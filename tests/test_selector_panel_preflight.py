import json
from pathlib import Path

import pytest

from core.research.ml.registries import load_registry_bundle
from core.research.ml.selector_panel_preflight import (
    CHALLENGERS,
    PRIMARY_MODEL,
    freeze_panel,
    powershell_commands,
    resolve_authoritative_panel,
    run_preflight,
)


MODELS = (PRIMARY_MODEL, *CHALLENGERS)


def _component(date, model, *, population="population", dataset="dataset", eligible=True):
    return {
        "artifact_link_hash": f"link:{date}:{model}", "model_id": model, "selector_version": "v1",
        "training_cutoff": "2023-12-01", "prediction_date": date,
        "dataset_id": "source", "dataset_checksum": dataset, "feature_schema_hash": f"feature:{model}",
        "target_contract_hash": "target", "ranking_contract_version": "ranking_metric_contract_v1",
        "row_population_hash": population, "verification_status": "VERIFIED_STRICT_OOS" if eligible else "INSUFFICIENT_EVIDENCE",
        "lineage_contract_version": "artifact_link_contract_v1", "eligible": eligible, "rejection_reasons": [],
    }


def _components(dates=("2024-01-02", "2024-01-05")):
    return {(date, model): _component(date, model, population=f"population:{date}", dataset=f"dataset:{date}") for date in dates for model in MODELS}


def _panel(tmp_path, requested=("2024-01-01",), components=None):
    bundle = load_registry_bundle()
    panel = resolve_authoritative_panel(
        panel_name="test", panel_version="v1", requested_dates=requested,
        components=components if components is not None else _components(), primary_model=PRIMARY_MODEL,
        challengers=CHALLENGERS, registry_set_hash=bundle.registry_set_hash,
        policy_registry_hash=bundle.documents["portfolio_policies"].registry_hash,
    )
    path = tmp_path / "panel.json"; freeze_panel(path, panel)
    return panel, path


def test_resolution_is_deterministic_and_prefers_forward():
    bundle = load_registry_bundle(); kwargs = dict(panel_name="x", panel_version="v1", requested_dates=["2024-01-03"], components=_components(), primary_model=PRIMARY_MODEL, challengers=CHALLENGERS, registry_set_hash=bundle.registry_set_hash, policy_registry_hash=bundle.documents["portfolio_policies"].registry_hash)
    first = resolve_authoritative_panel(**kwargs); second = resolve_authoritative_panel(**kwargs)
    assert first["requested_to_resolved"] == second["requested_to_resolved"] == [{"requested": "2024-01-03", "resolved": "2024-01-05", "method": "forward"}]
    assert first["panel_checksum"] == second["panel_checksum"]


def test_missing_requested_date_is_explicitly_blocked(tmp_path):
    panel, _ = _panel(tmp_path, components={})
    assert panel["status"] == "BLOCKED"
    assert panel["exclusions"][0]["rejection_reason"] == "NO_SHARED_ELIGIBLE_SELECTOR_DATE"


def test_duplicate_resolved_date_is_rejected(tmp_path):
    panel, _ = _panel(tmp_path, requested=("2024-01-01", "2024-01-02"))
    assert panel["resolved_dates"] == ["2024-01-02"]
    assert panel["exclusions"][0]["rejection_reason"] == "DUPLICATE_RESOLVED_DATE"


@pytest.mark.parametrize("defect,reason", [
    ("strict", "NO_SHARED_ELIGIBLE_SELECTOR_DATE"),
    ("dataset", "NO_SHARED_ELIGIBLE_SELECTOR_DATE"),
    ("missing", "NO_SHARED_ELIGIBLE_SELECTOR_DATE"),
    ("population", "NO_SHARED_ELIGIBLE_SELECTOR_DATE"),
])
def test_shared_population_defects_fail_closed(tmp_path, defect, reason):
    components = _components(("2024-01-02",))
    if defect == "strict": components[("2024-01-02", "ridge")]["eligible"] = False
    if defect == "dataset": components[("2024-01-02", "ridge")]["dataset_checksum"] = "other"
    if defect == "missing": del components[("2024-01-02", "elastic_net")]
    if defect == "population": components[("2024-01-02", "ridge")]["row_population_hash"] = "other"
    panel, _ = _panel(tmp_path, requested=("2024-01-02",), components=components)
    assert panel["status"] == "BLOCKED" and panel["exclusions"][0]["rejection_reason"] == reason


def test_frozen_panel_cannot_be_silently_changed(tmp_path):
    panel, path = _panel(tmp_path)
    assert freeze_panel(path, panel) == "skipped_identical"
    changed = {**panel, "panel_checksum": "changed"}
    with pytest.raises(FileExistsError, match="conflict"):
        freeze_panel(path, changed)


def test_preflight_rejects_stale_registry_and_changed_checksum(tmp_path):
    panel, path = _panel(tmp_path)
    report = run_preflight(frozen_panel=path, output_root=tmp_path / "out", current_registry_set_hash="stale")
    assert report["exit_code"] == 2 and "STALE_REGISTRY_SET_HASH" in report["blocking_reasons"]
    payload = json.loads(path.read_text()); payload["resolved_dates"] = ["2099-01-01"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_preflight(frozen_panel=path, output_root=tmp_path / "out")
    assert "PANEL_CHECKSUM_MISMATCH" in report["blocking_reasons"]


def test_partition_plan_complete_incomplete_and_conflict_is_read_only(tmp_path):
    panel, path = _panel(tmp_path, requested=("2024-01-02",))
    output = tmp_path / "out"
    owners = []
    for policy in ("daily_top_k_equal_weight_v1",):
        for bps in (5, 10, 25):
            owner = output / "model=ordered_logit_ranker" / "date=2024-01-02" / f"policy={policy}" / f"cost_bps={bps}"
            owner.mkdir(parents=True); owners.append(owner)
    (owners[0] / "manifest.json").write_text(json.dumps({"status": "complete", "identity": {"date_panel_checksum": panel["panel_checksum"]}}))
    (owners[2] / "manifest.json").write_text(json.dumps({"status": "complete", "identity": {"date_panel_checksum": "wrong"}}))
    before = sorted(str(item) for item in output.rglob("*"))
    report = run_preflight(frozen_panel=path, output_root=output)
    after = sorted(str(item) for item in output.rglob("*"))
    assert before == after
    states = {(row["cost_bps"], row["status"], row["action"]) for row in report["partition_plan"] if row["policy_id"] == "daily_top_k_equal_weight_v1"}
    assert (5, "complete", "skip") in states
    assert (10, "incomplete", "resume") in states
    assert (25, "conflicting", "block") in states
    assert report["exit_code"] == 2


def test_exact_powershell_commands_include_bounded_controls(tmp_path):
    commands = powershell_commands(tmp_path / "panel.json", tmp_path / "out", tmp_path / "logs", concurrency=2, failure_threshold=1)
    command = commands["first_run"]
    for value in ("ordered_logit_ranker", "--concurrency 2", "--failure-threshold 1", "--resume", "--no-promotion"):
        assert value in command
    assert commands["resume"] == command
    assert "--require-all-components" in commands["aggregate_publication_blocked"]
