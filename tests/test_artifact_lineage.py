from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from application.services.ml_lineage_commands import run_artifact_lineage_verify, run_registry_verify

from core.research.ml.artifact_lineage import (
    CONFLICTING_EVIDENCE, DECLARED_STRICT_OOS_UNVERIFIED,
    INELIGIBLE_ARTIFACT_KIND, INSUFFICIENT_EVIDENCE, REJECTED_ARTIFACT,
    VERIFIED_STRICT_OOS, artifact_link_hash, build_artifact_link,
    promotion_eligibility, verify_lineage_graph, verify_selector_artifact,
    verify_upstream_set,
)


def _selector(**updates):
    values = dict(
        artifact_kind="BOUNDED_SELECTOR_PREDICTION", artifact_id="selector-2026-01-02",
        artifact_checksum="A" * 64, experiment_run_id="run-1",
        canonical_model_or_policy_id="ridge", model_or_policy_entry_hash="B" * 64,
        dataset_id="frozen-v2", dataset_checksum="C" * 64,
        row_population_hash="D" * 64, feature_schema_hash="E" * 64,
        target_contract_hash="F" * 64, decision_start="2026-01-02T00:00:00+00:00",
        decision_end="2026-01-02T00:00:00+00:00", training_cutoff="2026-01-01T00:00:00+00:00",
        maximum_label_available_timestamp="2026-01-02T00:00:00+00:00",
        strict_oos_claim=True, strict_oos_evidence={"prediction_quality_passed": True, "row_population_verified": True},
        completion_status="complete",
    )
    values.update(updates)
    return build_artifact_link(**values)


def test_strict_oos_selector_passes_and_hash_is_deterministic():
    first = _selector()
    second = _selector(created_at="later", artifact_manifest_path="C:/machine/repo/manifest.json")
    assert verify_selector_artifact(first).status == VERIFIED_STRICT_OOS
    assert artifact_link_hash(first) == artifact_link_hash(second)


@pytest.mark.parametrize(("updates", "reason"), [
    ({"maximum_label_available_timestamp": "2026-01-03T00:00:00+00:00"}, "LABEL_AVAILABILITY_AFTER_DECISION"),
    ({"training_cutoff": "2026-01-02T00:00:00+00:00"}, "TRAINING_CUTOFF_NOT_BEFORE_DECISION"),
    ({"target_contract_hash": None}, "TARGET_CONTRACT_MISSING"),
    ({"strict_oos_evidence": {"prediction_quality_passed": False, "row_population_verified": True}}, "PREDICTION_QUALITY_FAILED"),
])
def test_conflicting_or_missing_selector_evidence(updates, reason):
    result = verify_selector_artifact(_selector(**updates))
    assert reason in result.reason_codes
    assert result.status in {CONFLICTING_EVIDENCE, DECLARED_STRICT_OOS_UNVERIFIED}


def test_rejected_and_diagnostic_artifacts_never_verify():
    assert verify_selector_artifact(_selector(completion_status="rejected")).status == REJECTED_ARTIFACT
    assert verify_selector_artifact(_selector(artifact_kind="RESEARCH_DIAGNOSTIC")).status == INELIGIBLE_ARTIFACT_KIND


def test_legacy_artifact_has_explicit_insufficient_evidence():
    result = verify_selector_artifact({"artifact_kind": "BOUNDED_SELECTOR_PREDICTION", "completion_status": "complete"})
    assert result.status == INSUFFICIENT_EVIDENCE
    assert "LEGACY_IDENTITY_INSUFFICIENT" not in result.reason_codes
    assert "TRAINING_CUTOFF_MISSING" in result.reason_codes


def test_replay_upstream_set_requires_consistency_and_unique_decisions():
    first = _selector()
    second = _selector(artifact_id="selector-2")
    result = verify_upstream_set([first, second], promotion_mode=True)
    assert "DUPLICATE_DECISION_DATE_OWNERSHIP" in result.reason_codes
    second = _selector(artifact_id="selector-2", decision_start="2026-01-03T00:00:00+00:00", decision_end="2026-01-03T00:00:00+00:00", target_contract_hash="0" * 64)
    assert "TARGET_CONTRACT_MISMATCH" in verify_upstream_set([first, second], promotion_mode=True).reason_codes


def test_research_mode_records_ineligibility_and_promotion_blocks():
    legacy = _selector(training_cutoff=None)
    result = verify_upstream_set([legacy], promotion_mode=False)
    assert result.status == INSUFFICIENT_EVIDENCE
    promotion = promotion_eligibility({**legacy, "artifact_kind": "PORTFOLIO_REPLAY"}, result)
    assert promotion["promotion_eligible"] is False
    assert "UPSTREAM_NOT_VERIFIED_STRICT_OOS" in promotion["blocking_reasons"]


def _write(path: Path, link: dict):
    path.write_text(json.dumps({"artifact_link": link}), encoding="utf-8")


def test_valid_selector_replay_exposure_graph(tmp_path: Path):
    selector_path = tmp_path / "selector.json"
    selector = _selector(artifact_manifest_path=selector_path)
    _write(selector_path, selector)
    replay_path = tmp_path / "replay.json"
    selector_ref = {**selector, "artifact_manifest_path": selector_path.name}
    replay = build_artifact_link(
        artifact_kind="PORTFOLIO_REPLAY", artifact_id="replay", artifact_checksum="1" * 64,
        upstream_links=[selector_ref], verification_status=VERIFIED_STRICT_OOS,
        strict_oos_claim=True, completion_status="complete",
    )
    _write(replay_path, replay)
    exposure = build_artifact_link(
        artifact_kind="EXPOSURE_DATASET", artifact_id="exposure", artifact_checksum="2" * 64,
        upstream_links=[{**replay, "artifact_manifest_path": replay_path.name}],
        verification_status=VERIFIED_STRICT_OOS, strict_oos_claim=True,
        strict_oos_evidence={"target_maturity_guard_passed": True}, completion_status="complete",
    )
    exposure_path = tmp_path / "exposure.json"
    _write(exposure_path, exposure)
    result = verify_lineage_graph(exposure_path, require_promotion_grade=True)
    assert result["verification_status"] == VERIFIED_STRICT_OOS
    assert result["promotion"]["promotion_eligible"] is True


def test_graph_reports_missing_edge_and_cycle(tmp_path: Path):
    missing = build_artifact_link(
        artifact_kind="PORTFOLIO_REPLAY", artifact_id="missing", artifact_checksum="1" * 64,
        upstream_links=[{"artifact_id": "gone"}], verification_status=VERIFIED_STRICT_OOS,
        completion_status="complete",
    )
    missing_path = tmp_path / "missing.json"; _write(missing_path, missing)
    result = verify_lineage_graph(missing_path)
    assert result["failing_edge"] == "missing[0]"
    assert "UPSTREAM_LINK_MISSING" in result["verification_reasons"]

    cycle_path = tmp_path / "cycle.json"
    cycle = build_artifact_link(
        artifact_kind="PORTFOLIO_REPLAY", artifact_id="cycle", artifact_checksum="1" * 64,
        upstream_links=[{"artifact_id": "cycle", "artifact_manifest_path": cycle_path.name}],
        verification_status=VERIFIED_STRICT_OOS, completion_status="complete",
    )
    _write(cycle_path, cycle)
    assert "LINEAGE_CYCLE" in verify_lineage_graph(cycle_path)["verification_reasons"]


def test_graph_reports_link_hash_and_expected_kind_mismatch(tmp_path: Path):
    path = tmp_path / "selector.json"; link = _selector(); link["artifact_id"] = "tampered"; _write(path, link)
    result = verify_lineage_graph(path, expected_artifact_kind="PORTFOLIO_REPLAY")
    assert result["verification_status"] == CONFLICTING_EVIDENCE
    assert "EXPECTED_ARTIFACT_KIND_MISMATCH" in result["verification_reasons"]


def test_graph_detects_selector_artifact_checksum_mismatch(tmp_path: Path):
    artifact = tmp_path / "predictions.bin"; artifact.write_bytes(b"tampered")
    path = tmp_path / "selector.json"
    link = _selector(artifact_manifest_path=path, artifact_path=artifact.name)
    _write(path, link)
    result = verify_lineage_graph(path)
    assert result["verification_status"] == CONFLICTING_EVIDENCE
    assert result["verification_reasons"] == ["UPSTREAM_CHECKSUM_MISMATCH"]


def test_cli_writes_json_and_markdown_without_ledger_mutation(tmp_path: Path):
    manifest = tmp_path / "selector.json"; _write(manifest, _selector(artifact_manifest_path=manifest))
    output = tmp_path / "audit.json"
    args = SimpleNamespace(
        artifact_manifest=str(manifest), expected_artifact_kind="BOUNDED_SELECTOR_PREDICTION",
        require_promotion_grade=True, verification_output=str(output),
    )
    result = run_artifact_lineage_verify({}, args)
    assert result["promotion"]["promotion_eligible"] is True
    assert output.exists() and output.with_suffix(".md").exists()
    assert not (tmp_path / "experiment_ledger.jsonl").exists()


def test_cli_failure_exit_and_registry_verification(tmp_path: Path):
    manifest = tmp_path / "legacy.json"; _write(manifest, build_artifact_link(artifact_kind="ORDINARY_SELECTOR_PREDICTION", artifact_id="legacy"))
    args = SimpleNamespace(artifact_manifest=str(manifest), expected_artifact_kind=None, require_promotion_grade=True, verification_output=None)
    with pytest.raises(SystemExit) as exc:
        run_artifact_lineage_verify({}, args)
    assert exc.value.code == 2
    registry = run_registry_verify({}, SimpleNamespace(verification_output=None))
    assert registry["status"] == "VERIFIED"
    assert registry["entry_count"] > 0
