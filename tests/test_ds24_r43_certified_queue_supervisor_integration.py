from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


R42_QUEUE = [
    "random_forest",
    "elastic_net",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
]


def write_manifest_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queue: list[str] | None = None) -> Path:
    manifest = tmp_path / "R42_ready_family_queue.json"
    matrix = tmp_path / "R42_full_family_readiness_matrix.json"
    families = list(queue or R42_QUEUE)
    manifest.write_text(
        json.dumps(
            {
                "ticket": supervisor.R42_READY_QUEUE_AUTHORITY_ID,
                "generated_at_utc": "2026-09-09T00:00:00+00:00",
                "ready_family_queue": families,
            }
        ),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps(
            {
                "families": [
                    {"family": family, "current_r42_state": "READY_TO_LAUNCH"}
                    for family in families
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    return manifest


def row(family: str, state: str, *, pid_alive: bool = False, duplicate: int = 0) -> dict[str, object]:
    return {
        "family": family,
        "state": state,
        "pid_alive": pid_alive,
        "duplicate_worker_count": duplicate,
        "checkpoint": f"{family}/progress.json",
        "namespace_lease_state": "STALE_RECOVERABLE",
        "metrics_rows": 0,
        "heavy": family not in {"elastic_net"},
    }


def board(**overrides: str) -> list[dict[str, object]]:
    defaults = {family: "V3_CERTIFIED_READY" for family in R42_QUEUE}
    defaults["elastic_net"] = "CERTIFIED_READY"
    defaults.update(overrides)
    return [row(family, state) for family, state in defaults.items()]


def test_ready_queue_manifest_validates_r42_authority_and_excludes_tft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)

    authority = supervisor.validate_ready_family_queue_manifest(manifest)

    assert authority["authority"] == "R42"
    assert authority["ready_family_queue"] == R42_QUEUE
    assert authority["manifest_hash"]

    bad = write_manifest_pair(tmp_path, monkeypatch, [*R42_QUEUE, "Temporal Fusion Transformer"])
    with pytest.raises(RuntimeError, match="TFT_FORBIDDEN"):
        supervisor.validate_ready_family_queue_manifest(bad)


def test_certified_queue_preserves_manifest_order_and_worker_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)

    plan = supervisor.certified_queue_admission_plan(board(), ready_family_queue_manifest=manifest)

    assert plan["queue"] == R42_QUEUE
    assert plan["first_eligible_family"] == "random_forest"
    routes = {family: supervisor.execution_registry_row(family)["worker_kind"] for family in R42_QUEUE}
    assert routes["random_forest"] == "TABULAR"
    assert routes["elastic_net"] == "TABULAR"
    assert routes["lightgbm_rank_xendcg"] == "LIGHTGBM_RANKING"
    assert routes["lightgbm_lambdarank"] == "LIGHTGBM_RANKING"
    assert routes["DLinear"] == "PYTORCH_SEQUENCE"
    assert supervisor.execution_registry_row("Temporal Fusion Transformer")["launch_enabled"] is False


def test_r42_capability_authority_is_explicit_opt_in() -> None:
    default_decision = supervisor.forward_metrics_contract_admission_decision("elastic_net", evaluation_version="v3")
    r43_decision = supervisor.forward_metrics_contract_admission_decision(
        "elastic_net",
        evaluation_version="v3",
        include_r42_authority=True,
    )

    assert default_decision["admitted"] is False
    assert r43_decision["admitted"] is True


def test_free_slot_scenarios_skip_running_and_family_specific_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)

    scenario_a = supervisor.certified_queue_admission_plan(board(), ready_family_queue_manifest=manifest)
    scenario_b_board = board()
    scenario_b_board[0] = row("random_forest", "V3_CERTIFIED_READY", pid_alive=True)
    scenario_b = supervisor.certified_queue_admission_plan(scenario_b_board, ready_family_queue_manifest=manifest)
    scenario_c_board = list(scenario_b_board)
    scenario_c_board[1] = row("elastic_net", "CRASHED_BLOCKED")
    scenario_c = supervisor.certified_queue_admission_plan(scenario_c_board, ready_family_queue_manifest=manifest)

    assert scenario_a["first_eligible_family"] == "random_forest"
    assert scenario_b["first_eligible_family"] == "elastic_net"
    assert scenario_c["first_eligible_family"] == "lightgbm_rank_xendcg"
    assert any(item["family"] == "elastic_net" and item["block_scope"] == "FAMILY_SPECIFIC_BLOCK" for item in scenario_c["skipped_families"])


def test_reboot_reconstruction_does_not_duplicate_running_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    recovered = board()
    recovered[0] = row("random_forest", "RUNNING", pid_alive=True)
    recovered[1] = row("elastic_net", "CERTIFIED_READY", duplicate=1)

    plan = supervisor.certified_queue_admission_plan(recovered, ready_family_queue_manifest=manifest)

    assert plan["first_eligible_family"] == "lightgbm_rank_xendcg"
    reasons = {item["family"]: item["reason"] for item in plan["skipped_families"]}
    assert reasons["random_forest"] == "RUNNING"
    assert reasons["elastic_net"] == "DUPLICATE_NAMESPACE_OWNER"


def test_dry_run_admission_reports_no_slot_but_next_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "build_family_board", lambda: board())
    monkeypatch.setattr(
        supervisor,
        "resource_gate",
        lambda _board, _config: {
            "admitted": False,
            "blocked_reasons": ["MAX_ACTIVE_MODEL_PROCESSES"],
            "resource_snapshot": {"available_ram_bytes": 10, "system_commit_percent": 50.0, "disk_free_bytes": 100},
            "zero_full_prediction_guard": {"paper_orders": 0, "live_orders": 0, "holdout_accessed": False, "full_prediction_files_in_metrics_namespaces": 0},
        },
    )
    monkeypatch.setattr(supervisor, "active_running", lambda _board: [row("mlp", "RUNNING", pid_alive=True), row("extra_trees", "RUNNING", pid_alive=True), row("gradient_boosting", "RUNNING", pid_alive=True)])
    monkeypatch.setattr(supervisor, "task_state", lambda name: {"TaskName": name, "State": "Absent"})
    task_plan = tmp_path / "task_plan.json"
    monkeypatch.setattr(supervisor, "R42_REBOOT_TASK_PLAN_PATH", task_plan)
    monkeypatch.setattr(supervisor, "R43_DRY_RUN_ADMISSION_PATH", tmp_path / "dry_run.json")
    args = argparse.Namespace(
        ready_family_queue_manifest=str(manifest),
        admit_crashed_recoverable=True,
        max_active_model_processes=3,
        max_policy_workers=3,
        max_restarts_per_family=2,
        max_system_commit_percent=95.0,
        admission_commit_percent=92.0,
        min_available_ram_gb=6.0,
        poll_seconds=20.0,
        evaluation_version="v3",
        metrics_root_name="metrics_only_v3",
        refit_policy="daily_session_v1",
    )

    dry = supervisor.dry_run_admission(args)

    assert dry["status"] == "NO_SLOT_AVAILABLE"
    assert dry["next_when_slot_available"] == "random_forest"
    assert dry["first_eligible_family"] == ""
    assert dry["worker_launches"] == 0
    assert dry["workers_stopped"] == 0
    assert dry["reboot_task_plan"]["activation_deferred"] is True
    assert task_plan.exists()
