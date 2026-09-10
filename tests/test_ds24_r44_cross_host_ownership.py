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
    monkeypatch.setattr(supervisor, "R42_READY_QUEUE_PATH", manifest)
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    return manifest


def write_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "R44_cross_host_family_ownership.json"
    monkeypatch.setattr(supervisor, "R44_CROSS_HOST_OWNERSHIP_PATH", path)
    supervisor.build_cross_host_ownership_authority(updated_at_utc="2026-09-09T01:00:00+00:00", path=path)
    return path


def release_dlinear_to_dell(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["families"]:
        if row["family"] == "DLinear":
            row.update(
                {
                    "execution_owner": "DELL",
                    "ownership_state": "DELL_OWNED",
                    "owner_state": "DELL_OWNED",
                    "dell_eligible": True,
                    "mac_eligible": False,
                    "ownership_reason": "R44 reservation released to Dell without R42 recertification.",
                }
            )
            row["authority_hash"] = supervisor.state_hash(row)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def row(family: str, state: str, *, pid_alive: bool = False, duplicate: int = 0) -> dict[str, object]:
    return {
        "family": family,
        "state": state,
        "pid_alive": pid_alive,
        "duplicate_worker_count": duplicate,
        "checkpoint": f"{family}/progress.json",
        "namespace_lease_state": "STALE_RECOVERABLE",
        "metrics_rows": 0,
        "heavy": family != "elastic_net",
    }


def board(**overrides: str) -> list[dict[str, object]]:
    defaults = {family: "V3_CERTIFIED_READY" for family in R42_QUEUE}
    defaults["elastic_net"] = "CERTIFIED_READY"
    defaults.update(overrides)
    return [row(family, state) for family, state in defaults.items()]


def dry_run_args(manifest: Path, ownership: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ready_family_queue_manifest=str(manifest),
        cross_host_ownership_manifest=str(ownership),
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


def test_r44_authority_and_effective_dell_queue_preserve_r42_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)
    effective_path = tmp_path / "R44_dell_effective_ready_queue.json"

    authority = supervisor.validate_cross_host_ownership_authority(ownership)
    effective = supervisor.write_dell_effective_ready_queue(
        ready_family_queue_manifest=manifest,
        cross_host_ownership_manifest=ownership,
        path=effective_path,
    )

    assert authority["excluded_mac_owned"] == ["lightgbm_rank_xendcg", "lightgbm_lambdarank"]
    assert authority["excluded_mac_reserved"] == ["DLinear"]
    assert effective["dell_effective_ready_queue"] == [
        "random_forest",
        "elastic_net",
        "PatchTST",
        "Transformer",
        "iTransformer",
        "Momentum Transformer",
        "Market Context Encoder",
    ]
    assert "lightgbm_rank_xendcg" in supervisor.validate_ready_family_queue_manifest(manifest)["ready_family_queue"]
    assert effective_path.exists()


def test_r44_planner_skips_mac_owned_and_falls_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)
    current = board()
    current[0] = row("random_forest", "RUNNING", pid_alive=True)
    current[1] = row("elastic_net", "CRASHED_BLOCKED")

    plan = supervisor.certified_queue_admission_plan(
        current,
        ready_family_queue_manifest=manifest,
        cross_host_ownership_manifest=ownership,
    )

    assert plan["first_eligible_family"] == "PatchTST"
    reasons = {item["family"]: item["reason"] for item in plan["skipped_families"]}
    assert reasons["elastic_net"] == "STATE_NOT_ELIGIBLE:CRASHED_BLOCKED"
    assert reasons["lightgbm_rank_xendcg"] == "SKIP_MAC_OWNED"
    assert reasons["lightgbm_lambdarank"] == "SKIP_MAC_OWNED"
    assert reasons["DLinear"] == "SKIP_MAC_RESERVED"


def test_r44_dlinear_reservation_release_makes_dell_admissible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)
    release_dlinear_to_dell(ownership)
    current = board(
        random_forest="COMPLETE",
        elastic_net="COMPLETE",
        lightgbm_rank_xendcg="COMPLETE",
        lightgbm_lambdarank="COMPLETE",
    )

    plan = supervisor.certified_queue_admission_plan(
        current,
        ready_family_queue_manifest=manifest,
        cross_host_ownership_manifest=ownership,
    )

    assert plan["first_eligible_family"] == "DLinear"
    dlinear = supervisor.validate_cross_host_ownership_authority(ownership)["by_family"]["DLinear"]
    assert dlinear["dell_eligible"] is True
    assert dlinear["readiness_state"] == "READY_TO_LAUNCH"


def test_r44_duplicate_compute_guard_blocks_manual_mac_owned_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)

    plan = supervisor.certified_queue_admission_plan(
        board(),
        family_queue="lightgbm_rank_xendcg",
        ready_family_queue_manifest="",
        cross_host_ownership_manifest=ownership,
    )

    assert plan["first_eligible_family"] == ""
    assert plan["skipped_families"][0]["reason"] == "SKIP_MAC_OWNED"
    with pytest.raises(RuntimeError, match="CROSS_HOST_OWNERSHIP_ADMISSION_BLOCKED"):
        supervisor.assert_dell_ownership_admission("lightgbm_rank_xendcg", ownership)
    assert supervisor.validate_ready_family_queue_manifest(manifest)["ready_family_queue"][2] == "lightgbm_rank_xendcg"


def test_r44_reboot_reconstruction_does_not_duplicate_mac_owned_or_running_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)
    recovered = board()
    recovered[0] = row("random_forest", "RUNNING", pid_alive=True)
    recovered[1] = row("elastic_net", "CERTIFIED_READY", duplicate=1)

    plan = supervisor.certified_queue_admission_plan(
        recovered,
        ready_family_queue_manifest=manifest,
        cross_host_ownership_manifest=ownership,
    )

    assert plan["first_eligible_family"] == "PatchTST"
    reasons = {item["family"]: item["reason"] for item in plan["skipped_families"]}
    assert reasons["random_forest"] == "RUNNING"
    assert reasons["elastic_net"] == "DUPLICATE_NAMESPACE_OWNER"
    assert reasons["lightgbm_rank_xendcg"] == "SKIP_MAC_OWNED"


def test_r44_dry_run_reports_no_slot_and_mac_exclusions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = write_manifest_pair(tmp_path, monkeypatch)
    ownership = write_ownership(tmp_path, monkeypatch)
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
    monkeypatch.setattr(supervisor, "R44_REBOOT_TASK_PLAN_PATH", tmp_path / "R44_reboot_supervisor_task_plan.json")
    monkeypatch.setattr(supervisor, "R44_ADMISSION_VALIDATION_PATH", tmp_path / "R44_cross_host_admission_validation.json")

    dry = supervisor.dry_run_admission(dry_run_args(manifest, ownership))

    assert dry["status"] == "NO_SLOT_AVAILABLE"
    assert dry["next_when_slot_available"] == "random_forest"
    assert dry["first_eligible_family"] == ""
    assert dry["excluded_mac_owned"] == ["lightgbm_rank_xendcg", "lightgbm_lambdarank"]
    assert dry["excluded_mac_reserved"] == ["DLinear"]
    assert dry["worker_launches"] == 0
    assert dry["workers_stopped"] == 0
    assert dry["reboot_task_plan"]["r44_cross_host_ownership_manifest"]
