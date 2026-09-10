from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from core.research.ml import ds24_metrics_only_evaluator as ev
from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


R50_SEQUENCE_FAMILIES = [
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
]


def complete_forward_capability(family: str) -> dict[str, object]:
    registry = ev.extended_performance_metrics_contract_registry()
    return {
        "family": family,
        "writer_initialised_before_first_prediction": True,
        "base_evaluation_contract_hash": ev.resolved_performance_contract_v3_hash(),
        "extended_contract_hash": ev.extended_performance_metrics_contract_hash(),
        "available_per_timestamp_fields": registry["mandatory_per_timestamp_fields"],
        "available_artifacts": registry["mandatory_artifacts"],
        "atomic_retention_gate_available": True,
    }


def write_r42_r44(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publish_capabilities: bool = True,
    corrupt_family: str = "",
) -> tuple[Path, Path]:
    stage = tmp_path / "stage"
    policy_root = stage / "r7_r14_policy_workers"
    capability_root = stage / "R42_forward_metrics_capabilities"
    capability_root.mkdir(parents=True)
    r42 = stage / "R42_ready_family_queue.json"
    matrix = stage / "R42_full_family_readiness_matrix.json"
    r44 = stage / "R44_cross_host_family_ownership.json"
    effective = stage / "R44_dell_effective_ready_queue.json"
    queue = list(supervisor.R42_ALLOWED_READY_FAMILIES)
    r42.write_text(
        json.dumps(
            {
                "ticket": supervisor.R42_READY_QUEUE_AUTHORITY_ID,
                "generated_at_utc": "2026-09-10T00:00:00+00:00",
                "ready_family_queue": queue,
            }
        ),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps(
            {
                "families": [
                    {"family": family, "current_r42_state": "READY_TO_LAUNCH"}
                    for family in queue
                ]
                + [
                    {
                        "family": "Temporal Fusion Transformer",
                        "current_r42_state": "CONFIGURATION_AUTHORITY_REQUIRED",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    if publish_capabilities:
        for family in [*queue, "Temporal Fusion Transformer"]:
            payload = complete_forward_capability(family)
            if family == corrupt_family:
                payload["extended_contract_hash"] = "invalid"
            (capability_root / f"{supervisor.family_slug(family)}.forward_metrics_capability.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
    monkeypatch.setattr(supervisor, "STAGE", stage)
    monkeypatch.setattr(supervisor, "POLICY_ROOT", policy_root)
    monkeypatch.setattr(supervisor, "R42_READY_QUEUE_PATH", r42)
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    monkeypatch.setattr(supervisor, "R44_CROSS_HOST_OWNERSHIP_PATH", r44)
    monkeypatch.setattr(supervisor, "R44_DELL_EFFECTIVE_READY_QUEUE_PATH", effective)
    supervisor.build_cross_host_ownership_authority(updated_at_utc="2026-09-10T01:00:00+00:00", path=r44)
    supervisor.write_dell_effective_ready_queue(
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
        path=effective,
    )
    return r42, r44


def running_row(family: str, pid: int) -> dict[str, object]:
    return {
        "ProcessId": pid,
        "ParentProcessId": 7560,
        "CreationDate": "2026-09-10T12:00:00+00:00",
        "CommandLine": f"python scripts/local/ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py --family {family}",
        "WorkingSetSize": 10,
        "PageFileUsage": 10,
    }


def dry_args(r42: Path, r44: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ready_family_queue_manifest=str(r42),
        cross_host_ownership_manifest=str(r44),
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


def test_r50_reboot_reconstructs_absent_r42_sequence_as_certified_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r42, r44 = write_r42_r44(tmp_path, monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "python_processes",
        lambda: [running_row("random_forest", 8332), running_row("elastic_net", 12608)],
    )

    board = supervisor.build_family_board()
    states = {row["family"]: row for row in board}
    plan = supervisor.certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
        admit_crashed_recoverable=True,
    )

    assert states["PatchTST"]["state"] == "V3_CERTIFIED_READY"
    assert states["PatchTST"]["runtime_state"] == "ABSENT"
    assert states["PatchTST"]["scientific_readiness"] == "READY_TO_LAUNCH"
    assert plan["first_eligible_family"] == "PatchTST"
    assert supervisor.execution_registry_row("PatchTST", include_r42_authority=True)["worker_kind"] == "PYTORCH_SEQUENCE"


def test_r50_clean_machine_keeps_all_r42_sequence_families_admission_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r42, r44 = write_r42_r44(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "python_processes", lambda: [])

    board = supervisor.build_family_board()
    states = {row["family"]: row["state"] for row in board}
    plan = supervisor.certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
    )

    assert [states[family] for family in R50_SEQUENCE_FAMILIES] == ["V3_CERTIFIED_READY"] * len(R50_SEQUENCE_FAMILIES)
    assert plan["eligible_families"][0]["family"] == "random_forest"


def test_r50_missing_or_invalid_r42_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_r42_r44(tmp_path, monkeypatch, corrupt_family="PatchTST")
    monkeypatch.setattr(supervisor, "python_processes", lambda: [])

    row = {item["family"]: item for item in supervisor.build_family_board()}["PatchTST"]

    assert row["state"] == "V3_CERTIFICATION_REQUIRED"
    assert row["state_reconciliation_reason"] == "R42_FORWARD_METRICS_CAPABILITY_INVALID_OR_MISSING"
    assert "extended_contract_hash" in row["sequence_certification_reconciliation"]["missing_requirements"]

    write_r42_r44(tmp_path / "missing", monkeypatch, publish_capabilities=False)
    row = {item["family"]: item for item in supervisor.build_family_board()}["PatchTST"]
    assert row["state"] == "V3_CERTIFICATION_REQUIRED"


def test_r50_sequence_runtime_failure_with_valid_r42_authority_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r42, r44 = write_r42_r44(tmp_path, monkeypatch)
    patchtst_root = supervisor.POLICY_ROOT / "PatchTST"
    patchtst_root.mkdir(parents=True)
    (patchtst_root / "stderr_r34_v3_gen1.log").write_text("Traceback\nRuntimeError: transient", encoding="utf-8")
    monkeypatch.setattr(
        supervisor,
        "python_processes",
        lambda: [running_row("random_forest", 8332), running_row("elastic_net", 12608)],
    )

    board = supervisor.build_family_board()
    row = {item["family"]: item for item in board}["PatchTST"]
    plan = supervisor.certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
        admit_crashed_recoverable=True,
    )

    assert row["state"] == "CRASHED_RECOVERABLE"
    assert plan["first_eligible_family"] == "PatchTST"


def test_r50_owner_filter_preserves_science_and_dell_eligibility_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r42, r44 = write_r42_r44(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "python_processes", lambda: [])
    board = supervisor.build_family_board()

    plan = supervisor.certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
    )
    reasons = {row["family"]: row["reason"] for row in plan["skipped_families"]}
    rows = {row["family"]: row for row in board}

    assert rows["PatchTST"]["state"] == "V3_CERTIFIED_READY"
    assert rows["DLinear"]["state"] == "V3_CERTIFIED_READY"
    assert reasons["DLinear"] == "SKIP_MAC_RESERVED"
    assert reasons["lightgbm_rank_xendcg"] == "SKIP_MAC_OWNED"
    assert rows["Temporal Fusion Transformer"]["state"] == "CONFIGURATION_AUTHORITY_REQUIRED"


def test_r50_dry_run_plans_patchtst_for_free_third_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r42, r44 = write_r42_r44(tmp_path, monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "python_processes",
        lambda: [running_row("random_forest", 8332), running_row("elastic_net", 12608)],
    )
    monkeypatch.setattr(
        supervisor,
        "system_snapshot",
        lambda: {
            "available_ram_bytes": 16 * 1024**3,
            "disk_free_bytes": 20 * 1024**3,
            "system_commit_percent": 45.0,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "zero_full_prediction_guard",
        lambda: {
            "paper_orders": 0,
            "live_orders": 0,
            "holdout_accessed": False,
            "full_prediction_files_in_metrics_namespaces": 0,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "exact_manifest",
        lambda: {"pid_alive": False, "terminal_complete": True, "working_set": 0, "private_memory": 0},
    )
    monkeypatch.setattr(supervisor, "cleanup_processes", lambda: [])
    monkeypatch.setattr(supervisor, "r36_admission_hold_active", lambda: {})
    monkeypatch.setattr(supervisor, "recent_hard_resource_containment_state", lambda **_kwargs: {"recent": False})
    monkeypatch.setattr(supervisor, "R44_REBOOT_TASK_PLAN_PATH", tmp_path / "reboot_task.json")
    monkeypatch.setattr(supervisor, "R44_ADMISSION_VALIDATION_PATH", tmp_path / "dry_run.json")

    dry = supervisor.dry_run_admission(dry_args(r42, r44))

    assert dry["status"] == "READY_TO_ADMIT"
    assert dry["first_eligible_family"] == "PatchTST"
    assert dry["next_when_slot_available"] == "PatchTST"
    assert dry["eligible_family_details"][0]["worker_kind"] == "PYTORCH_SEQUENCE"
    assert dry["worker_launches"] == 0
