from __future__ import annotations

from pathlib import Path

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor
from scripts.local import ds24_p8_r14_e3g_c2_r7_r44a_disk_capacity_stabilisation as r44a


def _quiet_resource_gate(monkeypatch, *, disk_gib: int = 24, recent_disk_containment: bool = False) -> None:
    monkeypatch.setattr(
        supervisor,
        "system_snapshot",
        lambda: {
            "available_ram_bytes": 16 * 1024**3,
            "total_ram_bytes": 32 * 1024**3,
            "system_commit_percent": 50.0,
            "disk_free_bytes": disk_gib * 1024**3,
            "disk_total_bytes": 100 * 1024**3,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "exact_manifest",
        lambda: {"pid_alive": True, "terminal_complete": True, "working_set": 0, "private_memory": 0},
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
    monkeypatch.setattr(supervisor, "disallowed_process_manifest", lambda: {"ds26_workers": [], "r20_compactor": []})
    monkeypatch.setattr(supervisor, "task_state", lambda _name: {})
    monkeypatch.setattr(supervisor, "r36_admission_hold_active", lambda: False)
    monkeypatch.setattr(supervisor, "read_json", lambda _path: {})
    monkeypatch.setattr(
        supervisor,
        "recent_hard_resource_containment_state",
        lambda **_kwargs: {
            "contract_id": supervisor.R44A_DISK_READMISSION_CONTRACT_ID,
            "recent": recent_disk_containment,
            "reason": "DISK_BELOW_12_GIB_PAUSE_NEWEST" if recent_disk_containment else "",
            "stopped_worker_count": 1 if recent_disk_containment else 0,
        },
    )


def test_allowed_cleanup_root_accepts_repo_pycache_and_rejects_outside(tmp_path: Path, monkeypatch) -> None:
    code_root = tmp_path / "core"
    allowed = code_root / "pkg" / "__pycache__"
    outside = tmp_path / "outside" / "__pycache__"
    allowed.mkdir(parents=True)
    outside.mkdir(parents=True)
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [code_root])

    assert r44a.allowed_cleanup_root(allowed) == str(allowed.resolve())
    assert r44a.allowed_cleanup_root(outside) == ""


def test_reparse_point_refused_before_safe_cache_classification(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "core" / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [tmp_path / "core"])
    monkeypatch.setattr(r44a, "is_reparse_point", lambda _path: True)

    row = r44a.classify_cleanup_candidate(
        cache,
        "python_bytecode_cache",
        manifest_rows=[],
        processes=[],
        tracked_counter=lambda _path: 0,
    )

    assert row["classification"] == "UNKNOWN_FAIL_CLOSED"
    assert row["proposed_action"] == "KEEP"
    assert "reparse_point_refused" in row["evidence"]


def test_tracked_file_refusal_overrides_disposable_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "core" / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.pyc").write_bytes(b"cache")
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [tmp_path / "core"])

    row = r44a.classify_cleanup_candidate(
        cache,
        "python_bytecode_cache",
        manifest_rows=[],
        processes=[],
        tracked_counter=lambda _path: 1,
    )

    assert row["classification"] == "PROTECTED_GIT_STATE"
    assert row["expected_reclaimed_bytes"] == 0


def test_pagefile_protection_is_windows_managed() -> None:
    row = r44a.classify_cleanup_candidate(
        Path("C:/pagefile.sys"),
        "windows_managed_virtual_memory",
        manifest_rows=[],
        processes=[],
        tracked_counter=lambda _path: 0,
    )

    assert row["classification"] == "WINDOWS_MANAGED_DO_NOT_DELETE"
    assert row["proposed_action"] == "KEEP"


def test_active_namespace_manifest_protection_rejects_candidate(tmp_path: Path, monkeypatch) -> None:
    active_root = tmp_path / "stage" / "r7_r14_policy_workers" / "huber" / "metrics_only_v3_r37_huber_replay"
    cache = active_root / "__pycache__"
    cache.mkdir(parents=True)
    manifest = [
        {
            "category": "PROTECTED_ACTIVE_RUNTIME",
            "path": str(active_root),
            "resolved_path": str(active_root.resolve()),
            "reason": "huber_active_metrics_namespace",
        }
    ]
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [tmp_path])

    row = r44a.classify_cleanup_candidate(
        cache,
        "python_bytecode_cache",
        manifest_rows=manifest,
        processes=[],
        tracked_counter=lambda _path: 0,
    )

    assert row["classification"] == "PROTECTED_ACTIVE_RUNTIME"
    assert row["lease_or_manifest_reference"] == "huber_active_metrics_namespace"


def test_pre_deletion_inventory_only_records_safe_delete_candidates(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "core" / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.pyc").write_bytes(b"12345")
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [tmp_path / "core"])
    safe = r44a.classify_cleanup_candidate(
        cache,
        "python_bytecode_cache",
        manifest_rows=[],
        processes=[],
        tracked_counter=lambda _path: 0,
    )
    protected = {**safe, "classification": "PROTECTED_GIT_STATE", "proposed_action": "KEEP"}

    rows = r44a.pre_deletion_inventory([safe, protected])

    assert len(rows) == 1
    assert rows[0]["inventory_file_count"] == 1
    assert rows[0]["inventory_size_bytes"] == 5


def test_delete_safe_candidates_rechecks_and_deletes_only_safe_tmp_path(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "scripts" / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.pyc").write_bytes(b"cache")
    monkeypatch.setattr(r44a, "CODE_CACHE_SEARCH_ROOTS", [tmp_path / "scripts"])
    monkeypatch.setattr(r44a, "protected_path_rows", lambda: [])
    monkeypatch.setattr(r44a, "process_snapshot", lambda: [])
    monkeypatch.setattr(r44a, "system_resource_snapshot", lambda: {"disk_free_bytes": 100})
    monkeypatch.setattr(r44a.r44, "checkpoint_integrity_rows", lambda: [])
    row = r44a.classify_cleanup_candidate(
        cache,
        "python_bytecode_cache",
        manifest_rows=[],
        processes=[],
        tracked_counter=lambda _path: 0,
    )

    ledger, result = r44a.delete_safe_candidates([row], execute=True)

    assert ledger[0]["status"] == "DELETED"
    assert not cache.exists()
    assert result["deleted_count"] == 1


def test_supervisor_readmission_hysteresis_blocks_after_recent_hard_resource_containment(monkeypatch) -> None:
    _quiet_resource_gate(monkeypatch, disk_gib=19, recent_disk_containment=True)

    gate = supervisor.resource_gate([], supervisor.GateConfig(max_policy_workers=3, max_active_model_processes=3))

    assert gate["admitted"] is False
    assert "RESOURCE_CONTAINMENT_READMISSION_HYSTERESIS" in gate["blocked_reasons"]
    assert gate["disk_readmission_policy"]["post_disk_containment_readmission_floor_bytes"] == 20 * 1024**3


def test_supervisor_emergency_disk_containment_floor_is_not_weakened(monkeypatch) -> None:
    _quiet_resource_gate(monkeypatch, disk_gib=2, recent_disk_containment=False)

    gate = supervisor.resource_gate([], supervisor.GateConfig())

    assert supervisor.GateConfig().hard_disk_floor_bytes == supervisor.MIN_EXECUTION_FREE_DISK_BYTES
    assert "DISK_HARD_FLOOR" in gate["blocked_reasons"]


def test_exact_queue_and_future_metrics_gate_are_preserved() -> None:
    assert supervisor.selected_queue("rff_ridge,huber,mlp") == ["rff_ridge", "huber", "mlp"]

    future = supervisor.forward_metrics_contract_admission_decision(
        "random_forest",
        evaluation_version="v3",
        metrics_root_name="metrics_only_v3",
    )

    assert future["admitted"] is True
    assert future["capability_decision"] == "ADMITTED_EXTENDED_PERFORMANCE_METRICS_READY"
