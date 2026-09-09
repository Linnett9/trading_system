from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


def _write_r42_r44_authorities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    r42 = tmp_path / "R42_ready_family_queue.json"
    matrix = tmp_path / "R42_full_family_readiness_matrix.json"
    r44 = tmp_path / "R44_cross_host_family_ownership.json"
    effective = tmp_path / "R44_dell_effective_ready_queue.json"
    r42.write_text(
        json.dumps(
            {
                "ticket": supervisor.R42_READY_QUEUE_AUTHORITY_ID,
                "generated_at_utc": "2026-09-09T00:00:00+00:00",
                "ready_family_queue": list(supervisor.R42_ALLOWED_READY_FAMILIES),
            }
        ),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps(
            {
                "families": [
                    {"family": family, "current_r42_state": "READY_TO_LAUNCH"}
                    for family in supervisor.R42_ALLOWED_READY_FAMILIES
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "R42_READY_QUEUE_PATH", r42)
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    monkeypatch.setattr(supervisor, "R44_CROSS_HOST_OWNERSHIP_PATH", r44)
    monkeypatch.setattr(supervisor, "R44_DELL_EFFECTIVE_READY_QUEUE_PATH", effective)
    supervisor.build_cross_host_ownership_authority(updated_at_utc="2026-09-09T01:00:00+00:00", path=r44)
    supervisor.write_dell_effective_ready_queue(
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
        path=effective,
    )
    return r42, r44


def _board() -> list[dict[str, object]]:
    ready = [
        {
            "family": family,
            "state": "V3_CERTIFIED_READY",
            "pid_alive": False,
            "duplicate_worker_count": 0,
            "checkpoint": "",
            "namespace_lease_state": "STALE_RECOVERABLE",
            "metrics_rows": 0,
            "heavy": family != "elastic_net",
        }
        for family in supervisor.R42_ALLOWED_READY_FAMILIES
    ]
    workers = [
        {
            "family": family,
            "state": "RUNNING",
            "pid": pid,
            "pid_alive": True,
            "creation_time": f"2026-09-09T01:0{idx}:00+00:00",
            "command_line": f"python worker --family {family}",
            "cursor": "2020-01-01T21:00:00+00:00",
            "namespace_lease_state": "LIVE_VERIFIED",
            "duplicate_worker_count": 0,
            "metrics_rows": 10,
            "heavy": True,
        }
        for idx, (family, pid) in enumerate(zip(supervisor.R45_PROTECTED_DELL_WORKERS, (101, 102, 103)))
    ]
    return ready + workers


def _resource_gate(_board: list[dict[str, object]], _config: supervisor.GateConfig) -> dict[str, object]:
    return {
        "active_model_processes": 3,
        "zero_full_prediction_guard": {
            "holdout_accessed": False,
            "full_prediction_files_in_metrics_namespaces": 0,
            "paper_orders": 0,
            "live_orders": 0,
        },
        "resource_snapshot": {"available_ram_bytes": 8 * 1024**3, "system_commit_percent": 50.0, "disk_free_bytes": 20 * 1024**3},
    }


def test_r45_launch_authority_uses_r42_r44_and_no_legacy_family_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    path = tmp_path / "R45_supervisor_launch_authority.json"

    payload = supervisor.r45_supervisor_launch_authority(path)

    assert payload["uses_legacy_family_queue"] is False
    assert "--family-queue" not in payload["command"]
    assert "--ready-family-queue-manifest" in payload["command"]
    assert "--cross-host-ownership-manifest" in payload["command"]
    assert payload["scheduled_task_launcher"].endswith("launch_ds24_r45_supervisor.ps1")
    assert payload["effective_dell_queue"] == list(supervisor.R44_DELL_READY_FAMILIES)
    assert path.exists()


def test_r45_pre_cutover_snapshot_fails_closed_on_missing_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "lightweight_certified_queue_board", lambda _queue: _board()[:-1])
    monkeypatch.setattr(supervisor, "resource_gate", _resource_gate)
    monkeypatch.setattr(
        supervisor,
        "supervisor_daemon_processes",
        lambda: [{"ProcessId": 500, "CommandLine": "python supervisor --daemon"}],
    )

    with pytest.raises(RuntimeError, match="PROTECTED_WORKER_SET_MISMATCH"):
        supervisor.r45_pre_cutover_snapshot(tmp_path / "pre.json")


def test_r45_pre_cutover_snapshot_records_required_hashes_and_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    lease_path = tmp_path / "lease.json"
    lease_path.write_text(json.dumps({"pid": 500}), encoding="utf-8")
    monkeypatch.setattr(supervisor, "LEASE_PATH", lease_path)
    monkeypatch.setattr(supervisor, "lightweight_certified_queue_board", lambda _queue: _board())
    monkeypatch.setattr(supervisor, "resource_gate", _resource_gate)
    monkeypatch.setattr(
        supervisor,
        "supervisor_daemon_processes",
        lambda: [{"ProcessId": 500, "CommandLine": "python supervisor --daemon"}],
    )

    payload = supervisor.r45_pre_cutover_snapshot(tmp_path / "pre.json")

    assert payload["classification"] == "PASS"
    assert payload["active_model_processes"] == 3
    assert sorted(row["family"] for row in payload["protected_workers"]) == sorted(supervisor.R45_PROTECTED_DELL_WORKERS)
    assert payload["r42_manifest_hash"]
    assert payload["r44_ownership_manifest_hash"]
    assert payload["r44_effective_dell_queue_hash"]


def test_r45_reboot_simulation_and_queue_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "lightweight_certified_queue_board", lambda _queue: _board()[: len(supervisor.R42_ALLOWED_READY_FAMILIES)])
    monkeypatch.setattr(supervisor, "process_exists", lambda _pid: False)

    payload = supervisor.r45_reboot_simulation(tmp_path / "reboot.json")

    fallback = payload["queue_fallback_simulation"]
    assert payload["fresh_supervisor_would_recover_lease"] is True
    assert fallback["rf_running_next"] == "elastic_net"
    assert fallback["elastic_net_blocked_next"] == "PatchTST"
    assert fallback["patchtst_unavailable_order"][:4] == [
        "Transformer",
        "iTransformer",
        "Momentum Transformer",
        "Market Context Encoder",
    ]
    assert fallback["mac_exclusions_preserved"]["DLinear"] == "SKIP_MAC_RESERVED"


def test_r45_duplicate_daemon_is_refused_by_live_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "read_json", lambda _path: {"pid": 321, "lease_generation": 4})
    monkeypatch.setattr(supervisor, "process_exists", lambda pid: pid == 321)

    acquired, lease = supervisor.acquire_lease(resume=True)

    assert acquired is False
    assert lease["reason"] == "LIVE_LEASE_OWNER"
