from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


def _write_r42_r44_authorities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r42 = tmp_path / "R42_ready_family_queue.json"
    matrix = tmp_path / "R42_full_family_readiness_matrix.json"
    r44 = tmp_path / "R44_cross_host_family_ownership.json"
    effective = tmp_path / "R44_dell_effective_ready_queue.json"
    task_blocker = tmp_path / "R45_windows_task_registration.json"
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
    task_blocker.write_text(
        json.dumps({"registration_result": {"blocker": "WINDOWS_TASK_SCHEDULER_ACCESS_DENIED"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "R42_READY_QUEUE_PATH", r42)
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    monkeypatch.setattr(supervisor, "R44_CROSS_HOST_OWNERSHIP_PATH", r44)
    monkeypatch.setattr(supervisor, "R44_DELL_EFFECTIVE_READY_QUEUE_PATH", effective)
    monkeypatch.setattr(supervisor, "R45_WINDOWS_TASK_REGISTRATION_PATH", task_blocker)
    supervisor.build_cross_host_ownership_authority(updated_at_utc="2026-09-09T01:00:00+00:00", path=r44)
    supervisor.write_dell_effective_ready_queue(
        ready_family_queue_manifest=r42,
        cross_host_ownership_manifest=r44,
        path=effective,
    )


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
            "cursor": "2020-01-01T21:00:00+00:00",
            "namespace_lease_state": "LIVE_VERIFIED",
            "duplicate_worker_count": 0,
            "metrics_rows": 10,
            "heavy": True,
        }
        for family, pid in zip(supervisor.R45_PROTECTED_DELL_WORKERS, (101, 102, 103))
    ]
    return ready + workers


def test_r46_startup_entry_invokes_only_checked_in_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = tmp_path / "launch_ds24_r45_supervisor.ps1"
    monkeypatch.setattr(supervisor, "R45_SUPERVISOR_LAUNCHER_PATH", launcher)

    text = supervisor.r46_startup_entry_text()

    assert str(launcher) in text
    assert "powershell.exe" in text
    assert "--daemon" not in text
    assert "--family-queue" not in text


def test_r46_autostart_authority_and_validation_use_startup_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    appdata = tmp_path / "AppData" / "Roaming"
    startup_dir = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup_dir.mkdir(parents=True)
    launcher = tmp_path / "launch_ds24_r45_supervisor.ps1"
    launcher.write_text("SUPERVISOR_ALREADY_RUNNING", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(supervisor, "R45_SUPERVISOR_LAUNCHER_PATH", launcher)
    entry = supervisor.r46_startup_entry_path()
    entry.write_bytes(supervisor.r46_startup_entry_text().encode("utf-8"))

    authority = supervisor.r46_user_autostart_authority(tmp_path / "authority.json")
    validation = supervisor.r46_validate_autostart_entry(tmp_path / "validation.json")

    assert authority["requires_admin"] is False
    assert authority["mechanism"] == "WINDOWS_CURRENT_USER_STARTUP_FOLDER"
    assert authority["windows_task_scheduler_historical_blocker"] == "WINDOWS_TASK_SCHEDULER_ACCESS_DENIED"
    assert validation["classification"] == supervisor.R46_CLASSIFICATION
    assert validation["startup_entry_path"] == str(entry)
    assert validation["uses_legacy_family_queue"] is False


def test_r46_singleton_live_validation_does_not_change_supervisor_or_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    launcher = tmp_path / "launch_ds24_r45_supervisor.ps1"
    launcher.write_text("SUPERVISOR_ALREADY_RUNNING", encoding="utf-8")
    monkeypatch.setattr(supervisor, "R45_SUPERVISOR_LAUNCHER_PATH", launcher)
    monkeypatch.setattr(
        supervisor,
        "r46_pre_autostart_snapshot",
        lambda: {
            "protected_workers": [
                {"family": "mlp", "pid": 101},
                {"family": "extra_trees", "pid": 102},
                {"family": "gradient_boosting", "pid": 103},
            ]
        },
    )
    monkeypatch.setattr(supervisor, "supervisor_daemon_processes", lambda: [{"ProcessId": 500}])
    monkeypatch.setattr(supervisor, "lightweight_certified_queue_board", lambda _queue: _board())
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout="SUPERVISOR_ALREADY_RUNNING\n", stderr=""),
    )

    payload = supervisor.r46_singleton_live_validation(tmp_path / "singleton.json")

    assert payload["classification"] == "SUPERVISOR_ALREADY_RUNNING"
    assert payload["supervisor_count_before"] == 1
    assert payload["supervisor_count_after"] == 1
    assert payload["supervisor_pid_before"] == payload["supervisor_pid_after"] == 500
    assert payload["protected_worker_pids_before"] == payload["protected_worker_pids_after"]
    assert payload["worker_launches"] == 0


def test_r46_reboot_recovery_simulation_preserves_r44_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_r42_r44_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "lightweight_certified_queue_board", lambda _queue: _board()[: len(supervisor.R42_ALLOWED_READY_FAMILIES)])

    payload = supervisor.r46_reboot_recovery_simulation(tmp_path / "reboot.json")

    assert payload["classification"] == "PASS"
    assert payload["stale_supervisor_simulation"]["new_supervisor_permitted"] is True
    assert payload["clean_boot_simulation"]["first_dell_family"] == "random_forest"
    assert payload["clean_boot_simulation"]["effective_dell_queue"] == list(supervisor.R44_DELL_READY_FAMILIES)
    assert payload["partial_recovery_simulation"]["recoverable_dell_worker_selected"] is True
    assert payload["partial_recovery_simulation"]["mac_owned_family_substituted"] is False


def test_r45_launcher_contains_singleton_fail_closed_guard() -> None:
    text = supervisor.R45_SUPERVISOR_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "SUPERVISOR_ALREADY_RUNNING" in text
    assert "SUPERVISOR_IDENTITY_CONFLICT" in text
    assert "--family-queue" in text
    assert "-not $commandLine.Contains(\"--family-queue\")" in text
