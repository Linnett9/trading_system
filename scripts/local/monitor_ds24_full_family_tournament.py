from __future__ import annotations

import argparse
import datetime as dt
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
SUPERVISOR = "scripts\\local\\ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py"
SUMMARY_NAME = "r31_performance_summary.json"
V2_SUMMARY_NAME = "resolved_performance_summary_v2.json"
V2_CHECKPOINT_NAME = "resolved_performance_checkpoint_v2.json"
V3_SUMMARY_NAME = "resolved_performance_summary_v3.json"
V3_CHECKPOINT_NAME = "resolved_performance_checkpoint_v3.json"
R37_TERMINAL_NAME = "R7_R37_10_terminal_validation.json"
R36_TERMINAL_NAME = "R7_R36_07_terminal_validation.json"
R35_TERMINAL_NAME = "R7_R35_12_terminal_validation.json"
R35_ETA_NAME = "R7_R35_02_per_family_eta_report.json"
R35_READINESS_NAME = "R7_R35_05_forward_readiness_matrix.json"
R35_WATCHDOG_NAME = "R7_R35_10_watchdog_continuity_report.json"
R33_QUARANTINED_STATUS = "PROVISIONAL_UNVALIDATED_R33"
SUPERVISOR_LAUNCH_LEDGER = "R7_R27_07_automatic_launch_ledger.csv"
QUARANTINED_DISPLAY_VALUE = "QUARANTINED"

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor_api


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def heartbeat_age_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        value = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        return (dt.datetime.now(dt.timezone.utc) - value).total_seconds()
    except Exception:
        return None


def ps_json(script: str) -> Any:
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], cwd=ROOT, text=True, capture_output=True, timeout=20, check=False)
    if not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def process_status(pid: int) -> dict[str, Any]:
    data = ps_json(
        f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        "if ($p) { @{alive=$true;ProcessId=$p.Id;WorkingSetSize=$p.WorkingSet64;CPU=$p.CPU} | ConvertTo-Json -Depth 4 } "
        "else { @{alive=$false} | ConvertTo-Json }"
    )
    if isinstance(data, dict):
        return data
    return {"ProcessId": pid, "alive": False}


def process_command_line(pid: int) -> str:
    data = ps_json(f"Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" | Select-Object -ExpandProperty CommandLine | ConvertTo-Json")
    if data:
        return str(data)
    path = STAGE / SUPERVISOR_LAUNCH_LEDGER
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        rows = []
    for row in reversed(rows):
        if str(row.get("pid") or "") == str(pid) and row.get("event") == "SUPERVISOR_STARTED":
            return str(row.get("command") or "")
    return ""


def supervisor_gate_config(supervisor_pid: int, heartbeat: Mapping[str, Any]) -> supervisor_api.GateConfig:
    command_line = process_command_line(supervisor_pid) if supervisor_pid else ""

    def numeric_arg(name: str, default: float) -> float:
        value = supervisor_api.command_arg(command_line, name)
        if value in (None, ""):
            value = heartbeat.get(name.lstrip("-").replace("-", "_"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return supervisor_api.GateConfig(
        max_active_model_processes=int(numeric_arg("--max-active-model-processes", float(heartbeat.get("max_active_model_processes", 3) or 3))),
        max_policy_workers=int(numeric_arg("--max-policy-workers", float(heartbeat.get("max_policy_workers", 2) or 2))),
        admission_commit_percent=numeric_arg("--admission-commit-percent", 85.0),
        max_system_commit_percent=numeric_arg("--max-system-commit-percent", 90.0),
        min_available_ram_bytes=int(numeric_arg("--min-available-ram-gb", 6.0) * 1024**3),
    )


def task_state(name: str) -> str:
    data = ps_json(f"Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue | Select-Object TaskName,State | ConvertTo-Json")
    if isinstance(data, dict):
        return str(data.get("State"))
    return "Absent"


def supervisor_autostart_state() -> dict[str, Any]:
    authority = read_json(STAGE / "R46_user_autostart_authority.json")
    validation = read_json(STAGE / "R46_autostart_registration_validation.json")
    startup_entry = Path(str(authority.get("startup_entry_path") or validation.get("startup_entry_path") or ""))
    launcher = Path(str(authority.get("launcher_path") or validation.get("launcher_path") or ""))
    active = bool(
        authority
        and validation.get("classification") == supervisor_api.R46_CLASSIFICATION
        and startup_entry.exists()
        and launcher.exists()
    )
    return {
        "state": "USER_STARTUP_ACTIVE" if active else "USER_STARTUP_INACTIVE",
        "mechanism": authority.get("mechanism") or validation.get("mechanism") or "",
        "launcher_sha256": authority.get("launcher_sha256") or validation.get("launcher_sha256") or "",
        "singleton_protection": "LAUNCHER_PREFLIGHT_ACTIVE" if launcher.exists() else "LAUNCHER_MISSING",
    }


def latest_classification() -> str:
    for name in (
        "R7_R40_terminal_result.json",
        R37_TERMINAL_NAME,
        R36_TERMINAL_NAME,
        R35_TERMINAL_NAME,
        "R7_R34_09_terminal_validation.json",
        "R7_R33_13_terminal_validation.json",
        "R7_R32_10_terminal_validation.json",
        "R7_R31_12_terminal_validation.json",
        "R7_R30_09_terminal_validation.json",
        "R7_R27_14_terminal_validation.json",
    ):
        payload = read_json(STAGE / name)
        if payload.get("classification"):
            return str(payload["classification"])
    return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--eta-window-minutes", type=float, default=60.0)
    return parser.parse_args()


def r35_eta_text() -> list[str]:
    eta = read_json(STAGE / R35_ETA_NAME)
    readiness = read_json(STAGE / R35_READINESS_NAME)
    watchdog = read_json(STAGE / R35_WATCHDOG_NAME)
    lines: list[str] = []
    rows = eta.get("eta", []) if isinstance(eta.get("eta"), list) else []
    if rows:
        parts = []
        for row in rows:
            status = row.get("status")
            central = row.get("central_hours")
            confidence = row.get("confidence")
            if central is None:
                parts.append(f"{row.get('family')}={status}/{confidence}")
            else:
                parts.append(f"{row.get('family')}={central}h/{confidence}")
        lines.append("R35 ETA: " + "; ".join(parts))
    families = readiness.get("families", []) if isinstance(readiness.get("families"), list) else []
    if families:
        counts: dict[str, int] = {}
        for row in families:
            state = str(row.get("readiness_state", "UNKNOWN"))
            counts[state] = counts.get(state, 0) + 1
        next_family = read_json(STAGE / "R7_R35_09_post_r34_queue_manifest.json").get("first_launch_when_current_slot_frees", "")
        lines.append("R35 readiness: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) + f" | next_post_r34={next_family}")
    if watchdog:
        lines.append(
            "R35 watchdog: "
            f"reboot_logout_risk={watchdog.get('reboot_or_logout_risk', 'UNKNOWN')} "
            f"ds26_interference={watchdog.get('ds26_can_interfere', 'UNKNOWN')}"
        )
    return lines


def v3_candidate_roots(family: str) -> list[Path]:
    family_root = STAGE / "r7_r14_policy_workers" / family
    names: list[str] = []
    for filename in ("worker_inventory.json", "progress.json", "initialization_telemetry.json"):
        name = read_json(family_root / filename).get("metrics_root_name")
        if str(name).startswith("metrics_only_v3"):
            names.append(str(name))
    names.extend(["metrics_only_v3_r37_huber_replay", "metrics_only_v3_r37_rff_replay", "metrics_only_v3_r37_mlp_replay", "metrics_only_v3", "metrics_only_v3_r36_replay"])
    if family_root.exists():
        names.extend(path.name for path in family_root.glob("metrics_only_v3*") if path.is_dir())
    roots: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        root = family_root / name
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key in seen:
            continue
        seen.add(key)
        if root.exists():
            roots.append(root)
    return roots


def active_v3_root(family: str) -> Path:
    roots = v3_candidate_roots(family)
    for root in roots:
        lease = read_json(root / "namespace_writer_lease.json")
        pid = int(lease.get("pid", 0) or 0)
        if pid and process_status(pid).get("alive"):
            return root
    for root in roots:
        if (root / V3_SUMMARY_NAME).exists() or (root / V3_CHECKPOINT_NAME).exists():
            return root
    for root in roots:
        if (root / "resolved_performance_contract_v3.json").exists():
            return root
    return STAGE / "r7_r14_policy_workers" / family / "metrics_only_v3"


def family_summary(family: str) -> dict[str, Any]:
    v3_root = active_v3_root(family)
    v3 = read_json(v3_root / V3_SUMMARY_NAME)
    if v3:
        return {"_summary_version": "resolved_performance_v3", "_summary_root": str(v3_root), **v3}
    if (
        (v3_root / "resolved_performance_contract_v3.json").exists()
        or (v3_root / supervisor_api.NAMESPACE_WRITER_LEASE_NAME).exists()
    ):
        return {
            "_summary_version": "resolved_performance_v3",
            "_summary_root": str(v3_root),
            "rank_ic": {},
            "returns": {},
            "coverage": {},
            "status": "AWAITING_FIRST_V3_COMMIT",
        }
    v2 = read_json(STAGE / "r7_r14_policy_workers" / family / "metrics_only" / V2_SUMMARY_NAME)
    if v2:
        return {"_summary_version": "resolved_performance_v2", **v2}
    direct = STAGE / "r7_r14_policy_workers" / family / SUMMARY_NAME
    stage_names = {
        "ridge_policy_v1_control": "R7_R31_06_performance_summary_ridge.json",
        "spline_additive_ridge": "R7_R31_07_performance_summary_spline.json",
        "pca_ridge_policy_v1_control": "R7_R31_08_performance_summary_pca.json",
    }
    for path in (direct, STAGE / stage_names.get(family, "")):
        if path.name:
            payload = read_json(path)
            if payload:
                return payload
    return {}


def certified_queue_monitor_state(board: list[dict[str, Any]], blocked: list[str]) -> dict[str, Any]:
    try:
        queue = supervisor_api.validate_ready_family_queue_manifest(supervisor_api.R42_READY_QUEUE_PATH)
        ownership = supervisor_api.validate_cross_host_ownership_authority(supervisor_api.R44_CROSS_HOST_OWNERSHIP_PATH)
        plan = supervisor_api.certified_queue_admission_plan(
            board,
            ready_family_queue_manifest=supervisor_api.R42_READY_QUEUE_PATH,
            cross_host_ownership_manifest=supervisor_api.R44_CROSS_HOST_OWNERSHIP_PATH,
            retired_families_path=supervisor_api.R54_RETIRED_FAMILIES_PATH,
        )
    except Exception as exc:
        return {
            "authority": "R42_INVALID",
            "error": f"{type(exc).__name__}:{exc}",
            "remaining": 0,
            "next_family": "",
            "next_route": "",
            "global_block": bool(blocked),
            "family_specific_blocks": [],
            "cross_host_authority": "R44_INVALID",
            "cross_host_error": f"{type(exc).__name__}:{exc}",
        }
    next_family = str(plan.get("first_eligible_family", ""))
    route = supervisor_api.execution_registry_row(next_family) if next_family else {}
    fallback = ""
    if any(row.get("family") == "elastic_net" for row in plan.get("skipped_families", [])):
        for row in plan.get("eligible_families", []):
            if row.get("family") != "elastic_net":
                fallback = str(row.get("family", ""))
                break
    return {
        "authority": queue["authority"],
        "manifest_hash": queue["manifest_hash"],
        "cross_host_authority": ownership["authority"],
        "cross_host_manifest_hash": ownership["manifest_hash"],
        "excluded_mac_owned": ownership["excluded_mac_owned"],
        "excluded_mac_reserved": ownership["excluded_mac_reserved"],
        "excluded_families": plan.get("excluded_families", []),
        "retired_family_authority": plan.get("retired_family_authority", {}),
        "dell_ready_remaining": len(plan.get("eligible_families", [])),
        "remaining": len(plan.get("eligible_families", [])),
        "next_family": next_family,
        "next_route": route.get("worker_kind", ""),
        "next_fallback": fallback,
        "global_block": bool(blocked),
        "global_block_reasons": blocked,
        "family_specific_blocks": plan.get("skipped_families", []),
    }


def ownership_monitor_rows() -> dict[str, Any]:
    r49 = read_json(STAGE / "R49_cross_host_ownership_state.json")
    by_family = r49.get("by_family")
    if isinstance(by_family, dict) and by_family:
        return by_family
    r47a = read_json(STAGE / "R47A_cross_host_ownership_state.json")
    by_family = r47a.get("by_family")
    if isinstance(by_family, dict) and by_family:
        return by_family
    r47 = read_json(STAGE / "R47_cross_host_ownership_state.json")
    by_family = r47.get("by_family")
    if isinstance(by_family, dict) and by_family:
        return by_family
    return (
        supervisor_api.validate_cross_host_ownership_authority(supervisor_api.R44_CROSS_HOST_OWNERSHIP_PATH)
        .get("by_family", {})
    )


def display_ownership_state(family: str, row: Mapping[str, Any], ownership: Mapping[str, Any]) -> str:
    owner_state = str(ownership.get("owner_state", ""))
    execution_owner = str(ownership.get("execution_owner", ""))
    state = str(row.get("state", ""))
    if owner_state == "COMPLETE_IMPORTED":
        return "COMPLETE_IMPORTED (Mac)"
    if state == "COMPLETE":
        return "COMPLETE_LOCAL"
    if owner_state == "MAC_RUNNING":
        return "MAC_RUNNING"
    if owner_state == "MAC_COMPLETE":
        return "MAC_COMPLETE"
    if owner_state == "MAC_RESERVED_NEXT":
        return "MAC_RESERVED"
    if execution_owner == "DELL" and row.get("pid_alive"):
        return "DELL_RUNNING"
    if execution_owner == "DELL":
        return "DELL_READY"
    return ""


def compact_route_label(route: object) -> str:
    if route == "TABULAR":
        return "classical"
    return str(route or "")


def family_validation_state(family: str) -> str:
    payload = read_json(STAGE / "r7_r14_policy_workers" / family / "metrics_only" / "summary_validation_state_r33.json")
    return str(payload.get("status") or payload.get("validation_state") or "")


def summary_is_r33_quarantined(family: str, payload: Mapping[str, Any]) -> bool:
    marker = payload.get("r33_quarantine", {})
    accepted = payload.get("accepted_performance_results", {})
    return (
        family_validation_state(family) == R33_QUARANTINED_STATUS
        or payload.get("status") == R33_QUARANTINED_STATUS
        or (isinstance(marker, Mapping) and marker.get("status") == R33_QUARANTINED_STATUS)
        or (isinstance(accepted, Mapping) and accepted.get("status") == R33_QUARANTINED_STATUS)
    )


def display_metric(quarantined: bool, value: Any) -> Any:
    return QUARANTINED_DISPLAY_VALUE if quarantined else value


def compact_summary_text(family: str) -> str:
    payload = family_summary(family)
    if not payload:
        return ""
    if payload.get("_summary_version") == "resolved_performance_v3":
        root = Path(str(payload.get("_summary_root")))
        checkpoint = read_json(root / V3_CHECKPOINT_NAME)
        ic = payload.get("rank_ic", {})
        ci = ic.get("dependence_aware_95_ci") or {}
        returns = payload.get("returns", {})
        coverage = payload.get("coverage", {})
        lease = read_json(root / supervisor_api.NAMESPACE_WRITER_LEASE_NAME)
        return (
            " | v3="
            f"contract={checkpoint.get('evaluation_contract_version')} "
            f"lease_pid={lease.get('pid', '')} "
            f"lease_gen={lease.get('lease_generation', '')} "
            f"lease_phase={lease.get('phase', '')} "
            f"lease_cursor={lease.get('cursor', '')} "
            f"lease_heartbeat={lease.get('heartbeat_utc', '')} "
            f"raw_metric_rows={checkpoint.get('raw_metric_rows', '')} "
            f"resolved_rows={checkpoint.get('resolved_performance_rows')} "
            f"valid_ic_ts={checkpoint.get('rank_ic_valid_rows')} "
            f"mean_ic={ic.get('mean_spearman_rank_ic')} "
            f"ci=({ci.get('lower')},{ci.get('upper')}) "
            f"daily_net={returns.get('last_daily_net_return')} "
            f"cum_net={returns.get('cumulative_net_return')} "
            f"daily_sharpe={returns.get('daily_sharpe')} "
            f"max_dd={returns.get('maximum_drawdown')} "
            f"pending={coverage.get('pending_score_rows')} "
            f"censored={coverage.get('terminal_censored_rows')} "
            f"status={payload.get('status')} "
            f"durable={checkpoint.get('durable_metrics_bytes')} "
            f"tmp={checkpoint.get('current_temporary_bytes')} "
            f"pending_bytes={checkpoint.get('pending_buffer_bytes')} "
            f"disk_growth={checkpoint.get('disk_growth_since_launch_bytes', '')} "
            f"parts={checkpoint.get('partition_count')} "
            f"dupes={checkpoint.get('duplicate_count')} "
            f"daily_refits={checkpoint.get('daily_refit_count')} "
            f"last_refit_session={checkpoint.get('last_refit_session')}"
        )
    if payload.get("_summary_version") == "resolved_performance_v2":
        checkpoint = read_json(STAGE / "r7_r14_policy_workers" / family / "metrics_only" / V2_CHECKPOINT_NAME)
        ic = payload.get("rank_ic", {})
        returns = payload.get("returns", {})
        quarantined = summary_is_r33_quarantined(family, payload)
        return (
            " | v2="
            f"contract={checkpoint.get('evaluation_contract_version')} "
            f"resolved_rows={checkpoint.get('resolved_performance_rows')} "
            f"valid_ic={checkpoint.get('rank_ic_valid_rows')} "
            f"pending={checkpoint.get('pending_score_rows')} "
            f"pending_ts={checkpoint.get('pending_timestamp_count')}/{checkpoint.get('pending_timestamp_limit')} "
            f"oldest_pending={checkpoint.get('oldest_pending_timestamp')} "
            f"censored={checkpoint.get('terminal_censored_rows')} "
            f"tmp={checkpoint.get('current_temporary_bytes')} "
            f"peak_tmp={checkpoint.get('peak_temporary_bytes')} "
            f"durable={checkpoint.get('durable_metrics_bytes')} "
            f"rank_ic={display_metric(quarantined, ic.get('mean_spearman_rank_ic'))} "
            f"net={display_metric(quarantined, returns.get('cumulative_net_return'))} "
            f"sharpe={display_metric(quarantined, returns.get('sharpe', returns.get('sharpe_ratio')))} "
            f"max_dd={display_metric(quarantined, returns.get('maximum_drawdown', returns.get('max_drawdown')))} "
            f"status={payload.get('status')}"
            f"{' validation=' + family_validation_state(family) if family_validation_state(family) else ''}"
        )
    ic = payload.get("information_coefficient", {})
    port = payload.get("portfolio_performance", {}).get("top_n_equal_weight", {})
    coverage = payload.get("coverage", {})
    ci = ic.get("newey_west_95_ci") or {}
    quarantined = summary_is_r33_quarantined(family, payload)
    return (
        " | summary="
        f"rank_ic={display_metric(quarantined, ic.get('mean_spearman_rank_ic'))} "
        f"ci=({display_metric(quarantined, ci.get('lower'))},{display_metric(quarantined, ci.get('upper'))}) "
        f"return={display_metric(quarantined, port.get('cumulative_net_return', port.get('cumulative_return')))} "
        f"net={display_metric(quarantined, port.get('cumulative_net_return', port.get('net_return')))} "
        f"sharpe={display_metric(quarantined, port.get('sharpe_ratio', port.get('sharpe')))} "
        f"drawdown={display_metric(quarantined, port.get('maximum_drawdown', port.get('max_drawdown')))} "
        f"max_dd={display_metric(quarantined, port.get('maximum_drawdown', port.get('max_drawdown')))} "
        f"resolved={port.get('resolved_portfolio_observations') or coverage.get('resolved_outcome_count')} "
        f"censored={coverage.get('censored_terminal_count')}"
        f"{' status=' + str(payload.get('status')) if payload.get('status') else ''}"
    )


def main() -> int:
    args = parse_args()
    heartbeat = read_json(STAGE / "R7_R27_01_supervisor_heartbeat.json")
    lease = read_json(STAGE / "R7_R27_tournament_supervisor.lease.json")
    heartbeat_pid = int(heartbeat.get("pid") or 0)
    lease_pid = int(lease.get("pid") or 0)
    lease_proc = process_status(lease_pid) if lease_pid else {"alive": False}
    heartbeat_proc = process_status(heartbeat_pid) if heartbeat_pid else {"alive": False}
    supervisor_pid = lease_pid if lease_proc.get("alive") else heartbeat_pid
    supervisor_proc = lease_proc if lease_proc.get("alive") else heartbeat_proc
    gate_config = supervisor_gate_config(supervisor_pid, heartbeat) if supervisor_proc.get("alive") else supervisor_api.GateConfig()
    gate: dict[str, Any] = {}
    try:
        if args.compact:
            queue = supervisor_api.validate_ready_family_queue_manifest(supervisor_api.R42_READY_QUEUE_PATH)
            board = supervisor_api.lightweight_certified_queue_board(queue["ready_family_queue"])
        else:
            board = supervisor_api.build_family_board()
        gate = supervisor_api.resource_gate(board, gate_config)
        resources = {"resource_snapshot": gate.get("resource_snapshot", {})}
    except Exception:
        board = read_json(STAGE / "R7_R27_02_family_state_board.json").get("families", [])
        resources = read_json(STAGE / "R7_R27_05_resource_snapshot.json")
    exact = gate.get("protected_exact", {}) or heartbeat.get("exact_worker", {})
    disk = shutil.disk_usage(str(ROOT.anchor or "C:\\"))
    guard = heartbeat.get("zero_full_prediction_guard", {})
    print(f"DS24 Full-Family Tournament | {latest_classification()}")
    print(
        f"Supervisor PID {supervisor_pid}: {'alive' if supervisor_proc.get('alive') else 'dead'} "
        f"heartbeat_pid={heartbeat_pid} heartbeat_age={heartbeat_age_seconds(heartbeat.get('heartbeat_utc'))}"
    )
    print(f"Supervisor task: {task_state('DreamSystem_DS24_TournamentSupervisor')} | DS26 task: {task_state('DreamSystem_DS26_ProspectiveNewsCapture')}")
    autostart = supervisor_autostart_state()
    print(
        "Supervisor autostart: "
        f"{autostart.get('state')} "
        f"mechanism={autostart.get('mechanism') or 'none'} "
        f"launcher_hash={autostart.get('launcher_sha256') or 'none'} "
        f"singleton={autostart.get('singleton_protection')}"
    )
    print(f"Lease owner: {lease.get('pid')}")
    print(f"Active model processes: {gate.get('active_model_processes', heartbeat.get('active_model_processes'))} / {heartbeat.get('max_active_model_processes')}")
    exact_state = "COMPLETE" if exact.get("terminal_complete") else ("alive" if exact.get("pid_alive") else "dead")
    print(f"Exact PID {exact.get('pid')}: {exact_state} metric_rows={exact.get('metrics_rows')} topn_rows={exact.get('topn_rows')} cursor={exact.get('cursor')}")
    print(f"Resource block: {gate.get('blocked_reasons', heartbeat.get('resource_block_reason'))}")
    snap = resources.get("resource_snapshot", heartbeat.get("resource_snapshot", {}))
    print(
        "Resources: "
        f"commit={snap.get('system_commit_percent')}% "
        f"available_ram={snap.get('available_ram_bytes')} "
        f"free_disk={snap.get('disk_free_bytes', disk.free)} bytes "
        f"free_disk_gib={round((snap.get('disk_free_bytes', disk.free) or 0) / 1024**3, 3)}"
    )
    terminal = (
        read_json(STAGE / R36_TERMINAL_NAME)
        or read_json(STAGE / "R7_R33_13_terminal_validation.json")
        or read_json(STAGE / "R7_R32_10_terminal_validation.json")
        or read_json(STAGE / "R7_R31_12_terminal_validation.json")
        or read_json(STAGE / "R7_R27_14_terminal_validation.json")
    )
    terminal_guard = terminal.get("guard", {}) if isinstance(terminal.get("guard"), dict) else {}
    paper_orders = terminal.get("paper_orders", terminal_guard.get("paper_orders"))
    live_orders = terminal.get("live_orders", terminal_guard.get("live_orders"))
    holdout_accessed = terminal.get("holdout_accessed", terminal_guard.get("holdout_accessed"))
    print(f"Zero-full-prediction guard: {guard} | paper={paper_orders} live={live_orders} holdout={holdout_accessed}")
    print(f"Cleanup active: {read_json(STAGE / 'R7_R27_10_cleanup_ledger_reference.json').get('cleanup_active')}")
    blocked = gate.get("blocked_reasons", heartbeat.get("resource_block_reason", []))
    certified = certified_queue_monitor_state(board, blocked)
    next_family = "" if blocked else (certified.get("next_family") or (supervisor_api.next_ready_family(board) if board else heartbeat.get("next_ready_family")))
    print(f"Next automatic family: {next_family or 'none'}")
    print(
        "Certified queue authority: "
        f"{certified.get('authority')} hash={certified.get('manifest_hash', '')} "
        f"remaining={certified.get('remaining')} "
        f"next={certified.get('next_family') or 'none'} "
        f"route={certified.get('next_route') or 'none'}"
    )
    print(
        "Certified queue blocks: "
        f"GLOBAL_RESOURCE_BLOCK={certified.get('global_block')} "
        f"reasons={certified.get('global_block_reasons', [])} "
        f"FAMILY_SPECIFIC_BLOCK={len(certified.get('family_specific_blocks', []))}"
    )
    print(
        "Cross-host authority: "
        f"{certified.get('cross_host_authority', 'R44')} "
        f"hash={certified.get('cross_host_manifest_hash', '')} "
        f"Mac owned: {', '.join(certified.get('excluded_mac_owned', [])) or 'none'} "
        f"Mac reserved: {', '.join(certified.get('excluded_mac_reserved', [])) or 'none'} "
        f"Operator excluded: {', '.join(certified.get('excluded_families', [])) or 'none'} "
        f"Dell ready remaining: {certified.get('dell_ready_remaining', certified.get('remaining'))} "
        f"Next Dell family: {certified.get('next_family') or 'none'} "
        f"Next Dell route: {compact_route_label(certified.get('next_route')) or 'none'}"
    )
    if certified.get("next_fallback"):
        print(f"Next Dell fallback: {certified.get('next_fallback')}")
    ownership_rows = ownership_monitor_rows()
    if args.eta_window_minutes:
        print(f"ETA window: {args.eta_window_minutes:g} minutes")
    for line in r35_eta_text():
        print(line)
    print("Families:")
    for row in board:
        line = (
            f"  {row.get('family')}: {row.get('state')} "
            f"pid={row.get('pid')} alive={row.get('pid_alive')} metric_rows={row.get('metrics_rows')} topn_rows={row.get('topn_rows')} "
            f"cursor={row.get('cursor')} lease={row.get('namespace_lease_state', '')}/{row.get('namespace_lease_pid', '')} "
            f"stderr={row.get('stderr_state')} eval={row.get('evaluation_contract_version') or ''} "
            f"resolved={row.get('resolved_performance_rows', 0)} ic_ts={row.get('rank_ic_valid_rows', 0)} pending={row.get('pending_score_rows', 0)}"
        )
        ownership = ownership_rows.get(str(row.get("family")), {})
        if ownership:
            display_state = display_ownership_state(str(row.get("family")), row, ownership)
            line += (
                f" readiness={ownership.get('readiness_state', '')} "
                f"owner={ownership.get('execution_owner', '')}/{ownership.get('owner_state', '')} "
                f"dell_eligible={ownership.get('dell_eligible')}"
                f"{' display=' + display_state if display_state else ''}"
            )
        if not args.compact:
            line += (
                f" terminal={row.get('registered_terminal_cursor')} "
                f"lease_phase={row.get('namespace_lease_phase', '')} lease_cursor={row.get('namespace_lease_cursor', '')} "
                f"tmp={row.get('current_temporary_bytes', 0)} peak_tmp={row.get('peak_temporary_bytes', 0)} "
                f"durable={row.get('durable_metrics_bytes', 0)}"
                f"{compact_summary_text(str(row.get('family')))}"
            )
        print(line)
    print(f"Monitor command: python scripts\\local\\monitor_ds24_full_family_tournament.py --compact --eta-window-minutes {args.eta_window_minutes:g}")
    print(f"Supervisor status: python {SUPERVISOR} --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
