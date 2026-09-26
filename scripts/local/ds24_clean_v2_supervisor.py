from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import authority_bundle, load_contract, stable_hash
from core.research.ml.ds24.clean_v2_runtime import (
    MAX_DELL_MODEL_WORKERS,
    REFIT_POLICY_ID,
    RUN_ID,
    validate_ownership,
)


RUN_ROOT = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID
STATUS_PATH = RUN_ROOT / "supervisor_status.json"
LEASE_PATH = RUN_ROOT / "supervisor_lease.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_inventory_matches(
    root: Path,
    partitions: list[dict[str, Any]],
    *,
    relative_path_key: str,
    bytes_key: str,
) -> bool:
    for partition in partitions:
        if not isinstance(partition, dict):
            return False
        relative = Path(str(partition.get(relative_path_key, "")))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        candidate = root / relative
        try:
            if not candidate.is_file() or candidate.stat().st_size != int(partition[bytes_key]):
                return False
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return True


def preflight(host: str) -> dict[str, Any]:
    feature = load_contract("feature_authority.json")
    target = load_contract("target_contract.json")
    tournament = load_contract("tournament_contract.json")
    models = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    validate_ownership(ownership["hosts"])
    bundle = authority_bundle()
    reasons: list[str] = []

    sidecar_root = ROOT / feature["sidecar"]["path"]
    sidecar_manifest_path = sidecar_root / "authority_manifest.json"
    sidecar_manifest = _read_json(sidecar_manifest_path) if sidecar_manifest_path.is_file() else {}
    expected_rows = int(feature["base_authority"]["physical_snapshot_row_count"])
    sidecar_logical_hash = stable_hash(
        {key: value for key, value in sidecar_manifest.items() if key != "logical_sha256"}
    )
    sidecar_partitions = sidecar_manifest.get("partitions", [])
    base_root = ROOT / feature["base_authority"]["path"]
    if not (
        sidecar_manifest.get("complete") is True
        and sidecar_manifest.get("authority_id") == feature["authority_id"]
        and sidecar_manifest.get("repair_columns") == feature["sidecar"]["repair_columns"]
        and int(sidecar_manifest.get("row_count", -1)) == expected_rows
        and len(sidecar_partitions) == feature["base_authority"]["physical_snapshot_stock_partitions"]
        and _partition_inventory_matches(
            sidecar_root,
            sidecar_partitions,
            relative_path_key="relative_path",
            bytes_key="bytes",
        )
        and _partition_inventory_matches(
            base_root,
            sidecar_partitions,
            relative_path_key="base_relative_path",
            bytes_key="base_bytes",
        )
        and sidecar_manifest.get("logical_sha256") == sidecar_logical_hash
    ):
        reasons.append("V2_SIDECAR_NOT_FULLY_MATERIALIZED")

    certificate_path = RUN_ROOT / "causality_certificate.json"
    certificate = _read_json(certificate_path) if certificate_path.is_file() else {}
    fixture = certificate.get("fixture", {})
    certificate_hash = stable_hash(
        {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    )
    if not (
        certificate.get("passed") is True
        and certificate.get("predictor_count") == 101
        and certificate.get("terminal_classification")
        == "101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION"
        and certificate.get("feature_authority_id") == feature["authority_id"]
        and certificate.get("static_authority_bundle_sha256") == bundle["bundle_sha256"]
        and fixture.get("years") == [2017, 2024]
        and set(fixture.get("symbols", [])) >= {"AAA", "BBB", "SPY", "QQQ", "GLD", "TLT", "XLK"}
        and set(fixture.get("session_types", []))
        >= {"REGULAR", "EARLY_CLOSE", "PRE_MARKET", "AFTER_HOURS"}
        and certificate.get("certificate_sha256") == certificate_hash
    ):
        reasons.append("FUTURE_BAR_PERTURBATION_CERTIFICATE_MISSING_OR_FAILED")
    if target.get("resolved_contract_sha256") != "8e2d5458044c17cf60bc1d8e46b71357599f3837ddb53d64d9b00d322d17f419":
        reasons.append("TARGET_CONTRACT_HASH_MISMATCH")
    target_delta_path = ROOT / target["physical_authority"]["delta_path"] / "authority_manifest.json"
    target_delta = _read_json(target_delta_path) if target_delta_path.is_file() else {}
    target_authority_hash = stable_hash(
        {key: value for key, value in target_delta.items() if key != "logical_sha256"}
    )
    target_partitions = target_delta.get("partitions", [])
    target_base_snapshot = target_delta.get("base_snapshot") or {}
    target_base_partitions = target_base_snapshot.get("partitions", [])
    target_code_hashes = {
        code_hash
        for partition in target_delta.get("partitions", [])
        for code_hash in partition.get("target_code_hashes", [])
    }
    if not (
        target_delta.get("complete") is True
        and target_delta.get("authority_id") == target["authority_id"]
        and target_delta.get("target_id") == target["target_id"]
        and target_delta.get("symbol_count") == feature["base_authority"]["physical_snapshot_asset_count"]
        and len(target_partitions) == target_delta.get("symbol_count")
        and _partition_inventory_matches(
            target_delta_path.parent,
            target_partitions,
            relative_path_key="relative_path",
            bytes_key="bytes",
        )
        and len(target_base_partitions)
        == feature["base_authority"]["physical_snapshot_stock_partitions"]
        and target_base_snapshot.get("aggregate_partition_sha256")
        == stable_hash(target_base_partitions)
        and _partition_inventory_matches(
            ROOT / target["physical_authority"]["base_path"],
            target_base_partitions,
            relative_path_key="base_relative_path",
            bytes_key="base_bytes",
        )
        and target_delta.get("maximum_historical_outcome_timestamp_consumed")
        == target["maximum_historical_outcome_timestamp_consumed"].replace("Z", "+00:00")
        and target_code_hashes == {target["target_code_hash"]}
        and target_delta.get("logical_sha256") == target_authority_hash
    ):
        reasons.append("CLEAN_TARGET_DELTA_NOT_FULLY_MATERIALIZED")

    configured = set(models["families"])
    assigned = set(ownership["hosts"].get(host, []))
    missing_models = sorted(family for family in assigned if family not in configured and family not in models["controls"])
    if missing_models:
        reasons.append(f"OWNED_FAMILY_CONFIG_MISSING:{','.join(missing_models)}")
    worker_commands = tournament.get("worker_commands", {}).get(host, {})
    launch_families = (
        list(tournament["lanes"]["FULL_CLEAN"])
        if host == "dell"
        else [*tournament["lanes"]["SHORT_REQUALIFICATION"], *tournament["lanes"]["UNSCORED_DISCOVERY"]]
    )
    launch_families = [family for family in launch_families if family in assigned]
    commandless = sorted(family for family in launch_families if family not in worker_commands)
    if commandless:
        reasons.append(f"CLEAN_V2_WORKER_COMMANDS_UNRESOLVED:{','.join(commandless)}")

    disk = shutil.disk_usage(ROOT.anchor or ROOT)
    try:
        import psutil

        ram_available_bytes = int(psutil.virtual_memory().available)
    except ImportError:
        ram_available_bytes = None
    required_gib = float(tournament["storage"]["minimum_post_launch_free_gib"])
    required_gib += float(tournament["storage"]["projected_tournament_gib_upper_bound"])
    if not sidecar_manifest.get("complete"):
        required_gib += float(tournament["storage"]["projected_sidecar_gib_upper_bound"])
    if disk.free < int(required_gib * 1024**3):
        reasons.append("INSUFFICIENT_DISK_FOR_DECLARED_RESERVE")

    feature_hash = sidecar_manifest.get("logical_sha256") or stable_hash(feature)
    if not reasons:
        classification = "DS24_CLEAN_V2_READY_FOR_SUPERVISOR_LAUNCH"
    elif "FUTURE_BAR_PERTURBATION_CERTIFICATE_MISSING_OR_FAILED" in reasons:
        classification = "DS24_CLEAN_V2_BLOCKED_FEATURE_CAUSALITY"
    elif "INSUFFICIENT_DISK_FOR_DECLARED_RESERVE" in reasons:
        classification = "DS24_CLEAN_V2_BLOCKED_RESOURCE"
    else:
        classification = "DS24_CLEAN_V2_BLOCKED_DATA_OR_CONFIG_AUTHORITY"
    return {
        "run_id": RUN_ID,
        "host_role": host,
        "hostname": socket.gethostname(),
        "checked_at_utc": _utc_now(),
        "feature_authority_id": feature["authority_id"],
        "feature_authority_hash": feature_hash,
        "target_authority_hash": target_delta.get("logical_sha256"),
        "target_contract_hash": target["resolved_contract_sha256"],
        "static_authority_bundle_sha256": bundle["bundle_sha256"],
        "results_ledger_hash": load_contract("prior_evidence_manifest.json")["workbook"]["captured_xlsx_export_sha256"],
        "refit_policy": REFIT_POLICY_ID,
        "launch_families": launch_families,
        "mac_owned_families": ownership["hosts"]["mac"],
        "maximum_model_workers": MAX_DELL_MODEL_WORKERS if host == "dell" else 1,
        "disk_free_bytes": disk.free,
        "ram_available_bytes": ram_available_bytes,
        "blocking_reasons": reasons,
        "ready": not reasons,
        "classification": classification,
        "paper_orders": 0,
        "live_orders": 0,
    }


def launch(host: str) -> dict[str, Any]:
    report = preflight(host)
    if not report["ready"]:
        _write_json_atomic(STATUS_PATH, report)
        return report
    tournament = load_contract("tournament_contract.json")
    commands = tournament["worker_commands"][host]
    limit = MAX_DELL_MODEL_WORKERS if host == "dell" else 1
    launched: list[dict[str, Any]] = []
    lease = {
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "host_role": host,
        "started_at_utc": _utc_now(),
        "generation": 1,
        "adopted_r40_state": False,
    }
    _write_json_atomic(LEASE_PATH, lease)
    for family in report["launch_families"][:limit]:
        command = list(commands[family])
        process = subprocess.Popen(command, cwd=ROOT)  # noqa: S603 - command is frozen static authority
        launched.append({"family": family, "pid": process.pid, "command": command})
    report.update(
        {
            "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
            "supervisor_pid": os.getpid(),
            "active_workers": launched,
            "ready": True,
        }
    )
    _write_json_atomic(STATUS_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed DS24 clean-V2 supervisor.")
    parser.add_argument("--host", choices=("dell", "mac"), default="dell")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status and STATUS_PATH.is_file():
        report = _read_json(STATUS_PATH)
    else:
        report = launch(args.host) if args.launch else preflight(args.host)
        if not args.launch:
            _write_json_atomic(STATUS_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
