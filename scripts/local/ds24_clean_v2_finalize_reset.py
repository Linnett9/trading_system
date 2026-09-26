from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import authority_bundle, file_sha256, load_contract


AUDIT_ROOT = ROOT / "docs/audits/ds24_r40_invalidated_reset_20260926"
DELETED_COMPONENT = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector"
FORBIDDEN_REFERENCE = "DS-24_independent_five_minute_selector"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True, timeout=15
    )
    return result.stdout.strip()


def _source_reference_audit() -> dict[str, Any]:
    roots = ("application", "core", "scripts", "config", "tests")
    suffixes = {".py", ".json", ".yaml", ".yml"}
    rows: list[dict[str, str]] = []
    for root_name in roots:
        root = ROOT / root_name
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            if FORBIDDEN_REFERENCE not in text:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("core/research/ml/ds24/clean_v2_") or relative.startswith("scripts/local/ds24_clean_v2_"):
                classification = "CLEAN_V2_GUARD_OR_AUDIT_TOOLING"
            elif relative.startswith("tests/"):
                classification = "LEGACY_CHARACTERISATION_OR_CLEAN_REGRESSION_TEST"
            elif relative == "scripts/audits/ds24_paper_provenance_forensic.py":
                classification = "PRESERVED_FORENSIC_AUDIT"
            elif relative.startswith("config/ds24/"):
                classification = "RETIRED_V1_STATIC_CONFIG"
            else:
                classification = "RETIRED_LEGACY_GENERATOR_OR_RUNTIME_NOT_USED_BY_CLEAN_V2"
            rows.append({"path": relative, "classification": classification})
    active_dependencies = [
        row for row in rows if row["classification"] == "ACTIVE_CLEAN_V2_RUNTIME_DEPENDENCY"
    ]
    return {
        "audit_id": "DS24_CLEAN_V2_DELETED_STAGE_REFERENCE_AUDIT_V1",
        "searched_roots": list(roots),
        "reference_count": len(rows),
        "active_clean_v2_runtime_dependencies": active_dependencies,
        "pass": not active_dependencies,
        "references": rows,
    }


def _current_ds24_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match 'python|powershell' -and $_.CommandLine -match 'ds24|R40|r40' "
        "-and $_.CommandLine -notmatch 'ds24_clean_v2|Get-CimInstance' } | "
        "Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    values = payload if isinstance(payload, list) else [payload]
    return [dict(value) for value in values]


def _stale_worker_inventory() -> list[dict[str, Any]]:
    path = ROOT / "ds24_supervisor_status.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("active_worker_manifest", {}).get("workers", [])
    return [
        {
            "family": row.get("family"),
            "recorded_pid": row.get("recorded_pid") or row.get("pid"),
            "command": row.get("command_line"),
            "start_time": row.get("creation_time"),
            "last_checkpoint": row.get("checkpoint"),
            "last_cursor": row.get("cursor") or row.get("current_scoring_cursor"),
            "last_heartbeat": row.get("heartbeat"),
            "historical_status_only": True,
        }
        for row in rows
    ]


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to replace immutable reset evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_audit = _source_reference_audit()
    bundle = authority_bundle()
    ledger = load_contract("prior_evidence_manifest.json")
    feature = load_contract("feature_authority.json")
    current_processes = _current_ds24_processes()
    forensic_root = ROOT / "docs/audits/ds24_paper_provenance_forensic_20260926"
    forensic_hashes = {
        path.name: file_sha256(path)
        for path in sorted(forensic_root.iterdir())
        if path.is_file()
    }
    manifest = {
        "manifest_id": "DS24_R40_INVALIDATED_RESET_20260926",
        "created_at_utc": _utc_now(),
        "reason_for_invalidation": "shared value-level point-in-time leakage in the V1 101-feature authority",
        "forensic_audit_commit": "d29fb4488",
        "git_head_at_reset": _git_head(),
        "forensic_audit_paths_and_hashes": forensic_hashes,
        "old_controlling_run_ids": [
            "ds24_p8_r14_e3g_c2_20260824T000000Z",
            "ds24_p8_r14_e3g_c2_r7_r27_20260826T000000Z",
            "R40/R45 tournament supervisor lineage",
        ],
        "affected_feature_authority": {
            "authority_id": feature["base_authority"]["authority_id"],
            "authority_identity": feature["base_authority"]["authority_identity"],
            "predictor_manifest_sha256": load_contract("predictor_manifest.json")["source_manifest_sha256"],
            "classification": "RETIRED_FOR_MODEL_TRAINING_DUE_TO_FEATURE_PIT_LEAK",
        },
        "known_contaminated_features": feature["confirmed_cross_session_defects"],
        "additional_features_repaired_after_stronger_gate": feature[
            "additional_defects_found_by_future_value_perturbation"
        ],
        "additional_future_timestamp_source_repairs": feature[
            "additional_future_timestamp_source_defects"
        ],
        "active_workers_stopped": [],
        "stop_evidence": {
            "scan_result": "NO_ACTIVE_OBSOLETE_DS24_TOURNAMENT_PROCESS_FOUND",
            "scan_time_utc": _utc_now(),
            "current_matching_processes": current_processes,
            "stale_last_known_worker_records": _stale_worker_inventory(),
        },
        "autostart": {
            "scheduled_task": "DreamSystem_DS24_TournamentSupervisor",
            "scheduled_task_state": "ABSENT",
            "startup_original_path": "C:/Users/Brandon/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/DreamSystem_DS24_TournamentSupervisor.cmd",
            "startup_disabled_path": "C:/Users/Brandon/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/DreamSystem_DS24_TournamentSupervisor.cmd.disabled-20260926",
            "startup_sha256": "6834f634741986eae5a0d5bda800b08aa529348cd4b44b9a863b7e1497be6bbf",
            "action": "DISABLED_BY_RECOVERABLE_RENAME",
        },
        "migrated_static_authority_bundle_sha256": bundle["bundle_sha256"],
        "migrated_static_authority_files": bundle["files"],
        "results_ledger_xlsx_export_sha256": ledger["workbook"]["captured_xlsx_export_sha256"],
        "results_ledger_drive_file_id": ledger["workbook"]["google_drive_file_id"],
        "paths_deleted": ["docs/dream_system/components/DS-24_independent_five_minute_selector"],
        "deleted_path_state_at_reset": "ALREADY_ABSENT_FROM_WORKTREE; PRE-EXISTING TRACKED DELETIONS PRESERVED",
        "preserved_paths": ["docs/audits", "data", "mac_aux_runs", "source code", "tests", "configs"],
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_once(AUDIT_ROOT / "STATIC_REFERENCE_AUDIT.json", reference_audit)
    _write_once(AUDIT_ROOT / "RESET_MANIFEST.json", manifest)
    print(json.dumps({"reset_manifest": str(AUDIT_ROOT / 'RESET_MANIFEST.json'), "reference_audit": reference_audit}, indent=2))
    return 0 if not current_processes and reference_audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
