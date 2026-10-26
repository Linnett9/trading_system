"""Safety-gated exact-path SIP retirement orchestration."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from application.services.hf_data_cutover_service import (
    CutoverAuditError,
    CutoverPaths,
    _load_json,
    atomic_csv,
    atomic_json,
    utc_now,
)
from core.research.market_data_cutover import canonical_sha256
from infrastructure.data.market_data_cutover_audit import (
    directory_stats,
    resource_state,
    scheduled_task_details,
    sha256_file,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _directory_identity(path: Path) -> tuple[Any, str]:
    stats = directory_stats(path)
    material = f"{path.resolve()}|{stats.bytes}|{stats.files}|{stats.maximum_mtime_ns}".encode("utf-8")
    return stats, hashlib.sha256(material).hexdigest()


def _checkpoint_inventory(repository_root: Path) -> dict[str, Any]:
    roots = [repository_root / "artifacts", repository_root / "data/research"]
    rows = []
    for root in roots:
        stats = directory_stats(root)
        rows.append({"path": str(root.resolve()), "exists": root.exists(), "bytes": stats.bytes, "files": stats.files, "maximum_mtime_ns": stats.maximum_mtime_ns})
    return {"roots": rows, "identity": canonical_sha256(rows)}


def _validate_capsule(paths: CutoverPaths) -> dict[str, Any]:
    manifest_path = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE_MANIFEST.json"
    manifest = _load_json(manifest_path)
    failures = []
    for artifact in manifest.get("artifacts", []):
        artifact_path = paths.output_root / str(artifact["path"])
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            failures.append(str(artifact_path))
    if failures:
        raise CutoverAuditError(f"Science capsule validation failed: {failures}")
    return {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path), "artifact_count": len(manifest["artifacts"]), "total_bytes": int(manifest["total_bytes"]), "all_hashes_verified": True}


def _active_consumer_gate(state: Mapping[str, Any], candidates: Sequence[Path]) -> dict[str, Any]:
    candidate_text = [str(path.resolve()).lower() for path in candidates]
    protected = []
    blockers = []
    for process in state.get("relevant_processes", []):
        command = str(process.get("command_line") or "")
        lowered = command.lower()
        if "hf_data_cutover_r1.py" in lowered:
            continue
        record = {"pid": process.get("pid"), "command_line": command}
        if any(value in lowered for value in candidate_text) or any(
            token in lowered
            for token in (
                "ds24_clean",
                "ds24_p8",
                "historical_bar_backfill",
                "hf_alpha_sip_r2",
                "canonical_5m_feature_authority",
                "five_minute_targets",
            )
        ):
            blockers.append(record)
        elif any(token in lowered for token in ("ds26", "paper", "parity")):
            protected.append(record)
    return {"passed": not blockers, "blocking_consumers": blockers, "protected_unrelated_processes": protected}


def execute_retirement(paths: CutoverPaths, *, expected_intent_sha256: str) -> dict[str, Any]:
    """Delete only the immutable, exact paths in the approved retirement intent."""

    intent_path = paths.output_root / "SIP_RETIREMENT_INTENT.csv"
    actual_intent_sha256 = sha256_file(intent_path)
    if actual_intent_sha256.lower() != expected_intent_sha256.lower():
        raise CutoverAuditError(f"Retirement intent hash mismatch: {actual_intent_sha256}")
    rows = _read_csv(intent_path)
    if len(rows) != 6:
        raise CutoverAuditError(f"Expected exactly six retirement candidates, found {len(rows)}")
    data_root = (paths.repository_root / "data").resolve()
    candidates: list[Path] = []
    for row in rows:
        candidate = Path(row["exact_path"]).resolve()
        if data_root not in candidate.parents or candidate == data_root:
            raise CutoverAuditError(f"Retirement candidate escaped governed data root: {candidate}")
        if not candidate.is_dir():
            raise CutoverAuditError(f"Retirement candidate missing before delete: {candidate}")
        stats, identity = _directory_identity(candidate)
        if stats.bytes != int(row["bytes"]) or stats.files != int(row["files"]) or identity != row["pre_delete_hash_or_identity"]:
            raise CutoverAuditError(f"Retirement candidate changed after intent freeze: {candidate}")
        candidates.append(candidate)

    capsule = _validate_capsule(paths)
    ledger = paths.repository_root / "docs/research/results/Trading_Research_Evidence_Register.xlsx"
    ledger_sha256 = sha256_file(ledger)
    checkpoint_inventory = _checkpoint_inventory(paths.repository_root)
    before = resource_state(repository_root=paths.repository_root)
    before["captured_at_utc"] = utc_now()
    consumer_gate = _active_consumer_gate(before, candidates)
    if not consumer_gate["passed"]:
        raise CutoverAuditError(f"Active SIP consumer gate failed: {consumer_gate['blocking_consumers']}")

    watcher_name = "HF_ALPHA_SIP_R2_20261025T125948Z_LOCAL_SIP_COLLECTOR"
    watcher_before = scheduled_task_details(watcher_name)
    disable = subprocess.run(
        ["schtasks", "/Change", "/TN", watcher_name, "/Disable"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if watcher_before["present"] and disable.returncode != 0:
        raise CutoverAuditError(f"Failed to disable SIP watcher: {disable.stderr.strip()}")
    watcher_after = scheduled_task_details(watcher_name)

    predelete_gate = {
        "schema_version": "hf_data_cutover_r1.pre_retirement_gate.v1",
        "captured_at_utc": utc_now(),
        "intent_path": str(intent_path.resolve()),
        "intent_sha256": actual_intent_sha256,
        "candidate_count": len(candidates),
        "logical_bytes": sum(int(row["bytes"]) for row in rows),
        "capsule": capsule,
        "consumer_gate": consumer_gate,
        "watcher_before": watcher_before,
        "watcher_disable": {"return_code": disable.returncode, "stdout": disable.stdout.strip(), "stderr": disable.stderr.strip()},
        "watcher_after": watcher_after,
        "ledger_sha256": ledger_sha256,
        "checkpoint_inventory": checkpoint_inventory,
    }
    atomic_json(paths.output_root / "PRE_SIP_RETIREMENT_GATE.json", predelete_gate)

    journal: list[dict[str, Any]] = []
    for row, candidate in zip(rows, candidates, strict=True):
        free_before = int(resource_state(repository_root=paths.repository_root)["disk"]["free_bytes"])
        started_at = utc_now()
        shutil.rmtree(candidate)
        if candidate.exists():
            raise CutoverAuditError(f"Retirement verification failed; path remains: {candidate}")
        free_after = int(resource_state(repository_root=paths.repository_root)["disk"]["free_bytes"])
        journal.append(
            {
                "exact_path": str(candidate),
                "status": "DELETED_AND_VERIFIED_ABSENT",
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "logical_bytes": int(row["bytes"]),
                "files": int(row["files"]),
                "pre_delete_hash_or_identity": row["pre_delete_hash_or_identity"],
                "free_bytes_before": free_before,
                "free_bytes_after": free_after,
                "measured_free_space_change": free_after - free_before,
                "recoverability": "Not recoverable from local SIP data; compact scientific evidence retained in the validated capsule.",
            }
        )
        atomic_csv(paths.output_root / "SIP_RETIREMENT_JOURNAL.csv", journal)

    after = resource_state(repository_root=paths.repository_root)
    after["captured_at_utc"] = utc_now()
    post_consumer_gate = _active_consumer_gate(after, candidates)
    if sha256_file(ledger) != ledger_sha256:
        raise CutoverAuditError("Canonical results ledger changed during SIP retirement")
    if _checkpoint_inventory(paths.repository_root)["identity"] != checkpoint_inventory["identity"]:
        raise CutoverAuditError("Model/checkpoint inventory changed during SIP retirement")
    _validate_capsule(paths)
    if any(candidate.exists() for candidate in candidates):
        raise CutoverAuditError("At least one journaled candidate still exists after retirement")
    logical_bytes = sum(int(row["bytes"]) for row in rows)
    post_state = {
        "schema_version": "hf_data_cutover_r1.post_sip_retirement_state.v1",
        "captured_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "logical_bytes_retired": logical_bytes,
        "starting_free_bytes": int(before["disk"]["free_bytes"]),
        "final_free_bytes": int(after["disk"]["free_bytes"]),
        "measured_free_space_change": int(after["disk"]["free_bytes"]) - int(before["disk"]["free_bytes"]),
        "retained_capsule_bytes": capsule["total_bytes"],
        "net_recovery_after_capsule": int(after["disk"]["free_bytes"]) - int(before["disk"]["free_bytes"]) - int(capsule["total_bytes"]),
        "minimum_final_reserve_bytes": 15 * 1024**3,
        "reserve_gate_passed": int(after["disk"]["free_bytes"]) >= 15 * 1024**3,
        "journal_sha256": sha256_file(paths.output_root / "SIP_RETIREMENT_JOURNAL.csv"),
        "capsule": capsule,
        "ledger_sha256_unchanged": ledger_sha256,
        "checkpoint_inventory_unchanged": checkpoint_inventory,
        "watcher": watcher_after,
        "active_consumer_gate_after": post_consumer_gate,
        "protected_processes_preserved": post_consumer_gate["protected_unrelated_processes"],
    }
    atomic_json(paths.output_root / "POST_SIP_RETIREMENT_STATE.json", post_state)
    run_manifest = _load_json(paths.output_root / "RUN_MANIFEST.json")
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "SIP_RETIRED_IEX_PILOT_PENDING",
            "retirement": {
                "journal_sha256": post_state["journal_sha256"],
                "logical_bytes_retired": logical_bytes,
                "measured_free_space_change": post_state["measured_free_space_change"],
                "final_free_bytes": post_state["final_free_bytes"],
                "watcher_disabled": True,
            },
            "next_gate": "Authenticate Alpaca IEX market-data-only source and run bounded capability pilot.",
        }
    )
    atomic_json(paths.output_root / "RUN_MANIFEST.json", run_manifest)
    return run_manifest


__all__ = ["execute_retirement"]
