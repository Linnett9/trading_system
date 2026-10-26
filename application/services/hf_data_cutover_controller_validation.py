"""Manifest and integrity gates for HF-DATA-CUTOVER-R1 stage handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from application.services.hf_data_cutover_service import CutoverAuditError
from infrastructure.data.iex_market_data_store import sha256_file


class StageGateError(CutoverAuditError):
    """A stage's durable output does not satisfy its publication contract."""


def validate_raw_authority(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    run = _read_json(audit_root / "RUN_MANIFEST.json")
    acquisition = run.get("iex_raw_backfill") or {}
    five = _read_json(audit_root / "IEX_RAW_5M_MANIFEST.json")
    daily = _read_json(audit_root / "IEX_DAILY_MANIFEST.json")
    _require_raw_manifest(five, int(acquisition["planned_5m_partitions"]), "raw_5m")
    _require_raw_manifest(daily, int(acquisition["planned_daily_partitions"]), "raw_daily")
    five_hash = sha256_file(audit_root / "IEX_RAW_5M_MANIFEST.json")
    daily_hash = sha256_file(audit_root / "IEX_DAILY_MANIFEST.json")
    if acquisition.get("raw_5m_manifest_sha256") != five_hash or acquisition.get("daily_manifest_sha256") != daily_hash:
        raise StageGateError("RUN_MANIFEST raw authority hashes are absent or mismatched")
    five_partition_hashes = _partition_manifest_hashes(
        repository_root / "data/market/iex/raw/5m", "symbol=*/year=*/manifest.json"
    )
    daily_partition_hashes = _partition_manifest_hashes(
        repository_root / "data/market/iex/raw/1d", "symbol=*/manifest.json"
    )
    if five_partition_hashes != sorted(five.get("partition_manifest_sha256") or []):
        raise StageGateError("Raw 5m partition manifest hashes/counts drifted")
    if daily_partition_hashes != sorted(daily.get("partition_manifest_sha256") or []):
        raise StageGateError("Raw daily partition manifest hashes/counts drifted")
    if run.get("state") == "IEX_RAW_BACKFILL_IN_PROGRESS":
        raise StageGateError("Raw IEX command has not published its successful terminal state")
    return {
        "raw_5m_sha256": five_hash,
        "raw_daily_sha256": daily_hash,
        "partitions": len(five_partition_hashes) + len(daily_partition_hashes),
    }


def validate_adjusted_daily(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "IEX_DAILY_ADJUSTED_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("partition_count") != 514:
        raise StageGateError("Adjusted daily authority is incomplete")
    if manifest.get("feed") != "IEX" or manifest.get("adjustment") != "all":
        raise StageGateError("Adjusted daily feed/adjustment identity drift")
    if manifest.get("identity") != "SEPARATE_ADJUSTED_RESEARCH_VIEW_NOT_RAW_SOURCE_TRUTH":
        raise StageGateError("Adjusted daily authority is not physically identified as separate")
    hashes = _partition_manifest_hashes(repository_root / "data/market/iex/derived/1d_adjusted", "symbol=*/manifest.json")
    if hashes != sorted(manifest.get("partition_manifest_sha256") or []):
        raise StageGateError("Adjusted daily partition manifest hashes/counts drifted")
    return {"manifest_sha256": sha256_file(manifest_path), "partitions": len(hashes), "daily_h5_h10_gate": "PASSED"}


def validate_hourly(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "IEX_1H_DERIVATION_MANIFEST.json"
    manifest = _read_json(manifest_path)
    raw_hash = sha256_file(audit_root / "IEX_RAW_5M_MANIFEST.json")
    if manifest.get("source_raw_5m_manifest_sha256") != raw_hash:
        raise StageGateError("1h source raw authority hash drift")
    expected = int(_read_json(audit_root / "IEX_RAW_5M_MANIFEST.json")["completed_partition_count"])
    if manifest.get("partition_count") != expected:
        raise StageGateError("1h partition count does not match raw 5m authority")
    hashes = _partition_manifest_hashes(repository_root / "data/market/iex/derived/1h", "symbol=*/year=*/manifest.json")
    if hashes != sorted(manifest.get("partition_manifest_sha256") or []):
        raise StageGateError("1h partition manifest hashes/counts drifted")
    return {"manifest_sha256": sha256_file(manifest_path), "source_raw_sha256": raw_hash, "partitions": len(hashes)}


def validate_feature_bank(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "IEX_FEATURE_V1_MANIFEST.json"
    manifest = _read_json(manifest_path)
    spec_path = audit_root / "IEX_FEATURE_BANK_V1_SPEC.json"
    if manifest.get("status") != "COMPLETE" or manifest.get("missing_tables"):
        raise StageGateError("Feature Bank V1 manifest is incomplete")
    if manifest.get("predictor_count") != 80 or manifest.get("feature_spec_sha256") != sha256_file(spec_path):
        raise StageGateError("Feature Bank V1 frozen specification identity drift")
    if manifest.get("targets_read_by_feature_pipeline") is not False:
        raise StageGateError("Feature Bank V1 target-independence gate failed")
    paths = sorted((repository_root / "data/market/iex/features/v1").glob("**/manifest.json"))
    expected = sum(int(row["partitions"]) for row in manifest.get("tables", {}).values())
    if len(paths) != expected:
        raise StageGateError(f"Feature partition inventory mismatch: {len(paths)}/{expected}")
    return {"manifest_sha256": sha256_file(manifest_path), "feature_spec_sha256": sha256_file(spec_path), "partitions": len(paths), "predictors": 80}


def validate_feature_quality(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    report_path = audit_root / "IEX_FEATURE_V1_PRETARGET_VALIDATION.json"
    report = _read_json(report_path)
    if report.get("status") != "PASSED" or report.get("failures"):
        raise StageGateError("Pre-target feature determinism/quality gate did not pass")
    if report.get("target_data_read") is not False or report.get("predictor_count") != 80:
        raise StageGateError("Pre-target feature validation contract drift")
    current_feature_hash = sha256_file(audit_root / "IEX_FEATURE_V1_MANIFEST.json")
    if report.get("feature_manifest_sha256") != current_feature_hash:
        raise StageGateError("Feature authority changed after pre-target validation")
    if not (report.get("determinism_sentinel") or {}).get("feature_exact"):
        raise StageGateError("Feature deterministic replay is not exact")
    feature_root = repository_root / "data/market/iex/features/v1"
    inventory = [
        {"path": path.relative_to(feature_root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(feature_root.glob("**/manifest.json"))
    ]
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if len(inventory) != report.get("partition_count") or inventory_hash != report.get("partition_manifest_set_sha256"):
        raise StageGateError("Feature partition manifests changed after pre-target validation")
    return {"report_sha256": sha256_file(report_path), "feature_manifest_sha256": current_feature_hash, "partitions": report.get("partition_count")}


def validate_targets(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    validate_adjusted_daily(repository_root, audit_root)
    validate_feature_quality(repository_root, audit_root)
    manifest_path = audit_root / "IEX_TARGET_V1_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("target_horizons") != ["H30", "H60", "H5", "H10"]:
        raise StageGateError("Target authority horizons/status drift")
    if manifest.get("maturity_violation_count") or manifest.get("duplicate_identity_count"):
        raise StageGateError("Target maturity or identity gate failed")
    if manifest.get("feature_manifest_sha256_at_target_build") != sha256_file(audit_root / "IEX_FEATURE_V1_MANIFEST.json"):
        raise StageGateError("Target authority feature lineage hash drift")
    expected_intraday = int(_read_json(audit_root / "IEX_RAW_5M_MANIFEST.json")["completed_partition_count"])
    expected_daily = int(_read_json(audit_root / "IEX_DAILY_ADJUSTED_MANIFEST.json")["partition_count"])
    table_counts = manifest.get("table_counts") or {}
    if (table_counts.get("intraday_h30_h60") or {}).get("partitions") != expected_intraday:
        raise StageGateError("H30/H60 target partition count drift")
    if (table_counts.get("daily_h5_h10") or {}).get("partitions") != expected_daily:
        raise StageGateError("H5/H10 target partition count drift")
    hashes = _partition_manifest_hashes(repository_root / "data/market/iex/targets/v1", "**/manifest.json")
    if hashes != sorted(manifest.get("partition_manifest_sha256") or []):
        raise StageGateError("Target partition manifest hashes/counts drifted")
    return {"manifest_sha256": sha256_file(manifest_path), "partitions": len(hashes), "horizons": manifest["target_horizons"]}


def validate_final_closeout(_repository_root: Path, audit_root: Path) -> dict[str, Any]:
    run = _read_json(audit_root / "RUN_MANIFEST.json")
    quality_path = audit_root / "IEX_DATA_QUALITY_REPORT.json"
    migration_path = audit_root / "SIP_IEX_FEATURE_MIGRATION_REPORT.json"
    final_path = audit_root / "FINAL_REPORT.md"
    quality = _read_json(quality_path)
    validation = run.get("validation") or {}
    expected = {
        "quality_report_sha256": sha256_file(quality_path),
        "migration_report_sha256": sha256_file(migration_path),
        "final_report_sha256": sha256_file(final_path),
    }
    if run.get("state") != "IEX_DATA_AND_FEATURE_V1_READY" or quality.get("all_required_gates_passed") is not True:
        raise StageGateError("Final cutover validation has not reached the required terminal state")
    if any(validation.get(key) != value for key, value in expected.items()):
        raise StageGateError("Final closeout hash evidence drifted")
    return {**expected, "terminal_state": "IEX_DATA_AND_FEATURE_V1_READY"}


def _require_raw_manifest(manifest: Mapping[str, Any], expected: int, dataset: str) -> None:
    if manifest.get("dataset") != dataset or manifest.get("feed") != "IEX" or manifest.get("adjustment") != "raw":
        raise StageGateError(f"{dataset} authority identity drift")
    if manifest.get("planned_partition_count") != expected or manifest.get("completed_partition_count") != expected:
        raise StageGateError(f"{dataset} partition count is incomplete")
    if manifest.get("failed_partition_count") or manifest.get("duplicate_count") or manifest.get("off_grid_count"):
        raise StageGateError(f"{dataset} integrity gates failed")
    if len(manifest.get("partition_manifest_sha256") or []) != expected:
        raise StageGateError(f"{dataset} partition manifest hash count drift")


def _partition_manifest_hashes(root: Path, pattern: str) -> list[str]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise StageGateError(f"No partition manifests found below {root}")
    return sorted(sha256_file(path) for path in paths)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageGateError(f"Expected JSON object: {path}")
    return payload


__all__ = [
    "StageGateError",
    "validate_adjusted_daily",
    "validate_feature_bank",
    "validate_feature_quality",
    "validate_final_closeout",
    "validate_hourly",
    "validate_raw_authority",
    "validate_targets",
]
