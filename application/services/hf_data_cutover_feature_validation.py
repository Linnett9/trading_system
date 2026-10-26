"""Pre-target determinism and quality gate for IEX Feature Bank V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from application.services.hf_data_cutover_service import (
    CutoverAuditError,
    CutoverPaths,
    atomic_json,
    utc_now,
)
from core.research.iex_feature_builders import build_five_minute_features
from infrastructure.data.iex_market_data_store import sha256_file


def run_iex_feature_validation(paths: CutoverPaths) -> dict[str, Any]:
    """Validate the published feature authority without reading any target data."""
    run_manifest = _read_json(paths.output_root / "RUN_MANIFEST.json")
    if run_manifest.get("state") != "IEX_FEATURE_V1_COMPLETE_TARGET_BUILD_PENDING":
        raise CutoverAuditError(
            "Pre-target feature validation requires the completed feature authority, "
            f"found {run_manifest.get('state')}"
        )

    feature_manifest_path = paths.output_root / "IEX_FEATURE_V1_MANIFEST.json"
    feature_manifest = _read_json(feature_manifest_path)
    feature_root = paths.repository_root / "data/market/iex/features/v1"
    partition_paths = sorted(feature_root.glob("**/manifest.json"))
    partitions = [_read_json(path) for path in partition_paths]
    observed_tables: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    manifest_hashes: list[dict[str, str]] = []
    for path, partition in zip(partition_paths, partitions, strict=True):
        identity = partition.get("identity") or {}
        quality = partition.get("quality") or {}
        lineage = partition.get("lineage") or {}
        table = str(identity.get("table"))
        counts = observed_tables.setdefault(table, {"partitions": 0, "rows": 0, "bytes": 0})
        counts["partitions"] += 1
        counts["rows"] += int(partition.get("row_count", 0))
        counts["bytes"] += int(partition.get("parquet_bytes", 0))
        relative = path.relative_to(feature_root).as_posix()
        manifest_hashes.append({"path": relative, "sha256": sha256_file(path)})
        parquet_path_value = partition.get("parquet_path")
        parquet_path = Path(str(parquet_path_value)) if parquet_path_value else None
        if (
            parquet_path is None
            or not parquet_path.is_file()
            or partition.get("parquet_sha256") != sha256_file(parquet_path)
        ):
            failures.append(f"{relative}: parquet hash mismatch or missing artifact")
        if partition.get("status") != "COMPLETE":
            failures.append(f"{relative}: status is not COMPLETE")
        if identity.get("feature_spec_sha256") != feature_manifest.get("feature_spec_sha256"):
            failures.append(f"{relative}: feature specification hash drift")
        if lineage.get("targets_read") is not False:
            failures.append(f"{relative}: target-independence lineage is absent")
        if int(quality.get("duplicate_count", -1)) != 0:
            failures.append(f"{relative}: duplicate row identities")
        if int(quality.get("availability_violation_count", -1)) != 0:
            failures.append(f"{relative}: availability violation")
        if any(float(value or 0.0) != 0.0 for value in quality.get("nonfinite_rate_by_feature", {}).values()):
            failures.append(f"{relative}: non-finite feature values")

    expected_tables = feature_manifest.get("tables") or {}
    if observed_tables != expected_tables:
        failures.append("published partition inventory does not match IEX_FEATURE_V1_MANIFEST.json")
    if feature_manifest.get("status") != "COMPLETE" or feature_manifest.get("missing_tables"):
        failures.append("feature dataset manifest is incomplete")
    if feature_manifest.get("predictor_count") != 80:
        failures.append("frozen predictor count is not 80")
    if feature_manifest.get("targets_read_by_feature_pipeline") is not False:
        failures.append("feature dataset manifest does not prove target independence")
    if feature_manifest.get("raw_ohlcv_duplicated_in_feature_tables") is not False:
        failures.append("raw OHLCV duplication contract changed")

    determinism = _feature_determinism_sentinel(paths)
    if not determinism["feature_exact"]:
        failures.append("AAPL 2024 exact deterministic rebuild failed")

    inventory_payload = json.dumps(manifest_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report = {
        "schema_version": "hf_data_cutover_r1.iex_feature_pretarget_validation.v1",
        "created_at_utc": utc_now(),
        "status": "PASSED" if not failures else "FAILED",
        "target_data_read": False,
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_spec_sha256": feature_manifest.get("feature_spec_sha256"),
        "predictor_count": feature_manifest.get("predictor_count"),
        "partition_count": len(partition_paths),
        "tables": observed_tables,
        "partition_manifest_set_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "quality_contract": {
            "duplicate_count": 0,
            "availability_violation_count": 0,
            "nonfinite_rate": 0.0,
            "missing_values": "permitted where required lookback is unavailable",
        },
        "determinism_sentinel": determinism,
        "failures": failures,
    }
    output_path = paths.output_root / "IEX_FEATURE_V1_PRETARGET_VALIDATION.json"
    atomic_json(output_path, report)
    if failures:
        raise CutoverAuditError(f"Pre-target feature validation failed: {failures[:5]}")
    return report


def _feature_determinism_sentinel(paths: CutoverPaths) -> dict[str, Any]:
    data_root = paths.repository_root / "data/market/iex"
    aapl = _read_symbol_years(data_root / "raw/5m/symbol=AAPL")
    spy = _read_symbol_years(data_root / "raw/5m/symbol=SPY")
    qqq = _read_symbol_years(data_root / "raw/5m/symbol=QQQ")
    published = _read_partition(data_root / "features/v1/5m/symbol=AAPL/year=2024")
    first = build_five_minute_features(aapl, raw_spy=spy, raw_qqq=qqq)
    first = first.loc[first["timestamp"].dt.year.eq(2024)].reset_index(drop=True)
    second = build_five_minute_features(
        aapl.copy(deep=True), raw_spy=spy.copy(deep=True), raw_qqq=qqq.copy(deep=True)
    )
    second = second.loc[second["timestamp"].dt.year.eq(2024)].reset_index(drop=True)
    rebuilds_equal = _frames_equal(first, second)
    matches_published = _frames_equal(first, published)
    return {
        "sample": {"symbol": "AAPL", "year": 2024},
        "two_rebuilds_equal": rebuilds_equal,
        "matches_published": matches_published,
        "feature_exact": rebuilds_equal and matches_published,
        "comparison": "exact values, columns, dtypes, order, and null masks",
    }


def _read_partition(directory: Path) -> pd.DataFrame:
    manifest = _read_json(directory / "manifest.json")
    parquet_path = Path(manifest["parquet_path"])
    if sha256_file(parquet_path) != manifest["parquet_sha256"]:
        raise CutoverAuditError(f"Feature validation source hash mismatch: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _read_symbol_years(directory: Path) -> pd.DataFrame:
    frames = [_read_partition(path.parent) for path in sorted(directory.glob("year=*/manifest.json"))]
    if not frames:
        raise CutoverAuditError(f"No raw feature-validation source partitions found: {directory}")
    return pd.concat(frames, ignore_index=True)


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True)
    except AssertionError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CutoverAuditError(f"Expected JSON object: {path}")
    return payload


__all__ = ["run_iex_feature_validation"]
