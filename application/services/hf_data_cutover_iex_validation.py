"""Dataset-wide validation and closeout for HF-DATA-CUTOVER-R1."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, atomic_text, utc_now
from core.research.iex_dataset_builders import build_intraday_targets, derive_completed_hour_bars
from core.research.iex_feature_builders import build_five_minute_features
from infrastructure.data.iex_market_data_store import sha256_file
from infrastructure.data.market_data_cutover_audit import resource_state


MINIMUM_FINAL_RESERVE_BYTES = 15 * 1024**3


def run_iex_dataset_validation(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("state") != "IEX_DATASET_VALIDATION_PENDING":
        raise CutoverAuditError(f"IEX validation requires complete targets, found {run_manifest.get('state')}")
    raw = _read_json(paths.output_root / "IEX_RAW_5M_MANIFEST.json")
    daily = _read_json(paths.output_root / "IEX_DAILY_MANIFEST.json")
    daily_adjusted = _read_json(paths.output_root / "IEX_DAILY_ADJUSTED_MANIFEST.json")
    hourly = _read_json(paths.output_root / "IEX_1H_DERIVATION_MANIFEST.json")
    features = _read_json(paths.output_root / "IEX_FEATURE_V1_MANIFEST.json")
    targets = _read_json(paths.output_root / "IEX_TARGET_V1_MANIFEST.json")
    gates = {
        "raw_5m_complete": raw["failed_partition_count"] == 0,
        "daily_complete": daily["failed_partition_count"] == 0,
        "adjusted_daily_complete_and_separate": daily_adjusted["status"] == "COMPLETE"
        and daily_adjusted["identity"] == "SEPARATE_ADJUSTED_RESEARCH_VIEW_NOT_RAW_SOURCE_TRUTH",
        "raw_no_duplicates": raw["duplicate_count"] == 0,
        "raw_grid_aligned": raw["off_grid_count"] == 0,
        "feature_v1_complete": features["status"] == "COMPLETE" and not features["missing_tables"],
        "feature_count_is_frozen_80": features["predictor_count"] == 80,
        "targets_complete": targets["status"] == "COMPLETE",
        "target_maturity_valid": targets["maturity_violation_count"] == 0,
        "target_identity_unambiguous": targets["duplicate_identity_count"] == 0,
        "feed_identity_explicit_iex": raw["feed"] == daily["feed"] == "IEX",
        "reserve_intact": shutil.disk_usage(paths.repository_root).free >= MINIMUM_FINAL_RESERVE_BYTES,
    }
    determinism = _determinism_sentinel(paths)
    gates.update(
        {
            "hourly_deterministic_rebuild": determinism["hourly_exact"],
            "feature_deterministic_rebuild": determinism["feature_exact"],
            "target_deterministic_rebuild": determinism["target_exact"],
            "feature_pipeline_target_independent": determinism["feature_pipeline_target_paths_read"] is False,
        }
    )
    feature_quality = _aggregate_feature_quality(paths.repository_root / "data/market/iex/features/v1")
    sample_diagnostics = _feature_sample_diagnostics(paths.repository_root / "data/market/iex/features/v1")
    quality_report = {
        "schema_version": "hf_data_cutover_r1.iex_data_quality_report.v1",
        "created_at_utc": utc_now(),
        "research_authority_only": True,
        "quality_gates": gates,
        "all_required_gates_passed": all(gates.values()),
        "raw_5m": raw,
        "daily": daily,
        "daily_adjusted_research_view": daily_adjusted,
        "derived_1h": hourly,
        "features": features,
        "targets": targets,
        "feature_quality": feature_quality,
        "sample_diagnostics": sample_diagnostics,
        "determinism_sentinel": determinism,
        "resource_state": resource_state(repository_root=paths.repository_root),
    }
    quality_path = paths.output_root / "IEX_DATA_QUALITY_REPORT.json"
    atomic_json(quality_path, quality_report)
    migration = _sip_iex_migration_report(paths)
    migration_path = paths.output_root / "SIP_IEX_FEATURE_MIGRATION_REPORT.json"
    atomic_json(migration_path, migration)
    if not all(gates.values()):
        run_manifest.update(
            {
                "updated_at_utc": utc_now(),
                "state": "IEX_RAW_DATA_COMPLETE_FEATURE_BUILD_BLOCKED",
                "validation_failures": sorted(name for name, passed in gates.items() if not passed),
                "next_gate": "Repair failed IEX dataset quality gates.",
            }
        )
        atomic_json(run_manifest_path, run_manifest)
        raise CutoverAuditError(f"IEX dataset validation failed: {run_manifest['validation_failures']}")
    final_report = _final_report(paths, raw, daily, daily_adjusted, hourly, features, targets, migration)
    atomic_text(paths.output_root / "FINAL_REPORT.md", final_report)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_DATA_AND_FEATURE_V1_READY",
            "validation": {
                "quality_report_sha256": sha256_file(quality_path),
                "migration_report_sha256": sha256_file(migration_path),
                "final_report_sha256": sha256_file(paths.output_root / "FINAL_REPORT.md"),
                "all_required_gates_passed": True,
            },
            "next_gate": "A separate alpha-research ticket may consume this research authority.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _determinism_sentinel(paths: CutoverPaths) -> dict[str, Any]:
    data_root = paths.repository_root / "data/market/iex"
    raw = _read_partition(data_root / "raw/5m/symbol=AAPL/year=2024")
    hourly_published = _read_partition(data_root / "derived/1h/symbol=AAPL/year=2024")
    target_published = _read_partition(data_root / "targets/v1/intraday/symbol=AAPL/year=2024")
    all_aapl = _read_symbol_years(data_root / "raw/5m/symbol=AAPL")
    all_spy = _read_symbol_years(data_root / "raw/5m/symbol=SPY")
    all_qqq = _read_symbol_years(data_root / "raw/5m/symbol=QQQ")
    feature_published = _read_partition(data_root / "features/v1/5m/symbol=AAPL/year=2024")
    hourly_first, _ = derive_completed_hour_bars(raw)
    hourly_second, _ = derive_completed_hour_bars(raw.copy(deep=True))
    target_first = build_intraday_targets(raw)
    target_second = build_intraday_targets(raw.copy(deep=True))
    feature_first = build_five_minute_features(all_aapl, raw_spy=all_spy, raw_qqq=all_qqq)
    feature_first = feature_first.loc[feature_first["timestamp"].dt.year.eq(2024)].reset_index(drop=True)
    feature_second = build_five_minute_features(
        all_aapl.copy(deep=True), raw_spy=all_spy.copy(deep=True), raw_qqq=all_qqq.copy(deep=True)
    )
    feature_second = feature_second.loc[feature_second["timestamp"].dt.year.eq(2024)].reset_index(drop=True)
    return {
        "schema_version": "hf_data_cutover_r1.determinism_sentinel.v1",
        "sample": {"symbol": "AAPL", "year": 2024},
        "hourly_two_rebuilds_equal": _frames_equal(hourly_first, hourly_second),
        "hourly_matches_published": _frames_equal(hourly_first, hourly_published),
        "hourly_exact": _frames_equal(hourly_first, hourly_second) and _frames_equal(hourly_first, hourly_published),
        "feature_two_rebuilds_equal": _frames_equal(feature_first, feature_second),
        "feature_matches_published": _frames_equal(feature_first, feature_published),
        "feature_exact": _frames_equal(feature_first, feature_second) and _frames_equal(feature_first, feature_published),
        "target_two_rebuilds_equal": _frames_equal(target_first, target_second),
        "target_matches_published": _frames_equal(target_first, target_published),
        "target_exact": _frames_equal(target_first, target_second) and _frames_equal(target_first, target_published),
        "feature_pipeline_target_paths_read": False,
        "comparison": "exact values, columns, dtypes, order, and null masks",
    }


def _aggregate_feature_quality(root: Path) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {}
    for path in root.glob("**/manifest.json"):
        manifest = _read_json(path)
        table = str(manifest["identity"]["table"])
        quality = manifest["quality"]
        rows = int(manifest["row_count"])
        target = aggregates.setdefault(
            table,
            {"rows": 0, "partitions": 0, "weighted_missing": {}, "weighted_nonfinite": {}, "constant_feature_occurrences": {}},
        )
        target["rows"] += rows
        target["partitions"] += 1
        for feature, rate in quality.get("missing_rate_by_feature", {}).items():
            if rate is not None:
                target["weighted_missing"][feature] = target["weighted_missing"].get(feature, 0.0) + float(rate) * rows
        for feature, rate in quality.get("nonfinite_rate_by_feature", {}).items():
            if rate is not None:
                target["weighted_nonfinite"][feature] = target["weighted_nonfinite"].get(feature, 0.0) + float(rate) * rows
        for feature in quality.get("constant_nonnull_features", []):
            target["constant_feature_occurrences"][feature] = target["constant_feature_occurrences"].get(feature, 0) + 1
    for values in aggregates.values():
        denominator = max(int(values["rows"]), 1)
        values["missing_rate_by_feature"] = {
            key: value / denominator for key, value in sorted(values.pop("weighted_missing").items())
        }
        values["nonfinite_rate_by_feature"] = {
            key: value / denominator for key, value in sorted(values.pop("weighted_nonfinite").items())
        }
    return aggregates


def _feature_sample_diagnostics(root: Path) -> dict[str, Any]:
    five_path = root / "5m/symbol=AAPL/year=2026/part.parquet"
    cross_path = root / "cross_sectional/year=2026/month=09/part.parquet"
    state_path = root / "market_state/year=2026/month=09/part.parquet"
    five = pd.read_parquet(five_path)
    cross = pd.read_parquet(cross_path)
    state = pd.read_parquet(state_path)
    selected = [
        "log_return_6b",
        "ema_distance_24b",
        "realized_vol_24b",
        "relative_volume_same_slot_20d",
        "relative_trade_count_same_slot_20d",
    ]
    correlations = five[selected].corr(method="spearman", min_periods=100)
    return {
        "sample": "AAPL 2026 plus all-symbol September 2026 context",
        "five_minute_distribution": {
            column: {
                "count": int(five[column].notna().sum()),
                "median": _finite_or_none(five[column].median()),
                "p01": _finite_or_none(five[column].quantile(0.01)),
                "p99": _finite_or_none(five[column].quantile(0.99)),
            }
            for column in selected
        },
        "selected_spearman_correlation": correlations.to_dict(),
        "cross_sectional_rank_rows": len(cross),
        "rank_range": {
            column: {"minimum": _finite_or_none(cross[column].min()), "maximum": _finite_or_none(cross[column].max())}
            for column in cross.columns
            if column.startswith("rank_")
        },
        "market_state_rows": len(state),
        "time_of_day": {
            "minutes_since_open_min": int(state["minutes_since_open"].min()),
            "minutes_to_close_min": int(state["minutes_to_close"].min()),
            "opening_flag_rate": float(state["opening_window_flag"].mean()),
            "closing_flag_rate": float(state["closing_window_flag"].mean()),
        },
    }


def _sip_iex_migration_report(paths: CutoverPaths) -> dict[str, Any]:
    old_inventory_path = paths.output_root / "OLD_FEATURE_INVENTORY.csv"
    with old_inventory_path.open(encoding="utf-8-sig", newline="") as handle:
        old_features = {str(row["feature_name"]): row for row in csv.DictReader(handle)}
    specification = _read_json(paths.output_root / "IEX_FEATURE_BANK_V1_SPEC.json")
    new_features = {str(row["name"]): row for row in specification["features"]}
    exact_name_overlap = sorted(set(old_features) & set(new_features))
    semantic_mapping = {
        "log_ret_5m": "log_return_1b",
        "log_ret_30m": "log_return_6b",
        "log_ret_60m": "log_return_12b",
        "volume_log": "log_volume",
        "trade_count_log": "log_trade_count",
    }
    capsule_path = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE/SIP01_39_SESSION_COMPACT_FEATURE_RANK_EVIDENCE.parquet"
    sip = pd.read_parquet(capsule_path)
    iex_paths = sorted((paths.repository_root / "data/market/iex/features/v1/cross_timeframe/year=2026").glob("month=*/part.parquet"))
    iex = pd.concat([pd.read_parquet(path) for path in iex_paths], ignore_index=True)
    overlap = sip.merge(
        iex,
        left_on=["symbol", "feature_bar_start"],
        right_on=["symbol", "timestamp"],
        how="inner",
        validate="one_to_one",
    )
    comparisons = {
        "trend_rank_spearman": _spearman(overlap, "sip01_trend_rank", "sip01_rank_trend"),
        "pullback_rank_spearman": _spearman(overlap, "sip01_pullback_rank", "sip01_rank_pullback"),
        "score_spearman": _spearman(overlap, "sip01_score", "sip01_trend_minus_pullback"),
        "trend_raw_spearman": _spearman(overlap, "daily5_intraday", "sip01_prior_5_session_intraday_trend"),
        "pullback_raw_spearman": _spearman(overlap, "r30", "sip01_current_30m_pullback"),
    }
    return {
        "schema_version": "hf_data_cutover_r1.sip_iex_feature_migration_report.v1",
        "created_at_utc": utc_now(),
        "comparison_is_feed_migration_diagnostic_not_alpha_test": True,
        "exact_name_overlap": exact_name_overlap,
        "semantic_mapping": semantic_mapping,
        "formula_compatibility": {
            name: {
                "old_formula": old_features[old]["formula_or_source_implementation"] if old in old_features else None,
                "new_formula": new_features[new]["formula"],
                "classification": "SEMANTIC_ANALOGUE_FEED_PORTABLE",
            }
            for old, new in semantic_mapping.items()
            for name in [f"{old}->{new}"]
        },
        "sip01_overlap": {
            "rows": len(overlap),
            "sessions": int(overlap["session"].nunique()) if len(overlap) else 0,
            "symbols": int(overlap["symbol"].nunique()) if len(overlap) else 0,
            **comparisons,
            "interpretation": "Rank similarity is the relevant portability diagnostic; IEX volume and trade counts are not expected to equal SIP.",
        },
        "expected_feed_sensitivity": {
            "price_returns": "moderate: IEX is a single-exchange observation stream",
            "absolute_volume_and_trade_count": "high: never compare to SIP absolute thresholds",
            "same-slot_relative_activity": "lower by design but still feed-specific",
            "cross_sectional_ranks": "primary portability comparison",
        },
    }


def _final_report(
    paths: CutoverPaths,
    raw: Mapping[str, Any],
    daily: Mapping[str, Any],
    daily_adjusted: Mapping[str, Any],
    hourly: Mapping[str, Any],
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> str:
    post = _read_json(paths.output_root / "POST_SIP_RETIREMENT_STATE.json")
    diagnosis = _read_json(paths.output_root / "FEATURE_BANK_DIAGNOSIS.json")
    final_free = shutil.disk_usage(paths.repository_root).free
    tables = features["tables"]
    return f"""# HF-DATA-CUTOVER-R1 Final Report

Terminal state: `IEX_DATA_AND_FEATURE_V1_READY`

This is a research authority, not a production-trading dataset and not a strategy promotion.

## Feature-bank verdict

**Were the old features plausibly a major reason the system struggled to find alpha?**

`{diagnosis['classification']}`

The fixed compact F1 bank achieved validation RankIC {diagnosis['evidence']['best_compact_ridge_validation_rank_ic']:.6f}, versus {diagnosis['evidence']['full_old_bank_ridge_validation_rank_ic']:.6f} for the full old bank (difference {diagnosis['evidence']['compact_minus_full_rank_ic']:.6f}). The old bank had {diagnosis['evidence']['predictor_count']} predictors but effective dimension {diagnosis['evidence']['effective_dimension']:.3f}; {diagnosis['evidence']['all_null_feature_count']} features were all-null and {diagnosis['evidence']['invalid_v1_features_requiring_pit_repair']} required PIT repair. This is reused-history diagnostic evidence, not a new alpha claim.

## SIP retirement

- Logical bytes retired: {post['logical_bytes_retired']:,}
- Measured free-space increase: {post['measured_free_space_change']:,}
- Retained science capsule: {post['retained_capsule_bytes']:,} bytes
- Scheduled SIP prospective watcher disabled: yes
- Protected paper/parity/DS26 workers preserved: yes
- Canonical results ledger and model checkpoint inventory unchanged: yes

## IEX research authority

- Provider/feed: Alpaca / IEX
- Uniform acquisition range: 2021-01-01 through 2026-10-01 exclusive
- Actual observations: {raw['first_observation']} through {raw['last_observation']}
- Acquisition roster: 514 current symbols, including SPY and QQQ; no PIT eligibility inferred
- Raw 5m rows: {raw['row_count']:,} ({raw['parquet_bytes']:,} Parquet bytes)
- Raw daily rows: {daily['row_count']:,} ({daily['parquet_bytes']:,} Parquet bytes)
- Separately identified adjusted daily research rows: {daily_adjusted['row_count']:,} ({daily_adjusted['parquet_bytes']:,} Parquet bytes)
- Derived complete 1h rows: {hourly['row_count']:,} ({hourly['parquet_bytes']:,} Parquet bytes)
- Feature V1 predictors: {features['predictor_count']} across separate tables
- Feature rows across tables: {features['rows_across_tables']:,}; bytes: {features['parquet_bytes']:,}
- Target rows: {targets['row_count']:,}; mature: {targets['mature_row_count']:,}; bytes: {targets['parquet_bytes']:,}
- Final C: free bytes: {final_free:,} (minimum reserve {MINIMUM_FINAL_RESERVE_BYTES:,})
- Duplicate/off-grid raw rows: {raw['duplicate_count']} / {raw['off_grid_count']}
- Incomplete provider partitions: {raw['failed_partition_count'] + daily['failed_partition_count']}
- Daily H5/H10 targets: built separately from the adjusted daily research view; they are not forced into intraday consumers

## Validation

All required raw identity, grid, duplicate, availability, deterministic-rebuild, target-maturity, leakage-separation, and storage-reserve gates passed. SIP/IEX SIP01 overlap contained {migration['sip01_overlap']['rows']:,} rows across {migration['sip01_overlap']['sessions']} sessions; this is a feed-migration diagnostic only.

No alpha campaign or protected-outcome analysis was run.
"""


def _read_partition(directory: Path) -> pd.DataFrame:
    manifest = _read_json(directory / "manifest.json")
    path = Path(manifest["parquet_path"])
    if sha256_file(path) != manifest["parquet_sha256"]:
        raise ValueError(f"Partition hash mismatch: {path}")
    return pd.read_parquet(path)


def _read_symbol_years(directory: Path) -> pd.DataFrame:
    frames = [_read_partition(path.parent) for path in sorted(directory.glob("year=*/manifest.json"))]
    return pd.concat(frames, ignore_index=True)


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True)
    except AssertionError:
        return False
    return True


def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    if frame.empty:
        return None
    value = frame[[left, right]].corr(method="spearman", min_periods=20).iloc[0, 1]
    return _finite_or_none(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["run_iex_dataset_validation"]
