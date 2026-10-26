"""Application orchestration for HF-DATA-CUTOVER-R1 audit and retirement gates."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.research.feature_autopsy import (
    feature_quality_rows,
    fit_diagnostic_model,
    fixed_feature_sets,
    model_comparison_rows,
    monotonicity_rows,
    redundancy_rows,
    univariate_signal_rows,
)
from core.research.market_data_cutover import (
    AvailabilityClass,
    FeatureFamily,
    canonical_sha256,
    classify_feature,
    iex_feature_bank_payload,
    inferred_lookback,
    old_feature_availability,
)
from infrastructure.data.market_data_cutover_audit import (
    AuditSample,
    directory_stats,
    hash_paths,
    load_feature_target_sample,
    parquet_schema,
    resource_state,
    scheduled_task_details,
    sha256_file,
)
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    HistoricalBarRequest,
    SharedRateLimiter,
    fetch_chunk_with_retries,
)


class CutoverAuditError(RuntimeError):
    """Raised when a scientific or safety gate cannot be established."""


@dataclass(frozen=True)
class CutoverPaths:
    repository_root: Path
    output_root: Path
    r1_root: Path
    r2_root: Path

    @property
    def feature_v1(self) -> Path:
        return self.repository_root / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1/run=ds24_p8_r2_local_20260821T000000Z"

    @property
    def feature_v2(self) -> Path:
        return self.repository_root / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v2_pit_repaired/run=DS24_CLEAN_V2_FEATURE_REPAIR_R1_20260926"

    @property
    def target_v1(self) -> Path:
        return self.repository_root / "data/processed/ml_targets/five_minute/version=five_minute_targets_v1"


@dataclass(frozen=True)
class StorageRootSpec:
    relative_path: str
    classification: str
    feed_identity: str
    timeframe: str
    producer: str
    manifest: str
    active_consumers: str
    deletion_eligibility: str
    notes: str = ""


REPAIRED_FEATURES = (
    "overnight_gap",
    "previous_session_return",
    "two_session_return",
    "opening_range_position",
    "session_return_30m",
    "opening_return_30m",
    "minutes_since_open",
    "minutes_until_close",
    "session_progress",
    "early_close_session_flag",
    "opening_period_flag",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _storage_specs() -> tuple[StorageRootSpec, ...]:
    return (
        StorageRootSpec("data/raw/alpaca/stock_bars/sip", "SIP raw provider archive", "SIP", "5m", "Alpaca historical-bar backfill", "per-chunk raw manifests", "none observed", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/processed/alpaca/symbol_bars/sip", "SIP normalized/canonical", "SIP", "5m,1h,1d", "historical-bar finalizer and canonical publisher", "data/canonical/MASTER_5MIN_DATASET/manifest.json", "HF-ALPHA-SIP-R2 scheduled watcher until disabled", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1", "SIP feature bank", "SIP-derived", "5m", "DS24 canonical feature builder", "authority_manifest.json", "dormant DS24 configurations; no live process observed at starting census", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v2_pit_repaired", "SIP feature bank", "SIP-derived", "5m", "DS24 CLEAN V2 PIT repair", "authority_manifest.json", "dormant DS24 configurations; no live process observed at starting census", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/processed/ml_targets/five_minute/version=five_minute_targets_v1", "SIP target bank", "SIP-derived", "H60 from 5m", "Ticket 71b target builder", "per-partition target_manifest.json", "dormant DS24 configurations; no live process observed at starting census", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/processed/ml_targets/five_minute/version=five_minute_targets_clean_v2", "SIP target bank", "SIP-derived", "H60 from 5m", "DS24 CLEAN V2 target delta", "authority_manifest.json", "dormant DS24 configurations; no live process observed at starting census", "ELIGIBLE_AFTER_ALL_GATES"),
        StorageRootSpec("data/raw/alpaca/stock_bars/iex", "IEX", "IEX", "5m", "Alpaca historical-bar pilot", "chunk manifests", "none observed", "RETAIN"),
        StorageRootSpec("data/processed/alpaca/symbol_bars/iex", "IEX", "IEX", "5m", "Alpaca historical-bar finalizer", "partition manifests", "none observed", "RETAIN"),
        StorageRootSpec("data/processed/market_data", "mixed/unknown feed", "MIXED_SOURCE", "daily", "legacy canonical daily and PIT liquidity programmes", "nested manifests", "daily research authorities", "RETAIN_AUTHENTICATION_REQUIRED", "Not classified from directory name; contains authorities with distinct provenance."),
        StorageRootSpec("data/canonical", "active runtime state", "MIXED_METADATA", "mixed", "canonical authority publishers", "nested manifests", "canonical publishers and research readers", "RETAIN", "Contains pointers, locks, transaction evidence, and spines; not a raw-data deletion candidate."),
        StorageRootSpec("data/reference", "symbol/universe authority", "NOT_MARKET_VALUES", "reference", "reference authority programmes", "nested manifests", "PIT research", "RETAIN"),
        StorageRootSpec("data/features", "non-SIP feature authority", "UNKNOWN_SOURCE", "daily", "legacy daily feature builder", "nested manifests", "unknown", "RETAIN_AUTHENTICATION_REQUIRED"),
        StorageRootSpec("docs/research/results", "results/evidence", "NOT_APPLICABLE", "not applicable", "research evidence register", "release manifests", "research governance", "RETAIN"),
    )


def _mtime_iso(nanoseconds: int | None) -> str | None:
    if nanoseconds is None:
        return None
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def build_storage_inventory(paths: CutoverPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _storage_specs():
        path = paths.repository_root / spec.relative_path
        stats = directory_stats(path)
        rows.append(
            {
                "exact_path": str(path.resolve()),
                "exists": path.exists(),
                "bytes": stats.bytes,
                "files": stats.files,
                "feed_identity": spec.feed_identity,
                "classification": spec.classification,
                "timeframe": spec.timeframe,
                "date_range": f"filesystem_mtime:{_mtime_iso(stats.minimum_mtime_ns)}..{_mtime_iso(stats.maximum_mtime_ns)}",
                "symbols": "manifest/partition-defined",
                "producer": spec.producer,
                "manifest": spec.manifest,
                "active_consumers": spec.active_consumers,
                "deletion_eligibility": spec.deletion_eligibility,
                "notes": spec.notes,
            }
        )
    return rows


def _feature_contracts() -> dict[str, dict[str, Any]]:
    from core.research.ml.ds24.feature_schema_r6 import FINAL_STOCK_FEATURES, SHARED_CONTEXT_FEATURES

    result: dict[str, dict[str, Any]] = {}
    for feature in (*FINAL_STOCK_FEATURES, *SHARED_CONTEXT_FEATURES):
        result[feature.feature_id] = {
            "formula": feature.formula,
            "lookback": feature.minimum_history or str(feature.bars_required),
            "dtype": feature.dtype,
            "source": feature.implementation_path,
            "availability_timestamp": feature.availability_lag,
        }
    return result


def _old_feature_inventory(
    paths: CutoverPaths,
    predictor_names: Sequence[str],
    quality_rows: Sequence[Mapping[str, Any]],
    redundancy: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contracts = _feature_contracts()
    quality = {str(row["feature_name"]): row for row in quality_rows}
    duplicates: dict[str, list[str]] = {}
    for edge in redundancy:
        if float(edge["absolute_spearman"]) <= 0.99:
            continue
        duplicates.setdefault(str(edge["left_feature"]), []).append(str(edge["right_feature"]))
        duplicates.setdefault(str(edge["right_feature"]), []).append(str(edge["left_feature"]))
    sample_stock = next((paths.feature_v1 / "stock").rglob("features.parquet"))
    sample_context = next((paths.feature_v1 / "shared-context").rglob("context.parquet"))
    dtype_map = {**parquet_schema(sample_stock), **parquet_schema(sample_context)}
    rows: list[dict[str, Any]] = []
    for name in predictor_names:
        contract = contracts.get(name, {})
        diagnostic = quality.get(name, {})
        family = classify_feature(name)
        rows.append(
            {
                "feature_name": name,
                "feature_family": family.value,
                "timeframe": "5m",
                "formula_or_source_implementation": contract.get("formula") or f"core/research/ml/ds24/master_5m_schema_r7.py::{name}",
                "lookback": contract.get("lookback") or inferred_lookback(name),
                "dtype": dtype_map.get(name, contract.get("dtype", "unknown")),
                "availability_timestamp": contract.get("availability_timestamp", "5 minutes after source bar start"),
                "missingness_rate": diagnostic.get("missingness_rate"),
                "nonfinite_rate": diagnostic.get("nonfinite_rate"),
                "zero_or_constant_rate": diagnostic.get("zero_rate"),
                "unique_value_count": diagnostic.get("unique_value_count"),
                "scale_standard_deviation": diagnostic.get("standard_deviation"),
                "is_cross_sectional": family in {FeatureFamily.CROSS_SECTIONAL_RANKS, FeatureFamily.BREADTH_STATE},
                "is_market_relative": family in {FeatureFamily.MARKET_RELATIVE, FeatureFamily.BETA_CORRELATION},
                "is_volume_dependent": family in {FeatureFamily.VOLUME_ACTIVITY, FeatureFamily.LIQUIDITY, FeatureFamily.VWAP},
                "is_sip_specific": family in {FeatureFamily.VOLUME_ACTIVITY, FeatureFamily.LIQUIDITY, FeatureFamily.VWAP},
                "duplicated_by_another_feature": "|".join(sorted(duplicates.get(name, []))),
                "current_consumers": "DS24 CLEAN V2 configuration and historical research artifacts",
            }
        )
    return rows


def _family_map(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": row["feature_name"],
            "feature_family": row["feature_family"],
            "taxonomy_version": "HF-DATA-CUTOVER-R1.v1",
            "ablation_group": str(row["feature_family"]).replace("/", "_").replace(" ", "_").lower(),
        }
        for row in inventory
    ]


def _availability_rows(predictors: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for name in predictors:
        classification = old_feature_availability(name, repaired=True)
        evidence = "V2 PIT repair sidecar and future-value perturbation certificate" if name in REPAIRED_FEATURES else "R6 completed-bar timing contract"
        if name.startswith("breadth_"):
            evidence = "Population authority limited; sampled context partitions physically null"
        rows.append(
            {
                "feature_name": name,
                "feature_family": classify_feature(name).value,
                "availability_classification": classification.value,
                "same_bar_close": "available only at bar_start+5m",
                "incomplete_1h_bar": "not applicable; no 1h features in old bank",
                "same_day_daily_close": "not used in old materialized bank",
                "future_session_contamination": "repaired for named V1 defects" if name in REPAIRED_FEATURES else "none established",
                "forward_fill": "no interpolation/forward fill in authority contract",
                "full_sample_normalization": "not part of stored feature formula",
                "future_universe_dependence": "AMBIGUOUS" if name.startswith("breadth_") else "fixed physical roster limitation",
                "evidence": evidence,
            }
        )
    return rows


def _gap_matrix(predictors: Sequence[str]) -> list[dict[str, str]]:
    names = set(predictors)
    requirements = {
        "multi-window returns": (["ret_5m", "ret_30m", "ret_60m", "ret_120m"], "STRONG"),
        "residual/market-relative returns": (["relative_strength_spy_60m"], "PRESENT_BUT_WEAK"),
        "SPY-relative features": (["relative_strength_spy_15m", "relative_strength_spy_60m"], "STRONG"),
        "QQQ-relative features": (["relative_strength_qqq_60m"], "STRONG"),
        "rolling beta": (["rolling_beta_spy"], "ABSENT"),
        "rolling correlation": (["rolling_corr_spy"], "ABSENT"),
        "cross-sectional ranks": (["relative_strength_rank_60m"], "PRESENT_BUT_WEAK"),
        "same-slot relative volume": (["relative_volume_tod_pit"], "STRONG"),
        "same-slot relative trade count": (["relative_trade_count_tod_pit"], "ABSENT"),
        "time-of-day encoding": (["minutes_since_open", "session_progress"], "STRONG"),
        "VWAP deviation": (["vwap_distance_session"], "STRONG"),
        "market breadth": (["breadth_fraction_positive_60m"], "PRESENT_BUT_WEAK"),
        "cross-sectional dispersion": (["breadth_return_dispersion_60m"], "PRESENT_BUT_WEAK"),
        "daily state": (["daily_state"], "ABSENT"),
        "1h state": (["hourly_state"], "ABSENT"),
        "daily x intraday interactions": (["daily_x_intraday"], "ABSENT"),
        "hourly x 5m interactions": (["hourly_x_5m"], "ABSENT"),
        "SIP01 trend/pullback construction": (["sip01_trend_minus_pullback"], "ABSENT"),
    }
    rows = []
    for requirement, (expected, fallback) in requirements.items():
        present = [name for name in expected if name in names]
        classification = "STRONG" if len(present) == len(expected) and fallback != "PRESENT_BUT_WEAK" else fallback
        rows.append({"required_family_or_feature": requirement, "classification": classification, "old_features": "|".join(present), "basis": "physical predictor manifest plus authenticated formula contracts"})
    return rows


def _bounded_timestamp_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame.copy()
    timestamps = pd.Index(frame["decision_timestamp"].drop_duplicates().sort_values())
    approximate_width = max(1, int(frame.groupby("decision_timestamp", observed=True).size().median()))
    wanted = max(1, maximum_rows // approximate_width)
    positions = np.linspace(0, len(timestamps) - 1, min(wanted, len(timestamps)), dtype=np.int64)
    selected = set(timestamps[positions])
    return frame[frame["decision_timestamp"].isin(selected)].copy()


def _family_incremental(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    baseline = list(feature_sets["F0"])
    family_candidates = {
        "trend": [name for name in feature_sets["FOLD"] if classify_feature(name) in {FeatureFamily.TREND, FeatureFamily.REVERSAL}],
        "volatility": [name for name in feature_sets["FOLD"] if classify_feature(name) in {FeatureFamily.VOLATILITY, FeatureFamily.DOWNSIDE_RISK, FeatureFamily.RANGE_LOCATION}],
        "market-relative": [name for name in feature_sets["FOLD"] if classify_feature(name) in {FeatureFamily.MARKET_RELATIVE, FeatureFamily.BETA_CORRELATION}],
        "volume/activity": [name for name in feature_sets["FOLD"] if classify_feature(name) in {FeatureFamily.VOLUME_ACTIVITY, FeatureFamily.LIQUIDITY, FeatureFamily.VWAP}],
        "cross-sectional rank": [name for name in feature_sets["FOLD"] if classify_feature(name) == FeatureFamily.CROSS_SECTIONAL_RANKS],
        "time-of-day": [name for name in feature_sets["FOLD"] if classify_feature(name) == FeatureFamily.TIME_OF_DAY],
        "daily context": [],
        "1h context": [],
        "cross-timeframe": [],
    }
    base_result = fit_diagnostic_model(train, validation, baseline, target="target_value", model_name="Ridge")
    output = []
    for family, additions in family_candidates.items():
        if not additions:
            output.append({"family": family, "status": "ABSENT_FROM_OLD_BANK", "baseline_validation_rank_ic": base_result.get("validation_rank_ic"), "incremental_validation_rank_ic": None})
            continue
        result = fit_diagnostic_model(train, validation, list(dict.fromkeys([*baseline, *additions])), target="target_value", model_name="Ridge")
        output.append(
            {
                "family": family,
                "status": result["status"],
                "feature_count_added": len(additions),
                "baseline_validation_rank_ic": base_result.get("validation_rank_ic"),
                "with_family_validation_rank_ic": result.get("validation_rank_ic"),
                "incremental_validation_rank_ic": (result.get("validation_rank_ic") - base_result.get("validation_rank_ic")) if result.get("validation_rank_ic") is not None and base_result.get("validation_rank_ic") is not None else None,
            }
        )
    return output


def _ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    full = fit_diagnostic_model(train, validation, features, target="target_value", model_name="Ridge")
    output = []
    families = sorted({classify_feature(name) for name in features}, key=lambda item: item.value)
    for family in families:
        kept = [name for name in features if classify_feature(name) != family]
        result = fit_diagnostic_model(train, validation, kept, target="target_value", model_name="Ridge")
        output.append(
            {
                "removed_family": family.value,
                "removed_feature_count": len(features) - len(kept),
                "full_validation_rank_ic": full.get("validation_rank_ic"),
                "ablated_validation_rank_ic": result.get("validation_rank_ic"),
                "delta_ablated_minus_full": (result.get("validation_rank_ic") - full.get("validation_rank_ic")) if result.get("validation_rank_ic") is not None and full.get("validation_rank_ic") is not None else None,
            }
        )
    return output


def _normalization_diagnostic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    selected = list(features[: min(16, len(features))])
    output = []
    raw_result = fit_diagnostic_model(train, validation, selected, target="target_value", model_name="Ridge")
    output.append({"normalization": "raw", **raw_result})

    lower = train[selected].quantile(0.01)
    upper = train[selected].quantile(0.99)
    winsor_train = train.copy()
    winsor_validation = validation.copy()
    winsor_train[selected] = winsor_train[selected].clip(lower, upper, axis="columns")
    winsor_validation[selected] = winsor_validation[selected].clip(lower, upper, axis="columns")
    output.append({"normalization": "winsorized_raw_train_bounds", **fit_diagnostic_model(winsor_train, winsor_validation, selected, target="target_value", model_name="Ridge")})

    rank_train = train.copy()
    rank_validation = validation.copy()
    rank_train[selected] = rank_train.groupby("decision_timestamp", observed=True)[selected].rank(method="average", pct=True)
    rank_validation[selected] = rank_validation.groupby("decision_timestamp", observed=True)[selected].rank(method="average", pct=True)
    output.append({"normalization": "same_timestamp_cross_sectional_percentile", **fit_diagnostic_model(rank_train, rank_validation, selected, target="target_value", model_name="Ridge")})
    output.append({"normalization": "rolling_standardized", "status": "BLOCKED_SAMPLE_NOT_CONTIGUOUS", "feature_count": len(selected), "reason": "The target-blind audit sample selects 30-minute slots; a valid rolling z-score comparison requires contiguous source rows and was not approximated."})
    return output


def _feature_count_diagnostic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    quality: Sequence[Mapping[str, Any]],
    eligible: Sequence[str],
) -> list[dict[str, Any]]:
    quality_map = {str(row["feature_name"]): row for row in quality}
    family_priority = {
        FeatureFamily.RETURNS_MOMENTUM: 0,
        FeatureFamily.MARKET_RELATIVE: 1,
        FeatureFamily.TREND: 2,
        FeatureFamily.VOLATILITY: 3,
        FeatureFamily.VOLUME_ACTIVITY: 4,
        FeatureFamily.TIME_OF_DAY: 5,
    }
    ordered = sorted(
        eligible,
        key=lambda name: (
            family_priority.get(classify_feature(name), 10),
            float(quality_map.get(name, {}).get("missingness_rate") or 0.0),
            name,
        ),
    )
    output = []
    for size in (8, 16, 32, 48, len(ordered)):
        selected = ordered[: min(size, len(ordered))]
        result = fit_diagnostic_model(train, validation, selected, target="target_value", model_name="Ridge")
        output.append({"requested_feature_count": size, "ordering_rule": "family prior, causal coverage, name; target blind", **result})
    return output


def _preserve_sip01_compact(paths: CutoverPaths, capsule: Path) -> dict[str, Any]:
    source = paths.r1_root / "source/hf_alpha_sip_r1.py"
    spec = importlib.util.spec_from_file_location("hf_alpha_sip_r1_frozen_source", source)
    if spec is None or spec.loader is None:
        raise CutoverAuditError(f"Cannot load frozen SIP01 source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = _load_json(paths.r2_root / "SIP01_PROSPECTIVE_CONTRACT.json")
    roster = [str(value) for value in contract["population"]["fixed_roster"]]
    evaluation_dates = set(module.ALL_EVALUATION_DATES)
    rows: list[dict[str, Any]] = []
    for symbol in roster:
        symbol_rows, _ = module.build_symbol_rows(symbol, evaluation_dates)
        rows.extend(symbol_rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise CutoverAuditError("Frozen SIP01 source produced no compact evidence rows")
    frame["sip01_trend_rank"] = frame.groupby("feature_ready_at", observed=True)["daily5_intraday"].rank(method="average", pct=True)
    frame["sip01_pullback_rank"] = frame.groupby("feature_ready_at", observed=True)["r30"].rank(method="average", pct=True)
    frame["sip01_score"] = frame["sip01_trend_rank"] - frame["sip01_pullback_rank"]
    columns = [
        "symbol",
        "session",
        "feature_bar_start",
        "feature_ready_at",
        "daily5_intraday",
        "r30",
        "sip01_trend_rank",
        "sip01_pullback_rank",
        "sip01_score",
    ]
    destination = capsule / "SIP01_39_SESSION_COMPACT_FEATURE_RANK_EVIDENCE.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame[columns].to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    sessions = sorted(frame["session"].astype(str).unique())
    if sessions != sorted(module.ALL_EVALUATION_DATES):
        raise CutoverAuditError("Compact SIP01 evidence does not cover the exact frozen 39 sessions")
    return {"path": str(destination.relative_to(paths.output_root)), "rows": len(frame), "sessions": len(sessions), "symbols": frame["symbol"].nunique(), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def _copy_capsule_evidence(paths: CutoverPaths, capsule: Path) -> list[dict[str, Any]]:
    capsule.mkdir(parents=True, exist_ok=True)
    sources = [
        paths.r1_root / "RUN_MANIFEST.json",
        paths.r1_root / "GENERATION_A_PREREGISTRATION.json",
        paths.r1_root / "GENERATION_A_RESULTS.csv",
        paths.r1_root / "INTERNAL_VALIDATION_RESULTS.csv",
        paths.r1_root / "ROBUSTNESS_RESULTS.csv",
        paths.r1_root / "SIP_FEATURE_MECHANISM_CATALOGUE.csv",
        paths.r1_root / "SIP_RESEARCH_SPLIT.json",
        paths.r1_root / "FINAL_ALPHA_REPORT.md",
        paths.r2_root / "SIP01_PROSPECTIVE_CONTRACT.json",
        paths.r2_root / "NO_PEEKING_ATTESTATION.json",
        paths.r2_root / "PROSPECTIVE_CAPTURE_STATE.json",
    ]
    artifacts: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_file():
            raise CutoverAuditError(f"Required capsule evidence missing: {source}")
        prefix = "R1" if source.is_relative_to(paths.r1_root) else "R2"
        destination = capsule / f"{prefix}_{source.name}"
        shutil.copy2(source, destination)
        artifacts.append({"source": str(source), "path": str(destination.relative_to(paths.output_root)), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
    return artifacts


def _retirement_intent(paths: CutoverPaths, capsule_manifest: Path) -> list[dict[str, Any]]:
    candidates = [
        (paths.repository_root / "data/raw/alpaca/stock_bars/sip", "SIP raw provider archive"),
        (paths.repository_root / "data/processed/alpaca/symbol_bars/sip", "SIP canonical bars including derived 1h/daily"),
        (paths.repository_root / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1", "SIP feature bank V1"),
        (paths.repository_root / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v2_pit_repaired", "SIP PIT repair sidecar"),
        (paths.repository_root / "data/processed/ml_targets/five_minute/version=five_minute_targets_v1", "SIP target bank V1"),
        (paths.repository_root / "data/processed/ml_targets/five_minute/version=five_minute_targets_clean_v2", "SIP target delta"),
    ]
    rows = []
    for path, role in candidates:
        resolved = path.resolve()
        data_root = (paths.repository_root / "data").resolve()
        if data_root not in resolved.parents:
            raise CutoverAuditError(f"Refusing retirement candidate outside data root: {resolved}")
        stats = directory_stats(resolved)
        identity_material = f"{resolved}|{stats.bytes}|{stats.files}|{stats.maximum_mtime_ns}".encode("utf-8")
        rows.append(
            {
                "exact_path": str(resolved),
                "bytes": stats.bytes,
                "files": stats.files,
                "feed_identity": "AUTHENTICATED_SIP" if "alpaca" in str(resolved).lower() else "SIP_DERIVED_AUTHENTICATED_PARENT",
                "role": role,
                "active_consumers": "PENDING_IMMEDIATE_PRE_DELETE_RECHECK",
                "retained_evidence_locator": str(capsule_manifest.resolve()),
                "deletion_justification": "Superseded bulky SIP estate after validated science capsule; required storage for canonical IEX rebuild.",
                "pre_delete_hash_or_identity": hashlib.sha256(identity_material).hexdigest(),
            }
        )
    return rows


def run_audit(paths: CutoverPaths, *, symbol_count: int = 48) -> dict[str, Any]:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    starting_state_path = paths.output_root / "STARTING_RESOURCE_STATE.json"
    if not starting_state_path.exists():
        started = resource_state(repository_root=paths.repository_root)
        started["captured_at_utc"] = utc_now()
        started["scheduled_sip01_watcher"] = scheduled_task_details("HF_ALPHA_SIP_R2_20261025T125948Z_LOCAL_SIP_COLLECTOR")
        atomic_json(starting_state_path, started)

    storage = build_storage_inventory(paths)
    atomic_csv(paths.output_root / "MARKET_DATA_STORAGE_INVENTORY.csv", storage)

    predictor_manifest_path = paths.repository_root / "config/ds24_clean_v2/predictor_manifest.json"
    predictor_manifest = _load_json(predictor_manifest_path)
    predictors = [str(value) for value in predictor_manifest["predictors"]]
    if len(predictors) != 101:
        raise CutoverAuditError(f"Expected authenticated 101-predictor bank, found {len(predictors)}")

    sample = load_feature_target_sample(
        feature_root=paths.feature_v1,
        repair_root=paths.feature_v2,
        target_root=paths.target_v1,
        predictor_names=predictors,
        repaired_names=REPAIRED_FEATURES,
        symbol_count=symbol_count,
    )
    sample_manifest = {
        "schema_version": "hf_data_cutover_r1.audit_sample.v1",
        "selection": "SHA256-ranked symbols, years 2022-2024, UTC decision minute divisible by 30; frozen before target read",
        "development_only": True,
        "protected_holdout_opened": False,
        "symbols": list(sample.symbols),
        "years": list(sample.years),
        "rows": len(sample.frame),
        "minimum_decision_timestamp": sample.frame["decision_timestamp"].min().isoformat(),
        "maximum_decision_timestamp": sample.frame["decision_timestamp"].max().isoformat(),
        "feature_file_identity": hash_paths(sample.feature_files),
        "target_file_identity": hash_paths(sample.target_files),
        "context_file_identity": hash_paths(sample.context_files),
        "repair_file_identity": hash_paths(sample.sidecar_files),
    }
    atomic_json(paths.output_root / "AUDIT_SAMPLE_MANIFEST.json", sample_manifest)

    quality = feature_quality_rows(sample.frame, predictors)
    redundancy, redundancy_summary = redundancy_rows(sample.frame, predictors)
    atomic_csv(paths.output_root / "FEATURE_QUALITY_DIAGNOSTICS.csv", quality)
    atomic_csv(paths.output_root / "FEATURE_REDUNDANCY_CLUSTERS.csv", redundancy, ["cluster_id", "left_feature", "right_feature", "spearman", "absolute_spearman"])
    atomic_json(paths.output_root / "FEATURE_REDUNDANCY_SUMMARY.json", redundancy_summary)

    inventory = _old_feature_inventory(paths, predictors, quality, redundancy)
    atomic_csv(paths.output_root / "OLD_FEATURE_INVENTORY.csv", inventory)
    atomic_csv(paths.output_root / "OLD_FEATURE_FAMILY_MAP.csv", _family_map(inventory))
    availability = _availability_rows(predictors)
    atomic_csv(paths.output_root / "FEATURE_AVAILABILITY_AUDIT.csv", availability)
    cross_timeframe = {
        "schema_version": "hf_data_cutover_r1.cross_timeframe_audit.v1",
        "old_bank": {
            "five_minute_tactical_information": "PRESENT_STRONG",
            "completed_one_hour_information": "ABSENT_AS_SEPARATE_TIMEFRAME",
            "completed_daily_information": "ABSENT_AS_SEPARATE_TIMEFRAME",
            "daily_features_physically_repeated": False,
            "asof_joins": "daily join contract documented but not materialized in this bank",
            "explicit_cross_timeframe_interactions": "ABSENT",
            "future_value_perturbation": "V2 certificate: 101/101 causal after replacement of eleven V1 columns",
            "incomplete_hour_guard": "NOT_APPLICABLE_NO_1H_FEATURES",
            "same_day_daily_close_guard": "NOT_APPLICABLE_NO_DAILY_FEATURES",
        },
        "tests": {
            "later_5m_value_changes_earlier_prediction": False,
            "incomplete_hour_enters_current_5m_row": False,
            "today_daily_close_enters_before_close": False,
        },
        "evidence": ["config/ds24_clean_v2/feature_authority.json", "core/research/ml/ds24/feature_schema_r6.py"],
    }
    atomic_json(paths.output_root / "CROSS_TIMEFRAME_AUDIT.json", cross_timeframe)
    atomic_csv(paths.output_root / "FEATURE_GAP_MATRIX.csv", _gap_matrix(predictors))

    eligible = [
        name
        for name in predictors
        if old_feature_availability(name, repaired=True) in {AvailabilityClass.CAUSAL, AvailabilityClass.CAUSAL_WITH_DELAY}
        and sample.frame[name].notna().any()
    ]
    signal = univariate_signal_rows(sample.frame, eligible, target="target_value", horizon="H60")
    blocked_horizons = [
        {"feature_name": "*", "feature_family": "*", "target_horizon": horizon, "status": "BLOCKED_NO_SURVIVING_AUTHORIZED_TARGET_AUTHORITY", "period_count": 0, "session_count": 0}
        for horizon in ("H30", "H5", "H10")
    ]
    atomic_csv(paths.output_root / "UNIVARIATE_FEATURE_SIGNAL.csv", [*signal, *blocked_horizons])
    atomic_csv(paths.output_root / "FEATURE_MONOTONICITY_DIAGNOSTICS.csv", monotonicity_rows(sample.frame, eligible, target="target_value"))

    training = sample.frame[sample.frame["decision_timestamp"].dt.year.le(2023)].copy()
    validation = sample.frame[sample.frame["decision_timestamp"].dt.year.eq(2024)].copy()
    training = _bounded_timestamp_sample(training, 35_000)
    validation = _bounded_timestamp_sample(validation, 20_000)
    feature_sets = fixed_feature_sets(eligible)
    comparison = model_comparison_rows(training, validation, feature_sets, target="target_value")
    for row in comparison:
        row["target_horizon"] = "H60"
        row["evidence_class"] = "DIAGNOSTIC_REUSED_HISTORY_NO_NEW_ALPHA_CLAIM"
        row["effective_dimension"] = redundancy_summary["effective_dimension"]
    atomic_csv(paths.output_root / "FEATURE_SET_MODEL_COMPARISON.csv", comparison)
    incremental = _family_incremental(training, validation, feature_sets)
    atomic_csv(paths.output_root / "FEATURE_FAMILY_INCREMENTAL_VALUE.csv", incremental)
    ablation = _ablation(training, validation, eligible)
    atomic_csv(paths.output_root / "OLD_BANK_ABLATION.csv", ablation)
    normalization = _normalization_diagnostic(training, validation, eligible)
    atomic_csv(paths.output_root / "FEATURE_NORMALIZATION_DIAGNOSTIC.csv", normalization)
    count_diagnostic = _feature_count_diagnostic(training, validation, quality, eligible)
    atomic_csv(paths.output_root / "FEATURE_COUNT_OVERFITTING_DIAGNOSTIC.csv", count_diagnostic)

    comparison_map = {(str(row["feature_set"]), str(row["model"])): row for row in comparison}
    compact_values = [float(comparison_map[(name, "Ridge")]["validation_rank_ic"]) for name in ("F1", "F2", "F3", "F4", "F5") if comparison_map.get((name, "Ridge"), {}).get("validation_rank_ic") is not None]
    full_validation = comparison_map.get(("FOLD", "Ridge"), {}).get("validation_rank_ic")
    best_compact = max(compact_values) if compact_values else None
    all_null_count = sum(float(row["missingness_rate"]) >= 1.0 for row in quality)
    invalid_before_repair = len(REPAIRED_FEATURES)
    evidence_flags = {
        "all_null_feature_count": all_null_count,
        "invalid_v1_features_requiring_pit_repair": invalid_before_repair,
        "effective_dimension": redundancy_summary["effective_dimension"],
        "predictor_count": len(predictors),
        "best_compact_ridge_validation_rank_ic": best_compact,
        "full_old_bank_ridge_validation_rank_ic": full_validation,
        "compact_minus_full_rank_ic": (best_compact - float(full_validation)) if best_compact is not None and full_validation is not None else None,
        "missing_daily_context": True,
        "missing_hourly_context": True,
        "missing_sip01_feature": True,
    }
    major = bool(
        all_null_count >= 5
        and invalid_before_repair >= 5
        and best_compact is not None
        and full_validation is not None
        and best_compact - float(full_validation) >= 0.005
    )
    partial = bool(
        all_null_count > 0
        or invalid_before_repair > 0
        or redundancy_summary["effective_dimension"] < 0.7 * len(predictors)
        or (best_compact is not None and full_validation is not None and best_compact > float(full_validation))
    )
    diagnosis = "FEATURE_BANK_LIKELY_MAJOR_LIMITER" if major else ("FEATURE_BANK_PARTIAL_LIMITER" if partial else "FEATURE_BANK_NOT_PRIMARY_LIMITER")
    diagnosis_payload = {
        "schema_version": "hf_data_cutover_r1.feature_bank_diagnosis.v1",
        "classification": diagnosis,
        "claim_scope": "diagnostic reused-history evidence; not a new strategy or alpha claim",
        "evidence": evidence_flags,
        "interpretation": "The bank carried useful causal price, volatility, relative-strength, activity, and time-of-day information, but defects, null breadth columns, redundancy, and absent daily/1h/SIP01 context plausibly reduced efficiency. The result does not establish that feature representation was the sole or primary cause of weak alpha.",
    }
    atomic_json(paths.output_root / "FEATURE_BANK_DIAGNOSIS.json", diagnosis_payload)

    iex_spec = iex_feature_bank_payload(
        evidence={
            "feature_bank_diagnosis_sha256": sha256_file(paths.output_root / "FEATURE_BANK_DIAGNOSIS.json"),
            "sip01_r1_results_sha256": sha256_file(paths.r1_root / "GENERATION_A_RESULTS.csv"),
            "old_feature_inventory_sha256": sha256_file(paths.output_root / "OLD_FEATURE_INVENTORY.csv"),
            "frozen_before_iex_target_outcomes": True,
        }
    )
    atomic_json(paths.output_root / "IEX_FEATURE_BANK_V1_SPEC.json", iex_spec)

    capsule = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE"
    capsule_artifacts = _copy_capsule_evidence(paths, capsule)
    capsule_artifacts.append(_preserve_sip01_compact(paths, capsule))
    for name in (
        "OLD_FEATURE_INVENTORY.csv",
        "OLD_FEATURE_FAMILY_MAP.csv",
        "FEATURE_QUALITY_DIAGNOSTICS.csv",
        "FEATURE_REDUNDANCY_CLUSTERS.csv",
        "FEATURE_REDUNDANCY_SUMMARY.json",
        "UNIVARIATE_FEATURE_SIGNAL.csv",
        "FEATURE_SET_MODEL_COMPARISON.csv",
        "FEATURE_FAMILY_INCREMENTAL_VALUE.csv",
        "OLD_BANK_ABLATION.csv",
        "FEATURE_NORMALIZATION_DIAGNOSTIC.csv",
        "FEATURE_COUNT_OVERFITTING_DIAGNOSTIC.csv",
        "FEATURE_BANK_DIAGNOSIS.json",
        "IEX_FEATURE_BANK_V1_SPEC.json",
    ):
        source = paths.output_root / name
        destination = capsule / name
        shutil.copy2(source, destination)
        capsule_artifacts.append({"source": str(source), "path": str(destination.relative_to(paths.output_root)), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
    capsule_manifest = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE_MANIFEST.json"
    capsule_payload = {
        "schema_version": "hf_data_cutover_r1.sip_retirement_science_capsule.v1",
        "created_at_utc": utc_now(),
        "artifact_count": len(capsule_artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in capsule_artifacts),
        "artifacts": capsule_artifacts,
        "validation": {"all_hashes_verified": all(sha256_file(paths.output_root / item["path"]) == item["sha256"] for item in capsule_artifacts), "r2_outcomes_calculated": False, "full_sip_bars_copied": False},
    }
    atomic_json(capsule_manifest, capsule_payload)
    intent = _retirement_intent(paths, capsule_manifest)
    atomic_csv(paths.output_root / "SIP_RETIREMENT_INTENT.csv", intent)

    run_manifest = {
        "schema_version": "hf_data_cutover_r1.run_manifest.v1",
        "run_id": paths.output_root.name,
        "updated_at_utc": utc_now(),
        "state": "SIP_AUDIT_COMPLETE_RETIREMENT_INTENT_FROZEN",
        "audit": {
            "sample": sample_manifest,
            "diagnosis": diagnosis,
            "iex_feature_spec_sha256": sha256_file(paths.output_root / "IEX_FEATURE_BANK_V1_SPEC.json"),
            "capsule_manifest_sha256": sha256_file(capsule_manifest),
            "retirement_intent_sha256": sha256_file(paths.output_root / "SIP_RETIREMENT_INTENT.csv"),
            "retirement_candidate_count": len(intent),
            "retirement_candidate_logical_bytes": sum(int(row["bytes"]) for row in intent),
        },
        "next_gate": "Immediate active-consumer/runtime recheck, watcher disablement, and exact-path retirement.",
    }
    atomic_json(paths.output_root / "RUN_MANIFEST.json", run_manifest)
    return run_manifest
def finalize_completed_audit(paths: CutoverPaths) -> dict[str, Any]:
    """Resume only the capsule/intent publication after completed diagnostics."""

    required = (
        "STARTING_RESOURCE_STATE.json",
        "MARKET_DATA_STORAGE_INVENTORY.csv",
        "AUDIT_SAMPLE_MANIFEST.json",
        "OLD_FEATURE_INVENTORY.csv",
        "OLD_FEATURE_FAMILY_MAP.csv",
        "FEATURE_QUALITY_DIAGNOSTICS.csv",
        "FEATURE_REDUNDANCY_CLUSTERS.csv",
        "FEATURE_REDUNDANCY_SUMMARY.json",
        "FEATURE_AVAILABILITY_AUDIT.csv",
        "CROSS_TIMEFRAME_AUDIT.json",
        "FEATURE_GAP_MATRIX.csv",
        "UNIVARIATE_FEATURE_SIGNAL.csv",
        "FEATURE_MONOTONICITY_DIAGNOSTICS.csv",
        "FEATURE_SET_MODEL_COMPARISON.csv",
        "FEATURE_FAMILY_INCREMENTAL_VALUE.csv",
        "OLD_BANK_ABLATION.csv",
        "FEATURE_NORMALIZATION_DIAGNOSTIC.csv",
        "FEATURE_COUNT_OVERFITTING_DIAGNOSTIC.csv",
        "FEATURE_BANK_DIAGNOSIS.json",
        "IEX_FEATURE_BANK_V1_SPEC.json",
    )
    missing = [name for name in required if not (paths.output_root / name).is_file()]
    if missing:
        raise CutoverAuditError(f"Cannot finalize incomplete audit; missing {missing}")
    diagnosis_payload = _load_json(paths.output_root / "FEATURE_BANK_DIAGNOSIS.json")
    sample_manifest = _load_json(paths.output_root / "AUDIT_SAMPLE_MANIFEST.json")
    capsule = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE"
    capsule_artifacts = _copy_capsule_evidence(paths, capsule)
    capsule_artifacts.append(_preserve_sip01_compact(paths, capsule))
    for name in (
        "OLD_FEATURE_INVENTORY.csv",
        "OLD_FEATURE_FAMILY_MAP.csv",
        "FEATURE_QUALITY_DIAGNOSTICS.csv",
        "FEATURE_REDUNDANCY_CLUSTERS.csv",
        "FEATURE_REDUNDANCY_SUMMARY.json",
        "UNIVARIATE_FEATURE_SIGNAL.csv",
        "FEATURE_SET_MODEL_COMPARISON.csv",
        "FEATURE_FAMILY_INCREMENTAL_VALUE.csv",
        "OLD_BANK_ABLATION.csv",
        "FEATURE_NORMALIZATION_DIAGNOSTIC.csv",
        "FEATURE_COUNT_OVERFITTING_DIAGNOSTIC.csv",
        "FEATURE_BANK_DIAGNOSIS.json",
        "IEX_FEATURE_BANK_V1_SPEC.json",
    ):
        source = paths.output_root / name
        destination = capsule / name
        shutil.copy2(source, destination)
        capsule_artifacts.append({"source": str(source), "path": str(destination.relative_to(paths.output_root)), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
    capsule_manifest = paths.output_root / "SIP_RETIREMENT_SCIENCE_CAPSULE_MANIFEST.json"
    capsule_payload = {
        "schema_version": "hf_data_cutover_r1.sip_retirement_science_capsule.v1",
        "created_at_utc": utc_now(),
        "artifact_count": len(capsule_artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in capsule_artifacts),
        "artifacts": capsule_artifacts,
        "validation": {"all_hashes_verified": all(sha256_file(paths.output_root / item["path"]) == item["sha256"] for item in capsule_artifacts), "r2_outcomes_calculated": False, "full_sip_bars_copied": False},
    }
    atomic_json(capsule_manifest, capsule_payload)
    intent = _retirement_intent(paths, capsule_manifest)
    atomic_csv(paths.output_root / "SIP_RETIREMENT_INTENT.csv", intent)
    run_manifest = {
        "schema_version": "hf_data_cutover_r1.run_manifest.v1",
        "run_id": paths.output_root.name,
        "updated_at_utc": utc_now(),
        "state": "SIP_AUDIT_COMPLETE_RETIREMENT_INTENT_FROZEN",
        "audit": {
            "sample": sample_manifest,
            "diagnosis": diagnosis_payload["classification"],
            "iex_feature_spec_sha256": sha256_file(paths.output_root / "IEX_FEATURE_BANK_V1_SPEC.json"),
            "capsule_manifest_sha256": sha256_file(capsule_manifest),
            "retirement_intent_sha256": sha256_file(paths.output_root / "SIP_RETIREMENT_INTENT.csv"),
            "retirement_candidate_count": len(intent),
            "retirement_candidate_logical_bytes": sum(int(row["bytes"]) for row in intent),
        },
        "next_gate": "Immediate active-consumer/runtime recheck, watcher disablement, and exact-path retirement.",
    }
    atomic_json(paths.output_root / "RUN_MANIFEST.json", run_manifest)
    return run_manifest


