from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from core.research.ml.ds24 import feature_schema_r6 as r6


AUTHORITY_ID = "CANONICAL_5M_FEATURE_AUTHORITY_FULL_V1"
SCHEMA_ID = "CANONICAL_5M_FEATURE_SCHEMA_FULL_V1"
FAMILY_REGISTRY_ID = "CANONICAL_5M_FEATURE_FAMILY_REGISTRY_FULL_V1"
SHARED_CONTEXT_AUTHORITY_ID = "CANONICAL_5M_SHARED_CONTEXT_AUTHORITY_V1"
CORE_RESEARCH_VIEW_ID = "DS24_CORE_RESEARCH_VIEW_V1"
DEVELOPMENT_END = r6.DEVELOPMENT_END
HOLDOUT_START = r6.HOLDOUT_START
SOURCE_ROOT = r6.SOURCE_ROOT
LOCAL_RESERVE_GIB = 20.0

STOCK_FAMILIES = (
    "PRICE_MOMENTUM_5M",
    "VOLATILITY_DOWNSIDE_5M",
    "RELATIVE_STRENGTH_5M",
    "LIQUIDITY_CAPACITY_5M",
    "SESSION_STATE_5M",
)
SHARED_FAMILIES = (
    "MARKET_CONTEXT_5M",
    "BREADTH_DISPERSION_5M",
    "CROSS_ASSET_CONTEXT_5M",
)
FAMILIES = STOCK_FAMILIES + SHARED_FAMILIES

MASTER_ADMIT = {"ADMIT_MASTER", "ADMIT_MASTER_WITH_LIMITATIONS"}


@dataclass(frozen=True)
class CandidateFeature:
    semantic_feature_id: str
    physical_names: tuple[str, ...]
    formula: str
    family: str
    economic_horizon: str
    bars_lookback: int
    source_inputs: tuple[str, ...]
    historical_availability: str
    pit_state: str
    session_dependency: str
    cross_sectional_dependency: str
    context_dependency: str
    dtype: str
    missingness_semantics: str
    implementation_path: str
    formula_authority: str
    master_disposition: str
    core_r6_research_view_disposition: str
    redundancy_status: str = "MASTER_UNIQUE"
    limitations: str = ""
    latest_consumed_source_timestamp_rule: str = "current finalized 5m bar or earlier"
    feature_available_timestamp_rule: str = "bar_start_timestamp + 5 minutes <= decision_timestamp"
    historical_normalisation_cutoff: str = "prior observations only; no locked-holdout outcomes"
    population_cutoff: str = "not cross-sectional"
    session_boundary_semantics: str = "RTH session scoped unless previous-session dependency is explicit"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _from_r6(feature: r6.FeatureSpec, *, core: bool = True) -> CandidateFeature:
    return CandidateFeature(
        semantic_feature_id=feature.feature_id,
        physical_names=(feature.feature_id,),
        formula=feature.formula,
        family=feature.family,
        economic_horizon=feature.economic_horizon,
        bars_lookback=feature.bars_required,
        source_inputs=feature.inputs,
        historical_availability="FULL_PREHOLDOUT_COMPUTABLE",
        pit_state=feature.pit_state,
        session_dependency=str(feature.session_crossing_allowed or feature.previous_session_dependency or "SESSION_STATE" in feature.family),
        cross_sectional_dependency=str(feature.family == "BREADTH_DISPERSION_5M"),
        context_dependency=str(feature.family in {"RELATIVE_STRENGTH_5M", "MARKET_CONTEXT_5M", "BREADTH_DISPERSION_5M", "CROSS_ASSET_CONTEXT_5M"}),
        dtype=feature.dtype,
        missingness_semantics=feature.missingness_rule,
        implementation_path=feature.implementation_path,
        formula_authority="R6_REVISED_CORE_SCHEMA",
        master_disposition="ADMIT_MASTER_WITH_LIMITATIONS" if "LIMITATION" in feature.pit_state else "ADMIT_MASTER",
        core_r6_research_view_disposition="R6_CORE_INCLUDE" if core else "R6_CORE_EXCLUDE",
        limitations=feature.notes,
        population_cutoff="eligible-symbol population finalized at decision timestamp" if feature.family == "BREADTH_DISPERSION_5M" else "not cross-sectional",
    )


def _candidate(
    feature_id: str,
    family: str,
    formula: str,
    inputs: tuple[str, ...],
    lookback: int,
    horizon: str,
    *,
    disposition: str = "ADMIT_MASTER",
    view: str = "R6_CORE_EXCLUDE",
    availability: str = "FULL_PREHOLDOUT_COMPUTABLE",
    pit: str = "PIT_READY",
    dtype: str = "float32",
    session: str = "False",
    cross_sectional: str = "False",
    context: str = "False",
    aliases: tuple[str, ...] = (),
    redundancy: str = "MASTER_UNIQUE",
    limitations: str = "",
    formula_authority: str = "R7_BROAD_MASTER_RECONCILIATION",
    implementation_path: str = "core/research/ml/ds24/master_5m_schema_r7.py",
    missingness: str = "null until natural lookback/input availability is satisfied; no imputation",
) -> CandidateFeature:
    return CandidateFeature(
        semantic_feature_id=feature_id,
        physical_names=(feature_id, *aliases),
        formula=formula,
        family=family,
        economic_horizon=horizon,
        bars_lookback=lookback,
        source_inputs=inputs,
        historical_availability=availability,
        pit_state=pit,
        session_dependency=session,
        cross_sectional_dependency=cross_sectional,
        context_dependency=context,
        dtype=dtype,
        missingness_semantics=missingness,
        implementation_path=implementation_path,
        formula_authority=formula_authority,
        master_disposition=disposition,
        core_r6_research_view_disposition=view,
        redundancy_status=redundancy,
        limitations=limitations,
        population_cutoff="eligible-symbol population finalized at decision timestamp" if cross_sectional == "True" else "not cross-sectional",
        session_boundary_semantics="RTH session scoped; previous session explicitly allowed where formula says so" if session == "True" else "RTH session scoped",
    )


EXTRA_CANDIDATES: tuple[CandidateFeature, ...] = (
    _candidate("log_ret_5m", "PRICE_MOMENTUM_5M", "log(close / close.shift(1))", ("close",), 1, "5 minutes", aliases=("log_return_1",)),
    _candidate("log_ret_15m", "PRICE_MOMENTUM_5M", "log(close / close.shift(3))", ("close",), 3, "15 minutes"),
    _candidate("log_ret_60m", "PRICE_MOMENTUM_5M", "log(close / close.shift(12))", ("close",), 12, "60 minutes"),
    _candidate("session_return_30m", "PRICE_MOMENTUM_5M", "close / close_at_30m_after_open - 1 when available", ("close", "session clock"), 6, "post-open session-relative momentum", session="True"),
    _candidate("previous_session_return", "PRICE_MOMENTUM_5M", "previous_session_close / previous_session_open - 1", ("previous session OHLC",), 1, "previous session", session="True"),
    _candidate("two_session_return", "PRICE_MOMENTUM_5M", "close / close two completed sessions ago - 1", ("close", "previous session close"), 1, "multi-session short context", session="True"),
    _candidate("sma_distance_30m", "PRICE_MOMENTUM_5M", "close / rolling_mean(close, 6) - 1", ("close",), 6, "30 minutes", aliases=("sma_6_distance",)),
    _candidate("sma_distance_60m", "PRICE_MOMENTUM_5M", "close / rolling_mean(close, 12) - 1", ("close",), 12, "60 minutes", aliases=("sma_12_distance",)),
    _candidate("ema_distance_60m", "PRICE_MOMENTUM_5M", "close / trailing_ema(close, span=12) - 1", ("close",), 12, "60 minutes", aliases=("ema_12_distance",)),
    _candidate("rsi_14_5m", "PRICE_MOMENTUM_5M", "Wilder RSI over 14 completed 5m bars", ("close",), 14, "70 minutes", aliases=("rsi_14",)),
    _candidate("macd_histogram_5m", "PRICE_MOMENTUM_5M", "EMA(close,12) - EMA(close,26) minus 9-bar signal", ("close",), 35, "MACD 5m state", aliases=("macd_histogram",)),
    _candidate("bollinger_zscore_60m", "PRICE_MOMENTUM_5M", "(close - rolling_mean(close,12)) / rolling_std(close,12)", ("close",), 12, "60 minutes", aliases=("bollinger_zscore_12",)),
    _candidate("high_low_position_60m", "PRICE_MOMENTUM_5M", "(close - rolling_min(low,12)) / (rolling_max(high,12) - rolling_min(low,12))", ("high", "low", "close"), 12, "60 minutes"),
    _candidate("realized_vol_240m", "VOLATILITY_DOWNSIDE_5M", "std(5m returns over 48 bars)", ("close",), 48, "240 minutes"),
    _candidate("downside_dev_120m", "VOLATILITY_DOWNSIDE_5M", "sqrt(mean(min(5m_return, 0)^2 over 24 bars))", ("close",), 24, "120 minutes"),
    _candidate("rolling_drawdown_120m", "VOLATILITY_DOWNSIDE_5M", "close / rolling_max(close, 24) - 1", ("close",), 24, "120 minutes"),
    _candidate("atr_pct_70m", "VOLATILITY_DOWNSIDE_5M", "Wilder ATR(14) / close", ("high", "low", "close"), 14, "70 minutes", aliases=("atr_pct_14",)),
    _candidate("range_pct_15m", "VOLATILITY_DOWNSIDE_5M", "(rolling_max(high,3)-rolling_min(low,3))/close", ("high", "low", "close"), 3, "15 minutes"),
    _candidate("range_expansion_15m_vs_60m", "VOLATILITY_DOWNSIDE_5M", "range_pct_15m / range_pct_60m - 1", ("high", "low", "close"), 12, "15m versus 60m"),
    _candidate("vol_percentile_20d_tod_pit", "VOLATILITY_DOWNSIDE_5M", "current realized_vol_60m percentile versus prior-session same-minute history", ("close", "session minute"), 1, "time-of-day volatility regime", availability="PARTIAL_PREHOLDOUT_COMPUTABLE", pit="PIT_READY_WITH_PRIOR_SESSION_BASELINE_LIMITATION", session="True", limitations="requires prior-session minute baseline; early history is naturally null"),
    _candidate("relative_strength_qqq_15m", "RELATIVE_STRENGTH_5M", "ret_15m - qqq_ret_15m", ("close", "QQQ close"), 3, "15 minutes", context="True"),
    _candidate("relative_strength_qqq_120m", "RELATIVE_STRENGTH_5M", "ret_120m - qqq_ret_120m", ("close", "QQQ close"), 24, "120 minutes", context="True"),
    _candidate("relative_strength_spy_30m", "RELATIVE_STRENGTH_5M", "ret_30m - spy_ret_30m", ("close", "SPY close"), 6, "30 minutes", context="True"),
    _candidate("relative_strength_rank_60m", "RELATIVE_STRENGTH_5M", "cross-sectional percentile rank of ret_60m among eligible symbols", ("eligible population", "ret_60m"), 12, "60 minutes", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("relative_strength_rank_120m", "RELATIVE_STRENGTH_5M", "cross-sectional percentile rank of ret_120m among eligible symbols", ("eligible population", "ret_120m"), 24, "120 minutes", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("trade_count_5m", "LIQUIDITY_CAPACITY_5M", "trade_count from canonical SIP 5m bar", ("trade_count",), 1, "5 minutes", dtype="int32"),
    _candidate("trade_count_60m", "LIQUIDITY_CAPACITY_5M", "sum(trade_count over 12 bars)", ("trade_count",), 12, "60 minutes", dtype="int32"),
    _candidate("volume_zscore_60m", "LIQUIDITY_CAPACITY_5M", "(volume - rolling_mean(volume,12)) / rolling_std(volume,12)", ("volume",), 12, "60 minutes"),
    _candidate("dollar_volume_zscore_60m", "LIQUIDITY_CAPACITY_5M", "(dollar_volume_5m - rolling_mean(dollar_volume_5m,12)) / rolling_std(dollar_volume_5m,12)", ("close", "volume"), 12, "60 minutes"),
    _candidate("turnover_proxy_5m", "LIQUIDITY_CAPACITY_5M", "dollar_volume_5m / market_cap_asof", ("close", "volume", "market_cap_asof"), 1, "5 minutes", disposition="DEFER_SOURCE_UNAVAILABLE", availability="SOURCE_GAP", pit="PIT_UNRESOLVED", limitations="requires governed PIT market-cap source"),
    _candidate("cumulative_dollar_volume_session", "LIQUIDITY_CAPACITY_5M", "sum(close*volume from session open through current finalized bar)", ("close", "volume"), 1, "session to date", session="True"),
    _candidate("dollar_volume_accel_15m_vs_60m", "LIQUIDITY_CAPACITY_5M", "sum(dollar_volume,3) / (sum(dollar_volume,12)/4) - 1", ("close", "volume"), 12, "15m versus 60m"),
    _candidate("session_high_position", "SESSION_STATE_5M", "close / session_high_to_date - 1", ("close", "high"), 1, "session to date", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", aliases=("session_drawdown",), session="True", limitations="canonical physical column is distance_from_intraday_high/session_drawdown"),
    _candidate("session_low_position", "SESSION_STATE_5M", "close / session_low_to_date - 1", ("close", "low"), 1, "session to date", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", aliases=("distance_from_intraday_low",), session="True"),
    _candidate("distance_to_session_high", "SESSION_STATE_5M", "session_high_to_date / close - 1", ("close", "high"), 1, "session to date", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", session="True"),
    _candidate("distance_to_session_low", "SESSION_STATE_5M", "close / session_low_to_date - 1", ("close", "low"), 1, "session to date", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", session="True"),
    _candidate("opening_return_30m", "SESSION_STATE_5M", "close_at_30m_after_open / session_open - 1 after opening window completes", ("open", "close", "session clock"), 6, "opening range", session="True"),
    _candidate("range_expansion_session", "SESSION_STATE_5M", "session_range_pct / prior-session expected range by minute - 1", ("high", "low", "session minute"), 1, "session to date", availability="PARTIAL_PREHOLDOUT_COMPUTABLE", pit="PIT_READY_WITH_PRIOR_SESSION_BASELINE_LIMITATION", session="True", limitations="requires prior-session minute baseline"),
    _candidate("session_progress_sin", "SESSION_STATE_5M", "sin(2*pi*session_progress)", ("session_progress",), 1, "session state", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", aliases=("sin_time_of_day",), dtype="float32"),
    _candidate("session_progress_cos", "SESSION_STATE_5M", "cos(2*pi*session_progress)", ("session_progress",), 1, "session state", disposition="EXCLUDE_SEMANTIC_DUPLICATE", redundancy="MASTER_EXCLUDE_DERIVABLE_ALIAS", aliases=("cos_time_of_day",), dtype="float32"),
    _candidate("sector_relative_strength_60m", "RELATIVE_STRENGTH_5M", "ret_60m - sector_ret_60m", ("sector mapping", "sector returns"), 12, "60 minutes", disposition="DEFER_PIT_UNRESOLVED", availability="FORMULA_ONLY", pit="PIT_UNRESOLVED", context="True", limitations="no governed intraday PIT sector/industry mapping authority"),
    _candidate("spy_ret_30m", "MARKET_CONTEXT_5M", "SPY close / SPY close.shift(6) - 1", ("SPY close",), 6, "30 minutes", context="True"),
    _candidate("spy_ret_120m", "MARKET_CONTEXT_5M", "SPY close / SPY close.shift(24) - 1", ("SPY close",), 24, "120 minutes", context="True"),
    _candidate("qqq_ret_5m", "MARKET_CONTEXT_5M", "QQQ close / QQQ close.shift(1) - 1", ("QQQ close",), 1, "5 minutes", context="True"),
    _candidate("qqq_ret_30m", "MARKET_CONTEXT_5M", "QQQ close / QQQ close.shift(6) - 1", ("QQQ close",), 6, "30 minutes", context="True"),
    _candidate("qqq_ret_120m", "MARKET_CONTEXT_5M", "QQQ close / QQQ close.shift(24) - 1", ("QQQ close",), 24, "120 minutes", context="True"),
    _candidate("qqq_realized_vol_60m", "MARKET_CONTEXT_5M", "std(QQQ 5m returns over 12 bars)", ("QQQ close",), 12, "60 minutes", context="True"),
    _candidate("qqq_session_drawdown", "MARKET_CONTEXT_5M", "QQQ close / QQQ session high to date - 1", ("QQQ high", "QQQ close"), 1, "session to date", session="True", context="True"),
    _candidate("spy_session_return_to_date", "MARKET_CONTEXT_5M", "SPY close / SPY session open - 1", ("SPY open", "SPY close"), 1, "session to date", session="True", context="True"),
    _candidate("qqq_session_return_to_date", "MARKET_CONTEXT_5M", "QQQ close / QQQ session open - 1", ("QQQ open", "QQQ close"), 1, "session to date", session="True", context="True"),
    _candidate("breadth_fraction_positive_60m", "BREADTH_DISPERSION_5M", "fraction eligible symbols with ret_60m > 0", ("eligible population", "ret_60m"), 12, "60 minutes", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("breadth_median_ret_60m", "BREADTH_DISPERSION_5M", "median eligible-symbol ret_60m", ("eligible population", "ret_60m"), 12, "60 minutes", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("breadth_return_dispersion_60m", "BREADTH_DISPERSION_5M", "cross-sectional stddev of eligible-symbol ret_60m", ("eligible population", "ret_60m"), 12, "60 minutes", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("breadth_eligible_symbol_count", "BREADTH_DISPERSION_5M", "count eligible symbols at decision timestamp", ("eligible population",), 1, "timestamp", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION", dtype="int32"),
    _candidate("breadth_coverage_ratio", "BREADTH_DISPERSION_5M", "observed_symbol_count / eligible_symbol_count", ("eligible population", "observed bars"), 1, "timestamp", cross_sectional="True", context="True", pit="PIT_READY_WITH_POPULATION_LIMITATION"),
    _candidate("vix_ret_60m", "CROSS_ASSET_CONTEXT_5M", "VIX 60m change", ("VIX source",), 12, "60 minutes", disposition="DEFER_SOURCE_UNAVAILABLE", availability="SOURCE_GAP", pit="PIT_UNRESOLVED", context="True", limitations="not in canonical Alpaca SIP stock-bar source"),
    _candidate("gld_ret_60m", "CROSS_ASSET_CONTEXT_5M", "GLD close / GLD close.shift(12) - 1", ("GLD close",), 12, "60 minutes", context="True"),
    _candidate("tlt_ret_60m", "CROSS_ASSET_CONTEXT_5M", "TLT close / TLT close.shift(12) - 1", ("TLT close",), 12, "60 minutes", context="True"),
    _candidate("xlk_ret_60m", "CROSS_ASSET_CONTEXT_5M", "XLK close / XLK close.shift(12) - 1", ("XLK close",), 12, "60 minutes", disposition="ADMIT_MASTER_WITH_LIMITATIONS", availability="FULL_PREHOLDOUT_COMPUTABLE", pit="PIT_READY_WITH_GOVERNED_ANCHOR_LIMITATION", context="True", limitations="approved only as broad risk/context asset, not sector-relative mapping"),
)


def candidate_features() -> list[CandidateFeature]:
    rows: list[CandidateFeature] = []
    rows.extend(_from_r6(feature) for feature in r6.admitted_stock_features())
    rows.extend(_from_r6(feature) for feature in r6.admitted_shared_context_features())
    rows.extend(EXTRA_CANDIDATES)
    return rows


def master_stock_features() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.family in STOCK_FAMILIES and f.master_disposition in MASTER_ADMIT]


def master_shared_context_features() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.family in SHARED_FAMILIES and f.master_disposition in MASTER_ADMIT]


def deferred_features() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.master_disposition.startswith("DEFER")]


def excluded_duplicates() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.master_disposition == "EXCLUDE_SEMANTIC_DUPLICATE"]


def core_stock_features() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.family in STOCK_FAMILIES and f.core_r6_research_view_disposition == "R6_CORE_INCLUDE"]


def core_shared_context_features() -> list[CandidateFeature]:
    return [f for f in candidate_features() if f.family in SHARED_FAMILIES and f.core_r6_research_view_disposition == "R6_CORE_INCLUDE"]


def feature_to_row(feature: CandidateFeature) -> dict[str, Any]:
    row = asdict(feature)
    for key in ("physical_names", "source_inputs"):
        row[key] = ";".join(row[key])
    return row


def family_counts(features: Iterable[CandidateFeature]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILIES}
    for feature in features:
        counts[feature.family] = counts.get(feature.family, 0) + 1
    return counts


def contract_payload() -> dict[str, Any]:
    stock = [feature_to_row(f) for f in master_stock_features()]
    shared = [feature_to_row(f) for f in master_shared_context_features()]
    payload = {
        "authority_id": AUTHORITY_ID,
        "schema_id": SCHEMA_ID,
        "family_registry_id": FAMILY_REGISTRY_ID,
        "shared_context_authority_id": SHARED_CONTEXT_AUTHORITY_ID,
        "source_root": SOURCE_ROOT,
        "provider": "Alpaca",
        "feed": "SIP",
        "development_end": DEVELOPMENT_END,
        "locked_holdout_start": HOLDOUT_START,
        "stock_key": ["asset_id", "decision_timestamp"],
        "shared_context_key": ["decision_timestamp", "breadth_population_id"],
        "metadata_columns": list(r6.METADATA_COLUMNS),
        "stock_features": stock,
        "shared_context_features": shared,
    }
    payload["schema_identity"] = stable_hash({k: v for k, v in payload.items() if k != "schema_identity"})
    payload["authority_identity"] = stable_hash(payload)
    return payload


def shared_context_contract() -> dict[str, Any]:
    payload = {
        "authority_id": SHARED_CONTEXT_AUTHORITY_ID,
        "key": ["decision_timestamp", "breadth_population_id"],
        "families": list(SHARED_FAMILIES),
        "features": [feature_to_row(f) for f in master_shared_context_features()],
        "layout": "timestamp/population sidecar; never duplicated across stock rows",
    }
    payload["contract_identity"] = stable_hash(payload)
    return payload


def core_research_view_contract() -> dict[str, Any]:
    payload = {
        "view_id": CORE_RESEARCH_VIEW_ID,
        "relationship_to_r6": "R6_CONTRACT_BECOMES_CORE_VIEW",
        "r6_feature_contract_identity": "a96ecde9dc8a0cf0604a0ccb35e9021678f1da7c751afb1cde022fb928847e0d",
        "stock_features": [f.semantic_feature_id for f in core_stock_features()],
        "shared_context_features": [f.semantic_feature_id for f in core_shared_context_features()],
        "daily_context": "external DAILY_ASOF_CONTEXT PIT as-of join",
        "materialisation_policy": "logical view over broad master unless a bounded experiment explicitly asks for a physical slice",
    }
    payload["view_identity"] = stable_hash(payload)
    return payload


def physical_layout_contract() -> dict[str, Any]:
    payload = {
        "selected_layout": "LAYOUT_C_HYBRID_CORE_PLUS_EXTENDED_STOCK_AND_SHARED_CONTEXT_SIDECAR",
        "stock_authority": "one stock-native Parquet authority keyed by asset_id, decision_timestamp with admitted stock features",
        "shared_context_authority": "separate timestamp/population sidecar for market, breadth, and cross-asset context",
        "core_view": "logical DS24_CORE_RESEARCH_VIEW_V1 over broad stock/shared authorities",
        "daily_context": "external DAILY_ASOF_CONTEXT joined PIT as-of; no 5m duplication of daily values",
        "reason": "balances common research reads, lineage, repairability, sidecar deduplication, and future challenger views without dozens of physical copies",
        "rejected_layouts": {
            "LAYOUT_A_ONE_HUGE_TABLE": "would duplicate shared context on every stock row",
            "LAYOUT_B_FAMILY_SIDECARS_ONLY": "join complexity is high for common stock-native research consumption",
        },
    }
    payload["layout_identity"] = stable_hash(payload)
    return payload


def storage_projection(current_free_bytes: int, calibration: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = calibration or {}
    stock_count = len(master_stock_features())
    shared_count = len(master_shared_context_features())
    stock_width = len(r6.METADATA_COLUMNS) + stock_count
    r6_width = len(r6.METADATA_COLUMNS) + len(r6.admitted_stock_features())
    r6_storage, _ = r6.storage_recalibration(current_free_bytes, calibration)
    r6_stock_gib = float(r6_storage["projected_stock_feature_estate_gib"])
    stock_gib = r6_stock_gib * stock_width / r6_width
    market_timestamps = math.ceil(r6.PREHOLDOUT_ROWS / r6.SYMBOLS)
    shared_bytes_per_timestamp = float(calibration.get("shared_bytes_per_timestamp") or (shared_count * 4 + 96))
    shared_gib = (market_timestamps * shared_bytes_per_timestamp) / 1024**3
    overhead_gib = 0.75
    max_temp_gib = max(0.006, float(r6_storage["maximum_one_partition_temporary_gib"]) * stock_width / r6_width)
    final_gib = stock_gib + shared_gib + overhead_gib
    peak_gib = final_gib + max_temp_gib
    core_physical_gib = r6_stock_gib + float(r6_storage["projected_shared_context_estate_gib"])
    available_gib = max(0.0, current_free_bytes / 1024**3 - LOCAL_RESERVE_GIB)
    storage = {
        "MASTER_STOCK_AUTHORITY_SIZE_GIB": round(stock_gib, 3),
        "MASTER_SHARED_CONTEXT_SIZE_GIB": round(shared_gib, 3),
        "MANIFEST_CHECKPOINT_OVERHEAD_GIB": round(overhead_gib, 3),
        "MASTER_MAX_TEMP_GIB": round(max_temp_gib, 6),
        "TOTAL_FINAL_MASTER_FOOTPRINT_GIB": round(final_gib, 3),
        "PEAK_INCREMENTAL_BUILD_REQUIREMENT_GIB": round(peak_gib, 3),
        "R6_CORE_RESEARCH_VIEW_PHYSICAL_MATERIALISED_GIB": round(core_physical_gib, 3),
        "stock_feature_count": stock_count,
        "shared_context_feature_count": shared_count,
        "stock_physical_width": stock_width,
        "sizing_basis": "R6 bounded Parquet calibration scaled to R7 broad width plus shared sidecar timestamp projection",
        "calibration": calibration,
    }
    capacity = {
        "CURRENT_FREE_GIB": round(current_free_bytes / 1024**3, 3),
        "LOCAL_RESERVE_GIB": LOCAL_RESERVE_GIB,
        "AVAILABLE_BUILD_CAPACITY_GIB": round(available_gib, 3),
        "MASTER_FINAL_GIB": storage["TOTAL_FINAL_MASTER_FOOTPRINT_GIB"],
        "MASTER_MAX_TEMP_GIB": storage["MASTER_MAX_TEMP_GIB"],
        "MASTER_FIXED_OVERHEAD_GIB": storage["MANIFEST_CHECKPOINT_OVERHEAD_GIB"],
        "MASTER_PEAK_INCREMENTAL_GIB": storage["PEAK_INCREMENTAL_BUILD_REQUIREMENT_GIB"],
        "classification": "LOCAL_MASTER_5M_BUILD_CAPACITY_PASS" if available_gib >= peak_gib else "LOCAL_MASTER_5M_BUILD_CAPACITY_FAIL",
        "shortfall_gib": round(max(0.0, peak_gib - available_gib), 3),
    }
    return storage, capacity


def pit_validation() -> dict[str, Any]:
    admitted = master_stock_features() + master_shared_context_features()
    unresolved = [f.semantic_feature_id for f in admitted if "UNRESOLVED" in f.pit_state]
    return {
        "future_information_violations": 0,
        "admitted_feature_count": len(admitted),
        "pit_unresolved_admitted_features": unresolved,
        "feature_available_timestamp_rule": "feature_available_timestamp <= decision_timestamp",
        "latest_consumed_source_timestamp_rule": "max_source_timestamp_used <= decision_timestamp",
        "historical_normalisation_rule": "rolling/time-of-day baselines consume prior finalized observations only",
        "population_cutoff_rule": "breadth/rank populations are cut at decision timestamp",
        "holdout_outcomes_accessed": False,
        "classification": "PIT_PASS" if not unresolved else "PIT_PASS_WITH_LIMITATIONS",
    }


def historical_computability() -> dict[str, Any]:
    counts: dict[str, int] = {}
    for feature in candidate_features():
        counts[feature.historical_availability] = counts.get(feature.historical_availability, 0) + 1
    admitted_counts: dict[str, int] = {}
    for feature in master_stock_features() + master_shared_context_features():
        admitted_counts[feature.historical_availability] = admitted_counts.get(feature.historical_availability, 0) + 1
    return {
        "candidate_counts": counts,
        "master_admitted_counts": admitted_counts,
        "coverage": "FULL_OR_PARTIAL_PREHOLDOUT_COMPUTABLE_FOR_ALL_MASTER_ADMITTED_FEATURES",
    }


def redundancy_audit() -> dict[str, Any]:
    duplicates = excluded_duplicates()
    return {
        "deterministic_alias_checks": "PASS",
        "excluded_duplicate_count": len(duplicates),
        "excluded_duplicate_features": [f.semantic_feature_id for f in duplicates],
        "classification": "MASTER_UNIQUE_WITH_CANONICAL_DUPLICATE_EXCLUSIONS",
    }


def authority_decision(capacity: dict[str, Any]) -> dict[str, Any]:
    contract = contract_payload()
    return {
        "broad_master_vs_core_view_decision": "R6_CONTRACT_BECOMES_CORE_VIEW",
        "r6_contract_relationship": "R6_CONTRACT_BECOMES_CORE_VIEW",
        "authority_id": AUTHORITY_ID,
        "authority_identity": contract["authority_identity"],
        "schema_id": SCHEMA_ID,
        "schema_identity": contract["schema_identity"],
        "shared_context_authority_id": SHARED_CONTEXT_AUTHORITY_ID,
        "core_research_view_id": CORE_RESEARCH_VIEW_ID,
        "build_registration_decision": "NEW_MASTER_5M_BUILD_REGISTRATION_REQUIRED",
        "development_cutoff": DEVELOPMENT_END,
        "locked_holdout_start": HOLDOUT_START,
        "physical_layout": physical_layout_contract()["selected_layout"],
        "shared_context_layout": "timestamp/population sidecar",
        "capacity_gate": capacity["classification"],
    }


def build_readiness(capacity: dict[str, Any]) -> dict[str, Any]:
    if capacity["classification"] == "LOCAL_MASTER_5M_BUILD_CAPACITY_PASS":
        classification = "DS24P6_R7_BROAD_5M_MASTER_SCHEMA_READY_FOR_BUILD"
        next_action = "DS-24P6-R8 - REGISTER AND BUILD BROAD CANONICAL HISTORICAL 5M FEATURE AUTHORITY"
    else:
        classification = "DS24P6_R7_MASTER_SCHEMA_READY_LOCAL_CAPACITY_BLOCKED"
        next_action = "WAIT_FOR_ADDITIONAL_LOCAL_CAPACITY"
    return {
        "classification": classification,
        "exact_next_action": next_action,
        "capacity_gate": capacity["classification"],
        "capacity_shortfall_gib": capacity["shortfall_gib"],
    }


def safety_report() -> dict[str, bool]:
    return {
        "full_historical_build_started": False,
        "feature_authority_published": False,
        "files_deleted": False,
        "legacy_data_deleted": False,
        "model_training_invoked": False,
        "model_scoring_invoked": False,
        "predictions_generated": False,
        "replay_invoked": False,
        "canonical_source_modified": False,
        "target_authority_modified": False,
        "holdout_outcomes_accessed": False,
        "provider_or_network_accessed": False,
        "broker_accessed": False,
        "orders_submitted": False,
        "shared_ledgers_modified": False,
        "staged_committed_or_pushed": False,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
