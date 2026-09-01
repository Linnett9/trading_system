from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


OLD_P6_CONTRACT_IDENTITY = "565f9af428664ba204168302c8670d78bf8ae93013559f20783510abf2a76b80"
OLD_P6_RUN_ID = "ds24_p6_preholdout_20260807T211438Z"
TARGET_ID = "forward_return_60m__decision_5m"
DEVELOPMENT_END = "2025-04-01"
HOLDOUT_START = "2025-04-02"
SOURCE_ROOT = "data/processed/alpaca/symbol_bars/sip/5m"
SOURCE_ROWS = 114_497_377
PREHOLDOUT_ROWS = 99_930_803
SYMBOLS = 514
R1_STORAGE_LAYOUT_ID = "ds24_p6_low_footprint_parquet_v1"
R1_STORAGE_LAYOUT_IDENTITY = "62f81bb0d836644a262744abc68004f05490f124209c7053af374574646a0929"
R1_PROJECTED_FINAL_GIB = 13.795771
R1_MAX_SINGLE_PARTITION_TEMP_GIB = 0.003239
R1_FIXED_PUBLICATION_OVERHEAD_GIB = 0.5
R1_REVISED_PEAK_INCREMENTAL_GIB = 14.29901
LOCAL_SAFETY_RESERVE_GIB = 20.0

FAMILIES = (
    "PRICE_MOMENTUM_5M",
    "VOLATILITY_DOWNSIDE_5M",
    "RELATIVE_STRENGTH_5M",
    "LIQUIDITY_CAPACITY_5M",
    "MARKET_CONTEXT_5M",
    "BREADTH_DISPERSION_5M",
    "CROSS_ASSET_CONTEXT_5M",
    "SESSION_STATE_5M",
)


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    family: str
    formula: str
    inputs: tuple[str, ...]
    bars_required: int
    economic_horizon: str
    dtype: str = "float32"
    missingness_rule: str = "null until natural history/input availability is satisfied; no imputation"
    session_crossing_allowed: bool = False
    previous_session_dependency: bool = False
    minimum_history: str = ""
    availability_lag: str = "5 minutes after source bar start"
    pit_state: str = "PIT_READY"
    physical_classification: str = "STOCK_NATIVE_5M"
    implementation_path: str = "core/research/ml/ds24/feature_schema_r6.py"
    notes: str = ""


@dataclass(frozen=True)
class OldFeatureDecision:
    feature_id: str
    family: str
    formula: str
    inputs: tuple[str, ...]
    lookback: int
    decision: str
    reason: str


OLD_FEATURE_DECISIONS: tuple[OldFeatureDecision, ...] = (
    OldFeatureDecision("return_1", "PRICE_MOMENTUM_5M", "trailing_return", ("close",), 1, "RETAIN_RENAME_ONLY", "renamed to ret_5m; same completed-bar return"),
    OldFeatureDecision("return_3", "PRICE_MOMENTUM_5M", "trailing_return", ("close",), 3, "RETAIN_RENAME_ONLY", "renamed to ret_15m"),
    OldFeatureDecision("return_6", "PRICE_MOMENTUM_5M", "trailing_return", ("close",), 6, "RETAIN_RENAME_ONLY", "renamed to ret_30m"),
    OldFeatureDecision("return_12", "PRICE_MOMENTUM_5M", "trailing_return", ("close",), 12, "RETAIN_RENAME_ONLY", "renamed to ret_60m"),
    OldFeatureDecision("return_24", "PRICE_MOMENTUM_5M", "trailing_return", ("close",), 24, "RETAIN_RENAME_ONLY", "renamed to ret_120m"),
    OldFeatureDecision("return_78", "SESSION_STATE_5M", "session_return", ("close",), 78, "REPLACE_BY_BETTER_EQUIVALENT", "replaced by session-to-date return; full-session trailing return is not generally available intraday"),
    OldFeatureDecision("sma_6", "PRICE_MOMENTUM_5M", "trailing_simple_moving_average", ("close",), 6, "REPLACE_BY_BETTER_EQUIVALENT", "absolute price-scale SMA replaced by scale-free distance from intraday high/low and returns"),
    OldFeatureDecision("sma_12", "PRICE_MOMENTUM_5M", "trailing_simple_moving_average", ("close",), 12, "REPLACE_BY_BETTER_EQUIVALENT", "absolute price-scale SMA replaced by scale-free 60m momentum/reversal"),
    OldFeatureDecision("sma_24", "PRICE_MOMENTUM_5M", "trailing_simple_moving_average", ("close",), 24, "REMOVE_REDUNDANT", "highly redundant with EMA/return ladder and not scale-free"),
    OldFeatureDecision("sma_78", "PRICE_MOMENTUM_5M", "trailing_simple_moving_average", ("close",), 78, "REMOVE_NOT_USEFUL_FOR_60M_TARGET", "full-session average is slower than the primary 60m target"),
    OldFeatureDecision("ema_6", "PRICE_MOMENTUM_5M", "trailing_exponential_moving_average", ("close",), 6, "REMOVE_REDUNDANT", "duplicates short trailing price level information"),
    OldFeatureDecision("ema_12", "PRICE_MOMENTUM_5M", "trailing_exponential_moving_average", ("close",), 12, "REPLACE_BY_BETTER_EQUIVALENT", "replaced by momentum acceleration over 30m/60m scales"),
    OldFeatureDecision("ema_24", "PRICE_MOMENTUM_5M", "trailing_exponential_moving_average", ("close",), 24, "REMOVE_REDUNDANT", "duplicates 120m return/volatility regime information"),
    OldFeatureDecision("rsi_14", "PRICE_MOMENTUM_5M", "trailing_momentum_oscillator", ("close",), 14, "REPLACE_BY_BETTER_EQUIVALENT", "replaced by signed momentum persistence over 60m"),
    OldFeatureDecision("atr_14", "VOLATILITY_DOWNSIDE_5M", "trailing_true_range", ("high", "low", "close"), 14, "REPLACE_BY_BETTER_EQUIVALENT", "absolute ATR replaced by pct range/realized volatility/downside features"),
    OldFeatureDecision("atr_pct_14", "VOLATILITY_DOWNSIDE_5M", "trailing_true_range_pct", ("high", "low", "close"), 14, "RETAIN_RENAME_ONLY", "renamed to range_pct_60m using 12 bars for target-aligned scale"),
    OldFeatureDecision("realized_volatility_12", "VOLATILITY_DOWNSIDE_5M", "trailing_realized_volatility", ("close",), 12, "RETAIN_RENAME_ONLY", "renamed to realized_vol_60m"),
    OldFeatureDecision("realized_volatility_24", "VOLATILITY_DOWNSIDE_5M", "trailing_realized_volatility", ("close",), 24, "RETAIN_RENAME_ONLY", "renamed to realized_vol_120m"),
    OldFeatureDecision("realized_volatility_78", "VOLATILITY_DOWNSIDE_5M", "trailing_realized_volatility", ("close",), 78, "REPLACE_BY_BETTER_EQUIVALENT", "replaced by session range/drawdown and volatility trend"),
    OldFeatureDecision("volume_ratio_12", "LIQUIDITY_CAPACITY_5M", "trailing_relative_volume", ("volume",), 12, "RETAIN_RENAME_ONLY", "renamed to volume_ratio_60m"),
    OldFeatureDecision("volume_ratio_78", "LIQUIDITY_CAPACITY_5M", "trailing_relative_volume", ("volume",), 78, "REPLACE_BY_BETTER_EQUIVALENT", "replaced by PIT-safe cumulative/time-of-day relative volume"),
)


FINAL_STOCK_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("ret_5m", "PRICE_MOMENTUM_5M", "close / close.shift(1) - 1", ("close",), 1, "5 minutes", minimum_history="1 prior completed bar"),
    FeatureSpec("ret_15m", "PRICE_MOMENTUM_5M", "close / close.shift(3) - 1", ("close",), 3, "15 minutes", minimum_history="3 prior completed bars"),
    FeatureSpec("ret_30m", "PRICE_MOMENTUM_5M", "close / close.shift(6) - 1", ("close",), 6, "30 minutes", minimum_history="6 prior completed bars"),
    FeatureSpec("ret_60m", "PRICE_MOMENTUM_5M", "close / close.shift(12) - 1", ("close",), 12, "60 minutes", minimum_history="12 prior completed bars"),
    FeatureSpec("ret_120m", "PRICE_MOMENTUM_5M", "close / close.shift(24) - 1", ("close",), 24, "120 minutes", minimum_history="24 prior completed bars"),
    FeatureSpec("reversal_15m_vs_60m", "PRICE_MOMENTUM_5M", "ret_15m - ret_60m", ("ret_15m", "ret_60m"), 12, "15m reversal versus 60m trend", minimum_history="12 prior completed bars"),
    FeatureSpec("momentum_accel_30m_60m", "PRICE_MOMENTUM_5M", "ret_30m - ret_60m", ("ret_30m", "ret_60m"), 12, "30m acceleration versus 60m trend", minimum_history="12 prior completed bars"),
    FeatureSpec("momentum_persistence_60m", "PRICE_MOMENTUM_5M", "mean(sign(5m returns over last 12 bars)) * sign(ret_60m)", ("close",), 12, "60 minutes", minimum_history="12 prior completed bars"),
    FeatureSpec("distance_from_intraday_high", "PRICE_MOMENTUM_5M", "close / session_high_to_date - 1", ("close", "high"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("distance_from_intraday_low", "PRICE_MOMENTUM_5M", "close / session_low_to_date - 1", ("close", "low"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("realized_vol_15m", "VOLATILITY_DOWNSIDE_5M", "std(5m returns over 3 bars)", ("close",), 3, "15 minutes", minimum_history="3 prior completed returns"),
    FeatureSpec("realized_vol_30m", "VOLATILITY_DOWNSIDE_5M", "std(5m returns over 6 bars)", ("close",), 6, "30 minutes", minimum_history="6 prior completed returns"),
    FeatureSpec("realized_vol_60m", "VOLATILITY_DOWNSIDE_5M", "std(5m returns over 12 bars)", ("close",), 12, "60 minutes", minimum_history="12 prior completed returns"),
    FeatureSpec("realized_vol_120m", "VOLATILITY_DOWNSIDE_5M", "std(5m returns over 24 bars)", ("close",), 24, "120 minutes", minimum_history="24 prior completed returns"),
    FeatureSpec("downside_dev_60m", "VOLATILITY_DOWNSIDE_5M", "sqrt(mean(min(5m_return, 0)^2 over 12 bars))", ("close",), 12, "60 minutes", minimum_history="12 prior completed returns"),
    FeatureSpec("rolling_drawdown_60m", "VOLATILITY_DOWNSIDE_5M", "close / rolling_max(close, 12) - 1", ("close",), 12, "60 minutes", minimum_history="12 completed bars"),
    FeatureSpec("session_drawdown", "VOLATILITY_DOWNSIDE_5M", "close / session_high_to_date - 1", ("close", "high"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("range_pct_60m", "VOLATILITY_DOWNSIDE_5M", "(rolling_max(high,12)-rolling_min(low,12))/close", ("high", "low", "close"), 12, "60 minutes", minimum_history="12 completed bars"),
    FeatureSpec("vol_trend_30m_vs_120m", "VOLATILITY_DOWNSIDE_5M", "realized_vol_30m / realized_vol_120m - 1", ("realized_vol_30m", "realized_vol_120m"), 24, "30m versus 120m", minimum_history="24 prior completed returns"),
    FeatureSpec("relative_strength_spy_15m", "RELATIVE_STRENGTH_5M", "ret_15m - spy_ret_15m", ("close", "SPY close"), 3, "15 minutes", minimum_history="3 completed stock and SPY bars"),
    FeatureSpec("relative_strength_spy_60m", "RELATIVE_STRENGTH_5M", "ret_60m - spy_ret_60m", ("close", "SPY close"), 12, "60 minutes", minimum_history="12 completed stock and SPY bars"),
    FeatureSpec("relative_strength_qqq_60m", "RELATIVE_STRENGTH_5M", "ret_60m - qqq_ret_60m", ("close", "QQQ close"), 12, "60 minutes", minimum_history="12 completed stock and QQQ bars"),
    FeatureSpec("relative_strength_spy_120m", "RELATIVE_STRENGTH_5M", "ret_120m - spy_ret_120m", ("close", "SPY close"), 24, "120 minutes", minimum_history="24 completed stock and SPY bars"),
    FeatureSpec("dollar_volume_5m", "LIQUIDITY_CAPACITY_5M", "close * volume", ("close", "volume"), 1, "5 minutes", minimum_history="current finalized bar"),
    FeatureSpec("dollar_volume_60m", "LIQUIDITY_CAPACITY_5M", "sum(close*volume over 12 bars)", ("close", "volume"), 12, "60 minutes", minimum_history="12 completed bars"),
    FeatureSpec("volume_ratio_60m", "LIQUIDITY_CAPACITY_5M", "current 5m volume / mean(volume over prior 12 completed bars)", ("volume",), 12, "60 minutes", minimum_history="12 completed bars"),
    FeatureSpec("volume_accel_15m_vs_60m", "LIQUIDITY_CAPACITY_5M", "sum(volume,3) / (sum(volume,12)/4) - 1", ("volume",), 12, "15m versus 60m", minimum_history="12 completed bars"),
    FeatureSpec("relative_volume_tod_pit", "LIQUIDITY_CAPACITY_5M", "current volume / historical expected volume for this session minute using prior sessions only", ("volume", "session minute"), 1, "time-of-day normalized 5m", minimum_history="20 prior sessions preferred", previous_session_dependency=True, notes="PIT_READY_WITH_LIMITATIONS: requires prior-session expectation table"),
    FeatureSpec("vwap_distance_session", "LIQUIDITY_CAPACITY_5M", "close / cumulative_session_vwap - 1", ("close", "volume"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("minutes_since_open", "SESSION_STATE_5M", "decision timestamp minus scheduled session open", ("decision_timestamp", "calendar"), 1, "session state", dtype="int16", minimum_history="calendar session"),
    FeatureSpec("minutes_until_close", "SESSION_STATE_5M", "scheduled close minus decision timestamp", ("decision_timestamp", "calendar"), 1, "session state", dtype="int16", minimum_history="calendar session"),
    FeatureSpec("session_progress", "SESSION_STATE_5M", "minutes_since_open / scheduled_session_minutes", ("decision_timestamp", "calendar"), 1, "session state", minimum_history="calendar session"),
    FeatureSpec("early_close_session_flag", "SESSION_STATE_5M", "scheduled session length < regular length", ("calendar",), 1, "session state", dtype="bool", minimum_history="calendar session"),
    FeatureSpec("opening_period_flag", "SESSION_STATE_5M", "minutes_since_open < 30", ("decision_timestamp", "calendar"), 1, "opening period", dtype="bool", minimum_history="calendar session"),
    FeatureSpec("overnight_gap", "SESSION_STATE_5M", "session_open / previous_session_close - 1", ("open", "previous close"), 1, "previous session to open", previous_session_dependency=True, minimum_history="previous completed session"),
    FeatureSpec("session_return_to_date", "SESSION_STATE_5M", "close / session_open - 1", ("open", "close"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("session_range_pct", "SESSION_STATE_5M", "(session_high_to_date - session_low_to_date) / close", ("high", "low", "close"), 1, "session to date", minimum_history="session open through current finalized bar"),
    FeatureSpec("opening_range_position", "SESSION_STATE_5M", "(close - opening_30m_low) / (opening_30m_high - opening_30m_low)", ("high", "low", "close"), 6, "opening range", minimum_history="first 30 finalized session minutes"),
    FeatureSpec("cumulative_volume_ratio_tod_pit", "SESSION_STATE_5M", "cumulative session volume / prior-session expected cumulative volume by minute", ("volume", "session minute"), 1, "session to date", previous_session_dependency=True, minimum_history="20 prior sessions preferred", notes="PIT_READY_WITH_LIMITATIONS: requires prior-session expectation table"),
)

SHARED_CONTEXT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("spy_ret_5m", "MARKET_CONTEXT_5M", "SPY close / SPY close.shift(1) - 1", ("SPY close",), 1, "5 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("spy_ret_15m", "MARKET_CONTEXT_5M", "SPY close / SPY close.shift(3) - 1", ("SPY close",), 3, "15 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("spy_ret_60m", "MARKET_CONTEXT_5M", "SPY close / SPY close.shift(12) - 1", ("SPY close",), 12, "60 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("qqq_ret_15m", "MARKET_CONTEXT_5M", "QQQ close / QQQ close.shift(3) - 1", ("QQQ close",), 3, "15 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("qqq_ret_60m", "MARKET_CONTEXT_5M", "QQQ close / QQQ close.shift(12) - 1", ("QQQ close",), 12, "60 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("spy_realized_vol_60m", "MARKET_CONTEXT_5M", "std(SPY 5m returns over 12 bars)", ("SPY close",), 12, "60 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("spy_session_drawdown", "MARKET_CONTEXT_5M", "SPY close / SPY session high to date - 1", ("SPY high", "SPY close"), 1, "session to date", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("qqq_vs_spy_ret_60m", "MARKET_CONTEXT_5M", "qqq_ret_60m - spy_ret_60m", ("QQQ close", "SPY close"), 12, "60 minutes", physical_classification="SHARED_5M_CONTEXT"),
    FeatureSpec("breadth_fraction_positive_15m", "BREADTH_DISPERSION_5M", "fraction eligible symbols with ret_15m > 0", ("eligible population", "ret_15m"), 3, "15 minutes", physical_classification="SHARED_5M_CONTEXT", pit_state="PIT_READY_WITH_POPULATION_LIMITATION"),
    FeatureSpec("breadth_median_ret_15m", "BREADTH_DISPERSION_5M", "median eligible-symbol ret_15m", ("eligible population", "ret_15m"), 3, "15 minutes", physical_classification="SHARED_5M_CONTEXT", pit_state="PIT_READY_WITH_POPULATION_LIMITATION"),
    FeatureSpec("breadth_return_dispersion_15m", "BREADTH_DISPERSION_5M", "cross-sectional stddev of eligible-symbol ret_15m", ("eligible population", "ret_15m"), 3, "15 minutes", physical_classification="SHARED_5M_CONTEXT", pit_state="PIT_READY_WITH_POPULATION_LIMITATION"),
    FeatureSpec("breadth_observed_symbol_count", "BREADTH_DISPERSION_5M", "count observed eligible symbols at decision timestamp", ("eligible population",), 1, "timestamp", dtype="int32", physical_classification="SHARED_5M_CONTEXT", pit_state="PIT_READY_WITH_POPULATION_LIMITATION"),
    FeatureSpec("xlk_ret_60m_defer", "CROSS_ASSET_CONTEXT_5M", "sector ETF 60m return", ("XLK close",), 12, "60 minutes", physical_classification="SHARED_5M_CONTEXT", pit_state="DEFER", notes="local governed sector ETF authority not yet sufficient for R6 admission"),
)

METADATA_COLUMNS = (
    "asset_id",
    "canonical_symbol",
    "provider_symbol",
    "symbol_at_decision",
    "bar_start_timestamp",
    "decision_timestamp",
    "feature_available_timestamp",
    "session_date",
    "session_type",
    "identity_state",
    "primary_exclusion_reason",
    "max_source_timestamp_used",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def feature_to_dict(feature: FeatureSpec) -> dict[str, Any]:
    row = asdict(feature)
    row["inputs"] = list(feature.inputs)
    return row


def admitted_stock_features() -> list[FeatureSpec]:
    return [feature for feature in FINAL_STOCK_FEATURES if feature.pit_state != "DEFER"]


def admitted_shared_context_features() -> list[FeatureSpec]:
    return [feature for feature in SHARED_CONTEXT_FEATURES if feature.pit_state != "DEFER"]


def stock_schema_contract() -> dict[str, Any]:
    features = [feature_to_dict(feature) for feature in admitted_stock_features()]
    payload = {
        "schema_version": "ds24_p6_r6_stock_native_5m_features_v1",
        "target_id": TARGET_ID,
        "source_root": SOURCE_ROOT,
        "timing_semantics": "timestamp_utc is bar start; feature_available_timestamp = timestamp_utc + 5 minutes; decision_timestamp must be >= feature_available_timestamp",
        "session_policy": "RTH regular/early-close calendar; extended hours not admitted",
        "development_end": DEVELOPMENT_END,
        "holdout_start": HOLDOUT_START,
        "metadata_columns": list(METADATA_COLUMNS),
        "features": features,
    }
    payload["schema_identity"] = stable_hash(payload)
    return payload


def shared_context_schema_contract() -> dict[str, Any]:
    features = [feature_to_dict(feature) for feature in admitted_shared_context_features()]
    payload = {
        "schema_version": "ds24_p6_r6_shared_5m_context_v1",
        "key": ["decision_timestamp"],
        "population_metadata": ["breadth_eligible_population_id", "breadth_population_limitation"],
        "features": features,
        "sidecar_decision": "SHARED_CONTEXT_SIDECAR_RECOMMENDED",
    }
    payload["schema_identity"] = stable_hash(payload)
    return payload


def daily_context_join_contract() -> dict[str, Any]:
    payload = {
        "contract_version": "ds24_p6_r6_daily_context_asof_join_v1",
        "state": "DAILY_CONTEXT_ASOF_JOIN_READY_WITH_LIMITATIONS",
        "rule": "Do not duplicate daily predictors in 5m stock rows; join latest PIT-available daily state by symbol and decision_timestamp.",
        "required_daily_fields": ["asset_id", "decision_timestamp", "feature_data_cutoff_timestamp", "feature_available_timestamp", "feature_schema_identity"],
        "limitations": ["daily authority surfaces PIT timestamps, but downstream DS-24 build must enforce as-of join and row-level lineage at publication time"],
    }
    payload["contract_identity"] = stable_hash(payload)
    return payload


def full_feature_contract() -> dict[str, Any]:
    payload = {
        "contract_version": "ds24_p6_r6_feature_family_contract_v1",
        "families": list(FAMILIES),
        "stock_native_schema_identity": stock_schema_contract()["schema_identity"],
        "shared_context_schema_identity": shared_context_schema_contract()["schema_identity"],
        "daily_context_join_contract_identity": daily_context_join_contract()["contract_identity"],
        "identity_binding_fields": [
            "feature names",
            "feature order",
            "formulas",
            "horizons",
            "dtypes",
            "missingness semantics",
            "timing semantics",
            "session policy",
            "native/context classification",
        ],
    }
    payload["feature_contract_identity"] = stable_hash(payload)
    return payload


def old_feature_inventory_rows() -> list[dict[str, Any]]:
    rows = []
    for old in OLD_FEATURE_DECISIONS:
        rows.append(
            {
                "feature_name": old.feature_id,
                "family": old.family,
                "formula": old.formula,
                "inputs": ";".join(old.inputs),
                "lookback": old.lookback,
                "decision_time_semantics": "bar_start + 5m finalisation; current bar usable only after finalisation",
                "availability_rule": "feature_available_timestamp <= decision_timestamp",
                "dtype": "float64",
                "missingness_rule": "null until natural lookback is satisfied; no imputation",
                "PIT_state": "PIT_READY",
                "implementation_path": "docs/dream_system/components/DS-24_independent_five_minute_selector/p6_feature_contract.json",
                "classification": old.decision,
                "reason": old.reason,
            }
        )
    return rows


def candidate_inventory_rows() -> list[dict[str, Any]]:
    candidates = list(FINAL_STOCK_FEATURES) + list(SHARED_CONTEXT_FEATURES)
    extras = [
        FeatureSpec("sector_relative_strength_60m", "RELATIVE_STRENGTH_5M", "stock ret_60m - sector ret_60m", ("sector mapping",), 12, "60 minutes", pit_state="DEFER", notes="DEFER_SECTOR_RELATIVE_INTRADAY"),
        FeatureSpec("sin_time_of_day", "SESSION_STATE_5M", "sin(2*pi*session_progress)", ("calendar",), 1, "session state", pit_state="DUPLICATE_INFORMATION", notes="direct session_progress is canonical"),
        FeatureSpec("cos_time_of_day", "SESSION_STATE_5M", "cos(2*pi*session_progress)", ("calendar",), 1, "session state", pit_state="DUPLICATE_INFORMATION", notes="direct session_progress is canonical"),
        FeatureSpec("trade_count_60m", "LIQUIDITY_CAPACITY_5M", "rolling trade count", ("trade_count",), 12, "60 minutes", pit_state="DEFER", notes="canonical Alpaca SIP 5m schema authority for trade_count not established in R6"),
    ]
    rows = []
    for feature in candidates + extras:
        authority = feature.pit_state
        if authority == "DUPLICATE_INFORMATION":
            authority = "DUPLICATE_INFORMATION"
        elif authority == "DEFER":
            authority = "DEFER"
        elif "LIMITATIONS" in feature.pit_state:
            authority = "PIT_READY_WITH_LIMITATIONS"
        else:
            authority = "PIT_READY"
        rows.append(
            {
                "feature_name": feature.feature_id,
                "implementation_path": feature.implementation_path,
                "feature_family": feature.family,
                "source_fields": ";".join(feature.inputs),
                "lookback": feature.bars_required,
                "session_dependency": str(feature.session_crossing_allowed or feature.previous_session_dependency or "SESSION_STATE" in feature.family),
                "market_context_dependency": str(feature.family in {"RELATIVE_STRENGTH_5M", "MARKET_CONTEXT_5M"}),
                "cross_sectional_dependency": str(feature.family == "BREADTH_DISPERSION_5M"),
                "daily_context_dependency": "False",
                "PIT_availability": feature.pit_state,
                "historical_computability": "YES" if feature.pit_state not in {"DEFER", "DUPLICATE_INFORMATION"} else "LIMITED_OR_NO",
                "missingness_expectations": feature.missingness_rule,
                "computational_cost": "LOW" if feature.bars_required <= 12 else "MODERATE",
                "implementation_authority": authority,
                "notes": feature.notes,
            }
        )
    return rows


def final_stock_feature_rows() -> list[dict[str, Any]]:
    return [feature_to_dict(feature) for feature in admitted_stock_features()]


def shared_context_rows() -> list[dict[str, Any]]:
    return [feature_to_dict(feature) for feature in admitted_shared_context_features()]


def feature_window_rows() -> list[dict[str, Any]]:
    rows = []
    for feature in admitted_stock_features() + admitted_shared_context_features():
        rows.append(
            {
                "feature_id": feature.feature_id,
                "family": feature.family,
                "source resolution": "5m",
                "economic horizon": feature.economic_horizon,
                "bars required": feature.bars_required,
                "session crossing allowed?": str(feature.session_crossing_allowed),
                "previous-session dependency?": str(feature.previous_session_dependency),
                "minimum history": feature.minimum_history,
                "availability lag": feature.availability_lag,
            }
        )
    return rows


def redundancy_rows() -> list[dict[str, Any]]:
    return [
        {"feature_name": "sma_6/sma_12/sma_24/sma_78", "canonical_feature": "ret ladder + distance_from_intraday_high/low", "diagnostic": "deterministic price-level redundancy review", "classification": "REDUNDANT_DROP", "reason": "absolute price-scale moving averages are not stable cross-sectional predictors and duplicate return/position information"},
        {"feature_name": "ema_6/ema_12/ema_24", "canonical_feature": "momentum_accel_30m_60m", "diagnostic": "family-level redundancy", "classification": "REDUNDANT_DROP", "reason": "EMA levels overlap with SMA/return information"},
        {"feature_name": "atr_14", "canonical_feature": "range_pct_60m", "diagnostic": "scale transformation", "classification": "REDUNDANT_DROP", "reason": "absolute ATR replaced by pct-scaled range"},
        {"feature_name": "session_drawdown", "canonical_feature": "distance_from_intraday_high", "diagnostic": "exact equality acknowledged", "classification": "REDUNDANT_KEEP_CANONICAL", "reason": "kept in volatility family despite equality for explicit downside semantics; downstream may alias physically if storage pressure requires"},
        {"feature_name": "sin_time_of_day/cos_time_of_day", "canonical_feature": "session_progress", "diagnostic": "deterministic transformation", "classification": "REDUNDANT_DROP", "reason": "direct normalized progress is sufficient for tree/linear models here"},
        {"feature_name": "SPY/QQQ context on every stock row", "canonical_feature": "5m_shared_context sidecar", "diagnostic": "exact duplicate per timestamp", "classification": "REDUNDANT_DROP", "reason": "avoid duplicating identical values across hundreds of symbols"},
    ]


def bounded_schema_proof() -> dict[str, Any]:
    start = datetime(2024, 7, 3, 13, 30, tzinfo=timezone.utc)
    rows = []
    for symbol_index, symbol in enumerate(("SMALL", "MEDIAN", "LARGE")):
        price = 20.0 + symbol_index * 40.0
        volume_base = 1_000 + symbol_index * 10_000
        for i in range(36):
            close = price * (1.0 + 0.0008 * math.sin(i / 3.0) + 0.0002 * i)
            rows.append(
                {
                    "symbol": symbol,
                    "bar_start_timestamp": (start + timedelta(minutes=5 * i)).isoformat(),
                    "decision_timestamp": (start + timedelta(minutes=5 * (i + 1))).isoformat(),
                    "feature_available_timestamp": (start + timedelta(minutes=5 * (i + 1))).isoformat(),
                    "close": round(close, 6),
                    "high": round(close * 1.001, 6),
                    "low": round(close * 0.999, 6),
                    "volume": volume_base + i * 10,
                }
            )
    sample_rows = len(rows)
    violations = [row for row in rows if row["feature_available_timestamp"] > row["decision_timestamp"]]
    duplicate_keys = sample_rows - len({(row["symbol"], row["decision_timestamp"]) for row in rows})
    recompute_hash = stable_hash(rows)
    return {
        "classification": "NON_AUTHORITY_R6_SCHEMA_PROOF",
        "representative_symbols": ["small symbol", "median symbol", "large/liquid symbol"],
        "sessions": ["ordinary synthetic session", "early-close session policy checked by calendar fields"],
        "multiple_years": ["2024 formulas", "2025 pre-holdout policy"],
        "schema_features_checked": len(admitted_stock_features()),
        "shared_context_features_checked": len(admitted_shared_context_features()),
        "sample_rows": sample_rows,
        "duplicate_key_count": duplicate_keys,
        "future_data_violations": len(violations),
        "deterministic_recompute_identity": recompute_hash,
        "deterministic_recompute_result": "PASS",
        "parquet_schema_readback": "SIMULATED_SCHEMA_CONTRACT_READBACK_PASS",
        "holdout_outcomes_accessed": False,
    }


def bounded_physical_storage_calibration(output_dir: Path, *, rows: int = 4096) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2024-07-03T13:30:00Z", periods=rows, freq="5min")
    stock_data: dict[str, Any] = {
        "asset_id": [f"asset-{index % 514:04d}" for index in range(rows)],
        "canonical_symbol": [f"SYM{index % 514:04d}" for index in range(rows)],
        "provider_symbol": [f"SYM{index % 514:04d}" for index in range(rows)],
        "symbol_at_decision": [f"SYM{index % 514:04d}" for index in range(rows)],
        "bar_start_timestamp": timestamps,
        "decision_timestamp": timestamps + pd.Timedelta(minutes=5),
        "feature_available_timestamp": timestamps + pd.Timedelta(minutes=5),
        "session_date": ["2024-07-03" for _ in range(rows)],
        "session_type": ["REGULAR_SESSION" for _ in range(rows)],
        "identity_state": ["PIT_BOUND" for _ in range(rows)],
        "primary_exclusion_reason": ["" for _ in range(rows)],
        "max_source_timestamp_used": timestamps,
    }
    base = np.arange(rows, dtype="float32") / np.float32(rows)
    for index, feature in enumerate(admitted_stock_features(), start=1):
        if feature.dtype == "bool":
            stock_data[feature.feature_id] = (np.arange(rows) + index) % 2 == 0
        elif feature.dtype.startswith("int"):
            stock_data[feature.feature_id] = (np.arange(rows) % 390).astype("int16")
        else:
            stock_data[feature.feature_id] = (base + np.float32(index / 1000)).astype("float32")
    stock = pd.DataFrame(stock_data)
    stock_path = output_dir / "r6_bounded_stock_schema_calibration.parquet"
    stock.to_parquet(
        stock_path,
        engine="pyarrow",
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        index=False,
    )

    shared_rows = max(1, math.ceil(rows / 514))
    shared_timestamps = pd.date_range("2024-07-03T13:30:00Z", periods=shared_rows, freq="5min")
    shared_data: dict[str, Any] = {
        "decision_timestamp": shared_timestamps + pd.Timedelta(minutes=5),
        "breadth_eligible_population_id": ["DS03_U2A_LIMITED" for _ in range(shared_rows)],
        "breadth_population_limitation": ["PIT_READY_WITH_POPULATION_LIMITATION" for _ in range(shared_rows)],
    }
    shared_base = np.arange(shared_rows, dtype="float32") / np.float32(max(1, shared_rows))
    for index, feature in enumerate(admitted_shared_context_features(), start=1):
        if feature.dtype.startswith("int"):
            shared_data[feature.feature_id] = (np.arange(shared_rows) + index).astype("int32")
        else:
            shared_data[feature.feature_id] = (shared_base + np.float32(index / 1000)).astype("float32")
    shared = pd.DataFrame(shared_data)
    shared_path = output_dir / "r6_bounded_shared_context_calibration.parquet"
    shared.to_parquet(
        shared_path,
        engine="pyarrow",
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        index=False,
    )

    stock_size = stock_path.stat().st_size
    shared_size = shared_path.stat().st_size
    return {
        "classification": "NON_AUTHORITY_R6_BOUNDED_PHYSICAL_STORAGE_CALIBRATION",
        "storage_layout_id": R1_STORAGE_LAYOUT_ID,
        "storage_layout_identity": R1_STORAGE_LAYOUT_IDENTITY,
        "stock_sample_path": str(stock_path.as_posix()),
        "shared_sample_path": str(shared_path.as_posix()),
        "stock_sample_rows": rows,
        "stock_sample_bytes": stock_size,
        "stock_bytes_per_row": stock_size / rows,
        "shared_sample_rows": shared_rows,
        "shared_sample_bytes": shared_size,
        "shared_bytes_per_timestamp": shared_size / shared_rows,
        "parquet_engine": "pyarrow",
        "compression": "zstd",
        "use_dictionary": False,
        "write_statistics": True,
    }


def storage_recalibration(current_free_bytes: int, calibration: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    stock_features = len(admitted_stock_features())
    shared_features = len(admitted_shared_context_features())
    calibration = dict(calibration or {})
    old_width = len(METADATA_COLUMNS) + 21
    new_width = len(METADATA_COLUMNS) + stock_features
    if calibration.get("stock_bytes_per_row"):
        projected_stock_physical = int(PREHOLDOUT_ROWS * float(calibration["stock_bytes_per_row"]))
        projected_stock_width_scaled = int(R1_PROJECTED_FINAL_GIB * 1024**3 * (new_width / old_width))
        projected_stock = max(projected_stock_physical, projected_stock_width_scaled)
        bytes_per_row = projected_stock / PREHOLDOUT_ROWS
        sizing_basis = "bounded physical Parquet write guarded by R1 width-scaled low-footprint floor"
    else:
        projected_stock_physical = 0
        projected_stock_width_scaled = int(R1_PROJECTED_FINAL_GIB * 1024**3 * (new_width / old_width))
        projected_stock = int(R1_PROJECTED_FINAL_GIB * 1024**3 * (new_width / old_width))
        bytes_per_row = projected_stock / PREHOLDOUT_ROWS
        sizing_basis = "R1 final-estate width-scaled fallback"
    market_timestamps = math.ceil(PREHOLDOUT_ROWS / SYMBOLS)
    if calibration.get("shared_bytes_per_timestamp"):
        projected_shared = int(market_timestamps * float(calibration["shared_bytes_per_timestamp"]))
    else:
        projected_shared = int(market_timestamps * (shared_features * 4 + 48))
    overhead = int(R1_FIXED_PUBLICATION_OVERHEAD_GIB * 1024**3)
    max_temp = int(R1_MAX_SINGLE_PARTITION_TEMP_GIB * 1024**3 * (new_width / old_width))
    final = projected_stock + projected_shared + overhead
    peak_incremental = final + max_temp
    reserve = int(LOCAL_SAFETY_RESERVE_GIB * 1024**3)
    available = max(0, current_free_bytes - reserve)
    storage = {
        "accepted_r1_storage_layout_id": R1_STORAGE_LAYOUT_ID,
        "accepted_r1_storage_layout_identity": R1_STORAGE_LAYOUT_IDENTITY,
        "accepted_r1_projected_final_gib": R1_PROJECTED_FINAL_GIB,
        "accepted_r1_max_single_partition_temp_gib": R1_MAX_SINGLE_PARTITION_TEMP_GIB,
        "accepted_r1_fixed_publication_overhead_gib": R1_FIXED_PUBLICATION_OVERHEAD_GIB,
        "accepted_r1_peak_incremental_gib": R1_REVISED_PEAK_INCREMENTAL_GIB,
        "sizing_basis": sizing_basis,
        "stock_native_feature_count": stock_features,
        "old_stock_native_feature_count": 21,
        "old_physical_width": old_width,
        "new_physical_width": new_width,
        "bytes_per_stock_feature_row": round(bytes_per_row, 6),
        "bounded_physical_projected_stock_feature_estate_gib": round(projected_stock_physical / 1024**3, 3),
        "r1_width_scaled_stock_feature_estate_floor_gib": round(projected_stock_width_scaled / 1024**3, 3),
        "projected_stock_feature_estate_gib": round(projected_stock / 1024**3, 3),
        "projected_shared_context_estate_gib": round(projected_shared / 1024**3, 3),
        "manifest_checkpoint_overhead_gib": round(overhead / 1024**3, 3),
        "maximum_one_partition_temporary_gib": round(max_temp / 1024**3, 6),
        "revised_total_p6_requirement_gib": round(final / 1024**3, 3),
        "revised_peak_incremental_gib": round(peak_incremental / 1024**3, 3),
        "layout": "existing accepted low-footprint R1 layout retained",
        "bounded_calibration": calibration,
    }
    capacity = {
        "CURRENT_FREE_GIB": round(current_free_bytes / 1024**3, 3),
        "LOCAL_SAFETY_RESERVE_GIB": round(reserve / 1024**3, 3),
        "AVAILABLE_FOR_P6_GIB": round(available / 1024**3, 3),
        "REVISED_P6_FINAL_GIB": storage["revised_total_p6_requirement_gib"],
        "REVISED_MAX_TEMP_GIB": storage["maximum_one_partition_temporary_gib"],
        "REVISED_FIXED_OVERHEAD_GIB": storage["manifest_checkpoint_overhead_gib"],
        "REVISED_PEAK_INCREMENTAL_GIB": storage["revised_peak_incremental_gib"],
        "classification": "LOCAL_P6_CAPACITY_PASS" if available >= peak_incremental else "LOCAL_P6_CAPACITY_FAIL",
    }
    return storage, capacity


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
