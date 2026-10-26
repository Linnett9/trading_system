"""Pure contracts for the SIP-to-IEX research-data cutover.

The module intentionally contains no filesystem, provider, target, or model IO.
It centralises the feature taxonomy and the frozen IEX V1 feature proposal used
by the cutover audit so generated evidence has one deterministic source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable


class FeatureFamily(StrEnum):
    RETURNS_MOMENTUM = "returns/momentum"
    REVERSAL = "reversal"
    TREND = "trend"
    RANGE_LOCATION = "range/location"
    VOLATILITY = "volatility"
    DOWNSIDE_RISK = "downside risk/drawdown"
    VOLUME_ACTIVITY = "volume/activity"
    LIQUIDITY = "liquidity proxies"
    VWAP = "VWAP"
    MARKET_RELATIVE = "market-relative"
    BETA_CORRELATION = "beta/correlation"
    CROSS_SECTIONAL_RANKS = "cross-sectional ranks"
    BREADTH_STATE = "breadth/state"
    TIME_OF_DAY = "time-of-day"
    DAILY_CONTEXT = "daily context"
    HOURLY_CONTEXT = "1h context"
    CROSS_TIMEFRAME = "cross-timeframe interactions"
    CLASSIC_TECHNICAL = "classic technical indicators"
    OTHER = "other/unclassified"


class AvailabilityClass(StrEnum):
    CAUSAL = "CAUSAL"
    CAUSAL_WITH_DELAY = "CAUSAL_WITH_DELAY"
    RETROSPECTIVE_ONLY = "RETROSPECTIVE_ONLY"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


@dataclass(frozen=True)
class IexFeatureSpec:
    name: str
    table: str
    family: FeatureFamily
    formula: str
    source_columns: tuple[str, ...]
    lookback: str
    dtype: str = "float32"
    availability_rule: str = "All source bars must be completed by decision_timestamp."
    null_policy: str = "Null until the complete causal lookback is available; never forward-fill."
    causal_test: str = "Perturb every later source value and assert the feature is unchanged."


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_feature(name: str) -> FeatureFamily:
    """Classify a feature name using deterministic, ordered domain rules."""

    value = name.lower()
    if "rank" in value or "percentile" in value:
        return FeatureFamily.CROSS_SECTIONAL_RANKS
    if "breadth" in value or "dispersion" in value or "observed_symbol" in value:
        return FeatureFamily.BREADTH_STATE
    if any(token in value for token in ("minutes_since", "minutes_until", "session_progress", "opening_period", "early_close", "time_sin", "time_cos")):
        return FeatureFamily.TIME_OF_DAY
    if any(token in value for token in ("beta", "correlation", "corr_")):
        return FeatureFamily.BETA_CORRELATION
    if any(token in value for token in ("relative_strength", "relative_return", "residual", "_vs_spy", "spy_relative", "qqq_relative")):
        return FeatureFamily.MARKET_RELATIVE
    if "vwap" in value:
        return FeatureFamily.VWAP
    if any(token in value for token in ("volume", "trade_count", "activity")):
        if any(token in value for token in ("dollar_volume", "liquidity")):
            return FeatureFamily.LIQUIDITY
        return FeatureFamily.VOLUME_ACTIVITY
    if any(token in value for token in ("drawdown", "downside")):
        return FeatureFamily.DOWNSIDE_RISK
    if any(token in value for token in ("vol_", "volatility", "atr", "bollinger_width", "vol_of_vol")):
        return FeatureFamily.VOLATILITY
    if any(token in value for token in ("range", "high_low_position", "distance_from_intraday")):
        return FeatureFamily.RANGE_LOCATION
    if any(token in value for token in ("rsi", "macd", "bollinger", "obv", "pvt", "typical_price")):
        return FeatureFamily.CLASSIC_TECHNICAL
    if any(token in value for token in ("sma", "ema", "slope", "trend")):
        return FeatureFamily.TREND
    if "reversal" in value or "pullback" in value:
        return FeatureFamily.REVERSAL
    if any(token in value for token in ("ret_", "return", "momentum", "overnight_gap")):
        return FeatureFamily.RETURNS_MOMENTUM
    return FeatureFamily.OTHER


def inferred_lookback(name: str) -> str:
    matches = [int(value) for value in re.findall(r"(?<!\d)(\d+)(?:m|h|d)?(?!\d)", name.lower())]
    if not matches:
        return "session/current causal state"
    return str(max(matches))


def old_feature_availability(name: str, *, repaired: bool) -> AvailabilityClass:
    """Project the authenticated V1/V2 PIT evidence onto one old feature."""

    repaired_names = {
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
    }
    if name in repaired_names:
        return AvailabilityClass.CAUSAL if repaired else AvailabilityClass.INVALID
    if name.startswith("breadth_"):
        return AvailabilityClass.AMBIGUOUS
    return AvailabilityClass.CAUSAL_WITH_DELAY


def _feature(
    name: str,
    table: str,
    family: FeatureFamily,
    formula: str,
    sources: Iterable[str],
    lookback: str,
    **overrides: str,
) -> IexFeatureSpec:
    return IexFeatureSpec(
        name=name,
        table=table,
        family=family,
        formula=formula,
        source_columns=tuple(sources),
        lookback=lookback,
        **overrides,
    )


def iex_feature_bank_v1() -> tuple[IexFeatureSpec, ...]:
    """Return the frozen, storage-normalised IEX V1 predictor contract."""

    features: list[IexFeatureSpec] = []
    for bars in (1, 3, 6, 12, 24, 48, 78):
        features.append(_feature(f"log_return_{bars}b", "5m", FeatureFamily.RETURNS_MOMENTUM, f"log(close/close.shift({bars}))", ("close",), f"{bars} completed 5m bars"))
    features.append(_feature("session_to_date_log_return", "5m", FeatureFamily.RETURNS_MOMENTUM, "log(close/session_open)", ("open", "close", "calendar"), "current completed session"))

    for bars in (6, 12, 24, 78):
        features.append(_feature(f"ema_distance_{bars}b", "5m", FeatureFamily.TREND, f"close/EMA(close,{bars})-1", ("close",), f"{bars} completed bars"))
    for bars in (12, 24, 78):
        features.append(_feature(f"log_price_slope_{bars}b", "5m", FeatureFamily.TREND, f"OLS slope of log(close) over {bars} completed bars", ("close",), f"{bars} completed bars"))
    features.append(_feature("distance_from_session_open", "5m", FeatureFamily.TREND, "close/session_open-1", ("open", "close", "calendar"), "current completed session"))

    for bars in (6, 12, 24, 78):
        features.append(_feature(f"realized_vol_{bars}b", "5m", FeatureFamily.VOLATILITY, f"std(log_return_1b,{bars})", ("close",), f"{bars + 1} completed bars"))
    features.extend(
        [
            _feature("downside_vol_24b", "5m", FeatureFamily.DOWNSIDE_RISK, "sqrt(mean(min(log_return_1b,0)^2,24))", ("close",), "25 completed bars"),
            _feature("atr_14b_over_price", "5m", FeatureFamily.VOLATILITY, "ATR(high,low,close,14)/close", ("high", "low", "close"), "15 completed bars"),
            _feature("bar_range_over_price", "5m", FeatureFamily.RANGE_LOCATION, "(high-low)/close", ("high", "low", "close"), "current completed bar"),
            _feature("session_range_over_price", "5m", FeatureFamily.RANGE_LOCATION, "(session_high-session_low)/close", ("high", "low", "close", "calendar"), "current completed session"),
            _feature("vol_of_vol_24b", "5m", FeatureFamily.VOLATILITY, "std(realized_vol_6b,24)", ("close",), "30 completed bars"),
            _feature("bollinger_z_24b", "5m", FeatureFamily.CLASSIC_TECHNICAL, "(close-SMA(close,24))/std(close,24)", ("close",), "24 completed bars"),
            _feature("bollinger_width_24b", "5m", FeatureFamily.VOLATILITY, "4*std(close,24)/SMA(close,24)", ("close",), "24 completed bars"),
            _feature("drawdown_from_session_high", "5m", FeatureFamily.DOWNSIDE_RISK, "close/session_high_to_date-1", ("high", "close", "calendar"), "current completed session"),
        ]
    )

    for benchmark in ("spy", "qqq"):
        for minutes, bars in ((30, 6), (60, 12)):
            features.append(_feature(f"{benchmark}_relative_return_{minutes}m", "5m", FeatureFamily.MARKET_RELATIVE, f"stock_log_return_{bars}b-{benchmark}_log_return_{bars}b", ("close", f"{benchmark}_close"), f"{bars} aligned completed bars"))
    features.extend(
        [
            _feature("rolling_beta_spy_20d", "5m", FeatureFamily.BETA_CORRELATION, "rolling cov(stock,SPY)/var(SPY) using prior completed sessions", ("close", "spy_close", "calendar"), "20 prior completed sessions"),
            _feature("rolling_corr_spy_20d", "5m", FeatureFamily.BETA_CORRELATION, "rolling correlation(stock,SPY) using prior completed sessions", ("close", "spy_close", "calendar"), "20 prior completed sessions"),
            _feature("beta_residual_return_60m", "5m", FeatureFamily.MARKET_RELATIVE, "stock_return_60m-beta_spy*spy_return_60m", ("close", "spy_close", "rolling_beta_spy_20d"), "12 bars plus prior beta"),
            _feature("beta_residual_momentum_24b", "5m", FeatureFamily.MARKET_RELATIVE, "sum(beta-residual 5m returns,24)", ("close", "spy_close", "rolling_beta_spy_20d"), "24 bars plus prior beta"),
            _feature("market_relative_session_return", "5m", FeatureFamily.MARKET_RELATIVE, "stock_session_return-spy_session_return", ("close", "spy_close", "calendar"), "current completed session"),
        ]
    )

    features.extend(
        [
            _feature("log_volume", "5m", FeatureFamily.VOLUME_ACTIVITY, "log1p(volume)", ("volume",), "current completed bar"),
            _feature("log_trade_count", "5m", FeatureFamily.VOLUME_ACTIVITY, "log1p(trade_count)", ("trade_count",), "current completed bar"),
            _feature("relative_volume_same_slot_20d", "5m", FeatureFamily.VOLUME_ACTIVITY, "volume/median(prior 20 sessions same slot volume)", ("volume", "calendar"), "20 prior completed sessions"),
            _feature("relative_trade_count_same_slot_20d", "5m", FeatureFamily.VOLUME_ACTIVITY, "trade_count/median(prior 20 sessions same slot trade_count)", ("trade_count", "calendar"), "20 prior completed sessions"),
            _feature("relative_dollar_volume_same_slot_20d", "5m", FeatureFamily.LIQUIDITY, "close*volume/median(prior 20 sessions same slot close*volume)", ("close", "volume", "calendar"), "20 prior completed sessions"),
            _feature("volume_zscore_24b", "5m", FeatureFamily.VOLUME_ACTIVITY, "zscore(log1p(volume),24)", ("volume",), "24 completed bars"),
            _feature("trade_count_zscore_24b", "5m", FeatureFamily.VOLUME_ACTIVITY, "zscore(log1p(trade_count),24)", ("trade_count",), "24 completed bars"),
            _feature("volume_acceleration_6v24", "5m", FeatureFamily.VOLUME_ACTIVITY, "sum(volume,6)/(sum(volume,24)/4)-1", ("volume",), "24 completed bars"),
            _feature("price_activity_disagreement", "5m", FeatureFamily.VOLUME_ACTIVITY, "sign(log_return_6b)*(-relative_volume_same_slot_20d)", ("close", "volume", "calendar"), "20 prior sessions plus 6 bars"),
        ]
    )

    features.extend(
        [
            _feature("close_vwap_deviation", "5m", FeatureFamily.VWAP, "close/provider_bar_vwap-1", ("close", "vwap"), "current completed bar"),
            _feature("session_vwap_deviation", "5m", FeatureFamily.VWAP, "close/session_cumulative_vwap-1", ("close", "vwap", "volume", "calendar"), "current completed session"),
            _feature("vwap_trend_12b", "5m", FeatureFamily.VWAP, "provider_bar_vwap/provider_bar_vwap.shift(12)-1", ("vwap",), "12 completed bars"),
            _feature("rolling_vwap_deviation_24b", "5m", FeatureFamily.VWAP, "close/volume_weighted_mean(close,24)-1", ("close", "volume"), "24 completed bars"),
        ]
    )

    for base in ("log_return_6b", "log_return_12b", "ema_distance_24b", "beta_residual_return_60m", "realized_vol_24b", "relative_volume_same_slot_20d", "relative_trade_count_same_slot_20d"):
        features.append(_feature(f"rank_{base}", "cross_sectional", FeatureFamily.CROSS_SECTIONAL_RANKS, f"same-timestamp percentile rank of {base} over available symbols", (base, "acquisition_roster"), "same decision timestamp", availability_rule="Rank only values available by decision_timestamp; omit unavailable symbols."))

    features.extend(
        [
            _feature("minutes_since_open", "market_state", FeatureFamily.TIME_OF_DAY, "decision_timestamp-XNYS session open", ("decision_timestamp", "calendar"), "current session", dtype="int16"),
            _feature("minutes_to_close", "market_state", FeatureFamily.TIME_OF_DAY, "XNYS session close-decision_timestamp", ("decision_timestamp", "calendar"), "current session", dtype="int16"),
            _feature("time_sin", "market_state", FeatureFamily.TIME_OF_DAY, "sin(2*pi*minutes_since_open/session_minutes)", ("calendar", "decision_timestamp"), "current session"),
            _feature("time_cos", "market_state", FeatureFamily.TIME_OF_DAY, "cos(2*pi*minutes_since_open/session_minutes)", ("calendar", "decision_timestamp"), "current session"),
            _feature("opening_window_flag", "market_state", FeatureFamily.TIME_OF_DAY, "minutes_since_open<30", ("calendar", "decision_timestamp"), "current session", dtype="bool"),
            _feature("closing_window_flag", "market_state", FeatureFamily.TIME_OF_DAY, "minutes_to_close<=30", ("calendar", "decision_timestamp"), "current session", dtype="bool"),
            _feature("market_breadth_positive_30m", "market_state", FeatureFamily.BREADTH_STATE, "fraction available symbols with log_return_6b>0", ("log_return_6b", "acquisition_roster"), "same decision timestamp"),
            _feature("market_return_dispersion_30m", "market_state", FeatureFamily.BREADTH_STATE, "cross-sectional std(log_return_6b)", ("log_return_6b", "acquisition_roster"), "same decision timestamp"),
            _feature("spy_market_state", "market_state", FeatureFamily.BREADTH_STATE, "categorical sign/trend/volatility state from completed SPY bars", ("spy_close",), "78 completed bars", dtype="int8"),
        ]
    )

    for days in (1, 5, 10, 20, 63):
        features.append(_feature(f"daily_log_return_{days}d", "1d", FeatureFamily.DAILY_CONTEXT, f"log(close/close.shift({days}))", ("close",), f"{days} completed daily bars", availability_rule="Available only after the source daily close plus the frozen publication lag."))
    features.extend(
        [
            _feature("daily_trend_63d", "1d", FeatureFamily.DAILY_CONTEXT, "log-price OLS slope over 63 completed daily bars", ("close",), "63 completed daily bars"),
            _feature("daily_realized_vol_20d", "1d", FeatureFamily.DAILY_CONTEXT, "std(daily log return,20)", ("close",), "21 completed daily bars"),
            _feature("daily_atr14_over_price", "1d", FeatureFamily.DAILY_CONTEXT, "ATR(14)/close", ("high", "low", "close"), "15 completed daily bars"),
            _feature("daily_drawdown_63d", "1d", FeatureFamily.DAILY_CONTEXT, "close/rolling_max(close,63)-1", ("close",), "63 completed daily bars"),
            _feature("daily_spy_relative_strength_20d", "1d", FeatureFamily.DAILY_CONTEXT, "stock_return_20d-spy_return_20d", ("close", "spy_close"), "20 aligned completed daily bars"),
        ]
    )

    for hours in (1, 3, 6, 12, 24):
        features.append(_feature(f"hourly_log_return_{hours}h", "1h", FeatureFamily.HOURLY_CONTEXT, f"log(close/close.shift({hours}))", ("close",), f"{hours} completed derived hours", availability_rule="Only fully completed derived hours may be joined as-of to a 5m decision."))
    features.extend(
        [
            _feature("hourly_trend_12h", "1h", FeatureFamily.HOURLY_CONTEXT, "log-price OLS slope over 12 completed hours", ("close",), "12 completed hours"),
            _feature("hourly_realized_vol_12h", "1h", FeatureFamily.HOURLY_CONTEXT, "std(hourly log return,12)", ("close",), "13 completed hours"),
            _feature("hourly_spy_relative_return_6h", "1h", FeatureFamily.HOURLY_CONTEXT, "stock_return_6h-spy_return_6h", ("close", "spy_close"), "6 aligned completed hours"),
        ]
    )

    features.extend(
        [
            _feature("sip01_prior_5_session_intraday_trend", "cross_timeframe", FeatureFamily.CROSS_TIMEFRAME, "sum(log(session_close/session_open), prior 5 completed sessions)", ("open", "close", "calendar"), "5 prior completed sessions"),
            _feature("sip01_current_30m_pullback", "cross_timeframe", FeatureFamily.CROSS_TIMEFRAME, "log(close/close.shift(6))", ("close",), "6 completed 5m bars"),
            _feature("sip01_rank_trend", "cross_timeframe", FeatureFamily.CROSS_SECTIONAL_RANKS, "same-timestamp percentile rank of sip01_prior_5_session_intraday_trend", ("sip01_prior_5_session_intraday_trend",), "same decision timestamp"),
            _feature("sip01_rank_pullback", "cross_timeframe", FeatureFamily.CROSS_SECTIONAL_RANKS, "same-timestamp percentile rank of sip01_current_30m_pullback", ("sip01_current_30m_pullback",), "same decision timestamp"),
            _feature("sip01_trend_minus_pullback", "cross_timeframe", FeatureFamily.CROSS_TIMEFRAME, "sip01_rank_trend-sip01_rank_pullback", ("sip01_rank_trend", "sip01_rank_pullback"), "same decision timestamp"),
            _feature("daily_trend_x_30m_pullback", "cross_timeframe", FeatureFamily.CROSS_TIMEFRAME, "daily_trend_63d*(-log_return_6b)", ("daily_trend_63d", "log_return_6b"), "as-of daily state plus 6 bars"),
            _feature("hourly_trend_x_5m_residual", "cross_timeframe", FeatureFamily.CROSS_TIMEFRAME, "hourly_trend_12h*beta_residual_return_5m", ("hourly_trend_12h", "close", "spy_close"), "as-of completed hour plus current completed bar"),
        ]
    )
    # Keep V1 inside the ticket's compact 60--80 predictor envelope.  These
    # candidates are intentionally deferred because the retained feature is a
    # closer, less redundant expression of the same information family.
    deferred_redundant = {
        "ema_distance_6b",
        "ema_distance_12b",
        "log_price_slope_12b",
        "log_price_slope_24b",
        "bar_range_over_price",
        "vol_of_vol_24b",
        "bollinger_width_24b",
        "volume_zscore_24b",
        "trade_count_zscore_24b",
        "vwap_trend_12b",
        "rolling_vwap_deviation_24b",
    }
    admitted = tuple(item for item in features if item.name not in deferred_redundant)
    if len(admitted) != 80 or len({item.name for item in admitted}) != len(admitted):
        raise AssertionError("IEX V1 must contain exactly 80 uniquely named predictors")
    return admitted


def iex_feature_bank_payload(*, evidence: dict[str, Any]) -> dict[str, Any]:
    specs = [
        {**asdict(item), "family": item.family.value, "source_columns": list(item.source_columns)}
        for item in iex_feature_bank_v1()
    ]
    payload: dict[str, Any] = {
        "schema_version": "iex_feature_bank_v1_spec.v1",
        "status": "FROZEN_BEFORE_IEX_TARGET_OUTCOME_ANALYSIS",
        "predictor_count": len(specs),
        "storage_contract": {
            "format": "Parquet",
            "compression": "ZSTD",
            "raw_ohlcv_duplicated_in_features": False,
            "daily_and_hourly_context_repeated_in_5m_rows": False,
            "targets_separate": True,
        },
        "sip01_formula_frozen": "pct_rank(sum(previous 5 completed session open-to-close log returns))-pct_rank(current 30m return)",
        "evidence": evidence,
        "features": specs,
    }
    payload["spec_sha256"] = canonical_sha256(payload)
    return payload
