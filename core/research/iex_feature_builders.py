"""Pure causal feature builders for the frozen IEX Feature Bank V1."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.research.iex_dataset_builders import regular_session_bars, rolling_log_price_slope
from infrastructure.data.calendar_authority import default_calendar_authority


FIVE_MINUTE_FEATURES = (
    "log_return_1b",
    "log_return_3b",
    "log_return_6b",
    "log_return_12b",
    "log_return_24b",
    "log_return_48b",
    "log_return_78b",
    "session_to_date_log_return",
    "ema_distance_24b",
    "ema_distance_78b",
    "log_price_slope_78b",
    "distance_from_session_open",
    "realized_vol_6b",
    "realized_vol_12b",
    "realized_vol_24b",
    "realized_vol_78b",
    "downside_vol_24b",
    "atr_14b_over_price",
    "session_range_over_price",
    "bollinger_z_24b",
    "drawdown_from_session_high",
    "spy_relative_return_30m",
    "spy_relative_return_60m",
    "qqq_relative_return_30m",
    "qqq_relative_return_60m",
    "rolling_beta_spy_20d",
    "rolling_corr_spy_20d",
    "beta_residual_return_60m",
    "beta_residual_momentum_24b",
    "market_relative_session_return",
    "log_volume",
    "log_trade_count",
    "relative_volume_same_slot_20d",
    "relative_trade_count_same_slot_20d",
    "relative_dollar_volume_same_slot_20d",
    "volume_acceleration_6v24",
    "price_activity_disagreement",
    "close_vwap_deviation",
    "session_vwap_deviation",
)

DAILY_FEATURES = (
    "daily_log_return_1d",
    "daily_log_return_5d",
    "daily_log_return_10d",
    "daily_log_return_20d",
    "daily_log_return_63d",
    "daily_trend_63d",
    "daily_realized_vol_20d",
    "daily_atr14_over_price",
    "daily_drawdown_63d",
    "daily_spy_relative_strength_20d",
)

HOURLY_FEATURES = (
    "hourly_log_return_1h",
    "hourly_log_return_3h",
    "hourly_log_return_6h",
    "hourly_log_return_12h",
    "hourly_log_return_24h",
    "hourly_trend_12h",
    "hourly_realized_vol_12h",
    "hourly_spy_relative_return_6h",
)

CROSS_SECTIONAL_FEATURES = (
    "rank_log_return_6b",
    "rank_log_return_12b",
    "rank_ema_distance_24b",
    "rank_beta_residual_return_60m",
    "rank_realized_vol_24b",
    "rank_relative_volume_same_slot_20d",
    "rank_relative_trade_count_same_slot_20d",
)

MARKET_STATE_FEATURES = (
    "minutes_since_open",
    "minutes_to_close",
    "time_sin",
    "time_cos",
    "opening_window_flag",
    "closing_window_flag",
    "market_breadth_positive_30m",
    "market_return_dispersion_30m",
    "spy_market_state",
)

CROSS_TIMEFRAME_FEATURES = (
    "sip01_prior_5_session_intraday_trend",
    "sip01_current_30m_pullback",
    "sip01_rank_trend",
    "sip01_rank_pullback",
    "sip01_trend_minus_pullback",
    "daily_trend_x_30m_pullback",
    "hourly_trend_x_5m_residual",
)


def build_five_minute_features(
    raw_symbol: pd.DataFrame,
    *,
    raw_spy: pd.DataFrame,
    raw_qqq: pd.DataFrame,
) -> pd.DataFrame:
    """Build all 39 per-symbol 5m predictors from completed RTH bars only."""

    frame = regular_session_bars(raw_symbol)
    if frame.empty:
        return _empty_feature_frame(FIVE_MINUTE_FEATURES)
    symbols = set(frame["symbol"].astype(str))
    if len(symbols) != 1:
        raise ValueError("5m feature builder accepts exactly one canonical symbol")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    log_close = np.log(close)
    for window in (1, 3, 6, 12, 24, 48, 78):
        frame[f"log_return_{window}b"] = log_close - log_close.shift(window)
    sessions = frame.groupby("session_date", sort=False, observed=True)
    session_open = sessions["open"].transform("first")
    session_high = sessions["high"].cummax()
    session_low = sessions["low"].cummin()
    frame["session_to_date_log_return"] = np.log(close / session_open)
    frame["distance_from_session_open"] = close / session_open - 1.0
    for window in (24, 78):
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        frame[f"ema_distance_{window}b"] = close / ema - 1.0
    frame["log_price_slope_78b"] = rolling_log_price_slope(close, 78)
    returns_1 = frame["log_return_1b"]
    for window in (6, 12, 24, 78):
        frame[f"realized_vol_{window}b"] = returns_1.rolling(window, min_periods=window).std(ddof=0)
    downside_square = returns_1.clip(upper=0).pow(2)
    frame["downside_vol_24b"] = downside_square.rolling(24, min_periods=24).mean().pow(0.5)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14b_over_price"] = true_range.rolling(14, min_periods=14).mean() / close
    frame["session_range_over_price"] = (session_high - session_low) / close
    rolling_mean = close.rolling(24, min_periods=24).mean()
    rolling_std = close.rolling(24, min_periods=24).std(ddof=0)
    frame["bollinger_z_24b"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    frame["drawdown_from_session_high"] = close / session_high - 1.0

    spy = _benchmark_intraday(raw_spy, prefix="spy")
    qqq = _benchmark_intraday(raw_qqq, prefix="qqq")
    frame = frame.merge(spy, on="timestamp", how="left", validate="many_to_one")
    frame = frame.merge(qqq, on="timestamp", how="left", validate="many_to_one")
    frame["spy_relative_return_30m"] = frame["log_return_6b"] - frame["spy_log_return_6b"]
    frame["spy_relative_return_60m"] = frame["log_return_12b"] - frame["spy_log_return_12b"]
    frame["qqq_relative_return_30m"] = frame["log_return_6b"] - frame["qqq_log_return_6b"]
    frame["qqq_relative_return_60m"] = frame["log_return_12b"] - frame["qqq_log_return_12b"]

    session_returns = sessions.apply(
        lambda group: float(np.log(group["close"].iloc[-1] / group["open"].iloc[0])),
        include_groups=False,
    ).rename("stock_session_return")
    spy_sessions = _session_returns(raw_spy).rename("spy_session_return")
    paired = pd.concat([session_returns, spy_sessions], axis=1).sort_index()
    prior_stock = paired["stock_session_return"].shift(1)
    prior_spy = paired["spy_session_return"].shift(1)
    covariance = prior_stock.rolling(20, min_periods=20).cov(prior_spy)
    variance = prior_spy.rolling(20, min_periods=20).var()
    paired["rolling_beta_spy_20d"] = covariance / variance.replace(0, np.nan)
    paired["rolling_corr_spy_20d"] = prior_stock.rolling(20, min_periods=20).corr(prior_spy)
    frame["rolling_beta_spy_20d"] = frame["session_date"].map(paired["rolling_beta_spy_20d"])
    frame["rolling_corr_spy_20d"] = frame["session_date"].map(paired["rolling_corr_spy_20d"])
    frame["beta_residual_return_60m"] = (
        frame["log_return_12b"] - frame["rolling_beta_spy_20d"] * frame["spy_log_return_12b"]
    )
    residual_1b = frame["log_return_1b"] - frame["rolling_beta_spy_20d"] * frame["spy_log_return_1b"]
    frame["beta_residual_momentum_24b"] = residual_1b.rolling(24, min_periods=24).sum()
    frame["market_relative_session_return"] = (
        frame["session_to_date_log_return"] - frame["spy_session_to_date_log_return"]
    )

    frame["log_volume"] = np.log1p(frame["volume"])
    frame["log_trade_count"] = np.log1p(frame["trade_count"])
    frame["slot_index"] = (
        (frame["timestamp"] - frame["session_open"]).dt.total_seconds().div(300).astype("int16")
    )
    dollar_volume = close * frame["volume"]
    frame["relative_volume_same_slot_20d"] = _prior_slot_ratio(frame["volume"], frame["slot_index"])
    frame["relative_trade_count_same_slot_20d"] = _prior_slot_ratio(frame["trade_count"], frame["slot_index"])
    frame["relative_dollar_volume_same_slot_20d"] = _prior_slot_ratio(dollar_volume, frame["slot_index"])
    sum_6 = frame["volume"].rolling(6, min_periods=6).sum()
    sum_24 = frame["volume"].rolling(24, min_periods=24).sum()
    frame["volume_acceleration_6v24"] = sum_6 / (sum_24 / 4.0).replace(0, np.nan) - 1.0
    frame["price_activity_disagreement"] = np.sign(frame["log_return_6b"]) * -frame["relative_volume_same_slot_20d"]
    frame["close_vwap_deviation"] = close / frame["vwap"].replace(0, np.nan) - 1.0
    vwap_notional = frame["vwap"] * frame["volume"]
    cumulative_notional = vwap_notional.groupby(frame["session_date"], sort=False).cumsum()
    cumulative_volume = frame["volume"].groupby(frame["session_date"], sort=False).cumsum()
    session_vwap = cumulative_notional / cumulative_volume.replace(0, np.nan)
    frame["session_vwap_deviation"] = close / session_vwap - 1.0
    return _project_feature_frame(frame, FIVE_MINUTE_FEATURES)


def build_daily_features(raw_daily: pd.DataFrame, *, raw_spy_daily: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_daily(raw_daily)
    if frame.empty:
        return _empty_feature_frame(DAILY_FEATURES, include_session_date=True)
    if frame["symbol"].nunique() != 1:
        raise ValueError("Daily feature builder accepts exactly one canonical symbol")
    close = frame["close"]
    log_close = np.log(close)
    for window in (1, 5, 10, 20, 63):
        frame[f"daily_log_return_{window}d"] = log_close - log_close.shift(window)
    frame["daily_trend_63d"] = rolling_log_price_slope(close, 63)
    daily_returns = frame["daily_log_return_1d"]
    frame["daily_realized_vol_20d"] = daily_returns.rolling(20, min_periods=20).std(ddof=0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["daily_atr14_over_price"] = true_range.rolling(14, min_periods=14).mean() / close
    frame["daily_drawdown_63d"] = close / close.rolling(63, min_periods=63).max() - 1.0
    spy = _prepare_daily(raw_spy_daily)
    spy["spy_daily_log_return_20d"] = np.log(spy["close"]) - np.log(spy["close"]).shift(20)
    frame = frame.merge(
        spy[["session_date", "spy_daily_log_return_20d"]],
        on="session_date",
        how="left",
        validate="one_to_one",
    )
    frame["daily_spy_relative_strength_20d"] = (
        frame["daily_log_return_20d"] - frame["spy_daily_log_return_20d"]
    )
    return _project_feature_frame(
        frame,
        DAILY_FEATURES,
        available_column="available_at",
        include_session_date=True,
    )


def build_hourly_features(hourly: pd.DataFrame, *, hourly_spy: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return _empty_feature_frame(HOURLY_FEATURES, available_column="available_at")
    frame = hourly.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    if frame["symbol"].nunique() != 1:
        raise ValueError("Hourly feature builder accepts exactly one canonical symbol")
    log_close = np.log(frame["close"])
    for window in (1, 3, 6, 12, 24):
        frame[f"hourly_log_return_{window}h"] = log_close - log_close.shift(window)
    frame["hourly_trend_12h"] = rolling_log_price_slope(frame["close"], 12)
    frame["hourly_realized_vol_12h"] = frame["hourly_log_return_1h"].rolling(12, min_periods=12).std(ddof=0)
    spy = hourly_spy.sort_values("timestamp", kind="mergesort").copy()
    spy["spy_hourly_log_return_6h"] = np.log(spy["close"]) - np.log(spy["close"]).shift(6)
    frame = frame.merge(
        spy[["timestamp", "spy_hourly_log_return_6h"]], on="timestamp", how="left", validate="many_to_one"
    )
    frame["hourly_spy_relative_return_6h"] = frame["hourly_log_return_6h"] - frame["spy_hourly_log_return_6h"]
    return _project_feature_frame(frame, HOURLY_FEATURES, available_column="available_at", include_session_date=True)


def build_cross_sectional_and_market_state(
    five_minute_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build timestamp-local ranks and one-row-per-timestamp market context."""

    required = {
        "symbol",
        "timestamp",
        "decision_timestamp",
        "log_return_6b",
        "log_return_12b",
        "ema_distance_24b",
        "beta_residual_return_60m",
        "realized_vol_24b",
        "relative_volume_same_slot_20d",
        "relative_trade_count_same_slot_20d",
    }
    missing = sorted(required - set(five_minute_features.columns))
    if missing:
        raise ValueError(f"Cross-sectional input is missing: {missing}")
    frame = five_minute_features.sort_values(["timestamp", "symbol"], kind="mergesort").copy()
    rank_sources = {
        "rank_log_return_6b": "log_return_6b",
        "rank_log_return_12b": "log_return_12b",
        "rank_ema_distance_24b": "ema_distance_24b",
        "rank_beta_residual_return_60m": "beta_residual_return_60m",
        "rank_realized_vol_24b": "realized_vol_24b",
        "rank_relative_volume_same_slot_20d": "relative_volume_same_slot_20d",
        "rank_relative_trade_count_same_slot_20d": "relative_trade_count_same_slot_20d",
    }
    timestamp_groups = frame.groupby("timestamp", sort=False, observed=True)
    for output, source in rank_sources.items():
        frame[output] = timestamp_groups[source].rank(method="average", pct=True)
    cross = frame[["symbol", "timestamp", "decision_timestamp", *CROSS_SECTIONAL_FEATURES]].copy()
    spy = frame.loc[frame["symbol"].eq("SPY"), ["timestamp", "log_return_6b", "ema_distance_24b", "realized_vol_24b"]].copy()
    spy["spy_market_state"] = (
        np.sign(spy["log_return_6b"]).fillna(0).astype("int8")
        + 2 * np.sign(spy["ema_distance_24b"]).fillna(0).astype("int8")
        + 4 * spy["realized_vol_24b"].gt(spy["realized_vol_24b"].rolling(78, min_periods=20).median()).astype("int8")
    )
    context = timestamp_groups["log_return_6b"].agg(
        market_breadth_positive_30m=lambda values: float(values.dropna().gt(0).mean()),
        market_return_dispersion_30m=lambda values: float(values.std(ddof=0)),
    ).reset_index()
    context = context.merge(spy[["timestamp", "spy_market_state"]], on="timestamp", how="left", validate="one_to_one")
    context["decision_timestamp"] = context["timestamp"] + pd.Timedelta(minutes=5)
    timing = _intraday_timing(context["decision_timestamp"])
    context = pd.concat([context.reset_index(drop=True), timing], axis=1)
    market_state = context[["timestamp", "decision_timestamp", *MARKET_STATE_FEATURES]].copy()
    for column in CROSS_SECTIONAL_FEATURES:
        cross[column] = cross[column].astype("float32")
    for column in ("time_sin", "time_cos", "market_breadth_positive_30m", "market_return_dispersion_30m"):
        market_state[column] = market_state[column].astype("float32")
    return cross, market_state


def build_cross_timeframe_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen SIP01 transfer and two bounded context interactions."""

    required = {
        "symbol",
        "timestamp",
        "decision_timestamp",
        "sip01_prior_5_session_intraday_trend",
        "sip01_current_30m_pullback",
        "log_return_6b",
        "daily_trend_63d",
        "hourly_trend_12h",
        "beta_residual_return_5m",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Cross-timeframe input is missing: {missing}")
    frame = candidates.sort_values(["timestamp", "symbol"], kind="mergesort").copy()
    grouped = frame.groupby("timestamp", sort=False, observed=True)
    frame["sip01_rank_trend"] = grouped["sip01_prior_5_session_intraday_trend"].rank(
        method="average", pct=True
    )
    frame["sip01_rank_pullback"] = grouped["sip01_current_30m_pullback"].rank(
        method="average", pct=True
    )
    frame["sip01_trend_minus_pullback"] = frame["sip01_rank_trend"] - frame["sip01_rank_pullback"]
    frame["daily_trend_x_30m_pullback"] = frame["daily_trend_63d"] * -frame["log_return_6b"]
    frame["hourly_trend_x_5m_residual"] = frame["hourly_trend_12h"] * frame["beta_residual_return_5m"]
    for column in CROSS_TIMEFRAME_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame[
        ["symbol", "timestamp", "decision_timestamp", *CROSS_TIMEFRAME_FEATURES]
    ].reset_index(drop=True)


def _benchmark_intraday(raw: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    frame = regular_session_bars(raw).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    log_close = np.log(frame["close"])
    for window in (1, 6, 12):
        frame[f"{prefix}_log_return_{window}b"] = log_close - log_close.shift(window)
    session_open = frame.groupby("session_date", sort=False, observed=True)["open"].transform("first")
    frame[f"{prefix}_session_to_date_log_return"] = np.log(frame["close"] / session_open)
    return frame[
        [
            "timestamp",
            f"{prefix}_log_return_1b",
            f"{prefix}_log_return_6b",
            f"{prefix}_log_return_12b",
            f"{prefix}_session_to_date_log_return",
        ]
    ]


def _session_returns(raw: pd.DataFrame) -> pd.Series:
    frame = regular_session_bars(raw)
    return frame.groupby("session_date", sort=True, observed=True).apply(
        lambda group: float(np.log(group["close"].iloc[-1] / group["open"].iloc[0])),
        include_groups=False,
    )


def _prior_slot_ratio(values: pd.Series, slots: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    medians = numeric.groupby(slots, sort=False).transform(
        lambda group: group.shift(1).rolling(20, min_periods=20).median()
    )
    return numeric / medians.replace(0, np.nan)


def _prepare_daily(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    frame = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    authority = default_calendar_authority()
    available = {
        day: authority.session(pd.Timestamp(day).date()).close_timestamp
        for day in frame["session_date"].unique()
    }
    # A frozen five-minute publication lag prevents an intraday close decision
    # from consuming the same session's daily bar at the exact close instant.
    frame["available_at"] = pd.to_datetime(frame["session_date"].map(available), utc=True) + pd.Timedelta(minutes=5)
    return frame


def _project_feature_frame(
    frame: pd.DataFrame,
    features: Iterable[str],
    *,
    available_column: str = "decision_timestamp",
    include_session_date: bool = False,
) -> pd.DataFrame:
    feature_names = tuple(features)
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    identity = ["symbol", "timestamp"]
    if include_session_date and "session_date" in frame:
        identity.append("session_date")
    if available_column == "decision_timestamp" and available_column in frame:
        frame["feature_available_at"] = frame["decision_timestamp"]
        identity.extend(["decision_timestamp", "feature_available_at"])
    elif available_column in frame:
        frame["feature_available_at"] = frame[available_column]
        identity.append("feature_available_at")
    output = frame[[*identity, *feature_names]].copy()
    return output


def _empty_feature_frame(
    features: Iterable[str],
    *,
    available_column: str = "decision_timestamp",
    include_session_date: bool = False,
) -> pd.DataFrame:
    identity = ["symbol", "timestamp"]
    if include_session_date:
        identity.append("session_date")
    identity.append("feature_available_at")
    return pd.DataFrame(columns=[*identity, *tuple(features)])


def _intraday_timing(decisions: pd.Series) -> pd.DataFrame:
    authority = default_calendar_authority()
    local_dates = decisions.dt.tz_convert("America/New_York").dt.date
    schedule = {
        str(day): authority.session(day)
        for day in sorted(set(local_dates))
    }
    opens = pd.to_datetime([schedule[str(day)].open_timestamp for day in local_dates], utc=True)
    closes = pd.to_datetime([schedule[str(day)].close_timestamp for day in local_dates], utc=True)
    minutes_since = (decisions.reset_index(drop=True) - pd.Series(opens)).dt.total_seconds().div(60)
    minutes_to = (pd.Series(closes) - decisions.reset_index(drop=True)).dt.total_seconds().div(60)
    session_minutes = minutes_since + minutes_to
    phase = 2 * np.pi * minutes_since / session_minutes
    return pd.DataFrame(
        {
            "minutes_since_open": minutes_since.astype("int16"),
            "minutes_to_close": minutes_to.astype("int16"),
            "time_sin": np.sin(phase),
            "time_cos": np.cos(phase),
            "opening_window_flag": minutes_since.lt(30),
            "closing_window_flag": minutes_to.le(30),
        }
    )


__all__ = [
    "CROSS_SECTIONAL_FEATURES",
    "CROSS_TIMEFRAME_FEATURES",
    "DAILY_FEATURES",
    "FIVE_MINUTE_FEATURES",
    "HOURLY_FEATURES",
    "MARKET_STATE_FEATURES",
    "build_cross_sectional_and_market_state",
    "build_cross_timeframe_features",
    "build_daily_features",
    "build_five_minute_features",
    "build_hourly_features",
]
