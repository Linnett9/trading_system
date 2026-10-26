from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.research.iex_feature_builders import (
    CROSS_SECTIONAL_FEATURES,
    CROSS_TIMEFRAME_FEATURES,
    DAILY_FEATURES,
    FIVE_MINUTE_FEATURES,
    HOURLY_FEATURES,
    MARKET_STATE_FEATURES,
    build_cross_sectional_and_market_state,
    build_cross_timeframe_features,
    build_daily_features,
    build_five_minute_features,
    build_hourly_features,
)
from infrastructure.data.market_sessions import trading_sessions


UTC = timezone.utc


def _intraday(symbol: str, multiplier: float = 1.0) -> pd.DataFrame:
    rows = []
    for index in range(40):
        price = 100.0 * multiplier + index * 0.1
        rows.append(
            {
                "symbol": symbol,
                "timestamp": datetime(2021, 1, 4, 14, 30, tzinfo=UTC) + timedelta(minutes=5 * index),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.05,
                "volume": 1000 + index,
                "trade_count": 100 + index,
                "vwap": price + 0.02,
            }
        )
    return pd.DataFrame(rows)


def _daily(symbol: str, multiplier: float = 1.0) -> pd.DataFrame:
    sessions = trading_sessions(date(2021, 1, 4), date(2021, 5, 28))[:80]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": [datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=4) for day in sessions],
            "open": [100.0 * multiplier + index for index in range(len(sessions))],
            "high": [101.0 * multiplier + index for index in range(len(sessions))],
            "low": [99.0 * multiplier + index for index in range(len(sessions))],
            "close": [100.5 * multiplier + index for index in range(len(sessions))],
            "volume": [10000 + index for index in range(len(sessions))],
            "trade_count": [1000 + index for index in range(len(sessions))],
            "vwap": [100.25 * multiplier + index for index in range(len(sessions))],
        }
    )


def test_five_minute_feature_builder_matches_frozen_surface_and_is_causal() -> None:
    raw = _intraday("AAPL")
    spy = _intraday("SPY", 2.0)
    qqq = _intraday("QQQ", 1.5)

    baseline = build_five_minute_features(raw, raw_spy=spy, raw_qqq=qqq)
    altered = raw.copy()
    altered.loc[altered.index[-1], ["close", "high", "volume"]] = [999.0, 1000.0, 999999]
    rebuilt = build_five_minute_features(altered, raw_spy=spy, raw_qqq=qqq)

    assert tuple(baseline.columns[4:]) == FIVE_MINUTE_FEATURES
    assert not {"open", "high", "low", "close", "volume", "vwap"}.intersection(baseline.columns)
    pd.testing.assert_frame_equal(baseline.iloc[:-1], rebuilt.iloc[:-1])
    assert (baseline["feature_available_at"] > baseline["timestamp"]).all()


def test_daily_and_hourly_feature_surfaces_are_separate_and_timed() -> None:
    daily = build_daily_features(_daily("AAPL"), raw_spy_daily=_daily("SPY", 2.0))
    hourly_raw = pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": pd.date_range("2021-01-04T14:30:00Z", periods=30, freq="h"),
            "available_at": pd.date_range("2021-01-04T15:30:00Z", periods=30, freq="h"),
            "session_date": "2021-01-04",
            "close": np.linspace(100, 110, 30),
        }
    )
    spy = hourly_raw.assign(symbol="SPY", close=np.linspace(200, 220, 30))
    hourly = build_hourly_features(hourly_raw, hourly_spy=spy)

    assert tuple(daily.columns[-len(DAILY_FEATURES) :]) == DAILY_FEATURES
    assert tuple(hourly.columns[-len(HOURLY_FEATURES) :]) == HOURLY_FEATURES
    assert (daily["feature_available_at"] > daily["timestamp"]).all()
    assert (hourly["feature_available_at"] > hourly["timestamp"]).all()


def test_cross_sectional_ranks_use_only_available_same_timestamp_values() -> None:
    aapl = build_five_minute_features(_intraday("AAPL"), raw_spy=_intraday("SPY", 2.0), raw_qqq=_intraday("QQQ", 1.5))
    msft = aapl.copy()
    msft["symbol"] = "MSFT"
    msft["log_return_6b"] = msft["log_return_6b"] * 2
    spy = aapl.copy()
    spy["symbol"] = "SPY"
    combined = pd.concat([aapl, msft, spy], ignore_index=True)

    cross, state = build_cross_sectional_and_market_state(combined)

    assert tuple(cross.columns[-len(CROSS_SECTIONAL_FEATURES) :]) == CROSS_SECTIONAL_FEATURES
    assert tuple(state.columns[-len(MARKET_STATE_FEATURES) :]) == MARKET_STATE_FEATURES
    assert not cross.duplicated(["symbol", "timestamp"]).any()
    assert not state.duplicated(["timestamp"]).any()


def test_cross_timeframe_builder_preserves_frozen_sip01_formula() -> None:
    candidates = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.to_datetime(["2021-01-04T15:00:00Z"] * 2),
            "decision_timestamp": pd.to_datetime(["2021-01-04T15:05:00Z"] * 2),
            "sip01_prior_5_session_intraday_trend": [0.03, 0.01],
            "sip01_current_30m_pullback": [-0.02, 0.01],
            "log_return_6b": [-0.02, 0.01],
            "daily_trend_63d": [0.001, 0.002],
            "hourly_trend_12h": [0.003, -0.001],
            "beta_residual_return_5m": [0.004, 0.005],
        }
    )

    result = build_cross_timeframe_features(candidates)

    assert tuple(result.columns[-len(CROSS_TIMEFRAME_FEATURES) :]) == CROSS_TIMEFRAME_FEATURES
    aapl = result.loc[result["symbol"].eq("AAPL")].iloc[0]
    assert np.isclose(aapl["sip01_trend_minus_pullback"], 1.0 - 0.5)
