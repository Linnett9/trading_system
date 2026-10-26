from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.research.iex_dataset_builders import (
    build_intraday_targets,
    build_daily_targets,
    derive_completed_hour_bars,
    regular_session_bars,
    rolling_log_price_slope,
)
from infrastructure.data.market_sessions import trading_sessions


UTC = timezone.utc


def _bars(count: int, *, start: datetime = datetime(2021, 1, 4, 14, 30, tzinfo=UTC)) -> pd.DataFrame:
    rows = []
    for index in range(count):
        price = 100.0 + index
        rows.append(
            {
                "symbol": "AAPL",
                "timestamp": start + timedelta(minutes=5 * index),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 100 + index,
                "trade_count": 10 + index,
                "vwap": price + 0.25,
            }
        )
    return pd.DataFrame(rows)


def test_regular_session_filter_assigns_explicit_completion_time() -> None:
    raw = pd.concat(
        [
            _bars(2),
            _bars(1, start=datetime(2021, 1, 4, 13, 0, tzinfo=UTC)),
        ],
        ignore_index=True,
    )

    result = regular_session_bars(raw)

    assert len(result) == 2
    assert result["timestamp"].min() == pd.Timestamp("2021-01-04T14:30:00Z")
    assert (result["decision_timestamp"] == result["timestamp"] + pd.Timedelta(minutes=5)).all()


def test_hour_derivation_requires_all_twelve_grid_bars() -> None:
    complete, complete_quality = derive_completed_hour_bars(_bars(12))
    incomplete, incomplete_quality = derive_completed_hour_bars(_bars(11))

    assert len(complete) == 1
    assert complete.iloc[0]["open"] == 100.0
    assert complete.iloc[0]["close"] == 111.5
    assert complete.iloc[0]["available_at"] == pd.Timestamp("2021-01-04T15:30:00Z")
    assert complete.iloc[0]["source_5m_rows"] == 12
    assert complete_quality["complete_hours"] == 1
    assert incomplete.empty
    assert incomplete_quality["incomplete_hours_excluded"] == 1


def test_intraday_targets_use_next_bar_open_and_never_cross_missing_grid() -> None:
    targets = build_intraday_targets(_bars(13))
    first_h30 = targets.loc[targets["target_horizon"].eq("H30")].iloc[0]
    first_h60 = targets.loc[targets["target_horizon"].eq("H60")].iloc[0]

    assert first_h30["target_status"] == "MATURE"
    assert first_h30["entry_timestamp"] == pd.Timestamp("2021-01-04T14:35:00Z")
    assert first_h30["maturity_timestamp"] == pd.Timestamp("2021-01-04T15:05:00Z")
    assert np.isclose(first_h30["target_value"], np.log(106.5 / 101.0))
    assert first_h60["target_status"] == "MATURE"
    assert first_h60["maturity_timestamp"] > first_h60["decision_timestamp"]

    missing = _bars(13).drop(index=1).reset_index(drop=True)
    missing_targets = build_intraday_targets(missing)
    first = missing_targets.loc[missing_targets["target_horizon"].eq("H30")].iloc[0]
    assert first["target_status"] == "UNAVAILABLE_GAP_OR_SESSION_END"
    assert pd.isna(first["target_value"])


def test_rolling_log_price_slope_is_linear_and_has_frozen_warmup() -> None:
    prices = pd.Series(np.exp(np.arange(8, dtype="float64") * 0.02))

    result = rolling_log_price_slope(prices, 5)

    assert result.iloc[:4].isna().all()
    assert np.allclose(result.iloc[4:], 0.02)


def test_daily_targets_use_adjusted_next_session_open_and_outcome_close() -> None:
    session_days = trading_sessions(
        pd.Timestamp("2021-01-04").date(), pd.Timestamp("2021-02-01").date()
    )[:13]
    sessions = pd.DatetimeIndex(session_days).tz_localize("UTC")
    adjusted = pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": sessions + pd.Timedelta(hours=4),
            "open": np.arange(100.0, 113.0),
            "high": np.arange(101.0, 114.0),
            "low": np.arange(99.0, 112.0),
            "close": np.arange(100.5, 113.5),
            "volume": 1000,
            "trade_count": 100,
            "vwap": np.arange(100.25, 113.25),
        }
    )

    targets = build_daily_targets(adjusted)
    first_h5 = targets.loc[targets["target_horizon"].eq("H5")].iloc[0]
    first_h10 = targets.loc[targets["target_horizon"].eq("H10")].iloc[0]

    assert first_h5["target_status"] == "MATURE"
    assert np.isclose(first_h5["target_value"], np.log(105.5 / 101.0))
    assert first_h10["target_status"] == "MATURE"
    assert first_h10["maturity_timestamp"] > first_h10["decision_timestamp"]
