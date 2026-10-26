"""Deterministic scientific builders for the canonical IEX research estate.

This module is deliberately filesystem- and provider-independent. It accepts
already-authenticated bar frames and returns causal derived bars, predictors,
or targets with explicit availability timestamps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from infrastructure.data.calendar_authority import default_calendar_authority


FIVE_MINUTES = pd.Timedelta(minutes=5)
ONE_HOUR = pd.Timedelta(hours=1)


def regular_session_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Return provider bars whose full interval lies inside an XNYS session."""

    if raw.empty:
        return raw.assign(
            session_date=pd.Series(dtype="string"),
            session_open=pd.Series(dtype="datetime64[ns, UTC]"),
            session_close=pd.Series(dtype="datetime64[ns, UTC]"),
            decision_timestamp=pd.Series(dtype="datetime64[ns, UTC]"),
        )
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    local_dates = frame["timestamp"].dt.tz_convert("America/New_York").dt.date
    schedule = _schedule_frame(local_dates.unique())
    frame["session_date"] = local_dates.astype(str)
    frame = frame.merge(schedule, on="session_date", how="left", validate="many_to_one")
    frame["decision_timestamp"] = frame["timestamp"] + FIVE_MINUTES
    eligible = (
        frame["session_open"].notna()
        & frame["timestamp"].ge(frame["session_open"])
        & frame["decision_timestamp"].le(frame["session_close"])
    )
    return frame.loc[eligible].sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def derive_completed_hour_bars(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate only complete 12-bar, session-anchored hours from 5m bars."""

    rth = regular_session_bars(raw)
    columns = [
        "symbol",
        "timestamp",
        "available_at",
        "session_date",
        "hour_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "source_5m_rows",
    ]
    if rth.empty:
        return pd.DataFrame(columns=columns), {
            "input_rows": int(len(raw)),
            "rth_rows": 0,
            "candidate_hours": 0,
            "complete_hours": 0,
            "incomplete_hours_excluded": 0,
        }
    elapsed_minutes = (rth["timestamp"] - rth["session_open"]).dt.total_seconds().div(60).astype("int32")
    rth["hour_index"] = elapsed_minutes.floordiv(60).astype("int8")
    rth["hour_start"] = rth["session_open"] + pd.to_timedelta(rth["hour_index"].astype(int), unit="h")
    rth["hour_end"] = rth["hour_start"] + ONE_HOUR
    rth = rth.loc[rth["hour_end"].le(rth["session_close"])].copy()
    group_columns = ["symbol", "session_date", "hour_index", "hour_start", "hour_end"]
    grouped = rth.groupby(group_columns, sort=True, observed=True)
    counts = grouped.size().rename("source_5m_rows")
    unique_counts = grouped["timestamp"].nunique().rename("unique_5m_rows")
    first_timestamp = grouped["timestamp"].min().rename("first_5m_timestamp")
    last_timestamp = grouped["timestamp"].max().rename("last_5m_timestamp")
    eligibility = pd.concat([counts, unique_counts, first_timestamp, last_timestamp], axis=1).reset_index()
    complete_keys = eligibility.loc[
        eligibility["source_5m_rows"].eq(12)
        & eligibility["unique_5m_rows"].eq(12)
        & eligibility["first_5m_timestamp"].eq(eligibility["hour_start"])
        & eligibility["last_5m_timestamp"].eq(eligibility["hour_end"] - FIVE_MINUTES),
        group_columns,
    ]
    complete = rth.merge(complete_keys, on=group_columns, how="inner", validate="many_to_one")
    if complete.empty:
        output = pd.DataFrame(columns=columns)
    else:
        complete["vwap_notional"] = complete["vwap"] * complete["volume"]
        aggregations = complete.groupby(group_columns, sort=True, observed=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", lambda values: values.sum(min_count=1)),
            vwap_notional=("vwap_notional", lambda values: values.sum(min_count=1)),
            source_5m_rows=("timestamp", "size"),
        ).reset_index()
        aggregations["vwap"] = aggregations["vwap_notional"].div(
            aggregations["volume"].where(aggregations["volume"].ne(0))
        )
        output = aggregations.rename(columns={"hour_start": "timestamp", "hour_end": "available_at"})
        output = output.loc[:, columns].sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    return output, {
        "input_rows": int(len(raw)),
        "rth_rows": int(len(rth)),
        "candidate_hours": int(len(eligibility)),
        "complete_hours": int(len(output)),
        "incomplete_hours_excluded": int(len(eligibility) - len(output)),
        "aggregation": {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trade_count": "sum_min_count_1",
            "vwap": "volume_weighted_provider_vwap",
        },
    }


def build_intraday_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Build conservative next-bar-open H30/H60 targets without crossing sessions."""

    rth = regular_session_bars(raw)
    outputs: list[pd.DataFrame] = []
    for (_, _), session in rth.groupby(["symbol", "session_date"], sort=True, observed=True):
        ordered = session.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        for bars, horizon in ((6, "H30"), (12, "H60")):
            entry_timestamp = ordered["timestamp"].shift(-1)
            entry_price = ordered["open"].shift(-1)
            outcome_index_shift = -bars
            outcome_bar_timestamp = ordered["timestamp"].shift(outcome_index_shift)
            outcome_price = ordered["close"].shift(outcome_index_shift)
            maturity = ordered["decision_timestamp"] + pd.to_timedelta(bars * 5, unit="m")
            expected_entry = ordered["decision_timestamp"]
            expected_outcome_bar = ordered["timestamp"] + pd.to_timedelta(bars * 5, unit="m")
            mature = entry_timestamp.eq(expected_entry) & outcome_bar_timestamp.eq(expected_outcome_bar)
            target = np.log(outcome_price / entry_price).where(mature)
            status = np.where(mature & target.notna(), "MATURE", "UNAVAILABLE_GAP_OR_SESSION_END")
            outputs.append(
                pd.DataFrame(
                    {
                        "symbol": ordered["symbol"],
                        "decision_timestamp": ordered["decision_timestamp"],
                        "feature_ready_timestamp": ordered["decision_timestamp"],
                        "entry_timestamp": entry_timestamp.where(mature),
                        "outcome_timestamp": maturity.where(mature),
                        "maturity_timestamp": maturity.where(mature),
                        "target_horizon": horizon,
                        "target_status": status,
                        "target_value": target.astype("float64"),
                    }
                )
            )
    if not outputs:
        return pd.DataFrame(
            columns=[
                "symbol",
                "decision_timestamp",
                "feature_ready_timestamp",
                "entry_timestamp",
                "outcome_timestamp",
                "maturity_timestamp",
                "target_horizon",
                "target_status",
                "target_value",
            ]
        )
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["symbol", "decision_timestamp", "target_horizon"], kind="mergesort"
    ).reset_index(drop=True)


def build_daily_targets(adjusted_daily: pd.DataFrame) -> pd.DataFrame:
    """Build adjusted next-session-open H5/H10 targets with explicit maturity."""

    if adjusted_daily.empty:
        return _empty_target_frame()
    frame = adjusted_daily.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    authority = default_calendar_authority()
    schedule = {
        day: authority.session(pd.Timestamp(day).date())
        for day in frame["session_date"].unique()
    }
    frame["decision_timestamp"] = pd.to_datetime(
        frame["session_date"].map({day: record.close_timestamp for day, record in schedule.items()}),
        utc=True,
    ) + pd.Timedelta(minutes=5)
    outputs: list[pd.DataFrame] = []
    for _, symbol_rows in frame.groupby("symbol", sort=True, observed=True):
        ordered = symbol_rows.reset_index(drop=True)
        for sessions, horizon in ((5, "H5"), (10, "H10")):
            entry_open = ordered["open"].shift(-1)
            entry_date = ordered["session_date"].shift(-1)
            outcome_close = ordered["close"].shift(-sessions)
            outcome_date = ordered["session_date"].shift(-sessions)
            entry_timestamp = pd.to_datetime(
                entry_date.map(
                    lambda day: authority.session(pd.Timestamp(day).date()).open_timestamp if pd.notna(day) else None
                ),
                utc=True,
            )
            maturity_timestamp = pd.to_datetime(
                outcome_date.map(
                    lambda day: authority.session(pd.Timestamp(day).date()).close_timestamp if pd.notna(day) else None
                ),
                utc=True,
            )
            mature = entry_open.notna() & outcome_close.notna() & maturity_timestamp.gt(ordered["decision_timestamp"])
            target = np.log(outcome_close / entry_open).where(mature)
            outputs.append(
                pd.DataFrame(
                    {
                        "symbol": ordered["symbol"],
                        "decision_timestamp": ordered["decision_timestamp"],
                        "feature_ready_timestamp": ordered["decision_timestamp"],
                        "entry_timestamp": entry_timestamp.where(mature),
                        "outcome_timestamp": maturity_timestamp.where(mature),
                        "maturity_timestamp": maturity_timestamp.where(mature),
                        "target_horizon": horizon,
                        "target_status": np.where(
                            mature & target.notna(), "MATURE", "UNAVAILABLE_LISTING_OR_HISTORY_END"
                        ),
                        "target_value": target.astype("float64"),
                    }
                )
            )
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["symbol", "decision_timestamp", "target_horizon"], kind="mergesort"
    ).reset_index(drop=True)


def rolling_log_price_slope(values: pd.Series, window: int) -> pd.Series:
    """Compute a rolling OLS slope in O(n) with complete-window semantics."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    output = np.full(len(numeric), np.nan, dtype="float64")
    if len(numeric) < window:
        return pd.Series(output, index=values.index)
    y = np.log(numeric)
    indices = np.arange(len(y), dtype="float64")
    finite = np.isfinite(y)
    y_clean = np.where(finite, y, 0.0)
    cumulative_y = np.concatenate(([0.0], np.cumsum(y_clean)))
    cumulative_ky = np.concatenate(([0.0], np.cumsum(indices * y_clean)))
    cumulative_valid = np.concatenate(([0], np.cumsum(finite.astype("int64"))))
    starts = np.arange(0, len(y) - window + 1, dtype="int64")
    ends = starts + window
    sum_y = cumulative_y[ends] - cumulative_y[starts]
    sum_ky = cumulative_ky[ends] - cumulative_ky[starts]
    weighted = sum_ky - starts * sum_y
    sum_x = window * (window - 1) / 2.0
    sum_x2 = window * (window - 1) * (2 * window - 1) / 6.0
    denominator = window * sum_x2 - sum_x**2
    slopes = (window * weighted - sum_x * sum_y) / denominator
    valid = (cumulative_valid[ends] - cumulative_valid[starts]) == window
    output[window - 1 :] = np.where(valid, slopes, np.nan)
    return pd.Series(output, index=values.index)


def _schedule_frame(days: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    authority = default_calendar_authority()
    for raw_day in sorted(set(days)):
        record = authority.session(raw_day)
        rows.append(
            {
                "session_date": str(raw_day),
                "session_open": record.open_timestamp,
                "session_close": record.close_timestamp,
                "calendar_authority_version": record.authority_version_identity,
                "early_close": bool(record.early_close),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["session_open"] = pd.to_datetime(frame["session_open"], utc=True).astype("datetime64[ns, UTC]")
        frame["session_close"] = pd.to_datetime(frame["session_close"], utc=True).astype("datetime64[ns, UTC]")
    return frame


def _empty_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "decision_timestamp",
            "feature_ready_timestamp",
            "entry_timestamp",
            "outcome_timestamp",
            "maturity_timestamp",
            "target_horizon",
            "target_status",
            "target_value",
        ]
    )


__all__ = [
    "build_intraday_targets",
    "build_daily_targets",
    "derive_completed_hour_bars",
    "regular_session_bars",
    "rolling_log_price_slope",
]
