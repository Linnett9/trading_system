from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.research.ml.ds24.clean_v2_features import PerturbationCase


CERTIFICATION_SYMBOLS = ("AAA", "BBB", "SPY", "QQQ", "GLD", "TLT", "XLK")


def _session_bars(session_date: str, *, early_close: bool) -> pd.DatetimeIndex:
    periods = 42 if early_close else 78
    start = pd.Timestamp(f"{session_date} 14:30:00", tz="UTC")
    return pd.date_range(start, periods=periods, freq="5min")


def synthetic_multiyear_raw_frames() -> dict[str, pd.DataFrame]:
    """Deterministic normal/early-close fixture spanning two calendar years."""

    sessions = (
        ("2017-11-21", False),
        ("2017-11-22", False),
        ("2017-11-24", True),
        ("2024-11-25", False),
        ("2024-11-26", False),
        ("2024-11-29", True),
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol_ordinal, symbol in enumerate(CERTIFICATION_SYMBOLS):
        pieces: list[pd.DataFrame] = []
        running = 0
        for session_date, early_close in sessions:
            timestamps = _session_bars(session_date, early_close=early_close)
            extended_timestamps = timestamps.insert(0, timestamps[0] - pd.Timedelta(minutes=5)).append(
                pd.DatetimeIndex([timestamps[-1] + pd.Timedelta(minutes=5)])
            )
            session_types = ["PRE_MARKET", *(["EARLY_CLOSE" if early_close else "REGULAR"] * len(timestamps)), "AFTER_HOURS"]
            row = np.arange(len(extended_timestamps), dtype="float64") + running
            base = 40.0 + 7.0 * symbol_ordinal + 0.025 * row
            wave = 0.12 * np.sin((row + symbol_ordinal) / 7.0)
            close = base + wave
            pieces.append(
                pd.DataFrame(
                    {
                        "asset_id": symbol,
                        "canonical_symbol": symbol,
                        "provider_symbol": symbol,
                        "timestamp_utc": extended_timestamps,
                        "session_date": session_date,
                        "session_type": session_types,
                        "open": close - 0.02,
                        "high": close + 0.08,
                        "low": close - 0.09,
                        "close": close,
                        "volume": 100_000.0 + 53.0 * row + 1000.0 * symbol_ordinal,
                        "trade_count": 1000 + row.astype("int64") + 10 * symbol_ordinal,
                        "vwap": close - 0.005,
                    }
                )
            )
            running += len(extended_timestamps)
        frames[symbol] = pd.concat(pieces, ignore_index=True)
    return frames


def certification_cases(frames: Mapping[str, pd.DataFrame]) -> list[PerturbationCase]:
    cases: list[PerturbationCase] = []
    selections = (
        ("AAA", "2017-11-22", 0, "2017_normal_open"),
        ("BBB", "2017-11-22", 35, "2017_normal_mid_session"),
        ("AAA", "2017-11-22", 76, "2017_normal_near_close"),
        ("BBB", "2017-11-24", 0, "2017_early_close_open"),
        ("AAA", "2017-11-24", 20, "2017_early_close_mid_session"),
        ("BBB", "2017-11-24", 40, "2017_early_close_near_close"),
        ("AAA", "2024-11-26", 0, "2024_normal_open"),
        ("BBB", "2024-11-26", 35, "2024_normal_mid_session"),
        ("AAA", "2024-11-29", 40, "2024_early_close_near_close"),
    )
    for symbol, session_date, offset, label in selections:
        frame = frames[symbol]
        rows = frame[
            (frame["session_date"] == session_date)
            & (frame["session_type"].isin(["REGULAR", "EARLY_CLOSE"]))
        ].reset_index(drop=True)
        bar_start = pd.Timestamp(rows.loc[offset, "timestamp_utc"])
        cases.append(PerturbationCase(symbol, bar_start + pd.Timedelta(minutes=5), label))
    return cases


def source_fixture_summary(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    sessions = pd.concat(
        [frame[["session_date", "session_type"]] for frame in frames.values()],
        ignore_index=True,
    ).drop_duplicates()
    return {
        "symbols": sorted(frames),
        "years": sorted({int(value[:4]) for value in sessions["session_date"]}),
        "session_types": sorted(sessions["session_type"].unique()),
        "rows": int(sum(len(frame) for frame in frames.values())),
    }
