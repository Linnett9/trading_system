"""Pure contracts and validation for the canonical Alpaca IEX bar estate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


RAW_BAR_COLUMNS = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)


@dataclass(frozen=True, slots=True)
class IexPartitionRequest:
    """One independently resumable raw-bar acquisition partition."""

    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    year: int | None = None

    @property
    def partition_id(self) -> str:
        values = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": _utc_iso(self.start),
            "end": _utc_iso(self.end),
            "year": self.year,
            "feed": "iex",
            "adjustment": "raw",
            "currency": "USD",
        }
        digest = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"iex-{self.timeframe}-{self.symbol}-{self.year or 'all'}-{digest}"

    def identity(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": _utc_iso(self.start),
            "end": _utc_iso(self.end),
            "year": self.year,
            "feed": "IEX",
            "adjustment": "raw",
            "currency": "USD",
        }


def plan_iex_partitions(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> tuple[IexPartitionRequest, ...]:
    """Create stable symbol/year 5m partitions or one per-symbol daily partition."""

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("IEX partition bounds must be timezone-aware")
    if start >= end:
        raise ValueError("IEX partition start must precede end")
    normalized_symbols = tuple(sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}))
    if not normalized_symbols:
        raise ValueError("IEX partition plan requires at least one symbol")
    if timeframe == "1Day":
        return tuple(
            IexPartitionRequest(symbol=symbol, timeframe=timeframe, start=start, end=end)
            for symbol in normalized_symbols
        )
    if timeframe != "5m":
        raise ValueError(f"Unsupported canonical IEX timeframe: {timeframe}")
    partitions: list[IexPartitionRequest] = []
    for symbol in normalized_symbols:
        for year in range(start.year, end.year + 1):
            year_start = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
            year_end = min(end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
            if year_start < year_end:
                partitions.append(
                    IexPartitionRequest(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=year_start,
                        end=year_end,
                        year=year,
                    )
                )
    return tuple(partitions)


def canonical_raw_bar_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Project provider rows into the immutable raw-field contract."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="int64"),
                "trade_count": pd.Series(dtype="Int64"),
                "vwap": pd.Series(dtype="float64"),
            }
        )
    missing = sorted(set(RAW_BAR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Provider rows are missing canonical raw columns: {missing}")
    frame = frame.loc[:, RAW_BAR_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise").astype(
        "datetime64[ns, UTC]"
    )
    for column in ("open", "high", "low", "close", "vwap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise").astype("int64")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce").astype("Int64")
    return frame.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def validate_raw_bar_partition(
    frame: pd.DataFrame,
    request: IexPartitionRequest,
) -> dict[str, Any]:
    """Fail closed on identity, chronology, values, duplicates, and grid."""

    if tuple(frame.columns) != RAW_BAR_COLUMNS:
        raise ValueError(f"Raw IEX schema mismatch: {tuple(frame.columns)}")
    if frame.empty:
        return _quality_payload(frame, request)
    if set(frame["symbol"].dropna().astype(str)) != {request.symbol}:
        raise ValueError(f"Partition {request.partition_id} contains a foreign symbol")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if bool((timestamps < request.start).any()) or bool((timestamps >= request.end).any()):
        raise ValueError(f"Partition {request.partition_id} contains a timestamp outside its request bounds")
    if bool(frame.duplicated(["symbol", "timestamp"]).any()):
        raise ValueError(f"Partition {request.partition_id} contains duplicate row identities")
    numeric = frame[["open", "high", "low", "close", "volume"]]
    if bool(numeric.isna().any().any()):
        raise ValueError(f"Partition {request.partition_id} has null required values")
    if bool((frame["volume"] < 0).any()) or bool((frame["trade_count"].dropna() < 0).any()):
        raise ValueError(f"Partition {request.partition_id} has negative activity values")
    if bool((frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any()):
        raise ValueError(f"Partition {request.partition_id} violates high-price invariants")
    if bool((frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError(f"Partition {request.partition_id} violates low-price invariants")
    return _quality_payload(frame, request)


def _quality_payload(frame: pd.DataFrame, request: IexPartitionRequest) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "duplicate_count": 0,
            "off_grid_count": 0,
            "vwap_null_count": 0,
            "trade_count_null_count": 0,
        }
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if request.timeframe == "5m":
        off_grid = timestamps.dt.minute.mod(5).ne(0) | timestamps.dt.second.ne(0) | timestamps.dt.microsecond.ne(0)
    else:
        off_grid = timestamps.dt.minute.ne(0) | timestamps.dt.second.ne(0) | timestamps.dt.microsecond.ne(0)
    return {
        "row_count": int(len(frame)),
        "first_timestamp": timestamps.min().isoformat(),
        "last_timestamp": timestamps.max().isoformat(),
        "duplicate_count": int(frame.duplicated(["symbol", "timestamp"]).sum()),
        "off_grid_count": int(off_grid.sum()),
        "vwap_null_count": int(frame["vwap"].isna().sum()),
        "trade_count_null_count": int(frame["trade_count"].isna().sum()),
    }


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "IexPartitionRequest",
    "RAW_BAR_COLUMNS",
    "canonical_raw_bar_frame",
    "plan_iex_partitions",
    "validate_raw_bar_partition",
]
