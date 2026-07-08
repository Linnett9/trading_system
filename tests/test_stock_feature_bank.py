from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.stock_feature_bank import (
    build_stock_feature_bank,
    feature_columns,
    inspect_bar_file,
    summarize_market_inventory,
    validate_stock_feature_bank,
)


def test_daily_features_are_backward_looking_and_future_append_invariant(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(280)
    _write_bars(root, "SPY", "1Day", dates, start=200.0, step=0.2)
    _write_bars(root, "QQQ", "1Day", dates, start=250.0, step=0.3)
    _write_bars(root, "AAA", "1Day", dates[:260], start=100.0, step=1.0)

    first = build_stock_feature_bank("1Day", data_root=root, output_dir=tmp_path / "features", symbols=["AAA", "SPY", "QQQ"])
    row_before = _row(first.path, "AAA", dates[249])
    _write_bars(root, "AAA", "1Day", dates, start=100.0, step=1.0)
    second = build_stock_feature_bank("1Day", data_root=root, output_dir=tmp_path / "features", symbols=["AAA", "SPY", "QQQ"])
    row_after = _row(second.path, "AAA", dates[249])

    for name in ("return_20", "sma_200", "rsi_14", "atr_14", "realized_volatility_63", "relative_strength_spy_63"):
        assert row_after[name] == pytest.approx(row_before[name])


def test_duplicate_key_rejection_and_deterministic_sorting_are_reported(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(80)
    unsorted = list(reversed(dates))
    _write_bars(root, "SPY", "1h", dates, start=200.0, step=0.2)
    _write_bars(root, "QQQ", "1h", dates, start=250.0, step=0.3)
    _write_bars(root, "BBB", "1h", unsorted, start=100.0, step=1.0)

    result = build_stock_feature_bank("1h", data_root=root, output_dir=tmp_path / "features", symbols=["BBB", "SPY", "QQQ"])
    table = pq.read_table(result.path, columns=["timestamp", "symbol", "timeframe"])
    rows = table.to_pylist()

    assert result.duplicate_key_count == 0
    assert rows == sorted(rows, key=lambda row: (row["timestamp"], row["symbol"]))


def test_parquet_round_trip_schema_stability_and_no_infinite_values(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(90)
    _write_bars(root, "CCC", "5m", dates, start=10.0, step=0.1)

    result = build_stock_feature_bank("5m", data_root=root, output_dir=tmp_path / "features", symbols=["CCC"])
    validated = validate_stock_feature_bank(result.path)
    table = pq.read_table(result.path)

    assert table.column_names == list(feature_columns("5m"))
    assert validated.infinite_value_count == 0
    assert validated.column_count == len(feature_columns("5m"))


def test_spy_qqq_alignment_missing_benchmark_rows_and_nan_warmup(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(260)
    _write_bars(root, "SPY", "1Day", dates[10:], start=200.0, step=0.2)
    _write_bars(root, "QQQ", "1Day", dates[20:], start=250.0, step=0.3)
    _write_bars(root, "DDD", "1Day", dates, start=100.0, step=0.5)

    result = build_stock_feature_bank("1Day", data_root=root, output_dir=tmp_path / "features", symbols=["DDD", "SPY", "QQQ"])
    early = _row(result.path, "DDD", dates[5])
    late = _row(result.path, "DDD", dates[-1])

    assert early["spy_return_20"] is None
    assert early["sma_200"] is None
    assert late["spy_return_20"] is not None
    assert late["qqq_return_20"] is not None


def test_safe_division_returns_null_without_runtime_warnings(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(40)
    flat = [100.0 for _ in dates]
    _write_bars(root, "SPY", "1Day", dates, closes=flat)
    _write_bars(root, "QQQ", "1Day", dates, closes=flat)
    _write_bars(root, "EEE", "1Day", dates, closes=flat, volumes=[0.0 for _ in dates])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_stock_feature_bank("1Day", data_root=root, output_dir=tmp_path / "features", symbols=["EEE", "SPY", "QQQ"])

    row = _row(result.path, "EEE", dates[-1])
    assert not caught
    assert row["volume_ratio_20"] is None
    assert row["bollinger_zscore_20"] is None


def test_fast_inventory_marks_duplicate_and_sort_checks_as_assumptions(tmp_path: Path):
    root = tmp_path / "processed"
    dates = _dates(5)
    _write_bars(root, "FFF", "1Day", dates, start=10.0, step=1.0)

    row = inspect_bar_file(root / "FFF" / "1Day" / "bars.parquet")
    summary = summarize_market_inventory([row])

    assert row["duplicate_check_status"] == "not_scanned_fast_inventory_assumed_from_canonical_import"
    assert row["sorted_check_status"] == "not_scanned_fast_inventory_assumed_from_canonical_import"
    assert summary["timeframes"]["1Day"]["duplicate_check_status"] == row["duplicate_check_status"]
    assert summary["timeframes"]["1Day"]["sorted_check_status"] == row["sorted_check_status"]


def _dates(count: int):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(days=index) for index in range(count)]


def _write_bars(
    root: Path,
    symbol: str,
    timeframe: str,
    timestamps,
    *,
    start: float = 10.0,
    step: float = 1.0,
    closes=None,
    volumes=None,
) -> None:
    path = root / symbol / timeframe / "bars.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    ordered = list(timestamps)
    for index, timestamp in enumerate(ordered):
        close = closes[index] if closes is not None else start + index * step
        rows.append(
            {
                "timestamp": timestamp,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": float(volumes[index] if volumes is not None else 1_000 + index),
                "symbol": symbol,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(path: str, symbol: str, timestamp: datetime):
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row["symbol"] == symbol and row["timestamp"] == timestamp:
            return row
    raise AssertionError(f"Missing row {symbol} {timestamp}")
