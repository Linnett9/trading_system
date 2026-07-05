from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from application.services.market_parquet_commands import run_market_parquet_import
from core.entities.candle import Candle
from core.research.ml.multi_timeframe import add_intraday_summary_features
from infrastructure.data.market_parquet import (
    MarketParquetDataFeed,
    MarketParquetImporter,
    assert_market_parquet_output_path,
    market_parquet_path,
    migrate_legacy_daily_parquet,
    normalize_timeframe_source_folder,
)


def test_market_parquet_import_writes_structured_utc_schema_and_feed_reads_intraday(tmp_path):
    raw_dir = tmp_path / "downloads" / "5m"
    raw_dir.mkdir(parents=True)
    (raw_dir / "AAPL_5m.csv").write_text(
        "timestamp,open,high,low,close,volume,symbol\n"
        "2024-01-02T14:30:00-05:00,10,12,9,11,100,AAPL\n"
        "2024-01-02T14:35:00-05:00,11,13,10,12,200,AAPL\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "processed"

    result = MarketParquetImporter(raw_dir, output_root).import_timeframe("5m")[0]
    feed = MarketParquetDataFeed(output_root)
    candles = feed.get_historical_bars(
        "AAPL",
        "5m",
        datetime(2024, 1, 2, 19, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 20, 0, tzinfo=timezone.utc),
    )

    assert result.output_path == str(market_parquet_path(output_root, "AAPL", "5m"))
    assert candles[0].timestamp == datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert feed.get_last_request_metadata("AAPL")["timestamp_semantics"] == "bar_close"


def test_market_parquet_import_keeps_timeframes_in_separate_output_paths(tmp_path):
    five_min_raw = tmp_path / "downloads" / "5_us_txt" / "data"
    hourly_raw = tmp_path / "downloads" / "h_us_txt" / "data"
    five_min_raw.mkdir(parents=True)
    hourly_raw.mkdir(parents=True)
    five_min_raw.joinpath("AAPL_5m.csv").write_text(
        "timestamp,open,high,low,close,volume,symbol\n"
        "2024-01-02T14:30:00Z,10,12,9,11,100,AAPL\n"
        "2024-01-02T14:35:00Z,11,13,10,12,200,AAPL\n",
        encoding="utf-8",
    )
    hourly_raw.joinpath("AAPL_1h.csv").write_text(
        "timestamp,open,high,low,close,volume,symbol\n"
        "2024-01-02T15:00:00Z,10,12,9,11,100,AAPL\n"
        "2024-01-02T16:00:00Z,11,13,10,12,200,AAPL\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "processed"

    five_min_result = MarketParquetImporter(five_min_raw, output_root).import_timeframe("5m")[0]
    hourly_result = MarketParquetImporter(hourly_raw, output_root).import_timeframe("1h")[0]

    five_min_path = market_parquet_path(output_root, "AAPL", "5m")
    hourly_path = market_parquet_path(output_root, "AAPL", "1h")
    assert five_min_path.exists()
    assert hourly_path.exists()
    assert five_min_path != hourly_path
    assert five_min_result.output_path == str(five_min_path)
    assert hourly_result.output_path == str(hourly_path)


def test_market_parquet_discovers_vendor_timeframe_folders(tmp_path, caplog):
    raw_root = tmp_path / "downloads" / "data"
    five_min_raw = raw_root / "5 min" / "us" / "nasdaq etfs"
    hourly_raw = raw_root / "hourly" / "us" / "nasdaq etfs"
    five_min_raw.mkdir(parents=True)
    hourly_raw.mkdir(parents=True)
    five_min_raw.joinpath("AAPL.US.txt").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T14:30:00Z,10,12,9,11,100\n"
        "2024-01-02T14:35:00Z,11,13,10,12,200\n",
        encoding="utf-8",
    )
    hourly_raw.joinpath("AAPL.US.txt").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T15:00:00Z,20,22,19,21,100\n"
        "2024-01-02T16:00:00Z,21,23,20,22,200\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "processed"

    with caplog.at_level(logging.INFO):
        five_min_result = MarketParquetImporter(raw_root, output_root).import_timeframe("5m")[0]
        hourly_result = MarketParquetImporter(raw_root, output_root).import_timeframe("1h")[0]

    assert five_min_result.source_path.endswith(str(Path("5 min") / "us" / "nasdaq etfs" / "AAPL.US.txt"))
    assert hourly_result.source_path.endswith(str(Path("hourly") / "us" / "nasdaq etfs" / "AAPL.US.txt"))
    assert market_parquet_path(output_root, "AAPL", "5m").exists()
    assert market_parquet_path(output_root, "AAPL", "1h").exists()
    assert "Detected raw folder: 5 min -> canonical: 5m -> files: 1" in caplog.text
    assert "Detected raw folder: hourly -> canonical: 1h -> files: 1" in caplog.text


def test_normalize_timeframe_source_folder_maps_vendor_names():
    assert normalize_timeframe_source_folder("5 min") == "5m"
    assert normalize_timeframe_source_folder("5min") == "5m"
    assert normalize_timeframe_source_folder("hourly") == "1h"
    assert normalize_timeframe_source_folder("1 hour") == "1h"
    assert normalize_timeframe_source_folder("daily") == "1Day"
    assert normalize_timeframe_source_folder("1 day") == "1Day"


def test_market_parquet_output_path_guard_rejects_collapsed_symbol_path(tmp_path):
    with pytest.raises(AssertionError, match="must include symbol and timeframe"):
        assert_market_parquet_output_path(
            tmp_path / "processed" / "AAPL" / "bars.parquet",
            tmp_path / "processed",
            "AAPL",
            "1h",
        )


def test_market_parquet_import_skips_malformed_files_and_continues(tmp_path, caplog):
    raw_dir = tmp_path / "h_us_txt" / "data"
    raw_dir.mkdir(parents=True)
    (raw_dir / "MSFT_1h.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T15:00:00Z,10,12,9,11,100\n"
        "2024-01-02T15:00:00Z,10,12,9,11,100\n",
        encoding="utf-8",
    )
    (raw_dir / "AAPL_1h.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T15:00:00Z,10,12,9,11,100\n"
        "2024-01-02T16:00:00Z,11,13,10,12,200\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        results = MarketParquetImporter(raw_dir, tmp_path / "processed").import_timeframe("1h")

    assert [result.symbol for result in results] == ["AAPL"]
    assert "MSFT_1h.csv" in caplog.text
    assert "duplicate timestamps" in caplog.text


def test_market_parquet_import_skips_empty_files(tmp_path, caplog):
    raw_dir = tmp_path / "h_us_txt" / "data"
    raw_dir.mkdir(parents=True)
    (raw_dir / "SIXG_1h.csv").write_text("", encoding="utf-8")
    (raw_dir / "QQQ_1h.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T15:00:00Z,10,12,9,11,100\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        results = MarketParquetImporter(raw_dir, tmp_path / "processed").import_timeframe("1h")

    assert [result.symbol for result in results] == ["QQQ"]
    assert "SIXG_1h.csv" in caplog.text
    assert "contains no OHLCV rows" in caplog.text


def test_legacy_daily_parquet_migration_writes_structured_daily_layout(tmp_path):
    legacy_dir = tmp_path / "stooq_parquet"
    legacy_dir.mkdir()
    _write_legacy_daily_parquet(legacy_dir / "SPY.parquet")
    output_root = tmp_path / "processed"

    result = migrate_legacy_daily_parquet(legacy_dir, output_root, symbols=["SPY"])[0]
    feed = MarketParquetDataFeed(output_root)
    candles = feed.get_historical_bars(
        "SPY",
        "1Day",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
    )

    assert result.output_path == str(market_parquet_path(output_root, "SPY", "1Day"))
    assert [candle.close for candle in candles] == [11.0, 12.0]


def test_intraday_summary_features_use_asof_cutoff_without_future_bars():
    rows = [{"feature_date": "2024-01-02", "daily_feature": 1.0}]
    candles = {
        "5m": {
            "SPY": [
                Candle("SPY", datetime(2024, 1, 2, 20, 55, tzinfo=timezone.utc), 10, 11, 9, 10, 100),
                Candle("SPY", datetime(2024, 1, 2, 21, 0, tzinfo=timezone.utc), 10, 11, 9, 11, 200),
                Candle("SPY", datetime(2024, 1, 2, 21, 5, tzinfo=timezone.utc), 11, 13, 10, 12, 999),
            ]
        }
    }

    enriched = add_intraday_summary_features(
        rows,
        candles,
        benchmark_symbols=["SPY"],
        session_close_utc="21:00",
    )

    assert enriched[0]["5m_spy_return_last_bar"] == pytest.approx(0.10)
    assert enriched[0]["5m_spy_volume_last_bar"] == 200.0


def test_market_parquet_import_rejects_missing_intraday_directory(tmp_path):
    config = {
        "ml": {
            "market_data": {
                "processed_root": str(tmp_path / "processed"),
                "raw_root": str(tmp_path / "downloads"),
                "enabled_timeframes": ["5m", "1h"],
                "timeframes": {
                    "5m": {"raw_dir": str(tmp_path / "downloads" / "5_us_txt" / "data")},
                    "1h": {"raw_dir": str(tmp_path / "downloads" / "h_us_txt" / "data")},
                },
            }
        }
    }

    with pytest.raises(FileNotFoundError, match="5m: .*5_us_txt.*1h: .*h_us_txt"):
        run_market_parquet_import(config)


def _write_legacy_daily_parquet(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [
            {
                "timestamp": datetime(2024, 1, 1),
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
                "volume": 100.0,
            },
            {
                "timestamp": datetime(2024, 1, 2),
                "open": 11.0,
                "high": 13.0,
                "low": 10.0,
                "close": 12.0,
                "volume": 200.0,
            },
        ]
    )
    pq.write_table(table, path)
