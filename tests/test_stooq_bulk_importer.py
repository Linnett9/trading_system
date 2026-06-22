from datetime import datetime
import zipfile

import pytest

from application.services.market_data_loader import load_candles_with_metadata
from infrastructure.data.stooq_bulk_importer import StooqBulkImporter
from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed


HEADER = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
)


def test_stooq_bulk_import_writes_parquet_and_feed_reads_it(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    _write_bulk_file(extracted_dir / "AAPL.txt")
    parquet_dir = tmp_path / "parquet"

    result = StooqBulkImporter(
        str(extracted_dir), str(parquet_dir), minimum_history_years=9,
    ).import_symbol("AAPL")
    feed = StooqParquetDataFeed(str(parquet_dir))
    candles = feed.get_historical_bars(
        "AAPL", "1Day", datetime(2016, 1, 1), datetime(2026, 1, 3)
    )

    assert result.row_count == 2
    assert (parquet_dir / "AAPL.parquet").exists()
    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert feed.get_last_request_metadata("AAPL")["source"] == "stooq_parquet"


def test_stooq_bulk_import_reads_matching_zip_member(tmp_path):
    zip_path = tmp_path / "us_daily_ascii.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("data/AAPL.txt", _bulk_text())

    result = StooqBulkImporter(
        str(tmp_path / "missing-extracted"),
        str(tmp_path / "parquet"),
        zip_path=str(zip_path),
    ).import_symbol("AAPL")

    assert "!data/AAPL.txt" in result.source_path


def test_stooq_bulk_import_rejects_missing_symbol(tmp_path):
    importer = StooqBulkImporter(str(tmp_path), str(tmp_path / "parquet"))

    with pytest.raises(FileNotFoundError, match="MSFT"):
        importer.import_symbol("MSFT")


def test_stooq_bulk_import_rejects_short_history_and_bad_ohlcv(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    (extracted_dir / "MSFT.txt").write_text(
        HEADER + "MSFT.US,D,20240102,000000,10,12,9,11,100,0\n",
        encoding="utf-8",
    )
    (extracted_dir / "NVDA.txt").write_text(
        HEADER
        + "NVDA.US,D,20160101,000000,10,8,9,11,100,0\n"
        + "NVDA.US,D,20260102,000000,11,13,10,12,100,0\n",
        encoding="utf-8",
    )
    importer = StooqBulkImporter(str(extracted_dir), str(tmp_path / "parquet"))

    with pytest.raises(ValueError, match="insufficient history"):
        importer.import_symbol("MSFT")
    with pytest.raises(ValueError, match="invalid OHLCV"):
        importer.import_symbol("NVDA")


def test_stooq_parquet_does_not_reuse_alpaca_cache(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    _write_bulk_file(extracted_dir / "AAPL.txt")
    parquet_dir = tmp_path / "parquet"
    StooqBulkImporter(str(extracted_dir), str(parquet_dir)).import_symbol("AAPL")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "AAPL_alpaca_1Day_9y_iex_all_10000bars.json").write_text(
        "[]", encoding="utf-8"
    )
    config = {
        "backtest": {
            "years": 9,
            "timeframe": "1Day",
            "provider": "stooq_parquet",
            "data_feed": "iex",
            "data_adjustment": "all",
            "historical_bar_limit": 10_000,
        },
        "cache": {"enabled": True, "data_dir": str(cache_dir)},
    }

    result = load_candles_with_metadata(
        "AAPL", config, StooqParquetDataFeed(str(parquet_dir))
    )

    assert result.candles[-1].close == 12.0
    assert result.metadata["source"] == "stooq_parquet"
    assert result.metadata["cache_hit"] is False


def _write_bulk_file(path):
    path.write_text(_bulk_text(), encoding="utf-8")


def _bulk_text():
    return (
        HEADER
        + "AAPL.US,D,20160101,000000,10,12,9,11,100,0\n"
        + "AAPL.US,D,20260102,000000,11,13,10,12,200,0\n"
    )
