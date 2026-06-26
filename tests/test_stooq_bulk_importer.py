from datetime import datetime
import json
import zipfile

import pytest

from application.services.stooq_bulk_commands import run_stooq_bulk_import
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


def test_stooq_bulk_import_reads_official_recursive_directory_layout(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nested_dir = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nested_dir.mkdir(parents=True)
    _write_bulk_file(nested_dir / "aapl.us.txt")

    result = StooqBulkImporter(
        str(stooq_root),
        str(tmp_path / "parquet"),
    ).import_symbol("AAPL")

    assert result.row_count == 2
    assert result.source_path.endswith("aapl.us.txt")
    assert (tmp_path / "parquet" / "AAPL.parquet").exists()


def test_stooq_bulk_import_finds_official_layout_from_missing_extracted_dir(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nested_dir = stooq_root / "data" / "daily" / "us" / "nyse etfs" / "2"
    nested_dir.mkdir(parents=True)
    (nested_dir / "spy.us.txt").write_text(
        _bulk_text(symbol="SPY"),
        encoding="utf-8",
    )

    result = StooqBulkImporter(
        str(stooq_root / "extracted"),
        str(tmp_path / "parquet"),
    ).import_symbol("SPY")

    assert result.source_path.endswith("spy.us.txt")
    assert (tmp_path / "parquet" / "SPY.parquet").exists()


def test_stooq_bulk_import_reuses_recursive_symbol_index(tmp_path, monkeypatch):
    stooq_root = tmp_path / "stooq_bulk"
    nasdaq = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nyse_etfs = stooq_root / "data" / "daily" / "us" / "nyse etfs" / "2"
    nasdaq.mkdir(parents=True)
    nyse_etfs.mkdir(parents=True)
    _write_bulk_file(nasdaq / "aapl.us.txt")
    (nyse_etfs / "spy.us.txt").write_text(
        _bulk_text(symbol="SPY"),
        encoding="utf-8",
    )
    importer = StooqBulkImporter(str(stooq_root), str(tmp_path / "parquet"))
    original_build = importer._build_symbol_index
    build_calls = 0

    def counted_build():
        nonlocal build_calls
        build_calls += 1
        return original_build()

    first = importer.import_symbol("AAPL")
    monkeypatch.setattr(importer, "_build_symbol_index", counted_build)

    second = importer.import_symbol("SPY")

    assert first.source_path.endswith("aapl.us.txt")
    assert second.source_path.endswith("spy.us.txt")
    assert (tmp_path / "parquet" / "SPY.parquet").exists()
    assert build_calls == 0


def test_stooq_bulk_import_prefers_flat_extracted_layout(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nested_dir = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nested_dir.mkdir(parents=True)
    _write_bulk_file(stooq_root / "AAPL.txt")
    (nested_dir / "aapl.us.txt").write_text(
        _bulk_text(symbol="AAPL", close=99.0),
        encoding="utf-8",
    )

    result = StooqBulkImporter(str(stooq_root), str(tmp_path / "parquet")).import_symbol(
        "AAPL"
    )
    feed = StooqParquetDataFeed(str(tmp_path / "parquet"))
    candles = feed.get_historical_bars(
        "AAPL", "1Day", datetime(2016, 1, 1), datetime(2026, 1, 3)
    )

    assert result.source_path.endswith("AAPL.txt")
    assert candles[-1].close == 12.0


def test_stooq_bulk_raw_candidates_filter_and_rank_recursive_layout(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nasdaq_stocks = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nyse_stocks = stooq_root / "data" / "daily" / "us" / "nyse stocks" / "2"
    nyse_etfs = stooq_root / "data" / "daily" / "us" / "nyse etfs" / "3"
    nasdaq_stocks.mkdir(parents=True)
    nyse_stocks.mkdir(parents=True)
    nyse_etfs.mkdir(parents=True)
    (nasdaq_stocks / "aapl.us.txt").write_text(
        _bulk_text_with_rows("AAPL", 4),
        encoding="utf-8",
    )
    (nyse_stocks / "msft.us.txt").write_text(
        _bulk_text_with_rows("MSFT", 6),
        encoding="utf-8",
    )
    (nasdaq_stocks / "brunw.us.txt").write_text(
        _bulk_text_with_rows("BRUNW", 8),
        encoding="utf-8",
    )
    (nyse_etfs / "spy.us.txt").write_text(
        _bulk_text_with_rows("SPY", 10),
        encoding="utf-8",
    )
    importer = StooqBulkImporter(str(stooq_root), str(tmp_path / "parquet"))

    candidates = importer.select_raw_symbols(
        top=2,
        asset_class="stocks",
        min_rows=4,
        exclude_warrants_units_rights=True,
    )

    assert [candidate.symbol for candidate in candidates] == ["MSFT", "AAPL"]
    assert [candidate.asset_class for candidate in candidates] == [
        "stocks",
        "stocks",
    ]
    assert [candidate.row_count for candidate in candidates] == [6, 4]


def test_stooq_bulk_raw_candidates_support_etf_filter(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nasdaq_stocks = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nyse_etfs = stooq_root / "data" / "daily" / "us" / "nyse etfs" / "2"
    nasdaq_stocks.mkdir(parents=True)
    nyse_etfs.mkdir(parents=True)
    (nasdaq_stocks / "aapl.us.txt").write_text(
        _bulk_text_with_rows("AAPL", 5),
        encoding="utf-8",
    )
    (nyse_etfs / "spy.us.txt").write_text(
        _bulk_text_with_rows("SPY", 3),
        encoding="utf-8",
    )
    importer = StooqBulkImporter(str(stooq_root), str(tmp_path / "parquet"))

    candidates = importer.select_raw_symbols(asset_class="etfs", min_rows=3)

    assert [candidate.symbol for candidate in candidates] == ["SPY"]
    assert candidates[0].asset_class == "etfs"


def test_stooq_bulk_command_imports_top_raw_candidates(tmp_path):
    stooq_root = tmp_path / "stooq_bulk"
    nasdaq_stocks = stooq_root / "data" / "daily" / "us" / "nasdaq stocks" / "1"
    nyse_etfs = stooq_root / "data" / "daily" / "us" / "nyse etfs" / "2"
    nasdaq_stocks.mkdir(parents=True)
    nyse_etfs.mkdir(parents=True)
    (nasdaq_stocks / "aapl.us.txt").write_text(
        _bulk_text_with_rows("AAPL", 3),
        encoding="utf-8",
    )
    (nasdaq_stocks / "msft.us.txt").write_text(
        _bulk_text_with_rows("MSFT", 5),
        encoding="utf-8",
    )
    (nyse_etfs / "spy.us.txt").write_text(
        _bulk_text_with_rows("SPY", 7),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    parquet_dir = tmp_path / "parquet"
    config = {
        "ml": {
            "stooq_bulk_extracted_dir": str(stooq_root / "extracted"),
            "stooq_parquet_dir": str(parquet_dir),
            "output_dir": str(report_dir),
            "minimum_history_years": 0,
        },
        "research": {"stooq_import_symbols": ["SHOULD_NOT_IMPORT"]},
        "backtest": {"symbols": ["ALSO_IGNORED"]},
    }

    run_stooq_bulk_import(
        config,
        top=500,
        asset_class="stocks",
        min_rows=4,
    )

    payload = json.loads(
        (report_dir / "stooq_bulk_import.json").read_text(encoding="utf-8")
    )
    assert payload["requested_symbol_count"] == 1
    assert payload["symbols"][0]["symbol"] == "MSFT"
    assert (parquet_dir / "MSFT.parquet").exists()
    assert not (parquet_dir / "SPY.parquet").exists()


def test_stooq_bulk_import_manifest_reports_resume_and_missing_data(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    _write_bulk_file(extracted_dir / "AAPL.txt")
    parquet_dir = tmp_path / "parquet"
    manifest_path = tmp_path / "manifest.json"
    importer = StooqBulkImporter(str(extracted_dir), str(parquet_dir))

    first = importer.import_symbols_with_manifest(
        ["AAPL", "MSFT"],
        manifest_path=manifest_path,
        resume=True,
    )
    second = importer.import_symbols_with_manifest(
        ["AAPL"],
        manifest_path=manifest_path,
        resume=True,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(first.imported) == 1
    assert first.missing_symbols[0]["symbol"] == "MSFT"
    assert first.imported[0].missing_trading_day_gaps == 1
    assert second.skipped_existing[0].symbol == "AAPL"
    assert payload["skipped_existing_symbol_count"] == 1
    assert payload["research_only"] is True


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


def _bulk_text(symbol: str = "AAPL", close: float = 12.0):
    return (
        HEADER
        + f"{symbol}.US,D,20160101,000000,10,12,9,11,100,0\n"
        + f"{symbol}.US,D,20260102,000000,11,13,10,{close},200,0\n"
    )


def _bulk_text_with_rows(symbol: str, row_count: int):
    rows = []
    for index in range(row_count):
        day = index + 1
        rows.append(
            f"{symbol}.US,D,202001{day:02d},000000,10,12,9,11,100,0\n"
        )
    return HEADER + "".join(rows)
