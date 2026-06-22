from datetime import datetime
from application.services.market_data_loader import load_candles_with_metadata
from infrastructure.data.stooq_csv_data_feed import StooqCsvDataFeed


def test_stooq_csv_data_feed_reads_downloaded_daily_csv(tmp_path):
    (tmp_path / "AAPL.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2015-01-02,10,12,9,11,100\n"
        "2015-01-05,11,13,10,12,200\n",
        encoding="utf-8",
    )
    feed = StooqCsvDataFeed(str(tmp_path))

    candles = feed.get_historical_bars(
        "AAPL",
        "1Day",
        datetime(2015, 1, 1),
        datetime(2015, 1, 31),
    )

    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert feed.get_last_request_metadata("AAPL")["source"] == "stooq_csv"


def test_stooq_csv_data_feed_reports_missing_symbol_file(tmp_path):
    feed = StooqCsvDataFeed(str(tmp_path))

    try:
        feed.get_historical_bars(
            "MSFT",
            "1Day",
            datetime(2015, 1, 1),
            datetime(2015, 1, 31),
        )
    except FileNotFoundError as error:
        assert "MSFT.csv" in str(error)
    else:
        raise AssertionError("Expected a missing Stooq CSV error")


def test_stooq_csv_mode_does_not_reuse_alpaca_cache(tmp_path):
    data_dir = tmp_path / "stooq"
    data_dir.mkdir()
    (data_dir / "AAPL.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-02,10,12,9,11,100\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "AAPL_alpaca_1Day_9y_iex_all_10000bars.json").write_text(
        "[]", encoding="utf-8"
    )
    config = {
        "backtest": {
            "years": 9,
            "timeframe": "1Day",
            "provider": "stooq_csv",
            "data_feed": "iex",
            "data_adjustment": "all",
            "historical_bar_limit": 10_000,
        },
        "cache": {"enabled": True, "data_dir": str(cache_dir)},
    }

    result = load_candles_with_metadata(
        "AAPL", config, StooqCsvDataFeed(str(data_dir))
    )

    assert result.candles[0].close == 11.0
    assert result.metadata["source"] == "stooq_csv"
    assert result.metadata["cache_hit"] is False
