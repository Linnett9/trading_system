from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import application.services.dual_momentum_commands as commands


def test_feedless_comparison_loads_canonical_market_parquet_with_warmup(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("pyarrow")
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    parquet_dir = tmp_path / "processed"
    for symbol in ["AAA", "BBB", "SPY"]:
        _write_market_parquet(
            parquet_dir / symbol / "1Day" / "bars.parquet",
            symbol=symbol,
            start=date(2024, 12, 20),
            days=25,
        )
    captured = {}

    def fake_writer(config, dual_config, candles_by_symbol, strategy_names=None):
        captured["symbols"] = sorted(candles_by_symbol)
        captured["first_aaa"] = candles_by_symbol["AAA"][0].timestamp.date()
        captured["last_aaa"] = candles_by_symbol["AAA"][-1].timestamp.date()
        captured["strategy_names"] = strategy_names
        return SimpleNamespace(
            json_path=tmp_path / "result.json",
            csv_path=tmp_path / "result.csv",
            markdown_path=tmp_path / "result.md",
        )

    monkeypatch.setattr(commands, "write_stock_ml_dual_momentum_comparison", fake_writer)
    monkeypatch.setattr(
        commands,
        "load_candles",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("feed-backed load_candles should not be used")
        ),
    )

    commands.run_stock_ml_dual_momentum_comparison(
        _config(artifact_path, parquet_dir),
        feed=None,
        strategy_names=["elastic_net_oos"],
    )

    assert captured["symbols"] == ["AAA", "BBB", "SPY"]
    assert captured["first_aaa"] == date(2024, 12, 20)
    assert captured["last_aaa"] >= date(2025, 1, 10)
    assert captured["strategy_names"] == ["elastic_net_oos"]
    assert commands._local_history_provider(
        _config(artifact_path, parquet_dir)
    ) == "market_parquet"


def test_feedless_comparison_requires_canonical_symbol_timeframe_data(tmp_path):
    pytest.importorskip("pyarrow")
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    parquet_dir = tmp_path / "processed"
    _write_market_parquet(
        parquet_dir / "AAA" / "1Day" / "bars.parquet",
        symbol="AAA",
        start=date(2024, 12, 20),
        days=25,
    )

    with pytest.raises(FileNotFoundError, match="Missing local historical data"):
        commands.load_stock_ml_comparison_local_candles(
            _config(artifact_path, parquet_dir),
            _dual_config(),
            symbols=["AAA", "BBB"],
        )


def test_comparison_provider_override_ignores_obsolete_stooq_default(tmp_path):
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    config = _config(artifact_path, tmp_path / "processed")

    assert config["ml"]["historical_data_provider"] == "stooq_parquet"
    assert commands._local_history_provider(config) == "market_parquet"


def test_broad_universe_requests_artifact_symbols_and_allows_missing_market_data(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("pyarrow")
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    parquet_dir = tmp_path / "processed"
    _write_market_parquet(
        parquet_dir / "AAA" / "1Day" / "bars.parquet",
        symbol="AAA",
        start=date(2024, 12, 20),
        days=25,
    )
    config = _config(artifact_path, parquet_dir)
    config["ml"]["stock_ml_dual_momentum_comparison"]["mode"] = (
        "broad_research_universe"
    )

    captured = {}

    def fake_writer(config, dual_config, candles_by_symbol, strategy_names=None):
        captured["symbols"] = sorted(candles_by_symbol)
        return SimpleNamespace(
            json_path=tmp_path / "result.json",
            csv_path=tmp_path / "result.csv",
            markdown_path=tmp_path / "result.md",
        )

    monkeypatch.setattr(commands, "write_stock_ml_dual_momentum_comparison", fake_writer)

    commands.run_stock_ml_dual_momentum_comparison(config, feed=None)

    assert captured["symbols"] == ["AAA"]


def test_feed_backed_comparison_path_is_unchanged(monkeypatch, tmp_path):
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    feed = object()
    calls = []

    monkeypatch.setattr(
        commands,
        "load_candles",
        lambda symbol, config, received_feed: calls.append(
            (symbol, received_feed)
        )
        or [object()],
    )
    monkeypatch.setattr(
        commands,
        "load_stock_ml_comparison_local_candles",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local loader should not run when feed is provided")
        ),
    )
    monkeypatch.setattr(
        commands,
        "write_stock_ml_dual_momentum_comparison",
        lambda **kwargs: SimpleNamespace(
            json_path=tmp_path / "result.json",
            csv_path=tmp_path / "result.csv",
            markdown_path=tmp_path / "result.md",
        ),
    )

    commands.run_stock_ml_dual_momentum_comparison(
        _config(artifact_path, tmp_path / "processed"),
        feed=feed,
    )

    assert calls == [("AAA", feed), ("BBB", feed), ("SPY", feed)]


def test_local_date_window_keeps_required_warmup_history(tmp_path):
    artifact_path = _write_oos_artifact(tmp_path / "oos.csv")
    config = _config(artifact_path, tmp_path / "processed")

    start, end = commands.stock_ml_comparison_local_date_window(
        config,
        {
            **_dual_config(),
            "momentum_periods": [5],
            "regime_sma_period": 5,
            "use_asset_trend_filter": False,
        },
        {date(2025, 1, 10), date(2025, 1, 11)},
    )

    assert start.date() <= date(2024, 12, 26)
    assert end.date() == date(2025, 1, 11)


def _config(artifact_path: Path, parquet_dir: Path) -> dict:
    return {
        "backtest": {
            "starting_equity": 500,
            "symbols": ["AAA", "BBB", "SPY"],
            "timeframe": "1Day",
            "provider": "alpaca",
        },
        "ml": {
            "historical_data_provider": "stooq_parquet",
            "stooq_parquet_dir": str(parquet_dir),
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact_path),
                "output_dir": str(artifact_path.parent / "out"),
                "local_data_provider": "market_parquet",
                "market_parquet_dir": str(parquet_dir),
                "timeframe": "1Day",
            },
        },
        "research": {"dual_momentum": _dual_config()},
    }


def _dual_config() -> dict:
    return {
        "symbols": ["AAA", "BBB", "SPY"],
        "top_n": 1,
        "momentum_periods": [2],
        "regime_sma_period": 2,
        "use_asset_trend_filter": False,
        "transaction_cost_bps": 0,
    }


def _write_oos_artifact(path: Path) -> Path:
    path.write_text(
        "rebalance_date,symbol,fold_id,actual_forward_return_10d,"
        "stock_level_predicted_forward_return_10d_elastic_net,"
        "stock_level_predicted_forward_return_10d_random_forest\n"
        "2025-01-10,AAA,fold_1,0.01,0.3,0.2\n"
        "2025-01-10,BBB,fold_1,0.02,0.4,0.5\n"
        "2025-01-11,AAA,fold_1,0.01,0.5,0.6\n"
        "2025-01-11,BBB,fold_1,0.02,0.2,0.1\n",
        encoding="utf-8",
    )
    return path


def _write_market_parquet(
    path: Path,
    *,
    symbol: str,
    start: date,
    days: int,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [
        datetime.combine(start + timedelta(days=index), datetime.min.time())
        for index in range(days)
    ]
    prices = [10.0 + index for index in range(days)]
    table = pa.table({
        "timestamp": timestamps,
        "open": prices,
        "high": [price + 0.5 for price in prices],
        "low": [price - 0.5 for price in prices],
        "close": prices,
        "volume": [1000.0] * days,
        "symbol": [symbol] * days,
    })
    pq.write_table(table, path)
