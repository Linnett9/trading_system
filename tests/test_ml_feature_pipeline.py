from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.entities.candle import Candle
from core.research.ml.config import MLExperimentConfig
from core.research.ml.pipelines.feature_pipeline import MLFeaturePipeline


def test_feature_pipeline_returns_empty_result_without_feed(tmp_path):
    config = {
        "reports": {"ml_dir": str(tmp_path / "reports")},
        "ml": {"output_dir": str(tmp_path / "reports")},
    }
    pipeline = MLFeaturePipeline(
        config,
        MLExperimentConfig.from_config(config),
        feed=None,
    )

    result = pipeline.build()

    assert result.feature_result.rows == []
    assert result.feature_result.dropped_rows == 0
    assert result.candles_by_symbol == {}
    assert result.champion_equity_curve == []
    assert result.champion_rebalance_dates == set()
    assert result.champion_selections == []
    assert result.history_data_metadata == {}
    assert result.champion_state_updated is False
    assert result.history_data_metadata_updated is False


def test_feature_pipeline_resolves_symbols_from_universe_and_benchmarks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "yaml", _yaml_module(["MSFT", "SPY"]))
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("symbols:\n  - MSFT\n  - SPY\n", encoding="utf-8")
    config = {
        "backtest": {"symbols": ["AAPL"]},
        "research": {
            "dual_momentum": {
                "symbols": ["AAPL"],
                "universe_path": str(universe_path),
            }
        },
        "ml": {"benchmark_symbols": ["SPY", "QQQ"]},
    }
    pipeline = MLFeaturePipeline(
        config,
        MLExperimentConfig.from_config(config),
    )

    assert pipeline.feature_symbols() == ["MSFT", "SPY", "QQQ"]


def test_feature_pipeline_expands_rebalance_universe_with_max_symbols(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "yaml", _yaml_module(["msft", "aapl"]))
    universe_path = tmp_path / "expanded.yaml"
    universe_path.write_text("symbols:\n  - msft\n  - aapl\n", encoding="utf-8")
    config = {
        "research": {"dual_momentum": {"symbols": ["fallback"]}},
        "ml": {
            "label_type": "should_reduce_exposure",
            "expanded_rebalance_dataset": {
                "universe_paths": [str(universe_path)],
                "max_symbols": 1,
            },
            "benchmark_symbols": ["SPY", "QQQ"],
        },
    }
    pipeline = MLFeaturePipeline(
        config,
        MLExperimentConfig.from_config(config),
    )

    assert pipeline.feature_symbols() == ["fallback", "MSFT", "SPY", "QQQ"]


def test_feature_pipeline_writes_history_coverage_report(tmp_path):
    report_dir = tmp_path / "reports"
    config = {
        "reports": {"ml_dir": str(report_dir)},
        "ml": {
            "output_dir": str(report_dir),
            "minimum_history_years": 1,
            "history_coverage_tolerance_days": 10,
        },
    }
    pipeline = MLFeaturePipeline(
        config,
        MLExperimentConfig.from_config(config),
        research_label="TEST_RESEARCH",
    )

    pipeline.validate_history_coverage(
        {"SPY": _candles("SPY", 370)},
        config["ml"],
        {"SPY": {"source": "unit"}},
    )

    payload = json.loads(
        (report_dir / "history_coverage.json").read_text(encoding="utf-8")
    )
    assert payload["coverage_sufficient"] is True
    assert payload["research_label"] == "TEST_RESEARCH"
    assert payload["short_history_allowed_for_smoke_test"] is False
    assert payload["symbols"]["SPY"]["request"] == {"source": "unit"}


def test_feature_pipeline_rejects_insufficient_history(tmp_path):
    report_dir = tmp_path / "reports"
    config = {
        "reports": {"ml_dir": str(report_dir)},
        "ml": {
            "output_dir": str(report_dir),
            "minimum_history_years": 2,
        },
    }
    pipeline = MLFeaturePipeline(
        config,
        MLExperimentConfig.from_config(config),
    )

    with pytest.raises(RuntimeError, match="historical coverage is insufficient"):
        pipeline.validate_history_coverage(
            {"SPY": _candles("SPY", 30)},
            config["ml"],
            {},
        )

    assert (report_dir / "history_coverage.json").exists()


def _candles(symbol: str, count: int) -> list[Candle]:
    start = datetime(2020, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.0 + index,
            volume=1_000,
        )
        for index in range(count)
    ]


def _yaml_module(symbols: list[str]) -> SimpleNamespace:
    return SimpleNamespace(safe_load=lambda _: {"symbols": symbols})
