from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from core.entities.candle import Candle
from core.research.ml.config import MLExperimentConfig
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.label_pipeline import MLLabelPipeline


def test_label_pipeline_preserves_empty_dataset_result():
    pipeline = MLLabelPipeline(
        {"ml": {"label_type": "risk_regime"}},
        MLExperimentConfig.from_config({"ml": {"label_type": "risk_regime"}}),
        champion_equity_curve=[],
    )

    result = pipeline.build(
        MLFeatureBuildResult(rows=[], dropped_rows=0, date_range=None),
        candles_by_symbol={},
    )

    assert result.rows == []
    assert result.dropped_rows_insufficient_horizon == 0
    assert result.label_name == "risk_regime"


def test_label_pipeline_builds_champion_success_labels():
    config = {
        "ml": {
            "label_type": "champion_success",
            "label_horizon_days": 1,
            "benchmark_symbols": ["SPY", "QQQ"],
        }
    }
    pipeline = MLLabelPipeline(
        config,
        MLExperimentConfig.from_config(config),
        champion_equity_curve=[
            _equity_point("2024-01-01", 100.0),
            _equity_point("2024-01-02", 110.0),
            _equity_point("2024-01-03", 100.0),
        ],
    )

    result = pipeline.build(
        MLFeatureBuildResult(
            rows=[
                {"feature_date": "2024-01-01"},
                {"feature_date": "2024-01-02"},
                {"feature_date": "2024-01-03"},
            ],
            dropped_rows=0,
            date_range=("2024-01-01", "2024-01-03"),
        ),
        candles_by_symbol={
            "SPY": [
                _candle("SPY", "2024-01-01", 100.0),
                _candle("SPY", "2024-01-02", 105.0),
                _candle("SPY", "2024-01-03", 110.0),
            ]
        },
    )

    assert result.label_name == "champion_success"
    assert result.rows[0]["champion_success"] == 1
    assert result.rows[0]["label_start_date"] == "2024-01-02"
    assert result.rows[0]["label_end_date"] == "2024-01-02"
    assert result.dropped_rows_insufficient_horizon == 1


def test_label_pipeline_builds_should_reduce_exposure_labels():
    config = {"ml": {"label_type": "should_reduce_exposure"}}
    pipeline = MLLabelPipeline(
        config,
        MLExperimentConfig.from_config(config),
        champion_equity_curve=[],
    )

    result = pipeline.build(
        MLFeatureBuildResult(
            rows=[
                {
                    "feature_id": "feature-1",
                    "feature_date": "2024-01-01",
                    "label_start_date": "2024-01-02",
                    "label_end_date": "2024-01-05",
                    "should_reduce_exposure": 1,
                    "future_volatility": 0.2,
                    "future_drawdown": -0.1,
                    "future_max_drawdown": -0.12,
                    "champion_excess_return": -0.03,
                    "volatility_adjusted_excess_return": -0.4,
                }
            ],
            dropped_rows=0,
            date_range=("2024-01-01", "2024-01-01"),
        ),
        candles_by_symbol={"SPY": []},
    )

    assert result.label_name == "should_reduce_exposure"
    assert result.rows[0]["feature_id"] == "feature-1"
    assert result.rows[0]["should_reduce_exposure"] == 1
    assert result.rows[0]["future_max_drawdown"] == -0.12


def test_label_pipeline_rejects_unsupported_label_type():
    config = {"ml": {"label_type": "unsupported"}}
    pipeline = MLLabelPipeline(
        config,
        MLExperimentConfig.from_config(config),
        champion_equity_curve=[],
    )

    with pytest.raises(ValueError, match="Unsupported ML label type"):
        pipeline.build(
            MLFeatureBuildResult(
                rows=[{"feature_date": "2024-01-01"}],
                dropped_rows=0,
                date_range=("2024-01-01", "2024-01-01"),
            ),
            candles_by_symbol={"SPY": []},
        )


def _candle(symbol: str, date_value: str, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=datetime.fromisoformat(f"{date_value}T00:00:00"),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000,
    )


def _equity_point(date_value: str, equity: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=datetime.fromisoformat(f"{date_value}T00:00:00"),
        equity=equity,
    )
