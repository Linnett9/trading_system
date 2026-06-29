from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from core.research.ml.features.labels import MLLabelBuildResult
from core.research.ml.reports.ranking_outcomes import (
    future_drawdown_event,
    outcomes_by_feature_date,
    period_return,
)


def test_ranking_outcomes_return_none_for_missing_benchmark_data():
    label_result = _label_result("2024-01-01", "2024-01-03")
    equity_curve = [
        _equity_point("2024-01-01", 100.0),
        _equity_point("2024-01-03", 110.0),
    ]

    outcomes = outcomes_by_feature_date(
        {"ml": {"benchmark_symbols": ["QQQ"]}},
        label_result,
        {"SPY": [_candle("2024-01-01", 100.0), _candle("2024-01-03", 105.0)]},
        equity_curve,
    )

    assert list(outcomes["2024-01-01"]) == [
        "strategy_return",
        "excess_spy_return",
        "drawdown_event",
    ]
    outcome = outcomes["2024-01-01"]
    assert outcome["strategy_return"] == pytest.approx(0.1)
    assert outcome["excess_spy_return"] is None
    assert outcome["drawdown_event"] == 0.0


def test_ranking_outcomes_compute_strategy_and_excess_returns():
    label_result = _label_result("2024-01-01", "2024-01-03")
    equity_curve = [
        _equity_point("2024-01-01", 100.0),
        _equity_point("2024-01-02", 103.0),
        _equity_point("2024-01-03", 110.0),
    ]

    outcomes = outcomes_by_feature_date(
        {"ml": {"benchmark_symbols": ["QQQ"]}},
        label_result,
        {"QQQ": [_candle("2024-01-01", 200.0), _candle("2024-01-03", 210.0)]},
        equity_curve,
    )

    outcome = outcomes["2024-01-01"]
    assert outcome["strategy_return"] == pytest.approx(0.1)
    assert outcome["excess_spy_return"] == pytest.approx(0.05)
    assert outcome["drawdown_event"] == 0.0


def test_ranking_outcomes_detect_drawdown_event():
    label_result = _label_result("2024-01-01", "2024-01-04")
    equity_curve = [
        _equity_point("2024-01-01", 100.0),
        _equity_point("2024-01-02", 120.0),
        _equity_point("2024-01-03", 107.0),
        _equity_point("2024-01-04", 115.0),
    ]

    outcomes = outcomes_by_feature_date(
        {"ml": {"benchmark_symbols": ["SPY"]}},
        label_result,
        {"SPY": [_candle("2024-01-01", 100.0), _candle("2024-01-04", 101.0)]},
        equity_curve,
    )

    assert outcomes["2024-01-01"]["drawdown_event"] == 1.0


def test_ranking_outcome_helpers_preserve_legacy_missing_date_behavior():
    assert period_return({"2024-01-01": 100.0}, "2024-01-01", "2024-01-02") is None
    assert future_drawdown_event(
        ["2024-01-01"],
        {"2024-01-01": 100.0},
        {"2024-01-01": 0},
        "2024-01-01",
        "2024-01-02",
    ) is None


def _label_result(feature_date: str, label_end_date: str) -> MLLabelBuildResult:
    return MLLabelBuildResult(
        rows=[
            {
                "feature_date": feature_date,
                "label_start_date": feature_date,
                "label_end_date": label_end_date,
                "risk_regime": 1,
            }
        ],
        dropped_rows_insufficient_horizon=0,
        label_name="risk_regime",
    )


def _equity_point(date_text: str, equity: float) -> SimpleNamespace:
    return SimpleNamespace(timestamp=datetime.fromisoformat(date_text), equity=equity)


def _candle(date_text: str, close: float) -> SimpleNamespace:
    return SimpleNamespace(timestamp=datetime.fromisoformat(date_text), close=close)
