from datetime import datetime, timedelta
from types import SimpleNamespace

from main import paper_benchmark_metrics, paper_drift_rows


def test_paper_benchmark_metrics_compares_same_period():
    start = datetime(2026, 6, 1)
    status = {
        "starting_cash": 500,
        "cash": 0,
        "mark_to_market_equity": 550,
        "fills": [
            {
                "decision_timestamp": start.isoformat(),
            },
        ],
    }
    candles = [
        SimpleNamespace(timestamp=start, close=100),
        SimpleNamespace(timestamp=start + timedelta(days=10), close=120),
    ]

    metrics = paper_benchmark_metrics(status, candles, "SPY")

    assert round(metrics["paper_return"], 10) == 0.10
    assert round(metrics["benchmark_return"], 10) == 0.20
    assert round(metrics["excess_return"], 10) == -0.10


def test_paper_drift_rows_compare_target_to_current_weights():
    status = {
        "cash": 0,
        "mark_to_market_equity": 500,
        "positions": {"AAPL": 2, "SPY": 3},
        "prices_used": {"AAPL": 100, "SPY": 100},
    }
    decision_payload = {
        "target_weights": {"AAPL": 0.50, "SPY": 0.50},
        "exposure_target": 1.0,
    }

    rows = paper_drift_rows(status, decision_payload)
    rows_by_symbol = {row["symbol"]: row for row in rows}

    assert rows_by_symbol["AAPL"]["current_weight"] == 0.40
    assert rows_by_symbol["AAPL"]["target_weight"] == 0.50
    assert round(rows_by_symbol["AAPL"]["drift"], 10) == 0.10
    assert rows_by_symbol["SPY"]["current_weight"] == 0.60
    assert rows_by_symbol["SPY"]["target_weight"] == 0.50
    assert round(rows_by_symbol["SPY"]["drift"], 10) == -0.10
