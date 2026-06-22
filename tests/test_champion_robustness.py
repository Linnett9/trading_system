from datetime import datetime, timedelta
from types import SimpleNamespace

from core.research.champion_robustness import (
    build_champion_robustness_report,
    period_exclusion_summary,
)


def test_champion_robustness_reports_cost_and_concentration_metrics():
    points = [SimpleNamespace(timestamp=datetime(2020, 1, 1) + timedelta(days=index), equity=100 + index) for index in range(800)]
    selections = [SimpleNamespace(timestamp=points[700].timestamp, symbols=["AAPL", "MSFT"], target_weights={"AAPL": .6, "MSFT": .4}, scores={"AAPL": .3, "MSFT": .2}, exposure_target=.8)]
    result = SimpleNamespace(result=SimpleNamespace(equity_curve=points, sharpe=1.2), cagr=.2, calmar=1.5, turnover_percent=.5, annualized_turnover_percent=.1, selections=selections)
    candles = {symbol: [SimpleNamespace(timestamp=point.timestamp, close=100 + index) for index, point in enumerate(points)] for symbol in ("AAPL", "MSFT", "SPY")}
    report = build_champion_robustness_report(result, candles, {"AAPL": "Information Technology", "MSFT": "Information Technology"}, {"0": result}, {"remove_top": result})
    assert report["baseline"]["cagr"] == .2
    assert report["concentration"]["max_single_name_weight"] == .6
    assert "0" in report["transaction_cost_sensitivity"]
    period_summary = period_exclusion_summary(
        result, candles, "SPY", "2018-01-01", "2018-12-31"
    )
    assert period_summary["period_overlapped_backtest"] is False
    assert period_summary["excluded_point_count"] == 0
    assert period_summary["warning"] is not None
    assert period_summary["method"] == "post_hoc_equity_curve_exclusion"
