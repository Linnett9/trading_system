from datetime import datetime, timedelta

import pytest

from core.entities.trade import Trade
from core.research.capital_utilization_analyzer import (
    CapitalUtilizationAnalyzer,
)
from core.services.portfolio_engine import EquityPoint


def test_capital_utilization_tracks_exposure_across_equity_curve():
    start = datetime(2024, 1, 1)
    trade = Trade(
        symbol="AAPL",
        side="LONG",
        entry_price=100,
        entry_time=start + timedelta(days=1),
        exit_time=start + timedelta(days=3),
        quantity=1,
    )
    equity_curve = [
        EquityPoint(timestamp=start + timedelta(days=index), equity=500)
        for index in range(5)
    ]

    analysis = CapitalUtilizationAnalyzer().analyze(
        trades=[trade],
        equity_curve=equity_curve,
        starting_equity=500,
    )

    assert analysis.average_position_value == 100
    assert analysis.average_exposure_percent == pytest.approx(0.12)
    assert analysis.max_exposure_percent == 0.20
    assert analysis.average_cash_percent == 0.88
