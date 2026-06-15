from datetime import datetime

from core.entities.fill import Fill
from core.entities.portfolio import Portfolio
from core.entities.signal import Signal
from core.risk.simple_risk_manager import SimpleRiskManager


def test_duplicate_buy_is_blocked_when_portfolio_already_long():
    risk = SimpleRiskManager()
    portfolio = Portfolio()

    portfolio.update(Fill(
        symbol="AAPL",
        quantity=100,
        price=200,
        timestamp=datetime(2024, 1, 1),
    ))

    signal = Signal(
        symbol="AAPL",
        action="BUY",
        timestamp=datetime(2024, 1, 2),
    )

    assert risk.validate(signal, portfolio) is False
