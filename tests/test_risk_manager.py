from datetime import datetime

from core.entities.signal import Signal
from core.risk.simple_risk_manager import SimpleRiskManager


def make_signal(action):
    return Signal(
        symbol="AAPL",
        action=action,
        timestamp=datetime(2024, 1, 1),
    )


def test_hold_signal_is_blocked():
    risk = SimpleRiskManager()

    assert risk.validate(make_signal("HOLD")) is False


def test_buy_signal_is_allowed_without_portfolio_context():
    risk = SimpleRiskManager()

    assert risk.validate(make_signal("BUY")) is True


def test_position_sizing_returns_share_quantity():
    risk = SimpleRiskManager(
        max_risk_per_trade=0.01,
        max_exposure=0.2,
    )

    size = risk.position_size(
        make_signal("BUY"),
        account_equity=10_000,
        market_price=200,
    )

    assert size == 0.5
