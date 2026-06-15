from datetime import datetime

from core.entities.signal import Signal
from core.execution.simple_execution_model import SimpleExecutionModel


def make_signal(action):
    return Signal(
        symbol="AAPL",
        action=action,
        timestamp=datetime(2024, 1, 1),
    )


def test_buy_fill_is_above_market():
    model = SimpleExecutionModel(
        spread_bps=2,
        slippage_bps=1,
        seed=42,
    )

    assert model.create_fill_price(make_signal("BUY"), 100) > 100


def test_sell_fill_is_below_market():
    model = SimpleExecutionModel(
        spread_bps=2,
        slippage_bps=1,
        seed=42,
    )

    assert model.create_fill_price(make_signal("SELL"), 100) < 100


def test_zero_spread_and_slippage_fill_at_market():
    model = SimpleExecutionModel(
        spread_bps=0,
        slippage_bps=0,
    )

    assert model.create_fill_price(make_signal("BUY"), 100) == 100
    assert model.create_fill_price(make_signal("SELL"), 100) == 100
