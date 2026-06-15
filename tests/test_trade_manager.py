from datetime import datetime

from core.entities.fill import Fill
from core.services.trade_manager import TradeManager


def make_fill(quantity, price):
    return Fill(
        symbol="AAPL",
        quantity=quantity,
        price=price,
        timestamp=datetime(2024, 1, 1),
    )


def test_open_trade_stores_long_position():
    manager = TradeManager()

    trade = manager.open_trade(make_fill(quantity=10, price=100))

    assert trade is not None
    assert trade.side == "LONG"
    assert trade.quantity == 10
    assert manager.get_position("AAPL") == "LONG"


def test_duplicate_open_returns_existing_trade():
    manager = TradeManager()

    first = manager.open_trade(make_fill(quantity=10, price=100))
    second = manager.open_trade(make_fill(quantity=5, price=120))

    assert second is first
    assert manager.get_open_trade("AAPL").entry_price == 100


def test_close_trade_calculates_pnl_and_exit_reason():
    manager = TradeManager()
    manager.open_trade(make_fill(quantity=10, price=100))

    trade = manager.close_trade(
        make_fill(quantity=-10, price=120),
        exit_reason="manual exit",
    )

    assert trade is not None
    assert trade.is_open is False
    assert trade.pnl == 200
    assert trade.exit_reason == "manual exit"
    assert len(manager.closed_trades) == 1
    assert manager.get_open_trade("AAPL") is None
