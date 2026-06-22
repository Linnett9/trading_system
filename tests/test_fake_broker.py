from datetime import datetime

from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.order import Order
from infrastructure.brokers.fake_broker import FakeBroker


def test_fake_broker_applies_slippage_and_commission_to_buy():
    broker = FakeBroker(
        cash=1000,
        prices={"AMAT": 100},
        slippage_bps=10,
        commission_bps=5,
    )

    fill = broker.submit_order(Order(
        symbol="AMAT",
        side="BUY",
        quantity=2,
        timestamp=datetime(2026, 6, 1),
    ))

    assert fill.price == 100.1
    assert round(fill.fees, 4) == 0.1001
    assert round(broker.get_account()["cash"], 4) == 799.6999


def test_fake_broker_tracks_partial_fill_open_order():
    broker = FakeBroker(
        cash=1000,
        prices={"AMAT": 100},
        partial_fill_ratio=0.5,
    )

    fill = broker.submit_order(Order(
        symbol="AMAT",
        side="BUY",
        quantity=2,
        timestamp=datetime(2026, 6, 1),
    ))

    assert fill.quantity == 1
    assert broker.get_positions()["AMAT"] == 1
    assert broker.get_open_orders()[0]["quantity"] == 1
    assert broker.get_open_orders()[0]["reason"] == "partial_fill"


def test_fake_broker_exposes_capabilities():
    broker = FakeBroker(
        capabilities=BrokerCapabilities(
            supports_fractional_shares=False,
            supports_market_orders=True,
            supports_limit_orders=False,
            min_order_size=25,
            asset_class="equity",
            trading_hours="regular",
        )
    )

    capabilities = broker.get_capabilities()

    assert capabilities.supports_fractional_shares is False
    assert capabilities.supports_limit_orders is False
    assert capabilities.min_order_size == 25
