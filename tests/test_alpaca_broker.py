import json
from datetime import datetime

from core.entities.order import Order
from infrastructure.brokers.alpaca_broker import AlpacaBroker


class DummyResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_submit_order_uses_paper_endpoint_and_returns_fill():
    def opener(request, timeout=30):
        assert request.full_url == "https://paper-api.alpaca.markets/v2/orders"
        headers = dict(request.header_items())
        assert headers["Apca-api-key-id"] == "paper-key"
        assert headers["Apca-api-secret-key"] == "paper-secret"
        return DummyResponse({
            "id": "order-123",
            "symbol": "AAPL",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "123.45",
            "filled_at": "2026-06-23T12:00:00Z",
        })

    broker = AlpacaBroker(
        config={"api_key": "paper-key", "secret_key": "paper-secret"},
        opener=opener,
    )
    order = Order(
        symbol="AAPL",
        side="BUY",
        quantity=10,
        timestamp=datetime.utcnow(),
    )

    fill = broker.submit_order(order)

    assert fill.symbol == "AAPL"
    assert fill.quantity == 10.0
    assert fill.price == 123.45
