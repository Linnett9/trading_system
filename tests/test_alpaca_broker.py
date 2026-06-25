import json
from datetime import datetime

from application.services.paper_trading_service import PaperTradingService
from core.entities.order import Order
from infrastructure.brokers.alpaca_broker import AlpacaBroker
import pytest


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


def test_get_account_parses_buying_power_and_cash():
    def opener(request, timeout=30):
        assert request.full_url == "https://paper-api.alpaca.markets/v2/account"
        return DummyResponse({
            "cash": "123.45",
            "buying_power": "100000.0",
            "portfolio_value": "100123.45",
        })

    broker = AlpacaBroker(
        config={"api_key": "paper-key", "secret_key": "paper-secret"},
        opener=opener,
    )

    account = broker.get_account()

    assert account["cash"] == 123.45
    assert account["buying_power"] == 100000.0
    assert account["equity"] == 100123.45


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


def test_broker_reconciliation_passes_when_buying_power_is_sufficient():
    class StubBroker:
        def get_account(self):
            return {"cash": 500.0, "buying_power": 100000.0}

        def get_positions(self):
            return {}

        def get_open_orders(self):
            return []

        def get_capabilities(self):
            return None

        def get_fills(self):
            return []

    class DummyDecision:
        cash = 500.0
        current_positions = {}
        orders = [
            type("Order", (), {"quantity_delta": 10.0, "limit_price": 50.0})()
        ]

    service = PaperTradingService(
        config={
            "broker": {"adapter": "alpaca"},
            "paper_trading": {"execution_adapter": "broker"},
        },
        feed=None,
    )
    service._build_broker_for_decision = lambda decision: StubBroker()

    reconciliation = service._broker_reconciliation(DummyDecision())

    assert reconciliation["passed"] is True
    assert reconciliation["broker_buying_power"] == 100000.0
class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "id": "order-1",
            "symbol": "AAPL",
            "status": "filled",
            "filled_qty": "1",
            "filled_avg_price": "100",
            "submitted_at": "2026-06-24T10:00:00Z",
            "filled_at": "2026-06-24T10:00:01Z",
        }).encode("utf-8")


def test_alpaca_market_order_payload_uses_market_type():
    captured = {}

    def opener(request, timeout=30):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    broker = AlpacaBroker(
        config={"api_key": "key", "secret_key": "secret"},
        opener=opener,
    )

    broker.submit_order(Order(
        symbol="AAPL",
        side="BUY",
        quantity=1,
        timestamp=datetime.utcnow(),
        order_type="MARKET",
    ))

    assert captured["body"]["type"] == "market"
    assert "limit_price" not in captured["body"]


def test_alpaca_limit_order_payload_uses_limit_price():
    captured = {}

    def opener(request, timeout=30):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    broker = AlpacaBroker(
        config={"api_key": "key", "secret_key": "secret"},
        opener=opener,
    )

    broker.submit_order(Order(
        symbol="AAPL",
        side="BUY",
        quantity=1,
        timestamp=datetime.utcnow(),
        order_type="LIMIT",
        limit_price=99.50,
    ))

    assert captured["body"]["type"] == "limit"
    assert captured["body"]["limit_price"] == "99.5"


def test_alpaca_limit_order_requires_limit_price():
    broker = AlpacaBroker(
        config={"api_key": "key", "secret_key": "secret"},
        opener=lambda request, timeout=30: _FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Limit order requires"):
        broker.submit_order(Order(
            symbol="AAPL",
            side="BUY",
            quantity=1,
            timestamp=datetime.utcnow(),
            order_type="LIMIT",
        ))


def test_alpaca_unsupported_order_type_raises():
    broker = AlpacaBroker(
        config={"api_key": "key", "secret_key": "secret"},
        opener=lambda request, timeout=30: _FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Unsupported Alpaca order type"):
        broker.submit_order(Order(
            symbol="AAPL",
            side="BUY",
            quantity=1,
            timestamp=datetime.utcnow(),
            order_type="STOP",
        ))


def test_alpaca_broker_accepts_legacy_secret_env_alias(monkeypatch):
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)

    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET", "legacy-paper-secret")

    broker = AlpacaBroker(opener=lambda request, timeout=30: DummyResponse({}))

    assert broker.api_key == "paper-key"
    assert broker.secret_key == "legacy-paper-secret"