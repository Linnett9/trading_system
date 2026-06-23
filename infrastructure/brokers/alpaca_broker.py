from __future__ import annotations

from datetime import datetime
import json
from typing import Callable
from urllib.request import Request, urlopen

from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.fill import Fill
from core.entities.order import Order
from core.interfaces.broker import IBroker


class AlpacaBroker(IBroker):
    """Minimal Alpaca paper-trading broker adapter."""

    def __init__(
        self,
        config: dict | None = None,
        opener: Callable | None = None,
    ):
        broker_config = config or {}
        self.api_key = broker_config.get("api_key") or broker_config.get("API_KEY")
        self.secret_key = broker_config.get("secret_key") or broker_config.get("SECRET_KEY")
        self._opener = opener or urlopen
        self._fills: list[Fill] = []
        self._orders: list[dict] = []
        self._account: dict = {"cash": 0.0, "equity": 0.0}
        self._positions: dict[str, float] = {}
        self.capabilities = BrokerCapabilities()
        self.base_url = broker_config.get("base_url", "https://paper-api.alpaca.markets")

    def get_account(self) -> dict:
        if not self.api_key or not self.secret_key:
            return dict(self._account)

        request = Request(
            f"{self.base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            },
        )
        with self._opener(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        account = {
            "cash": float(payload.get("cash", 0) or 0),
            "equity": float(payload.get("portfolio_value", payload.get("equity", 0)) or 0),
            "buying_power": float(payload.get("buying_power", payload.get("cash", 0)) or 0),
            "raw": payload,
        }
        self._account = account
        return dict(account)

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_open_orders(self) -> list[dict]:
        return list(self._orders)

    def get_capabilities(self) -> BrokerCapabilities:
        return self.capabilities

    def submit_order(self, order: Order) -> Fill:
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.upper(),
            "type": "market",
            "time_in_force": "day",
            "order_class": "simple",
        }
        request = Request(
            "https://paper-api.alpaca.markets/v2/orders",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with self._opener(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        fill = Fill(
            symbol=response_payload.get("symbol", order.symbol),
            quantity=float(response_payload.get("filled_qty", 0) or 0),
            price=float(response_payload.get("filled_avg_price", 0) or 0),
            timestamp=datetime.fromisoformat(
                response_payload.get("filled_at", "2026-01-01T00:00:00Z").replace("Z", "+00:00")
            ),
            fees=0.0,
        )
        self._fills.append(fill)
        self._orders.append({
            "id": response_payload.get("id"),
            "symbol": response_payload.get("symbol", order.symbol),
            "status": response_payload.get("status", "filled"),
        })
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + fill.quantity
        return fill

    def cancel_order(self, order_id: str) -> dict:
        for order in self._orders:
            if order.get("id") == order_id:
                order["status"] = "canceled"
                return order
        return {"id": order_id, "status": "not_found"}

    def get_fills(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Fill]:
        fills = list(self._fills)
        if start is not None:
            fills = [fill for fill in fills if fill.timestamp >= start]
        if end is not None:
            fills = [fill for fill in fills if fill.timestamp <= end]
        return fills
