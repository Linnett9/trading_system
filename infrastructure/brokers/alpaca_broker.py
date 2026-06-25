from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.fill import Fill
from core.entities.order import Order
from core.interfaces.broker import IBroker


class AlpacaBroker(IBroker):
    """Minimal Alpaca paper-trading broker adapter.

    Uses Alpaca's paper trading API by default:
    https://paper-api.alpaca.markets
    """

    def __init__(
        self,
        config: dict | None = None,
        opener: Callable | None = None,
    ):
        broker_config = config or {}

        self.api_key = (
            broker_config.get("api_key")
            or broker_config.get("API_KEY")
            or os.getenv("ALPACA_API_KEY")
            or os.getenv("APCA_API_KEY_ID")
        )
        self.secret_key = (
            broker_config.get("secret_key")
            or broker_config.get("SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
            or os.getenv("ALPACA_SECRET")
            or os.getenv("APCA_API_SECRET_KEY")
        )

        self.base_url = str(
            broker_config.get("base_url", "https://paper-api.alpaca.markets")
        ).rstrip("/")

        self._opener = opener or urlopen
        self._fills: list[Fill] = []
        self._orders: list[dict[str, Any]] = []
        self._account: dict[str, Any] = {
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
        }
        self._positions: dict[str, float] = {}

        self.capabilities = BrokerCapabilities(
            supports_fractional_shares=bool(
                broker_config.get("supports_fractional", True)
            ),
            supports_market_orders=bool(
                broker_config.get("supports_market_orders", True)
            ),
            supports_limit_orders=bool(
                broker_config.get("supports_limit_orders", True)
            ),
            min_order_size=float(broker_config.get("min_order_notional", 1.0)),
            asset_class=str(broker_config.get("asset_class", "equity")),
            trading_hours=str(broker_config.get("trading_hours", "regular")),
        )

    def get_account(self) -> dict:
        payload = self._request_json("GET", "/v2/account")

        account = {
            "cash": self._to_float(payload.get("cash")),
            "equity": self._to_float(
                payload.get("portfolio_value", payload.get("equity"))
            ),
            "buying_power": self._to_float(
                payload.get("buying_power", payload.get("cash"))
            ),
            "raw": payload,
        }
        self._account = account
        return dict(account)

    def get_positions(self) -> dict[str, float]:
        payload = self._request_json("GET", "/v2/positions")

        positions: dict[str, float] = {}
        if isinstance(payload, list):
            for row in payload:
                symbol = str(row.get("symbol", "")).upper()
                if not symbol:
                    continue
                positions[symbol] = self._to_float(row.get("qty"))

        self._positions = positions
        return dict(positions)

    def get_open_orders(self) -> list[dict]:
        payload = self._request_json("GET", "/v2/orders?status=open")

        orders = payload if isinstance(payload, list) else []
        self._orders = list(orders)
        return list(orders)

    def get_capabilities(self) -> BrokerCapabilities:
        return self.capabilities

    def submit_order(self, order: Order) -> Fill:
        self._require_credentials()

        payload = self._order_payload(order)
        response_payload = self._request_json("POST", "/v2/orders", payload)

        broker_order = {
            "id": response_payload.get("id"),
            "symbol": response_payload.get("symbol", order.symbol.upper()),
            "side": response_payload.get("side", order.side.lower()),
            "type": response_payload.get("type", payload["type"]),
            "status": response_payload.get("status", "submitted"),
            "requested_qty": response_payload.get(
                "qty",
                str(abs(float(order.quantity))),
            ),
            "filled_qty": response_payload.get("filled_qty", "0"),
            "filled_avg_price": response_payload.get("filled_avg_price"),
            "submitted_at": response_payload.get("submitted_at"),
            "filled_at": response_payload.get("filled_at"),
            "raw": response_payload,
        }
        self._orders.append(broker_order)

        filled_qty = self._to_float(response_payload.get("filled_qty"))
        signed_qty = filled_qty if order.side.upper() == "BUY" else -filled_qty

        price = self._to_float(response_payload.get("filled_avg_price"))
        timestamp = self._parse_timestamp(
            response_payload.get("filled_at")
            or response_payload.get("submitted_at")
            or response_payload.get("created_at")
        )

        fill = Fill(
            symbol=response_payload.get("symbol", order.symbol.upper()),
            quantity=signed_qty,
            price=price,
            timestamp=timestamp,
            fees=0.0,
        )

        # Only record actual fills locally. An accepted/submitted Alpaca order
        # can return filled_qty=0, which should not be treated as a real fill.
        if filled_qty:
            self._fills.append(fill)
            self._positions[fill.symbol] = (
                self._positions.get(fill.symbol, 0.0) + fill.quantity
            )

        return fill

    def cancel_order(self, order_id: str) -> dict:
        payload = self._request_json("DELETE", f"/v2/orders/{order_id}")

        for order in self._orders:
            if order.get("id") == order_id:
                order["status"] = "canceled"

        return payload if isinstance(payload, dict) else {
            "id": order_id,
            "status": "canceled",
        }

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

    def _order_payload(self, order: Order) -> dict[str, Any]:
        order_type = str(order.order_type or "MARKET").lower()

        if order_type not in {"market", "limit"}:
            raise RuntimeError(f"Unsupported Alpaca order type: {order.order_type}")

        payload = {
            "symbol": order.symbol.upper(),
            "qty": str(abs(float(order.quantity))),
            "side": order.side.lower(),
            "type": order_type,
            "time_in_force": "day",
        }

        if order_type == "limit":
            if order.limit_price is None or float(order.limit_price) <= 0:
                raise RuntimeError("Limit order requires a positive limit_price.")
            payload["limit_price"] = str(float(order.limit_price))

        return payload

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self._require_credentials()

        data = None
        headers = {
            "APCA-API-KEY-ID": str(self.api_key),
            "APCA-API-SECRET-KEY": str(self.secret_key),
            "Accept": "application/json",
        }

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with self._opener(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Alpaca API request failed: {method} {path} "
                f"status={exc.code} body={error_body}"
            ) from exc

        if not body:
            return {}

        return json.loads(body)

    def _require_credentials(self) -> None:
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Missing Alpaca credentials. Expected ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or api_key/secret_key "
                "in broker config."
            )

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None or value == "":
            return 0.0
        return float(value)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not value:
            return datetime.now(timezone.utc)

        if isinstance(value, datetime):
            return value

        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))