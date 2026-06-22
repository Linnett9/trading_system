from __future__ import annotations

from datetime import datetime

from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.fill import Fill
from core.entities.order import Order
from core.interfaces.broker import IBroker


class FakeBroker(IBroker):
    """In-memory broker adapter for paper-pipeline testing."""

    def __init__(
        self,
        cash: float = 500,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
        open_orders: list[dict] | None = None,
        reject_symbols: set[str] | None = None,
        partial_fill_ratio: float = 1.0,
        slippage_bps: float = 0.0,
        commission_bps: float = 0.0,
        capabilities: BrokerCapabilities | None = None,
    ):
        self.cash = cash
        self.prices = prices or {}
        self.reject_symbols = reject_symbols or set()
        self.partial_fill_ratio = partial_fill_ratio
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.positions: dict[str, float] = positions or {}
        self.open_orders: list[dict] = open_orders or []
        self.fills: list[Fill] = []
        self.capabilities = capabilities or BrokerCapabilities()

    def get_account(self) -> dict:
        return {
            "cash": self.cash,
            "equity": self.cash + self._positions_value(),
        }

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_open_orders(self) -> list[dict]:
        return list(self.open_orders)

    def get_capabilities(self) -> BrokerCapabilities:
        return self.capabilities

    def submit_order(self, order: Order) -> Fill:
        if order.symbol in self.reject_symbols:
            self.open_orders.append({
                "id": self._order_id(order),
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "status": "rejected",
                "reason": "configured_reject_symbol",
            })
            raise RuntimeError(f"FakeBroker rejected order for {order.symbol}")

        market_price = self.prices.get(order.symbol)
        if market_price is None:
            raise RuntimeError(f"No fake price configured for {order.symbol}")

        price = self._execution_price(market_price, order.side)
        filled_quantity = order.quantity * self.partial_fill_ratio
        signed_quantity = (
            filled_quantity if order.side.upper() == "BUY" else -filled_quantity
        )
        signed_notional = signed_quantity * price
        fees = abs(signed_notional) * (self.commission_bps / 10000)
        self.cash -= signed_notional
        self.cash -= fees
        self.positions[order.symbol] = (
            self.positions.get(order.symbol, 0) + signed_quantity
        )
        if abs(self.positions[order.symbol]) < 1e-10:
            self.positions.pop(order.symbol)

        fill = Fill(
            symbol=order.symbol,
            quantity=signed_quantity,
            price=price,
            timestamp=datetime.utcnow(),
            fees=fees,
        )
        self.fills.append(fill)

        if self.partial_fill_ratio < 1:
            remaining_quantity = order.quantity - filled_quantity
            self.open_orders.append({
                "id": self._order_id(order),
                "symbol": order.symbol,
                "side": order.side,
                "quantity": remaining_quantity,
                "filled_quantity": filled_quantity,
                "requested_quantity": order.quantity,
                "status": "open",
                "reason": "partial_fill",
            })

        return fill

    def cancel_order(self, order_id: str) -> dict:
        for order in self.open_orders:
            if str(id(order)) == order_id:
                order["status"] = "cancelled"
                return order

        return {"order_id": order_id, "status": "not_found"}

    def get_fills(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Fill]:
        fills = self.fills
        if start is not None:
            fills = [fill for fill in fills if fill.timestamp >= start]
        if end is not None:
            fills = [fill for fill in fills if fill.timestamp <= end]
        return list(fills)

    def _positions_value(self) -> float:
        return sum(
            quantity * self.prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )

    def _order_id(self, order: Order) -> str:
        return f"fake-{order.symbol}-{order.timestamp.timestamp()}"

    def _execution_price(self, market_price: float, side: str) -> float:
        slippage = self.slippage_bps / 10000
        if side.upper() == "BUY":
            return market_price * (1 + slippage)
        return market_price * (1 - slippage)
