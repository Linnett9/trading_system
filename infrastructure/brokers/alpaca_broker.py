from __future__ import annotations

from datetime import datetime

from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.fill import Fill
from core.entities.order import Order
from core.interfaces.broker import IBroker


class AlpacaBroker(IBroker):
    """Placeholder for future Alpaca paper/live brokerage integration."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AlpacaBroker is intentionally disabled until broker submission, "
            "risk checks, and monitoring are production-ready."
        )

    def get_account(self) -> dict:
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def get_open_orders(self) -> list[dict]:
        raise NotImplementedError

    def get_capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def submit_order(self, order: Order) -> Fill:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError

    def get_fills(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Fill]:
        raise NotImplementedError
