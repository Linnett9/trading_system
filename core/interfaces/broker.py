# core/interfaces/broker.py

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.order import Order
from core.entities.fill import Fill


class IBroker(ABC):

    def get_account(self) -> dict:
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def get_open_orders(self) -> list[dict]:
        raise NotImplementedError

    def get_capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        pass

    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError

    def get_fills(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Fill]:
        raise NotImplementedError
