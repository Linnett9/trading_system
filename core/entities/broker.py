# core/interfaces/broker.py

from abc import ABC, abstractmethod
from core.entities.order import Order
from core.entities.fill import Fill


class IBroker(ABC):

    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        pass