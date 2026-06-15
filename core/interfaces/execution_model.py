# core/interfaces/execution_model.py

from abc import ABC, abstractmethod
from core.entities.signal import Signal


class IExecutionModel(ABC):

    @abstractmethod
    def create_fill_price(self, signal: Signal, market_price: float) -> float:
        """
        Convert signal + market price into an actual execution price.
        """
        pass