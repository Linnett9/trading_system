# core/interfaces/risk_manager.py

from abc import ABC, abstractmethod
from typing import Optional

from core.entities.risk_context import RiskContext
from core.entities.signal import Signal


class IRiskManager(ABC):

    @abstractmethod
    def validate(self, signal: Signal) -> bool:
        pass

    @abstractmethod
    def position_size(
        self,
        signal: Signal,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        pass
