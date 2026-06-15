# core/interfaces/strategy.py

from abc import ABC, abstractmethod
from core.entities.signal import Signal


class IStrategy(ABC):

    @abstractmethod
    def generate_signal(self, context) -> Signal:
        pass