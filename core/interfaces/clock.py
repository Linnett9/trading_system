from abc import ABC, abstractmethod
from datetime import date, datetime


class IClock(ABC):

    @abstractmethod
    def now(self) -> datetime:
        pass

    def today(self) -> date:
        return self.now().date()

    def is_market_open(self) -> bool:
        return False

    def next_market_open(self) -> datetime | None:
        return None
