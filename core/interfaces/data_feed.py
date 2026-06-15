from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from core.entities.candle import Candle


class IDataFeed(ABC):

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[Candle]:
        pass