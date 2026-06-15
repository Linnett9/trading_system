# core/entities/trade.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    symbol: str
    side: str                 # BUY / SELL
    entry_price: float
    entry_time: datetime

    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None

    quantity: float = 0.0
    pnl: float = 0.0

    is_open: bool = True
