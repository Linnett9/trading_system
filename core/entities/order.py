# core/entities/order.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str          # "BUY" / "SELL"
    quantity: float
    timestamp: datetime
    order_type: str = "MARKET"
    limit_price: float | None = None
