# core/entities/fill.py

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fill:
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
