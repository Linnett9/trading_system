# core/entities/position.py

from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    quantity: float = 0
    avg_price: float = 0