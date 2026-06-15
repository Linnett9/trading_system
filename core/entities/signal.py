from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Signal:

    symbol: str
    action: str          # "BUY", "SELL", "HOLD"
    timestamp: datetime
    confidence: float = 1.0
    reason: str = ""     # NEW