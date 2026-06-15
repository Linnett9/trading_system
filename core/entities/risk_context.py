from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskContext:
    atr: Optional[float] = None
    volatility: Optional[float] = None
