# core/entities/strategy_context.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class StrategyContext:
    symbol: str
    timestamp: datetime
    ema_fast: float
    ema_slow: float
    atr: Optional[float] = None
    volatility: Optional[float] = None
    volatility_percentile: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    relative_volume: Optional[float] = None
    current_position: Optional[str] = None
    close: Optional[float] = None
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    previous_sma_200: Optional[float] = None
    volatility_average: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_bandwidth: Optional[float] = None
    session_open: Optional[float] = None
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    vwap: Optional[float] = None
    market_regime: str = "unknown"
    volatility_regime: str = "unknown"
