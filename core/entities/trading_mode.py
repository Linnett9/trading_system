from enum import Enum


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    WALK_FORWARD = "walk_forward"
    PAPER = "paper"
    LIVE = "live"
