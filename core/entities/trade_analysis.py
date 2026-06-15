from dataclasses import dataclass


@dataclass(frozen=True)
class TradeAnalysis:
    total_trades: int = 0
    win_rate: float = 0
    average_win: float = 0
    average_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    expectancy: float = 0
    profit_factor: float = 0
    average_trade_duration_days: float = 0
    median_trade_duration_days: float = 0
    max_trade_duration_days: float = 0
    time_in_market_percent: float = 0

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "average_trade_duration_days": self.average_trade_duration_days,
            "median_trade_duration_days": self.median_trade_duration_days,
            "max_trade_duration_days": self.max_trade_duration_days,
            "time_in_market_percent": self.time_in_market_percent,
        }
