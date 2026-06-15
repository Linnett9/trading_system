from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class EquityPoint:
    timestamp: object
    equity: float


class PortfolioEngine:

    def __init__(self, starting_cash: float = 10_000):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.equity_curve: List[EquityPoint] = []

        # -------------------------
        # PERFORMANCE METRICS
        # -------------------------
        self.peak_equity = starting_cash
        self.max_drawdown = 0.0
        self.returns: List[float] = []
        self._last_trade_manager = None

    # -------------------------
    # MAIN UPDATE LOOP
    # -------------------------
    def update(self, trade_manager, latest_price, timestamp):

        self._last_trade_manager = trade_manager

        # -------------------------
        # 1. UNREALISED PnL
        # -------------------------
        unrealized = 0.0

        for trade in trade_manager.open_trades.values():

            if trade.side == "LONG":
                unrealized += (latest_price - trade.entry_price) * trade.quantity
            else:
                unrealized += (trade.entry_price - latest_price) * trade.quantity

        # -------------------------
        # 2. REALISED PnL
        # -------------------------
        realized = sum(t.pnl for t in trade_manager.closed_trades)

        # -------------------------
        # 3. TOTAL EQUITY
        # -------------------------
        total_equity = self.starting_cash + realized + unrealized

        # -------------------------
        # 4. EQUITY CURVE
        # -------------------------
        self.equity_curve.append(
            EquityPoint(timestamp=timestamp, equity=total_equity)
        )

        # -------------------------
        # 5. RETURNS TRACKING
        # -------------------------
        if len(self.equity_curve) > 1:
            prev = self.equity_curve[-2].equity
            if prev != 0:
                self.returns.append((total_equity - prev) / prev)

        # -------------------------
        # 6. DRAWDOWN CALCULATION
        # -------------------------
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        drawdown = (self.peak_equity - total_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        return total_equity

    @property
    def current_equity(self) -> float:

        if not self.equity_curve:
            return self.starting_cash

        return self.equity_curve[-1].equity

    # -------------------------
    # SUMMARY REPORT
    # -------------------------
    def summary(self, trade_manager=None) -> Dict:

        total_return = (self.equity_curve[-1].equity / self.starting_cash - 1) if self.equity_curve else 0

        avg_return = sum(self.returns) / len(self.returns) if self.returns else 0

        sharpe = 0
        if self.returns:
            mean = avg_return
            std = math.sqrt(sum((r - mean) ** 2 for r in self.returns) / len(self.returns))
            sharpe = (mean / std) * math.sqrt(252) if std != 0 else 0

        summary = {
            "starting_cash": self.starting_cash,
            "final_equity": self.equity_curve[-1].equity if self.equity_curve else self.starting_cash,
            "total_return": total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe,
            "num_returns_samples": len(self.returns),
        }

        manager = trade_manager or self._last_trade_manager
        if manager is None:
            return summary

        closed_trades = manager.closed_trades
        trade_count = len(closed_trades)

        wins = [trade.pnl for trade in closed_trades if trade.pnl > 0]
        losses = [trade.pnl for trade in closed_trades if trade.pnl <= 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        win_rate = len(wins) / trade_count if trade_count else 0
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0
        )
        expectancy = (
            sum(trade.pnl for trade in closed_trades) / trade_count
            if trade_count
            else 0
        )

        max_consecutive_losses = 0
        current_losses = 0

        for trade in closed_trades:
            if trade.pnl <= 0:
                current_losses += 1
                max_consecutive_losses = max(
                    max_consecutive_losses,
                    current_losses,
                )
            else:
                current_losses = 0

        summary.update({
            "closed_trades": trade_count,
            "win_rate": win_rate,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_consecutive_losses": max_consecutive_losses,
        })

        return summary
