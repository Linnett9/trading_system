from statistics import median

from core.entities.trade_analysis import TradeAnalysis


class TradeAnalyzer:

    def analyze(
        self,
        trades,
        period_start=None,
        period_end=None,
    ) -> TradeAnalysis:
        closed_trades = [
            trade for trade in trades
            if trade.exit_time is not None
        ]
        total_trades = len(closed_trades)

        if total_trades == 0:
            return TradeAnalysis()

        pnls = [trade.pnl for trade in closed_trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        durations = [
            self._duration_days(trade)
            for trade in closed_trades
        ]

        return TradeAnalysis(
            total_trades=total_trades,
            win_rate=len(wins) / total_trades,
            average_win=gross_profit / len(wins) if wins else 0,
            average_loss=sum(losses) / len(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            expectancy=sum(pnls) / total_trades,
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss > 0
                else 0
            ),
            average_trade_duration_days=sum(durations) / len(durations),
            median_trade_duration_days=median(durations),
            max_trade_duration_days=max(durations),
            time_in_market_percent=self._time_in_market_percent(
                durations,
                period_start,
                period_end,
            ),
        )

    def _duration_days(self, trade):
        seconds = (
            trade.exit_time
            - trade.entry_time
        ).total_seconds()

        return max(seconds / 86_400, 0)

    def _time_in_market_percent(
        self,
        durations,
        period_start,
        period_end,
    ):
        if period_start is None or period_end is None:
            return 0

        total_seconds = (period_end - period_start).total_seconds()

        if total_seconds <= 0:
            return 0

        invested_days = sum(durations)
        total_days = total_seconds / 86_400

        return min(invested_days / total_days, 1)
