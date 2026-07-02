from core.entities.backtest_result import BacktestResult
from core.research.capital_utilization_analyzer import CapitalUtilizationAnalyzer
from core.research.trade_analyzer import TradeAnalyzer


class BacktestEngineResultsMixin:
    def summary(self):

        if self._last_result is not None:
            return self._last_result.to_dict()

        if hasattr(self.portfolio_engine, "summary"):
            stats = self.portfolio_engine.summary(self.trade_manager)
            return self._result_from_summary(stats).to_dict()

        return {
            "open_trades": len(self.trade_manager.open_trades),
            "closed_trades": len(self.trade_manager.closed_trades),
        }
    def save_report(
        self,
        result=None,
        report_dir: str = "reports/backtests",
    ):

        backtest_result = result or self._last_result

        if backtest_result is None:
            backtest_result = self._build_result()

        return backtest_result.save_json(
            symbol=self.symbol,
            timeframe=self.timeframe,
            report_dir=report_dir,
        )
    def _build_result(self):

        if hasattr(self.portfolio_engine, "summary"):
            stats = self.portfolio_engine.summary(self.trade_manager)
            return self._result_from_summary(stats)

        return BacktestResult(
            starting_equity=self.account_equity,
            final_equity=self.account_equity,
            total_return=0,
            max_drawdown=0,
            sharpe=0,
            closed_trades=len(self.trade_manager.closed_trades),
            open_trades=len(self.trade_manager.open_trades),
            equity_curve=[],
            profit_factor=0,
            trade_analysis=TradeAnalyzer().analyze([]),
            capital_utilization=CapitalUtilizationAnalyzer().analyze(
                [],
                [],
                self.account_equity,
            ),
            signal_diagnostics=self.signal_diagnostics,
        )
    def _result_from_summary(self, stats):
        equity_curve = list(self.portfolio_engine.equity_curve)
        period_start = equity_curve[0].timestamp if equity_curve else None
        period_end = equity_curve[-1].timestamp if equity_curve else None
        trade_analysis = TradeAnalyzer().analyze(
            self.trade_manager.closed_trades,
            period_start=period_start,
            period_end=period_end,
        )
        all_trades = [
            *self.trade_manager.closed_trades,
            *self.trade_manager.open_trades.values(),
        ]
        capital_utilization = CapitalUtilizationAnalyzer().analyze(
            trades=all_trades,
            equity_curve=equity_curve,
            starting_equity=stats["starting_cash"],
        )

        return BacktestResult(
            starting_equity=stats["starting_cash"],
            final_equity=stats["final_equity"],
            total_return=stats["total_return"],
            max_drawdown=stats["max_drawdown"],
            sharpe=stats["sharpe_ratio"],
            closed_trades=len(self.trade_manager.closed_trades),
            open_trades=len(self.trade_manager.open_trades),
            equity_curve=equity_curve,
            profit_factor=stats.get("profit_factor", 0),
            trade_analysis=trade_analysis,
            capital_utilization=capital_utilization,
            signal_diagnostics=self.signal_diagnostics,
        )
