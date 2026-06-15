from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
import json
from typing import Any, List

from core.entities.capital_utilization import CapitalUtilization
from core.entities.signal_diagnostics import SignalDiagnostics
from core.entities.trade_analysis import TradeAnalysis


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float
    closed_trades: int
    open_trades: int
    equity_curve: List[Any]
    profit_factor: float = 0
    trade_analysis: TradeAnalysis = field(default_factory=TradeAnalysis)
    capital_utilization: CapitalUtilization = field(
        default_factory=CapitalUtilization
    )
    signal_diagnostics: SignalDiagnostics = field(
        default_factory=SignalDiagnostics
    )

    def to_dict(self) -> dict:
        return {
            "starting_equity": self.starting_equity,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "closed_trades": self.closed_trades,
            "open_trades": self.open_trades,
            "profit_factor": self.profit_factor,
            "trade_analysis": self.trade_analysis.to_dict(),
            "capital_utilization": self.capital_utilization.to_dict(),
            "signal_diagnostics": self.signal_diagnostics.to_dict(),
            "equity_curve": [
                self._serialize_equity_point(point)
                for point in self.equity_curve
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict):
        from core.entities.capital_utilization import CapitalUtilization
        from core.entities.signal_diagnostics import SignalDiagnostics
        from core.entities.trade_analysis import TradeAnalysis

        trade_analysis = TradeAnalysis(
            **payload.get("trade_analysis", {})
        )
        capital_utilization = CapitalUtilization(
            **payload.get("capital_utilization", {})
        )
        signal_diagnostics = SignalDiagnostics(
            **payload.get("signal_diagnostics", {})
        )

        return cls(
            starting_equity=payload.get("starting_equity", 0),
            final_equity=payload.get("final_equity", 0),
            total_return=payload.get("total_return", 0),
            max_drawdown=payload.get("max_drawdown", 0),
            sharpe=payload.get("sharpe", 0),
            closed_trades=payload.get("closed_trades", 0),
            open_trades=payload.get("open_trades", 0),
            equity_curve=[],
            profit_factor=payload.get("profit_factor", 0),
            trade_analysis=trade_analysis,
            capital_utilization=capital_utilization,
            signal_diagnostics=signal_diagnostics,
        )

    def save_json(
        self,
        symbol: str,
        timeframe: str,
        report_dir: str = "reports/backtests",
        run_date: date | None = None,
    ) -> Path:
        report_date = run_date or date.today()
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / f"{report_date.isoformat()}_{symbol}_{timeframe}.json"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            **self.to_dict(),
        }

        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        return path

    def _serialize_equity_point(self, point) -> dict:
        data = asdict(point)
        timestamp = data.get("timestamp")

        if isinstance(timestamp, datetime):
            data["timestamp"] = timestamp.isoformat()

        return data
