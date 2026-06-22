import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class PaperMonitoringService:

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def weekly_summary(self) -> Path:
        report_dir = self._report_dir()
        dashboard_rows = self._read_csv(self._dashboard_path())
        journal_rows = self._read_csv(report_dir / "journal.csv")
        fill_rows = self._read_csv(Path(
            self.config.get("paper_trading", {}).get(
                "fill_log_path",
                "data/paper/fills.csv",
            )
        ))
        today = datetime.utcnow().strftime("%Y%m%d")
        path = report_dir / f"weekly_summary_{today}.md"
        latest = dashboard_rows[-1] if dashboard_rows else {}
        latest_state = self._latest_state_snapshot()
        latest_equity = self._latest_metric(latest, latest_state, "portfolio_value")
        latest_cash = self._latest_metric(latest, latest_state, "cash")
        latest_exposure = self._latest_exposure(latest, latest_equity, latest_cash)
        risk_events = [
            row for row in journal_rows
            if row.get("risk_status") in {"ERROR", "CRITICAL"}
            or row.get("post_trade_status") in {"ERROR", "CRITICAL"}
            or row.get("notes")
        ]

        content = [
            "# Weekly Paper Summary",
            "",
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Runs: {len(journal_rows)}",
            f"Dashboard rows: {len(dashboard_rows)}",
            f"Fill rows: {len(fill_rows)}",
            f"Latest equity: {latest_equity}",
            f"Latest cash: {latest_cash}",
            f"Latest exposure: {latest_exposure}",
            f"Risk events: {len(risk_events)}",
            "",
            "## Latest Run",
            "",
            f"Run ID: {latest.get('run_id', 'n/a')}",
            f"Orders: {latest.get('orders', 'n/a')}",
            f"Risk status: {latest.get('risk_status', 'n/a')}",
            f"Post-trade status: {latest.get('post_trade_status', 'n/a')}",
            "",
            "## Notes",
            "",
            "- Review any ERROR/CRITICAL rows before relying on paper results.",
            "- Compare paper turnover and drawdown against the champion backtest.",
        ]
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        return path

    def promotion_checklist(self) -> Path:
        report_dir = self._report_dir()
        dashboard_rows = self._read_csv(self._dashboard_path())
        journal_rows = self._read_csv(report_dir / "journal.csv")
        today = datetime.utcnow().strftime("%Y%m%d")
        path = report_dir / f"promotion_checklist_{today}.md"
        unique_dates = {row.get("date") for row in dashboard_rows if row.get("date")}
        critical_rows = [
            row for row in journal_rows
            if row.get("risk_status") == "CRITICAL"
            or row.get("post_trade_status") == "CRITICAL"
        ]
        error_rows = [
            row for row in journal_rows
            if row.get("risk_status") == "ERROR"
            or row.get("post_trade_status") == "ERROR"
            or row.get("notes")
        ]
        checks = [
            (
                "30-60 paper trading days completed",
                len(unique_dates) >= 30,
                f"{len(unique_dates)} unique paper dates recorded",
            ),
            (
                "No unresolved critical errors",
                not critical_rows,
                f"{len(critical_rows)} critical rows",
            ),
            (
                "No unresolved reconciliation/order errors",
                not error_rows,
                f"{len(error_rows)} error/block rows",
            ),
            (
                "Orders match model intent",
                bool(dashboard_rows),
                "dashboard present" if dashboard_rows else "no dashboard rows",
            ),
            (
                "Manual review completed",
                False,
                "requires human sign-off",
            ),
            (
                "Latest backtest and walk-forward still acceptable",
                False,
                "requires fresh research run",
            ),
        ]
        lines = [
            "# Paper Trading Promotion Checklist",
            "",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
        ]
        for label, passed, detail in checks:
            mark = "x" if passed else " "
            lines.append(f"- [{mark}] {label} - {detail}")

        lines.extend([
            "",
            "Promotion decision: do not promote to live until every box is checked.",
        ])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _report_dir(self) -> Path:
        return Path(
            self.config.get("paper_trading", {}).get(
                "report_dir",
                self.config.get("reports", {}).get("paper_dir", "reports/paper"),
            )
        )

    def _dashboard_path(self) -> Path:
        return Path(
            self.config.get("paper_trading", {}).get(
                "dashboard_path",
                self._report_dir() / "dashboard.csv",
            )
        )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _state_path(self) -> Path:
        return self._report_dir() / "paper_state.json"

    def _latest_state_snapshot(self) -> dict[str, float]:
        path = self._state_path()
        if not path.exists():
            return {}

        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        cash = self._to_float(state.get("cash"))
        latest_equity = None
        equity_history = state.get("equity_history", [])
        if equity_history:
            latest_equity = self._to_float(
                equity_history[-1].get("equity"),
            )

        if latest_equity is None:
            latest_fill = None
            fills = state.get("fills", [])
            if fills:
                latest_fill = fills[-1]
            if latest_fill:
                latest_equity = self._to_float(latest_fill.get("equity_after"))

        snapshot = {}
        if latest_equity is not None:
            snapshot["portfolio_value"] = latest_equity
        if cash is not None:
            snapshot["cash"] = cash
        return snapshot

    def _latest_metric(
        self,
        dashboard_row: dict[str, str],
        state_snapshot: dict[str, float],
        key: str,
    ) -> str:
        dashboard_value = self._to_float(dashboard_row.get(key))
        state_value = state_snapshot.get(key)

        if (
            key == "portfolio_value"
            and dashboard_value is not None
            and state_value is not None
            and dashboard_value <= self._to_float(dashboard_row.get("cash"), 0)
            and state_value > dashboard_value
        ):
            return self._format_float(state_value)

        if dashboard_value is not None:
            return self._format_float(dashboard_value)
        if state_value is not None:
            return self._format_float(state_value)
        return "n/a"

    def _latest_exposure(
        self,
        dashboard_row: dict[str, str],
        latest_equity: str,
        latest_cash: str,
    ) -> str:
        dashboard_exposure = self._to_float(dashboard_row.get("exposure"))
        equity = self._to_float(latest_equity)
        cash = self._to_float(latest_cash)

        if (
            dashboard_exposure is not None
            and dashboard_exposure > 0
        ):
            return self._format_float(dashboard_exposure)
        if equity is None or cash is None or equity <= 0:
            return "n/a"
        return self._format_float((equity - cash) / equity)

    def _to_float(self, value: Any, default: float | None = None) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _format_float(self, value: float) -> str:
        return f"{value:.4f}"
