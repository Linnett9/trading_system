from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from application.services.paper_trading_reporting import report_dir
from core.entities.trading_mode import TradingMode
from core.risk.paper_risk import risk_status
from infrastructure.alerts.console_alert_service import ConsoleAlertService


def send_alerts(
    config: dict[str, Any],
    run_id: str,
    blocked_reason: str | None,
    risk_checks: list[Any],
    post_trade_checks: list[Any],
) -> None:
    alert_config = config.get("alerts", {})

    if not alert_config.get("enabled", True):
        return

    risk_state = risk_status(risk_checks)
    post_state = risk_status(post_trade_checks) if post_trade_checks else "PASS"

    if blocked_reason:
        ConsoleAlertService().send_alert(
            title="Paper trading blocked",
            message=f"run_id={run_id} reason={blocked_reason}",
            severity="ERROR",
        )
        return

    if risk_state in {"ERROR", "CRITICAL"}:
        ConsoleAlertService().send_alert(
            title="Paper pre-trade risk failure",
            message=f"run_id={run_id} risk_status={risk_state}",
            severity=risk_state,
        )

    if post_state in {"ERROR", "CRITICAL"}:
        ConsoleAlertService().send_alert(
            title="Paper post-trade reconciliation failure",
            message=f"run_id={run_id} post_trade_status={post_state}",
            severity=post_state,
        )


def validate_mode(config: dict[str, Any]) -> None:
    trading_config = config.get("trading", {})
    mode = trading_config.get("mode", "paper")
    trading_mode = TradingMode(mode)

    if (
        trading_mode == TradingMode.LIVE
        and not trading_config.get("live_enabled", False)
    ):
        raise RuntimeError("Refusing live trading because live_enabled=false.")

    if trading_mode not in {
        TradingMode.PAPER,
        TradingMode.BACKTEST,
        TradingMode.WALK_FORWARD,
    }:
        raise RuntimeError(f"Unsupported trading mode for paper service: {mode}")


def blocked_reason(config: dict[str, Any], decision: Any) -> str | None:
    paper_config = config.get("paper_trading", {})

    if (
        paper_config.get("refuse_stale_data", True)
        and decision.data_freshness.get("is_stale")
    ):
        return "stale_data"

    return None


def equity_history(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    dashboard_path = Path(
        config.get("paper_trading", {}).get(
            "dashboard_path",
            report_dir(config) / "dashboard.csv",
        )
    )
    if dashboard_path.exists():
        try:
            with dashboard_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append({
                        "timestamp": row.get("date"),
                        "equity": row.get("portfolio_value"),
                    })
        except OSError:
            return rows
    return rows
