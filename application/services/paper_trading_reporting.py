from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.risk.paper_risk import risk_status


def report_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("paper_trading", {}).get(
            "report_dir",
            config.get("reports", {}).get("paper_dir", "reports/paper"),
        )
    )


def save_model_triggered_rebalance_audit(
    config: dict[str, Any],
    payload: dict[str, Any],
) -> Path:
    path = report_dir(config) / "model_triggered_rebalance_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_order_preview(config: dict[str, Any], decision: Any) -> Path:
    path = (
        report_dir(config)
        / f"order_preview_{decision.timestamp.strftime('%Y%m%d')}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "current_weight",
        "target_weight",
        "target_value",
        "current_value",
        "order_side",
        "order_quantity",
        "order_type",
        "limit_price",
        "estimated_price",
        "estimated_notional",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for order in decision.orders:
            target_value = decision.equity * order.target_weight
            current_value = decision.equity * order.current_weight
            writer.writerow({
                "symbol": order.symbol,
                "current_weight": order.current_weight,
                "target_weight": order.target_weight,
                "target_value": target_value,
                "current_value": current_value,
                "order_side": order.side,
                "order_quantity": abs(order.quantity_delta),
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "estimated_price": order.price,
                "estimated_notional": abs(order.dollar_delta),
                "reason": order.reason,
            })
    return path


def save_risk_report(
    config: dict[str, Any],
    decision: Any,
    risk_checks: list[Any],
    reproducibility: dict[str, Any],
) -> Path:
    path = (
        report_dir(config)
        / f"risk_check_{decision.timestamp.strftime('%Y%m%d')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision_timestamp": decision.timestamp.isoformat(),
        "risk_status": risk_status(risk_checks),
        "reproducibility": reproducibility,
        "checks": [
            {
                "passed": check.passed,
                "severity": check.severity.value,
                "reason": check.reason,
                "details": check.details,
            }
            for check in risk_checks
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_metrics_summary(
    config: dict[str, Any],
    run_id: str,
    candidate_id: str,
    decision: Any,
    fill_record: dict[str, Any] | None,
    risk_checks: list[Any],
    post_trade_checks: list[Any],
    blocked_reason: str | None,
    reproducibility: dict[str, Any],
) -> Path:
    path = (
        report_dir(config)
        / f"metrics_summary_{decision.timestamp.strftime('%Y%m%d')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_rows = read_csv_rows(Path(
        config.get("paper_trading", {}).get(
            "dashboard_path",
            report_dir(config) / "dashboard.csv",
        )
    ))
    journal_rows = read_csv_rows(report_dir(config) / "journal.csv")
    equity_values = [
        float(row["portfolio_value"])
        for row in dashboard_rows
        if is_number(row.get("portfolio_value"))
    ]
    latest_equity = (
        float(fill_record.get("equity_after"))
        if fill_record and fill_record.get("equity_after") is not None
        else float(decision.equity)
    )
    peak_equity = max(equity_values + [latest_equity]) if latest_equity else 0
    drawdown = (
        (latest_equity / peak_equity) - 1
        if peak_equity
        else 0
    )
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "candidate_id": candidate_id,
        "latest_equity": latest_equity,
        "latest_cash": (
            fill_record.get("cash_after")
            if fill_record
            else decision.cash
        ),
        "orders_generated": len(decision.orders),
        "fills_count": (
            len(fill_record.get("fills", []))
            if fill_record
            else 0
        ),
        "runs_recorded": len(journal_rows),
        "blocked_runs_recorded": len([
            row for row in journal_rows
            if row.get("notes")
        ]),
        "risk_status": risk_status(risk_checks),
        "post_trade_status": (
            risk_status(post_trade_checks)
            if post_trade_checks
            else None
        ),
        "blocked_reason": blocked_reason,
        "max_drawdown_observed": drawdown,
        "reproducibility": reproducibility,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_reconciliation_report(
    config: dict[str, Any],
    decision: Any,
    fill_record: dict[str, Any],
    post_trade_checks: list[Any],
) -> Path:
    path = (
        report_dir(config)
        / f"reconciliation_{decision.timestamp.strftime('%Y%m%d')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision_timestamp": decision.timestamp.isoformat(),
        "post_trade_status": risk_status(post_trade_checks),
        "fill_status": fill_record.get("status"),
        "fills": fill_record.get("fills", []),
        "cash_after": fill_record.get("cash_after"),
        "equity_after": fill_record.get("equity_after"),
        "positions_after": fill_record.get("positions_after", {}),
        "checks": [
            {
                "passed": check.passed,
                "severity": check.severity.value,
                "reason": check.reason,
                "details": check.details,
            }
            for check in post_trade_checks
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_broker_reconciliation_report(
    config: dict[str, Any],
    decision: Any,
    reconciliation: dict[str, Any],
) -> Path:
    path = (
        report_dir(config)
        / f"broker_reconciliation_{decision.timestamp.strftime('%Y%m%d')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision_timestamp": decision.timestamp.isoformat(),
        **reconciliation,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def append_journal(
    config: dict[str, Any],
    run_id: str,
    candidate_id: str,
    dry_run: bool,
    decision: Any,
    fill_record: dict[str, Any] | None,
    risk_checks: list[Any],
    post_trade_checks: list[Any],
    blocked_reason: str | None,
) -> Path:
    path = report_dir(config) / "journal.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_id",
        "candidate_id",
        "mode",
        "dry_run",
        "orders_generated",
        "orders_sent",
        "orders_filled",
        "portfolio_value",
        "cash",
        "risk_status",
        "post_trade_status",
        "notes",
    ]
    exists = path.exists()
    orders_filled = (
        len(fill_record.get("fills", []))
        if fill_record is not None
        else 0
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "candidate_id": candidate_id,
            "mode": config.get("trading", {}).get("mode", "paper"),
            "dry_run": dry_run,
            "orders_generated": len(decision.orders),
            "orders_sent": orders_filled,
            "orders_filled": orders_filled,
            "portfolio_value": decision.equity,
            "cash": decision.cash,
            "risk_status": risk_status(risk_checks),
            "post_trade_status": (
                risk_status(post_trade_checks)
                if post_trade_checks
                else ""
            ),
            "notes": blocked_reason or "",
        })
    return path


def append_dashboard(
    config: dict[str, Any],
    run_id: str,
    decision: Any,
    fill_record: dict[str, Any] | None,
    risk_checks: list[Any],
    post_trade_checks: list[Any],
    blocked_reason: str | None,
) -> Path:
    path = Path(
        config.get("paper_trading", {}).get(
            "dashboard_path",
            report_dir(config) / "dashboard.csv",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "run_id",
        "portfolio_value",
        "cash",
        "exposure",
        "daily_return",
        "drawdown",
        "benchmark_return",
        "excess_return",
        "orders",
        "risk_status",
        "post_trade_status",
        "blocked_reason",
    ]
    equity = (
        float(fill_record.get("equity_after"))
        if fill_record and fill_record.get("equity_after") is not None
        else decision.equity
    )
    cash = (
        float(fill_record.get("cash_after"))
        if fill_record and fill_record.get("cash_after") is not None
        else decision.cash
    )
    exposure = (equity - cash) / equity if equity else 0
    previous_equity = latest_dashboard_equity(path)
    daily_return = (
        (equity / previous_equity) - 1
        if previous_equity
        else 0
    )
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "date": decision.timestamp.date().isoformat(),
            "run_id": run_id,
            "portfolio_value": equity,
            "cash": cash,
            "exposure": exposure,
            "daily_return": daily_return,
            "drawdown": "",
            "benchmark_return": "",
            "excess_return": "",
            "orders": len(decision.orders),
            "risk_status": risk_status(risk_checks),
            "post_trade_status": (
                risk_status(post_trade_checks)
                if post_trade_checks
                else ""
            ),
            "blocked_reason": blocked_reason or "",
        })

    return path


def append_event_log(
    config: dict[str, Any],
    run_id: str,
    candidate_id: str,
    dry_run: bool,
    submit: bool,
    decision: Any,
    fill_record: dict[str, Any] | None,
    risk_checks: list[Any],
    post_trade_checks: list[Any],
    blocked_reason: str | None,
    reproducibility: dict[str, Any],
    artifact_paths: dict[str, Path | None],
) -> Path:
    path = Path(
        config.get("paper_trading", {}).get(
            "event_log_path",
            report_dir(config) / "events.jsonl",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_type": "paper_trading_run",
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "strategy_id": "dual_momentum",
        "candidate_id": candidate_id,
        "dry_run": dry_run,
        "submit": submit,
        "portfolio_value": decision.equity,
        "cash": decision.cash,
        "risk_status": risk_status(risk_checks),
        "post_trade_status": (
            risk_status(post_trade_checks)
            if post_trade_checks
            else None
        ),
        "orders_count": len(decision.orders),
        "fills_count": (
            len(fill_record.get("fills", []))
            if fill_record
            else 0
        ),
        "blocked_reason": blocked_reason,
        "reproducibility": reproducibility,
        "artifacts": {
            key: str(value)
            for key, value in artifact_paths.items()
            if value is not None
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def latest_dashboard_equity(path: Path) -> float | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None

    if not rows:
        return None

    try:
        return float(rows[-1].get("portfolio_value") or 0)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
