from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

from application.services.paper_service import (
    build_paper_engine,
    create_paper_decision,
)
from application.services.broker_factory import build_broker
from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.order import Order
from core.research.dual_momentum_factory import build_dual_momentum_tester
from core.entities.trading_mode import TradingMode
from core.risk.paper_risk import (
    model_kill_switch_checks,
    post_trade_risk_checks,
    portfolio_kill_switch_checks,
    pre_trade_risk_checks,
    risk_blocks_submission,
    risk_status,
)
from infrastructure.alerts.console_alert_service import ConsoleAlertService


@dataclass(frozen=True)
class PaperTradingRunResult:
    run_id: str
    candidate_id: str
    dry_run: bool
    decision: Any
    fill_record: dict[str, Any] | None
    risk_checks: list[Any]
    post_trade_checks: list[Any]
    risk_report_path: Path
    reconciliation_report_path: Path | None
    order_preview_path: Path
    journal_path: Path
    fill_log_path: Path
    event_log_path: Path
    dashboard_path: Path
    metrics_summary_path: Path
    approval_path: Path | None
    blocked_reason: str | None = None


class PaperTradingService:
    """Application service for safe paper-trading orchestration."""

    def __init__(self, config: dict[str, Any], feed: Any):
        self.config = config
        self.feed = feed

    def run(
        self,
        dry_run: bool = True,
        submit: bool = False,
    ) -> PaperTradingRunResult:
        self._validate_mode()
        paper_config = self.config.get("paper_trading", {})
        submission_disabled = submit and not paper_config.get("submit_orders", True)

        if submission_disabled:
            submit = False

        candidate_id = (
            paper_config.get("paper_candidate_id")
            or self.config.get("paper_candidate_id", "")
        )
        run_id = self._run_id(candidate_id)

        decision = create_paper_decision(
            self.config,
            self.feed,
            build_dual_momentum_tester,
        )

        reproducibility = self._reproducibility_metadata(candidate_id)
        blocked_reason = self._blocked_reason(decision)

        if submission_disabled:
            blocked_reason = "paper_submission_disabled"

        risk_checks = pre_trade_risk_checks(decision, self.config)

        risk_checks.extend(
            portfolio_kill_switch_checks(
                current_equity=float(decision.equity),
                equity_history=self._equity_history(),
                config=self.config,
            )
        )

        risk_checks.extend(
            model_kill_switch_checks(
                decision=decision,
                config=self.config,
                reproducibility=reproducibility,
            )
        )

        if risk_blocks_submission(risk_checks):
            blocked_reason = blocked_reason or "risk_check_failed"

        if submit or paper_config.get("inspect_broker_state", False):
            blocked_reason = blocked_reason or self._broker_blocked_reason(decision)

        order_preview_path = self._save_order_preview(decision)

        risk_report_path = self._save_risk_report(
            decision,
            risk_checks,
            reproducibility,
        )

        approval_path = None
        fill_record = None
        post_trade_checks = []
        reconciliation_report_path = None
        broker_reconciliation = None

        if submit or paper_config.get("inspect_broker_state", False):
            broker_reconciliation = self._broker_reconciliation(decision)

            if broker_reconciliation is not None:
                reconciliation_report_path = self._save_broker_reconciliation_report(
                    decision,
                    broker_reconciliation,
                )

                if not broker_reconciliation["passed"]:
                    blocked_reason = (
                        blocked_reason
                        or "broker_reconciliation_failed"
                    )

        target_hash, order_hash = self._decision_hashes(decision)

        if dry_run:
            approval_path = self._save_dry_run_approval(
                decision=decision,
                target_hash=target_hash,
                order_hash=order_hash,
                risk_checks=risk_checks,
                reproducibility=reproducibility,
            )

        should_fill = submit and not dry_run and blocked_reason is None

        if should_fill:
            engine = build_paper_engine(self.config)

            if not decision.orders:
                fill_record = engine.fill_latest_decision(decision.report_path)
            else:
                approval_error = self._approval_error(target_hash, order_hash)

                if approval_error:
                    blocked_reason = approval_error
                elif paper_config.get("execution_adapter", "local_ledger") == "broker":
                    try:
                        fill_record = self._submit_with_broker(decision)
                    except RuntimeError as exc:
                        blocked_reason = f"broker_submission_failed:{exc}"
                else:
                    fill_record = engine.fill_latest_decision(decision.report_path)

        if fill_record is not None:
            post_trade_checks = post_trade_risk_checks(
                decision,
                fill_record,
                self.config,
            )

            reconciliation_report_path = self._save_reconciliation_report(
                decision,
                fill_record,
                post_trade_checks,
            )

        journal_path = self._append_journal(
            run_id=run_id,
            candidate_id=candidate_id,
            dry_run=dry_run,
            decision=decision,
            fill_record=fill_record,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
            blocked_reason=blocked_reason,
        )

        dashboard_path = self._append_dashboard(
            run_id=run_id,
            decision=decision,
            fill_record=fill_record,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
            blocked_reason=blocked_reason,
        )

        metrics_summary_path = self._save_metrics_summary(
            run_id=run_id,
            candidate_id=candidate_id,
            decision=decision,
            fill_record=fill_record,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
            blocked_reason=blocked_reason,
            reproducibility=reproducibility,
        )

        event_log_path = self._append_event_log(
            run_id=run_id,
            candidate_id=candidate_id,
            dry_run=dry_run,
            submit=submit,
            decision=decision,
            fill_record=fill_record,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
            blocked_reason=blocked_reason,
            reproducibility=reproducibility,
            artifact_paths={
                "order_preview": order_preview_path,
                "risk_report": risk_report_path,
                "reconciliation_report": reconciliation_report_path,
                "journal": journal_path,
                "dashboard": dashboard_path,
                "metrics_summary": metrics_summary_path,
            },
        )

        self._send_alerts(
            run_id=run_id,
            blocked_reason=blocked_reason,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
        )

        return PaperTradingRunResult(
            run_id=run_id,
            candidate_id=candidate_id,
            dry_run=dry_run,
            decision=decision,
            fill_record=fill_record,
            risk_checks=risk_checks,
            post_trade_checks=post_trade_checks,
            risk_report_path=risk_report_path,
            reconciliation_report_path=reconciliation_report_path,
            order_preview_path=order_preview_path,
            journal_path=journal_path,
            fill_log_path=Path(
                paper_config.get("fill_log_path", "data/paper/fills.csv")
            ),
            event_log_path=event_log_path,
            dashboard_path=dashboard_path,
            metrics_summary_path=metrics_summary_path,
            approval_path=approval_path,
            blocked_reason=blocked_reason,
        )

    def _send_alerts(
        self,
        run_id: str,
        blocked_reason: str | None,
        risk_checks: list[Any],
        post_trade_checks: list[Any],
    ) -> None:
        alert_config = self.config.get("alerts", {})

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

    def _validate_mode(self) -> None:
        trading_config = self.config.get("trading", {})
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

    def _blocked_reason(self, decision: Any) -> str | None:
        paper_config = self.config.get("paper_trading", {})

        if (
            paper_config.get("refuse_stale_data", True)
            and decision.data_freshness.get("is_stale")
        ):
            return "stale_data"

        return None

    def _broker_blocked_reason(self, decision: Any) -> str | None:
        paper_config = self.config.get("paper_trading", {})

        if paper_config.get("execution_adapter", "local_ledger") != "broker":
            return None

        broker = self._build_broker_for_decision(decision)

        try:
            account = broker.get_account()
        except Exception as exc:
            return f"broker_account_state_unreadable:{exc}"

        if not account:
            return "broker_account_state_unreadable"

        open_orders = broker.get_open_orders()
        open_symbols = {
            str(order.get("symbol"))
            for order in open_orders
            if str(order.get("status", "open")).lower()
            in {"open", "accepted", "new", "pending", "pending_new", "submitted"}
        }

        order_symbols = {order.symbol for order in decision.orders}
        conflicts = sorted(open_symbols & order_symbols)

        if conflicts:
            return "broker_open_order_conflict:" + ",".join(conflicts)

        return None

    def _broker_reconciliation(self, decision: Any) -> dict[str, Any] | None:
        paper_config = self.config.get("paper_trading", {})

        if paper_config.get("execution_adapter", "local_ledger") != "broker":
            return None

        broker_config = self.config.get("broker", {})
        broker = self._build_broker_for_decision(decision)
        account = broker.get_account()

        broker_positions = {
            symbol: float(quantity)
            for symbol, quantity in broker.get_positions().items()
        }

        local_positions = {
            symbol: float(quantity)
            for symbol, quantity in decision.current_positions.items()
        }

        broker_cash = float(account.get("cash", 0) or 0)
        broker_buying_power = float(account.get("buying_power", broker_cash) or 0)
        local_cash = float(decision.cash)
        cash_tolerance = float(broker_config.get("cash_tolerance", 1.0))
        position_tolerance = float(broker_config.get("position_tolerance", 1e-6))
        min_buying_power_buffer = float(
            broker_config.get("min_buying_power_buffer", 0.0)
        )
        cash_reconciliation = str(
            broker_config.get(
                "cash_reconciliation",
                "sleeve" if broker_config.get("sleeve_cash") is not None else "account",
            )
        ).lower()

        required_notional = sum(
            abs(float(getattr(order, "dollar_delta", 0.0) or 0.0))
            for order in decision.orders
            if float(getattr(order, "quantity_delta", 0.0) or 0.0) != 0
        )

        if required_notional <= 0:
            required_notional = sum(
                abs(float(order.quantity_delta))
                * float(
                    order.limit_price
                    if order.limit_price is not None
                    else getattr(order, "price", 0.0) or 0.0
                )
                for order in decision.orders
                if float(order.quantity_delta) != 0
            )

        required_notional = max(required_notional, 0.0)
        mismatches = []

        if (
            cash_reconciliation == "account"
            and abs(broker_cash - local_cash) > cash_tolerance
        ):
            mismatches.append(
                {
                    "reason": "cash_mismatch",
                    "local_cash": local_cash,
                    "broker_cash": broker_cash,
                    "delta": broker_cash - local_cash,
                    "tolerance": cash_tolerance,
                }
            )

        if broker_buying_power < required_notional + min_buying_power_buffer:
            mismatches.append(
                {
                    "reason": "insufficient_buying_power",
                    "local_cash": local_cash,
                    "broker_cash": broker_cash,
                    "broker_buying_power": broker_buying_power,
                    "required_notional": required_notional,
                    "buffer": min_buying_power_buffer,
                }
            )

        symbols = sorted(set(local_positions) | set(broker_positions))

        for symbol in symbols:
            local_quantity = local_positions.get(symbol, 0.0)
            broker_quantity = broker_positions.get(symbol, 0.0)
            quantity_delta = broker_quantity - local_quantity

            if abs(quantity_delta) > position_tolerance:
                mismatches.append(
                    {
                        "reason": "position_mismatch",
                        "symbol": symbol,
                        "local_quantity": local_quantity,
                        "broker_quantity": broker_quantity,
                        "delta": quantity_delta,
                        "tolerance": position_tolerance,
                    }
                )

        open_orders = broker.get_open_orders()
        recent_fills = [
            self._fill_to_dict(fill)
            for fill in broker.get_fills()
        ]

        capabilities = broker.get_capabilities() or BrokerCapabilities()

        return {
            "passed": not mismatches,
            "timestamp": datetime.utcnow().isoformat(),
            "broker_adapter": broker_config.get("adapter", "fake"),
            "broker_capabilities": capabilities.to_dict(),
            "local_cash": local_cash,
            "broker_cash": broker_cash,
            "broker_buying_power": broker_buying_power,
            "cash_reconciliation": cash_reconciliation,
            "broker_sleeve_cash": broker_config.get("sleeve_cash"),
            "required_notional": required_notional,
            "local_positions": local_positions,
            "broker_positions": broker_positions,
            "open_orders": open_orders,
            "recent_fills": recent_fills,
            "mismatches": mismatches,
        }

    def _submit_with_broker(self, decision: Any) -> dict[str, Any]:
        broker = self._build_broker_for_decision(decision)
        fills: list[dict[str, Any]] = []
        submitted_orders: list[dict[str, Any]] = []

        for paper_order in decision.orders:
            requested_quantity = abs(float(paper_order.quantity_delta))

            order = Order(
                symbol=paper_order.symbol,
                side=paper_order.side,
                quantity=requested_quantity,
                timestamp=datetime.utcnow(),
                order_type=paper_order.order_type,
                limit_price=paper_order.limit_price,
            )

            fill = broker.submit_order(order)

            filled_quantity = abs(float(getattr(fill, "quantity", 0.0) or 0.0))
            unfilled_quantity = max(requested_quantity - filled_quantity, 0.0)

            submitted_orders.append(
                {
                    "symbol": paper_order.symbol,
                    "side": paper_order.side,
                    "requested_quantity": requested_quantity,
                    "filled_quantity": filled_quantity,
                    "unfilled_quantity": unfilled_quantity,
                    "order_type": paper_order.order_type,
                    "limit_price": paper_order.limit_price,
                    "reason": paper_order.reason,
                }
            )

            # Alpaca may accept/submit an order without filling it immediately.
            # Only record an actual fill when the broker reports non-zero quantity.
            if filled_quantity > 0:
                fills.append(
                    {
                        "symbol": fill.symbol,
                        "side": paper_order.side,
                        "quantity_delta": fill.quantity,
                        "dollar_delta": fill.quantity * fill.price,
                        "price": fill.price,
                        "fees": fill.fees,
                        "requested_quantity": requested_quantity,
                        "filled_quantity": filled_quantity,
                        "unfilled_quantity": unfilled_quantity,
                        "order_type": paper_order.order_type,
                        "limit_price": paper_order.limit_price,
                        "reason": paper_order.reason,
                    }
                )

        account = broker.get_account()
        positions_after = broker.get_positions()
        open_orders_after = broker.get_open_orders()

        open_order_statuses = {
            "open",
            "accepted",
            "new",
            "pending",
            "pending_new",
            "partially_filled",
            "submitted",
        }

        open_orders_remaining = [
            order for order in open_orders_after
            if str(order.get("status", "")).lower() in open_order_statuses
        ]

        if fills and open_orders_remaining:
            status = "partial"
        elif fills:
            status = "filled"
        elif submitted_orders:
            status = "submitted"
        else:
            status = "submitted"

        fill_record = {
            "status": status,
            "already_filled": False,
            "no_orders": False,
            "filled_at": (
                datetime.utcnow().isoformat()
                if status in {"filled", "partial"}
                else None
            ),
            "decision_path": str(decision.report_path),
            "decision_timestamp": decision.timestamp.isoformat(),
            "fills": fills,
            "submitted_orders": submitted_orders,
            "cash_after": account.get("cash"),
            "positions_after": positions_after,
            "equity_after": account.get("equity"),
            "broker_adapter": self.config.get("broker", {}).get("adapter", "fake"),
            "open_orders_after": open_orders_after,
        }

        engine = build_paper_engine(self.config)
        return engine.apply_external_fill_record(decision.report_path, fill_record)
    
    def _build_broker_for_decision(self, decision: Any) -> Any:
        return build_broker(self.config, prices=self._decision_prices(decision))

    def _decision_prices(self, decision: Any) -> dict[str, float]:
        return {
            order.symbol: float(order.price)
            for order in decision.orders
            if order.price is not None
        }

    def _report_dir(self) -> Path:
        return Path(
            self.config.get("paper_trading", {}).get(
                "report_dir",
                self.config.get("reports", {}).get("paper_dir", "reports/paper"),
            )
        )

    def _run_id(self, candidate_id: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        clean_candidate = (candidate_id or "unknown").replace("/", "_")
        return f"{timestamp}_{clean_candidate}"

    def _reproducibility_metadata(self, candidate_id: str) -> dict[str, Any]:
        dual_config = self.config.get("research", {}).get("dual_momentum", {})
        candidate_config_path = dual_config.get("champion_config_path")
        return {
            "candidate_id": candidate_id,
            "config_hash": self._stable_hash(self.config),
            "candidate_config_path": candidate_config_path,
            "candidate_config_hash": self._file_hash(candidate_config_path),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "execution_adapter": self.config.get("paper_trading", {}).get(
                "execution_adapter",
                "local_ledger",
            ),
            "broker_adapter": self.config.get("broker", {}).get(
                "adapter",
                "fake",
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _stable_hash(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _file_hash(self, path_value: str | None) -> str | None:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fill_to_dict(self, fill: Any) -> dict[str, Any]:
        return {
            "symbol": fill.symbol,
            "quantity": fill.quantity,
            "price": fill.price,
            "timestamp": fill.timestamp.isoformat(),
            "fees": fill.fees,
        }

    def _equity_history(self) -> list[dict[str, Any]]:
        rows = []
        dashboard_path = Path(
            self.config.get("paper_trading", {}).get(
                "dashboard_path",
                self._report_dir() / "dashboard.csv",
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

    def _decision_hashes(self, decision: Any) -> tuple[str, str]:
        target_payload = json.dumps(
            {
                "exposure_target": round(float(decision.exposure_target), 6),
                "target_weights": {
                    symbol: round(float(weight), 6)
                    for symbol, weight in sorted(
                        decision.target_weights.items(),
                    )
                },
            },
            sort_keys=True,
        )
        order_payload = json.dumps(
            [
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity_delta": round(order.quantity_delta, 8),
                    "dollar_delta": round(order.dollar_delta, 8),
                    "target_weight": round(order.target_weight, 8),
                    "order_type": order.order_type,
                    "limit_price": (
                        round(order.limit_price, 8)
                        if order.limit_price is not None
                        else None
                    ),
                }
                for order in decision.orders
            ],
            sort_keys=True,
        )
        return (
            hashlib.sha256(target_payload.encode("utf-8")).hexdigest(),
            hashlib.sha256(order_payload.encode("utf-8")).hexdigest(),
        )

    def _approval_file(self) -> Path:
        return self._report_dir() / "dry_run_approval.json"

    def _save_dry_run_approval(
        self,
        decision: Any,
        target_hash: str,
        order_hash: str,
        risk_checks: list[Any],
        reproducibility: dict[str, Any],
    ) -> Path:
        path = self._approval_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_timestamp": decision.timestamp.isoformat(),
            "target_portfolio_hash": target_hash,
            "order_list_hash": order_hash,
            "risk_status": risk_status(risk_checks),
            "orders": len(decision.orders),
            "decision_path": str(decision.report_path),
            "reproducibility": reproducibility,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _approval_error(self, target_hash: str, order_hash: str) -> str | None:
        path = self._approval_file()
        if not path.exists():
            return "approval_required"

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "approval_invalid"

        if payload.get("target_portfolio_hash") != target_hash:
            return "approval_target_hash_mismatch"
        if payload.get("order_list_hash") != order_hash:
            return "approval_order_hash_mismatch"
        if payload.get("risk_status") in {"ERROR", "CRITICAL"}:
            return "approval_has_blocking_risk"

        return None

    def _save_order_preview(self, decision: Any) -> Path:
        path = (
            self._report_dir()
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

    def _save_risk_report(
        self,
        decision: Any,
        risk_checks: list[Any],
        reproducibility: dict[str, Any],
    ) -> Path:
        path = (
            self._report_dir()
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

    def _save_metrics_summary(
        self,
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
            self._report_dir()
            / f"metrics_summary_{decision.timestamp.strftime('%Y%m%d')}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_rows = self._read_csv_rows(Path(
            self.config.get("paper_trading", {}).get(
                "dashboard_path",
                self._report_dir() / "dashboard.csv",
            )
        ))
        journal_rows = self._read_csv_rows(self._report_dir() / "journal.csv")
        equity_values = [
            float(row["portfolio_value"])
            for row in dashboard_rows
            if self._is_number(row.get("portfolio_value"))
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

    def _save_reconciliation_report(
        self,
        decision: Any,
        fill_record: dict[str, Any],
        post_trade_checks: list[Any],
    ) -> Path:
        path = (
            self._report_dir()
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

    def _save_broker_reconciliation_report(
        self,
        decision: Any,
        reconciliation: dict[str, Any],
    ) -> Path:
        path = (
            self._report_dir()
            / f"broker_reconciliation_{decision.timestamp.strftime('%Y%m%d')}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decision_timestamp": decision.timestamp.isoformat(),
            **reconciliation,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _append_journal(
        self,
        run_id: str,
        candidate_id: str,
        dry_run: bool,
        decision: Any,
        fill_record: dict[str, Any] | None,
        risk_checks: list[Any],
        post_trade_checks: list[Any],
        blocked_reason: str | None,
    ) -> Path:
        path = self._report_dir() / "journal.csv"
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
                "mode": self.config.get("trading", {}).get("mode", "paper"),
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

    def _append_dashboard(
        self,
        run_id: str,
        decision: Any,
        fill_record: dict[str, Any] | None,
        risk_checks: list[Any],
        post_trade_checks: list[Any],
        blocked_reason: str | None,
    ) -> Path:
        path = Path(
            self.config.get("paper_trading", {}).get(
                "dashboard_path",
                self._report_dir() / "dashboard.csv",
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
        previous_equity = self._latest_dashboard_equity(path)
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

    def _latest_dashboard_equity(self, path: Path) -> float | None:
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

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError:
            return []

    def _is_number(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _append_event_log(
        self,
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
            self.config.get("paper_trading", {}).get(
                "event_log_path",
                self._report_dir() / "events.jsonl",
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