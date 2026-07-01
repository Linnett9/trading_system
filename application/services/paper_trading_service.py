from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from application.services.paper_service import (
    build_paper_engine,
    create_paper_decision,
)
from application.services.paper_trading_approval import (
    approval_error,
    approval_file,
    decision_hashes,
    file_hash,
    reproducibility_metadata,
    run_id,
    save_dry_run_approval,
    stable_hash,
)
from application.services.paper_trading_broker import (
    broker_blocked_reason,
    broker_reconciliation,
    build_broker_for_decision,
    decision_prices,
    fill_to_dict,
    submit_with_broker,
)
from application.services.paper_trading_reporting import (
    append_dashboard,
    append_event_log,
    append_journal,
    is_number,
    latest_dashboard_equity,
    read_csv_rows,
    report_dir,
    save_broker_reconciliation_report,
    save_metrics_summary,
    save_model_triggered_rebalance_audit,
    save_order_preview,
    save_reconciliation_report,
    save_risk_report,
)
from application.services.paper_trading_runtime import (
    blocked_reason,
    equity_history,
    send_alerts,
    validate_mode,
)
from application.services.paper_trading_types import PaperTradingRunResult
from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.rebalance.model_triggered import (
    evaluate_model_triggered_rebalance,
    validate_rebalance_policy,
)
from core.risk.paper_risk import (
    model_kill_switch_checks,
    post_trade_risk_checks,
    portfolio_kill_switch_checks,
    pre_trade_risk_checks,
    risk_blocks_submission,
    risk_status,
)


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
        validate_rebalance_policy(self.config)
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

        rebalance_evaluation = evaluate_model_triggered_rebalance(
            self.config, decision, submit_requested=submit and not dry_run,
        )
        if rebalance_evaluation is not None:
            save_model_triggered_rebalance_audit(
                self.config, rebalance_evaluation.to_dict(),
            )

        reproducibility = self._reproducibility_metadata(candidate_id)
        blocked_reason = self._blocked_reason(decision)

        if rebalance_evaluation is not None:
            if rebalance_evaluation.decision == "NO_TRADE":
                blocked_reason = "model_triggered_no_trade"
            elif (
                submit
                and not dry_run
                and not rebalance_evaluation.paper_submit_allowed
            ):
                blocked_reason = "model_triggered_submission_not_allowed"

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

        if rebalance_evaluation is not None:
            rebalance_evaluation = rebalance_evaluation.with_submission_result(
                fill_record is not None,
            )
            save_model_triggered_rebalance_audit(
                self.config, rebalance_evaluation.to_dict(),
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
        send_alerts(
            self.config,
            run_id,
            blocked_reason,
            risk_checks,
            post_trade_checks,
        )

    def _validate_mode(self) -> None:
        validate_mode(self.config)

    def _blocked_reason(self, decision: Any) -> str | None:
        return blocked_reason(self.config, decision)

    def _broker_blocked_reason(self, decision: Any) -> str | None:
        return broker_blocked_reason(self.config, decision)

    def _broker_reconciliation(self, decision: Any) -> dict[str, Any] | None:
        return broker_reconciliation(self.config, decision)

    def _submit_with_broker(self, decision: Any) -> dict[str, Any]:
        return submit_with_broker(self.config, decision)

    def _build_broker_for_decision(self, decision: Any) -> Any:
        return build_broker_for_decision(self.config, decision)

    def _decision_prices(self, decision: Any) -> dict[str, float]:
        return decision_prices(decision)

    def _report_dir(self) -> Path:
        return report_dir(self.config)

    def _run_id(self, candidate_id: str) -> str:
        return run_id(candidate_id)

    def _reproducibility_metadata(self, candidate_id: str) -> dict[str, Any]:
        return reproducibility_metadata(self.config, candidate_id)

    def _stable_hash(self, payload: Any) -> str:
        return stable_hash(payload)

    def _file_hash(self, path_value: str | None) -> str | None:
        return file_hash(path_value)

    def _fill_to_dict(self, fill: Any) -> dict[str, Any]:
        return fill_to_dict(fill)

    def _equity_history(self) -> list[dict[str, Any]]:
        return equity_history(self.config)

    def _decision_hashes(self, decision: Any) -> tuple[str, str]:
        return decision_hashes(decision)

    def _approval_file(self) -> Path:
        return approval_file(self.config)

    def _save_dry_run_approval(
        self,
        decision: Any,
        target_hash: str,
        order_hash: str,
        risk_checks: list[Any],
        reproducibility: dict[str, Any],
    ) -> Path:
        return save_dry_run_approval(
            self.config,
            decision,
            target_hash,
            order_hash,
            risk_checks,
            reproducibility,
        )

    def _approval_error(self, target_hash: str, order_hash: str) -> str | None:
        return approval_error(self.config, target_hash, order_hash)

    def _save_order_preview(self, decision: Any) -> Path:
        return save_order_preview(self.config, decision)

    def _save_risk_report(
        self,
        decision: Any,
        risk_checks: list[Any],
        reproducibility: dict[str, Any],
    ) -> Path:
        return save_risk_report(self.config, decision, risk_checks, reproducibility)

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
        return save_metrics_summary(
            self.config,
            run_id,
            candidate_id,
            decision,
            fill_record,
            risk_checks,
            post_trade_checks,
            blocked_reason,
            reproducibility,
        )

    def _save_reconciliation_report(
        self,
        decision: Any,
        fill_record: dict[str, Any],
        post_trade_checks: list[Any],
    ) -> Path:
        return save_reconciliation_report(
            self.config,
            decision,
            fill_record,
            post_trade_checks,
        )

    def _save_broker_reconciliation_report(
        self,
        decision: Any,
        reconciliation: dict[str, Any],
    ) -> Path:
        return save_broker_reconciliation_report(
            self.config,
            decision,
            reconciliation,
        )

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
        return append_journal(
            self.config,
            run_id,
            candidate_id,
            dry_run,
            decision,
            fill_record,
            risk_checks,
            post_trade_checks,
            blocked_reason,
        )

    def _append_dashboard(
        self,
        run_id: str,
        decision: Any,
        fill_record: dict[str, Any] | None,
        risk_checks: list[Any],
        post_trade_checks: list[Any],
        blocked_reason: str | None,
    ) -> Path:
        return append_dashboard(
            self.config,
            run_id,
            decision,
            fill_record,
            risk_checks,
            post_trade_checks,
            blocked_reason,
        )

    def _latest_dashboard_equity(self, path: Path) -> float | None:
        return latest_dashboard_equity(path)

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        return read_csv_rows(path)

    def _is_number(self, value: Any) -> bool:
        return is_number(value)

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
        return append_event_log(
            self.config,
            run_id,
            candidate_id,
            dry_run,
            submit,
            decision,
            fill_record,
            risk_checks,
            post_trade_checks,
            blocked_reason,
            reproducibility,
            artifact_paths,
        )
