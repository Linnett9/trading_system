from __future__ import annotations

from typing import Any

from core.risk.paper_risk_types import RiskCheckResult, RiskSeverity
from core.risk.paper_risk_utils import _is_positive_number


def portfolio_kill_switch_checks(
    current_equity: float,
    equity_history: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[RiskCheckResult]:
    kill_config = config.get("risk", {}).get("kill_switch", {})
    if not kill_config.get("enabled", False):
        return []

    checks = []
    history = [
        float(item["equity"])
        for item in equity_history
        if _is_positive_number(item.get("equity"))
    ]
    current_equity = float(current_equity)
    if current_equity <= 0:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="portfolio_equity_invalid",
            details={"current_equity": current_equity},
        )]

    if not history:
        return []

    max_daily_loss = kill_config.get("max_daily_loss")
    previous_equity = history[-1]
    daily_return = (current_equity / previous_equity) - 1 if previous_equity else 0
    if max_daily_loss is not None and daily_return < -float(max_daily_loss):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="portfolio_daily_loss_kill_switch",
            details={
                "daily_return": daily_return,
                "limit": -float(max_daily_loss),
                "previous_equity": previous_equity,
                "current_equity": current_equity,
            },
        ))

    max_weekly_loss = kill_config.get("max_weekly_loss")
    weekly_start = history[-5] if len(history) >= 5 else history[0]
    weekly_return = (current_equity / weekly_start) - 1 if weekly_start else 0
    if max_weekly_loss is not None and weekly_return < -float(max_weekly_loss):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="portfolio_weekly_loss_kill_switch",
            details={
                "weekly_return": weekly_return,
                "limit": -float(max_weekly_loss),
                "start_equity": weekly_start,
                "current_equity": current_equity,
            },
        ))

    max_drawdown = kill_config.get("max_drawdown_from_paper_start")
    peak_equity = max(history + [current_equity])
    drawdown = (current_equity / peak_equity) - 1 if peak_equity else 0
    if max_drawdown is not None and drawdown < -float(max_drawdown):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="portfolio_drawdown_kill_switch",
            details={
                "drawdown": drawdown,
                "limit": -float(max_drawdown),
                "peak_equity": peak_equity,
                "current_equity": current_equity,
            },
        ))

    return checks
def model_kill_switch_checks(
    decision: Any,
    config: dict[str, Any],
    reproducibility: dict[str, Any] | None = None,
) -> list[RiskCheckResult]:
    kill_config = config.get("risk", {}).get("model_kill_switch", {})
    if not kill_config.get("enabled", False):
        return []

    checks = []
    model_context = getattr(decision, "model_context", {}) or {}
    reproducibility = reproducibility or {}

    if (
        kill_config.get("block_stale_data", True)
        and getattr(decision, "data_freshness", {}).get("is_stale")
    ):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="model_stale_data_kill_switch",
            details=getattr(decision, "data_freshness", {}),
        ))

    if kill_config.get("require_model_context", True) and not model_context:
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="model_signal_unavailable",
            details={"model_context_present": False},
        ))

    if model_context.get("rebalance_failed"):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="latest_rebalance_failed",
            details={"model_context": model_context},
        ))

    expected_hash = kill_config.get("expected_candidate_config_hash")
    actual_hash = reproducibility.get("candidate_config_hash")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.CRITICAL,
            reason="candidate_config_hash_drift",
            details={
                "expected_candidate_config_hash": expected_hash,
                "actual_candidate_config_hash": actual_hash,
                "candidate_config_path": reproducibility.get(
                    "candidate_config_path",
                ),
            },
        ))

    return checks
