from __future__ import annotations

from typing import Any

from core.risk.paper_risk_types import RiskCheckResult, RiskSeverity


def post_trade_risk_checks(
    decision: Any,
    fill_record: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[RiskCheckResult]:
    if fill_record is None:
        return [RiskCheckResult(
            passed=True,
            severity=RiskSeverity.INFO,
            reason="no_fill_to_reconcile",
            details={},
        )]

    if fill_record.get("no_orders"):
        return [RiskCheckResult(
            passed=True,
            severity=RiskSeverity.INFO,
            reason="no_orders_to_reconcile",
            details={},
        )]

    if fill_record.get("already_filled"):
        return [RiskCheckResult(
            passed=True,
            severity=RiskSeverity.INFO,
            reason="decision_already_filled",
            details={},
        )]

    risk_config = config.get("risk", {}).get("paper", {})
    portfolio_config = config.get("portfolio", {})
    checks = []
    checks.extend(_fill_count_checks(decision, fill_record))
    checks.extend(_post_trade_cash_checks(decision, fill_record, portfolio_config))
    checks.extend(_unexpected_position_checks(decision, fill_record))
    checks.extend(_target_drift_checks(decision, fill_record, risk_config))

    if not checks:
        checks.append(RiskCheckResult(
            passed=True,
            severity=RiskSeverity.INFO,
            reason="post_trade_checks_passed",
            details={},
        ))

    return checks
def _fill_count_checks(
    decision: Any,
    fill_record: dict[str, Any],
) -> list[RiskCheckResult]:
    expected = len(decision.orders)
    actual = len(fill_record.get("fills", []))
    if actual != expected:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="fill_count_mismatch",
            details={
                "expected_fills": expected,
                "actual_fills": actual,
            },
        )]
    return []
def _post_trade_cash_checks(
    decision: Any,
    fill_record: dict[str, Any],
    portfolio_config: dict[str, Any],
) -> list[RiskCheckResult]:
    cash_buffer_percent = portfolio_config.get("cash_buffer_percent", 0.02)
    cash_after = float(fill_record.get("cash_after", 0) or 0)
    equity_after = float(fill_record.get("equity_after", decision.equity) or 0)
    min_cash = equity_after * cash_buffer_percent

    if cash_after < min_cash:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="post_trade_cash_buffer_breached",
            details={
                "cash_after": cash_after,
                "min_cash": min_cash,
                "cash_buffer_percent": cash_buffer_percent,
            },
        )]

    return []
def _unexpected_position_checks(
    decision: Any,
    fill_record: dict[str, Any],
) -> list[RiskCheckResult]:
    positions_after = fill_record.get("positions_after", {}) or {}
    target_symbols = {
        symbol
        for symbol, weight in decision.target_weights.items()
        if abs(weight * decision.exposure_target) > 1e-8
    }
    unexpected = [
        symbol
        for symbol, quantity in positions_after.items()
        if abs(float(quantity)) > 1e-8 and symbol not in target_symbols
    ]

    if unexpected:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="unexpected_positions_after_fill",
            details={"symbols": sorted(unexpected)},
        )]

    return []
def _target_drift_checks(
    decision: Any,
    fill_record: dict[str, Any],
    risk_config: dict[str, Any],
) -> list[RiskCheckResult]:
    tolerance = risk_config.get("post_trade_drift_tolerance", 0.005)
    positions_after = fill_record.get("positions_after", {}) or {}
    equity_after = float(fill_record.get("equity_after", decision.equity) or 0)
    prices = {order.symbol: order.price for order in decision.orders}
    checks = []

    if equity_after <= 0:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="post_trade_equity_invalid",
            details={"equity_after": equity_after},
        )]

    for symbol, raw_target_weight in decision.target_weights.items():
        price = prices.get(symbol)
        if price is None:
            continue

        current_value = float(positions_after.get(symbol, 0) or 0) * price
        current_weight = current_value / equity_after
        target_weight = raw_target_weight * decision.exposure_target
        drift = current_weight - target_weight

        if abs(drift) > tolerance:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.WARNING,
                reason="post_trade_target_drift",
                details={
                    "symbol": symbol,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift": drift,
                    "tolerance": tolerance,
                },
            ))

    return checks
