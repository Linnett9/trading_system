from __future__ import annotations

from typing import Any

from core.risk.paper_risk_types import RiskCheckResult, RiskSeverity


def pre_trade_risk_checks(decision: Any, config: dict[str, Any]) -> list[RiskCheckResult]:
    risk_config = config.get("risk", {}).get("paper", {})
    portfolio_config = config.get("portfolio", {})
    broker_config = config.get("broker", {})
    paper_config = config.get("paper_trading", {})
    checks = []

    checks.extend(_data_checks(decision))
    checks.extend(_exposure_checks(decision, risk_config))
    checks.extend(_order_checks(decision, risk_config, broker_config))
    checks.extend(_cash_buffer_checks(decision, portfolio_config))
    checks.extend(_portfolio_concentration_checks(decision, config))
    checks.extend(_unpriced_current_position_checks(decision))
    checks.extend(_broker_capability_checks(decision, broker_config, paper_config))

    if not checks:
        checks.append(RiskCheckResult(
            passed=True,
            severity=RiskSeverity.INFO,
            reason="risk_checks_passed",
            details={},
        ))

    return checks
def _unpriced_current_position_checks(decision: Any) -> list[RiskCheckResult]:
    """Block legacy holdings that cannot be priced or explicitly traded today."""
    order_symbols = {order.symbol for order in decision.orders}
    unpriced_symbols = sorted(
        symbol
        for symbol, quantity in getattr(decision, "current_positions", {}).items()
        if quantity and symbol not in decision.target_weights and symbol not in order_symbols
    )
    if not unpriced_symbols:
        return []
    return [RiskCheckResult(
        passed=False,
        severity=RiskSeverity.ERROR,
        reason="unpriced_current_holdings",
        details={"symbols": unpriced_symbols},
    )]
def _portfolio_concentration_checks(decision: Any, config: dict[str, Any]) -> list[RiskCheckResult]:
    from core.research.ml.data.sector_reference import load_sector_by_symbol

    risk_config = config.get("risk", {}).get("paper", {})
    if "max_sector_weight" not in risk_config:
        return []
    sectors = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        config.get("ml", {}).get("sector_by_symbol", {}),
    )
    weights = {symbol: weight * decision.exposure_target for symbol, weight in decision.target_weights.items()}
    sector_weights = {}
    for symbol, weight in weights.items():
        sector = sectors.get(symbol)
        if sector is None:
            return [RiskCheckResult(False, RiskSeverity.ERROR, "sector_mapping_missing", {"symbol": symbol})]
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    checks = []
    max_sector = max(sector_weights.values(), default=0.0)
    if max_sector > float(risk_config.get("max_sector_weight", 0.60)):
        checks.append(RiskCheckResult(False, RiskSeverity.ERROR, "max_sector_weight_exceeded", {"weight": max_sector, "limit": risk_config.get("max_sector_weight", 0.60)}))
    correlation = (getattr(decision, "model_context", {}) or {}).get("max_pairwise_correlation")
    limit = float(risk_config.get("max_pairwise_correlation", 0.90))
    if correlation is not None and correlation > limit:
        checks.append(RiskCheckResult(False, RiskSeverity.ERROR, "max_pairwise_correlation_exceeded", {"correlation": correlation, "limit": limit}))
    return checks
def _data_checks(decision: Any) -> list[RiskCheckResult]:
    checks = []
    freshness = decision.data_freshness or {}
    context = getattr(decision, "model_context", {}) or {}
    if not context.get("benchmark_available", True):
        checks.append(RiskCheckResult(False, RiskSeverity.ERROR, "benchmark_data_missing", {}))
    if freshness.get("is_stale"):
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="stale_data",
            details=freshness,
        ))

    data_quality = getattr(decision, "data_quality", {}) or {}
    issues_by_symbol = data_quality.get("issues_by_symbol", {}) or {}
    relevant_symbols = _decision_relevant_symbols(decision)

    for symbol in sorted(relevant_symbols):
        for issue in issues_by_symbol.get(symbol, []):
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity(issue.get("severity", "ERROR")),
                reason=f"data_quality_{issue.get('reason', 'unknown')}",
                details={
                    "symbol": symbol,
                    **(issue.get("details", {}) or {}),
                },
            ))

    return checks
def _decision_relevant_symbols(decision: Any) -> set[str]:
    symbols = {
        symbol
        for symbol, weight in decision.target_weights.items()
        if abs(weight * decision.exposure_target) > 1e-8
    }
    symbols.update(order.symbol for order in decision.orders)
    symbols.update(getattr(decision, "selected_symbols", []) or [])
    return symbols
def _exposure_checks(decision: Any, risk_config: dict[str, Any]) -> list[RiskCheckResult]:
    checks = []
    max_position_weight = risk_config.get("max_position_weight", 0.30)
    max_gross_exposure = risk_config.get("max_gross_exposure", 1.0)
    gross_exposure = sum(abs(weight) for weight in decision.target_weights.values())
    effective_gross_exposure = gross_exposure * decision.exposure_target

    for symbol, weight in decision.target_weights.items():
        effective_weight = abs(weight * decision.exposure_target)
        if effective_weight > max_position_weight:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="max_position_weight_exceeded",
                details={
                    "symbol": symbol,
                    "weight": effective_weight,
                    "limit": max_position_weight,
                },
            ))

    if effective_gross_exposure > max_gross_exposure:
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="max_gross_exposure_exceeded",
            details={
                "gross_exposure": effective_gross_exposure,
                "limit": max_gross_exposure,
            },
        ))

    return checks
def _order_checks(
    decision: Any,
    risk_config: dict[str, Any],
    broker_config: dict[str, Any],
) -> list[RiskCheckResult]:
    checks = []
    orders = decision.orders
    max_orders = risk_config.get("max_orders", 10)
    max_single_order_notional = risk_config.get("max_single_order_notional", 0.50)
    max_turnover = risk_config.get("max_turnover", 1.0)
    min_order_notional = broker_config.get("min_order_notional", 1.0)
    turnover = sum(abs(order.dollar_delta) for order in orders)
    turnover_fraction = turnover / decision.equity if decision.equity else 0

    if len(orders) > max_orders:
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="max_orders_exceeded",
            details={"orders": len(orders), "limit": max_orders},
        ))

    if turnover_fraction > max_turnover:
        checks.append(RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="max_turnover_exceeded",
            details={"turnover": turnover_fraction, "limit": max_turnover},
        ))

    for order in orders:
        notional_fraction = (
            abs(order.dollar_delta) / decision.equity
            if decision.equity
            else 0
        )
        if abs(order.dollar_delta) < min_order_notional:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.WARNING,
                reason="order_below_min_notional",
                details={
                    "symbol": order.symbol,
                    "notional": abs(order.dollar_delta),
                    "limit": min_order_notional,
                },
            ))
        if notional_fraction > max_single_order_notional:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="max_single_order_notional_exceeded",
                details={
                    "symbol": order.symbol,
                    "notional_fraction": notional_fraction,
                    "limit": max_single_order_notional,
                },
            ))

    return checks
def _cash_buffer_checks(
    decision: Any,
    portfolio_config: dict[str, Any],
) -> list[RiskCheckResult]:
    cash_buffer_percent = portfolio_config.get("cash_buffer_percent", 0.02)
    buy_notional = sum(
        order.dollar_delta
        for order in decision.orders
        if order.dollar_delta > 0
    )
    sell_notional = abs(sum(
        order.dollar_delta
        for order in decision.orders
        if order.dollar_delta < 0
    ))
    projected_cash = decision.cash - buy_notional + sell_notional
    min_cash = decision.equity * cash_buffer_percent

    if projected_cash < min_cash:
        return [RiskCheckResult(
            passed=False,
            severity=RiskSeverity.ERROR,
            reason="cash_buffer_breached",
            details={
                "projected_cash": projected_cash,
                "min_cash": min_cash,
                "cash_buffer_percent": cash_buffer_percent,
            },
        )]

    return []
def _broker_capability_checks(
    decision: Any,
    broker_config: dict[str, Any],
    paper_config: dict[str, Any],
) -> list[RiskCheckResult]:
    if paper_config.get("execution_adapter", "local_ledger") != "broker":
        return []

    checks = []
    supports_fractional = broker_config.get("supports_fractional", True)
    supports_market_orders = broker_config.get("supports_market_orders", True)
    supports_limit_orders = broker_config.get("supports_limit_orders", True)
    min_order_size = float(broker_config.get("min_order_notional", 1.0))

    for order in decision.orders:
        order_type = str(getattr(order, "order_type", "MARKET")).upper()
        quantity = abs(float(getattr(order, "quantity_delta", 0) or 0))
        notional = abs(float(getattr(order, "dollar_delta", 0) or 0))

        if order_type == "MARKET" and not supports_market_orders:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="broker_market_orders_unsupported",
                details={"symbol": order.symbol, "order_type": order_type},
            ))

        if order_type == "LIMIT" and not supports_limit_orders:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="broker_limit_orders_unsupported",
                details={"symbol": order.symbol, "order_type": order_type},
            ))

        if not supports_fractional and not float(quantity).is_integer():
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="broker_fractional_quantity_unsupported",
                details={"symbol": order.symbol, "quantity": quantity},
            ))

        if notional < min_order_size:
            checks.append(RiskCheckResult(
                passed=False,
                severity=RiskSeverity.ERROR,
                reason="broker_min_order_size_breached",
                details={
                    "symbol": order.symbol,
                    "notional": notional,
                    "min_order_size": min_order_size,
                },
            ))

    return checks
