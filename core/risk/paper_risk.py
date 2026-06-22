from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    severity: RiskSeverity
    reason: str
    details: dict[str, Any]


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
    from core.research.ml.sector_reference import load_sector_by_symbol

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


def risk_status(checks: list[RiskCheckResult]) -> str:
    if any(check.severity == RiskSeverity.CRITICAL for check in checks):
        return "CRITICAL"
    if any(check.severity == RiskSeverity.ERROR for check in checks):
        return "ERROR"
    if any(check.severity == RiskSeverity.WARNING for check in checks):
        return "WARNING"
    return "PASS"


def risk_blocks_submission(checks: list[RiskCheckResult]) -> bool:
    return any(
        check.severity in {RiskSeverity.ERROR, RiskSeverity.CRITICAL}
        for check in checks
    )


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


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


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
