from __future__ import annotations

from datetime import datetime
from typing import Any

from application.services.broker_factory import build_broker
from application.services.paper_service import build_paper_engine
from core.entities.broker_capabilities import BrokerCapabilities
from core.entities.order import Order


def decision_prices(decision: Any) -> dict[str, float]:
    return {
        order.symbol: float(order.price)
        for order in decision.orders
        if order.price is not None
    }


def build_broker_for_decision(config: dict[str, Any], decision: Any) -> Any:
    return build_broker(config, prices=decision_prices(decision))


def fill_to_dict(fill: Any) -> dict[str, Any]:
    return {
        "symbol": fill.symbol,
        "quantity": fill.quantity,
        "price": fill.price,
        "timestamp": fill.timestamp.isoformat(),
        "fees": fill.fees,
    }


def broker_blocked_reason(config: dict[str, Any], decision: Any) -> str | None:
    paper_config = config.get("paper_trading", {})

    if paper_config.get("execution_adapter", "local_ledger") != "broker":
        return None

    broker = build_broker_for_decision(config, decision)

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


def broker_reconciliation(
    config: dict[str, Any],
    decision: Any,
) -> dict[str, Any] | None:
    paper_config = config.get("paper_trading", {})

    if paper_config.get("execution_adapter", "local_ledger") != "broker":
        return None

    broker_config = config.get("broker", {})
    broker = build_broker_for_decision(config, decision)
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
        fill_to_dict(fill)
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


def submit_with_broker(config: dict[str, Any], decision: Any) -> dict[str, Any]:
    broker = build_broker_for_decision(config, decision)
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
        "broker_adapter": config.get("broker", {}).get("adapter", "fake"),
        "open_orders_after": open_orders_after,
    }

    engine = build_paper_engine(config)
    return engine.apply_external_fill_record(decision.report_path, fill_record)
