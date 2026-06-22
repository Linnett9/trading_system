from __future__ import annotations

from typing import Any

from core.entities.broker_capabilities import BrokerCapabilities
from core.interfaces.broker import IBroker
from infrastructure.brokers.alpaca_broker import AlpacaBroker
from infrastructure.brokers.fake_broker import FakeBroker


def broker_capabilities_from_config(config: dict[str, Any]) -> BrokerCapabilities:
    broker_config = config.get("broker", {})
    return BrokerCapabilities(
        supports_fractional_shares=bool(
            broker_config.get("supports_fractional", True)
        ),
        supports_market_orders=bool(
            broker_config.get("supports_market_orders", True)
        ),
        supports_limit_orders=bool(
            broker_config.get("supports_limit_orders", True)
        ),
        min_order_size=float(broker_config.get("min_order_notional", 1.0)),
        asset_class=str(broker_config.get("asset_class", "equity")),
        trading_hours=str(broker_config.get("trading_hours", "regular")),
    )


def build_broker(config: dict[str, Any], prices: dict[str, float] | None = None) -> IBroker:
    broker_config = config.get("broker", {})
    adapter = broker_config.get("adapter", "fake")

    if adapter == "fake":
        return FakeBroker(
            cash=float(broker_config.get("starting_cash", 500)),
            positions={
                symbol: float(quantity)
                for symbol, quantity in broker_config.get("positions", {}).items()
            },
            prices=prices or broker_config.get("prices", {}),
            open_orders=list(broker_config.get("open_orders", [])),
            reject_symbols=set(broker_config.get("reject_symbols", [])),
            partial_fill_ratio=float(broker_config.get("partial_fill_ratio", 1.0)),
            slippage_bps=float(
                broker_config.get(
                    "slippage_bps",
                    config.get("execution", {}).get("assumed_slippage_bps", 0.0),
                )
            ),
            commission_bps=float(
                broker_config.get(
                    "commission_bps",
                    config.get("execution", {}).get("commission_bps", 0.0),
                )
            ),
            capabilities=broker_capabilities_from_config(config),
        )

    if adapter == "alpaca":
        return AlpacaBroker(config=broker_config)

    raise RuntimeError(f"Unsupported broker adapter: {adapter}")
