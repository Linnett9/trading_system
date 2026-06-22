from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerCapabilities:
    supports_fractional_shares: bool = True
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    min_order_size: float = 1.0
    asset_class: str = "equity"
    trading_hours: str = "regular"

    def to_dict(self) -> dict[str, bool | float | str]:
        return {
            "supports_fractional_shares": self.supports_fractional_shares,
            "supports_market_orders": self.supports_market_orders,
            "supports_limit_orders": self.supports_limit_orders,
            "min_order_size": self.min_order_size,
            "asset_class": self.asset_class,
            "trading_hours": self.trading_hours,
        }
