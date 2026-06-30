from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    side: str
    quantity_delta: float
    dollar_delta: float
    current_weight: float
    target_weight: float
    drift_weight: float
    price: float
    reason: str
    score: float | None = None
    order_type: str = "MARKET"
    limit_price: float | None = None

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity_delta": self.quantity_delta,
            "dollar_delta": self.dollar_delta,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "drift_weight": self.drift_weight,
            "price": self.price,
            "reason": self.reason,
            "score": self.score,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
        }


@dataclass(frozen=True)
class PaperDecision:
    timestamp: object
    regime_label: str
    risk_on: bool
    exposure_target: float
    target_weights: dict[str, float]
    current_positions: dict[str, float]
    cash: float
    equity: float
    orders: list[PaperOrder]
    selected_symbols: list[str]
    model_context: dict
    data_freshness: dict
    data_quality: dict
    report_path: Path
    state_path: Path

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime_label": self.regime_label,
            "risk_on": self.risk_on,
            "exposure_target": self.exposure_target,
            "target_weights": self.target_weights,
            "current_positions": self.current_positions,
            "cash": self.cash,
            "equity": self.equity,
            "orders": [order.to_dict() for order in self.orders],
            "selected_symbols": self.selected_symbols,
            "model_context": self.model_context,
            "data_freshness": self.data_freshness,
            "data_quality": self.data_quality,
            "state_path": str(self.state_path),
        }
