# core/risk/atr_risk_manager.py

from typing import Optional

from core.entities.risk_context import RiskContext
from core.entities.signal import Signal
from core.risk.position_sizer import PositionSizer
from core.risk.simple_risk_manager import SimpleRiskManager


class ATRRiskManager(SimpleRiskManager):

    def __init__(
        self,
        max_risk_per_trade: float = 0.0025,  # 0.25%
        max_exposure: float = 0.02,          # 2%
        atr_multiplier: float = 2.0,
        atr_value: float = 1.0,
        position_sizer: PositionSizer | None = None,
    ):
        super().__init__(
            max_risk_per_trade=max_risk_per_trade,
            max_exposure=max_exposure,
            position_sizer=position_sizer,
        )

        self.atr_multiplier = atr_multiplier
        self.atr_value = atr_value

    def position_size(
        self,
        signal: Signal,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:

        if market_price <= 0:
            raise ValueError("market_price must be greater than zero")

        if self.position_sizer is not None:
            return self.position_sizer.size(
                account_equity=account_equity,
                market_price=market_price,
                risk_context=risk_context,
            )

        atr_value = self.atr_value
        if risk_context is not None and risk_context.atr is not None:
            atr_value = risk_context.atr

        if atr_value <= 0:
            raise ValueError("atr_value must be greater than zero")

        risk_amount = account_equity * self.max_risk_per_trade
        stop_distance = atr_value * self.atr_multiplier

        quantity_by_risk = risk_amount / stop_distance

        max_notional = account_equity * self.max_exposure
        quantity_by_exposure = max_notional / market_price

        return min(
            quantity_by_risk,
            quantity_by_exposure,
        )
