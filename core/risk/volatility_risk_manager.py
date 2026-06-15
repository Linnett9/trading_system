# core/risk/volatility_risk_manager.py

from typing import Optional

from core.entities.risk_context import RiskContext
from core.entities.signal import Signal
from core.risk.position_sizer import PositionSizer
from core.risk.simple_risk_manager import SimpleRiskManager


class VolatilityRiskManager(SimpleRiskManager):

    def __init__(
        self,
        max_risk_per_trade: float = 0.0025,  # 0.25%
        max_exposure: float = 0.02,          # 2%
        volatility: float = 0.02,
        position_sizer: PositionSizer | None = None,
    ):
        super().__init__(
            max_risk_per_trade=max_risk_per_trade,
            max_exposure=max_exposure,
            position_sizer=position_sizer,
        )

        self.volatility = volatility

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

        volatility = self.volatility
        if risk_context is not None and risk_context.volatility is not None:
            volatility = risk_context.volatility

        if volatility <= 0:
            raise ValueError("volatility must be greater than zero")

        base_notional = account_equity * self.max_exposure

        volatility_adjusted_notional = base_notional / (1 + volatility * 10)

        max_risk_notional = account_equity * self.max_risk_per_trade

        notional = min(
            volatility_adjusted_notional,
            max_risk_notional,
        )

        return notional / market_price
