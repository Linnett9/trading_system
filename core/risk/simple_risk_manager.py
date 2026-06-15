# core/risk/simple_risk_manager.py

from typing import Optional

from core.interfaces.risk_manager import IRiskManager
from core.entities.signal import Signal
from core.entities.portfolio import Portfolio
from core.entities.risk_context import RiskContext
from core.risk.position_sizer import PositionSizer


class SimpleRiskManager(IRiskManager):

    def __init__(
        self,
        max_risk_per_trade: float = 0.0025,   # 0.25% of equity per trade
        max_exposure: float = 0.002,          # max 0.2% capital exposure
        position_sizer: PositionSizer | None = None,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_exposure = max_exposure
        self.position_sizer = position_sizer

    # -------------------------------------------------
    # SIGNAL VALIDATION
    # -------------------------------------------------
    def validate(
        self,
        signal: Signal,
        portfolio: Optional[Portfolio] = None
    ) -> bool:

        # 1. Ignore neutral signals
        if signal.action == "HOLD":
            return False

        # 2. If no portfolio provided, allow (backtest-safe fallback)
        if portfolio is None:
            return True

        position = portfolio.positions.get(signal.symbol)

        # 3. Block duplicate entries (already in position)
        if position:

            # already long
            if position.quantity > 0 and signal.action == "BUY":
                return False

            # already short
            if position.quantity < 0 and signal.action == "SELL":
                return False

        return True

    # -------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------
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

        risk_notional = account_equity * self.max_risk_per_trade
        max_notional = account_equity * self.max_exposure

        notional = min(risk_notional, max_notional)
        quantity = notional / market_price

        return quantity
