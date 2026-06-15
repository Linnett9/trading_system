from dataclasses import dataclass
from typing import Optional

from core.entities.risk_context import RiskContext


class PositionSizer:
    def size(
        self,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        raise NotImplementedError

    def _validate_price(self, market_price: float):
        if market_price <= 0:
            raise ValueError("market_price must be greater than zero")


@dataclass(frozen=True)
class FixedFractionalSizer(PositionSizer):
    target_exposure: float = 0.10
    max_exposure: float = 1.0

    def size(
        self,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        self._validate_price(market_price)
        exposure = min(self.target_exposure, self.max_exposure)
        notional = account_equity * exposure
        return notional / market_price


@dataclass(frozen=True)
class FixedDollarSizer(PositionSizer):
    dollar_amount: float
    max_exposure: float = 1.0

    def size(
        self,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        self._validate_price(market_price)
        max_notional = account_equity * self.max_exposure
        notional = min(self.dollar_amount, max_notional)
        return notional / market_price


@dataclass(frozen=True)
class ATRPositionSizer(PositionSizer):
    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.20
    atr_multiplier: float = 2.0
    fallback_atr: float = 1.0

    def size(
        self,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        self._validate_price(market_price)

        atr_value = self.fallback_atr
        if risk_context is not None and risk_context.atr is not None:
            atr_value = risk_context.atr

        if atr_value <= 0:
            raise ValueError("atr_value must be greater than zero")

        risk_amount = account_equity * self.max_risk_per_trade
        stop_distance = atr_value * self.atr_multiplier
        quantity_by_risk = risk_amount / stop_distance

        max_notional = account_equity * self.max_exposure
        quantity_by_exposure = max_notional / market_price

        return min(quantity_by_risk, quantity_by_exposure)


@dataclass(frozen=True)
class VolatilitySizer(PositionSizer):
    target_exposure: float = 0.20
    max_exposure: float = 0.20
    fallback_volatility: float = 0.02

    def size(
        self,
        account_equity: float,
        market_price: float,
        risk_context: Optional[RiskContext] = None,
    ) -> float:
        self._validate_price(market_price)

        volatility = self.fallback_volatility
        if risk_context is not None and risk_context.volatility is not None:
            volatility = risk_context.volatility

        if volatility <= 0:
            raise ValueError("volatility must be greater than zero")

        exposure = min(self.target_exposure, self.max_exposure)
        adjusted_exposure = exposure / (1 + volatility * 10)
        notional = account_equity * adjusted_exposure

        return notional / market_price


def build_position_sizer(config: dict) -> PositionSizer:
    risk_config = config["risk"]
    sizing_config = config.get("position_sizing", {})
    mode = sizing_config.get("mode", "atr")
    max_exposure = sizing_config.get(
        "max_exposure",
        risk_config.get("max_exposure", 1.0),
    )

    if mode == "fixed_fractional":
        return FixedFractionalSizer(
            target_exposure=sizing_config.get("target_exposure", 0.10),
            max_exposure=max_exposure,
        )

    if mode == "fixed_dollar":
        return FixedDollarSizer(
            dollar_amount=sizing_config.get("dollar_amount", 50),
            max_exposure=max_exposure,
        )

    if mode == "volatility":
        return VolatilitySizer(
            target_exposure=sizing_config.get("target_exposure", 0.20),
            max_exposure=max_exposure,
            fallback_volatility=sizing_config.get(
                "fallback_volatility",
                0.02,
            ),
        )

    if mode == "atr":
        return ATRPositionSizer(
            max_risk_per_trade=risk_config["max_risk_per_trade"],
            max_exposure=max_exposure,
            atr_multiplier=risk_config["atr_multiplier"],
            fallback_atr=sizing_config.get("fallback_atr", 1.0),
        )

    raise ValueError(f"Unknown position sizing mode: {mode}")
