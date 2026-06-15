class BullMarketFilter:

    def passes(self, context) -> bool:
        return context.market_regime == "bull"


class VolatilityFilter:

    def __init__(self, allowed_regimes=None):
        self.allowed_regimes = allowed_regimes or {"low", "normal"}

    def passes(self, context) -> bool:
        if context.volatility_regime == "unknown":
            return True

        return context.volatility_regime in self.allowed_regimes


class TrendStrengthFilter:

    def passes(self, context) -> bool:
        if context.sma_200 is None or context.previous_sma_200 is None:
            return False

        return context.sma_200 > context.previous_sma_200
