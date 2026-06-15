class BullMarketFilter:

    def passes(self, context) -> bool:
        return context.market_regime == "bull"


class VolatilityFilter:

    def __init__(self, allowed_regimes=None, max_percentile=None):
        self.allowed_regimes = allowed_regimes or {"low", "normal"}
        self.max_percentile = max_percentile

    def passes(self, context) -> bool:
        if (
            self.max_percentile is not None
            and context.volatility_percentile is not None
        ):
            return context.volatility_percentile <= self.max_percentile

        if context.volatility_regime == "unknown":
            return True

        return context.volatility_regime in self.allowed_regimes


class TrendStrengthFilter:

    def __init__(self, min_adx=None):
        self.min_adx = min_adx

    def passes(self, context) -> bool:
        if context.sma_200 is None or context.previous_sma_200 is None:
            return False

        if context.sma_200 <= context.previous_sma_200:
            return False

        if self.min_adx is not None:
            if context.adx is None:
                return False

            return context.adx >= self.min_adx

        return True


class VolumeConfirmationFilter:

    def __init__(self, min_relative_volume=1.0):
        self.min_relative_volume = min_relative_volume

    def passes(self, context) -> bool:
        if context.relative_volume is None:
            return True

        return context.relative_volume >= self.min_relative_volume
