from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRegime:
    market_regime: str = "unknown"
    volatility_regime: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
        }


class MarketRegimeAnalyzer:

    def classify(
        self,
        close,
        sma_200,
        previous_sma_200,
        volatility,
        volatility_average,
    ) -> MarketRegime:
        return MarketRegime(
            market_regime=self._market_regime(
                close,
                sma_200,
                previous_sma_200,
            ),
            volatility_regime=self._volatility_regime(
                volatility,
                volatility_average,
            ),
        )

    def _market_regime(self, close, sma_200, previous_sma_200):
        if close is None or sma_200 is None or previous_sma_200 is None:
            return "unknown"

        if close > sma_200 and sma_200 > previous_sma_200:
            return "bull"

        if close < sma_200 and sma_200 < previous_sma_200:
            return "bear"

        return "sideways"

    def _volatility_regime(self, volatility, volatility_average):
        if volatility is None or volatility_average is None:
            return "unknown"

        if volatility > volatility_average * 1.25:
            return "high"

        if volatility < volatility_average * 0.75:
            return "low"

        return "normal"
