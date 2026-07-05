


class DualMomentumRankingPrimitivesMixin:
    def _period_return(self, prices, timestamps, index, period):
        if period <= 0:
            return None

        if index is None or index < period:
            return None

        current = prices[timestamps[index]]
        previous = prices[timestamps[index - period]]

        if previous <= 0:
            return None

        return (current / previous) - 1
    def _momentum_score(self, prices, timestamps, index, periods=None):
        scores = []
        periods = periods or self.momentum_periods

        for period in periods:
            if index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]

            if previous <= 0:
                return None

            scores.append((current / previous) - 1)

        return sum(scores) / len(scores) if scores else None
    def _weighted_momentum_score(
        self,
        prices,
        timestamps,
        index,
        periods,
        weights,
    ):
        if not periods:
            return None

        if len(weights) != len(periods):
            weights = [1 / len(periods)] * len(periods)

        weighted_score = 0
        total_weight = 0

        for period, weight in zip(periods, weights):
            if index is None or index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]

            if previous <= 0:
                return None

            weighted_score += weight * ((current / previous) - 1)
            total_weight += weight

        return weighted_score / total_weight if total_weight else None
