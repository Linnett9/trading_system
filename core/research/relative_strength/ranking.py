class RelativeStrengthRankingMixin:

    def _rank_symbols(self, timestamp, prices_by_symbol):
        ranked = []

        for symbol, prices in prices_by_symbol.items():
            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)
            if index is None:
                continue

            if not self._above_sma_filter(prices, timestamps, index):
                continue

            score = self._momentum_score(prices, timestamps, index)
            if score is not None:
                ranked.append((symbol, score))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _above_sma_filter(self, prices, timestamps, index):
        if index < self.sma_period:
            return False

        close = prices[timestamps[index]]
        sma = sum(
            prices[timestamps[position]]
            for position in range(index - self.sma_period + 1, index + 1)
        ) / self.sma_period
        previous_sma = sum(
            prices[timestamps[position]]
            for position in range(index - self.sma_period, index)
        ) / self.sma_period

        return close > sma and sma >= previous_sma

    def _momentum_score(self, prices, timestamps, index):
        scores = []

        for period in self.momentum_periods:
            if index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]
            if previous <= 0:
                return None

            scores.append((current / previous) - 1)

        return sum(scores) / len(scores) if scores else None
