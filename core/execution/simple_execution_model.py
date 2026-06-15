# core/execution/simple_execution_model.py

import random
from core.interfaces.execution_model import IExecutionModel
from core.entities.signal import Signal


class SimpleExecutionModel(IExecutionModel):

    def __init__(
        self,
        spread_bps: float = 2.0,     # 0.02% default spread
        slippage_bps: float = 1.0,   # random slippage
        seed: int | None = None
    ):
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self._random = random.Random(seed)

    def create_fill_price(self, signal: Signal, market_price: float) -> float:

        # convert basis points → multiplier
        spread = market_price * (self.spread_bps / 10_000)
        slippage = market_price * (
            self._random.uniform(0, self.slippage_bps) / 10_000
        )

        if signal.action == "BUY":
            # you pay ask → worse price
            return market_price + spread + slippage

        elif signal.action == "SELL":
            # you hit bid → worse price
            return market_price - spread - slippage

        return market_price
