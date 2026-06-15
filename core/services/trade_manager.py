# core/services/trade_manager.py

from typing import Dict, Optional

from core.entities.trade import Trade
from core.entities.fill import Fill


class TradeManager:

    def __init__(self):
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []

    # -------------------------
    # OPEN TRADE
    # -------------------------
    def open_trade(self, fill: Fill) -> Optional[Trade]:

        # Prevent duplicate position
        existing = self.open_trades.get(fill.symbol)

        if existing is not None:
            return existing

        trade = Trade(
            symbol=fill.symbol,
            side="LONG" if fill.quantity > 0 else "SHORT",
            entry_price=fill.price,
            entry_time=fill.timestamp,
            quantity=abs(fill.quantity),
            is_open=True,
        )

        self.open_trades[fill.symbol] = trade

        return trade

    # -------------------------
    # CLOSE TRADE
    # -------------------------
    def close_trade(
        self,
        fill: Fill,
        exit_reason: str = ""
    ) -> Optional[Trade]:

        trade = self.open_trades.get(fill.symbol)

        if trade is None:
            return None

        trade.exit_price = fill.price
        trade.exit_time = fill.timestamp
        trade.exit_reason = exit_reason
        trade.is_open = False

        # PnL calculation
        if trade.side == "LONG":
            trade.pnl = (
                (fill.price - trade.entry_price)
                * trade.quantity
            )
        else:
            trade.pnl = (
                (trade.entry_price - fill.price)
                * trade.quantity
            )

        self.closed_trades.append(trade)

        del self.open_trades[fill.symbol]

        return trade

    # -------------------------
    # STATE HELPERS
    # -------------------------
    def get_open_trade(
        self,
        symbol: str
    ) -> Optional[Trade]:
        return self.open_trades.get(symbol)

    def has_position(
        self,
        symbol: str
    ) -> bool:
        return symbol in self.open_trades

    def get_all_positions(
        self
    ) -> Dict[str, Trade]:
        return self.open_trades

    def get_position(
        self,
        symbol: str
    ) -> Optional[str]:

        trade = self.open_trades.get(symbol)

        if trade is None:
            return None

        return trade.side  # LONG or SHORT

    def is_flat(
        self,
        symbol: str
    ) -> bool:

        return symbol not in self.open_trades
