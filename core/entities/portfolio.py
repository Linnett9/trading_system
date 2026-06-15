# core/entities/portfolio.py

from typing import Dict
from core.entities.position import Position


class Portfolio:

    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def update(self, fill):

        pos = self.positions.get(fill.symbol)

        if pos is None:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                avg_price=fill.price
            )
            return

        new_qty = pos.quantity + fill.quantity

        if new_qty == 0:
            del self.positions[fill.symbol]
            return

        pos.avg_price = (
            (pos.avg_price * pos.quantity + fill.price * fill.quantity)
            / new_qty
        )

        pos.quantity = new_qty