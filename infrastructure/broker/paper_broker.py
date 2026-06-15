# infrastructure/broker/paper_broker.py

from datetime import datetime
from core.interfaces.broker import IBroker
from core.entities.order import Order
from core.entities.fill import Fill


class PaperBroker(IBroker):

    def __init__(self, price_feed):
        self.price_feed = price_feed  # MarketDataService

    def submit_order(self, order: Order) -> Fill:

        # simulate execution at last price
        price = self.price_feed.latest().close

        print(f"📤 ORDER SENT: {order.side} {order.quantity} {order.symbol} @ {price}")

        return Fill(
            symbol=order.symbol,
            quantity=order.quantity if order.side == "BUY" else -order.quantity,
            price=price,
            timestamp=datetime.utcnow()
        )