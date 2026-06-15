# core/engine/execution_engine.py

from core.entities.fill import Fill


class ExecutionEngine:

    def __init__(self, broker, trade_manager, execution_model):
        self.broker = broker
        self.trade_manager = trade_manager
        self.execution_model = execution_model

    def execute(self, signal, size, market_price):

        # 1. compute realistic fill price
        fill_price = self.execution_model.create_fill_price(signal, market_price)

        # 2. create fill
        fill_quantity = size
        if signal.action == "SELL":
            fill_quantity = -size

        fill = Fill(
            symbol=signal.symbol,
            price=fill_price,
            quantity=fill_quantity,
            timestamp=signal.timestamp
        )

        # 3. send to trade manager
        if signal.action == "BUY":
            return self.trade_manager.open_trade(fill)

        elif signal.action == "SELL":
            return self.trade_manager.close_trade(
                fill,
                exit_reason=signal.reason,
            )

        return None
