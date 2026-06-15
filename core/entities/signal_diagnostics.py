from dataclasses import dataclass


@dataclass
class SignalDiagnostics:
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    duplicate_buy_skips: int = 0
    flat_sell_skips: int = 0
    risk_blocked_signals: int = 0
    stop_loss_exits: int = 0
    take_profit_exits: int = 0

    def record_signal(self, action: str):
        if action == "BUY":
            self.buy_signals += 1
        elif action == "SELL":
            self.sell_signals += 1
        elif action == "HOLD":
            self.hold_signals += 1

    def to_dict(self) -> dict:
        return {
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "duplicate_buy_skips": self.duplicate_buy_skips,
            "flat_sell_skips": self.flat_sell_skips,
            "risk_blocked_signals": self.risk_blocked_signals,
            "stop_loss_exits": self.stop_loss_exits,
            "take_profit_exits": self.take_profit_exits,
        }
