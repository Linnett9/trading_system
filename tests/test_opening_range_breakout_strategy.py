from datetime import datetime

from core.entities.strategy_context import StrategyContext
from strategies.opening_range_breakout_strategy import OpeningRangeBreakoutStrategy


def _context(**overrides) -> StrategyContext:
    values = {
        "symbol": "AAPL",
        "timestamp": datetime(2026, 6, 22, 14, 5),
        "ema_fast": 101.0,
        "ema_slow": 100.0,
        "close": 102.0,
        "opening_range_high": 101.0,
        "opening_range_low": 99.0,
        "vwap": 100.5,
        "relative_volume": 2.0,
        "market_regime": "bull",
    }
    values.update(overrides)
    return StrategyContext(**values)


def test_breakout_strategy_emits_buy_for_confirmed_breakout():
    signal = OpeningRangeBreakoutStrategy("AAPL").generate_signal(_context())

    assert signal.action == "BUY"
    assert "Opening-range breakout" in signal.reason


def test_breakout_strategy_blocks_low_volume_entry():
    signal = OpeningRangeBreakoutStrategy("AAPL").generate_signal(
        _context(relative_volume=1.1)
    )

    assert signal.action == "HOLD"
    assert "Relative-volume" in signal.reason


def test_breakout_strategy_exits_long_when_vwap_is_lost():
    signal = OpeningRangeBreakoutStrategy("AAPL").generate_signal(
        _context(current_position="LONG", close=100.0)
    )

    assert signal.action == "SELL"
    assert signal.reason == "VWAP loss exit"


def test_breakout_strategy_holds_when_required_intraday_data_is_missing():
    signal = OpeningRangeBreakoutStrategy("AAPL").generate_signal(
        _context(vwap=None)
    )

    assert signal.action == "HOLD"
    assert "vwap" in signal.reason
