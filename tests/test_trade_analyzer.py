from datetime import datetime, timedelta

from core.entities.trade import Trade
from core.research.trade_analyzer import TradeAnalyzer


def make_trade(entry_day, exit_day, pnl):
    start = datetime(2024, 1, 1)

    return Trade(
        symbol="AAPL",
        side="LONG",
        entry_price=100,
        entry_time=start + timedelta(days=entry_day),
        exit_price=100 + pnl,
        exit_time=start + timedelta(days=exit_day),
        quantity=1,
        pnl=pnl,
        is_open=False,
    )


def test_trade_analyzer_calculates_win_loss_metrics():
    analysis = TradeAnalyzer().analyze(
        [
            make_trade(0, 2, 10),
            make_trade(3, 4, -5),
            make_trade(5, 8, 20),
        ],
        period_start=datetime(2024, 1, 1),
        period_end=datetime(2024, 1, 11),
    )

    assert analysis.total_trades == 3
    assert analysis.win_rate == 2 / 3
    assert analysis.average_win == 15
    assert analysis.average_loss == -5
    assert analysis.largest_win == 20
    assert analysis.largest_loss == -5
    assert analysis.expectancy == 25 / 3
    assert analysis.profit_factor == 6


def test_trade_analyzer_calculates_duration_and_time_in_market():
    analysis = TradeAnalyzer().analyze(
        [
            make_trade(0, 2, 10),
            make_trade(3, 4, -5),
            make_trade(5, 8, 20),
        ],
        period_start=datetime(2024, 1, 1),
        period_end=datetime(2024, 1, 11),
    )

    assert analysis.average_trade_duration_days == 2
    assert analysis.median_trade_duration_days == 2
    assert analysis.max_trade_duration_days == 3
    assert analysis.time_in_market_percent == 0.6


def test_trade_analyzer_handles_no_closed_trades():
    analysis = TradeAnalyzer().analyze([])

    assert analysis.total_trades == 0
    assert analysis.time_in_market_percent == 0
