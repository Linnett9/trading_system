from core.entities.capital_utilization import CapitalUtilization
from core.entities.trade_analysis import TradeAnalysis


def prices_by_symbol(candles_by_symbol):
    return {
        symbol: {
            candle.timestamp: candle.close
            for candle in candles
        }
        for symbol, candles in candles_by_symbol.items()
        if candles
    }


def common_timestamps(prices_by_symbol, lookback=0):
    if not prices_by_symbol:
        return []

    common = set.intersection(
        *[
            set(prices.keys())
            for prices in prices_by_symbol.values()
        ]
    )
    return sorted(common)[lookback:]


def prices_at(prices_by_symbol, timestamp):
    return {
        symbol: prices[timestamp]
        for symbol, prices in prices_by_symbol.items()
        if timestamp in prices
    }


def timestamp_index(timestamps, timestamp):
    try:
        return timestamps.index(timestamp)
    except ValueError:
        return None


def rebalance_key(timestamp, frequency):
    if frequency == "daily":
        return timestamp.year, timestamp.month, timestamp.day

    if frequency == "weekly":
        calendar = timestamp.isocalendar()
        return calendar.year, calendar.week

    if frequency == "biweekly":
        calendar = timestamp.isocalendar()
        return calendar.year, (calendar.week - 1) // 2

    if frequency == "quarterly":
        quarter = ((timestamp.month - 1) // 3) + 1
        return timestamp.year, quarter

    if frequency == "bimonthly":
        period = ((timestamp.month - 1) // 2) + 1
        return timestamp.year, period

    if frequency == "annual":
        return (timestamp.year,)

    return timestamp.year, timestamp.month


def should_rebalance(timestamp, last_rebalance_key, frequency):
    return rebalance_key(timestamp, frequency) != last_rebalance_key


def position_value(positions, prices):
    return sum(
        quantity * prices[symbol]
        for symbol, quantity in positions.items()
        if symbol in prices
    )


def equity(cash, positions, prices):
    return cash + position_value(positions, prices)


def profit_factor(pnls):
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl <= 0))
    return gross_profit / gross_loss if gross_loss else 0


def trade_analysis(pnls, exposure_values):
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return TradeAnalysis(
        total_trades=len(pnls),
        win_rate=len(wins) / len(pnls) if pnls else 0,
        average_win=gross_profit / len(wins) if wins else 0,
        average_loss=sum(losses) / len(losses) if losses else 0,
        largest_win=max(wins) if wins else 0,
        largest_loss=min(losses) if losses else 0,
        expectancy=sum(pnls) / len(pnls) if pnls else 0,
        profit_factor=gross_profit / gross_loss if gross_loss else 0,
        time_in_market_percent=(
            sum(1 for exposure in exposure_values if exposure > 0)
            / len(exposure_values)
            if exposure_values
            else 0
        ),
    )


def capital_utilization(exposure_values, position_values):
    average_exposure = (
        sum(exposure_values) / len(exposure_values)
        if exposure_values
        else 0
    )
    return CapitalUtilization(
        average_position_value=(
            sum(position_values) / len(position_values)
            if position_values
            else 0
        ),
        average_exposure_percent=average_exposure,
        max_exposure_percent=max(exposure_values) if exposure_values else 0,
        average_cash_percent=1 - average_exposure,
        average_leverage=average_exposure,
    )


def equal_weight_return(prices_by_symbol, timestamps, excluded_symbols=None):
    if not timestamps:
        return 0

    excluded_symbols = set(excluded_symbols or [])
    returns = []

    for symbol, prices in prices_by_symbol.items():
        if symbol in excluded_symbols:
            continue

        start = prices.get(timestamps[0])
        end = prices.get(timestamps[-1])
        if start:
            returns.append((end / start) - 1)

    return sum(returns) / len(returns) if returns else 0


def benchmark_return(
    prices_by_symbol,
    timestamps,
    benchmark_symbol,
    excluded_equal_weight_symbols=None,
):
    if not timestamps:
        return 0

    prices = prices_by_symbol.get(benchmark_symbol)
    if not prices:
        return equal_weight_return(
            prices_by_symbol,
            timestamps,
            excluded_symbols=excluded_equal_weight_symbols,
        )

    start = prices.get(timestamps[0])
    end = prices.get(timestamps[-1])
    return (end / start) - 1 if start else 0
