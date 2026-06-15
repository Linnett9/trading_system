import math


TRADING_DAYS_PER_YEAR = 252


def total_return(starting_equity: float, final_equity: float) -> float:
    if starting_equity <= 0:
        return 0

    return (final_equity / starting_equity) - 1


def cagr(
    starting_equity: float,
    final_equity: float,
    elapsed_days: float,
) -> float:
    if starting_equity <= 0 or final_equity <= 0 or elapsed_days <= 0:
        return 0

    years = elapsed_days / 365.25
    if years <= 0:
        return 0

    return (final_equity / starting_equity) ** (1 / years) - 1


def sharpe_ratio(returns: list[float]) -> float:
    if not returns:
        return 0

    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    std = math.sqrt(variance)

    if std == 0:
        return 0

    return (mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0

    peak = values[0]
    max_dd = 0

    for value in values:
        peak = max(peak, value)
        if peak != 0:
            max_dd = max(max_dd, (peak - value) / peak)

    return max_dd


def profit_factor(pnls: list[float]) -> float:
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss == 0:
        return 0

    return gross_profit / gross_loss


def calmar_ratio(return_value: float, drawdown: float) -> float:
    if drawdown <= 0:
        return 0

    return return_value / drawdown


def trade_count_quality(closed_trades: int, target_trades: int) -> float:
    if target_trades <= 0:
        return 1

    return min(1, closed_trades / target_trades)


def composite_score(
    excess_return: float,
    sharpe: float,
    max_drawdown_value: float,
    profit_factor_value: float,
    closed_trades: int,
    target_trades: int,
    passed_folds: int = 0,
    total_folds: int = 0,
) -> float:
    consistency = (
        passed_folds / total_folds
        if total_folds
        else 0
    )
    capped_profit_factor = min(profit_factor_value, 3.0) / 3.0

    return (
        excess_return * 0.35
        + sharpe * 0.25
        - max_drawdown_value * 0.20
        + capped_profit_factor * 0.10
        + trade_count_quality(closed_trades, target_trades) * 0.10
        + consistency * 0.10
    )
