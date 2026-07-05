import json
from pathlib import Path


def dual_momentum_diagnosis(result, candles_by_symbol=None):
    annual = {}
    monthly = []
    prices_by_symbol = prices_by_symbol_from_candles(candles_by_symbol or {})
    selection_by_month = {
        selection.timestamp.strftime("%Y-%m"): selection
        for selection in result.selections
    }
    months_by_year = {}

    for month, bot_return in result.monthly_returns.items():
        year = int(month.split("-")[0])
        months_by_year.setdefault(year, []).append(month)
        selection = selection_by_month.get(month)

        monthly.append({
            "month": month,
            "bot_return": bot_return,
            "benchmark_returns": benchmark_returns(
                month,
                prices_by_symbol,
            ),
            "selected_asset_returns": selected_asset_returns(
                month,
                selection,
                prices_by_symbol,
            ),
            "entry_momentum": entry_momentum_snapshot(
                selection,
                prices_by_symbol,
            ),
            "benchmark_relative": benchmark_relative_snapshot(
                month,
                bot_return,
                selection,
                prices_by_symbol,
            ),
            "regime": (
                selection.regime_label
                if selection is not None
                else "unknown"
            ),
            "exposure_target": (
                selection.exposure_target
                if selection is not None
                else 0
            ),
            "selected_symbols": (
                selection.symbols
                if selection is not None
                else []
            ),
            "fallback_symbols": (
                selection.fallback_symbols or []
                if selection is not None
                else []
            ),
            "asset_contributions": (
                monthly_asset_contributions(
                    month,
                    selection,
                    prices_by_symbol,
                )
                if selection is not None
                else {}
            ),
            "missed_winners": monthly_missed_winners(
                month,
                selection,
                prices_by_symbol,
            ),
        })

    for year, bot_return in result.annual_returns.items():
        months = months_by_year.get(year, [])

        selections = [
            selection_by_month[month]
            for month in months
            if month in selection_by_month
        ]

        regime_counts = {}
        for selection in selections:
            regime_counts[selection.regime_label] = (
                regime_counts.get(selection.regime_label, 0) + 1
            )

        year_monthly_items = [
            item for item in monthly
            if item["month"].startswith(str(year))
        ]

        annual[year] = {
            "bot_return": bot_return,
            "benchmark_return": None,
            "equal_weight_return": None,
            "average_exposure_target": (
                sum(selection.exposure_target for selection in selections)
                / len(selections)
                if selections
                else 0
            ),
            "cash_months": regime_counts.get("cash", 0),
            "partial_risk_months": regime_counts.get("partial-risk", 0),
            "fast_reentry_months": regime_counts.get("fast-reentry", 0),
            "risk_on_months": regime_counts.get("risk-on", 0),
            "defensive_months": regime_counts.get("defensive", 0),
            "top_selected_symbols": top_selected_symbols(selections),
            "worst_months": sorted(
                year_monthly_items,
                key=lambda item: item["bot_return"],
            )[:3],
            "top_contributors": aggregate_contributors(
                year_monthly_items,
                reverse=True,
            ),
            "worst_contributors": aggregate_contributors(
                year_monthly_items,
                reverse=False,
            ),
            "benchmark_relative": aggregate_benchmark_relative(
                year_monthly_items,
            ),
            "missed_winners": aggregate_missed_winners(year_monthly_items),
        }

    return {
        "annual": annual,
        "monthly": monthly,
        "config": result.config,
        "summary": {
            "return": result.result.total_return,
            "benchmark_return": result.benchmark_return,
            "equal_weight_return": result.equal_weight_return,
            "excess_return": result.excess_return,
            "excess_vs_equal_weight": result.excess_vs_equal_weight,
            "sharpe": result.result.sharpe,
            "max_drawdown": result.result.max_drawdown,
            "annualized_turnover_percent": (
                result.annualized_turnover_percent
            ),
        },
    }


def prices_by_symbol_from_candles(candles_by_symbol):
    return {
        symbol: {
            candle.timestamp: candle.close
            for candle in candles
        }
        for symbol, candles in candles_by_symbol.items()
    }


def monthly_asset_contributions(month, selection, prices_by_symbol):
    contributions = {}

    if selection is None:
        return contributions

    for symbol, weight in (selection.target_weights or {}).items():
        prices = prices_by_symbol.get(symbol, {})

        month_prices = [
            (timestamp, price)
            for timestamp, price in sorted(prices.items())
            if timestamp.strftime("%Y-%m") == month
        ]

        if len(month_prices) < 2:
            continue

        start = month_prices[0][1]
        end = month_prices[-1][1]

        if start:
            contributions[symbol] = (
                selection.exposure_target
                * weight
                * ((end / start) - 1)
            )

    return contributions


def benchmark_returns(month, prices_by_symbol, symbols=("SPY", "QQQ")):
    returns = monthly_asset_returns(month, prices_by_symbol)
    return {
        symbol: returns[symbol]
        for symbol in symbols
        if symbol in returns
    }


def selected_asset_returns(month, selection, prices_by_symbol):
    if selection is None:
        return {}

    returns = monthly_asset_returns(month, prices_by_symbol)
    return {
        symbol: returns[symbol]
        for symbol in selection.symbols
        if symbol in returns
    }


def benchmark_relative_snapshot(
    month,
    bot_return,
    selection,
    prices_by_symbol,
):
    selected_returns = selected_asset_returns(
        month,
        selection,
        prices_by_symbol,
    )
    benchmarks = benchmark_returns(month, prices_by_symbol)
    spy_return = benchmarks.get("SPY")
    qqq_return = benchmarks.get("QQQ")
    selected_average = (
        sum(selected_returns.values()) / len(selected_returns)
        if selected_returns
        else 0
    )

    return {
        "selected_average_return": selected_average,
        "bot_vs_selected_average": bot_return - selected_average,
        "bot_vs_spy": (
            bot_return - spy_return
            if spy_return is not None
            else None
        ),
        "bot_vs_qqq": (
            bot_return - qqq_return
            if qqq_return is not None
            else None
        ),
        "selected_average_vs_spy": (
            selected_average - spy_return
            if spy_return is not None
            else None
        ),
        "selected_average_vs_qqq": (
            selected_average - qqq_return
            if qqq_return is not None
            else None
        ),
        "spy_return": spy_return,
        "qqq_return": qqq_return,
    }


def entry_momentum_snapshot(
    selection,
    prices_by_symbol,
    periods=(21, 63, 126),
    benchmarks=("SPY", "QQQ"),
):
    if selection is None:
        return {}

    symbols = list(selection.symbols)
    for benchmark in benchmarks:
        if benchmark not in symbols:
            symbols.append(benchmark)

    snapshot = {}

    for symbol in symbols:
        prices = prices_by_symbol.get(symbol)
        if not prices:
            continue

        timestamps = sorted(prices)
        index = timestamp_index_at_or_before(timestamps, selection.timestamp)

        if index is None:
            continue

        snapshot[symbol] = {
            str(period): period_return(prices, timestamps, index, period)
            for period in periods
        }

    return snapshot


def timestamp_index_at_or_before(timestamps, timestamp):
    index = None

    for position, item in enumerate(timestamps):
        if item <= timestamp:
            index = position
            continue

        break

    return index


def period_return(prices, timestamps, index, period):
    if index is None or index < period:
        return None

    current = prices[timestamps[index]]
    previous = prices[timestamps[index - period]]

    if previous <= 0:
        return None

    return (current / previous) - 1


def aggregate_contributors(monthly_items, reverse=True, limit=5):
    totals = {}

    for item in monthly_items:
        for symbol, contribution in item["asset_contributions"].items():
            totals[symbol] = totals.get(symbol, 0) + contribution

    return [
        {
            "symbol": symbol,
            "contribution": contribution,
        }
        for symbol, contribution in sorted(
            totals.items(),
            key=lambda item: item[1],
            reverse=reverse,
        )[:limit]
    ]


def monthly_missed_winners(month, selection, prices_by_symbol, limit=5):
    selected = set(selection.symbols if selection is not None else [])
    returns = monthly_asset_returns(month, prices_by_symbol)

    return [
        {
            "symbol": symbol,
            "return": value,
        }
        for symbol, value in sorted(
            returns.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if symbol not in selected
    ][:limit]


def monthly_asset_returns(month, prices_by_symbol):
    returns = {}

    for symbol, prices in prices_by_symbol.items():
        month_prices = [
            (timestamp, price)
            for timestamp, price in sorted(prices.items())
            if timestamp.strftime("%Y-%m") == month
        ]

        if len(month_prices) < 2:
            continue

        start = month_prices[0][1]
        end = month_prices[-1][1]

        if start:
            returns[symbol] = (end / start) - 1

    return returns


def aggregate_missed_winners(monthly_items, limit=5):
    totals = {}
    counts = {}

    for item in monthly_items:
        for winner in item["missed_winners"]:
            symbol = winner["symbol"]
            totals[symbol] = totals.get(symbol, 0) + winner["return"]
            counts[symbol] = counts.get(symbol, 0) + 1

    return [
        {
            "symbol": symbol,
            "average_return": totals[symbol] / counts[symbol],
            "months": counts[symbol],
        }
        for symbol in sorted(
            totals,
            key=lambda item: totals[item] / counts[item],
            reverse=True,
        )[:limit]
    ]


def aggregate_benchmark_relative(monthly_items):
    fields = (
        "bot_vs_spy",
        "bot_vs_qqq",
        "selected_average_vs_spy",
        "selected_average_vs_qqq",
    )
    summary = {}

    for field in fields:
        values = [
            item["benchmark_relative"].get(field)
            for item in monthly_items
            if item["benchmark_relative"].get(field) is not None
        ]
        summary[field] = (
            sum(values) / len(values)
            if values
            else None
        )

    return summary


def top_selected_symbols(selections, limit=5):
    counts = {}

    for selection in selections:
        for symbol in selection.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1

    return [
        symbol
        for symbol, _ in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


def save_dual_momentum_diagnosis(
    diagnosis,
    report_dir,
    filename="dual_momentum_diagnosis.json",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename
    path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")

    return path
