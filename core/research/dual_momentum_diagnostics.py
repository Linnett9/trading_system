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