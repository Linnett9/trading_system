import csv
from pathlib import Path

from application.services.market_data_loader import (
    data_cache_path,
    load_candles,
)
from application.reporting.walk_forward_reporter import (
    print_walk_forward_summary,
)
from core.research.backtest_runner import run_backtest
from core.research.parameter_optimizer import ParameterOptimizer
from core.research.strategy_comparison import StrategyComparison
from core.research.walk_forward import WalkForwardTester


def run_data_audit(config, feed):
    dual_config = config["research"].get("dual_momentum", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])
    cache_config = config.get("cache", {})
    backtest_config = config["backtest"]

    rows = []
    timestamps_by_symbol = {}

    for symbol in symbols:
        cache_path = data_cache_path(symbol, backtest_config, cache_config)
        cache_hit = (
            bool(cache_config.get("enabled", False))
            and cache_path is not None
            and cache_path.exists()
        )
        candles = load_candles(symbol, config, feed)
        timestamps = [candle.timestamp for candle in candles]
        timestamps_by_symbol[symbol] = set(timestamps)
        max_daily_move = max_absolute_daily_move(candles)

        rows.append({
            "symbol": symbol,
            "source": "cache" if cache_hit else "feed",
            "cache_path": str(cache_path) if cache_path else "",
            "candles": len(candles),
            "start": timestamps[0].date().isoformat() if timestamps else "",
            "end": timestamps[-1].date().isoformat() if timestamps else "",
            "first_close": f"{candles[0].close:.4f}" if candles else "",
            "last_close": f"{candles[-1].close:.4f}" if candles else "",
            "max_abs_daily_move": (
                f"{max_daily_move:.4f}" if max_daily_move is not None else ""
            ),
            "split_warning": (
                max_daily_move is not None and max_daily_move >= 0.35
            ),
            "missing": not bool(candles),
        })

    common_timestamps = (
        set.intersection(*timestamps_by_symbol.values())
        if timestamps_by_symbol
        else set()
    )
    common_dates = sorted(common_timestamps)

    report_path = save_data_audit(rows, config["reports"]["summary_dir"])
    print_data_audit(rows, common_dates, config, report_path)


def save_data_audit(rows, report_dir):
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / "data_audit.csv"

    fieldnames = [
        "symbol",
        "source",
        "cache_path",
        "candles",
        "start",
        "end",
        "first_close",
        "last_close",
        "max_abs_daily_move",
        "split_warning",
        "missing",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report_path


def print_data_audit(rows, common_dates, config, report_path):
    sources = {}
    for row in rows:
        sources[row["source"]] = sources.get(row["source"], 0) + 1

    missing = [row["symbol"] for row in rows if row["missing"]]
    split_warnings = [
        row for row in rows
        if row.get("split_warning")
    ]
    shortest = sorted(rows, key=lambda row: row["candles"])[:5]
    longest = sorted(rows, key=lambda row: row["candles"], reverse=True)[:5]

    print("\nDATA AUDIT")
    print(
        f"Symbols: {len(rows)} | "
        f"Timeframe: {config['backtest']['timeframe']} | "
        f"Years requested: {config['backtest']['years']} | "
        f"Sources: {sources}"
    )

    if common_dates:
        print(
            "Common overlap: "
            f"{common_dates[0].date()} to {common_dates[-1].date()} "
            f"({len(common_dates)} shared bars)"
        )
    else:
        print("Common overlap: none")

    print("\nShortest histories")
    for row in shortest:
        print(
            f"{row['symbol']:<6} | {row['candles']:>5} bars | "
            f"{row['start'] or 'n/a'} -> {row['end'] or 'n/a'} | "
            f"{row['source']}"
        )

    print("\nLongest histories")
    for row in longest:
        print(
            f"{row['symbol']:<6} | {row['candles']:>5} bars | "
            f"{row['start'] or 'n/a'} -> {row['end'] or 'n/a'} | "
            f"{row['source']}"
        )

    if missing:
        print(f"\nMissing symbols: {', '.join(missing)}")

    if split_warnings:
        print("\nSplit-adjustment warnings")
        for row in split_warnings[:10]:
            print(
                f"{row['symbol']:<6} | max daily move "
                f"{float(row['max_abs_daily_move']) * 100:>6.2f}% | "
                f"{row['source']}"
            )

    print(f"\nSaved data audit: {report_path}")


def max_absolute_daily_move(candles):
    if len(candles) < 2:
        return None

    max_move = 0

    for previous, current in zip(candles, candles[1:]):
        if previous.close <= 0:
            continue

        move = abs((current.close / previous.close) - 1)
        max_move = max(max_move, move)

    return max_move


def run_base_backtests(config, feed):
    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        result = run_backtest(
            candles=candles,
            symbol=symbol,
            config=config,
        )

        report_path = result.save_json(
            symbol=symbol,
            timeframe=config["backtest"]["timeframe"],
            report_dir=config["reports"]["backtest_dir"],
        )

        print_result(symbol, result, report_path)


def run_optimization(config, feed):
    research_config = config["research"]

    optimizer = ParameterOptimizer(
        config=config,
        metric_name=research_config["optimization_metric"],
        min_closed_trades=research_config.get(
            "optimizer_min_closed_trades",
            research_config.get("min_closed_trades", 0),
        ),
    )

    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        results = optimizer.run(
            candles=candles,
            symbol=symbol,
            grid=research_config["parameter_grid"],
        )

        if not results:
            print(f"{symbol} | no valid parameter combinations")
            continue

        best = results[0]

        print(
            f"{symbol} | best_{best.metric_name}={best.metric_value:.4f} | "
            f"params={best.parameters}"
        )


def run_walk_forward(config, feed, show_details=False):
    research_config = config["research"]

    tester = WalkForwardTester(
        config=config,
        metric_name=research_config["optimization_metric"],
    )

    results = []

    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        result = tester.run(
            candles=candles,
            symbol=symbol,
            folds=research_config["walk_forward_folds"],
            grid=research_config["parameter_grid"],
        )

        report_path = result.save_json(
            report_dir=config["reports"]["walk_forward_dir"],
        )

        results.append((symbol, result, report_path))

    print_walk_forward_summary(results, show_details=show_details)


def run_strategy_comparison(config, feed, show_all=False):
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in config["backtest"]["symbols"]
    }

    comparison = StrategyComparison(
        config=config,
        candles_by_symbol=candles_by_symbol,
    )

    results = comparison.run()

    report_path = comparison.save_csv(
        results,
        report_dir=config["reports"]["summary_dir"],
    )

    print("\nSTRATEGY COMPARISON")

    limit = None if show_all else config["research"].get("report_top_n", 10)

    print(comparison.to_table(results, limit=limit))

    if limit and len(results) > limit:
        print(
            f"\nShowing top {limit} of {len(results)} results. "
            "Use --all-results to print everything."
        )

    print(f"\nSaved summary: {report_path}")


def print_result(symbol, result, report_path):
    print(
        f"{symbol} | "
        f"return={result.total_return * 100:.2f}% | "
        f"max_dd={result.max_drawdown * 100:.2f}% | "
        f"sharpe={result.sharpe:.2f} | "
        f"closed={result.closed_trades} | "
        f"report={report_path}"
    )
