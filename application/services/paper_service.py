from datetime import datetime

from application.services.market_data_loader import (
    load_candles,
    latest_prices,
    latest_data_freshness,
)
from core.paper.paper_trading_engine import PaperTradingEngine


def create_paper_decision(config, feed, build_dual_momentum_tester):
    dual_config = config["research"].get("dual_momentum", {})
    paper_config = config.get("paper_trading", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])

    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    tester = build_dual_momentum_tester(config, dual_config)
    result = tester.run(candles_by_symbol)
    prices_by_symbol = latest_prices(candles_by_symbol)
    engine = build_paper_engine(config)

    return engine.create_decision(
        result,
        prices_by_symbol,
        data_freshness=latest_data_freshness(
            candles_by_symbol,
            max_age_days=paper_config.get("max_data_age_days", 3),
        ),
    )


def build_paper_engine(config):
    paper_config = config.get("paper_trading", {})
    report_dir = paper_config.get(
        "report_dir",
        config.get("reports", {}).get("paper_dir", "reports/paper"),
    )

    return PaperTradingEngine(
        report_dir=report_dir,
        starting_cash=config["backtest"]["starting_equity"],
        min_trade_value=paper_config.get("min_trade_value", 1.0),
        rebalance_threshold=paper_config.get("rebalance_threshold", 0.0),
    )


def paper_benchmark_metrics(status, candles, benchmark_symbol):
    fills = status.get("fills", [])

    if not fills or not candles:
        return None

    start_at = parse_datetime(fills[0].get("decision_timestamp"))

    if start_at is None:
        return None

    start_price = first_close_at_or_after(candles, start_at)
    end_price = candles[-1].close if candles else None

    if not start_price or not end_price:
        return None

    benchmark_return = (end_price / start_price) - 1
    starting_cash = status.get("starting_cash", 0) or 0
    equity = status.get("mark_to_market_equity", status["cash"])
    paper_return = (equity / starting_cash - 1) if starting_cash else 0

    return {
        "symbol": benchmark_symbol,
        "start": start_at.isoformat(),
        "start_price": start_price,
        "end_price": end_price,
        "paper_return": paper_return,
        "benchmark_return": benchmark_return,
        "excess_return": paper_return - benchmark_return,
    }


def paper_drift_rows(status, decision_payload):
    if not decision_payload:
        return []

    prices = status.get("prices_used", {})
    positions = status.get("positions", {})
    equity = status.get("mark_to_market_equity", status["cash"])
    target_weights = decision_payload.get("target_weights", {}) or {}
    exposure_target = float(decision_payload.get("exposure_target", 1) or 0)
    symbols = sorted(set(positions) | set(target_weights))
    rows = []

    for symbol in symbols:
        price = prices.get(symbol, 0)
        quantity = positions.get(symbol, 0)
        current_value = quantity * price
        current_weight = current_value / equity if equity else 0
        target_weight = float(target_weights.get(symbol, 0)) * exposure_target

        rows.append({
            "symbol": symbol,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "drift": target_weight - current_weight,
            "value": current_value,
        })

    return sorted(rows, key=lambda item: abs(item["drift"]), reverse=True)


def parse_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def first_close_at_or_after(candles, start_at):
    comparable_start = start_at.replace(tzinfo=None)

    for candle in candles:
        timestamp = candle.timestamp.replace(tzinfo=None)

        if timestamp >= comparable_start:
            return candle.close

    return None