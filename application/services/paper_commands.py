from application.services.market_data_loader import (
    load_candles,
    latest_prices,
    latest_data_freshness,
)
from application.services.paper_service import (
    create_paper_decision,
    build_paper_engine,
    paper_benchmark_metrics,
    paper_drift_rows,
)
from application.reporting.paper_reporter import (
    print_paper_trade_decision,
    print_paper_fill,
    print_paper_run,
    print_paper_status,
    print_paper_report,
    print_paper_repair,
    print_paper_reset_refused,
    print_paper_reset,
)
from core.research.dual_momentum_factory import build_dual_momentum_tester


def run_paper_trade(config, feed):
    decision = create_paper_decision(
        config,
        feed,
        build_dual_momentum_tester,
    )
    print_paper_trade_decision(decision)


def run_paper_fill(config, decision_file=None):
    engine = build_paper_engine(config)
    fill_record = engine.fill_latest_decision(decision_file)
    print_paper_fill(fill_record, engine.state_path)


def run_paper_status(config):
    engine = build_paper_engine(config)
    print_paper_status(engine.status())


def run_paper_report(config, feed):
    dual_config = config["research"].get("dual_momentum", {})
    benchmark_symbol = dual_config.get("regime_symbol", "SPY")

    symbols = sorted(set(
        dual_config.get("symbols", config["backtest"]["symbols"])
        + [benchmark_symbol]
    ))

    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    engine = build_paper_engine(config)
    status = engine.status(latest_prices(candles_by_symbol))
    decision_payload = engine.latest_decision_payload()

    print_paper_report(
        status=status,
        data_freshness=latest_data_freshness(candles_by_symbol),
        benchmark=paper_benchmark_metrics(
            status=status,
            candles=candles_by_symbol.get(benchmark_symbol, []),
            benchmark_symbol=benchmark_symbol,
        ),
        drift_rows=paper_drift_rows(
            status=status,
            decision_payload=decision_payload,
        ),
    )


def run_paper_repair(config, feed):
    dual_config = config["research"].get("dual_momentum", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])

    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    engine = build_paper_engine(config)
    repair = engine.repair_state(latest_prices(candles_by_symbol))

    print_paper_repair(repair)


def run_paper_reset(config, confirm_reset=False):
    engine = build_paper_engine(config)

    if not confirm_reset:
        print_paper_reset_refused(engine.state_path)
        return

    print_paper_reset(engine.reset_state())


def run_paper_run(config, feed):
    paper_config = config.get("paper_trading", {})

    decision = create_paper_decision(
        config,
        feed,
        build_dual_momentum_tester,
    )

    fill_record = None
    blocked_reason = None

    if (
        paper_config.get("refuse_stale_data", True)
        and decision.data_freshness.get("is_stale")
    ):
        blocked_reason = "stale_data"
    elif paper_config.get("auto_fill", True):
        engine = build_paper_engine(config)
        fill_record = engine.fill_latest_decision(decision.report_path)

    print_paper_run(
        decision=decision,
        fill_record=fill_record,
        blocked_reason=blocked_reason,
    )