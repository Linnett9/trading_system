def format_percent(value):
    return f"{value * 100:.2f}%"


def print_paper_trade_decision(decision):
    context = decision.model_context

    print("\nPAPER TRADE DECISION")
    print(
        f"date={decision.timestamp.date()} | "
        f"regime={decision.regime_label} | "
        f"risk_on={decision.risk_on} | "
        f"equity={decision.equity:.2f} | "
        f"cash={decision.cash:.2f} | "
        f"target_exposure={format_percent(decision.exposure_target)}"
    )

    selected = ", ".join(decision.selected_symbols) or "cash"
    print(f"selected={selected}")

    print_data_freshness(decision.data_freshness)

    print(
        "Why: "
        f"{context.get('explanation')} "
        f"Mode={context.get('selection_mode')}, "
        f"rank={context.get('ranking_score_mode')}, "
        f"weighting={context.get('weighting')}, "
        f"momentum={context.get('momentum_periods')}."
    )

    print(
        "Selection controls: "
        f"min_score={context.get('min_selection_score')}, "
        f"max_assets={context.get('max_selected_assets')}."
    )

    if context.get("selection_mode") == "all_positive":
        print(
            "Note: all-positive mode can buy more than top_n because it "
            "holds every asset with positive momentum that passes filters."
        )

    if decision.target_weights:
        print("Target weights:")

        for symbol, weight in sorted(
            decision.target_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {symbol:<6} {format_percent(weight)}")

    skipped_assets = context.get("skipped_assets") or []

    if skipped_assets:
        print("Skipped assets:")

        for item in skipped_assets:
            print(
                f"  {item['symbol']:<6} score={item['score']:.4f} | "
                f"{item['reason']}"
            )

    if decision.orders:
        print("Proposed orders:")

        for order in decision.orders:
            print(
                f"  {order.side:<4} {order.symbol:<6} "
                f"qty={order.quantity_delta:.4f} | "
                f"value={order.dollar_delta:.2f} | "
                f"target={format_percent(order.target_weight)} | "
                f"drift={format_percent(order.drift_weight)} | "
                f"price={order.price:.2f} | "
                f"{order.reason}"
            )
    else:
        print("Proposed orders: none")

    print(
        "Interpretation: paper mode only records what the bot would do; "
        "it does not place broker orders or assume fills."
    )
    print(f"Saved decision: {decision.report_path}")
    print(f"Paper state: {decision.state_path}")


def print_paper_fill(fill_record, state_path):
    print("\nPAPER FILL")

    if fill_record.get("no_orders"):
        print(
            f"decision={fill_record['decision_path']} | "
            "status=no_orders | fills=0"
        )
        print(
            "Interpretation: the latest decision had no proposed orders, "
            "so paper state was left unchanged."
        )
        print(f"Paper state: {state_path}")
        return

    if fill_record.get("already_filled"):
        print(
            f"decision={fill_record['decision_path']} | "
            "already_filled=True | fills=0"
        )
        print(
            "Interpretation: this decision was already applied, so no "
            "cash or position changes were made."
        )
        print(f"Paper state: {state_path}")
        return

    print(
        f"decision={fill_record['decision_path']} | "
        f"fills={len(fill_record['fills'])} | "
        f"cash_after={fill_record['cash_after']:.2f} | "
        f"equity_after={fill_record['equity_after']:.2f}"
    )

    if fill_record["fills"]:
        print("Applied fills:")

        for fill in fill_record["fills"]:
            print(
                f"  {fill['side']:<4} {fill['symbol']:<6} "
                f"qty={fill['quantity_delta']:.4f} | "
                f"value={fill['dollar_delta']:.2f} | "
                f"price={fill['price']:.2f} | "
                f"{fill['reason']}"
            )

    print(
        "Interpretation: the paper state now contains these simulated "
        "positions, so the next paper-trade run can propose sells/rebalances."
    )
    print(f"Paper state: {state_path}")


def print_paper_fill_refused(state_path):
    print("\nPAPER FILL REFUSED")
    print(
        "Direct paper-fill is disabled unless --confirm-fill is provided. "
        "Prefer the safer workflow: paper-trading --dry-run, then "
        "paper-trading --submit."
    )
    print(f"Paper state: {state_path}")


def print_paper_run(
    decision,
    fill_record,
    blocked_reason=None,
    risk_checks=None,
    post_trade_checks=None,
    artifact_paths=None,
    run_id=None,
):
    risk_checks = risk_checks or []
    post_trade_checks = post_trade_checks or []
    artifact_paths = artifact_paths or {}
    print("\nPAPER RUN")
    if run_id:
        print(f"run_id={run_id}")

    print(
        f"date={decision.timestamp.date()} | "
        f"regime={decision.regime_label} | "
        f"orders={len(decision.orders)} | "
        f"equity={decision.equity:.2f} | "
        f"cash={decision.cash:.2f}"
    )

    print_data_freshness(decision.data_freshness)

    selected = ", ".join(decision.selected_symbols) or "cash"
    print(f"selected={selected}")

    if risk_checks:
        status = paper_risk_status(risk_checks)
        print(f"risk_status={status}")
        for check in risk_checks:
            if check.severity.value in {"WARNING", "ERROR", "CRITICAL"}:
                print(
                    f"  {check.severity.value:<8} {check.reason} | "
                    f"{check.details}"
                )

    if post_trade_checks:
        status = paper_risk_status(post_trade_checks)
        print(f"post_trade_status={status}")
        for check in post_trade_checks:
            if check.severity.value in {"WARNING", "ERROR", "CRITICAL"}:
                print(
                    f"  {check.severity.value:<8} {check.reason} | "
                    f"{check.details}"
                )

    if blocked_reason == "stale_data":
        print(
            "Action: blocked. Data is stale and paper_trading.refuse_stale_data "
            "is enabled."
        )
    elif blocked_reason == "risk_check_failed":
        print("Action: blocked. One or more ERROR/CRITICAL risk checks failed.")
    elif blocked_reason and blocked_reason.startswith("approval_"):
        print(f"Action: blocked. Dry-run approval failed: {blocked_reason}.")
    elif fill_record is None:
        print("Action: decision saved only; auto-fill is disabled.")
    elif fill_record.get("no_orders"):
        print("Action: no orders required; paper state unchanged.")
    elif fill_record.get("already_filled"):
        print("Action: skipped; this decision was already filled.")
    else:
        print(
            f"Action: filled {len(fill_record['fills'])} paper orders | "
            f"cash_after={fill_record['cash_after']:.2f} | "
            f"equity_after={fill_record['equity_after']:.2f}"
        )

    if decision.orders:
        print("Orders:")

        for order in decision.orders:
            print(
                f"  {order.side:<4} {order.symbol:<6} "
                f"value={order.dollar_delta:.2f} | "
                f"target={format_percent(order.target_weight)} | "
                f"drift={format_percent(order.drift_weight)}"
            )

    print(f"Saved decision: {decision.report_path}")
    if artifact_paths.get("order_preview"):
        print(f"Order preview: {artifact_paths['order_preview']}")
    if artifact_paths.get("risk_report"):
        print(f"Risk report: {artifact_paths['risk_report']}")
    if artifact_paths.get("reconciliation_report"):
        print(f"Reconciliation: {artifact_paths['reconciliation_report']}")
    if artifact_paths.get("approval"):
        print(f"Dry-run approval: {artifact_paths['approval']}")
    if artifact_paths.get("journal"):
        print(f"Journal: {artifact_paths['journal']}")
    if artifact_paths.get("dashboard"):
        print(f"Dashboard: {artifact_paths['dashboard']}")
    if artifact_paths.get("event_log"):
        print(f"Event log: {artifact_paths['event_log']}")
    if artifact_paths.get("fill_log"):
        print(f"Fill log: {artifact_paths['fill_log']}")
    print(f"Paper state: {decision.state_path}")


def paper_risk_status(risk_checks):
    severities = {check.severity.value for check in risk_checks}
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "ERROR" in severities:
        return "ERROR"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


def print_paper_status(status):
    print("\nPAPER STATUS")

    print(
        f"cash={status['cash']:.2f} | "
        f"positions={len(status['positions'])} | "
        f"fills={len(status['fills'])}"
    )

    if status["positions"]:
        print("Positions:")

        for symbol, quantity in sorted(status["positions"].items()):
            print(f"  {symbol:<6} qty={quantity:.4f}")
    else:
        print("Positions: none")

    if status["equity_history"]:
        latest = status["equity_history"][-1]

        print(
            "Latest equity snapshot: "
            f"{latest.get('timestamp')} | "
            f"equity={latest.get('equity'):.2f} | "
            f"cash={latest.get('cash'):.2f}"
        )

    print(f"Paper state: {status['state_path']}")


def print_paper_report(status, data_freshness, benchmark=None, drift_rows=None):
    starting_cash = status.get("starting_cash", 0) or 0
    equity = status.get("mark_to_market_equity", status["cash"])
    total_return = (equity / starting_cash - 1) if starting_cash else 0
    latest_fill = status["fills"][-1] if status["fills"] else None

    print("\nPAPER REPORT")

    print(
        f"starting={starting_cash:.2f} | "
        f"equity={equity:.2f} | "
        f"return={format_percent(total_return)} | "
        f"cash={status['cash']:.2f} | "
        f"positions={len(status['positions'])} | "
        f"fills={len(status['fills'])}"
    )

    print_data_freshness(data_freshness)

    if benchmark:
        print(
            "Benchmark: "
            f"{benchmark['symbol']}={format_percent(benchmark['benchmark_return'])} | "
            f"paper={format_percent(benchmark['paper_return'])} | "
            f"excess={format_percent(benchmark['excess_return'])}"
        )

    if status["positions"]:
        print("Mark-to-market positions:")
        prices = status.get("prices_used", {})

        for symbol, quantity in sorted(status["positions"].items()):
            price = prices.get(symbol, 0)
            value = quantity * price
            weight = value / equity if equity else 0

            print(
                f"  {symbol:<6} qty={quantity:.4f} | "
                f"price={price:.2f} | value={value:.2f} | "
                f"weight={format_percent(weight)}"
            )
    else:
        print("Mark-to-market positions: none")

    if drift_rows:
        print("Target drift:")

        for item in drift_rows:
            print(
                f"  {item['symbol']:<6} "
                f"current={format_percent(item['current_weight'])} | "
                f"target={format_percent(item['target_weight'])} | "
                f"drift={format_percent(item['drift'])} | "
                f"value={item['value']:.2f}"
            )

    if latest_fill:
        print(
            "Latest fill: "
            f"{latest_fill.get('filled_at')} | "
            f"decision={latest_fill.get('decision_path')}"
        )

    if status.get("last_decision_path"):
        print(f"Last decision: {status['last_decision_path']}")

    print(
        "Interpretation: this is still a paper ledger; it marks positions "
        "with the latest cached prices and does not contact a broker."
    )
    print(f"Paper state: {status['state_path']}")


def print_paper_repair(repair):
    print("\nPAPER REPAIR")

    print(
        f"removed_empty_fills={repair['removed_empty_fills']} | "
        f"removed_bad_equity_snapshots={repair['removed_bad_equity_snapshots']} | "
        f"equity={repair['equity']:.2f} | "
        f"cash={repair['cash']:.2f} | "
        f"positions={len(repair['positions'])}"
    )

    print(
        "Interpretation: removed old no-order/zero-equity paper artefacts "
        "and wrote a fresh mark-to-market snapshot."
    )
    print(f"Paper state: {repair['state_path']}")


def print_paper_reset_refused(state_path):
    print("\nPAPER RESET")
    print("Action: refused. Reset is destructive and needs --confirm-reset.")
    print(f"Paper state: {state_path}")


def print_paper_reset(reset):
    print("\nPAPER RESET")

    print(
        f"cash={reset['cash']:.2f} | "
        f"positions={len(reset['positions'])}"
    )

    print("Interpretation: paper ledger has been reset to starting cash.")
    print(f"Paper state: {reset['state_path']}")


def print_paper_weekly_summary(path):
    print("\nPAPER WEEKLY SUMMARY")
    print(f"Saved summary: {path}")


def print_paper_promotion_checklist(path):
    print("\nPAPER PROMOTION CHECKLIST")
    print(f"Saved checklist: {path}")


def print_data_freshness(data_freshness):
    if not data_freshness:
        return

    latest = data_freshness.get("latest_timestamp")
    age_days = data_freshness.get("age_days")
    max_age_days = data_freshness.get("max_age_days")
    stale = data_freshness.get("is_stale")
    status = "STALE" if stale else "fresh"

    print(
        "Data freshness: "
        f"latest={latest} | age={age_days}d | "
        f"limit={max_age_days}d | {status}"
    )

    if stale:
        print(
            "Warning: decisions are based on old cached market data. "
            "Refresh data before trusting a paper or live decision."
        )
