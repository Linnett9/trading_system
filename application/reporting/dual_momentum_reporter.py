from core.research.dual_momentum_scoring import (
    classify_dual_momentum_result,
    classify_walk_forward_fold_result,
    fold_gap_label,
    production_constraint_gap_details,
    production_gap_score,
    risk_regime_score,
    dual_momentum_quality_score,
    dual_momentum_walk_forward_summary,
    paper_safe_dual_momentum_score,
    walk_forward_selection_score,
)


FROZEN_CHAMPION_ID = "ranked_top5_monthly_exposure90_v1"
FROZEN_CHAMPION_CONFIG_NAME = "ranked_top5_hyst2_rank10_replace2_min7_exposure90"
RISK_REGIME_REPORT_LIMIT = 25


def format_percent(value):
    return f"{value * 100:.2f}%"


def select_champion(results):
    if not results:
        return None

    production_candidates = _candidates_with_tag(
        results,
        {"production candidate"},
    )
    if production_candidates:
        return max(production_candidates, key=_champion_sort_key)

    near_production_candidates = _candidates_with_tag(
        results,
        {"near-production candidate"},
    )
    if near_production_candidates:
        return max(near_production_candidates, key=_champion_sort_key)

    paper_candidates = _candidates_with_tag(results, {"paper candidate"})
    if paper_candidates:
        return max(paper_candidates, key=_champion_sort_key)

    candidates = [
        item
        for item in results
        if not classify_dual_momentum_result(item["result"]).startswith("rejected")
    ]

    if not candidates:
        candidates = results

    return max(candidates, key=_champion_sort_key)


def select_raw_score_leader(results):
    if not results:
        return None

    return max(results, key=lambda item: risk_regime_score(item["result"]))


def champion_label(item):
    if item is None:
        return "Champion"

    tag = classify_dual_momentum_result(item["result"])
    if tag == "production candidate":
        return "Production champion"

    if tag == "near-production candidate":
        return "Near-production champion"

    if tag == "paper candidate":
        return "Best paper candidate"

    return "Best research candidate"


def _candidates_with_tag(results, tags):
    return [
        item
        for item in results
        if classify_dual_momentum_result(item["result"]) in tags
    ]


def _champion_sort_key(item):
    result = item["result"]
    return (
        -production_gap_score(result),
        paper_safe_dual_momentum_score(result),
        risk_regime_score(result),
        result.result.sharpe,
        result.calmar,
        -result.result.max_drawdown,
        -result.annualized_turnover_percent,
        -result.cost_drag_percent,
    )


def production_gap_label(result):
    details = production_constraint_gap_details(result)
    if not details:
        return "ready"

    labels = []
    for item in details:
        label = item["label"]
        gap = item["gap"]
        if item["key"] in {"turnover", "drawdown", "return", "cost", "excess"}:
            labels.append(f"{label} +{format_percent(gap)}")
        elif item["key"] == "sharpe":
            labels.append(f"{label} +{gap:.2f}")
        else:
            labels.append(label)

    return ", ".join(labels)


def risk_regime_report_items(results, champion_item, raw_leader_item):
    must_show = {
        id(item)
        for item in (champion_item, raw_leader_item)
        if item is not None
    }

    ranked = sorted(
        results,
        key=lambda item: (
            classify_dual_momentum_result(item["result"])
            == "production candidate",
            classify_dual_momentum_result(item["result"])
            == "near-production candidate",
            -production_gap_score(item["result"]),
            paper_safe_dual_momentum_score(item["result"]),
            risk_regime_score(item["result"]),
            item["result"].result.sharpe,
            -item["result"].result.max_drawdown,
            -item["result"].annualized_turnover_percent,
        ),
        reverse=True,
    )

    selected = []
    selected_ids = set()
    for item in ranked:
        if len(selected) < RISK_REGIME_REPORT_LIMIT or id(item) in must_show:
            selected.append(item)
            selected_ids.add(id(item))

    for item in (champion_item, raw_leader_item):
        if item is not None and id(item) not in selected_ids:
            selected.append(item)

    return selected


def walk_forward_readiness_label(summary):
    if (
        summary["worst_excess_return"] >= 0
        and summary["consistency"] >= (2 / 3)
        and summary["average_drawdown"] <= 0.18
        and summary["average_turnover"] <= 6.0
        and summary.get("rejected_fold_count", 0) == 0
    ):
        return "production-ready"

    if (
        summary["average_excess_return"] > 0
        and summary["consistency"] >= 0.5
        and summary["average_drawdown"] <= 0.20
        and summary["average_turnover"] <= 10.0
        and summary.get("rejected_fold_count", 0) <= 1
    ):
        return "paper-ready"

    return "research-only"


def format_optional_percent(value):
    if value is None:
        return "n/a"

    return format_percent(value)


def print_dual_momentum_diagnosis(diagnosis, report_path):
    print("\nDUAL MOMENTUM DIAGNOSIS")
    print(
        "Year | Return | Avg Target | RiskOn | Fast | Partial | Cash | "
        "Top Symbols | Worst Months"
    )
    print("-" * 118)

    for year, year_item in sorted(diagnosis["annual"].items()):
        worst_months = ", ".join(
            f"{month['month']} {format_percent(month['bot_return'])}"
            for month in year_item["worst_months"]
        )

        print(
            f"{year} | "
            f"{format_percent(year_item['bot_return']):>7} | "
            f"{format_percent(year_item['average_exposure_target']):>10} | "
            f"{year_item['risk_on_months']:>6} | "
            f"{year_item['fast_reentry_months']:>4} | "
            f"{year_item['partial_risk_months']:>7} | "
            f"{year_item['cash_months']:>4} | "
            f"{', '.join(year_item['top_selected_symbols']):<24} | "
            f"{worst_months}"
        )

        top = ", ".join(
            f"{contributor['symbol']} "
            f"{format_percent(contributor['contribution'])}"
            for contributor in year_item["top_contributors"][:3]
        )
        worst = ", ".join(
            f"{contributor['symbol']} "
            f"{format_percent(contributor['contribution'])}"
            for contributor in year_item["worst_contributors"][:3]
        )

        print(
            f"     contributors | best: {top or 'n/a'} | "
            f"worst: {worst or 'n/a'}"
        )

        missed = ", ".join(
            f"{winner['symbol']} "
            f"{format_percent(winner['average_return'])}"
            for winner in year_item["missed_winners"][:3]
        )

        print(f"     missed winners | {missed or 'n/a'}")

        relative = year_item.get("benchmark_relative", {})
        print(
            "     benchmark gaps | "
            "bot-vs-SPY: "
            f"{format_optional_percent(relative.get('bot_vs_spy'))} | "
            "selected-avg-vs-SPY: "
            f"{format_optional_percent(relative.get('selected_average_vs_spy'))}"
        )

        for month in year_item["worst_months"][:2]:
            print(
                f"     weak month {month['month']} | "
                f"bot={format_percent(month['bot_return'])} | "
                "SPY="
                f"{format_optional_percent(month['benchmark_returns'].get('SPY'))} | "
                "QQQ="
                f"{format_optional_percent(month['benchmark_returns'].get('QQQ'))} | "
                f"held={', '.join(month['selected_symbols']) or 'cash'}"
            )

    print(f"\nSaved diagnosis: {report_path}")


def print_dual_momentum_risk_regime_experiments(results, report_path):
    years = sorted({
        year
        for item in results
        for year in item["result"].annual_returns
    })
    year_headers = " | ".join(str(year) for year in years)
    champion_item = select_champion(results)
    champion_result = champion_item["result"] if champion_item is not None else None
    raw_leader_item = select_raw_score_leader(results)
    report_items = risk_regime_report_items(
        results,
        champion_item,
        raw_leader_item,
    )

    print("\nDUAL MOMENTUM RISK REGIME PERFORMANCE")
    print(f"Current frozen champion: {FROZEN_CHAMPION_ID}")
    frozen_champion_item = next(
        (
            item for item in results
            if item["name"] == FROZEN_CHAMPION_CONFIG_NAME
        ),
        None,
    )
    if frozen_champion_item is None:
        print(
            "Frozen champion guard: missing "
            f"{FROZEN_CHAMPION_CONFIG_NAME} from this run"
        )
    if not _candidates_with_tag(results, {"production candidate"}):
        print("Production champion: none")
    if not _candidates_with_tag(results, {"near-production candidate"}):
        print("Near-production champion: none")
    if champion_item is not None:
        print(
            f"{champion_label(champion_item)}: {champion_item['name']} | "
            f"Score={risk_regime_score(champion_result):.2f} | "
            f"Return={format_percent(champion_result.result.total_return)} | "
            f"Sharpe={champion_result.result.sharpe:.2f} | "
            f"DD={format_percent(champion_result.result.max_drawdown)} | "
            f"Turn={format_percent(champion_result.annualized_turnover_percent)} | "
            f"Cost={format_percent(champion_result.cost_drag_percent)} | "
            f"Gap={production_gap_label(champion_result)}"
        )
    if raw_leader_item is not None and raw_leader_item is not champion_item:
        raw_result = raw_leader_item["result"]
        print(
            f"Raw score leader: {raw_leader_item['name']} | "
            f"Score={risk_regime_score(raw_result):.2f} | "
            f"Return={format_percent(raw_result.result.total_return)} | "
            f"DD={format_percent(raw_result.result.max_drawdown)} | "
            f"Turn={format_percent(raw_result.annualized_turnover_percent)} | "
            f"Tag={classify_dual_momentum_result(raw_result)} | "
            f"Gap={production_gap_label(raw_result)}"
        )
    hidden_count = max(0, len(results) - len(report_items))
    if hidden_count:
        print(
            f"Showing top {len(report_items)} candidates; "
            f"{hidden_count} lower-ranked configs hidden. Full CSV still saved."
        )
    print(
        f"Config | Return | SPY | Ex SPY | EqWt | Ex EqWt | Sharpe | DD | "
        f"Turn | Cost | Score | vsChamp | Tag | Misses | {year_headers}"
    )
    print("-" * (150 + len(years) * 9))

    for item in report_items:
        result = item["result"]
        year_values = " | ".join(
            f"{format_percent(result.annual_returns.get(year, 0)):>7}"
            for year in years
        )

        print(
            f"{item['name']:<26} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.equal_weight_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.annualized_turnover_percent):>7} | "
            f"{format_percent(result.cost_drag_percent):>6} | "
            f"{risk_regime_score(result):>5.2f} | "
            f"{champion_delta_label(result, champion_result):<24} | "
            f"{classify_dual_momentum_result(result):<23} | "
            f"{production_gap_label(result):<18} | "
            f"{year_values}"
        )

    print("\nDUAL MOMENTUM RISK REGIME CONFIG")
    print(
        "Regime Config | Mix | Off | Fallback | Chop | ChopExp | "
        "BreadthScale | DDScale | VolShock | RankExit | HoldRank | MaxRep | "
        "Gap | MinTrade | Drift | Rebal | Regime | Quality | Cooldown | "
        "Decay | Lead | Bench | BenchPart | MaxSector | Rank"
    )
    print("-" * 220)

    for item in report_items:
        result = item["result"]

        print(
            f"{item['name']:<30} | "
            f"{format_percent(result.config['mixed_risk_exposure']):>5} | "
            f"{format_percent(result.config['risk_off_risk_exposure']):>5} | "
            f"{format_percent(result.config['fallback_allocation']):>8} | "
            f"{str(result.config['chop_filter_enabled']):>5} | "
            f"{format_percent(result.config['chop_risk_exposure']):>7} | "
            f"{str(result.config['breadth_scaled_exposure_enabled']):>12} | "
            f"{str(result.config['drawdown_recovery_scaling_enabled']):>7} | "
            f"{str(result.config['volatility_shock_filter_enabled']):>8} | "
            f"{str(result.config['rank_deterioration_exit_enabled']):>8} | "
            f"{str(result.config['rank_hysteresis_max_rank']):>8} | "
            f"{str(result.config['max_rebalance_replacements']):>6} | "
            f"{format_percent(result.config['replacement_score_gap']):>5} | "
            f"{format_percent(result.config['rebalance_min_trade_weight']):>8} | "
            f"{format_percent(result.config.get('rebalance_drift_band', 0)):>5} | "
            f"{result.config['rebalance_frequency']:<9} | "
            f"{result.config['regime_confirmation_mode']:<7} | "
            f"{str(result.config['quality_filter_enabled']):>7} | "
            f"{str(result.config['cooldown_enabled']):>8} | "
            f"{str(result.config['decay_exit_enabled']):>5} | "
            f"{str(result.config['leadership_filter_enabled']):>5} | "
            f"{format_percent(result.config['benchmark_sleeve_allocation']):>5} | "
            f"{str(result.config['benchmark_participation_filter_enabled']):>9} | "
            f"{format_percent(result.config['max_sector_weight'] or 0):>9} | "
            f"{result.config['ranking_score_mode']}"
        )

    if results:
        print(
            "\nInterpretation: "
            f"{dual_momentum_result_explanation(results[0]['result'])} "
            "Use this table to compare full-period configs, then confirm the "
            "winner with walk-forward."
        )

    print(f"\nSaved risk-regime experiments: {report_path}")


def champion_delta_label(result, champion):
    if champion is None:
        return "n/a"

    return_delta = result.result.total_return - champion.result.total_return
    drawdown_delta = (
        result.result.max_drawdown - champion.result.max_drawdown
    )
    turnover_delta = (
        result.annualized_turnover_percent
        - champion.annualized_turnover_percent
    )

    return (
        f"ret {format_percent(return_delta)} "
        f"dd {format_percent(drawdown_delta)} "
        f"turn {format_percent(turnover_delta)}"
    )


def print_dual_momentum_walk_forward(results, report_path):
    summary = dual_momentum_walk_forward_summary(results)

    print("\nDUAL MOMENTUM WALK-FORWARD")
    print(
        "Fold | Test | Selected | Mode | Weight | TopN | Mom | VolTgt | DDGuard | "
        "Return | SPY | Ex SPY | Ex EqWt | BullCap | Sharpe | DD | Turn | Cost | Tag"
    )
    print("-" * 180)

    for index, item in enumerate(results, start=1):
        fold = item["fold"]
        result = item["result"]
        selected_config = (
            item["training_result"].config
            if item.get("training_result") is not None
            else result.config
        )
        selected_name = selected_config.get("experiment_name") or "n/a"
        bull_capture = (
            result.result.total_return / result.benchmark_return
            if result.benchmark_return > 0
            else 0
        )

        print(
            f"{index:>4} | "
            f"{fold['test_start']}..{fold['test_end']} | "
            f"{selected_name:<34} | "
            f"{selected_config.get('selection_mode', 'ranked'):<12} | "
            f"{selected_config.get('weighting', 'equal'):<18} | "
            f"{selected_config['top_n']:>4} | "
            f"{'/'.join(str(value) for value in selected_config['momentum_periods']):<7} | "
            f"{str(selected_config['target_volatility']):<6} | "
            f"{str(selected_config['max_drawdown_guard']):<7} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{format_percent(bull_capture):>7} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.annualized_turnover_percent):>7} | "
            f"{format_percent(result.cost_drag_percent):>6} | "
            f"{classify_walk_forward_fold_result(result)} | "
            f"{fold_gap_label(result)}"
        )

        training_result = item.get("training_result")
        if training_result is not None:
            filter_reasons = getattr(
                training_result,
                "walk_forward_filter_reasons",
                [],
            )
            fallback_label = (
                "fallback"
                if getattr(
                    training_result,
                    "walk_forward_filter_fallback",
                    False,
                )
                else "filtered"
            )
            print(
                "     why selected | "
                f"train_score={walk_forward_selection_score(training_result):.2f} | "
                f"train_return={format_percent(training_result.result.total_return)} | "
                f"train_excess={format_percent(training_result.excess_return)} | "
                f"train_sharpe={training_result.result.sharpe:.2f} | "
                f"train_dd={format_percent(training_result.result.max_drawdown)} | "
                f"train_turn={format_percent(training_result.annualized_turnover_percent)} | "
                f"pool={fallback_label} | "
                f"filter={'pass' if not filter_reasons else ','.join(filter_reasons)}"
            )

    if results:
        print("-" * 158)
        print(
            "Average | "
            f"excess_spy={format_percent(summary['average_excess_return'])} | "
            f"excess_eq={format_percent(summary['average_excess_vs_equal_weight'])} | "
            f"drawdown={format_percent(summary['average_drawdown'])} | "
            f"worst_excess={format_percent(summary['worst_excess_return'])} | "
            f"consistency={format_percent(summary['consistency'])} | "
            f"bull_capture={format_percent(summary['average_bull_capture'])} | "
            f"worst_capture={format_percent(summary['worst_bull_capture'])} | "
            f"turnover={format_percent(summary['average_turnover'])} | "
            f"rejected_folds={summary.get('rejected_fold_count', 0)} | "
            f"score={summary['score']:.2f} | "
            f"readiness={walk_forward_readiness_label(summary)}"
        )
        print(
            f"Interpretation: "
            f"{dual_momentum_walk_forward_explanation(summary)}"
        )

    print(f"\nSaved walk-forward: {report_path}")


def dual_momentum_result_explanation(result):
    parts = []

    if result.excess_return > 0:
        parts.append(
            f"beat SPY by {format_percent(result.excess_return)}"
        )
    else:
        parts.append(
            f"trailed SPY by {format_percent(abs(result.excess_return))}"
        )

    if result.excess_vs_equal_weight > 0:
        parts.append(
            "beat equal-weight by "
            f"{format_percent(result.excess_vs_equal_weight)}"
        )
    else:
        parts.append(
            "trailed equal-weight by "
            f"{format_percent(abs(result.excess_vs_equal_weight))}"
        )

    if result.result.max_drawdown <= 0.20:
        parts.append("drawdown is within the 20% target")
    else:
        parts.append("drawdown is above the 20% target")

    bull_capture = (
        result.result.total_return / result.benchmark_return
        if result.benchmark_return > 0
        else 0
    )

    if result.benchmark_return > 0:
        parts.append(f"bull capture is {format_percent(bull_capture)}")

    if result.annualized_turnover_percent > 7:
        parts.append("turnover is high")
    else:
        parts.append("turnover is controlled")

    return "; ".join(parts) + "."


def dual_momentum_walk_forward_explanation(summary):
    if (
        summary["average_excess_return"] > 0
        and summary["consistency"] >= 0.66
    ):
        verdict = "robust so far"
    elif summary["average_excess_return"] > 0:
        verdict = "promising but inconsistent"
    else:
        verdict = "not robust yet"

    return (
        f"{verdict}; average excess is "
        f"{format_percent(summary['average_excess_return'])}, worst fold is "
        f"{format_percent(summary['worst_excess_return'])}, consistency is "
        f"{format_percent(summary['consistency'])}, and bull capture is "
        f"{format_percent(summary['average_bull_capture'])}."
    )


def print_dual_momentum_experiments(results, report_path):
    print("\nDUAL MOMENTUM EXPERIMENTS")
    print(
        "Mode | Rank | Weight | MaxW | Mix | Off | Fallback | Decay | Chop | "
        "ChExp | Lead | Bench | Kill | TopN | Mom | Rebal | Asset SMA | "
        "Breadth | VolTgt | DD Guard | Return | CAGR | Calmar | Ex EqWt | "
        "Sharpe | DD | AnnTurn | Score | Tag"
    )
    print("-" * 258)

    for result in results[:10]:
        print(
            f"{result.config['selection_mode']:<12} | "
            f"{result.config['ranking_score_mode']:<16} | "
            f"{result.config['weighting']:<18} | "
            f"{format_percent(result.config['max_position_weight'] or 0):<6} | "
            f"{format_percent(result.config['mixed_risk_exposure']):<5} | "
            f"{format_percent(result.config['risk_off_risk_exposure']):<5} | "
            f"{format_percent(result.config['fallback_allocation']):<8} | "
            f"{str(result.config['decay_exit_enabled']):<5} | "
            f"{str(result.config['chop_filter_enabled']):<5} | "
            f"{format_percent(result.config['chop_risk_exposure']):<5} | "
            f"{str(result.config['leadership_filter_enabled']):<5} | "
            f"{format_percent(result.config['benchmark_sleeve_allocation']):<5} | "
            f"{str(result.config['strict_drawdown_kill_switch']):<5} | "
            f"{result.config['top_n']:>4} | "
            f"{'/'.join(str(value) for value in result.config['momentum_periods']):<7} | "
            f"{result.config['rebalance_frequency']:<7} | "
            f"{str(result.config['use_asset_trend_filter']):<9} | "
            f"{format_percent(result.config['min_breadth_percent']):<7} | "
            f"{str(result.config['target_volatility']):<6} | "
            f"{str(result.config['max_drawdown_guard']):<8} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.cagr):>6} | "
            f"{result.calmar:>6.2f} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.annualized_turnover_percent):>7} | "
            f"{dual_momentum_quality_score(result):>5.2f} | "
            f"{classify_dual_momentum_result(result)}"
        )

    print(f"\nSaved experiments: {report_path}")


def print_dual_momentum_result(result, report_path):
    backtest = result.result
    utilization = backtest.capital_utilization
    analysis = backtest.trade_analysis
    drawdown = result.drawdown_statistics
    champion_id = result.config.get("champion_id")
    champion_source = result.config.get("champion_source_config_name")

    print("\nDUAL MOMENTUM PORTFOLIO")
    if champion_id:
        source_suffix = f" ({champion_source})" if champion_source else ""
        print(f"Current frozen champion: {champion_id}{source_suffix}")

    print(
        f"return={format_percent(backtest.total_return)} | "
        f"benchmark={format_percent(result.benchmark_return)} | "
        f"excess={format_percent(result.excess_return)} | "
        f"equal_weight={format_percent(result.equal_weight_return)} | "
        f"ex_eq={format_percent(result.excess_vs_equal_weight)} | "
        f"cagr={format_percent(result.cagr)} | "
        f"calmar={result.calmar:.2f} | "
        f"sharpe={backtest.sharpe:.2f} | "
        f"max_dd={format_percent(backtest.max_drawdown)} | "
        f"closed={backtest.closed_trades} | "
        f"open={backtest.open_trades}"
    )

    print(
        f"time_in={format_percent(analysis.time_in_market_percent)} | "
        f"exposure={format_percent(utilization.average_exposure_percent)} | "
        f"cash={format_percent(utilization.average_cash_percent)} | "
        f"profit_factor={backtest.profit_factor:.2f} | "
        f"turnover={format_percent(result.turnover_percent)} | "
        f"ann_turn={format_percent(result.annualized_turnover_percent)} | "
        f"turn/rebal={format_percent(result.turnover_per_rebalance_percent)} | "
        f"cost={result.estimated_cost:.2f}"
    )

    print(
        "drawdown | "
        f"avg={format_percent(drawdown['average_drawdown'])} | "
        f"current={format_percent(drawdown['current_drawdown'])} | "
        f"longest={drawdown['longest_drawdown_days']}d"
    )

    print(f"Interpretation: {dual_momentum_result_explanation(result)}")
    print("Recent selections:")

    for selection in result.selections[-5:]:
        names = ", ".join(selection.symbols) if selection.symbols else "cash"
        regime = "risk-on" if selection.risk_on else "risk-off"
        print(f"  {selection.timestamp.date()} | {regime} | {names}")

    print(f"Saved summary: {report_path}")
