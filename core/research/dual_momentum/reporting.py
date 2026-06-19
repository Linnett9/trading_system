import csv
import json
from datetime import datetime
from pathlib import Path

from core.research.dual_momentum_scoring import (
    classify_dual_momentum_result,
    classify_walk_forward_fold_result,
    dual_momentum_quality_score,
    dual_momentum_walk_forward_summary,
    paper_safe_dual_momentum_score,
    risk_regime_score,
)


def parse_config_date(value):
    return datetime.fromisoformat(value)


def save_dual_momentum_walk_forward(
    results,
    report_dir,
    filename="dual_momentum_walk_forward.json",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    payload = {
        "summary": dual_momentum_walk_forward_summary(results),
        "folds": [],
    }

    for item in results:
        result = item["result"]
        training_result = item.get("training_result")

        bull_capture = (
            result.result.total_return / result.benchmark_return
            if result.benchmark_return > 0
            else None
        )

        payload["folds"].append({
            "fold": item["fold"],
            "selected_config": (
                training_result.config
                if training_result is not None
                else result.config
            ),
            "selected_name": (
                training_result.config.get("experiment_name")
                if training_result is not None
                else result.config.get("experiment_name")
            ),
            "train_return": (
                training_result.result.total_return
                if training_result is not None
                else None
            ),
            "train_quality_score": (
                dual_momentum_quality_score(training_result)
                if training_result is not None
                else None
            ),
            "train_paper_safe_score": (
                paper_safe_dual_momentum_score(training_result)
                if training_result is not None
                else None
            ),
            "return": result.result.total_return,
            "benchmark_return": result.benchmark_return,
            "equal_weight_return": result.equal_weight_return,
            "excess_return": result.excess_return,
            "excess_vs_equal_weight": result.excess_vs_equal_weight,
            "bull_capture_ratio": bull_capture,
            "sharpe": result.result.sharpe,
            "max_drawdown": result.result.max_drawdown,
            "cagr": result.cagr,
            "calmar": result.calmar,
            "annualized_turnover_percent": (
                result.annualized_turnover_percent
            ),
            "cost_drag_percent": result.cost_drag_percent,
            "closed_trades": result.result.closed_trades,
            "open_trades": result.result.open_trades,
        })

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path


def save_dual_momentum_filtered_walk_forward_candidates(
    folds,
    report_dir,
    filename="dual_momentum_walk_forward_candidates.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    fieldnames = [
        "fold",
        "train_start",
        "train_end",
        "rank",
        "name",
        "selector_mode",
        "filter_passed",
        "filter_fallback",
        "filter_reasons",
        "eligible_for_production_selector",
        "return",
        "excess_vs_benchmark",
        "excess_vs_equal_weight",
        "sharpe",
        "max_drawdown",
        "annualized_turnover_percent",
        "cost_drag_percent",
        "quality_score",
        "paper_safe_score",
        "tag",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for fold_index, item in enumerate(folds, start=1):
            fold = item["fold"]
            for rank, result in enumerate(item["training_results"], start=1):
                writer.writerow({
                    "fold": fold_index,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "rank": rank,
                    "name": result.config.get("experiment_name", "n/a"),
                    "selector_mode": getattr(
                        result,
                        "walk_forward_selector_mode",
                        "",
                    ),
                    "filter_passed": getattr(
                        result,
                        "walk_forward_filter_passed",
                        "",
                    ),
                    "filter_fallback": getattr(
                        result,
                        "walk_forward_filter_fallback",
                        "",
                    ),
                    "filter_reasons": ",".join(
                        getattr(
                            result,
                            "walk_forward_filter_reasons",
                            [],
                        )
                    ),
                    "eligible_for_production_selector": result.config.get(
                        "eligible_for_production_selector",
                        False,
                    ),
                    "return": result.result.total_return,
                    "excess_vs_benchmark": result.excess_return,
                    "excess_vs_equal_weight": result.excess_vs_equal_weight,
                    "sharpe": result.result.sharpe,
                    "max_drawdown": result.result.max_drawdown,
                    "annualized_turnover_percent": (
                        result.annualized_turnover_percent
                    ),
                    "cost_drag_percent": result.cost_drag_percent,
                    "quality_score": dual_momentum_quality_score(result),
                    "paper_safe_score": paper_safe_dual_momentum_score(result),
                    "tag": classify_dual_momentum_result(result),
                })

    return path


def save_dual_momentum_risk_regime_experiments(
    results,
    report_dir,
    filename="dual_momentum_risk_regimes.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    years = sorted({
        year
        for item in results
        for year in item["result"].annual_returns
    })

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=risk_regime_fieldnames(years),
        )
        writer.writeheader()

        for item in results:
            result = item["result"]
            row = risk_regime_row(item["name"], result)
            row.update({
                str(year): result.annual_returns.get(year, 0)
                for year in years
            })
            writer.writerow(row)

    return path


def save_dual_momentum_experiments(
    results,
    report_dir,
    filename="dual_momentum_experiments.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=experiment_fieldnames(),
        )
        writer.writeheader()

        for result in results:
            writer.writerow(experiment_row(result))

    return path


def risk_regime_fieldnames(years):
    return [
        "name",
        "return",
        "benchmark_return",
        "equal_weight_return",
        "excess_vs_benchmark",
        "excess_vs_equal_weight",
        "sharpe",
        "max_drawdown",
        "cagr",
        "calmar",
        "annualized_turnover_percent",
        "regime_confirmation_mode",
        "experiment_name",
        "rebalance_frequency",
        "mixed_risk_exposure",
        "risk_off_risk_exposure",
        "breadth_scaled_exposure_enabled",
        "drawdown_recovery_scaling_enabled",
        "volatility_shock_filter_enabled",
        "fallback_allocation",
        "decay_exit_enabled",
        "rank_drop_exit_top_n",
        "rank_deterioration_exit_enabled",
        "rank_deterioration_exit_rank",
        "chop_filter_enabled",
        "chop_risk_exposure",
        "quality_filter_enabled",
        "quality_require_momentum_improving",
        "cooldown_enabled",
        "short_term_weakness_penalty_enabled",
        "short_term_weakness_penalty_floor",
        "short_term_weakness_penalty_weight",
        "rank_hysteresis_enabled",
        "rank_hysteresis_margin",
        "rank_hysteresis_max_rank",
        "max_rebalance_replacements",
        "replacement_score_gap",
        "rebalance_min_trade_weight",
        "rebalance_drift_band",
        "leadership_filter_enabled",
        "relative_strength_filter_enabled",
        "relative_strength_filter_symbol",
        "relative_strength_filter_period",
        "relative_strength_filter_min_excess",
        "benchmark_sleeve_allocation",
        "benchmark_participation_filter_enabled",
        "benchmark_participation_period",
        "benchmark_participation_min_return",
        "benchmark_participation_max_selected_excess",
        "max_sector_weight",
        "ranking_score_mode",
        "relative_strength_weight",
        "transaction_cost_bps",
        "commission_bps",
        "slippage_bps",
        "spread_cost_bps",
        "effective_transaction_cost_bps",
        "score",
        "quality_score",
        "paper_safe_score",
    ] + [str(year) for year in years]


def risk_regime_row(name, result):
    return {
        "name": name,
        "return": result.result.total_return,
        "benchmark_return": result.benchmark_return,
        "equal_weight_return": result.equal_weight_return,
        "excess_vs_benchmark": result.excess_return,
        "excess_vs_equal_weight": result.excess_vs_equal_weight,
        "sharpe": result.result.sharpe,
        "max_drawdown": result.result.max_drawdown,
        "cagr": result.cagr,
        "calmar": result.calmar,
        "annualized_turnover_percent": result.annualized_turnover_percent,
        "regime_confirmation_mode": (
            result.config["regime_confirmation_mode"]
        ),
        "experiment_name": result.config.get("experiment_name"),
        "rebalance_frequency": result.config["rebalance_frequency"],
        "mixed_risk_exposure": result.config["mixed_risk_exposure"],
        "risk_off_risk_exposure": result.config["risk_off_risk_exposure"],
        "breadth_scaled_exposure_enabled": (
            result.config["breadth_scaled_exposure_enabled"]
        ),
        "drawdown_recovery_scaling_enabled": (
            result.config["drawdown_recovery_scaling_enabled"]
        ),
        "volatility_shock_filter_enabled": (
            result.config["volatility_shock_filter_enabled"]
        ),
        "fallback_allocation": result.config["fallback_allocation"],
        "decay_exit_enabled": result.config["decay_exit_enabled"],
        "rank_drop_exit_top_n": result.config["rank_drop_exit_top_n"],
        "rank_deterioration_exit_enabled": (
            result.config["rank_deterioration_exit_enabled"]
        ),
        "rank_deterioration_exit_rank": (
            result.config["rank_deterioration_exit_rank"]
        ),
        "chop_filter_enabled": result.config["chop_filter_enabled"],
        "chop_risk_exposure": result.config["chop_risk_exposure"],
        "quality_filter_enabled": result.config["quality_filter_enabled"],
        "quality_require_momentum_improving": (
            result.config["quality_require_momentum_improving"]
        ),
        "cooldown_enabled": result.config["cooldown_enabled"],
        "short_term_weakness_penalty_enabled": (
            result.config["short_term_weakness_penalty_enabled"]
        ),
        "short_term_weakness_penalty_floor": (
            result.config["short_term_weakness_penalty_floor"]
        ),
        "short_term_weakness_penalty_weight": (
            result.config["short_term_weakness_penalty_weight"]
        ),
        "rank_hysteresis_enabled": result.config["rank_hysteresis_enabled"],
        "rank_hysteresis_margin": result.config["rank_hysteresis_margin"],
        "rank_hysteresis_max_rank": (
            result.config["rank_hysteresis_max_rank"]
        ),
        "max_rebalance_replacements": (
            result.config["max_rebalance_replacements"]
        ),
        "replacement_score_gap": result.config["replacement_score_gap"],
        "rebalance_min_trade_weight": (
            result.config["rebalance_min_trade_weight"]
        ),
        "rebalance_drift_band": result.config.get("rebalance_drift_band", 0),
        "leadership_filter_enabled": result.config["leadership_filter_enabled"],
        "relative_strength_filter_enabled": (
            result.config["relative_strength_filter_enabled"]
        ),
        "relative_strength_filter_symbol": (
            result.config["relative_strength_filter_symbol"]
        ),
        "relative_strength_filter_period": (
            result.config["relative_strength_filter_period"]
        ),
        "relative_strength_filter_min_excess": (
            result.config["relative_strength_filter_min_excess"]
        ),
        "benchmark_sleeve_allocation": (
            result.config["benchmark_sleeve_allocation"]
        ),
        "benchmark_participation_filter_enabled": (
            result.config["benchmark_participation_filter_enabled"]
        ),
        "benchmark_participation_period": (
            result.config["benchmark_participation_period"]
        ),
        "benchmark_participation_min_return": (
            result.config["benchmark_participation_min_return"]
        ),
        "benchmark_participation_max_selected_excess": (
            result.config["benchmark_participation_max_selected_excess"]
        ),
        "max_sector_weight": result.config["max_sector_weight"],
        "ranking_score_mode": result.config["ranking_score_mode"],
        "relative_strength_weight": result.config["relative_strength_weight"],
        "transaction_cost_bps": result.config["transaction_cost_bps"],
        "commission_bps": result.config["commission_bps"],
        "slippage_bps": result.config["slippage_bps"],
        "spread_cost_bps": result.config["spread_cost_bps"],
        "effective_transaction_cost_bps": (
            result.config["effective_transaction_cost_bps"]
        ),
        "score": risk_regime_score(result),
        "quality_score": dual_momentum_quality_score(result),
        "paper_safe_score": paper_safe_dual_momentum_score(result),
    }


def experiment_fieldnames():
    return [
        "top_n",
        "experiment_name",
        "momentum_periods",
        "rebalance_frequency",
        "regime_confirmation_mode",
        "selection_mode",
        "weighting",
        "max_position_weight",
        "mixed_risk_exposure",
        "risk_off_risk_exposure",
        "fallback_allocation",
        "fallback_symbols",
        "decay_exit_enabled",
        "rank_drop_exit_top_n",
        "rank_deterioration_exit_enabled",
        "rank_deterioration_exit_rank",
        "chop_filter_enabled",
        "chop_risk_exposure",
        "leadership_filter_enabled",
        "relative_strength_filter_enabled",
        "relative_strength_filter_symbol",
        "relative_strength_filter_period",
        "relative_strength_filter_min_excess",
        "benchmark_sleeve_allocation",
        "benchmark_participation_filter_enabled",
        "benchmark_participation_period",
        "benchmark_participation_min_return",
        "benchmark_participation_max_selected_excess",
        "max_sector_weight",
        "ranking_score_mode",
        "relative_strength_weight",
        "quality_filter_enabled",
        "quality_require_momentum_improving",
        "cooldown_enabled",
        "short_term_weakness_penalty_enabled",
        "short_term_weakness_penalty_floor",
        "short_term_weakness_penalty_weight",
        "rank_hysteresis_enabled",
        "rank_hysteresis_margin",
        "rank_hysteresis_max_rank",
        "max_rebalance_replacements",
        "replacement_score_gap",
        "rebalance_min_trade_weight",
        "rebalance_drift_band",
        "strict_drawdown_kill_switch",
        "use_asset_trend_filter",
        "min_breadth_percent",
        "breadth_scaled_exposure_enabled",
        "drawdown_recovery_scaling_enabled",
        "volatility_shock_filter_enabled",
        "target_volatility",
        "max_drawdown_guard",
        "transaction_cost_bps",
        "commission_bps",
        "slippage_bps",
        "spread_cost_bps",
        "effective_transaction_cost_bps",
        "return",
        "cagr",
        "calmar",
        "benchmark_return",
        "equal_weight_return",
        "excess_vs_benchmark",
        "excess_vs_equal_weight",
        "sharpe",
        "max_drawdown",
        "turnover_percent",
        "annualized_turnover_percent",
        "turnover_per_rebalance_percent",
        "rebalance_count",
        "estimated_cost",
        "cost_drag_percent",
        "closed_trades",
        "open_trades",
        "quality_score",
        "paper_safe_score",
        "tag",
    ]


def experiment_row(result):
    return {
        "top_n": result.config["top_n"],
        "experiment_name": result.config.get("experiment_name"),
        "momentum_periods": result.config["momentum_periods"],
        "rebalance_frequency": result.config["rebalance_frequency"],
        "regime_confirmation_mode": (
            result.config["regime_confirmation_mode"]
        ),
        "selection_mode": result.config["selection_mode"],
        "weighting": result.config["weighting"],
        "max_position_weight": result.config["max_position_weight"],
        "mixed_risk_exposure": result.config["mixed_risk_exposure"],
        "risk_off_risk_exposure": result.config["risk_off_risk_exposure"],
        "fallback_allocation": result.config["fallback_allocation"],
        "fallback_symbols": result.config["fallback_symbols"],
        "decay_exit_enabled": result.config["decay_exit_enabled"],
        "rank_drop_exit_top_n": result.config["rank_drop_exit_top_n"],
        "rank_deterioration_exit_enabled": (
            result.config["rank_deterioration_exit_enabled"]
        ),
        "rank_deterioration_exit_rank": (
            result.config["rank_deterioration_exit_rank"]
        ),
        "chop_filter_enabled": result.config["chop_filter_enabled"],
        "chop_risk_exposure": result.config["chop_risk_exposure"],
        "leadership_filter_enabled": result.config["leadership_filter_enabled"],
        "relative_strength_filter_enabled": (
            result.config["relative_strength_filter_enabled"]
        ),
        "relative_strength_filter_symbol": (
            result.config["relative_strength_filter_symbol"]
        ),
        "relative_strength_filter_period": (
            result.config["relative_strength_filter_period"]
        ),
        "relative_strength_filter_min_excess": (
            result.config["relative_strength_filter_min_excess"]
        ),
        "benchmark_sleeve_allocation": (
            result.config["benchmark_sleeve_allocation"]
        ),
        "benchmark_participation_filter_enabled": (
            result.config["benchmark_participation_filter_enabled"]
        ),
        "benchmark_participation_period": (
            result.config["benchmark_participation_period"]
        ),
        "benchmark_participation_min_return": (
            result.config["benchmark_participation_min_return"]
        ),
        "benchmark_participation_max_selected_excess": (
            result.config["benchmark_participation_max_selected_excess"]
        ),
        "max_sector_weight": result.config["max_sector_weight"],
        "ranking_score_mode": result.config["ranking_score_mode"],
        "relative_strength_weight": result.config["relative_strength_weight"],
        "quality_filter_enabled": result.config["quality_filter_enabled"],
        "quality_require_momentum_improving": (
            result.config["quality_require_momentum_improving"]
        ),
        "cooldown_enabled": result.config["cooldown_enabled"],
        "short_term_weakness_penalty_enabled": (
            result.config["short_term_weakness_penalty_enabled"]
        ),
        "short_term_weakness_penalty_floor": (
            result.config["short_term_weakness_penalty_floor"]
        ),
        "short_term_weakness_penalty_weight": (
            result.config["short_term_weakness_penalty_weight"]
        ),
        "rank_hysteresis_enabled": result.config["rank_hysteresis_enabled"],
        "rank_hysteresis_margin": result.config["rank_hysteresis_margin"],
        "rank_hysteresis_max_rank": (
            result.config["rank_hysteresis_max_rank"]
        ),
        "max_rebalance_replacements": (
            result.config["max_rebalance_replacements"]
        ),
        "replacement_score_gap": result.config["replacement_score_gap"],
        "rebalance_min_trade_weight": (
            result.config["rebalance_min_trade_weight"]
        ),
        "rebalance_drift_band": result.config.get("rebalance_drift_band", 0),
        "strict_drawdown_kill_switch": (
            result.config["strict_drawdown_kill_switch"]
        ),
        "use_asset_trend_filter": result.config["use_asset_trend_filter"],
        "min_breadth_percent": result.config["min_breadth_percent"],
        "breadth_scaled_exposure_enabled": (
            result.config["breadth_scaled_exposure_enabled"]
        ),
        "drawdown_recovery_scaling_enabled": (
            result.config["drawdown_recovery_scaling_enabled"]
        ),
        "volatility_shock_filter_enabled": (
            result.config["volatility_shock_filter_enabled"]
        ),
        "target_volatility": result.config["target_volatility"],
        "max_drawdown_guard": result.config["max_drawdown_guard"],
        "transaction_cost_bps": result.config["transaction_cost_bps"],
        "commission_bps": result.config["commission_bps"],
        "slippage_bps": result.config["slippage_bps"],
        "spread_cost_bps": result.config["spread_cost_bps"],
        "effective_transaction_cost_bps": (
            result.config["effective_transaction_cost_bps"]
        ),
        "return": result.result.total_return,
        "cagr": result.cagr,
        "calmar": result.calmar,
        "benchmark_return": result.benchmark_return,
        "equal_weight_return": result.equal_weight_return,
        "excess_vs_benchmark": result.excess_return,
        "excess_vs_equal_weight": result.excess_vs_equal_weight,
        "sharpe": result.result.sharpe,
        "max_drawdown": result.result.max_drawdown,
        "turnover_percent": result.turnover_percent,
        "annualized_turnover_percent": result.annualized_turnover_percent,
        "turnover_per_rebalance_percent": (
            result.turnover_per_rebalance_percent
        ),
        "rebalance_count": result.rebalance_count,
        "estimated_cost": result.estimated_cost,
        "cost_drag_percent": result.cost_drag_percent,
        "closed_trades": result.result.closed_trades,
        "open_trades": result.result.open_trades,
        "quality_score": dual_momentum_quality_score(result),
        "paper_safe_score": paper_safe_dual_momentum_score(result),
        "tag": classify_walk_forward_fold_result(result),
    }
