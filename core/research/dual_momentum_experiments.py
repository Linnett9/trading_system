import csv
import json
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

from core.research.dual_momentum_factory import build_dual_momentum_tester
from core.research.dual_momentum_scoring import (
    risk_regime_score,
    dual_momentum_quality_score,
    paper_safe_dual_momentum_score,
    dual_momentum_walk_forward_summary,
)


def dual_momentum_risk_regime_configs(dual_config):
    grid = dual_config.get("risk_regime_experiments", [])

    if not grid:
        grid = [
            {
                "name": "baseline_inverse_vol",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "defensive_assets",
                "overrides": {
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "cash_risk_off",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "scaled_exposure",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "fast_reentry",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": True,
                },
            },
            {
                "name": "scaled_plus_fast_reentry",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.50,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                },
            },
            {
                "name": "scaled_fast_reentry_75",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.75,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                },
            },
            {
                "name": "scaled_fast_reentry_fallback",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.75,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                    "fallback_symbols": ["SPY", "QQQ"],
                    "fallback_allocation": 0.25,
                    "fallback_min_risk_assets": 3,
                },
            },
            {
                "name": "scaled_fast_reentry_decay",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.75,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                    "decay_exit_enabled": True,
                    "decay_momentum_period": 63,
                    "rank_drop_exit_top_n": 7,
                },
            },
            {
                "name": "scaled_fast_reentry_chop",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.75,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                    "chop_filter_enabled": True,
                    "chop_lookback": 63,
                    "min_chop_momentum": 0.02,
                    "chop_risk_exposure": 0.50,
                },
            },
        ]

    neutral_optional_modules = {
        "fallback_allocation": 0.0,
        "decay_exit_enabled": False,
        "rank_drop_exit_top_n": None,
        "chop_filter_enabled": False,
        "quality_filter_enabled": False,
        "quality_require_momentum_improving": False,
        "cooldown_enabled": False,
        "leadership_filter_enabled": False,
        "benchmark_sleeve_allocation": 0.0,
    }

    for item in grid:
        candidate = deepcopy(dual_config)
        candidate.update(neutral_optional_modules)
        candidate.update(item.get("overrides", {}))

        yield {
            "name": item["name"],
            "config": candidate,
        }


def run_dual_momentum_experiments(config, dual_config, candles_by_symbol):
    results = []

    for candidate_config in dual_momentum_candidate_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate_config)
        results.append(tester.run(candles_by_symbol))

    return sorted(
        results,
        key=lambda result: (
            paper_safe_dual_momentum_score(result),
            dual_momentum_quality_score(result),
            result.result.sharpe,
            result.calmar,
            -result.result.max_drawdown,
            -result.annualized_turnover_percent,
        ),
        reverse=True,
    )


def run_dual_momentum_fold_optimization(
    config,
    dual_config,
    candles_by_symbol,
    start_at,
    end_at,
):
    results = []
    candidate_configs = list(dual_momentum_candidate_configs(dual_config))

    print(
        f"Optimizing dual momentum over {len(candidate_configs)} "
        f"candidate configs for {start_at.date()}..{end_at.date()}"
    )

    for candidate_config in candidate_configs:
        tester = build_dual_momentum_tester(config, candidate_config)
        results.append(
            tester.run(
                candles_by_symbol,
                start_at=start_at,
                end_at=end_at,
            )
        )

    return sorted(
        results,
        key=lambda result: (
            paper_safe_dual_momentum_score(result),
            dual_momentum_quality_score(result),
            result.result.sharpe,
            result.calmar,
            -result.result.max_drawdown,
            -result.annualized_turnover_percent,
        ),
        reverse=True,
    )


def dual_momentum_candidate_configs(dual_config):
    grid = dual_config.get("experiment_grid", {})

    top_values = grid.get("top_n", [dual_config.get("top_n", 3)])
    rebalance_values = grid.get(
        "rebalance_frequency",
        [dual_config.get("rebalance_frequency", "monthly")],
    )
    momentum_values = grid.get(
        "momentum_periods",
        [dual_config.get("momentum_periods", [126, 252])],
    )
    asset_filter_values = grid.get(
        "use_asset_trend_filter",
        [dual_config.get("use_asset_trend_filter", True)],
    )
    volatility_values = grid.get(
        "target_volatility",
        [dual_config.get("target_volatility")],
    )
    drawdown_values = grid.get(
        "max_drawdown_guard",
        [dual_config.get("max_drawdown_guard")],
    )
    breadth_values = grid.get(
        "min_breadth_percent",
        [dual_config.get("min_breadth_percent", 0)],
    )
    selection_mode_values = grid.get(
        "selection_mode",
        [dual_config.get("selection_mode", "ranked")],
    )
    min_selection_score_values = grid.get(
        "min_selection_score",
        [dual_config.get("min_selection_score", 0)],
    )
    max_selected_assets_values = grid.get(
        "max_selected_assets",
        [dual_config.get("max_selected_assets")],
    )
    weighting_values = grid.get(
        "weighting",
        [dual_config.get("weighting", "equal")],
    )
    max_position_weight_values = grid.get(
        "max_position_weight",
        [dual_config.get("max_position_weight")],
    )
    strict_kill_switch_values = grid.get(
        "strict_drawdown_kill_switch",
        [dual_config.get("strict_drawdown_kill_switch", False)],
    )
    mixed_exposure_values = grid.get(
        "mixed_risk_exposure",
        [dual_config.get("mixed_risk_exposure", 0.50)],
    )
    risk_off_exposure_values = grid.get(
        "risk_off_risk_exposure",
        [dual_config.get("risk_off_risk_exposure", 0)],
    )
    fallback_allocation_values = grid.get(
        "fallback_allocation",
        [dual_config.get("fallback_allocation", 0)],
    )
    decay_exit_values = grid.get(
        "decay_exit_enabled",
        [dual_config.get("decay_exit_enabled", False)],
    )
    rank_drop_values = grid.get(
        "rank_drop_exit_top_n",
        [dual_config.get("rank_drop_exit_top_n")],
    )
    chop_filter_values = grid.get(
        "chop_filter_enabled",
        [dual_config.get("chop_filter_enabled", False)],
    )
    chop_exposure_values = grid.get(
        "chop_risk_exposure",
        [dual_config.get("chop_risk_exposure", 0.50)],
    )
    quality_filter_values = grid.get(
        "quality_filter_enabled",
        [dual_config.get("quality_filter_enabled", False)],
    )
    quality_improving_values = grid.get(
        "quality_require_momentum_improving",
        [dual_config.get("quality_require_momentum_improving", False)],
    )
    cooldown_values = grid.get(
        "cooldown_enabled",
        [dual_config.get("cooldown_enabled", False)],
    )
    leadership_values = grid.get(
        "leadership_filter_enabled",
        [dual_config.get("leadership_filter_enabled", False)],
    )
    benchmark_sleeve_values = grid.get(
        "benchmark_sleeve_allocation",
        [dual_config.get("benchmark_sleeve_allocation", 0)],
    )
    ranking_score_values = grid.get(
        "ranking_score_mode",
        [dual_config.get("ranking_score_mode", "average_momentum")],
    )

    for (
        top_n,
        rebalance,
        momentum_periods,
        use_asset_filter,
        target_volatility,
        max_drawdown_guard,
        min_breadth_percent,
        selection_mode,
        min_selection_score,
        max_selected_assets,
        weighting,
        max_position_weight,
        strict_drawdown_kill_switch,
        mixed_risk_exposure,
        risk_off_risk_exposure,
        fallback_allocation,
        decay_exit_enabled,
        rank_drop_exit_top_n,
        chop_filter_enabled,
        chop_risk_exposure,
        quality_filter_enabled,
        quality_require_momentum_improving,
        cooldown_enabled,
        leadership_filter_enabled,
        benchmark_sleeve_allocation,
        ranking_score_mode,
    ) in product(
        top_values,
        rebalance_values,
        momentum_values,
        asset_filter_values,
        volatility_values,
        drawdown_values,
        breadth_values,
        selection_mode_values,
        min_selection_score_values,
        max_selected_assets_values,
        weighting_values,
        max_position_weight_values,
        strict_kill_switch_values,
        mixed_exposure_values,
        risk_off_exposure_values,
        fallback_allocation_values,
        decay_exit_values,
        rank_drop_values,
        chop_filter_values,
        chop_exposure_values,
        quality_filter_values,
        quality_improving_values,
        cooldown_values,
        leadership_values,
        benchmark_sleeve_values,
        ranking_score_values,
    ):
        candidate = deepcopy(dual_config)
        candidate.update({
            "top_n": top_n,
            "momentum_periods": momentum_periods,
            "rebalance_frequency": rebalance,
            "use_asset_trend_filter": use_asset_filter,
            "target_volatility": target_volatility,
            "max_drawdown_guard": max_drawdown_guard,
            "min_breadth_percent": min_breadth_percent,
            "selection_mode": selection_mode,
            "min_selection_score": min_selection_score,
            "max_selected_assets": max_selected_assets,
            "weighting": weighting,
            "max_position_weight": max_position_weight,
            "strict_drawdown_kill_switch": strict_drawdown_kill_switch,
            "mixed_risk_exposure": mixed_risk_exposure,
            "risk_off_risk_exposure": risk_off_risk_exposure,
            "fallback_allocation": fallback_allocation,
            "decay_exit_enabled": decay_exit_enabled,
            "rank_drop_exit_top_n": rank_drop_exit_top_n,
            "chop_filter_enabled": chop_filter_enabled,
            "chop_risk_exposure": chop_risk_exposure,
            "quality_filter_enabled": quality_filter_enabled,
            "quality_require_momentum_improving": (
                quality_require_momentum_improving
            ),
            "cooldown_enabled": cooldown_enabled,
            "leadership_filter_enabled": leadership_filter_enabled,
            "benchmark_sleeve_allocation": benchmark_sleeve_allocation,
            "ranking_score_mode": ranking_score_mode,
        })

        yield candidate


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
            fieldnames=[
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
                "mixed_risk_exposure",
                "risk_off_risk_exposure",
                "fallback_allocation",
                "decay_exit_enabled",
                "rank_drop_exit_top_n",
                "chop_filter_enabled",
                "chop_risk_exposure",
                "quality_filter_enabled",
                "quality_require_momentum_improving",
                "cooldown_enabled",
                "leadership_filter_enabled",
                "benchmark_sleeve_allocation",
                "ranking_score_mode",
                "score",
                "quality_score",
                "paper_safe_score",
            ]
            + [str(year) for year in years],
        )
        writer.writeheader()

        for item in results:
            result = item["result"]

            row = {
                "name": item["name"],
                "return": result.result.total_return,
                "benchmark_return": result.benchmark_return,
                "equal_weight_return": result.equal_weight_return,
                "excess_vs_benchmark": result.excess_return,
                "excess_vs_equal_weight": result.excess_vs_equal_weight,
                "sharpe": result.result.sharpe,
                "max_drawdown": result.result.max_drawdown,
                "cagr": result.cagr,
                "calmar": result.calmar,
                "annualized_turnover_percent": (
                    result.annualized_turnover_percent
                ),
                "mixed_risk_exposure": result.config["mixed_risk_exposure"],
                "risk_off_risk_exposure": (
                    result.config["risk_off_risk_exposure"]
                ),
                "fallback_allocation": result.config["fallback_allocation"],
                "decay_exit_enabled": result.config["decay_exit_enabled"],
                "rank_drop_exit_top_n": result.config["rank_drop_exit_top_n"],
                "chop_filter_enabled": result.config["chop_filter_enabled"],
                "chop_risk_exposure": result.config["chop_risk_exposure"],
                "quality_filter_enabled": (
                    result.config["quality_filter_enabled"]
                ),
                "quality_require_momentum_improving": (
                    result.config["quality_require_momentum_improving"]
                ),
                "cooldown_enabled": result.config["cooldown_enabled"],
                "leadership_filter_enabled": (
                    result.config["leadership_filter_enabled"]
                ),
                "benchmark_sleeve_allocation": (
                    result.config["benchmark_sleeve_allocation"]
                ),
                "ranking_score_mode": result.config["ranking_score_mode"],
                "score": risk_regime_score(result),
                "quality_score": dual_momentum_quality_score(result),
                "paper_safe_score": paper_safe_dual_momentum_score(result),
            }

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
            fieldnames=[
                "top_n",
                "momentum_periods",
                "rebalance_frequency",
                "selection_mode",
                "weighting",
                "max_position_weight",
                "mixed_risk_exposure",
                "risk_off_risk_exposure",
                "fallback_allocation",
                "fallback_symbols",
                "decay_exit_enabled",
                "rank_drop_exit_top_n",
                "chop_filter_enabled",
                "chop_risk_exposure",
                "leadership_filter_enabled",
                "benchmark_sleeve_allocation",
                "ranking_score_mode",
                "quality_filter_enabled",
                "quality_require_momentum_improving",
                "cooldown_enabled",
                "strict_drawdown_kill_switch",
                "use_asset_trend_filter",
                "min_breadth_percent",
                "target_volatility",
                "max_drawdown_guard",
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
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "top_n": result.config["top_n"],
                "momentum_periods": result.config["momentum_periods"],
                "rebalance_frequency": result.config["rebalance_frequency"],
                "selection_mode": result.config["selection_mode"],
                "weighting": result.config["weighting"],
                "max_position_weight": result.config["max_position_weight"],
                "mixed_risk_exposure": (
                    result.config["mixed_risk_exposure"]
                ),
                "risk_off_risk_exposure": (
                    result.config["risk_off_risk_exposure"]
                ),
                "fallback_allocation": (
                    result.config["fallback_allocation"]
                ),
                "fallback_symbols": result.config["fallback_symbols"],
                "decay_exit_enabled": result.config["decay_exit_enabled"],
                "rank_drop_exit_top_n": result.config["rank_drop_exit_top_n"],
                "chop_filter_enabled": result.config["chop_filter_enabled"],
                "chop_risk_exposure": result.config["chop_risk_exposure"],
                "leadership_filter_enabled": (
                    result.config["leadership_filter_enabled"]
                ),
                "benchmark_sleeve_allocation": (
                    result.config["benchmark_sleeve_allocation"]
                ),
                "ranking_score_mode": result.config["ranking_score_mode"],
                "quality_filter_enabled": (
                    result.config["quality_filter_enabled"]
                ),
                "quality_require_momentum_improving": (
                    result.config["quality_require_momentum_improving"]
                ),
                "cooldown_enabled": result.config["cooldown_enabled"],
                "strict_drawdown_kill_switch": (
                    result.config["strict_drawdown_kill_switch"]
                ),
                "use_asset_trend_filter": (
                    result.config["use_asset_trend_filter"]
                ),
                "min_breadth_percent": (
                    result.config["min_breadth_percent"]
                ),
                "target_volatility": result.config["target_volatility"],
                "max_drawdown_guard": result.config["max_drawdown_guard"],
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
                "annualized_turnover_percent": (
                    result.annualized_turnover_percent
                ),
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
            })

    return path