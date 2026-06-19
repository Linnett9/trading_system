from copy import deepcopy
from dataclasses import replace
from itertools import product

from core.research.dual_momentum.reporting import (
    parse_config_date,
    save_dual_momentum_experiments,
    save_dual_momentum_filtered_walk_forward_candidates,
    save_dual_momentum_risk_regime_experiments,
    save_dual_momentum_walk_forward,
)
from core.research.dual_momentum_factory import build_dual_momentum_tester
from core.research.dual_momentum_scoring import (
    classify_dual_momentum_result,
    dual_momentum_quality_score,
    paper_safe_dual_momentum_score,
    walk_forward_selection_score,
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
        "replacement_score_gap": 0.0,
    }

    for item in grid:
        candidate = deepcopy(dual_config)
        candidate.update(neutral_optional_modules)
        candidate.update(item.get("overrides", {}))
        candidate["experiment_name"] = item["name"]

        yield {
            "name": item["name"],
            "config": candidate,
        }


def run_dual_momentum_experiments(config, dual_config, candles_by_symbol):
    results = []

    for candidate_config in dual_momentum_candidate_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate_config)
        results.append(tester.run(candles_by_symbol))

    sorted_results = sorted(
        results,
        key=lambda result: (
            walk_forward_selection_score(result),
            paper_safe_dual_momentum_score(result),
            dual_momentum_quality_score(result),
            result.result.sharpe,
            result.calmar,
            -result.result.max_drawdown,
            -result.annualized_turnover_percent,
        ),
        reverse=True,
    )

    return sorted_results


def _walk_forward_candidate_negative_years(result):
    return sum(
        1
        for value in getattr(result, "annual_returns", {}).values()
        if value < 0
    )


def walk_forward_candidate_hard_filter(result, dual_config, selector_mode=None):
    selector_mode = selector_mode or walk_forward_selector_mode(dual_config)

    if result.result.max_drawdown > dual_config.get(
        f"walk_forward_{selector_mode}_max_in_sample_drawdown",
        dual_config.get("walk_forward_max_in_sample_drawdown", 0.18),
    ):
        return False

    if result.annualized_turnover_percent > dual_config.get(
        f"walk_forward_{selector_mode}_max_in_sample_turnover",
        dual_config.get("walk_forward_max_in_sample_turnover", 6.0),
    ):
        return False

    if result.result.sharpe < dual_config.get(
        f"walk_forward_{selector_mode}_min_in_sample_sharpe",
        dual_config.get("walk_forward_min_in_sample_sharpe", 1.0),
    ):
        return False

    if _walk_forward_candidate_negative_years(result) > dual_config.get(
        "walk_forward_max_negative_years",
        1,
    ):
        return False

    require_excess = dual_config.get(
        f"walk_forward_{selector_mode}_require_positive_excess",
        selector_mode == "production",
    )

    if require_excess and result.excess_return <= 0:
        return False

    if require_excess and result.excess_vs_equal_weight <= 0:
        return False

    max_position_weight = result.config.get("max_position_weight")
    if (
        max_position_weight is not None
        and max_position_weight > dual_config.get(
            "walk_forward_max_position_weight",
            0.28,
        )
    ):
        return False

    return True


def walk_forward_filter_reasons(result, dual_config, selector_mode=None):
    selector_mode = selector_mode or walk_forward_selector_mode(dual_config)
    reasons = []

    max_drawdown = dual_config.get(
        f"walk_forward_{selector_mode}_max_in_sample_drawdown",
        dual_config.get("walk_forward_max_in_sample_drawdown", 0.18),
    )
    max_turnover = dual_config.get(
        f"walk_forward_{selector_mode}_max_in_sample_turnover",
        dual_config.get("walk_forward_max_in_sample_turnover", 6.0),
    )
    min_sharpe = dual_config.get(
        f"walk_forward_{selector_mode}_min_in_sample_sharpe",
        dual_config.get("walk_forward_min_in_sample_sharpe", 1.0),
    )
    require_excess = dual_config.get(
        f"walk_forward_{selector_mode}_require_positive_excess",
        selector_mode == "production",
    )

    if result.result.max_drawdown > max_drawdown:
        reasons.append("drawdown")

    if result.annualized_turnover_percent > max_turnover:
        reasons.append("turnover")

    if result.result.sharpe < min_sharpe:
        reasons.append("sharpe")

    if _walk_forward_candidate_negative_years(result) > dual_config.get(
        "walk_forward_max_negative_years",
        1,
    ):
        reasons.append("negative_years")

    if require_excess and result.excess_return <= 0:
        reasons.append("benchmark")

    if require_excess and result.excess_vs_equal_weight <= 0:
        reasons.append("equal_weight")

    max_position_weight = result.config.get("max_position_weight")
    if (
        max_position_weight is not None
        and max_position_weight > dual_config.get(
            "walk_forward_max_position_weight",
            0.28,
        )
    ):
        reasons.append("max_weight")

    return reasons


def run_dual_momentum_fold_optimization(
    config,
    dual_config,
    candles_by_symbol,
    start_at,
    end_at,
):
    results = []
    candidate_configs = list(walk_forward_candidate_configs(dual_config))
    selector_mode = walk_forward_selector_mode(dual_config)

    print(
        f"Optimizing dual momentum over {len(candidate_configs)} "
        f"candidate configs for {start_at.date()}..{end_at.date()} "
        f"(selector={selector_mode})"
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

    for index, result in enumerate(results):
        reasons = walk_forward_filter_reasons(
            result,
            dual_config,
            selector_mode,
        )
        results[index] = replace(
            result,
            walk_forward_filter_reasons=reasons,
            walk_forward_filter_passed=not reasons,
            walk_forward_selector_mode=selector_mode,
        )

    hard_filtered_results = results
    if selector_mode in {"paper", "production"}:
        hard_filtered_results = [
            result
            for result in results
            if walk_forward_candidate_hard_filter(
                result,
                dual_config,
                selector_mode,
            )
        ]

    print(
        "Walk-forward training filter | "
        f"before={len(results)} | after={len(hard_filtered_results)}"
    )

    filter_fallback = not bool(hard_filtered_results)

    for index, result in enumerate(results):
        results[index] = replace(
            result,
            walk_forward_filter_fallback=filter_fallback,
        )

    if hard_filtered_results:
        results = hard_filtered_results
    else:
        print(
            "Walk-forward training filter warning | "
            "no candidates survived; falling back to scored candidates"
        )

    sorted_results = sorted(
        results,
        key=lambda result: (
            walk_forward_selection_score(result),
            paper_safe_dual_momentum_score(result),
            dual_momentum_quality_score(result),
            result.result.sharpe,
            result.calmar,
            -result.result.max_drawdown,
            -result.annualized_turnover_percent,
        ),
        reverse=True,
    )

    allowed_tags = set(
        dual_config.get(
            "walk_forward_allowed_candidate_tags",
            [],
        )
        or []
    )
    if allowed_tags:
        preferred_results = [
            result
            for result in sorted_results
            if (
                classify_dual_momentum_result(result) in allowed_tags
                or filter_fallback
            )
        ]

        if preferred_results:
            return preferred_results

    if not dual_config.get("walk_forward_exclude_rejected_configs", False):
        return sorted_results

    accepted_results = [
        result
        for result in sorted_results
        if not classify_dual_momentum_result(result).startswith("rejected")
    ]

    return accepted_results or sorted_results


def walk_forward_candidate_configs(dual_config):
    candidate_configs = list(dual_momentum_candidate_configs(dual_config))
    initial_count = len(candidate_configs)
    selector_mode = walk_forward_selector_mode(dual_config)

    if not dual_config.get("walk_forward_named_variants_only", False):
        filtered = [
            candidate
            for candidate in candidate_configs
            if walk_forward_selector_config_allowed(candidate, selector_mode)
        ]
        print(
            "Walk-forward config filter | "
            f"selector={selector_mode} | before={initial_count} | "
            f"after={len(filtered)}"
        )
        return filtered or candidate_configs

    named_configs = [
        candidate
        for candidate in candidate_configs
        if candidate.get("experiment_name") != "grid"
    ]

    excluded_patterns = [
        pattern.lower()
        for pattern in dual_config.get(
            "walk_forward_excluded_name_patterns",
            [],
        )
    ]
    if excluded_patterns:
        filtered_configs = [
            candidate
            for candidate in named_configs
            if not _matches_any_name_pattern(
                candidate.get("experiment_name", ""),
                excluded_patterns,
            )
        ]
        named_configs = filtered_configs or named_configs

    before_selector_count = len(named_configs)
    named_configs = [
        candidate
        for candidate in named_configs
        if walk_forward_selector_config_allowed(candidate, selector_mode)
    ]
    print(
        "Walk-forward config filter | "
        f"selector={selector_mode} | before={initial_count} | "
        f"named={before_selector_count} | after={len(named_configs)}"
    )

    return named_configs or candidate_configs


def walk_forward_selector_mode(dual_config):
    return dual_config.get("walk_forward_selector_mode", "paper").lower()


def walk_forward_selector_config_allowed(candidate, selector_mode):
    if selector_mode == "research":
        return True

    if selector_mode == "production":
        return bool(candidate.get("eligible_for_production_selector", False))

    return bool(candidate.get("eligible_for_paper_selector", True))


def _matches_any_name_pattern(name, patterns):
    normalized_name = name.lower()
    return any(pattern in normalized_name for pattern in patterns)


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
    regime_confirmation_mode_values = grid.get(
        "regime_confirmation_mode",
        [dual_config.get("regime_confirmation_mode", "primary")],
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
    breadth_scaled_exposure_values = grid.get(
        "breadth_scaled_exposure_enabled",
        [dual_config.get("breadth_scaled_exposure_enabled", False)],
    )
    drawdown_recovery_scaling_values = grid.get(
        "drawdown_recovery_scaling_enabled",
        [dual_config.get("drawdown_recovery_scaling_enabled", False)],
    )
    volatility_shock_values = grid.get(
        "volatility_shock_filter_enabled",
        [dual_config.get("volatility_shock_filter_enabled", False)],
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
    rank_deterioration_exit_values = grid.get(
        "rank_deterioration_exit_enabled",
        [dual_config.get("rank_deterioration_exit_enabled", False)],
    )
    rank_deterioration_exit_rank_values = grid.get(
        "rank_deterioration_exit_rank",
        [dual_config.get("rank_deterioration_exit_rank")],
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
    short_term_penalty_values = grid.get(
        "short_term_weakness_penalty_enabled",
        [dual_config.get("short_term_weakness_penalty_enabled", False)],
    )
    short_term_penalty_floor_values = grid.get(
        "short_term_weakness_penalty_floor",
        [dual_config.get("short_term_weakness_penalty_floor", -0.02)],
    )
    short_term_penalty_weight_values = grid.get(
        "short_term_weakness_penalty_weight",
        [dual_config.get("short_term_weakness_penalty_weight", 1.0)],
    )
    rank_hysteresis_values = grid.get(
        "rank_hysteresis_enabled",
        [dual_config.get("rank_hysteresis_enabled", False)],
    )
    rank_hysteresis_margin_values = grid.get(
        "rank_hysteresis_margin",
        [dual_config.get("rank_hysteresis_margin", 2)],
    )
    rank_hysteresis_max_rank_values = grid.get(
        "rank_hysteresis_max_rank",
        [dual_config.get("rank_hysteresis_max_rank")],
    )
    max_rebalance_replacements_values = grid.get(
        "max_rebalance_replacements",
        [dual_config.get("max_rebalance_replacements")],
    )
    replacement_score_gap_values = grid.get(
        "replacement_score_gap",
        [dual_config.get("replacement_score_gap", 0)],
    )
    rebalance_min_trade_weight_values = grid.get(
        "rebalance_min_trade_weight",
        [dual_config.get("rebalance_min_trade_weight", 0)],
    )
    leadership_values = grid.get(
        "leadership_filter_enabled",
        [dual_config.get("leadership_filter_enabled", False)],
    )
    relative_strength_filter_values = grid.get(
        "relative_strength_filter_enabled",
        [dual_config.get("relative_strength_filter_enabled", False)],
    )
    relative_strength_filter_symbol_values = grid.get(
        "relative_strength_filter_symbol",
        [dual_config.get("relative_strength_filter_symbol", "SPY")],
    )
    relative_strength_filter_period_values = grid.get(
        "relative_strength_filter_period",
        [dual_config.get("relative_strength_filter_period", 63)],
    )
    relative_strength_filter_min_excess_values = grid.get(
        "relative_strength_filter_min_excess",
        [dual_config.get("relative_strength_filter_min_excess", 0)],
    )
    benchmark_sleeve_values = grid.get(
        "benchmark_sleeve_allocation",
        [dual_config.get("benchmark_sleeve_allocation", 0)],
    )
    ranking_score_values = grid.get(
        "ranking_score_mode",
        [dual_config.get("ranking_score_mode", "average_momentum")],
    )
    relative_strength_weight_values = grid.get(
        "relative_strength_weight",
        [dual_config.get("relative_strength_weight", 0.25)],
    )
    transaction_cost_values = grid.get(
        "transaction_cost_bps",
        [dual_config.get("transaction_cost_bps", 2.0)],
    )
    commission_values = grid.get(
        "commission_bps",
        [dual_config.get("commission_bps", 0.0)],
    )
    slippage_values = grid.get(
        "slippage_bps",
        [dual_config.get("slippage_bps", 0.0)],
    )
    spread_cost_values = grid.get(
        "spread_cost_bps",
        [dual_config.get("spread_cost_bps", 0.0)],
    )

    for (
        top_n,
        rebalance,
        momentum_periods,
        regime_confirmation_mode,
        use_asset_filter,
        target_volatility,
        max_drawdown_guard,
        min_breadth_percent,
        breadth_scaled_exposure_enabled,
        drawdown_recovery_scaling_enabled,
        volatility_shock_filter_enabled,
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
        rank_deterioration_exit_enabled,
        rank_deterioration_exit_rank,
        chop_filter_enabled,
        chop_risk_exposure,
        quality_filter_enabled,
        quality_require_momentum_improving,
        cooldown_enabled,
        short_term_weakness_penalty_enabled,
        short_term_weakness_penalty_floor,
        short_term_weakness_penalty_weight,
        rank_hysteresis_enabled,
        rank_hysteresis_margin,
        rank_hysteresis_max_rank,
        max_rebalance_replacements,
        replacement_score_gap,
        rebalance_min_trade_weight,
        leadership_filter_enabled,
        relative_strength_filter_enabled,
        relative_strength_filter_symbol,
        relative_strength_filter_period,
        relative_strength_filter_min_excess,
        benchmark_sleeve_allocation,
        ranking_score_mode,
        relative_strength_weight,
        transaction_cost_bps,
        commission_bps,
        slippage_bps,
        spread_cost_bps,
    ) in product(
        top_values,
        rebalance_values,
        momentum_values,
        regime_confirmation_mode_values,
        asset_filter_values,
        volatility_values,
        drawdown_values,
        breadth_values,
        breadth_scaled_exposure_values,
        drawdown_recovery_scaling_values,
        volatility_shock_values,
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
        rank_deterioration_exit_values,
        rank_deterioration_exit_rank_values,
        chop_filter_values,
        chop_exposure_values,
        quality_filter_values,
        quality_improving_values,
        cooldown_values,
        short_term_penalty_values,
        short_term_penalty_floor_values,
        short_term_penalty_weight_values,
        rank_hysteresis_values,
        rank_hysteresis_margin_values,
        rank_hysteresis_max_rank_values,
        max_rebalance_replacements_values,
        replacement_score_gap_values,
        rebalance_min_trade_weight_values,
        leadership_values,
        relative_strength_filter_values,
        relative_strength_filter_symbol_values,
        relative_strength_filter_period_values,
        relative_strength_filter_min_excess_values,
        benchmark_sleeve_values,
        ranking_score_values,
        relative_strength_weight_values,
        transaction_cost_values,
        commission_values,
        slippage_values,
        spread_cost_values,
    ):
        candidate = deepcopy(dual_config)
        candidate.update({
            "experiment_name": "grid",
            "top_n": top_n,
            "momentum_periods": momentum_periods,
            "rebalance_frequency": rebalance,
            "regime_confirmation_mode": regime_confirmation_mode,
            "use_asset_trend_filter": use_asset_filter,
            "target_volatility": target_volatility,
            "max_drawdown_guard": max_drawdown_guard,
            "min_breadth_percent": min_breadth_percent,
            "breadth_scaled_exposure_enabled": (
                breadth_scaled_exposure_enabled
            ),
            "drawdown_recovery_scaling_enabled": (
                drawdown_recovery_scaling_enabled
            ),
            "volatility_shock_filter_enabled": (
                volatility_shock_filter_enabled
            ),
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
            "rank_deterioration_exit_enabled": (
                rank_deterioration_exit_enabled
            ),
            "rank_deterioration_exit_rank": rank_deterioration_exit_rank,
            "chop_filter_enabled": chop_filter_enabled,
            "chop_risk_exposure": chop_risk_exposure,
            "quality_filter_enabled": quality_filter_enabled,
            "quality_require_momentum_improving": (
                quality_require_momentum_improving
            ),
            "cooldown_enabled": cooldown_enabled,
            "short_term_weakness_penalty_enabled": (
                short_term_weakness_penalty_enabled
            ),
            "short_term_weakness_penalty_floor": (
                short_term_weakness_penalty_floor
            ),
            "short_term_weakness_penalty_weight": (
                short_term_weakness_penalty_weight
            ),
            "rank_hysteresis_enabled": rank_hysteresis_enabled,
            "rank_hysteresis_margin": rank_hysteresis_margin,
            "rank_hysteresis_max_rank": rank_hysteresis_max_rank,
            "max_rebalance_replacements": max_rebalance_replacements,
            "replacement_score_gap": replacement_score_gap,
            "rebalance_min_trade_weight": rebalance_min_trade_weight,
            "leadership_filter_enabled": leadership_filter_enabled,
            "relative_strength_filter_enabled": (
                relative_strength_filter_enabled
            ),
            "relative_strength_filter_symbol": (
                relative_strength_filter_symbol
            ),
            "relative_strength_filter_period": (
                relative_strength_filter_period
            ),
            "relative_strength_filter_min_excess": (
                relative_strength_filter_min_excess
            ),
            "benchmark_sleeve_allocation": benchmark_sleeve_allocation,
            "ranking_score_mode": ranking_score_mode,
            "relative_strength_weight": relative_strength_weight,
            "transaction_cost_bps": transaction_cost_bps,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "spread_cost_bps": spread_cost_bps,
        })

        yield candidate

    for item in dual_config.get("experiment_variants", []):
        candidate = deepcopy(dual_config)
        candidate.update(item.get("overrides", {}))
        candidate["experiment_name"] = item["name"]
        yield candidate


__all__ = [
    "dual_momentum_candidate_configs",
    "dual_momentum_risk_regime_configs",
    "parse_config_date",
    "run_dual_momentum_experiments",
    "run_dual_momentum_fold_optimization",
    "save_dual_momentum_experiments",
    "save_dual_momentum_filtered_walk_forward_candidates",
    "save_dual_momentum_risk_regime_experiments",
    "save_dual_momentum_walk_forward",
    "walk_forward_candidate_configs",
]
