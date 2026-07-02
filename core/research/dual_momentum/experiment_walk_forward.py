from dataclasses import replace

from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.dual_momentum.scoring import (
    classify_dual_momentum_result,
    dual_momentum_quality_score,
    paper_safe_dual_momentum_score,
    walk_forward_selection_score,
)
from core.research.dual_momentum.experiment_candidates import dual_momentum_candidate_configs

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

