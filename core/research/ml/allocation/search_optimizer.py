from __future__ import annotations

from functools import partial
from typing import Any

from core.research.ml.allocation.allocation_optimizer import (
    bootstrap_paired_comparison,
    build_optimizer_sampler,
    optimizer_candidate_count,
    optimizer_objective_mode,
    score_optimizer_candidate,
)
from core.research.ml.allocation.allocation_v2_variants import grid_variant
from core.research.ml.allocation.exposures import (
    _baseline_definitions,
    _clip_exposure,
    _missing_requirements,
    _variant_exposures,
)
from core.research.ml.allocation.search_exposure_path import _selected_optimizer_exposure_path
from core.research.ml.allocation.search_objective import _drawdown_aware_objective
from core.research.ml.allocation.simulation import (
    _net_period_returns,
    _result_payload,
    _simulate_policy,
    _trading_rank_key,
)
from core.research.ml.allocation.types import AllocationPolicyDefinition, AllocationPolicyResult


def _evaluate_allocation_optimizer(
    rows: list[dict[str, str]],
    probabilities: list[float],
    diagnostics: dict[str, float | None],
    config: dict[str, Any],
    binary_holdout_exposures: list[float],
    *,
    selection_rows: list[dict[str, str]] | None,
    selection_probabilities: list[float] | None,
) -> dict[str, Any]:
    optimizer_config = config.get("allocation_optimizer", {})
    objective_mode = optimizer_objective_mode(config)
    requested_sampler = str(optimizer_config.get("sampler", "random"))
    if not bool(optimizer_config.get("enabled", True)):
        return {
            "method": "disabled",
            "objective_mode": objective_mode,
            "sampler_requested": requested_sampler,
            "sampler_used": "disabled",
            "optuna_available": False,
            "fallback_reason": None,
            "candidate_count": 0,
            "candidates": [],
            "selected_policy": None,
            "skip_reason": "allocation optimizer disabled by configuration",
        }
    sampler = build_optimizer_sampler(config)
    sampler_metadata = sampler.metadata()
    if not selection_rows or not selection_probabilities:
        return {
            "method": sampler.method,
            "objective_mode": objective_mode,
            **sampler_metadata,
            "candidate_count": 0,
            "candidates": [],
            "selected_policy": None,
            "skip_reason": "out-of-fold selection rows are required",
        }
    requirements = (
        "predicted_forward_return_10d|predicted_forward_return_5d",
        "predicted_future_drawdown|predicted_max_adverse_excursion",
        "predicted_future_volatility",
    )
    missing = sorted(set(
        _missing_requirements(requirements, selection_rows, selection_probabilities)
        + _missing_requirements(requirements, rows, probabilities)
    ))
    if missing:
        return {
            "method": sampler.method,
            "objective_mode": objective_mode,
            **sampler_metadata,
            "candidate_count": 0,
            "candidates": [],
            "selected_policy": None,
            "skip_reason": "missing required prediction columns: " + ", ".join(missing),
        }
    trial_count = optimizer_candidate_count(config, sampler)
    baseline_definition = _baseline_definitions()[0]
    selection_champion = _simulate_policy(
        baseline_definition,
        selection_rows,
        [1.0 for _ in selection_rows],
        float(config.get("allocation_transaction_cost_bps", 5.0)),
        diagnostics,
    )
    evaluations = []
    for trial_number in range(trial_count):
        candidate = sampler.suggest(trial_number)
        variant = grid_variant(candidate)
        definition = AllocationPolicyDefinition(
            policy_name=str(candidate["candidate_id"]),
            required_prediction_columns=requirements,
            exposure_builder=partial(_variant_exposures, variant=variant),
            policy_kind="optimizer_candidate",
            mapping_method=variant.mapping_method,
            threshold_fit_scope="out_of_fold_train_predictions",
            overfit_warning=None,
            transaction_cost_bps=variant.transaction_cost_bps,
            exposure_min=variant.min_exposure,
            exposure_max=variant.max_exposure,
        )
        try:
            exposures = [
                _clip_exposure(value, definition)
                for value in _variant_exposures(
                    selection_rows,
                    selection_probabilities,
                    config,
                    variant=variant,
                )
            ]
            result = _simulate_policy(
                definition,
                selection_rows,
                exposures,
                variant.transaction_cost_bps,
                diagnostics,
            )
        except (TypeError, ValueError):
            sampler.observe(candidate, None)
            continue
        diagnostic_objective = _drawdown_aware_objective(result, selection_champion, config)
        objective_metrics = score_optimizer_candidate(
            diagnostic_objective=diagnostic_objective,
            exposure_path=_selected_optimizer_exposure_path(
                selection_rows,
                exposures,
                variant.transaction_cost_bps,
                variant,
            ),
            config=config,
            candidate_name=str(candidate["candidate_id"]),
        )
        objective_value = float(objective_metrics["objective_value"])
        sampler.observe(candidate, objective_value)
        evaluations.append({
            "candidate": candidate,
            "variant": variant,
            "result": result,
            "objective": objective_value,
            "objective_metrics": objective_metrics,
        })
    sampler_metadata = sampler.metadata()
    minimize = sampler_metadata.get("study_direction") == "minimize"
    objective_ranked = sorted(
        evaluations,
        key=lambda row: (
            (float(row["objective"]) if minimize else -float(row["objective"]),)
            + _trading_rank_key(row["result"])
        ),
    )
    outcome_ranked = sorted(evaluations, key=lambda row: _trading_rank_key(row["result"]))
    objective_ranks = {
        row["candidate"]["candidate_id"]: rank
        for rank, row in enumerate(objective_ranked, start=1)
    }
    outcome_ranks = {
        row["candidate"]["candidate_id"]: rank
        for rank, row in enumerate(outcome_ranked, start=1)
    }
    candidate_rows = [
        {
            **row["candidate"],
            **_result_payload(row["result"]),
            "objective": row["objective"],
            "objective_value": row["objective"],
            "objective_rank": objective_ranks[row["candidate"]["candidate_id"]],
            "outcome_rank": outcome_ranks[row["candidate"]["candidate_id"]],
            "evaluation_split": "out_of_fold_selection",
            **row["objective_metrics"],
        }
        for row in objective_ranked
    ]
    for row in candidate_rows:
        row.update({
            "sampler_requested": sampler_metadata.get("sampler_requested"),
            "sampler_used": sampler_metadata.get("sampler_used"),
            "optuna_available": sampler_metadata.get("optuna_available"),
            "fallback_reason": sampler_metadata.get("fallback_reason"),
        })
    if not objective_ranked:
        return {
            "method": sampler.method,
            "objective_mode": objective_mode,
            **sampler_metadata,
            "candidate_count": 0,
            "candidates": [],
            "selected_policy": None,
            "skip_reason": "no optimizer candidates evaluated successfully",
        }
    selected = objective_ranked[0]
    selected_variant = selected["variant"]
    selected_definition = AllocationPolicyDefinition(
        policy_name=f"selected_{sampler.sampler_used}_optimizer_diagnostic_policy",
        required_prediction_columns=requirements,
        exposure_builder=partial(_variant_exposures, variant=selected_variant),
        policy_kind="optimizer_diagnostic",
        mapping_method=selected_variant.mapping_method,
        threshold_fit_scope="out_of_fold_train_predictions",
        overfit_warning=None,
        transaction_cost_bps=selected_variant.transaction_cost_bps,
        exposure_min=selected_variant.min_exposure,
        exposure_max=selected_variant.max_exposure,
    )
    holdout_exposures = [
        _clip_exposure(value, selected_definition)
        for value in _variant_exposures(
            rows,
            probabilities,
            config,
            variant=selected_variant,
            fit_rows=selection_rows,
        )
    ]
    holdout_result = _simulate_policy(
        selected_definition,
        rows,
        holdout_exposures,
        selected_variant.transaction_cost_bps,
        diagnostics,
    )
    selected_holdout_path = _selected_optimizer_exposure_path(
        rows,
        holdout_exposures,
        selected_variant.transaction_cost_bps,
        selected_variant,
    )
    holdout_champion = _simulate_policy(
        _baseline_definitions()[0],
        rows,
        [1.0 for _ in rows],
        float(config.get("allocation_transaction_cost_bps", 5.0)),
        diagnostics,
    )
    holdout_objective_metrics = score_optimizer_candidate(
        diagnostic_objective=_drawdown_aware_objective(
            holdout_result,
            holdout_champion,
            config,
        ),
        exposure_path=selected_holdout_path,
        config=config,
        candidate_name=selected_definition.policy_name,
    )
    selected_returns = _net_period_returns(
        rows,
        holdout_exposures,
        selected_variant.transaction_cost_bps,
    )
    baseline_returns = _net_period_returns(
        rows,
        binary_holdout_exposures,
        float(config.get("allocation_transaction_cost_bps", 5.0)),
    )
    bootstrap = bootstrap_paired_comparison(
        selected_returns,
        baseline_returns,
        iterations=int(optimizer_config.get("bootstrap_iterations", 1_000)),
        random_seed=int(optimizer_config.get("bootstrap_random_seed", 84)),
    )
    return {
        "method": sampler.method,
        "objective_mode": objective_mode,
        **sampler_metadata,
        "candidate_count": len(candidate_rows),
        "selection_protocol": "out_of_fold_random_search_then_frozen_holdout_evaluation",
        "objective": config.get("allocation_grid_objective", {}),
        "candidates": candidate_rows,
        "selected_policy": {
            "candidate_id": selected["candidate"]["candidate_id"],
            "parameters": selected["candidate"],
            "selected_params": selected["candidate"],
            "objective": selected["objective"],
            "objective_value": selected["objective"],
            "objective_mode": objective_mode,
            "objective_metrics": selected["objective_metrics"],
            "holdout_objective_metrics": holdout_objective_metrics,
            "selected_by_robustness_objective": (
                objective_mode == "robustness_adjusted_canonical_score"
            ),
            "selection_metrics": _result_payload(selected["result"]),
            "holdout_metrics": _result_payload(holdout_result),
            "frozen_holdout_metrics": _result_payload(holdout_result),
        },
        "selected_optimizer_exposure_path": selected_holdout_path,
        "paired_comparison_vs_binary_overlay": bootstrap,
        "skip_reason": None,
    }
