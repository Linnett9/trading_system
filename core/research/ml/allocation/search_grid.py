from __future__ import annotations

from functools import partial
from typing import Any

from core.research.ml.allocation.allocation_v2_variants import (
    HOLDOUT_OVERFIT_WARNING,
    OUT_OF_SAMPLE_SELECTION_NOTICE,
    grid_candidate_payloads,
    grid_variant,
)
from core.research.ml.allocation.exposures import (
    _baseline_definitions,
    _clip_exposure,
    _missing_requirements,
    _variant_exposures,
)
from core.research.ml.allocation.search_objective import _drawdown_aware_objective
from core.research.ml.allocation.simulation import _result_payload, _simulate_policy, _trading_rank_key
from core.research.ml.allocation.types import AllocationPolicyDefinition, AllocationPolicyResult


def _evaluate_policy_grid_search(
    rows: list[dict[str, str]],
    probabilities: list[float],
    diagnostics: dict[str, float | None],
    config: dict[str, Any],
    champion_result: AllocationPolicyResult,
    *,
    selection_rows: list[dict[str, str]] | None = None,
    selection_probabilities: list[float] | None = None,
) -> dict[str, Any]:
    requirements = (
        "predicted_forward_return_10d|predicted_forward_return_5d",
        "predicted_future_drawdown|predicted_max_adverse_excursion",
        "predicted_future_volatility",
    )
    uses_out_of_sample_selection = bool(selection_rows) and bool(selection_probabilities)
    fit_rows = selection_rows if uses_out_of_sample_selection else rows
    fit_probabilities = selection_probabilities if uses_out_of_sample_selection else probabilities
    selection_protocol = (
        "out_of_fold_train_selection_then_frozen_holdout_evaluation"
        if uses_out_of_sample_selection
        else "holdout_in_sample_selection_and_evaluation"
    )
    selection_notice = (
        OUT_OF_SAMPLE_SELECTION_NOTICE
        if uses_out_of_sample_selection
        else HOLDOUT_OVERFIT_WARNING
    )
    missing = sorted(set(
        _missing_requirements(requirements, rows, probabilities)
        + _missing_requirements(requirements, fit_rows or [], fit_probabilities or [])
    ))
    if missing:
        return {
            "candidates": [],
            "selected": None,
            "skip_reason": "missing required prediction columns: " + ", ".join(missing),
            "selection_protocol": selection_protocol,
            "selection_notice": selection_notice,
        }
    selection_champion = champion_result
    if uses_out_of_sample_selection:
        baseline_definition = _baseline_definitions()[0]
        selection_champion = _simulate_policy(
            baseline_definition,
            fit_rows or [],
            [1.0 for _ in fit_rows or []],
            float(config.get("allocation_transaction_cost_bps", 5.0)),
            diagnostics,
        )
    evaluations = []
    for candidate in grid_candidate_payloads(config):
        variant = grid_variant(candidate)
        definition = AllocationPolicyDefinition(
            policy_name=str(candidate["candidate_id"]),
            required_prediction_columns=requirements,
            exposure_builder=partial(_variant_exposures, variant=variant),
            policy_kind="grid_search_candidate",
            mapping_method=variant.mapping_method,
            threshold_fit_scope=(
                "out_of_fold_train_predictions"
                if uses_out_of_sample_selection
                else variant.threshold_fit_scope
            ),
            overfit_warning=selection_notice,
            transaction_cost_bps=variant.transaction_cost_bps,
            exposure_min=variant.min_exposure,
            exposure_max=variant.max_exposure,
        )
        try:
            exposures = [
                _clip_exposure(value, definition)
                for value in _variant_exposures(
                    fit_rows or [],
                    fit_probabilities or [],
                    config,
                    variant=variant,
                    fit_rows=None,
                )
            ]
            result = _simulate_policy(
                definition,
                fit_rows or [],
                exposures,
                variant.transaction_cost_bps,
                diagnostics,
            )
        except (TypeError, ValueError):
            continue
        objective = _drawdown_aware_objective(result, selection_champion, config)
        evaluations.append({
            "candidate": candidate,
            "variant": variant,
            "definition": definition,
            "exposures": exposures,
            "result": result,
            "objective": objective,
        })
    outcome_ranked = sorted(evaluations, key=lambda row: _trading_rank_key(row["result"]))
    objective_ranked = sorted(
        evaluations,
        key=lambda row: (-float(row["objective"]),) + _trading_rank_key(row["result"]),
    )
    outcome_ranks = {
        row["candidate"]["candidate_id"]: rank
        for rank, row in enumerate(outcome_ranked, start=1)
    }
    objective_ranks = {
        row["candidate"]["candidate_id"]: rank
        for rank, row in enumerate(objective_ranked, start=1)
    }
    candidate_rows = []
    for row in evaluations:
        payload = {
            **row["candidate"],
            **_result_payload(row["result"]),
            "objective": row["objective"],
            "outcome_rank": outcome_ranks[row["candidate"]["candidate_id"]],
            "objective_rank": objective_ranks[row["candidate"]["candidate_id"]],
            "evaluation_split": (
                "out_of_fold_selection" if uses_out_of_sample_selection else "holdout_in_sample"
            ),
            "selection_notice": selection_notice,
        }
        candidate_rows.append(payload)
    selected = objective_ranked[0] if objective_ranked else None
    if selected:
        variant = selected["variant"]
        best_definition = AllocationPolicyDefinition(
            policy_name="best_grid_search_diagnostic_policy",
            required_prediction_columns=requirements,
            exposure_builder=partial(_variant_exposures, variant=variant),
            policy_kind="allocation_policy",
            mapping_method=variant.mapping_method,
            threshold_fit_scope=(
                "out_of_fold_train_predictions"
                if uses_out_of_sample_selection
                else variant.threshold_fit_scope
            ),
            overfit_warning=selection_notice,
            transaction_cost_bps=variant.transaction_cost_bps,
            exposure_min=variant.min_exposure,
            exposure_max=variant.max_exposure,
        )
        holdout_exposures = [
            _clip_exposure(value, best_definition)
            for value in _variant_exposures(
                rows,
                probabilities,
                config,
                variant=variant,
                fit_rows=(fit_rows if uses_out_of_sample_selection else None),
            )
        ]
        selected = {
            **selected,
            "definition": best_definition,
            "selection_result": selected["result"],
            "exposures": holdout_exposures,
            "result": _simulate_policy(
                best_definition,
                rows,
                holdout_exposures,
                variant.transaction_cost_bps,
                diagnostics,
            ),
        }
    return {
        "candidates": candidate_rows,
        "selected": selected,
        "skip_reason": None,
        "selection_protocol": selection_protocol,
        "selection_notice": selection_notice,
        "selection_row_count": len(fit_rows or []),
        "holdout_row_count": len(rows),
    }
