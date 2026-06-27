from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from core.research.performance_metrics import calmar_ratio
from core.research.ml.allocation_v2_variants import (
    AllocationVariant,
    HOLDOUT_OVERFIT_WARNING,
    OUT_OF_SAMPLE_SELECTION_NOTICE,
    grid_candidate_payloads,
    grid_variant,
    map_variant_scores,
    named_policy_variants,
)
from core.research.ml.allocation_optimizer import (
    bootstrap_paired_comparison,
    build_optimizer_sampler,
    optimizer_candidate_count,
    write_optimizer_reports,
)


POLICY_VERSION = "2.0.0"
RESEARCH_METADATA = {
    "trading_impact": "none",
    "research_only": True,
    "production_validated": False,
}


@dataclass(frozen=True)
class AllocationPolicyDefinition:
    policy_name: str
    required_prediction_columns: tuple[str, ...]
    exposure_builder: Callable[
        [list[dict[str, str]], list[float], dict[str, Any]],
        list[float],
    ]
    policy_version: str = POLICY_VERSION
    policy_kind: str = "allocation_policy"
    mapping_method: str = "fixed"
    threshold_fit_scope: str = "fixed_configured_thresholds"
    overfit_warning: str | None = None
    transaction_cost_bps: float | None = None
    exposure_min: float = 0.0
    exposure_max: float = 1.0


@dataclass(frozen=True)
class AllocationPolicyResult:
    policy_name: str
    policy_version: str
    policy_kind: str
    mapping_method: str
    threshold_fit_scope: str
    overfit_warning: str | None
    transaction_cost_bps: float
    required_prediction_columns: tuple[str, ...]
    exposure_min: float
    exposure_max: float
    trading_impact: str
    research_only: bool
    production_validated: bool
    available: bool
    skip_reason: str | None
    forecast_source: str
    total_return: float
    annualized_return: float | None
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    turnover: float
    estimated_transaction_costs: float
    return_per_unit_drawdown: float | None
    mean_exposure: float
    median_exposure: float
    min_exposure: float
    max_exposure: float
    exposure_std: float
    days_at_0_exposure: int
    days_at_full_exposure: int
    number_of_exposure_changes: int
    average_exposure_change: float
    maximum_one_period_exposure_change: float
    pct_periods_at_0_exposure: float
    pct_periods_at_20_exposure: float
    pct_periods_at_50_exposure: float
    pct_periods_at_80_exposure: float
    pct_periods_at_100_exposure: float
    evaluated_periods: int
    performance_when_exposure_reduced: dict[str, float | int]
    performance_when_exposure_high: dict[str, float | int]
    performance_during_worst_drawdown_windows: dict[str, float | int]
    drawdown_impact: dict[str, float | str]
    prediction_to_exposure_diagnostics: dict[str, float | None]
    balanced_accuracy: float | None
    brier_score: float | None
    expected_calibration_error: float | None


@dataclass(frozen=True)
class AllocationV2Paths:
    comparison_json: Path
    comparison_csv: Path
    leaderboard_markdown: Path
    shadow_overlay_json: Path
    diagnostics_json: Path
    diagnostics_markdown: Path
    grid_search_csv: Path
    grid_search_json: Path
    grid_search_markdown: Path
    optimizer_candidates_csv: Path
    optimizer_results_json: Path
    optimizer_report_markdown: Path
    selected_optimizer_exposure_path_csv: Path
    selected_optimizer_exposure_path_json: Path


def write_allocation_v2_reports(
    output_dir: Path,
    rows: list[dict[str, str]],
    meta_probabilities: list[float],
    diagnostics: dict[str, float | None],
    config: dict[str, Any],
    selection_rows: list[dict[str, str]] | None = None,
    selection_meta_probabilities: list[float] | None = None,
) -> AllocationV2Paths:
    if len(rows) != len(meta_probabilities):
        raise ValueError("Allocation rows and meta probabilities must have equal length")
    if (selection_rows is None) != (selection_meta_probabilities is None):
        raise ValueError(
            "Allocation selection rows and probabilities must be provided together"
        )
    if selection_rows is not None and len(selection_rows) != len(
        selection_meta_probabilities or []
    ):
        raise ValueError(
            "Allocation selection rows and probabilities must have equal length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    transaction_cost_bps = float(config.get("allocation_transaction_cost_bps", 5.0))
    policy_definitions = _policy_definitions(config, selection_rows=selection_rows)
    exposures_by_policy, skipped = _evaluate_policy_exposures(
        rows,
        meta_probabilities,
        config,
        selection_rows=selection_rows,
    )
    simulations: list[AllocationPolicyResult] = []
    for definition in policy_definitions:
        exposures = exposures_by_policy.get(definition.policy_name)
        if exposures is None:
            continue
        try:
            simulations.append(_simulate_policy(
                definition,
                rows,
                exposures,
                definition.transaction_cost_bps
                if definition.transaction_cost_bps is not None
                else transaction_cost_bps,
                diagnostics,
            ))
        except (TypeError, ValueError) as exc:
            skipped[definition.policy_name] = f"policy_evaluation_failed: {exc}"

    baseline_exposures = {
        definition.policy_name: definition.exposure_builder(
            rows,
            meta_probabilities,
            config,
        )
        for definition in _baseline_definitions()
    }
    baseline_simulations = [
        _simulate_policy(
            definition,
            rows,
            baseline_exposures[definition.policy_name],
            transaction_cost_bps,
            diagnostics,
        )
        for definition in _baseline_definitions()
    ]
    champion_result = next(
        result
        for result in baseline_simulations
        if result.policy_name == "champion_baseline"
    )
    grid_search = _evaluate_policy_grid_search(
        rows,
        meta_probabilities,
        diagnostics,
        config,
        champion_result,
        selection_rows=selection_rows,
        selection_probabilities=selection_meta_probabilities,
    )
    selected_grid = grid_search.get("selected")
    if selected_grid:
        selected_definition = selected_grid["definition"]
        selected_exposures = selected_grid["exposures"]
        policy_definitions = policy_definitions + (selected_definition,)
        exposures_by_policy[selected_definition.policy_name] = selected_exposures
        simulations.append(selected_grid["result"])
    optimizer_report = _evaluate_allocation_optimizer(
        rows,
        meta_probabilities,
        diagnostics,
        config,
        exposures_by_policy.get("binary_exposure_overlay", []),
        selection_rows=selection_rows,
        selection_probabilities=selection_meta_probabilities,
    )
    ranked = sorted(simulations + baseline_simulations, key=_trading_rank_key)
    ranked_payloads = [
        {"rank": index, **_result_payload(result)}
        for index, result in enumerate(ranked, start=1)
    ]
    _add_robustness_flags(
        ranked_payloads,
        config,
    )
    available_payloads = [
        row for row in ranked_payloads
        if row["policy_kind"] == "allocation_policy"
    ]
    baseline_payloads = [
        row for row in ranked_payloads
        if row["policy_kind"] == "diagnostic_baseline"
    ]
    skipped_payloads = [
        _unavailable_policy_payload(definition, skipped[definition.policy_name])
        for definition in policy_definitions
        if definition.policy_name in skipped
    ]
    policy_payloads = available_payloads + skipped_payloads
    comparison_rows = ranked_payloads + skipped_payloads
    comparison = {
        "mode": "allocation_policy_comparison_v2_research_only",
        "policy_version": POLICY_VERSION,
        "ranking_basis": [
            "total_return",
            "max_drawdown",
            "sharpe",
            "sortino",
            "calmar",
            "return_per_unit_drawdown",
            "turnover",
            "estimated_transaction_costs",
        ],
        "classification_metrics_role": "diagnostics_only",
        "transaction_cost_bps": transaction_cost_bps,
        "policies": policy_payloads,
        "baselines": baseline_payloads,
        "ranking": [
            {
                "rank": row["rank"],
                "policy_name": row["policy_name"],
                "policy_kind": row["policy_kind"],
                "total_return": row["total_return"],
                "max_drawdown": row["max_drawdown"],
                "sharpe": row["sharpe"],
                "sortino": row["sortino"],
                "calmar": row["calmar"],
                "return_per_unit_drawdown": row["return_per_unit_drawdown"],
                "turnover": row["turnover"],
                "estimated_transaction_costs": row[
                    "estimated_transaction_costs"
                ],
            }
            for row in ranked_payloads
        ],
        "available_policy_count": len(available_payloads),
        "skipped_policy_count": len(skipped_payloads),
        "automatic_promotion": False,
        "winners": _comparison_winners(ranked_payloads),
        "grid_search_diagnostic": (
            {
                "policy_name": selected_grid["result"].policy_name,
                "candidate_id": selected_grid["candidate"]["candidate_id"],
                "objective": selected_grid["objective"],
                "selection_protocol": grid_search["selection_protocol"],
                "selection_notice": grid_search["selection_notice"],
                "selection_metrics": _result_payload(
                    selected_grid["selection_result"]
                ),
                "holdout_metrics": _result_payload(selected_grid["result"]),
            }
            if selected_grid
            else None
        ),
        **RESEARCH_METADATA,
    }

    paths = AllocationV2Paths(
        comparison_json=output_dir / "allocation_policy_comparison.json",
        comparison_csv=output_dir / "allocation_policy_comparison.csv",
        leaderboard_markdown=output_dir / "allocation_policy_leaderboard.md",
        shadow_overlay_json=output_dir / "allocation_shadow_overlay.json",
        diagnostics_json=output_dir / "allocation_policy_diagnostics.json",
        diagnostics_markdown=output_dir / "allocation_policy_diagnostics.md",
        grid_search_csv=output_dir / "allocation_policy_grid_search.csv",
        grid_search_json=output_dir / "allocation_policy_grid_search.json",
        grid_search_markdown=output_dir / "allocation_policy_grid_search.md",
        optimizer_candidates_csv=output_dir / "allocation_optimizer_candidates.csv",
        optimizer_results_json=output_dir / "allocation_optimizer_results.json",
        optimizer_report_markdown=output_dir / "allocation_optimizer_report.md",
        selected_optimizer_exposure_path_csv=(
            output_dir / "selected_optimizer_exposure_path.csv"
        ),
        selected_optimizer_exposure_path_json=(
            output_dir / "selected_optimizer_exposure_path.json"
        ),
    )
    paths.comparison_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_comparison_csv(paths.comparison_csv, comparison_rows)
    _write_leaderboard(paths.leaderboard_markdown, ranked_payloads, skipped_payloads)
    selected = ranked[0] if ranked else None
    paths.shadow_overlay_json.write_text(
        json.dumps({
            "mode": "allocation_shadow_overlay_v2_research_only",
            "policy_version": POLICY_VERSION,
            "selected_for_research_comparison": (
                selected.policy_name if selected else None
            ),
            "selection_is_not_promotion": True,
            "policies": {
                definition.policy_name: _shadow_policy_payload(
                    definition,
                    rows,
                    exposures_by_policy.get(definition.policy_name),
                    skipped.get(definition.policy_name),
                )
                for definition in policy_definitions
            },
            "baselines": {
                definition.policy_name: _shadow_policy_payload(
                    definition,
                    rows,
                    baseline_exposures[definition.policy_name],
                    None,
                )
                for definition in _baseline_definitions()
            },
            **RESEARCH_METADATA,
        }, indent=2),
        encoding="utf-8",
    )
    diagnostics_payload = {
        "mode": "allocation_policy_diagnostics_v2_research_only",
        "policy_version": POLICY_VERSION,
        "sanity_reports": comparison_rows,
        "robustness_thresholds": {
            "exposure_changes_too_often_rate": float(
                config.get("allocation_exposure_change_rate_warning", 0.80)
            ),
            "mostly_extreme_exposure_percentage": float(
                config.get("allocation_mostly_extreme_percentage", 80.0)
            ),
            "return_destruction_minimum": float(
                config.get("allocation_return_destruction_minimum", 0.02)
            ),
        },
        **RESEARCH_METADATA,
    }
    paths.diagnostics_json.write_text(
        json.dumps(diagnostics_payload, indent=2),
        encoding="utf-8",
    )
    _write_diagnostics_markdown(
        paths.diagnostics_markdown,
        ranked_payloads,
        skipped_payloads,
    )
    _write_grid_search_reports(paths, grid_search, config)
    write_optimizer_reports(output_dir, optimizer_report)
    _validate_output_consistency(paths)
    return paths


def _policy_definitions(
    config: dict[str, Any] | None = None,
    *,
    selection_rows: list[dict[str, str]] | None = None,
) -> tuple[AllocationPolicyDefinition, ...]:
    base_definitions = (
        AllocationPolicyDefinition(
            policy_name="binary_exposure_overlay",
            required_prediction_columns=("predicted_probability",),
            exposure_builder=_binary_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="return_only_allocation",
            required_prediction_columns=(
                "predicted_forward_return_10d|predicted_forward_return_5d",
            ),
            exposure_builder=_return_only_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="risk_adjusted_allocation",
            required_prediction_columns=(
                "predicted_forward_return_10d|predicted_forward_return_5d",
                "predicted_future_drawdown|predicted_max_adverse_excursion",
                "predicted_future_volatility",
            ),
            exposure_builder=_risk_adjusted_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="meta_ensemble_allocation",
            required_prediction_columns=("predicted_probability",),
            exposure_builder=_meta_exposures,
        ),
    )
    variant_definitions = []
    for variant in named_policy_variants(config or {}):
        uses_frozen_selection = bool(selection_rows) and variant.mapping_method == "quantile"
        required_columns = (
            ("predicted_forward_return_10d|predicted_forward_return_5d",)
            if variant.policy_family == "return_only_allocation"
            else (
                "predicted_forward_return_10d|predicted_forward_return_5d",
                "predicted_future_drawdown|predicted_max_adverse_excursion",
                "predicted_future_volatility",
            )
        )
        variant_definitions.append(AllocationPolicyDefinition(
            policy_name=variant.policy_name,
            required_prediction_columns=required_columns,
            exposure_builder=partial(
                _variant_exposures,
                variant=variant,
                fit_rows=selection_rows,
            ),
            mapping_method=variant.mapping_method,
            threshold_fit_scope=(
                "out_of_fold_train_predictions"
                if uses_frozen_selection
                else variant.threshold_fit_scope
            ),
            overfit_warning=(
                OUT_OF_SAMPLE_SELECTION_NOTICE
                if uses_frozen_selection
                else variant.overfit_warning
            ),
            transaction_cost_bps=variant.transaction_cost_bps,
            exposure_min=variant.min_exposure,
            exposure_max=variant.max_exposure,
        ))
    return base_definitions + tuple(variant_definitions)


def _baseline_definitions() -> tuple[AllocationPolicyDefinition, ...]:
    return (
        AllocationPolicyDefinition(
            policy_name="champion_baseline",
            required_prediction_columns=(),
            exposure_builder=_always_full_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_full_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_full_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_half_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_half_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_zero_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_zero_exposures,
            policy_kind="diagnostic_baseline",
        ),
    )


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
    uses_out_of_sample_selection = bool(selection_rows) and bool(
        selection_probabilities
    )
    fit_rows = selection_rows if uses_out_of_sample_selection else rows
    fit_probabilities = (
        selection_probabilities if uses_out_of_sample_selection else probabilities
    )
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
        + _missing_requirements(
            requirements,
            fit_rows or [],
            fit_probabilities or [],
        )
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
        objective = _drawdown_aware_objective(
            result,
            selection_champion,
            config,
        )
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
                "out_of_fold_selection"
                if uses_out_of_sample_selection
                else "holdout_in_sample"
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


def _drawdown_aware_objective(
    result: AllocationPolicyResult,
    champion: AllocationPolicyResult,
    config: dict[str, Any],
) -> float:
    weights = config.get("allocation_grid_objective", {})
    return (
        float(weights.get("total_return_weight", 1.0)) * result.total_return
        + float(weights.get("max_drawdown_improvement_weight", 0.5))
        * (champion.max_drawdown - result.max_drawdown)
        + float(weights.get("sharpe_weight", 0.25)) * result.sharpe
        - float(weights.get("turnover_penalty_weight", 0.1)) * result.turnover
        - float(weights.get("transaction_cost_penalty_weight", 1.0))
        * result.estimated_transaction_costs
    )


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
    requested_sampler = str(optimizer_config.get("sampler", "random"))
    if not bool(optimizer_config.get("enabled", True)):
        return {
            "method": "disabled",
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
        objective_value = _drawdown_aware_objective(
            result,
            selection_champion,
            config,
        )
        sampler.observe(candidate, objective_value)
        evaluations.append({
            "candidate": candidate,
            "variant": variant,
            "result": result,
            "objective": objective_value,
        })
    sampler_metadata = sampler.metadata()
    minimize = sampler_metadata.get("study_direction") == "minimize"
    objective_ranked = sorted(
        evaluations,
        key=lambda row: (
            float(row["objective"])
            if minimize
            else -float(row["objective"]),
        ) + _trading_rank_key(row["result"]),
    )
    outcome_ranked = sorted(
        evaluations,
        key=lambda row: _trading_rank_key(row["result"]),
    )
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
        **sampler_metadata,
        "candidate_count": len(candidate_rows),
        "selection_protocol": (
            "out_of_fold_random_search_then_frozen_holdout_evaluation"
        ),
        "objective": config.get("allocation_grid_objective", {}),
        "candidates": candidate_rows,
        "selected_policy": {
            "candidate_id": selected["candidate"]["candidate_id"],
            "parameters": selected["candidate"],
            "selected_params": selected["candidate"],
            "objective": selected["objective"],
            "objective_value": selected["objective"],
            "selection_metrics": _result_payload(selected["result"]),
            "holdout_metrics": _result_payload(holdout_result),
            "frozen_holdout_metrics": _result_payload(holdout_result),
        },
        "selected_optimizer_exposure_path": _selected_optimizer_exposure_path(
            rows,
            holdout_exposures,
            selected_variant.transaction_cost_bps,
            selected_variant,
        ),
        "paired_comparison_vs_binary_overlay": bootstrap,
        "skip_reason": None,
    }


def _net_period_returns(
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
) -> list[float]:
    if not exposures:
        return []
    previous_exposure = 1.0
    returns = []
    for _, period_return, exposure in _aggregate_periods(rows, exposures):
        turnover = abs(exposure - previous_exposure)
        cost = turnover * transaction_cost_bps / 10_000.0
        returns.append((period_return * exposure) - cost)
        previous_exposure = exposure
    return returns


def _policy_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> dict[str, list[float]]:
    exposures, _ = _evaluate_policy_exposures(rows, probabilities, config)
    return exposures


def _evaluate_policy_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
    *,
    selection_rows: list[dict[str, str]] | None = None,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    exposures: dict[str, list[float]] = {}
    skipped: dict[str, str] = {}
    for definition in _policy_definitions(config, selection_rows=selection_rows):
        missing = _missing_requirements(
            definition.required_prediction_columns,
            rows,
            probabilities,
        )
        if missing:
            skipped[definition.policy_name] = (
                "missing required prediction columns: " + ", ".join(missing)
            )
            continue
        try:
            raw_exposures = definition.exposure_builder(rows, probabilities, config)
            exposures[definition.policy_name] = [
                _clip_exposure(value, definition)
                for value in raw_exposures
            ]
        except (TypeError, ValueError) as exc:
            skipped[definition.policy_name] = f"invalid policy inputs: {exc}"
    return exposures, skipped


def _missing_requirements(
    requirements: tuple[str, ...],
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> list[str]:
    missing = []
    for requirement in requirements:
        alternatives = requirement.split("|")
        if alternatives == ["predicted_probability"]:
            if len(probabilities) != len(rows) or not all(
                _is_probability(value) for value in probabilities
            ):
                missing.append(requirement)
            continue
        if not rows or not all(
            any(_has_forecast(row, name) for name in alternatives)
            for row in rows
        ):
            missing.append(requirement)
    return missing


def _binary_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del rows
    threshold = float(config.get("decision_threshold", 0.5))
    reduced_exposure = float(config.get("promotion_reduced_exposure", 0.7))
    return [
        reduced_exposure if probability >= threshold else 1.0
        for probability in probabilities
    ]


def _return_only_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities
    return [_score_to_exposure(_return_forecast(row), config) for row in rows]


def _risk_adjusted_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities
    return [_risk_adjusted_exposure(row, config) for row in rows]


def _meta_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del rows, config
    return [_meta_probability_exposure(value) for value in probabilities]


def _variant_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
    *,
    variant: AllocationVariant,
    fit_rows: list[dict[str, str]] | None = None,
) -> list[float]:
    del probabilities, config
    scores = _variant_scores(rows, variant)
    fit_scores = _variant_scores(fit_rows, variant) if fit_rows else None
    dates = [
        str(row.get("rebalance_date") or row.get("date") or "")
        for row in rows
    ]
    return map_variant_scores(
        scores,
        dates,
        variant,
        fit_scores=fit_scores,
    )


def _variant_scores(
    rows: list[dict[str, str]],
    variant: AllocationVariant,
) -> list[float]:
    scores = []
    for row in rows:
        expected_return = _return_forecast(row)
        if variant.policy_family == "return_only_allocation":
            score = variant.return_weight * expected_return
        else:
            drawdowns = _forecast_values(row, "predicted_future_drawdown")
            adverse = _forecast_values(row, "predicted_max_adverse_excursion")
            volatility = _forecast_values(row, "predicted_future_volatility")
            drawdown = mean(drawdowns) if drawdowns else mean(adverse)
            score = (
                (variant.return_weight * expected_return)
                - (variant.drawdown_weight * abs(drawdown))
                - (variant.volatility_weight * mean(volatility))
            )
        scores.append(score)
    return scores


def _always_full_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [1.0 for _ in rows]


def _always_half_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [0.5 for _ in rows]


def _always_zero_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [0.0 for _ in rows]


def _return_forecast(row: dict[str, str]) -> float:
    ten_day = _forecast_values(row, "predicted_forward_return_10d")
    if ten_day:
        return mean(ten_day)
    five_day = _forecast_values(row, "predicted_forward_return_5d")
    return mean(five_day)


def _risk_adjusted_exposure(row: dict[str, str], config: dict[str, Any]) -> float:
    drawdowns = _forecast_values(row, "predicted_future_drawdown")
    adverse = _forecast_values(row, "predicted_max_adverse_excursion")
    volatility = _forecast_values(row, "predicted_future_volatility")
    drawdown = mean(drawdowns) if drawdowns else mean(adverse)
    risk = mean(volatility)
    severe_drawdown = float(config.get("allocation_severe_drawdown", -0.15))
    severe_volatility = float(config.get("allocation_severe_volatility", 0.40))
    if drawdown <= severe_drawdown or risk >= severe_volatility:
        return 0.0
    score = _return_forecast(row) - (0.5 * abs(drawdown)) - (0.25 * risk)
    return _score_to_exposure(score, config)


def _score_to_exposure(score: float, config: dict[str, Any]) -> float:
    thresholds = config.get("allocation_score_thresholds", {})
    levels = (
        (float(thresholds.get("strong", 0.02)), 1.0),
        (float(thresholds.get("good", 0.005)), 0.8),
        (float(thresholds.get("neutral", 0.0)), 0.5),
        (float(thresholds.get("weak", -0.02)), 0.2),
    )
    for threshold, exposure in levels:
        if score >= threshold:
            return exposure
    return 0.0


def _meta_probability_exposure(probability: float) -> float:
    probability = _finite_float(probability)
    if probability >= 0.80:
        return 0.0
    if probability >= 0.65:
        return 0.2
    if probability >= 0.50:
        return 0.5
    if probability >= 0.35:
        return 0.8
    return 1.0


def _forecast_values(row: dict[str, str], suffix: str) -> list[float]:
    meta_name = f"meta_{suffix}"
    if row.get(meta_name) not in (None, ""):
        return [_finite_float(row[meta_name])]
    values = []
    for name, raw_value in row.items():
        if name == meta_name:
            continue
        if name == suffix or name.endswith(f"_{suffix}"):
            if raw_value in (None, ""):
                continue
            values.append(_finite_float(raw_value))
    return values


def _has_forecast(row: dict[str, str], suffix: str) -> bool:
    try:
        return bool(_forecast_values(row, suffix))
    except (TypeError, ValueError):
        return False


def _forecast_source(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
) -> str:
    auxiliary_requirements = [
        requirement
        for requirement in definition.required_prediction_columns
        if requirement != "predicted_probability"
    ]
    if not auxiliary_requirements:
        return "probability_only"
    uses_meta_for_every_requirement = all(
        all(
            any(
                row.get(f"meta_{name}") not in (None, "")
                for name in requirement.split("|")
            )
            for row in rows
        )
        for requirement in auxiliary_requirements
    )
    return (
        "meta_auxiliary"
        if uses_meta_for_every_requirement
        else "source_model_auxiliary"
    )


def _clip_exposure(
    value: float,
    definition: AllocationPolicyDefinition,
) -> float:
    return min(
        definition.exposure_max,
        max(definition.exposure_min, _finite_float(value)),
    )


def _simulate_policy(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
    diagnostics: dict[str, float | None],
) -> AllocationPolicyResult:
    periods = _aggregate_periods(rows, exposures)
    equity = 1.0
    baseline_equity = 1.0
    curve = [equity]
    baseline_curve = [baseline_equity]
    net_returns = []
    turnover = 0.0
    estimated_costs = 0.0
    previous_exposure = 1.0
    records: list[dict[str, float | str]] = []
    for date, period_return, exposure in periods:
        change = abs(exposure - previous_exposure)
        cost = change * transaction_cost_bps / 10_000.0
        net_return = (period_return * exposure) - cost
        if net_return <= -1.0 or period_return <= -1.0:
            raise ValueError("Allocation return would zero or invert equity")
        equity *= 1.0 + net_return
        baseline_equity *= 1.0 + period_return
        curve.append(equity)
        baseline_curve.append(baseline_equity)
        net_returns.append(net_return)
        turnover += change
        estimated_costs += cost
        records.append({
            "date": date,
            "baseline_return": period_return,
            "allocated_return": net_return,
            "exposure": exposure,
            "baseline_drawdown": _current_drawdown(baseline_curve),
        })
        previous_exposure = exposure

    total_return = equity - 1.0
    annualized_return = _annualized_return(total_return, periods)
    drawdown = _max_drawdown(curve)
    baseline_drawdown = _max_drawdown(baseline_curve)
    exposure_values = [period[2] for period in periods]
    changes = [
        abs(current - previous)
        for previous, current in zip(exposure_values, exposure_values[1:])
        if not math.isclose(current, previous)
    ]
    all_changes = [
        abs(current - previous)
        for previous, current in zip(exposure_values, exposure_values[1:])
    ]
    worst_count = max(1, math.ceil(len(records) * 0.20)) if records else 0
    worst_records = sorted(
        records,
        key=lambda row: float(row["baseline_drawdown"]),
    )[:worst_count]
    drawdown_delta = baseline_drawdown - drawdown
    if drawdown_delta > 1e-12:
        drawdown_effect = "avoided"
    elif drawdown_delta < -1e-12:
        drawdown_effect = "worsened"
    else:
        drawdown_effect = "unchanged"

    return AllocationPolicyResult(
        policy_name=definition.policy_name,
        policy_version=definition.policy_version,
        policy_kind=definition.policy_kind,
        mapping_method=definition.mapping_method,
        threshold_fit_scope=definition.threshold_fit_scope,
        overfit_warning=definition.overfit_warning,
        transaction_cost_bps=transaction_cost_bps,
        required_prediction_columns=definition.required_prediction_columns,
        exposure_min=definition.exposure_min,
        exposure_max=definition.exposure_max,
        available=True,
        skip_reason=None,
        forecast_source=_forecast_source(definition, rows),
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=drawdown,
        sharpe=_sharpe_ratio(net_returns, periods),
        sortino=_sortino_ratio(net_returns, periods),
        calmar=calmar_ratio(
            annualized_return if annualized_return is not None else total_return,
            drawdown,
        ),
        turnover=turnover,
        estimated_transaction_costs=estimated_costs,
        return_per_unit_drawdown=(total_return / drawdown if drawdown else None),
        mean_exposure=mean(exposure_values) if exposure_values else 0.0,
        median_exposure=median(exposure_values) if exposure_values else 0.0,
        min_exposure=min(exposure_values, default=0.0),
        max_exposure=max(exposure_values, default=0.0),
        exposure_std=_population_std(exposure_values),
        days_at_0_exposure=sum(math.isclose(value, 0.0) for value in exposure_values),
        days_at_full_exposure=sum(math.isclose(value, 1.0) for value in exposure_values),
        number_of_exposure_changes=len(changes),
        average_exposure_change=mean(changes) if changes else 0.0,
        maximum_one_period_exposure_change=max(all_changes, default=0.0),
        pct_periods_at_0_exposure=_percentage_at_exposure(exposure_values, 0.0),
        pct_periods_at_20_exposure=_percentage_at_exposure(exposure_values, 0.2),
        pct_periods_at_50_exposure=_percentage_at_exposure(exposure_values, 0.5),
        pct_periods_at_80_exposure=_percentage_at_exposure(exposure_values, 0.8),
        pct_periods_at_100_exposure=_percentage_at_exposure(exposure_values, 1.0),
        evaluated_periods=len(periods),
        performance_when_exposure_reduced=_performance_summary(
            [row for row in records if float(row["exposure"]) < 0.5]
        ),
        performance_when_exposure_high=_performance_summary(
            [row for row in records if float(row["exposure"]) >= 0.8]
        ),
        performance_during_worst_drawdown_windows=_performance_summary(worst_records),
        drawdown_impact={
            "baseline_max_drawdown": baseline_drawdown,
            "policy_max_drawdown": drawdown,
            "drawdown_improvement": drawdown_delta,
            "effect": drawdown_effect,
        },
        prediction_to_exposure_diagnostics=(
            _prediction_to_exposure_diagnostics(rows, exposures)
        ),
        balanced_accuracy=diagnostics.get("balanced_accuracy"),
        brier_score=diagnostics.get("brier_score"),
        expected_calibration_error=diagnostics.get("expected_calibration_error"),
        **RESEARCH_METADATA,
    )


def _aggregate_periods(
    rows: list[dict[str, str]],
    exposures: list[float],
) -> list[tuple[str, float, float]]:
    if len(rows) != len(exposures):
        raise ValueError("Allocation rows and exposures must have equal length")
    by_date: dict[str, list[tuple[float, float]]] = {}
    for row, exposure in zip(rows, exposures):
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Allocation row is missing rebalance_date")
        period_return = _finite_float(
            row.get("champion_return_next_period", 0.0) or 0.0
        )
        by_date.setdefault(date, []).append((period_return, _finite_float(exposure)))
    return [
        (date, mean(value[0] for value in values), mean(value[1] for value in values))
        for date, values in sorted(by_date.items())
    ]


def _selected_optimizer_exposure_path(
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
    variant: AllocationVariant,
) -> list[dict[str, Any]]:
    if len(rows) != len(exposures):
        raise ValueError("Optimizer exposure path rows and exposures must align")
    scores = _variant_scores(rows, variant)
    grouped: dict[str, list[dict[str, float]]] = {}
    for row, exposure, score in zip(rows, exposures, scores):
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Optimizer exposure path row is missing rebalance_date")
        grouped.setdefault(date, []).append({
            "period_return": _finite_float(
                row.get("champion_return_next_period", 0.0) or 0.0
            ),
            "exposure": _finite_float(exposure),
            "score": _finite_float(score),
            "predicted_forward_return": _return_forecast(row),
            "predicted_future_drawdown": _mean_or_none(
                _forecast_values(row, "predicted_future_drawdown")
                or _forecast_values(row, "predicted_max_adverse_excursion")
            ),
            "predicted_future_volatility": _mean_or_none(
                _forecast_values(row, "predicted_future_volatility")
            ),
        })

    equity = 1.0
    peak = 1.0
    previous_exposure = 1.0
    path_rows = []
    for date, values in sorted(grouped.items()):
        period_return = mean(value["period_return"] for value in values)
        exposure = mean(value["exposure"] for value in values)
        turnover = abs(exposure - previous_exposure)
        cost = turnover * transaction_cost_bps / 10_000.0
        net_return = (period_return * exposure) - cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        path_rows.append({
            "rebalance_date": date,
            "source_row_count": len(values),
            "period_return": period_return,
            "exposure": exposure,
            "score": mean(value["score"] for value in values),
            "predicted_forward_return": mean(
                value["predicted_forward_return"] for value in values
            ),
            "predicted_future_drawdown": _mean_or_none([
                value["predicted_future_drawdown"]
                for value in values
                if value["predicted_future_drawdown"] is not None
            ]),
            "predicted_future_volatility": _mean_or_none([
                value["predicted_future_volatility"]
                for value in values
                if value["predicted_future_volatility"] is not None
            ]),
            "turnover": turnover,
            "transaction_cost_bps": transaction_cost_bps,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
            **RESEARCH_METADATA,
        })
        previous_exposure = exposure
    return path_rows


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None


def _performance_summary(
    records: list[dict[str, float | str]],
) -> dict[str, float | int]:
    baseline_returns = [float(row["baseline_return"]) for row in records]
    allocated_returns = [float(row["allocated_return"]) for row in records]
    return {
        "period_count": len(records),
        "baseline_total_return": _compound_returns(baseline_returns),
        "allocated_total_return": _compound_returns(allocated_returns),
        "return_difference": (
            _compound_returns(allocated_returns) - _compound_returns(baseline_returns)
        ),
        "mean_exposure": (
            mean(float(row["exposure"]) for row in records) if records else 0.0
        ),
    }


def _shadow_policy_payload(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
    exposures: list[float] | None,
    skip_reason: str | None,
) -> dict[str, Any]:
    payload = {
        **_policy_metadata(definition),
        "available": exposures is not None and skip_reason is None,
        "skip_reason": skip_reason,
        "forecast_source": (
            _forecast_source(definition, rows)
            if exposures is not None and skip_reason is None
            else "unavailable/skipped"
        ),
        **RESEARCH_METADATA,
    }
    payload["rows"] = (
        [
            {"date": date, "baseline_return": period_return, "exposure": exposure}
            for date, period_return, exposure in _aggregate_periods(rows, exposures)
        ]
        if exposures is not None and skip_reason is None
        else []
    )
    return payload


def _result_payload(result: AllocationPolicyResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["required_prediction_columns"] = list(
        result.required_prediction_columns
    )
    return payload


def _unavailable_policy_payload(
    definition: AllocationPolicyDefinition,
    reason: str,
) -> dict[str, Any]:
    return {
        "rank": None,
        **_policy_metadata(definition),
        "available": False,
        "skip_reason": reason,
        "forecast_source": "unavailable/skipped",
        **RESEARCH_METADATA,
    }


def _policy_metadata(definition: AllocationPolicyDefinition) -> dict[str, Any]:
    return {
        "policy_name": definition.policy_name,
        "policy_version": definition.policy_version,
        "policy_kind": definition.policy_kind,
        "mapping_method": definition.mapping_method,
        "threshold_fit_scope": definition.threshold_fit_scope,
        "overfit_warning": definition.overfit_warning,
        "transaction_cost_bps": definition.transaction_cost_bps,
        "required_prediction_columns": list(
            definition.required_prediction_columns
        ),
        "exposure_min": definition.exposure_min,
        "exposure_max": definition.exposure_max,
    }


def _write_comparison_csv(path: Path, policies: list[dict[str, Any]]) -> None:
    rows = [_csv_policy_row(policy) for policy in policies]
    fieldnames = list(rows[0]) if rows else list(_csv_policy_row({}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_policy_row(policy: dict[str, Any]) -> dict[str, Any]:
    scalar_columns = (
        "rank", "policy_name", "policy_version", "policy_kind", "mapping_method",
        "threshold_fit_scope", "overfit_warning", "transaction_cost_bps",
        "available", "skip_reason", "forecast_source", "exposure_min",
        "exposure_max", "total_return",
        "annualized_return", "max_drawdown", "sharpe", "sortino", "calmar",
        "return_per_unit_drawdown", "turnover", "estimated_transaction_costs",
        "mean_exposure", "median_exposure", "min_exposure", "max_exposure",
        "exposure_std",
        "days_at_0_exposure", "days_at_full_exposure",
        "number_of_exposure_changes", "average_exposure_change",
        "maximum_one_period_exposure_change", "pct_periods_at_0_exposure",
        "pct_periods_at_20_exposure", "pct_periods_at_50_exposure",
        "pct_periods_at_80_exposure", "pct_periods_at_100_exposure",
        "evaluated_periods", "balanced_accuracy", "brier_score",
        "expected_calibration_error", "trading_impact", "research_only",
        "production_validated",
    )
    row = {name: policy.get(name) for name in scalar_columns}
    for name in (
        "required_prediction_columns",
        "performance_when_exposure_reduced",
        "performance_when_exposure_high",
        "performance_during_worst_drawdown_windows",
        "drawdown_impact",
        "prediction_to_exposure_diagnostics",
        "robustness_flags",
        "dominated_by",
    ):
        row[name] = json.dumps(policy.get(name))
    return row


def _write_leaderboard(
    path: Path,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    columns = (
        "policy_name", "policy_kind", "total_return", "max_drawdown", "sharpe",
        "sortino", "calmar", "return_per_unit_drawdown", "turnover",
        "estimated_transaction_costs", "mean_exposure",
    )
    lines = [
        "# Allocation Policy Leaderboard v2",
        "",
        "|rank|" + "|".join(columns) + "|",
        "|---|" + "|".join("---" for _ in columns) + "|",
    ]
    for result in results:
        rank = result["rank"]
        values = [
            str(result[name])
            if name in {"policy_name", "policy_kind"}
            else _format_optional_float(result.get(name))
            for name in columns
        ]
        lines.append(f"|{rank}|" + "|".join(values) + "|")
    winners = _comparison_winners(results)
    if winners:
        lines.extend([
            "",
            "## Outcome Winners",
            "",
            f"- Total return: {winners['best_total_return']}",
            f"- Max drawdown: {winners['best_max_drawdown']}",
            f"- Sharpe: {winners['best_sharpe']}",
            f"- Sortino: {winners['best_sortino']}",
            f"- Calmar: {winners['best_calmar']}",
            "- Dominated: " + _format_name_list(winners["dominated_policies"]),
            "- Too defensive: " + _format_name_list(
                winners["too_defensive_policies"]
            ),
            "- Too choppy: " + _format_name_list(winners["too_choppy_policies"]),
        ])
    if skipped:
        lines.extend(["", "## Skipped Policies", ""])
        lines.extend(
            f"- {row['policy_name']}: {row['skip_reason']}"
            for row in skipped
        )
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostics_markdown(
    path: Path,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    lines = [
        "# Allocation Policy Diagnostics v2",
        "",
        "|policy|mean|median|min|max|std|changes|max_change|constant|dominated|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        flags = row.get("robustness_flags", {})
        lines.append(
            "|{policy}|{mean:.6f}|{median:.6f}|{minimum:.6f}|{maximum:.6f}|"
            "{std:.6f}|{changes}|{max_change:.6f}|{constant}|{dominated}|".format(
                policy=row["policy_name"],
                mean=float(row["mean_exposure"]),
                median=float(row["median_exposure"]),
                minimum=float(row["min_exposure"]),
                maximum=float(row["max_exposure"]),
                std=float(row["exposure_std"]),
                changes=row["number_of_exposure_changes"],
                max_change=float(row["maximum_one_period_exposure_change"]),
                constant=str(flags.get("exposure_is_constant", False)).lower(),
                dominated=str(flags.get("dominated_by_simpler_baseline", False)).lower(),
            )
        )
    lines.extend(["", "## Prediction To Exposure", ""])
    for row in results:
        diagnostics = row.get("prediction_to_exposure_diagnostics", {})
        lines.append(
            "- {name}: return_corr={return_corr}, volatility_corr={vol_corr}, "
            "drawdown_corr={drawdown_corr}".format(
                name=row["policy_name"],
                return_corr=_format_optional_float(
                    diagnostics.get(
                        "correlation_predicted_forward_return_10d_to_exposure"
                    )
                ),
                vol_corr=_format_optional_float(
                    diagnostics.get(
                        "correlation_predicted_future_volatility_to_exposure"
                    )
                ),
                drawdown_corr=_format_optional_float(
                    diagnostics.get(
                        "correlation_predicted_future_drawdown_to_exposure"
                    )
                ),
            )
        )
    if skipped:
        lines.extend(["", "## Skipped Policies", ""])
        lines.extend(
            f"- {row['policy_name']}: {row['skip_reason']}"
            for row in skipped
        )
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grid_search_reports(
    paths: AllocationV2Paths,
    grid_search: dict[str, Any],
    config: dict[str, Any],
) -> None:
    candidates = list(grid_search.get("candidates", []))
    selected = next(
        (row for row in candidates if row.get("objective_rank") == 1),
        None,
    )
    selected_evaluation = grid_search.get("selected")
    selected_payload = None
    if selected and selected_evaluation:
        selected_payload = {
            **selected,
            "selection_metrics": _result_payload(
                selected_evaluation["selection_result"]
            ),
            "holdout_metrics": _result_payload(selected_evaluation["result"]),
            "selection_protocol": grid_search["selection_protocol"],
            "selection_notice": grid_search["selection_notice"],
            "overfit_warning": grid_search["selection_notice"],
        }
    outcome_ranked = sorted(
        candidates,
        key=lambda row: int(row.get("outcome_rank", 10**9)),
    )
    objective_ranked = sorted(
        candidates,
        key=lambda row: int(row.get("objective_rank", 10**9)),
    )
    payload = {
        "mode": "allocation_policy_grid_search_v2_research_only",
        "grid_size": len(candidates),
        "selection_objective": config.get("allocation_grid_objective", {}),
        "selected_diagnostic_policy": selected_payload,
        "candidates": outcome_ranked,
        "outcome_ranking": [row["candidate_id"] for row in outcome_ranked],
        "objective_ranking": [row["candidate_id"] for row in objective_ranked],
        "skip_reason": grid_search.get("skip_reason"),
        "selection_protocol": grid_search.get("selection_protocol"),
        "selection_notice": grid_search.get("selection_notice"),
        "overfit_warning": grid_search.get("selection_notice"),
        "automatic_promotion": False,
        **RESEARCH_METADATA,
    }
    paths.grid_search_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    csv_rows = [_json_safe_csv_row(row) for row in payload["candidates"]]
    fieldnames = list(csv_rows[0]) if csv_rows else [
        "candidate_id",
        "objective_rank",
        "outcome_rank",
        "overfit_warning",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with paths.grid_search_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# Allocation Policy Grid Search v2",
        "",
        str(grid_search.get("selection_notice")),
        "",
        "|objective_rank|outcome_rank|candidate|objective|total_return|max_drawdown|sharpe|turnover|cost|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "|{objective_rank}|{outcome_rank}|{candidate_id}|{objective:.6f}|"
            "{total_return:.6f}|{max_drawdown:.6f}|{sharpe:.6f}|"
            "{turnover:.6f}|{cost:.6f}|".format(
                objective_rank=row["objective_rank"],
                outcome_rank=row["outcome_rank"],
                candidate_id=row["candidate_id"],
                objective=float(row["objective"]),
                total_return=float(row["total_return"]),
                max_drawdown=float(row["max_drawdown"]),
                sharpe=float(row["sharpe"]),
                turnover=float(row["turnover"]),
                cost=float(row["estimated_transaction_costs"]),
            )
        )
    if grid_search.get("skip_reason"):
        lines.extend(["", f"Skipped: {grid_search['skip_reason']}"])
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    paths.grid_search_markdown.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _json_safe_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, value in row.items():
        output[name] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    return output


def _format_name_list(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _comparison_winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    available = [row for row in rows if row.get("available")]
    return {
        "best_total_return": max(available, key=lambda row: float(row["total_return"]))[
            "policy_name"
        ],
        "best_max_drawdown": min(available, key=lambda row: float(row["max_drawdown"]))[
            "policy_name"
        ],
        "best_sharpe": max(available, key=lambda row: float(row["sharpe"]))[
            "policy_name"
        ],
        "best_sortino": max(available, key=lambda row: float(row["sortino"]))[
            "policy_name"
        ],
        "best_calmar": max(available, key=lambda row: float(row["calmar"]))[
            "policy_name"
        ],
        "dominated_policies": [
            row["policy_name"]
            for row in available
            if row.get("robustness_flags", {}).get("dominated_by_simpler_baseline")
        ],
        "too_defensive_policies": [
            row["policy_name"]
            for row in available
            if row.get("robustness_flags", {}).get("too_defensive")
        ],
        "too_choppy_policies": [
            row["policy_name"]
            for row in available
            if row.get("robustness_flags", {}).get("too_choppy")
        ],
    }


def _add_robustness_flags(
    payloads: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    champion = next(
        row for row in payloads if row["policy_name"] == "champion_baseline"
    )
    simple_baselines = [
        row for row in payloads if row["policy_kind"] == "diagnostic_baseline"
    ]
    change_rate_limit = float(
        config.get("allocation_exposure_change_rate_warning", 0.80)
    )
    extreme_limit = float(
        config.get("allocation_mostly_extreme_percentage", 80.0)
    )
    destruction_floor = float(
        config.get("allocation_return_destruction_minimum", 0.02)
    )
    defensive_mean_limit = float(
        config.get("allocation_too_defensive_mean_exposure", 0.25)
    )
    turnover_per_period_limit = float(
        config.get("allocation_turnover_per_period_warning", 0.50)
    )
    for row in payloads:
        possible_changes = max(int(row["evaluated_periods"]) - 1, 1)
        change_rate = float(row["number_of_exposure_changes"]) / possible_changes
        return_damage = float(champion["total_return"]) - float(row["total_return"])
        material_return_damage = max(
            destruction_floor,
            abs(float(champion["total_return"])) * 0.50,
        )
        dominated_by = [
            baseline["policy_name"]
            for baseline in simple_baselines
            if baseline["policy_name"] != row["policy_name"]
            and _dominates(baseline, row)
        ]
        row["dominated_by"] = dominated_by
        row["robustness_flags"] = {
            "exposure_is_constant": math.isclose(
                float(row["exposure_std"]),
                0.0,
                abs_tol=1e-12,
            ),
            "exposure_changes_too_often": change_rate > change_rate_limit,
            "exposure_is_mostly_zero": (
                float(row["pct_periods_at_0_exposure"]) >= extreme_limit
            ),
            "exposure_is_mostly_full": (
                float(row["pct_periods_at_100_exposure"]) >= extreme_limit
            ),
            "improves_return_but_worsens_drawdown": (
                float(row["total_return"]) > float(champion["total_return"])
                and float(row["max_drawdown"]) > float(champion["max_drawdown"])
            ),
            "improves_drawdown_but_destroys_return": (
                float(row["max_drawdown"]) < float(champion["max_drawdown"])
                and return_damage > material_return_damage
            ),
            "dominated_by_simpler_baseline": bool(dominated_by),
            "too_defensive": (
                float(row["mean_exposure"]) < defensive_mean_limit
            ),
            "too_choppy": (
                change_rate > change_rate_limit
                or float(row["turnover"]) / max(int(row["evaluated_periods"]), 1)
                > turnover_per_period_limit
            ),
        }


def _dominates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    no_worse = (
        float(baseline["total_return"]) >= float(candidate["total_return"])
        and float(baseline["max_drawdown"]) <= float(candidate["max_drawdown"])
        and float(baseline["turnover"]) <= float(candidate["turnover"])
        and float(baseline["estimated_transaction_costs"])
        <= float(candidate["estimated_transaction_costs"])
    )
    strictly_better = (
        float(baseline["total_return"]) > float(candidate["total_return"])
        or float(baseline["max_drawdown"]) < float(candidate["max_drawdown"])
        or float(baseline["turnover"]) < float(candidate["turnover"])
    )
    return no_worse and strictly_better


def _validate_output_consistency(paths: AllocationV2Paths) -> None:
    comparison = json.loads(paths.comparison_json.read_text(encoding="utf-8"))
    shadow = json.loads(paths.shadow_overlay_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    grid_search = json.loads(paths.grid_search_json.read_text(encoding="utf-8"))
    optimizer = json.loads(paths.optimizer_results_json.read_text(encoding="utf-8"))
    selected_optimizer_path = json.loads(
        paths.selected_optimizer_exposure_path_json.read_text(encoding="utf-8")
    )
    with paths.comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    with paths.selected_optimizer_exposure_path_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        selected_path_csv_rows = list(csv.DictReader(handle))
    markdown = paths.leaderboard_markdown.read_text(encoding="utf-8")
    diagnostics_markdown = paths.diagnostics_markdown.read_text(encoding="utf-8")
    grid_markdown = paths.grid_search_markdown.read_text(encoding="utf-8")
    optimizer_markdown = paths.optimizer_report_markdown.read_text(encoding="utf-8")
    expected_policy_names = {
        definition.policy_name for definition in _policy_definitions()
    }
    if comparison.get("grid_search_diagnostic"):
        expected_policy_names.add("best_grid_search_diagnostic_policy")
    expected_baseline_names = {
        definition.policy_name for definition in _baseline_definitions()
    }
    comparison_policy_names = {
        row.get("policy_name") for row in comparison.get("policies", [])
    }
    comparison_baseline_names = {
        row.get("policy_name") for row in comparison.get("baselines", [])
    }
    csv_names = {row.get("policy_name") for row in csv_rows}
    shadow_policy_names = set(shadow.get("policies", {}))
    shadow_baseline_names = set(shadow.get("baselines", {}))
    if not (
        expected_policy_names
        == comparison_policy_names
        == shadow_policy_names
        and expected_baseline_names
        == comparison_baseline_names
        == shadow_baseline_names
        and csv_names == expected_policy_names | expected_baseline_names
    ):
        raise RuntimeError("Allocation v2 outputs contain inconsistent policy sets")
    for payload in (
        comparison,
        shadow,
        diagnostics,
        grid_search,
        optimizer,
        selected_optimizer_path,
    ):
        if any(payload.get(name) != value for name, value in RESEARCH_METADATA.items()):
            raise RuntimeError("Allocation v2 JSON output has invalid research metadata")
    required_notice = (
        "Research only. Trading impact: none. Production validated: false."
    )
    if any(
        required_notice not in content
        for content in (
            markdown,
            diagnostics_markdown,
            grid_markdown,
            optimizer_markdown,
        )
    ):
        raise RuntimeError("Allocation v2 Markdown output is missing research metadata")
    if any(
        row.get("research_only") != "True"
        or row.get("trading_impact") != "none"
        or row.get("production_validated") != "False"
        for row in csv_rows
    ):
        raise RuntimeError("Allocation v2 CSV output has invalid research metadata")
    if any(
        row.get("research_only") != "True"
        or row.get("trading_impact") != "none"
        or row.get("production_validated") != "False"
        for row in selected_path_csv_rows
    ):
        raise RuntimeError(
            "Selected optimizer exposure CSV output has invalid research metadata"
        )


def _trading_rank_key(result: AllocationPolicyResult) -> tuple[float, ...]:
    return (
        -result.total_return,
        result.max_drawdown,
        -result.sharpe,
        -result.sortino,
        -result.calmar,
        result.turnover,
        result.estimated_transaction_costs,
    )


def _prediction_to_exposure_diagnostics(
    rows: list[dict[str, str]],
    exposures: list[float],
) -> dict[str, float | None]:
    return_values = [
        _mean_forecast_or_none(row, "predicted_forward_return_10d")
        for row in rows
    ]
    volatility_values = [
        _mean_forecast_or_none(row, "predicted_future_volatility")
        for row in rows
    ]
    drawdown_values = [
        _mean_forecast_or_none(row, "predicted_future_drawdown")
        for row in rows
    ]
    drawdown_risk_values = [
        abs(value) if value is not None else None
        for value in drawdown_values
    ]
    return {
        "correlation_predicted_forward_return_10d_to_exposure": (
            _paired_correlation(return_values, exposures)
        ),
        "correlation_predicted_future_volatility_to_exposure": (
            _paired_correlation(volatility_values, exposures)
        ),
        "correlation_predicted_future_drawdown_to_exposure": (
            _paired_correlation(drawdown_values, exposures)
        ),
        "average_exposure_predicted_return_top_quartile": _quartile_exposure(
            return_values,
            exposures,
            top=True,
        ),
        "average_exposure_predicted_return_bottom_quartile": _quartile_exposure(
            return_values,
            exposures,
            top=False,
        ),
        "average_exposure_predicted_drawdown_risk_top_quartile": (
            _quartile_exposure(drawdown_risk_values, exposures, top=True)
        ),
        "average_exposure_predicted_volatility_top_quartile": (
            _quartile_exposure(volatility_values, exposures, top=True)
        ),
    }


def _mean_forecast_or_none(
    row: dict[str, str],
    suffix: str,
) -> float | None:
    try:
        values = _forecast_values(row, suffix)
    except (TypeError, ValueError):
        return None
    return mean(values) if values else None


def _paired_correlation(
    values: list[float | None],
    exposures: list[float],
) -> float | None:
    pairs = [
        (float(value), float(exposure))
        for value, exposure in zip(values, exposures)
        if value is not None
    ]
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = mean(left)
    right_mean = mean(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0.0:
        return None
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    ) / denominator


def _quartile_exposure(
    values: list[float | None],
    exposures: list[float],
    *,
    top: bool,
) -> float | None:
    pairs = sorted(
        (
            (float(value), float(exposure))
            for value, exposure in zip(values, exposures)
            if value is not None
        ),
        key=lambda pair: pair[0],
        reverse=top,
    )
    if not pairs:
        return None
    count = max(1, math.ceil(len(pairs) * 0.25))
    return mean(exposure for _, exposure in pairs[:count])


def _annualized_return(
    total_return: float,
    periods: list[tuple[str, float, float]],
) -> float | None:
    if len(periods) < 2 or total_return <= -1.0:
        return None
    try:
        start = datetime.fromisoformat(periods[0][0][:10])
        end = datetime.fromisoformat(periods[-1][0][:10])
    except ValueError:
        return None
    elapsed_days = (end - start).days + _estimated_terminal_period_days(periods)
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _percentage_at_exposure(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(
        math.isclose(value, target, abs_tol=1e-12) for value in values
    ) / len(values)


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _estimated_terminal_period_days(
    periods: list[tuple[str, float, float]],
) -> int:
    try:
        dates = [datetime.fromisoformat(row[0][:10]) for row in periods]
    except ValueError:
        return 0
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    return max(1, round(median(gaps))) if gaps else 0


def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _current_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = max(values)
    return (values[-1] / peak) - 1.0 if peak else 0.0


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(mean((value - average) ** 2 for value in values))


def _sharpe_ratio(
    returns: list[float],
    periods: list[tuple[str, float, float]],
) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    standard_deviation = _population_std(returns)
    if standard_deviation == 0.0:
        return 0.0
    return (
        average
        / standard_deviation
        * math.sqrt(_observed_periods_per_year(periods))
    )


def _sortino_ratio(
    returns: list[float],
    periods: list[tuple[str, float, float]],
) -> float:
    if not returns:
        return 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return (
        mean(returns)
        / downside_deviation
        * math.sqrt(_observed_periods_per_year(periods))
    )


def _observed_periods_per_year(
    periods: list[tuple[str, float, float]],
) -> float:
    if len(periods) < 2:
        return 1.0
    try:
        start = datetime.fromisoformat(periods[0][0][:10])
        end = datetime.fromisoformat(periods[-1][0][:10])
    except ValueError:
        return 1.0
    elapsed_days = (end - start).days + _estimated_terminal_period_days(periods)
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(periods) * 365.25 / elapsed_days)


def _is_probability(value: Any) -> bool:
    try:
        result = _finite_float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= result <= 1.0


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Allocation inputs must be finite")
    return result
