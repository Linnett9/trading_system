from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.allocation.exposures import (
    _baseline_definitions,
    _evaluate_policy_exposures,
    _policy_definitions,
    _policy_exposures,
    _forecast_values,
)
from core.research.ml.allocation.search import (
    _evaluate_allocation_optimizer,
    _evaluate_policy_grid_search,
)
from core.research.ml.allocation.reporting import (
    _shadow_policy_payload,
    _unavailable_policy_payload,
    _validate_output_consistency,
    _write_comparison_csv,
    _write_diagnostics_markdown,
    _write_grid_search_reports,
    _write_leaderboard,
)
from core.research.ml.allocation.simulation import (
    _add_robustness_flags,
    _comparison_winners,
    _result_payload,
    _simulate_policy,
    _trading_rank_key,
)
from core.research.ml.allocation.allocation_optimizer import write_optimizer_reports
from core.research.ml.allocation.types import (
    AllocationPolicyDefinition,
    AllocationPolicyResult,
    AllocationV2Paths,
    POLICY_VERSION,
    RESEARCH_METADATA,
)


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
    _write_grid_search_reports(
        paths,
        grid_search,
        config,
        result_payload=_result_payload,
    )
    write_optimizer_reports(output_dir, optimizer_report)
    _validate_output_consistency(paths)
    return paths
