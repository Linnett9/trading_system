from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.audits.return_mechanics_checks import (
    _candidate_metrics_equal,
    _champion_baseline_audit,
    _data_sanity_checks,
    _global_red_flags,
    _leakage_check,
    _mechanics_summary,
    _return_unit,
    _same_range_as_policies,
    _scenario_matrix,
    _symbol_concentration,
)
from core.research.ml.audits.return_mechanics_candidate import (
    _aggregate_return_rows,
    _candidate_audit,
    _capped_return_sensitivity,
    _cost_sensitivity,
    _equity_records,
    _exposure_sanity,
    _missing_candidate,
    _records_summary,
    _requirement_columns,
    _return_concentration,
    _top_records,
)
from core.research.ml.audits.return_mechanics_loading import (
    _artifact_paths,
    _read_csv,
    _read_json,
)
from core.research.ml.audits.return_mechanics_math import _number
from core.research.ml.audits.return_mechanics_optimizer import (
    _load_selected_optimizer_series,
    _reconstruct_optimizer_series,
)
from core.research.ml.audits.return_mechanics_reporting import (
    _fmt,
    _markdown,
    _write_csv,
)
from core.research.ml.audits.return_mechanics_types import (
    AUDITED_CANDIDATES,
    ReturnMechanicsAuditPaths,
)


def write_return_mechanics_audit(config: dict[str, Any]) -> ReturnMechanicsAuditPaths:
    ml_config = config.get("ml", {})
    output_dir = Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    default_cost_bps = float(ml_config.get("allocation_transaction_cost_bps", 5.0))

    paths = _artifact_paths(config, output_dir)
    shadow = _read_json(paths["shadow_overlay"])
    comparison = _read_json(paths["allocation_comparison"])
    optimizer = _read_json(paths["optimizer_results"])
    grid_search = _read_json(paths["grid_search"])
    expanded_audit = _read_json(paths["expanded_audit"])
    meta_audit = _read_json(paths["meta_audit"])
    meta_rows = _read_csv(paths["meta_dataset"])
    expanded_rows = _read_csv(paths["expanded_dataset"])
    auxiliary_rows = _read_csv(paths["meta_auxiliary_predictions"])
    selected_optimizer_path = _read_json(paths["selected_optimizer_exposure_path_json"])

    reported_metrics = _reported_metrics_by_candidate(comparison, optimizer)
    series_by_candidate = _load_shadow_series(
        shadow,
        comparison,
        default_cost_bps=default_cost_bps,
    )
    optimizer_series = _load_selected_optimizer_series(
        selected_optimizer_path,
        optimizer,
    ) or _reconstruct_optimizer_series(
        config=config,
        optimizer=optimizer,
        meta_rows=meta_rows,
        auxiliary_rows=auxiliary_rows,
    )
    if optimizer_series is not None:
        series_by_candidate[optimizer_series["candidate_name"]] = optimizer_series

    candidate_audits = []
    for name in AUDITED_CANDIDATES:
        series = series_by_candidate.get(name)
        if series is None:
            candidate_audits.append(_missing_candidate(name))
            continue
        candidate_audits.append(
            _candidate_audit(
                name,
                series,
                reported_metrics.get(name, {}),
                default_cost_bps=default_cost_bps,
            )
        )

    mechanics = _mechanics_summary(
        shadow,
        meta_rows,
        candidate_audits,
        default_cost_bps=default_cost_bps,
    )
    champion_audit = _champion_baseline_audit(
        config,
        candidate_audits,
        comparison,
    )
    data_sanity = _data_sanity_checks(
        config,
        expanded_audit,
        meta_audit,
        expanded_rows,
        candidate_audits,
    )
    leakage_check = _leakage_check(comparison, optimizer)
    payload = {
        "mode": "return_mechanics_audit_research_only",
        "audited_candidates": list(AUDITED_CANDIDATES),
        "source_artifacts": {key: str(path) for key, path in paths.items()},
        "mechanics": mechanics,
        "data_sanity": data_sanity,
        "leakage_check": leakage_check,
        "champion_baseline_audit": champion_audit,
        "candidates": candidate_audits,
        "capped_return_sensitivity": _scenario_matrix(
            candidate_audits,
            "capped_return_sensitivity",
        ),
        "cost_sensitivity": _scenario_matrix(candidate_audits, "cost_sensitivity"),
        "red_flags": _global_red_flags(
            candidate_audits,
            mechanics,
            champion_audit,
            leakage_check,
        ),
        **RESEARCH_METADATA,
    }

    audit_paths = ReturnMechanicsAuditPaths(
        csv_path=output_dir / "benchmark_return_audit.csv",
        json_path=output_dir / "benchmark_return_audit.json",
        markdown_path=output_dir / "benchmark_return_audit.md",
    )
    _write_csv(audit_paths.csv_path, candidate_audits)
    audit_paths.json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    audit_paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return audit_paths


def _load_shadow_series(
    shadow: dict[str, Any],
    comparison: dict[str, Any],
    *,
    default_cost_bps: float,
) -> dict[str, dict[str, Any]]:
    output = {}
    comparison_metrics = _reported_metrics_by_candidate(comparison, {})
    for collection in ("policies", "baselines"):
        payloads = shadow.get(collection, {})
        if not isinstance(payloads, dict):
            continue
        for candidate_name, payload in payloads.items():
            if not isinstance(payload, dict) or not payload.get("available", True):
                continue
            rows = payload.get("rows", [])
            if not isinstance(rows, list):
                continue
            cost_bps = _number(payload.get("transaction_cost_bps"))
            if cost_bps is None:
                cost_bps = _number(comparison.get("transaction_cost_bps"))
            if cost_bps is None:
                cost_bps = default_cost_bps
            output[str(candidate_name)] = {
                "candidate_name": str(candidate_name),
                "policy_kind": payload.get("policy_kind"),
                "forecast_source": payload.get("forecast_source"),
                "required_prediction_columns": payload.get(
                    "required_prediction_columns",
                    [],
                ),
                "transaction_cost_bps": float(cost_bps),
                "period_source": "allocation_shadow_overlay_exact",
                "exact_period_path": True,
                "rows": rows,
                "reported_metrics": comparison_metrics.get(str(candidate_name), {}),
            }
    return output


def _reported_metrics_by_candidate(
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output = {}
    for collection in ("policies", "baselines"):
        for row in comparison.get(collection, []):
            if isinstance(row, dict) and row.get("policy_name"):
                output[str(row["policy_name"])] = row
    selected = optimizer.get("selected_policy")
    if isinstance(selected, dict):
        metrics = selected.get("frozen_holdout_metrics") or selected.get(
            "holdout_metrics"
        )
        if isinstance(metrics, dict) and metrics.get("policy_name"):
            output[str(metrics["policy_name"])] = metrics
    return output

