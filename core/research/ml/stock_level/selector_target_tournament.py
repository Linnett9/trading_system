from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from core.research.framework.ranking import CrossSectionalRankingEvaluator, finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import file_digest
from core.research.ml.stock_level.selector_portfolio_promotion import (
    DEFAULT_PROMOTION,
    build_selector_portfolio_promotion,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_artifact_io import read_stock_level_artifact
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    write_stock_level_artifact,
)
from core.research.ml.stock_level_benchmark_data import (
    _available_feature_columns,
    _build_oos_prediction_rows,
    _number,
    _prepare_rows,
)
from core.research.ml.stock_level_benchmark_models import _model_factories
from core.research.ml.stock_level_benchmark_types import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    PREDICTION_PREFIX,
    TARGET_COLUMN,
    TARGET_OUTPUT_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
    TABULAR_MODEL_NAMES,
)
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    build_stock_level_model_ranking_benchmark,
)


TARGET_CONTRACT_VERSION = "selector_target_contract_v1"
TOURNAMENT_SCHEMA_VERSION = "selector_target_tournament_v1"

DEFAULT_TARGETS = {
    "raw_return_10d": {
        "target_column": "actual_forward_return_10d",
        "task_type": "cross_sectional_regression",
        "prediction_semantics": "cross_sectional_score",
        "mathematical_definition": "10-session simple close-to-close forward return.",
        "normalisation_scope": "none",
    },
    "market_residual_return_10d": {
        "target_column": "actual_market_residual_return_10d",
        "task_type": "cross_sectional_regression",
        "prediction_semantics": "cross_sectional_score",
        "mathematical_definition": "10-session stock forward return minus matching benchmark 10-session forward return.",
        "normalisation_scope": "benchmark_residual",
    },
    "rank_normalized_return_10d": {
        "target_column": "actual_rank_normalized_forward_return_10d",
        "task_type": "ranking",
        "prediction_semantics": "cross_sectional_score",
        "mathematical_definition": "Per-decision-date cross-sectional rank of raw 10-session return, scaled to [0, 1].",
        "normalisation_scope": "decision_date_cross_section",
    },
    "volatility_adjusted_return_10d": {
        "target_column": "actual_vol_adjusted_forward_return_10d",
        "task_type": "cross_sectional_regression",
        "prediction_semantics": "cross_sectional_score",
        "mathematical_definition": "Raw 10-session forward return divided by trailing 20-session pre-decision volatility.",
        "normalisation_scope": "symbol_trailing_volatility",
    },
    "drawdown_adjusted_return_10d": {
        "target_column": "actual_drawdown_adjusted_forward_return_10d",
        "task_type": "cross_sectional_regression",
        "prediction_semantics": "cross_sectional_score",
        "mathematical_definition": "Raw 10-session forward return penalized by absolute adverse forward drawdown.",
        "normalisation_scope": "forward_adverse_drawdown",
    },
    "top_decile_label_10d": {
        "target_column": "actual_top_decile_label_10d",
        "task_type": "binary_classification",
        "prediction_semantics": "classification_label",
        "mathematical_definition": "Indicator for membership in the top raw-return decile on a decision date.",
        "normalisation_scope": "decision_date_cross_section",
    },
}

ECONOMIC_RETURN_COLUMN = "actual_forward_return_10d"
BENCHMARK_RETURN_COLUMN = "actual_benchmark_return_10d"
OUTCOME_DENYLIST = {
    ECONOMIC_RETURN_COLUMN,
    "actual_forward_return_5d",
    "actual_future_volatility",
    "actual_future_drawdown",
    "actual_max_adverse_excursion",
    BENCHMARK_RETURN_COLUMN,
    *TARGET_OUTPUT_COLUMNS,
    *TARGET_PROVENANCE_COLUMNS,
    "target_status",
}


@dataclass(frozen=True)
class SelectorTargetTournamentPaths:
    output_dir: Path
    real_artifact_audit_json_path: Path
    real_artifact_audit_markdown_path: Path
    contracts_path: Path
    plan_path: Path
    predictions_path: Path | None
    predictions_csv_path: Path | None
    forecast_metrics_path: Path
    model_comparisons_path: Path
    target_summary_path: Path
    report_json_path: Path
    report_markdown_path: Path
    promotion_json_path: Path | None


def write_selector_target_tournament(config: Mapping[str, Any]) -> SelectorTargetTournamentPaths:
    ml = dict(config.get("ml", {}) or {})
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.selector_target_tournament.enabled is false")
    source_path = Path(settings["source_dataset_path"])
    if not source_path.exists():
        raise FileNotFoundError(
            "Selector target tournament source dataset does not exist; "
            f"configured_path={source_path}. No legacy fallback is permitted."
        )
    rows = read_stock_level_artifact(
        source_path,
        required_columns={"rebalance_date", "symbol"},
        allow_csv_fallback=bool(settings["allow_csv_fallback"]),
    )
    if not rows:
        raise ValueError(f"Selector target tournament source dataset has no rows: {source_path}")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_selector_target_tournament(
        rows,
        config=config,
        source_path=source_path,
        settings=settings,
    )
    paths = SelectorTargetTournamentPaths(
        output_dir=output_dir,
        real_artifact_audit_json_path=output_dir / "selector_target_real_artifact_audit.json",
        real_artifact_audit_markdown_path=output_dir / "selector_target_real_artifact_audit.md",
        contracts_path=output_dir / "selector_target_contracts.json",
        plan_path=output_dir / "selector_target_tournament_plan.json",
        predictions_path=None if settings["plan_only"] else output_dir / "selector_target_oos_predictions.parquet",
        predictions_csv_path=(
            output_dir / "selector_target_oos_predictions.csv"
            if (not settings["plan_only"] and settings["write_debug_csv"])
            else None
        ),
        forecast_metrics_path=output_dir / "selector_target_forecast_metrics.csv",
        model_comparisons_path=output_dir / "selector_target_model_comparisons.csv",
        target_summary_path=output_dir / "selector_target_summary.csv",
        report_json_path=output_dir / "selector_target_tournament_report.json",
        report_markdown_path=output_dir / "selector_target_tournament_report.md",
        promotion_json_path=None if settings["plan_only"] else output_dir / "selector_promotion_report.json",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.real_artifact_audit_json_path, payload["real_artifact_audit"])
    writer.write_markdown(paths.real_artifact_audit_markdown_path, _artifact_audit_markdown(payload["real_artifact_audit"]))
    writer.write_json(paths.contracts_path, payload["target_contracts"])
    writer.write_json(paths.plan_path, payload["plan"])
    writer.write_csv(paths.forecast_metrics_path, payload["forecast_metrics"], fieldnames=_fields(payload["forecast_metrics"], ["candidate_id"]))
    writer.write_csv(paths.model_comparisons_path, payload["reference_target_deltas"], fieldnames=_fields(payload["reference_target_deltas"], ["target_id", "model_id"]))
    writer.write_csv(paths.target_summary_path, payload["target_summary"], fieldnames=_fields(payload["target_summary"], ["target_id"]))
    if paths.predictions_path is not None:
        identity = write_stock_level_artifact(
            paths.predictions_path,
            payload["oos_predictions"],
            fieldnames=_prediction_fields(payload["oos_predictions"]),
            config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
        )
        payload["prediction_artifact_identity"] = identity
    if paths.predictions_csv_path is not None:
        writer.write_csv(paths.predictions_csv_path, payload["oos_predictions"], fieldnames=_prediction_fields(payload["oos_predictions"]))
    if paths.promotion_json_path is not None:
        writer.write_json(paths.promotion_json_path, payload["promotion_results"])
    payload.setdefault("prediction_artifact_identity", None)
    writer.write_json(paths.report_json_path, payload)
    writer.write_markdown(paths.report_markdown_path, _markdown(payload))
    return paths


def build_selector_target_tournament(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    source_path: Path | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_duplicate_rows(rows)
    contracts = discover_target_contracts(rows, settings)
    selected_contracts = [row for row in contracts if row["target_id"] in settings["target_ids"]]
    blockers = _contract_blockers(selected_contracts)
    real_artifact_audit = _real_artifact_audit(rows, source_path, settings, contracts)
    blockers.extend(real_artifact_audit["blockers"])
    feature_columns = _feature_columns(rows, config, settings)
    leakage_audit = _leakage_audit(feature_columns, selected_contracts)
    if leakage_audit["blocked_feature_columns"]:
        blockers.append("outcome_columns_in_feature_set")
    eligible = _target_eligible_rows(rows, selected_contracts, settings)
    fold_plan = _shared_fold_plan(
        eligible["matched_rows"],
        feature_columns=feature_columns,
        min_train_dates=int(settings["min_train_dates"]),
        test_window_dates=int(settings["test_window_dates"]),
        embargo_dates=int(settings["embargo_dates"]),
        maximum_folds=settings.get("maximum_folds"),
    )
    if fold_plan.get("status") != "ready":
        blockers.append(str(fold_plan.get("blocked_reason", "shared_fold_plan_not_ready")))
    fit_count = _fit_count(
        target_count=len(selected_contracts),
        model_count=len(settings["model_ids"]),
        fold_count=len(fold_plan["folds"]),
        seed_count=len(settings["seeds"]),
        hyperparameter_candidate_count=1,
    )
    plan = {
        "schema_version": TOURNAMENT_SCHEMA_VERSION,
        "plan_only": bool(settings["plan_only"]),
        "dataset_identity": _dataset_identity(rows, source_path),
        "real_artifact_audit": real_artifact_audit,
        "target_eligibility": eligible["statistics"],
        "shared_fold_plan": fold_plan,
        "fit_count": fit_count,
        "models": list(settings["model_ids"]),
        "seeds": list(settings["seeds"]),
        "feature_columns": list(feature_columns),
        "leakage_audit": leakage_audit,
        "blockers": blockers,
    }
    if settings["plan_only"] or blockers:
        return _payload(
            rows=rows,
            source_path=source_path,
            settings=settings,
            config=config,
            contracts=selected_contracts,
            plan=plan,
            oos_predictions=[],
            forecast_metrics=[],
            execution_reconciliation=_execution_reconciliation(fit_count, None, settings),
            promotion_results={},
            target_summary=[],
            reference_target_deltas=[],
            blockers=blockers,
        )
    oos_predictions, wide_rows, forecast_metrics, execution = _run_tournament(
        eligible["matched_rows"],
        selected_contracts,
        feature_columns=feature_columns,
        settings=settings,
        fold_plan=fold_plan,
    )
    promotion_results = _promotion_results(wide_rows, selected_contracts, settings, config)
    target_summary = _target_summary(forecast_metrics, promotion_results, settings)
    deltas = _reference_deltas(forecast_metrics, promotion_results, settings)
    return _payload(
        rows=rows,
        source_path=source_path,
        settings=settings,
        config=config,
        contracts=selected_contracts,
        plan=plan,
        oos_predictions=oos_predictions,
        forecast_metrics=forecast_metrics,
        execution_reconciliation=_execution_reconciliation(fit_count, execution, settings),
        promotion_results=promotion_results,
        target_summary=target_summary,
        reference_target_deltas=deltas,
        blockers=[],
    )


def discover_target_contracts(rows: list[dict[str, Any]], settings: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    configured = dict((settings or {}).get("targets", {}) or {})
    contracts = []
    for target_id, defaults in DEFAULT_TARGETS.items():
        spec = {**defaults, **dict(configured.get(target_id, {}) or {})}
        column = str(spec["target_column"])
        task_type = str(spec["task_type"])
        present = bool(rows and column in rows[0])
        non_null = sum(1 for row in rows if finite_number(row.get(column)) is not None)
        provenance_complete = sum(
            1
            for row in rows
            if finite_number(row.get(column)) is not None
            and str(row.get("target_provenance_contract_version")) == TARGET_PROVENANCE_CONTRACT_VERSION
            and str(row.get("label_available_timestamp", "")).strip()
        )
        status = "available_and_validated"
        reason = None
        if not present:
            status, reason = "unavailable", "target_column_missing"
        elif non_null == 0:
            status, reason = "unavailable", "target_column_all_null"
        elif task_type == "binary_classification":
            status, reason = "incompatible_with_regression_training_path", "classification target requested for regression tournament"
        elif provenance_complete < non_null:
            status, reason = "blocked_by_missing_provenance", "one or more target rows lack canonical label availability provenance"
        contract = {
            "contract_version": TARGET_CONTRACT_VERSION,
            "target_id": target_id,
            "target_column": column,
            "task_type": task_type,
            "prediction_semantics": spec["prediction_semantics"],
            "horizon_sessions": 10,
            "higher_is_better": True,
            "benchmark_symbol": "row.benchmark_symbol",
            "benchmark_return_definition": BENCHMARK_RETURN_COLUMN,
            "normalisation_scope": spec["normalisation_scope"],
            "mathematical_definition": spec["mathematical_definition"],
            "required_provenance_columns": list(TARGET_PROVENANCE_COLUMNS),
            "label_availability_column": "label_available_timestamp",
            "boundary_status_column": "target_status",
            "target_start_timestamp_column": "target_start_timestamp",
            "target_end_timestamp_column": "label_end_timestamp",
            "missing_value_behavior": "rows with non-finite selected target are excluded",
            "boundary_row_behavior": "target_status=unrealized_boundary or missing label availability is excluded",
            "used_by_real_training_command": column in {"actual_forward_return_10d", "actual_market_residual_return_10d", "actual_vol_adjusted_forward_return_10d", "actual_rank_normalized_forward_return_10d"},
            "automated_tests": True,
            "point_in_time_safe": status == "available_and_validated",
            "classification": status,
            "classification_reason": reason,
        }
        contract["contract_identity"] = _hash({k: v for k, v in contract.items() if k != "contract_identity"})
        contracts.append(contract)
    return sorted(contracts, key=lambda row: row["target_id"])


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("selector_target_tournament", {}) or {})
    output = dict(raw.get("output", {}) or {})
    bounded = dict(raw.get("bounded", {}) or {})
    default_output = stock_alpha_output_dir(config) / "selector_target_tournament"
    return {
        "enabled": bool(raw.get("enabled", False)),
        "comparison_mode": str(raw.get("comparison_mode", "target_intersection")),
        "reference_target_id": str(raw.get("reference_target_id", "raw_return_10d")),
        "target_ids": list(raw.get("target_ids", ["raw_return_10d", "market_residual_return_10d"])),
        "targets": dict(raw.get("targets", {}) or {}),
        "model_ids": list(raw.get("model_ids", ["ridge", "elastic_net"])),
        "feature_set_id": str(raw.get("feature_set_id", "stock_level_default_features")),
        "seeds": list(raw.get("seeds", [int(ml.get("random_seed", 42))])),
        "plan_only": bool(raw.get("plan_only", False)),
        "source_dataset_path": str(raw.get("source_dataset_path", ml.get("stock_level_prediction_artifacts_path", stock_alpha_output_dir(config) / "stock_level_prediction_artifacts.parquet"))),
        "allow_csv_fallback": bool(raw.get("allow_csv_fallback", False)),
        "expected_dataset": dict(raw.get("expected_dataset", {}) or {}),
        "output_dir": str(output.get("dir", raw.get("output_dir", default_output))),
        "write_predictions": bool(output.get("write_predictions", True)),
        "write_target_summary": bool(output.get("write_target_summary", True)),
        "write_portfolio_promotion_report": bool(output.get("write_portfolio_promotion_report", True)),
        "write_debug_csv": bool(output.get("write_debug_csv", False)),
        "maximum_decision_dates": bounded.get("maximum_decision_dates"),
        "maximum_symbols": bounded.get("maximum_symbols"),
        "maximum_folds": bounded.get("maximum_folds"),
        "minimum_symbols_per_date": int(raw.get("minimum_symbols_per_date", 2)),
        "min_train_dates": int(raw.get("min_train_dates", ml.get("stock_ranker_min_train_dates", 52))),
        "test_window_dates": int(raw.get("test_window_dates", ml.get("stock_ranker_test_window_dates", 13))),
        "embargo_dates": int(raw.get("embargo_dates", ml.get("stock_ranker_embargo_dates", 2))),
        "include_engineered_features": bool(ml.get("stock_ranker_include_engineered_features", False)),
        "sklearn_n_jobs": int(ml.get("sklearn_n_jobs", 1)),
        "promotion_config": dict(ml.get("selector_promotion", {}) or {}),
    }


def _real_artifact_audit(
    rows: list[dict[str, Any]],
    source_path: Path | None,
    settings: Mapping[str, Any],
    contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = [_row_key(row) for row in rows]
    path = source_path.resolve() if source_path else None
    identity = (
        artifact_identity(source_path, rows=rows, fieldnames=list(rows[0])) if source_path and source_path.exists() and rows else {}
    )
    expected_assertions = _expected_dataset_assertions(rows, source_path, settings, identity)
    provenance = _target_provenance_audit(rows, contracts, settings)
    blockers = [
        f"expected_dataset:{row['name']}" for row in expected_assertions if row["status"] == "fail"
    ]
    blockers.extend(
        f"target_provenance:{target_id}:{issue}"
        for target_id, audit in provenance.items()
        for issue in audit.get("blocking_issues", [])
        if audit.get("selected_for_tournament")
    )
    return {
        "schema_version": "selector_target_real_artifact_audit_v1",
        "configured_path": str(settings["source_dataset_path"]),
        "resolved_absolute_path": str(path) if path else None,
        "config_key": "ml.selector_target_tournament.source_dataset_path",
        "path_resolution_owner": "selector_target_tournament._settings",
        "exists": bool(source_path and source_path.exists()),
        "legacy_fallback_allowed": bool(settings.get("allow_csv_fallback", False)),
        "legacy_fallback_routes": [] if not settings.get("allow_csv_fallback", False) else ["explicit_allow_csv_fallback_for_legacy_fixtures"],
        "silent_legacy_fallback_permitted": False,
        "artifact_identity": identity,
        "row_count": len(rows),
        "decision_date_count": len({date for date, _ in keys}),
        "symbol_count": len({symbol for _, symbol in keys}),
        "minimum_rebalance_date": min((date for date, _ in keys), default=None),
        "maximum_rebalance_date": max((date for date, _ in keys), default=None),
        "decision_frequency": _first_present(rows, "decision_frequency"),
        "decision_grid_identity": _first_present(rows, "decision_grid_identity"),
        "decision_grid_version": _first_present(rows, "decision_grid_version"),
        "universe_identity": _hash(sorted({symbol for _, symbol in keys})),
        "target_provenance_audit": provenance,
        "expected_dataset_assertions": expected_assertions,
        "blockers": blockers,
        "status": "blocked" if blockers else "passed",
    }


def _expected_dataset_assertions(
    rows: list[dict[str, Any]],
    source_path: Path | None,
    settings: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = dict(settings.get("expected_dataset", {}) or {})
    if not expected:
        return []
    keys = [_row_key(row) for row in rows]
    actual = {
        "path": str(source_path.resolve()) if source_path else None,
        "minimum_rows": len(rows),
        "minimum_symbols": len({symbol for _, symbol in keys}),
        "minimum_decision_dates": len({date for date, _ in keys}),
        "expected_decision_frequency": _first_present(rows, "decision_frequency"),
        "expected_decision_grid_identity": _first_present(rows, "decision_grid_identity"),
        "expected_universe_identity": _hash(sorted({symbol for _, symbol in keys})),
        "expected_artifact_sha256": identity.get("sha256"),
    }
    checks = []
    for name, expected_value in expected.items():
        if expected_value in (None, ""):
            continue
        observed = actual.get(name)
        if name == "path":
            wanted = str(Path(str(expected_value)).resolve())
            passed = observed == wanted
            observed_value = observed
            expected_value = wanted
        elif name.startswith("minimum_"):
            passed = int(observed or 0) >= int(expected_value)
            observed_value = observed
        else:
            passed = str(observed) == str(expected_value)
            observed_value = observed
        checks.append({
            "name": name,
            "expected": expected_value,
            "observed": observed_value,
            "status": "pass" if passed else "fail",
        })
    return checks


def _target_provenance_audit(rows: list[dict[str, Any]], contracts: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    selected = set(settings.get("target_ids", []))
    output = {}
    for contract in contracts:
        column = contract["target_column"]
        values = [finite_number(row.get(column)) for row in rows if finite_number(row.get(column)) is not None]
        labelled = [row for row in rows if finite_number(row.get(column)) is not None]
        issues: list[str] = []
        for row in labelled:
            if str(row.get("target_status", "realized")) != "realized":
                issues.append("unrealized_boundary_rows_present")
                break
        for row in labelled:
            if str(row.get("target_provenance_contract_version")) != TARGET_PROVENANCE_CONTRACT_VERSION:
                issues.append("target_provenance_contract_version_missing_or_mismatch")
                break
        if any(not str(row.get("label_available_timestamp", "")).strip() for row in labelled):
            issues.append("label_available_timestamp_missing")
        if _timestamp_order_failure(labelled, "target_start_timestamp", "label_end_timestamp", allow_equal=True):
            issues.append("target_start_after_label_end")
        if _timestamp_order_failure(labelled, "feature_data_cutoff_timestamp", "decision_timestamp", allow_equal=True):
            issues.append("feature_cutoff_after_decision_timestamp")
        if _timestamp_order_failure(labelled, "decision_timestamp", "first_actionable_session", allow_equal=False):
            issues.append("decision_timestamp_not_before_first_actionable_session")
        if _timestamp_order_failure(labelled, "label_end_timestamp", "label_available_timestamp", allow_equal=True):
            issues.append("label_available_before_label_end")
        if contract["task_type"] == "binary_classification":
            safety = "unsupported_for_regression_tournament"
        elif contract["classification"] != "available_and_validated":
            safety = "blocked"
        elif issues:
            safety = "blocked"
        elif contract["normalisation_scope"] == "none":
            safety = "safe_for_bounded_tournament"
        else:
            safety = "provisionally_safe_for_bounded_tournament"
        output[contract["target_id"]] = {
            "target_column": column,
            "selected_for_tournament": contract["target_id"] in selected,
            "present": bool(rows and column in rows[0]),
            "non_null_count": len(values),
            "null_count": len(rows) - len(values),
            "first_labelled_decision_date": min((str(row.get("rebalance_date")) for row in labelled), default=None),
            "last_labelled_decision_date": max((str(row.get("rebalance_date")) for row in labelled), default=None),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "provenance_contract_version": _first_present(labelled, "target_provenance_contract_version"),
            "label_availability_populated_count": sum(1 for row in labelled if str(row.get("label_available_timestamp", "")).strip()),
            "benchmark_provenance_populated_count": sum(1 for row in labelled if str(row.get("benchmark_label_available_timestamp", "")).strip()),
            "contract_classification": contract["classification"],
            "safety_classification": safety,
            "blocking_issues": sorted(set(issues)),
        }
    return output


def _timestamp_order_failure(rows: list[dict[str, Any]], left: str, right: str, *, allow_equal: bool) -> bool:
    for row in rows:
        left_value = _timestamp_key(row.get(left))
        right_value = _timestamp_key(row.get(right))
        if left_value is None or right_value is None:
            continue
        if allow_equal and left_value > right_value:
            return True
        if not allow_equal and left_value >= right_value:
            return True
    return False


def _timestamp_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).replace("Z", "+00:00")


def _text_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _target_eligible_rows(rows: list[dict[str, Any]], contracts: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    if settings["comparison_mode"] not in {"target_intersection", "native_target_coverage"}:
        raise ValueError("selector_target_tournament.comparison_mode must be target_intersection or native_target_coverage")
    bounded = _bounded_rows(rows, settings)
    keys_by_target: dict[str, set[tuple[str, str]]] = {}
    boundary_excluded: dict[str, int] = {}
    for contract in contracts:
        keys: set[tuple[str, str]] = set()
        boundary = 0
        for row in bounded:
            if str(row.get("target_status", "realized")) != "realized":
                boundary += 1
                continue
            if str(row.get("target_provenance_contract_version")) != TARGET_PROVENANCE_CONTRACT_VERSION:
                continue
            if not str(row.get("label_available_timestamp", "")).strip():
                continue
            if finite_number(row.get(contract["target_column"])) is None:
                continue
            keys.add(_row_key(row))
        keys_by_target[contract["target_id"]] = keys
        boundary_excluded[contract["target_id"]] = boundary
    common = set.intersection(*keys_by_target.values()) if keys_by_target else set()
    if settings["comparison_mode"] == "native_target_coverage":
        common = set.union(*keys_by_target.values()) if keys_by_target else set()
    common = _keys_with_min_symbols(common, settings["minimum_symbols_per_date"])
    matched = [row for row in bounded if _row_key(row) in common]
    dates = sorted({date for date, _ in common})
    symbols = sorted({symbol for _, symbol in common})
    stats = {
        "source_row_count": len(rows),
        "bounded_row_count": len(bounded),
        "eligible_rows_per_target": {target_id: len(keys) for target_id, keys in keys_by_target.items()},
        "common_target_intersection_row_count": len(common),
        "common_decision_date_count": len(dates),
        "common_symbol_count": len(symbols),
        "first_common_decision_date": dates[0] if dates else None,
        "last_common_decision_date": dates[-1] if dates else None,
        "rows_excluded_per_target": {target_id: len(keys - common) for target_id, keys in keys_by_target.items()},
        "dates_excluded_per_target": {target_id: len({date for date, _ in keys - common}) for target_id, keys in keys_by_target.items()},
        "boundary_rows_excluded": boundary_excluded,
    }
    return {"matched_rows": sorted(matched, key=lambda row: _row_key(row)), "statistics": stats}


def _bounded_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    dates = sorted({str(row.get("rebalance_date", "")) for row in rows if str(row.get("rebalance_date", "")).strip()})
    if settings.get("maximum_decision_dates"):
        dates = dates[: int(settings["maximum_decision_dates"])]
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if str(row.get("symbol", "")).strip()})
    if settings.get("maximum_symbols"):
        symbols = symbols[: int(settings["maximum_symbols"])]
    return [row for row in rows if str(row.get("rebalance_date")) in dates and str(row.get("symbol", "")).upper() in symbols]


def _shared_fold_plan(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    min_train_dates: int,
    test_window_dates: int,
    embargo_dates: int,
    maximum_folds: Any,
) -> dict[str, Any]:
    prepared, excluded = _prepare_rows(rows, feature_columns)
    dates = sorted({row["rebalance_date"] for row in prepared})
    first_test_index = min_train_dates + embargo_dates
    if len(dates) <= first_test_index:
        return {
            "status": "blocked",
            "blocked_reason": "insufficient_dates_for_shared_fold_plan",
            "folds": [],
            "fold_plan_identity": _hash({"rows": len(prepared), "dates": dates, "blocked": True}),
            "excluded_incomplete_row_count": excluded,
        }
    folds, _ = _build_oos_prediction_rows(
        prepared,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    )
    if maximum_folds:
        folds = folds[: int(maximum_folds)]
    identity = _hash({
        "min_train_dates": min_train_dates,
        "test_window_dates": test_window_dates,
        "embargo_dates": embargo_dates,
        "folds": folds,
        "row_keys": [_row_key(row) for row in prepared],
    })
    return {
        "status": "ready",
        "folds": folds,
        "fold_count": len(folds),
        "fold_plan_identity": identity,
        "excluded_incomplete_row_count": excluded,
    }


def _run_tournament(
    rows: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    settings: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if fold_plan.get("status") != "ready":
        raise ValueError("Shared fold plan is not ready")
    model_ids = _validate_model_ids(settings["model_ids"])
    seeds = [int(seed) for seed in settings["seeds"]]
    all_predictions: list[dict[str, Any]] = []
    wide_by_key: dict[tuple[str, str], dict[str, Any]] = {
        _row_key(row): _base_wide_row(row) for row in rows
    }
    forecast_metrics: list[dict[str, Any]] = []
    execution = {
        "attempted_fits": 0,
        "completed_fits": 0,
        "failed_fits": 0,
        "prediction_producing_fits": 0,
        "executed_seed_count": len(seeds),
        "executed_seeds": seeds,
        "model_count": len(model_ids),
        "target_count": len(contracts),
        "fold_count": len(fold_plan["folds"]),
    }
    for contract in contracts:
        for seed in seeds:
            factories = _model_factories(seed, int(settings["sklearn_n_jobs"]))
            factories = {name: factories[name] for name in model_ids}
            execution["attempted_fits"] += len(model_ids) * len(fold_plan["folds"])
            predictions, payload = build_stock_level_model_ranking_benchmark(
                rows,
                target_column=contract["target_column"],
                feature_columns=feature_columns,
                min_train_dates=int(settings["min_train_dates"]),
                test_window_dates=int(settings["test_window_dates"]),
                embargo_dates=int(settings["embargo_dates"]),
                random_seed=seed,
                sklearn_n_jobs=int(settings["sklearn_n_jobs"]),
                model_n_jobs=1,
                include_sequence_models=False,
                model_factories=factories,
                sequence_model_factories={},
            )
            execution["completed_fits"] += len(model_ids) * len(fold_plan["folds"])
            _assert_fold_compatibility(payload, fold_plan)
            allowed_oos_dates = {
                date
                for fold in fold_plan["folds"]
                for date in _dates_between(fold["oos_prediction_date_min"], fold["oos_prediction_date_max"], predictions)
            }
            predictions = [row for row in predictions if str(row["rebalance_date"]) in allowed_oos_dates]
            for model_id in model_ids:
                source_column = f"{PREDICTION_PREFIX}{model_id}"
                candidate_name = _candidate_name(contract["target_id"], model_id, seed)
                target_column = f"{PREDICTION_PREFIX}{candidate_name}"
                candidate_id = f"{contract['target_id']}::{model_id}::seed_{seed}"
                metric_rows = []
                for row in predictions:
                    key = _row_key(row)
                    source = wide_by_key[key]
                    prediction = row.get(source_column)
                    source["fold_id"] = row["fold_id"]
                    source[target_column] = prediction
                    metric_rows.append({
                        "rebalance_date": row["rebalance_date"],
                        "symbol": row["symbol"],
                        "actual_selected_target": source[contract["target_column"]],
                        target_column: prediction,
                    })
                    all_predictions.append({
                        "candidate_id": candidate_id,
                        "target_id": contract["target_id"],
                        "target_contract_identity": contract["contract_identity"],
                        "model_id": model_id,
                        "model_configuration_identity": _hash({"model_id": model_id, "seed": seed}),
                        "feature_contract_identity": _hash({"feature_set_id": settings["feature_set_id"], "feature_columns": feature_columns}),
                        "dataset_identity": _hash([_row_key(item) for item in rows]),
                        "fold_plan_identity": fold_plan["fold_plan_identity"],
                        "seed": seed,
                        "prediction_semantics": contract["prediction_semantics"],
                        "strict_oos": True,
                        "strict_oos_status": True,
                        "fold_id": row["fold_id"],
                        "rebalance_date": row["rebalance_date"],
                        "decision_timestamp": _text_or_none(source.get("decision_timestamp") or row["rebalance_date"]),
                        "symbol": row["symbol"],
                        "prediction": prediction,
                        "actual_selected_target": source[contract["target_column"]],
                        "actual_investable_return_10d": source[ECONOMIC_RETURN_COLUMN],
                        "target_availability_timestamp": _text_or_none(source.get("label_available_timestamp")),
                    })
                if any(finite_number(row.get(target_column)) is not None for row in metric_rows):
                    execution["prediction_producing_fits"] += len(fold_plan["folds"])
                forecast_metrics.append(_forecast_metric(candidate_id, contract["target_id"], model_id, seed, metric_rows, target_column))
    for row in wide_by_key.values():
        for contract in contracts:
            for seed in seeds:
                for model_id in model_ids:
                    row.setdefault(f"{PREDICTION_PREFIX}{_candidate_name(contract['target_id'], model_id, seed)}", "")
    _reject_duplicate_predictions(all_predictions)
    return (
        sorted(all_predictions, key=lambda row: (row["candidate_id"], row["rebalance_date"], row["symbol"])),
        list(wide_by_key.values()),
        sorted(forecast_metrics, key=lambda row: row["candidate_id"]),
        execution,
    )


def _promotion_results(wide_rows: list[dict[str, Any]], contracts: list[dict[str, Any]], settings: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    completed = [
        _candidate_name(contract["target_id"], model_id, int(seed))
        for contract in contracts
        for seed in settings["seeds"]
        for model_id in settings["model_ids"]
    ]
    promotion = {**DEFAULT_PROMOTION, **dict(settings.get("promotion_config") or {})}
    promotion["fixed_policy"] = {**DEFAULT_PROMOTION["fixed_policy"], **dict(promotion.get("fixed_policy", {}) or {})}
    promotion["ranking"] = {**DEFAULT_PROMOTION["ranking"], **dict(promotion.get("ranking", {}) or {})}
    promotion["gates"] = {**DEFAULT_PROMOTION["gates"], **dict(promotion.get("gates", {}) or {})}
    promotion["comparison_mode"] = "intersection"
    promotion["candidate_types"] = ["single_model"]
    promotion["gates"].setdefault("require_outperformance_of_baseline", False)
    return build_selector_portfolio_promotion(
        wide_rows,
        benchmark={
            "walk_forward": {"out_of_sample_only": True},
            "completed_models": completed,
            "feature_columns": [],
            "best_ml_model": None,
        },
        promotion_config=promotion,
        predictions_path=None,
        benchmark_path=None,
        config=config,
    )


def _forecast_metric(candidate_id: str, target_id: str, model_id: str, seed: int, rows: list[dict[str, Any]], prediction_column: str) -> dict[str, Any]:
    evaluation = CrossSectionalRankingEvaluator(target_column="actual_selected_target").evaluate(
        rows,
        name=candidate_id,
        signal_column=prediction_column,
        kind="target_model",
    )
    coverage = sum(1 for row in rows if finite_number(row.get(prediction_column)) is not None) / len(rows) if rows else 0.0
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "model_id": model_id,
        "seed": seed,
        "prediction_column": prediction_column,
        "target_fit_metric_scope": "selected_target",
        "economic_metric_scope": "portfolio_replay_uses_actual_forward_return_10d",
        "prediction_coverage": coverage,
        **evaluation,
    }


def _target_summary(forecast_metrics: list[dict[str, Any]], promotion: Mapping[str, Any], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    portfolio = {row["candidate_id"].replace("single_model:", "").replace("__", "::"): row for row in promotion.get("candidate_metrics", [])}
    gates = {row["candidate_id"].replace("single_model:", "").replace("__", "::"): row for row in promotion.get("gate_results", [])}
    summaries = []
    for target_id in settings["target_ids"]:
        metrics = [row for row in forecast_metrics if row["target_id"] == target_id]
        candidates = [row["candidate_id"] for row in metrics]
        portfolio_rows = [portfolio[candidate] for candidate in candidates if candidate in portfolio]
        gate_rows = [gates[candidate] for candidate in candidates if candidate in gates]
        best_forecast = max(metrics, key=lambda row: _metric_value(row.get("mean_spearman_ic")), default={})
        best_portfolio = max(portfolio_rows, key=lambda row: _metric_value(row.get("net_sharpe")), default={})
        summaries.append({
            "target_id": target_id,
            "eligible_model_count": len(metrics),
            "best_model_by_forecast_metric": best_forecast.get("model_id"),
            "best_model_by_net_portfolio_metric": _model_from_candidate(best_portfolio.get("candidate_id")),
            "median_model_net_cagr": _median([row.get("net_cagr") for row in portfolio_rows]),
            "median_model_net_sharpe": _median([row.get("net_sharpe") for row in portfolio_rows]),
            "median_model_drawdown": _median([row.get("max_drawdown") for row in portfolio_rows]),
            "median_model_turnover": _median([row.get("annualized_turnover") for row in portfolio_rows]),
            "fraction_of_models_beating_reference_target": None,
            "fraction_passing_promotion_gates": _fraction([row.get("overall_status") == "eligible" for row in gate_rows]),
        })
    return summaries


def _reference_deltas(forecast_metrics: list[dict[str, Any]], promotion: Mapping[str, Any], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = str(settings["reference_target_id"])
    portfolio = {row["candidate_id"].replace("single_model:", "").replace("__", "::"): row for row in promotion.get("candidate_metrics", [])}
    forecast = {(row["target_id"], row["model_id"], int(row["seed"])): row for row in forecast_metrics}
    rows = []
    for target_id in settings["target_ids"]:
        if target_id == reference:
            continue
        for seed in settings["seeds"]:
            for model_id in settings["model_ids"]:
                left_id = f"{target_id}::{model_id}::seed_{int(seed)}"
                right_id = f"{reference}::{model_id}::seed_{int(seed)}"
                left, right = portfolio.get(left_id), portfolio.get(right_id)
                left_f, right_f = forecast.get((target_id, model_id, int(seed))), forecast.get((reference, model_id, int(seed)))
                rows.append({
                    "target_id": target_id,
                    "model_id": model_id,
                    "seed": int(seed),
                    "reference_target_id": reference,
                    "delta_net_cagr": _delta(left, right, "net_cagr"),
                    "delta_net_sharpe": _delta(left, right, "net_sharpe"),
                    "delta_max_drawdown": _delta(left, right, "max_drawdown"),
                    "delta_turnover": _delta(left, right, "annualized_turnover"),
                    "delta_cost_drag": _delta(left, right, "cost_drag"),
                    "delta_ic": _delta(left_f, right_f, "mean_spearman_ic"),
                })
    return rows


def _payload(**kwargs: Any) -> dict[str, Any]:
    rows = kwargs["rows"]
    source_path = kwargs["source_path"]
    settings = kwargs["settings"]
    config = kwargs["config"]
    status = "plan_only" if settings["plan_only"] else ("blocked" if kwargs["blockers"] else "completed")
    return {
        "schema_version": TOURNAMENT_SCHEMA_VERSION,
        "mode": "selector_target_tournament_research_only",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_lineage": _dataset_identity(rows, source_path),
        "real_artifact_audit": kwargs["plan"]["real_artifact_audit"],
        "target_contracts": kwargs["contracts"],
        "plan": kwargs["plan"],
        "candidate_identities": sorted({row["candidate_id"] for row in kwargs["oos_predictions"]}),
        "fit_count_plan": kwargs["plan"]["fit_count"],
        "execution_reconciliation": kwargs["execution_reconciliation"],
        "worker_thread_configuration": _worker_thread_configuration(settings),
        "forecast_metrics": kwargs["forecast_metrics"],
        "oos_predictions": kwargs["oos_predictions"],
        "portfolio_metrics": kwargs["promotion_results"].get("candidate_metrics", []),
        "promotion_results": kwargs["promotion_results"],
        "target_summary": kwargs["target_summary"],
        "reference_target_deltas": kwargs["reference_target_deltas"],
        "lineage": {
            "source_dataset_path": str(source_path) if source_path else None,
            "source_artifact_hash": file_digest(source_path) if source_path and source_path.exists() else None,
            "target_contract_identities": {row["target_id"]: row["contract_identity"] for row in kwargs["contracts"]},
            "shared_fold_plan_identity": kwargs["plan"]["shared_fold_plan"].get("fold_plan_identity"),
            "model_configuration_identities": {model_id: _hash({"model_id": model_id, "seeds": settings["seeds"]}) for model_id in settings["model_ids"]},
            "seed_list": settings["seeds"],
            "tournament_configuration_hash": _hash(settings),
            "promotion_configuration_hash": _hash(settings.get("promotion_config", {})),
            "code_commit": MLCoreArtifactWriter.git_commit(),
            "config_hash": _hash(config),
        },
        "warnings": _warnings(kwargs["contracts"], settings),
        "blockers": kwargs["blockers"],
        "training_performed": not settings["plan_only"] and not kwargs["blockers"],
        "final_fit_performed": False,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "promotion_thresholds_changed": False,
    }


def _feature_columns(rows: list[dict[str, Any]], config: Mapping[str, Any], settings: Mapping[str, Any]) -> tuple[str, ...]:
    columns = _available_feature_columns(rows, include_engineered=bool(settings["include_engineered_features"]))
    blocked = [column for column in columns if column in OUTCOME_DENYLIST or column.startswith("actual_")]
    if blocked:
        raise ValueError(f"Feature set contains outcome columns: {blocked}")
    return columns


def _leakage_audit(feature_columns: tuple[str, ...], contracts: list[dict[str, Any]]) -> dict[str, Any]:
    denied = set(OUTCOME_DENYLIST) | {row["target_column"] for row in contracts}
    denied.update(column for column in ALL_FEATURE_COLUMNS if column.startswith("actual_"))
    blocked = sorted(column for column in feature_columns if column in denied or column.startswith("actual_"))
    return {
        "feature_columns": list(feature_columns),
        "denied_outcome_columns": sorted(denied),
        "blocked_feature_columns": blocked,
        "passed": not blocked,
    }


def _contract_blockers(contracts: list[dict[str, Any]]) -> list[str]:
    blockers = []
    for contract in contracts:
        if contract["classification"] != "available_and_validated":
            blockers.append(f"{contract['target_id']}:{contract['classification_reason']}")
    return blockers


def _execution_reconciliation(
    fit_count_plan: Mapping[str, Any],
    execution: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    if execution is None:
        return {
            "status": "not_executed",
            "reason": "plan_only_or_blocked",
            "expected_base_fits": fit_count_plan["expected_base_fits"],
            "attempted_fits": 0,
            "completed_fits": 0,
            "failed_fits": 0,
            "prediction_producing_fits": 0,
            "configured_seeds": list(settings["seeds"]),
            "executed_seeds": [],
            "all_configured_seeds_executed": False,
            "final_fit_performed": False,
        }
    expected = int(fit_count_plan["expected_base_fits"])
    completed = int(execution.get("completed_fits", 0))
    attempted = int(execution.get("attempted_fits", 0))
    return {
        "status": "reconciled" if attempted == expected and completed == expected else "mismatch",
        "expected_base_fits": expected,
        "attempted_fits": attempted,
        "completed_fits": completed,
        "failed_fits": int(execution.get("failed_fits", 0)),
        "prediction_producing_fits": int(execution.get("prediction_producing_fits", 0)),
        "configured_seeds": list(settings["seeds"]),
        "executed_seeds": list(execution.get("executed_seeds", [])),
        "all_configured_seeds_executed": sorted(map(int, settings["seeds"])) == sorted(map(int, execution.get("executed_seeds", []))),
        "target_count": int(execution.get("target_count", 0)),
        "model_count": int(execution.get("model_count", 0)),
        "fold_count": int(execution.get("fold_count", 0)),
        "final_fit_performed": False,
    }


def _worker_thread_configuration(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "process_id": os.getpid(),
        "sklearn_n_jobs": int(settings["sklearn_n_jobs"]),
        "model_n_jobs": 1,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def _fit_count(target_count: int, model_count: int, fold_count: int, seed_count: int, hyperparameter_candidate_count: int) -> dict[str, int]:
    base = target_count * model_count * fold_count * seed_count * hyperparameter_candidate_count
    return {
        "target_count": target_count,
        "model_count": model_count,
        "fold_count": fold_count,
        "seed_count": seed_count,
        "hyperparameter_candidate_count": hyperparameter_candidate_count,
        "expected_base_fits": base,
        "expected_validation_refits": 0,
        "expected_final_fits": 0,
        "total_expected_fits": base,
    }


def _validate_model_ids(model_ids: list[str]) -> list[str]:
    invalid = [model_id for model_id in model_ids if model_id not in TABULAR_MODEL_NAMES]
    if invalid:
        raise ValueError(f"Selector target tournament supports initial tabular model_ids only: invalid={invalid}")
    return list(model_ids)


def _base_wide_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "fold_id": "",
    }


def _assert_fold_compatibility(payload: Mapping[str, Any], fold_plan: Mapping[str, Any]) -> None:
    expected = [(fold["fold_id"], fold["oos_prediction_date_min"], fold["oos_prediction_date_max"]) for fold in fold_plan["folds"]]
    actual = [(fold["fold_id"], fold["oos_prediction_date_min"], fold["oos_prediction_date_max"]) for fold in payload.get("walk_forward", {}).get("folds", [])[: len(expected)]]
    if expected != actual:
        raise ValueError("Target tournament fold plan mismatch")


def _dates_between(start: str, end: str, prediction_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["rebalance_date"]) for row in prediction_rows if start <= str(row["rebalance_date"]) <= end}


def _reject_duplicate_rows(rows: list[dict[str, Any]]) -> None:
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Selector target tournament source rows must be unique by rebalance_date and symbol")


def _reject_duplicate_predictions(rows: list[dict[str, Any]]) -> None:
    keys = [(row["candidate_id"], row["rebalance_date"], row["symbol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate selector target OOS prediction rows")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper())


def _keys_with_min_symbols(keys: set[tuple[str, str]], minimum: int) -> set[tuple[str, str]]:
    counts: dict[str, int] = {}
    for date, _ in keys:
        counts[date] = counts.get(date, 0) + 1
    return {key for key in keys if counts[key[0]] >= minimum}


def _candidate_name(target_id: str, model_id: str, seed: int) -> str:
    return f"{target_id}__{model_id}__seed_{seed}"


def _dataset_identity(rows: list[dict[str, Any]], source_path: Path | None) -> dict[str, Any]:
    keys = [_row_key(row) for row in rows]
    return {
        "source_dataset_path": str(source_path) if source_path else None,
        "source_artifact_sha256": file_digest(source_path) if source_path and source_path.exists() else None,
        "row_count": len(rows),
        "decision_date_count": len({date for date, _ in keys}),
        "symbol_count": len({symbol for _, symbol in keys}),
        "decision_grid_identity": _first_present(rows, "decision_grid_identity"),
        "universe_identity": _hash(sorted({symbol for _, symbol in keys})),
        "logical_content_hash": _hash(keys),
    }


def _first_present(rows: list[dict[str, Any]], column: str) -> str | None:
    return next((str(row.get(column)) for row in rows if str(row.get(column, "")).strip()), None)


def _metric_value(value: Any) -> float:
    number = finite_number(value)
    return number if number is not None else -math.inf


def _median(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if finite_number(value) is not None]
    return median(finite) if finite else None


def _fraction(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def _model_from_candidate(candidate_id: Any) -> str | None:
    if not candidate_id:
        return None
    text = str(candidate_id).replace("single_model:", "").replace("__", "::")
    return text.split("::", 1)[1] if "::" in text else None


def _delta(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None, metric: str) -> float | None:
    if not left or not right:
        return None
    a, b = finite_number(left.get(metric)), finite_number(right.get(metric))
    return a - b if a is not None and b is not None else None


def _warnings(contracts: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[str]:
    warnings = ["BOUNDED DIAGNOSTIC ONLY / NOT PROMOTION EVIDENCE"]
    if any(row["normalisation_scope"] != "none" for row in contracts):
        warnings.append("target_scales_differ; portfolio comparison uses prediction ordering as cross-sectional scores")
    if settings["comparison_mode"] == "native_target_coverage":
        warnings.append("native target coverage is diagnostic and not a sole fair target comparison")
    return warnings


def _fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    return [*preferred, *[key for key in dict.fromkeys(key for row in rows for key in row) if key not in preferred]] if rows else preferred


def _prediction_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "target_id",
        "model_id",
        "seed",
        "fold_id",
        "rebalance_date",
        "decision_timestamp",
        "symbol",
        "prediction",
        "actual_selected_target",
        "actual_investable_return_10d",
        "target_availability_timestamp",
        "strict_oos",
        "strict_oos_status",
        "dataset_identity",
        "target_contract_identity",
        "feature_contract_identity",
        "fold_plan_identity",
        "model_configuration_identity",
        "prediction_semantics",
    ]
    return _fields(rows, preferred)


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Target Tournament",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source rows: {payload['dataset_lineage']['row_count']}",
        f"- Targets: {', '.join(row['target_id'] for row in payload['target_contracts'])}",
        f"- Models: {', '.join(payload['plan']['models'])}",
        f"- Shared folds: {payload['plan']['shared_fold_plan'].get('fold_count', 0)}",
        f"- Expected fits: {payload['fit_count_plan']['total_expected_fits']}",
        f"- Recommended portfolio candidate: `{payload.get('promotion_results', {}).get('recommended_portfolio_candidate')}`",
        "",
        "## Target Summary",
        "",
        "| Target | Models | Best Forecast | Best Portfolio | Median Net Sharpe | Passing Gate Fraction |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in payload.get("target_summary", []):
        lines.append(
            f"| {row['target_id']} | {row['eligible_model_count']} | {row['best_model_by_forecast_metric']} | {row['best_model_by_net_portfolio_metric']} | {row['median_model_net_sharpe']} | {row['fraction_passing_promotion_gates']} |"
        )
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in payload["blockers"])
    return "\n".join(lines) + "\n"


def _artifact_audit_markdown(audit: Mapping[str, Any]) -> str:
    identity = dict(audit.get("artifact_identity", {}) or {})
    lines = [
        "# Selector Target Real Artifact Audit",
        "",
        "BOUNDED DIAGNOSTIC ONLY / NOT PROMOTION EVIDENCE.",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- Configured path: `{audit.get('configured_path')}`",
        f"- Resolved absolute path: `{audit.get('resolved_absolute_path')}`",
        f"- Silent legacy fallback permitted: `{audit.get('silent_legacy_fallback_permitted')}`",
        f"- Rows: {audit.get('row_count')}",
        f"- Symbols: {audit.get('symbol_count')}",
        f"- Decision dates: {audit.get('decision_date_count')}",
        f"- Decision frequency: `{audit.get('decision_frequency')}`",
        f"- Decision grid identity: `{audit.get('decision_grid_identity')}`",
        f"- Universe identity: `{audit.get('universe_identity')}`",
        f"- Artifact sha256: `{identity.get('sha256')}`",
        f"- Schema fingerprint: `{identity.get('schema_fingerprint')}`",
        "",
        "## Expected Dataset Assertions",
        "",
        "| Assertion | Expected | Observed | Status |",
        "|---|---|---|---|",
    ]
    for row in audit.get("expected_dataset_assertions", []):
        lines.append(f"| {row['name']} | `{row['expected']}` | `{row['observed']}` | {row['status']} |")
    if not audit.get("expected_dataset_assertions"):
        lines.append("| none_configured |  |  | skipped |")
    lines.extend(["", "## Target Provenance", "", "| Target | Column | Non-null | Safety | Issues |", "|---|---|---:|---|---|"])
    for target_id, row in sorted((audit.get("target_provenance_audit", {}) or {}).items()):
        issues = ", ".join(row.get("blocking_issues", [])) or "none"
        lines.append(f"| {target_id} | `{row.get('target_column')}` | {row.get('non_null_count')} | {row.get('safety_classification')} | {issues} |")
    if audit.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in audit["blockers"])
    return "\n".join(lines) + "\n"
