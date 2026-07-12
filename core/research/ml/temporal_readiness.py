from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from application.services.research_profiles import apply_research_profile
from config.config_loader import load_config
from core.research.ml.data.datasets import (
    build_dataset,
    dataset_leakage_audit,
    forbidden_predictor_columns,
)
from core.research.ml.meta.meta_dataset import build_meta_dataset_rows
from core.research.ml.meta.meta_evaluation import _walk_forward_meta_evaluation
from core.research.ml.stock_level.prediction_artifacts.types import (
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
)


READINESS_READY = "READY_FOR_LARGER_HISTORY_PRICE_ONLY_RUN"
READINESS_NOT_READY = "NOT_READY_FOR_LARGER_HISTORY_PRICE_ONLY_RUN"


@dataclass(frozen=True)
class ResolvedReadinessPaths:
    selector_config_path: Path
    exposure_config_path: Path
    meta_config_path: Path
    selector_input_path: Path
    selector_output_dir: Path
    exposure_dataset_path: Path
    exposure_output_dir: Path
    meta_expanded_dataset_path: Path
    meta_dataset_path: Path
    meta_output_dir: Path
    meta_source_prediction_dirs: list[Path]


def resolve_readiness_paths(
    *,
    selector_config_path: Path = Path("configs/research/stock_level_alpha_benchmark.yaml"),
    exposure_config_path: Path = Path("configs/research/logistic_regression_should_reduce_exposure.yaml"),
    meta_config_path: Path = Path("configs/research/regime_transformer_meta_ensemble_v1.yaml"),
    profile_name: str = "development",
) -> ResolvedReadinessPaths:
    selector = apply_research_profile(
        load_config(str(selector_config_path), overlay_project_config=True),
        profile_name,
    )
    exposure = apply_research_profile(
        load_config(str(exposure_config_path), overlay_project_config=True),
        profile_name,
    )
    meta = apply_research_profile(
        load_config(str(meta_config_path), overlay_project_config=True),
        profile_name,
    )
    selector_ml = selector.get("ml", {}) or {}
    exposure_ml = exposure.get("ml", {}) or {}
    meta_ml = meta.get("ml", {}) or {}
    return ResolvedReadinessPaths(
        selector_config_path=selector_config_path,
        exposure_config_path=exposure_config_path,
        meta_config_path=meta_config_path,
        selector_input_path=Path(
            selector_ml.get(
                "stock_level_prediction_artifacts_path",
                "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_level_prediction_artifacts_enriched.csv",
            )
        ),
        selector_output_dir=Path(selector_ml.get("output_dir", "reports/ml")),
        exposure_dataset_path=Path(
            exposure_ml.get(
                "expanded_rebalance_dataset_path",
                "cache/ml/expanded_rebalance_dataset.csv",
            )
        ),
        exposure_output_dir=Path(
            exposure_ml.get(
                "output_dir",
                "reports/ml/logistic_regression_should_reduce_exposure",
            )
        ),
        meta_expanded_dataset_path=Path(
            meta_ml.get(
                "expanded_rebalance_dataset_path",
                "cache/ml/expanded_rebalance_dataset.csv",
            )
        ),
        meta_dataset_path=Path(
            meta_ml.get("meta_dataset_path", "cache/ml/meta_ensemble_dataset.csv")
        ),
        meta_output_dir=Path(
            meta_ml.get("output_dir", "reports/ml/regime_transformer_meta_ensemble_v1")
        ),
        meta_source_prediction_dirs=[
            Path(path) for path in meta_ml.get("source_prediction_dirs", [])
        ],
    )


def audit_selector_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv_if_exists(path)
    audit = _temporal_row_audit(
        rows,
        key_columns=("rebalance_date", "symbol"),
        target_column="actual_forward_return_10d",
        fail_closed_same_date=True,
    )
    audit.update({
        "artifact_path": str(path),
        "artifact_exists": path.exists(),
        "dataset_hash": _file_digest(path) if path.exists() else None,
        "target_convention": "forward stock return target; availability must be after decision date",
        **_selector_target_provenance_audit(rows),
    })
    return audit


def audit_exposure_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv_if_exists(path)
    audit = _temporal_row_audit(
        rows,
        key_columns=("feature_id",),
        target_column="should_reduce_exposure",
        fail_closed_same_date=True,
    )
    predictor_audit = _exposure_predictor_audit(rows)
    audit.update({
        "artifact_path": str(path),
        "artifact_exists": path.exists(),
        "dataset_hash": _file_digest(path) if path.exists() else None,
        "target_distribution": _target_distribution(rows, "should_reduce_exposure"),
        **predictor_audit,
        "rows_eligible_by_decision": _eligible_by_decision(rows),
        "minimum_history_skipped_decisions": _minimum_history_skips(rows, minimum_rows=3),
    })
    return audit


def audit_meta_inputs(
    *,
    expanded_dataset_path: Path,
    source_prediction_dirs: list[Path],
    fold_count: int = 3,
) -> dict[str, Any]:
    expanded_rows = _read_csv_if_exists(expanded_dataset_path)
    sources: dict[str, dict[str, dict[str, str]]] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    duplicate_source_predictions: dict[str, int] = {}
    for source_dir in source_prediction_dirs:
        prediction_path = source_dir / "prediction_artifacts.csv"
        metadata_path = source_dir / "prediction_artifacts.json"
        rows = _read_csv_if_exists(prediction_path)
        metadata = _read_json_if_exists(metadata_path)
        model = str(metadata.get("model_type") or (rows[0].get("model_type") if rows else source_dir.name))
        duplicate_source_predictions[model] = len(rows) - len({row.get("feature_id", "") for row in rows})
        source_metadata[model] = {
            "source_dir": str(source_dir),
            "prediction_artifact_exists": prediction_path.exists(),
            "metadata_exists": metadata_path.exists(),
            "dataset_hash": metadata.get("dataset_hash") or metadata.get("data_hash"),
            "validation_method": metadata.get("validation_method"),
            "temporal_policy": metadata.get("temporal_policy"),
            "model_input_contract_version": metadata.get("model_input_contract_version"),
            "row_count": len(rows),
        }
        sources[model] = {row["feature_id"]: row for row in rows if row.get("feature_id")}
    source_non_oof_feature_ids = {
        feature_id
        for predictions in sources.values()
        for feature_id, row in predictions.items()
        if row.get("split") != "out_of_fold"
    }
    meta_rows: list[dict[str, str]] = []
    dataset_audit: dict[str, Any] = {}
    if len(sources) >= 2 and expanded_rows:
        meta_rows, dataset_audit = build_meta_dataset_rows(expanded_rows, sources)
    walk_forward = (
        _walk_forward_meta_evaluation(
            meta_rows,
            model_type="logistic_regression",
            fold_count=fold_count,
            threshold=0.5,
            reduced_exposure=0.7,
            reduce_when="above_or_equal_threshold",
            random_seed=42,
            calibration_bin_count=5,
        )
        if meta_rows
        else {"fold_count": 0, "folds": [], "summary": {}, "leakage_checks_passed": False}
    )
    dataset_hashes = {
        str(value.get("dataset_hash"))
        for value in source_metadata.values()
        if value.get("dataset_hash")
    }
    temporal_identities = {
        json.dumps(
            {
                "validation_method": value.get("validation_method"),
                "temporal_policy": value.get("temporal_policy"),
                "model_input_contract_version": value.get("model_input_contract_version"),
            },
            sort_keys=True,
            default=str,
        )
        for value in source_metadata.values()
    }
    in_sample_training_rows = sum(
        1 for row in meta_rows if row.get("feature_id") in source_non_oof_feature_ids
    )
    return {
        "expanded_dataset_path": str(expanded_dataset_path),
        "expanded_dataset_exists": expanded_dataset_path.exists(),
        "source_model_count": len(sources),
        "source_models": sorted(sources),
        "source_metadata": source_metadata,
        "meta_row_count": len(meta_rows),
        "date_coverage": _date_coverage(meta_rows, "rebalance_date"),
        "missing_source_predictions": dataset_audit.get("missing_prediction_counts_by_model", {}),
        "duplicate_source_predictions": duplicate_source_predictions,
        "mixed_dataset_identities": len(dataset_hashes) > 1,
        "mixed_temporal_identities": len(temporal_identities) > 1,
        "missing_temporal_identities": any(
            not value.get("temporal_policy") for value in source_metadata.values()
        ),
        "in_sample_base_prediction_rows": in_sample_training_rows,
        "all_training_source_predictions_out_of_fold": in_sample_training_rows == 0,
        "walk_forward_fold_count": walk_forward.get("fold_count", 0),
        "minimum_history_skipped_folds": sum(
            1 for fold in walk_forward.get("folds", []) if not fold.get("train_sample_count")
        ),
        "immature_labels_excluded": sum(
            int(fold.get("purged_unavailable_label_count", 0))
            for fold in walk_forward.get("folds", [])
        ),
        "leakage_checks_passed": bool(walk_forward.get("leakage_checks_passed")),
        "walk_forward": walk_forward,
    }


def validate_temporal_artifacts(output_dir: Path, *, meta: bool = False) -> dict[str, Any]:
    names = (
        ("meta_ensemble_temporal_audit.json", "meta_ensemble_temporal_folds.csv")
        if meta
        else ("exposure_temporal_audit.json", "exposure_temporal_folds.csv")
    )
    audit_path = output_dir / names[0]
    folds_path = output_dir / names[1]
    audit = _read_json_if_exists(audit_path)
    folds = _read_csv_if_exists(folds_path)
    return {
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "folds_path": str(folds_path),
        "audit_exists": audit_path.exists(),
        "folds_exists": folds_path.exists(),
        "audit_parseable": bool(audit),
        "fold_count": len(folds),
        "leakage_checks_passed": audit.get("leakage_checks_passed"),
    }


def validate_immutable_output(output_dir: Path) -> dict[str, Any]:
    latest_path = output_dir / "latest_completed.json"
    latest = _read_json_if_exists(latest_path)
    run_dir = output_dir / "runs" / str(latest.get("run_id", ""))
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json_if_exists(manifest_path)
    return {
        "output_dir": str(output_dir),
        "latest_completed_exists": latest_path.exists(),
        "latest_run_id": latest.get("run_id"),
        "manifest_exists": manifest_path.exists(),
        "manifest_status": manifest.get("run_status"),
        "artifact_count": len(manifest.get("artifacts", [])) if isinstance(manifest.get("artifacts"), list) else 0,
        "champion_exists": (output_dir / "champion.json").exists(),
    }


def walk_forward_scope(
    selector_audit: dict[str, Any],
    exposure_audit: dict[str, Any],
    meta_audit: dict[str, Any],
    *,
    exposure_base_model_count: int,
    deep_epochs_per_fold: int,
) -> dict[str, Any]:
    selector_dates = int(selector_audit.get("date_coverage", {}).get("unique_dates") or 0)
    exposure_dates = int(exposure_audit.get("date_coverage", {}).get("unique_dates") or 0)
    meta_dates = int(meta_audit.get("date_coverage", {}).get("unique_dates") or 0)
    exposure_fits = exposure_dates * exposure_base_model_count
    total_fits = selector_dates + exposure_fits + meta_dates
    return {
        "selector_decision_dates": selector_dates,
        "selector_retraining_folds": selector_dates,
        "exposure_decision_dates": exposure_dates,
        "exposure_retraining_folds_per_model": exposure_dates,
        "exposure_base_model_count": exposure_base_model_count,
        "meta_decision_dates": meta_dates,
        "meta_retraining_folds": meta_dates,
        "deep_model_epochs_per_fold": deep_epochs_per_fold,
        "expected_total_model_fits": total_fits,
        "expected_checkpoint_directories": exposure_base_model_count,
        "minimum_history_skipped_folds": {
            "exposure": len(exposure_audit.get("minimum_history_skipped_decisions", [])),
            "meta": meta_audit.get("minimum_history_skipped_folds", 0),
        },
        "workload_classification": (
            "small"
            if total_fits < 100
            else "moderate"
            if total_fits < 1000
            else "large"
            if total_fits < 5000
            else "impractical on current hardware"
        ),
        "implementation_retrains_every_decision": True,
    }


def build_readiness_report(
    *,
    output_dir: Path,
    full_suite_status: dict[str, Any],
    smoke_status: dict[str, Any],
    paths: ResolvedReadinessPaths,
) -> dict[str, Any]:
    selector = audit_selector_csv(paths.selector_input_path)
    exposure = audit_exposure_csv(paths.exposure_dataset_path)
    meta = audit_meta_inputs(
        expanded_dataset_path=paths.meta_expanded_dataset_path,
        source_prediction_dirs=paths.meta_source_prediction_dirs,
    )
    artifact_validation = {
        "exposure": validate_temporal_artifacts(paths.exposure_output_dir),
        "meta": validate_temporal_artifacts(paths.meta_output_dir, meta=True),
        "selector_temporal_audit": {
            "path": str(paths.selector_output_dir / "stock_level_temporal_audit.json"),
            "exists": (paths.selector_output_dir / "stock_level_temporal_audit.json").exists(),
            "parseable": bool(_read_json_if_exists(paths.selector_output_dir / "stock_level_temporal_audit.json")),
        },
    }
    immutable = {
        "selector": validate_immutable_output(paths.selector_output_dir),
        "exposure": validate_immutable_output(paths.exposure_output_dir),
        "meta": validate_immutable_output(paths.meta_output_dir),
    }
    scope = walk_forward_scope(
        selector,
        exposure,
        meta,
        exposure_base_model_count=max(1, len(paths.meta_source_prediction_dirs)),
        deep_epochs_per_fold=3,
    )
    blockers = _remaining_blockers(
        full_suite_status=full_suite_status,
        selector=selector,
        exposure=exposure,
        meta=meta,
        artifact_validation=artifact_validation,
        smoke_status=smoke_status,
    )
    decision = READINESS_READY if not blockers else READINESS_NOT_READY
    report = {
        "final_readiness_decision": decision,
        "full_suite_status": full_suite_status,
        "resolved_paths": _paths_payload(paths),
        "selector_temporal_status": selector,
        "exposure_temporal_status": exposure,
        "meta_temporal_status": meta,
        "same_day_timing_convention": same_day_timing_convention(),
        "walk_forward_computational_scope": scope,
        "smoke_run_status": smoke_status,
        "artifact_validation_status": artifact_validation,
        "immutable_run_validation": immutable,
        "remaining_blockers": blockers,
    }
    write_readiness_report(output_dir, report)
    return report


def same_day_timing_convention() -> dict[str, Any]:
    return {
        "repository_owned_convention": "date-only decisions are treated as not proven later than same-date outcome close",
        "feature_availability": "features dated T are available for the decision timestamp T only when generated from information before that decision",
        "portfolio_decision": "research rebalance decisions are date-keyed; intraday ordering is not encoded in date-only artifacts",
        "outcome_close_publication": "an outcome ending at close T is safe only for a later decision unless an explicit later timestamp proves ordering",
        "same_day_label_eligible": False,
    }


def write_readiness_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (output_dir / "temporal_readiness_report.md").write_text(
        _readiness_markdown(report),
        encoding="utf-8",
    )


def _remaining_blockers(
    *,
    full_suite_status: dict[str, Any],
    selector: dict[str, Any],
    exposure: dict[str, Any],
    meta: dict[str, Any],
    artifact_validation: dict[str, Any],
    smoke_status: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not full_suite_status.get("completed"):
        blockers.append("full pytest suite did not complete")
    elif not full_suite_status.get("passed"):
        blockers.append("full pytest suite completed with failures")
    if not selector.get("all_availability_safe"):
        blockers.append("selector label availability is incomplete or unsafe")
    if not selector.get("target_provenance_contract_complete"):
        blockers.append("selector target provenance contract is incomplete or missing")
    if not exposure.get("all_availability_safe"):
        blockers.append("exposure label availability is incomplete or unsafe")
    if meta.get("missing_temporal_identities"):
        blockers.append("meta source artifacts are missing temporal-policy identities")
    if meta.get("mixed_temporal_identities"):
        blockers.append("meta source artifacts have mixed temporal identities")
    if not meta.get("all_training_source_predictions_out_of_fold"):
        blockers.append("meta training includes non-out-of-fold source predictions")
    if not artifact_validation.get("exposure", {}).get("audit_exists"):
        blockers.append("exposure temporal audit artifact is missing")
    if not artifact_validation.get("meta", {}).get("audit_exists"):
        blockers.append("meta temporal audit artifact is missing")
    if not smoke_status.get("completed"):
        blockers.append("development smoke workflow did not complete")
    return blockers


def _temporal_row_audit(
    rows: list[dict[str, str]],
    *,
    key_columns: tuple[str, ...],
    target_column: str,
    fail_closed_same_date: bool,
) -> dict[str, Any]:
    explicit = 0
    derived = 0
    missing = 0
    availability_before_feature = 0
    availability_equal_decision = 0
    availability_after_decision = 0
    rejected = 0
    duplicate_keys = _duplicate_keys(rows, key_columns)
    for row in rows:
        row_rejected = False
        feature = _first(row, "feature_timestamp", "feature_date", "rebalance_date")
        decision = _first(row, "decision_timestamp", "rebalance_date", "feature_date")
        explicit_value = str(row.get("label_available_timestamp") or "").strip()
        label_end = _first(
            row,
            "label_end_timestamp",
            "label_end_date",
            "outcome_end_date",
        )
        if explicit_value:
            explicit += 1
            available = explicit_value
        elif label_end:
            derived += 1
            available = label_end
        else:
            missing += 1
            row_rejected = True
            continue
        if feature and available < feature:
            availability_before_feature += 1
            row_rejected = True
        if decision and available == decision:
            availability_equal_decision += 1
            if fail_closed_same_date:
                row_rejected = True
        elif decision and available > decision:
            availability_after_decision += 1
        elif decision and available < decision and fail_closed_same_date:
            row_rejected = True
        if row_rejected:
            rejected += 1
    return {
        "total_rows": len(rows),
        "date_coverage": _date_coverage(rows, "rebalance_date"),
        "rows_with_explicit_label_available_timestamp": explicit,
        "rows_deriving_availability_from_true_outcome_end_timestamp": derived,
        "rows_missing_label_availability": missing,
        "rows_rejected_by_fail_closed_validation": rejected,
        "rows_where_label_availability_before_feature_time": availability_before_feature,
        "rows_where_label_availability_equals_decision_time": availability_equal_decision,
        "rows_where_label_availability_after_decision_time": availability_after_decision,
        "duplicate_decision_key_count": len(duplicate_keys),
        "duplicate_decision_keys_sample": duplicate_keys[:20],
        "target_column": target_column,
        "target_non_null_count": sum(1 for row in rows if str(row.get(target_column, "")).strip()),
        "all_availability_safe": bool(rows)
        and missing == 0
        and rejected == 0
        and availability_equal_decision == 0,
    }


def _selector_target_provenance_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    complete = 0
    missing = 0
    invalid_order = 0
    missing_benchmark = 0
    versions: dict[str, int] = {}
    horizons: dict[str, int] = {}
    for row in rows:
        version = str(row.get("target_provenance_contract_version") or "").strip()
        if version:
            versions[version] = versions.get(version, 0) + 1
        horizon = str(row.get("target_horizon") or "").strip()
        if horizon:
            horizons[horizon] = horizons.get(horizon, 0) + 1
        has_required = all(
            str(row.get(column) or "").strip()
            for column in TARGET_PROVENANCE_COLUMNS
        )
        if has_required:
            complete += 1
        else:
            missing += 1
        target_start = str(row.get("target_start_timestamp") or "").strip()
        label_start = str(row.get("label_start_timestamp") or "").strip()
        label_end = str(row.get("label_end_timestamp") or "").strip()
        label_available = str(row.get("label_available_timestamp") or "").strip()
        if (
            target_start
            and label_start
            and label_end
            and label_available
            and not (target_start < label_start <= label_end < label_available)
        ):
            invalid_order += 1
        if (
            row.get("actual_benchmark_return_10d")
            not in (None, "")
            and not str(row.get("benchmark_label_end_timestamp") or "").strip()
        ):
            missing_benchmark += 1
    return {
        "target_provenance_contract_version_expected": TARGET_PROVENANCE_CONTRACT_VERSION,
        "target_provenance_contract_versions": versions,
        "target_horizon_distribution": horizons,
        "rows_with_complete_target_provenance": complete,
        "rows_missing_target_provenance": missing,
        "rows_with_invalid_target_provenance_order": invalid_order,
        "rows_with_benchmark_return_missing_benchmark_provenance": missing_benchmark,
        "target_provenance_contract_complete": bool(rows)
        and missing == 0
        and invalid_order == 0
        and missing_benchmark == 0
        and set(versions) == {TARGET_PROVENANCE_CONTRACT_VERSION},
    }


def _exposure_predictor_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "predictor_count": 0,
            "target_derived_columns_present_in_predictors": [],
            "target_derived_columns_present_in_source": [],
        }
    source_columns = set(rows[0])
    target_derived_source = forbidden_predictor_columns(source_columns)
    try:
        dataset = build_dataset(rows, rows, "should_reduce_exposure")
        leakage = dataset_leakage_audit(dataset)
        predictor_columns = {name for features in dataset.features for name in features}
    except (KeyError, ValueError, TypeError) as exc:
        return {
            "predictor_count": 0,
            "target_derived_columns_present_in_predictors": ["audit_failed"],
            "predictor_audit_error": str(exc),
            "target_derived_columns_present_in_source": target_derived_source,
        }
    return {
        "predictor_count": len(predictor_columns),
        "target_derived_columns_present_in_predictors": leakage["forbidden_predictor_columns"],
        "target_derived_columns_present_in_source": target_derived_source,
    }


def _eligible_by_decision(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    decisions = sorted({_first(row, "decision_timestamp", "rebalance_date", "feature_date") for row in rows})
    result: dict[str, dict[str, int]] = {}
    for decision in decisions:
        if not decision:
            continue
        eligible = 0
        excluded = 0
        for row in rows:
            row_decision = _first(row, "decision_timestamp", "rebalance_date", "feature_date")
            available = str(row.get("label_available_timestamp") or row.get("label_end_date") or row.get("outcome_end_date") or "")
            if row_decision < decision and available <= decision:
                eligible += 1
            elif row_decision < decision and available > decision:
                excluded += 1
        result[decision] = {
            "eligible_training_rows": eligible,
            "excluded_unmatured_rows": excluded,
        }
    return result


def _minimum_history_skips(rows: list[dict[str, str]], *, minimum_rows: int) -> list[str]:
    return [
        decision
        for decision, counts in _eligible_by_decision(rows).items()
        if counts["eligible_training_rows"] < minimum_rows
    ]


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    return ""


def _duplicate_keys(rows: Iterable[dict[str, str]], columns: tuple[str, ...]) -> list[str]:
    seen: set[tuple[str, ...]] = set()
    duplicates: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in columns)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return ["|".join(key) for key in sorted(duplicates)]


def _date_coverage(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    dates = sorted({str(row.get(column, "")) for row in rows if row.get(column)})
    return {
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "unique_dates": len(dates),
    }


def _target_distribution(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _paths_payload(paths: ResolvedReadinessPaths) -> dict[str, Any]:
    return {
        "selector_config_path": str(paths.selector_config_path),
        "exposure_config_path": str(paths.exposure_config_path),
        "meta_config_path": str(paths.meta_config_path),
        "selector_input_path": str(paths.selector_input_path),
        "selector_output_dir": str(paths.selector_output_dir),
        "exposure_dataset_path": str(paths.exposure_dataset_path),
        "exposure_output_dir": str(paths.exposure_output_dir),
        "meta_expanded_dataset_path": str(paths.meta_expanded_dataset_path),
        "meta_dataset_path": str(paths.meta_dataset_path),
        "meta_output_dir": str(paths.meta_output_dir),
        "meta_source_prediction_dirs": [
            str(path) for path in paths.meta_source_prediction_dirs
        ],
    }


def _readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Temporal Readiness Report",
        "",
        f"- Final decision: `{report['final_readiness_decision']}`",
        f"- Full suite completed: `{report['full_suite_status'].get('completed')}`",
        f"- Full suite passed: `{report['full_suite_status'].get('passed')}`",
        f"- Selector rows: `{report['selector_temporal_status'].get('total_rows')}`",
        f"- Exposure rows: `{report['exposure_temporal_status'].get('total_rows')}`",
        f"- Meta rows: `{report['meta_temporal_status'].get('meta_row_count')}`",
        "",
        "## Remaining Blockers",
    ]
    blockers = report.get("remaining_blockers", [])
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend([
        "",
        "## Same-Day Timing",
        report["same_day_timing_convention"]["repository_owned_convention"],
        "",
        "## Walk-Forward Scope",
        f"- Expected total model fits: `{report['walk_forward_computational_scope'].get('expected_total_model_fits')}`",
        f"- Classification: `{report['walk_forward_computational_scope'].get('workload_classification')}`",
    ])
    return "\n".join(lines) + "\n"
