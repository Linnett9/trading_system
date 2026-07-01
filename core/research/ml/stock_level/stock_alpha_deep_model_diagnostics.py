from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_alpha_news_contract import validate_news_contract
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import _factories_for_model_set
from core.research.ml.stock_level.stock_alpha_model_sets import resolve_stock_alpha_model_set
from core.research.ml.stock_level_benchmark_data import _available_feature_columns, _prepare_rows
from core.research.ml.stock_level_benchmark_execution import _build_sequences, _walk_forward_partitions
from core.research.ml.stock_level_benchmark_models import _sequence_feature_columns
from core.research.ml.stock_level_benchmark_types import (
    AUXILIARY_TARGET_COLUMNS,
    MODEL_NAMES,
    PREDICTION_PREFIX,
    SEQUENCE_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
)
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps

GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}
REQUIRED_FIELDS = (
    "model_name",
    "model_set",
    "run_size",
    "target_column",
    "fold_id",
    "train_rows",
    "test_rows",
    "train_dates",
    "test_dates",
    "train_symbols",
    "test_symbols",
    "feature_column_count",
    "feature_non_null_ratio",
    "x_train_finite_ratio",
    "x_train_non_finite_count",
    "x_test_finite_ratio",
    "x_test_non_finite_count",
    "target_non_null_count",
    "y_train_finite_count",
    "y_train_non_finite_count",
    "y_train_mean",
    "y_train_std",
    "y_train_min",
    "y_train_max",
    "sequence_length",
    "sequence_windows_created",
    "sequence_train_windows",
    "sequence_test_windows",
    "auxiliary_target_columns",
    "auxiliary_target_count",
    "auxiliary_y_train_shape",
    "auxiliary_y_test_shape",
    "auxiliary_target_non_null_counts",
    "auxiliary_target_finite_counts",
    "auxiliary_alignment_status",
    "feature_imputation_strategy",
    "feature_nan_count_before_imputation",
    "feature_nan_count_after_imputation",
    "test_feature_nan_count_before_imputation",
    "test_feature_nan_count_after_imputation",
    "x_train_finite_ratio_after_preprocessing",
    "x_train_non_finite_count_after_preprocessing",
    "x_test_finite_ratio_after_preprocessing",
    "x_test_non_finite_count_after_preprocessing",
    "train_loss_by_epoch",
    "train_loss_finite",
    "model_parameters_finite",
    "raw_prediction_count",
    "raw_finite_prediction_count",
    "postprocessed_prediction_count",
    "postprocessed_finite_prediction_count",
    "prediction_count",
    "finite_prediction_count",
    "nan_prediction_count",
    "prediction_date_count",
    "prediction_symbol_count",
    "merge_key_columns",
    "merge_matched_rows",
    "merge_unmatched_rows",
    "output_prediction_column",
    "skip_reason",
    "error_message",
)


@dataclass(frozen=True)
class DeepModelDiagnosticPaths:
    json_path: Path
    markdown_path: Path
    csv_path: Path


def diagnose_sequence_fold(*, model_name: str, model_set: str, run_size: str, target_column: str, fold_id: int, all_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], feature_columns: tuple[str, ...], sequence_length: int, predictor: Callable[[list[list[list[float]]], list[list[list[float]]], list[dict[str, Any]]], list[Any]] | None, unavailable_reason: str | None = None, auxiliary_target_columns: tuple[str, ...] = ()) -> dict[str, Any]:
    train_sequences, sequence_train_rows = _build_sequences(all_rows, train_rows, feature_columns, sequence_length)
    test_sequences, prediction_rows = _build_sequences(all_rows, test_rows, feature_columns, sequence_length)
    fold_rows = [*train_rows, *test_rows]
    target_values = [float(r[target_column]) for r in train_rows if _finite(r.get(target_column))]
    auxiliary_counts = _auxiliary_counts(sequence_train_rows, auxiliary_target_columns)
    row = {"model_name": model_name, "model_set": model_set, "run_size": run_size, "target_column": target_column, "fold_id": fold_id, "train_rows": len(train_rows), "test_rows": len(test_rows), "train_dates": len({r["rebalance_date"] for r in train_rows}), "test_dates": len({r["rebalance_date"] for r in test_rows}), "train_symbols": len({r["symbol"] for r in train_rows}), "test_symbols": len({r["symbol"] for r in test_rows}), "feature_column_count": len(feature_columns), "feature_non_null_ratio": _feature_ratio(fold_rows, feature_columns), "target_non_null_count": len(target_values), "sequence_length": sequence_length, "sequence_windows_created": len(train_sequences) + len(test_sequences), "sequence_train_windows": len(train_sequences), "sequence_test_windows": len(test_sequences), "auxiliary_target_columns": list(auxiliary_target_columns), "auxiliary_target_count": len(auxiliary_target_columns), "auxiliary_y_train_shape": [len(sequence_train_rows), len(auxiliary_target_columns)] if auxiliary_target_columns else [0, 0], "auxiliary_y_test_shape": [len(prediction_rows), len(auxiliary_target_columns)] if auxiliary_target_columns else [0, 0], "auxiliary_target_non_null_counts": auxiliary_counts["non_null"], "auxiliary_target_finite_counts": auxiliary_counts["finite"], "auxiliary_alignment_status": "aligned" if not auxiliary_target_columns or len(sequence_train_rows) == len(train_sequences) else "misaligned", "x_train_finite_ratio": _sequence_finite_ratio(train_sequences), "x_train_non_finite_count": _sequence_non_finite_count(train_sequences), "x_test_finite_ratio": _sequence_finite_ratio(test_sequences), "x_test_non_finite_count": _sequence_non_finite_count(test_sequences), "y_train_finite_count": len(target_values), "y_train_non_finite_count": len(train_rows) - len(target_values), "y_train_mean": _average(target_values), "y_train_std": _std(target_values), "y_train_min": min(target_values) if target_values else None, "y_train_max": max(target_values) if target_values else None, "feature_imputation_strategy": None, "feature_nan_count_before_imputation": _sequence_non_finite_count(train_sequences), "feature_nan_count_after_imputation": None, "test_feature_nan_count_before_imputation": _sequence_non_finite_count(test_sequences), "test_feature_nan_count_after_imputation": None, "x_train_finite_ratio_after_preprocessing": None, "x_train_non_finite_count_after_preprocessing": None, "x_test_finite_ratio_after_preprocessing": None, "x_test_non_finite_count_after_preprocessing": None, "train_loss_by_epoch": [], "train_loss_finite": None, "model_parameters_finite": None, "raw_prediction_count": 0, "raw_finite_prediction_count": 0, "postprocessed_prediction_count": 0, "postprocessed_finite_prediction_count": 0, "prediction_count": 0, "finite_prediction_count": 0, "nan_prediction_count": 0, "prediction_date_count": 0, "prediction_symbol_count": 0, "merge_key_columns": ["rebalance_date", "symbol"], "merge_matched_rows": 0, "merge_unmatched_rows": len(test_rows), "output_prediction_column": f"{PREDICTION_PREFIX}{model_name}", "skip_reason": None, "error_message": None}
    if unavailable_reason:
        row["skip_reason"] = "missing_auxiliary_targets" if unavailable_reason.startswith("missing_auxiliary_targets") else "model_unavailable_by_design"
        row["error_message"] = unavailable_reason
        return row
    if not row["target_non_null_count"]: row["skip_reason"] = "target_missing"; return row
    if row["feature_non_null_ratio"] < 0.05: row["skip_reason"] = "feature_matrix_mostly_nan"; return row
    if not train_sequences: row["skip_reason"] = "no_sequence_windows_created"; return row
    if not test_sequences: row["skip_reason"] = "test_sequence_windows_missing"; return row
    try:
        predicted = predictor(train_sequences, test_sequences, sequence_train_rows) if predictor else []
        if isinstance(predicted, tuple) and len(predicted) == 3:
            values, override_keys, model_diagnostics = predicted
            row.update(dict(model_diagnostics or {}))
        elif isinstance(predicted, tuple):
            values, override_keys = predicted
        else:
            values, override_keys = predicted, None
        values = list(values)
    except Exception as exc: row["skip_reason"] = "model_error"; row["error_message"] = f"{type(exc).__name__}: {exc}"; return row
    finite = [value for value in values if _finite(value)]; keys = set(override_keys) if override_keys is not None else {(r["rebalance_date"], r["symbol"]) for r in prediction_rows[:len(values)]}; test_keys = {(r["rebalance_date"], r["symbol"]) for r in test_rows}; matched = keys & test_keys
    row.update({"prediction_count": len(values), "finite_prediction_count": len(finite), "nan_prediction_count": len(values) - len(finite), "prediction_date_count": len({k[0] for k in keys}), "prediction_symbol_count": len({k[1] for k in keys}), "merge_matched_rows": len(matched), "merge_unmatched_rows": len(test_keys - matched)})
    if row.get("x_train_non_finite_count_after_preprocessing") or row.get("x_test_non_finite_count_after_preprocessing"):
        row["skip_reason"] = "non_finite_sequence_inputs"
    elif row.get("train_loss_finite") is False:
        row["skip_reason"] = "loss_nan"
    elif row.get("model_parameters_finite") is False:
        row["skip_reason"] = "model_parameters_nan"
    elif len(values) != len(prediction_rows): row["skip_reason"] = "prediction_count_mismatch"; row["error_message"] = f"model returned {len(values)} predictions for {len(prediction_rows)} sequence test windows"
    elif values and not finite: row["skip_reason"] = "predictions_all_nan_after_model"
    elif not matched: row["skip_reason"] = "prediction_merge_failed"
    elif len(matched) < len(test_keys): row["skip_reason"] = "prediction_merge_partial"
    return row


def write_stock_alpha_deep_model_diagnostics(config: Mapping[str, Any]) -> DeepModelDiagnosticPaths:
    config = {**config, "ml": {**dict(config.get("ml", {}) or {}), "stock_alpha_run_size": "dev"}}
    ml = config["ml"]
    target = str(ml.get("stock_deep_diagnostic_target", TARGET_COLUMN))
    models = _requested_models(ml)
    for model_name in models:
        if model_name not in SEQUENCE_MODEL_NAMES:
            raise ValueError(f"stock_deep_diagnostic_model must be one of: {', '.join(SEQUENCE_MODEL_NAMES)}")
    settings = StockLevelResearchConfig.from_mapping(config); output = stock_alpha_output_dir(config); caps = apply_stock_alpha_worker_caps(dict(config))
    rows = CsvRowRepository().read(settings.artifact_path); features = _available_feature_columns(rows, include_engineered=settings.include_engineered_features)
    if target != TARGET_COLUMN: rows = [dict(row, **{TARGET_COLUMN: row.get(target, "")}) for row in rows]
    prepared, _ = _prepare_rows(rows, features); dates = sorted({r["rebalance_date"] for r in prepared}); model_set = resolve_stock_alpha_model_set("full")
    news_contract = validate_news_contract(config, rows)
    _, factories = _factories_for_model_set(settings, model_set, sklearn_n_jobs=1, torch_num_threads=caps["torch_num_threads"]); news = tuple(c for c in (rows[0] if rows else {}) if c.startswith("news_") or "sentiment" in c.lower())
    paths: DeepModelDiagnosticPaths | None = None
    for model_name in models:
        paths = _write_one_model_diagnostics(
            model_name=model_name,
            target=target,
            settings=settings,
            output=output,
            rows=rows,
            prepared=prepared,
            dates=dates,
            features=features,
            factories=factories,
            news=news,
            news_contract_available=news_contract.available,
        )
    _write_index(output)
    if paths is None:
        raise ValueError("No stock-alpha deep diagnostic models requested")
    return paths


def _write_one_model_diagnostics(
    *,
    model_name: str,
    target: str,
    settings: StockLevelResearchConfig,
    output: Path,
    rows: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
    dates: list[str],
    features: tuple[str, ...],
    factories: Mapping[str, Callable[[], Any]],
    news: tuple[str, ...],
    news_contract_available: bool,
) -> DeepModelDiagnosticPaths:
    model_output = output / "deep_diagnostics" / model_name
    unavailable = "missing point-in-time symbol-level news/sentiment features" if model_name == "news_analysis_transformer" and not news else None
    if model_name == "news_analysis_transformer" and news and not news_contract_available:
        unavailable = "news_analysis_transformer unavailable: missing valid point-in-time news contract"
    auxiliary_columns = AUXILIARY_TARGET_COLUMNS if model_name == "multitask_transformer" else ()
    missing_auxiliary = _missing_auxiliary_targets(rows, auxiliary_columns)
    if missing_auxiliary:
        available = sorted({column for row in rows for column in row if column.startswith("actual_")})
        unavailable = (
            "missing_auxiliary_targets: required="
            f"{list(auxiliary_columns)} available={available}"
        )
    feature_columns = _sequence_feature_columns(model_name, news, features); factory = factories.get(model_name)
    diagnostics = []
    for fold_id, train, test, _, _, _ in _walk_forward_partitions(prepared, dates, first_test_index=settings.min_train_dates + settings.embargo_dates, test_window_dates=settings.test_window_dates, embargo_dates=settings.embargo_dates):
        def predict(train_x, test_x, train_target_rows, factory=factory):
            auxiliary = [[r[column] for column in auxiliary_columns] for r in train_target_rows] if auxiliary_columns else None
            model = factory(); model.fit(train_x, [r[TARGET_COLUMN] for r in train_target_rows], auxiliary); values = model.predict(test_x); return values, None, getattr(model, "diagnostics", {})
        diagnostics.append(diagnose_sequence_fold(model_name=model_name, model_set="diagnostic_single_model", run_size="dev", target_column=target, fold_id=fold_id, all_rows=prepared, train_rows=train, test_rows=test, feature_columns=feature_columns, sequence_length=settings.sequence_length, predictor=predict if factory and not unavailable else None, unavailable_reason=unavailable or (None if factory else "Model factory unavailable"), auxiliary_target_columns=auxiliary_columns))
    payload = {"mode": "stock_alpha_deep_model_diagnostics", **GUARDRAILS, "model_name": model_name, "model_set": "diagnostic_single_model", "run_size": "dev", "target_column": target, "source_artifact_path": str(settings.artifact_path), "feature_column_count": len(feature_columns), "fold_count": len(diagnostics), "diagnostics": diagnostics}
    model_output.mkdir(parents=True, exist_ok=True); paths = DeepModelDiagnosticPaths(model_output / "stock_alpha_deep_model_diagnostics.json", model_output / "stock_alpha_deep_model_diagnostics.md", model_output / "stock_alpha_deep_model_diagnostics.csv")
    paths.json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"); paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    with paths.csv_path.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS); writer.writeheader(); writer.writerows(diagnostics)
    return paths


def _finite(value: Any) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


def _requested_models(ml: Mapping[str, Any]) -> list[str]:
    raw_models = ml.get("stock_deep_diagnostic_models")
    if raw_models is None:
        return [str(ml.get("stock_deep_diagnostic_model", "dlinear"))]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("ml.stock_deep_diagnostic_models must be a non-empty list")
    return [str(model) for model in raw_models]


def _feature_ratio(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> float:
    total = len(rows) * len(columns); return sum(_finite(r.get(c)) for r in rows for c in columns) / total if total else 0.0


def _sequence_finite_ratio(sequences: list[list[list[float]]]) -> float:
    total = sum(len(step) for sequence in sequences for step in sequence)
    return (total - _sequence_non_finite_count(sequences)) / total if total else 1.0


def _sequence_non_finite_count(sequences: list[list[list[float]]]) -> int:
    return sum(
        1
        for sequence in sequences
        for step in sequence
        for value in step
        if not _finite(value)
    )


def _auxiliary_counts(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        "non_null": {column: sum(row.get(column) is not None for row in rows) for column in columns},
        "finite": {column: sum(_finite(row.get(column)) for row in rows) for column in columns},
    }


def _missing_auxiliary_targets(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[str]:
    if not columns:
        return []
    return [
        column
        for column in columns
        if not any(column in row and _finite(row.get(column)) for row in rows)
    ]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# Stock Alpha Deep-Model Diagnostics", "", "Research/debug only. Trading impact: none. Production validated: false.", "", f"- Model: `{payload['model_name']}`", f"- Run size: `{payload['run_size']}`", "", "| Fold | Train windows | Test windows | Predictions | Finite | Matched | Skip reason |", "|---:|---:|---:|---:|---:|---:|---|"]
    for row in payload["diagnostics"]: lines.append(f"| {row['fold_id']} | {row['sequence_train_windows']} | {row['sequence_test_windows']} | {row['prediction_count']} | {row['finite_prediction_count']} | {row['merge_matched_rows']} | {row['skip_reason'] or ''} |")
    return "\n".join(lines) + "\n"


def _write_index(output_dir: Path) -> None:
    root = output_dir / "deep_diagnostics"
    entries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/stock_alpha_deep_model_diagnostics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        diagnostics = list(payload.get("diagnostics", []) or [])
        summary = _model_index_summary(diagnostics)
        entries.append(
            {
                "model_name": str(payload.get("model_name", path.parent.name)),
                **summary,
                "fold_count": int(payload.get("fold_count", len(diagnostics))),
                "json_path": str(path),
                "markdown_path": str(path.with_suffix(".md")),
                "csv_path": str(path.with_suffix(".csv")),
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            }
        )
    payload = {
        "mode": "stock_alpha_deep_model_diagnostics_index",
        "model_count": len(entries),
        "models": entries,
        "model_coverage": _model_coverage(entries),
        **GUARDRAILS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stock_alpha_deep_model_diagnostics_index.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "stock_alpha_deep_model_diagnostics_index.md").write_text(
        _index_markdown(payload),
        encoding="utf-8",
    )


def _index_markdown(payload: Mapping[str, Any]) -> str:
    coverage = dict(payload.get("model_coverage", {}) or {})
    lines = [
        "# Stock Alpha Deep-Model Diagnostics Index",
        "",
        "Research/debug only. Trading impact: none. Production validated: false.",
        "",
        "## Model Coverage",
        "",
        f"- Total stock-alpha models: {coverage.get('total_model_count')}",
        f"- Tested models: {coverage.get('tested_model_count')}",
        f"- Passed models: {coverage.get('passed_model_count')}",
        f"- Failed models: {coverage.get('failed_model_count')}",
        f"- Unavailable models: {coverage.get('unavailable_model_count')}",
        f"- Skipped missing required inputs: {coverage.get('skipped_missing_required_inputs_count')}",
        f"- Untested models: {coverage.get('untested_model_count')}",
        "",
        "| Model | Status | Folds | Finite predictions | Windows | Merge | Loss finite | Params finite | Report | Skip reasons |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in payload.get("models", []):
        reasons = ", ".join(row.get("skip_reasons", []))
        lines.append(
            f"| {row['model_name']} | {row['status']} | {row['fold_count']} | "
            f"{row['finite_prediction_count']}/{row['prediction_count']} | "
            f"{row['sequence_windows_created']} | {row['merge_back_matched']} | "
            f"{row['loss_finite']} | {row['model_parameters_finite']} | "
            f"{row['markdown_path']} | {reasons} |"
        )
    return "\n".join(lines) + "\n"


def _model_index_summary(diagnostics: list[Mapping[str, Any]]) -> dict[str, Any]:
    prediction_count = sum(int(row.get("prediction_count") or 0) for row in diagnostics)
    finite_prediction_count = sum(int(row.get("finite_prediction_count") or 0) for row in diagnostics)
    skip_reasons = sorted({str(row.get("skip_reason")) for row in diagnostics if row.get("skip_reason")})
    sequence_windows_created = bool(diagnostics) and all(
        int(row.get("sequence_train_windows") or 0) > 0 and int(row.get("sequence_test_windows") or 0) > 0
        for row in diagnostics
    )
    merge_back_matched = bool(diagnostics) and all(
        int(row.get("merge_matched_rows") or 0) == int(row.get("prediction_count") or 0)
        for row in diagnostics
    )
    loss_values = [row.get("train_loss_finite") for row in diagnostics if row.get("train_loss_finite") is not None]
    parameter_values = [row.get("model_parameters_finite") for row in diagnostics if row.get("model_parameters_finite") is not None]
    loss_finite = all(bool(value) for value in loss_values) if loss_values else None
    parameters_finite = all(bool(value) for value in parameter_values) if parameter_values else None
    if not diagnostics:
        status = "skipped"
    elif any(reason == "model_unavailable_by_design" for reason in skip_reasons):
        status = "unavailable"
    elif any(reason == "missing_auxiliary_targets" for reason in skip_reasons):
        status = "skipped_missing_required_inputs"
    elif skip_reasons or prediction_count == 0 or finite_prediction_count != prediction_count or not sequence_windows_created or not merge_back_matched or loss_finite is False or parameters_finite is False:
        status = "failed"
    else:
        status = "passed"
    return {
        "status": status,
        "prediction_count": prediction_count,
        "finite_prediction_count": finite_prediction_count,
        "finite_prediction_ratio": finite_prediction_count / prediction_count if prediction_count else 0.0,
        "skip_reasons": skip_reasons,
        "sequence_windows_created": sequence_windows_created,
        "merge_back_matched": merge_back_matched,
        "loss_finite": loss_finite,
        "model_parameters_finite": parameters_finite,
    }


def _model_coverage(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    tested = {str(row["model_name"]) for row in entries}
    statuses = [str(row.get("status")) for row in entries]
    deep_models = [model for model in SEQUENCE_MODEL_NAMES if model != "news_analysis_transformer"]
    return {
        "total_model_count": len(MODEL_NAMES),
        "tabular_model_count": len(TABULAR_MODEL_NAMES),
        "sequence_model_count": len(SEQUENCE_MODEL_NAMES),
        "deep_diagnostic_eligible_count": len(deep_models),
        "tested_model_count": len(tested),
        "passed_model_count": statuses.count("passed"),
        "failed_model_count": statuses.count("failed"),
        "unavailable_model_count": statuses.count("unavailable"),
        "skipped_missing_required_inputs_count": statuses.count("skipped_missing_required_inputs"),
        "untested_model_count": len([model for model in MODEL_NAMES if model not in tested]),
        "all_models": list(MODEL_NAMES),
        "tabular_models": list(TABULAR_MODEL_NAMES),
        "sequence_models": list(SEQUENCE_MODEL_NAMES),
        "deep_diagnostic_eligible_models": deep_models,
        "conditional_models": {
            "news_analysis_transformer": "requires real point-in-time symbol-level news/sentiment features",
        },
        "tested_models": sorted(tested),
        "untested_models": [model for model in MODEL_NAMES if model not in tested],
    }
