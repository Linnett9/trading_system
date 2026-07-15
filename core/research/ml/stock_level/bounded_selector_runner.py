from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from core.research.ml.stock_level.selector_dataset import BASELINE_CANDIDATES, DETERMINISTIC_SIGNAL_COLUMNS
from core.research.ml.stock_level.stock_level_alpha_features import ENGINEERED_FEATURE_COLUMNS
from core.research.ml.stock_level.selector_feature_schema import load_feature_schema
from core.research.ml.stock_level_benchmark_models import _build_tabular_model
from core.research.ml.stock_level_benchmark_types import TARGET_OUTPUT_COLUMNS

CONTRACT_VERSION = "bounded_daily_selector_v2"
PREDICTION_QUALITY_CONTRACT_VERSION = "fitted_candidate_prediction_quality_v1"
MIN_PREDICTION_STANDARD_DEVIATION = 1e-12
MIN_PREDICTION_RANGE = 1e-12
PREDICTION_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
SUPPORTED_MODELS = ("ridge", "elastic_net", "random_forest", "gradient_boosting")
SUPPORTED_BASELINES = tuple(BASELINE_CANDIDATES)
OUTCOME_PREFIX = "actual_"


class PredictionQualityError(RuntimeError):
    def __init__(self, reasons: list[str], metrics: Mapping[str, Any]):
        self.reasons = reasons
        self.metrics = dict(metrics)
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class BoundedSelectorSettings:
    dataset_root: Path
    output_root: Path
    oos_start_date: str | None
    oos_end_date: str | None
    max_oos_dates: int | None
    model_allowlist: tuple[str, ...]
    baseline_allowlist: tuple[str, ...]
    include_engineered_features: bool
    feature_schema_path: Path | None
    resume: bool
    overwrite_incomplete_dates: bool
    random_seed: int
    sklearn_n_jobs: int
    smoke_overrides: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> "BoundedSelectorSettings":
        ml = dict(config.get("ml", {}) or {})
        raw = dict(ml.get("stock_selector_bounded", {}) or {})
        raw.update({k: v for k, v in dict(overrides or {}).items() if v is not None})
        settings = cls(
            dataset_root=Path(raw.get("dataset_root", ml.get("stock_selector_dataset_root", ""))),
            output_root=Path(raw.get("output_root", "reports/ml/readiness/canonical_v2_selector_bounded")),
            oos_start_date=_text(raw.get("oos_start_date")), oos_end_date=_text(raw.get("oos_end_date")),
            max_oos_dates=int(raw["max_oos_dates"]) if raw.get("max_oos_dates") is not None else None,
            model_allowlist=tuple(raw.get("model_allowlist", ("ridge", "elastic_net"))),
            baseline_allowlist=tuple(raw.get("baseline_allowlist", SUPPORTED_BASELINES)),
            include_engineered_features=bool(raw.get("include_engineered_features", ml.get("stock_ranker_include_engineered_features", False))),
            feature_schema_path=Path(raw["feature_schema_path"]) if raw.get("feature_schema_path") else None,
            resume=bool(raw.get("resume", True)), overwrite_incomplete_dates=bool(raw.get("overwrite_incomplete_dates", True)),
            random_seed=int(raw.get("random_seed", ml.get("random_seed", 42))),
            sklearn_n_jobs=int(raw.get("sklearn_n_jobs", ml.get("sklearn_n_jobs", 1))),
            smoke_overrides={k: v for k, v in dict(raw.get("smoke_overrides", {}) or {}).items() if v is not None},
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.oos_end_date and self.max_oos_dates is None:
            raise ValueError("Bounded selector mode requires oos_end_date or max_oos_dates")
        if self.max_oos_dates is not None and self.max_oos_dates < 1:
            raise ValueError("max_oos_dates must be positive")
        if self.oos_start_date and self.oos_end_date and self.oos_start_date > self.oos_end_date:
            raise ValueError("oos_start_date cannot exceed oos_end_date")
        unknown_models = sorted(set(self.model_allowlist) - set(SUPPORTED_MODELS))
        unknown_baselines = sorted(set(self.baseline_allowlist) - set(SUPPORTED_BASELINES))
        if unknown_models or unknown_baselines:
            raise ValueError(f"Unsupported bounded candidates: models={unknown_models}; baselines={unknown_baselines}")
        allowed_overrides = {"random_forest_n_estimators", "random_forest_max_depth", "random_forest_min_samples_leaf", "gradient_boosting_n_estimators", "gradient_boosting_max_depth", "gradient_boosting_learning_rate", "training_row_cap"}
        unknown_overrides = sorted(set(self.smoke_overrides) - allowed_overrides)
        if unknown_overrides:
            raise ValueError(f"Unsupported bounded smoke overrides: {unknown_overrides}")
        if any(float(value) <= 0 for value in self.smoke_overrides.values()):
            raise ValueError("Bounded smoke override values must be positive")


def run_bounded_selector(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = BoundedSelectorSettings.from_config(config, overrides)
    manifest = _read_json(settings.dataset_root / "manifest.json")
    rows_path = settings.dataset_root / "rows.parquet"
    scores_path = settings.dataset_root / "baseline_scores.parquet"
    if not rows_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Frozen selector rows and baseline sidecar are required")
    feature_columns, selected_schema = _resolve_features(rows_path, settings.include_engineered_features, settings.feature_schema_path)
    selected_dates = _select_dates(rows_path, settings)
    run_identity = _run_identity(settings, manifest, feature_columns, selected_schema)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for decision_date in selected_dates:
        destination = settings.output_root / f"date={decision_date}"
        if settings.resume and _valid_completed_date(destination, decision_date, run_identity, rows_path):
            results.append({"decision_date": decision_date, "status": "skipped_complete"})
            continue
        if destination.exists():
            if not settings.overwrite_incomplete_dates:
                raise RuntimeError(f"Incomplete/incompatible date exists and overwrite is disabled: {destination}")
            _remove_date_dir(destination, settings.output_root)
        results.append(_run_date(settings, manifest, rows_path, scores_path, decision_date, feature_columns, run_identity))
    summary = {"contract_version": CONTRACT_VERSION, "dataset_id": manifest["dataset_id"], "dates": results, "feature_count": len(feature_columns), "feature_columns": list(feature_columns), **_feature_selection_reporting(settings, feature_columns, selected_schema), "output_root": str(settings.output_root)}
    _atomic_json(settings.output_root / "run_summary.json", summary)
    return summary


def _run_date(settings, dataset_manifest, rows_path, scores_path, decision_date, feature_columns, run_identity):
    started = time.perf_counter()
    rows_ds, scores_ds = ds.dataset(rows_path, format="parquet"), ds.dataset(scores_path, format="parquet")
    oos_filter = (ds.field("decision_session_date") == decision_date) & (ds.field("selector_eligible") == True)
    source_columns = [name for name in _source_columns(feature_columns) if name in rows_ds.schema.names]
    oos_source = rows_ds.to_table(columns=source_columns, filter=oos_filter)
    if not oos_source.num_rows:
        raise RuntimeError(f"No eligible OOS rows for {decision_date}")
    decision_timestamp = str(oos_source["decision_timestamp"][0].as_py())
    train_filter = (ds.field("decision_timestamp") < decision_timestamp) & (ds.field("label_available_timestamp") <= decision_timestamp) & (ds.field("selector_eligible") == True) & (ds.field("target_status") == "realized")
    train_source = rows_ds.to_table(columns=source_columns, filter=train_filter)
    score_columns = ["row_id", *DETERMINISTIC_SIGNAL_COLUMNS]
    train_scores = scores_ds.to_table(columns=score_columns, filter=ds.field("decision_timestamp") < decision_timestamp)
    oos_scores = scores_ds.to_table(columns=score_columns, filter=ds.field("decision_timestamp") == decision_timestamp)
    train = train_source.join(train_scores, keys="row_id", join_type="inner")
    oos = oos_source.join(oos_scores, keys="row_id", join_type="inner")
    if train.num_rows != train_source.num_rows or oos.num_rows != oos_source.num_rows:
        raise RuntimeError("Frozen rows and baseline sidecar populations do not match")
    training_row_count_before_cap = train.num_rows
    row_cap = settings.smoke_overrides.get("training_row_cap")
    if row_cap is not None and train.num_rows > int(row_cap):
        train = train.sort_by([("decision_timestamp", "ascending"), ("row_id", "ascending")]).slice(train.num_rows - int(row_cap))
    training_decision_max = str(pc.max(train["decision_timestamp"]).as_py())
    training_label_max = str(pc.max(train["label_available_timestamp"]).as_py())
    if training_decision_max >= decision_timestamp or training_label_max > decision_timestamp:
        raise RuntimeError("Bounded selector temporal training guard failed")
    _validate_features(train, feature_columns)
    training_row_count = train.num_rows
    feature_missingness = {name: {"training": train[name].null_count / train.num_rows, "oos": oos[name].null_count / oos.num_rows} for name in feature_columns}
    x_train = _matrix(train, feature_columns); y_train = _float_array(train["actual_forward_return_10d"])
    x_oos = _matrix(oos, feature_columns)
    del train, train_source, train_scores, oos_source, oos_scores
    gc.collect()
    prediction_columns: dict[str, pa.Array] = {}
    statuses, timings, model_details = {}, {}, {}
    for model_id in settings.model_allowlist:
        model = _bounded_model(model_id, settings)
        parameters = _reported_parameters(model_id, model)
        print(f"date={decision_date} model={model_id} status=starting rows={training_row_count} features={len(feature_columns)} parameters={json.dumps(parameters, sort_keys=True)} workers={parameters.get('workers', 1)}")
        try:
            fit_started = time.perf_counter(); model.fit(x_train, y_train); fit_elapsed = time.perf_counter() - fit_started
            predict_started = time.perf_counter(); raw_values = model.predict(x_oos); predict_elapsed = time.perf_counter() - predict_started
            quality = _prediction_quality(raw_values, oos.num_rows, require_dispersion=True)
            values = quality.pop("values")
        except BaseException as exc:
            statuses[model_id] = "failed"
            print(f"date={decision_date} model={model_id} status=failed error={type(exc).__name__}:{exc}")
            if isinstance(exc, PredictionQualityError):
                _atomic_json(settings.output_root / f"date={decision_date}.model={model_id}.failure.json", {"decision_date": decision_date, "model_id": model_id, "status": "rejected", "prediction_quality_contract": _prediction_quality_contract(), "prediction_quality": exc.metrics, "rejection_reasons": exc.reasons})
            raise
        column = f"stock_level_predicted_forward_return_10d_{model_id}"; prediction_columns[column] = pa.array(values, type=pa.float64())
        timings[model_id] = {"fit_seconds": fit_elapsed, "prediction_seconds": predict_elapsed, "total_seconds": fit_elapsed + predict_elapsed}
        model_details[model_id] = {"status": "complete", "parameters": parameters, "prediction_quality": quality, "prediction_coverage": quality["coverage"], "prediction_dispersion": quality["standard_deviation"], **timings[model_id]}
        statuses[model_id] = "complete"
        print(f"date={decision_date} model={model_id} status=complete fit_seconds={fit_elapsed:.6f} prediction_seconds={predict_elapsed:.6f}")
    for baseline_id in settings.baseline_allowlist:
        column = BASELINE_CANDIDATES[baseline_id]
        prediction_columns[column] = oos[column]
        baseline_quality = _prediction_quality(oos[column].to_numpy(zero_copy_only=False), oos.num_rows, require_dispersion=False)
        baseline_quality.pop("values")
        statuses[baseline_id] = "complete"; timings[baseline_id] = {"fit_seconds": 0.0, "prediction_seconds": 0.0, "total_seconds": 0.0}
        model_details[baseline_id] = {"status": "complete", "parameters": {"trainable": False}, "prediction_quality": baseline_quality, **timings[baseline_id]}
    keep = [name for name in ("row_id", "asset_id", "symbol", "rebalance_date", "decision_timestamp", "decision_session_date", "label_available_timestamp", "actual_forward_return_10d", *TARGET_OUTPUT_COLUMNS) if name in oos.schema.names]
    predictions = oos.select(keep)
    for name, values in prediction_columns.items(): predictions = predictions.append_column(name, values)
    _validate_predictions(predictions, decision_date, tuple(prediction_columns))
    temp_dir = settings.output_root / f".date={decision_date}.{uuid.uuid4().hex}.tmp"
    temp_dir.mkdir(parents=True)
    prediction_path = temp_dir / "predictions.parquet"
    pq.write_table(predictions, prediction_path, compression="zstd")
    checksum = _sha256(prediction_path)
    metrics = {"decision_date": decision_date, "training_row_count": training_row_count, "training_row_count_before_cap": training_row_count_before_cap, "oos_row_count": oos.num_rows, "oos_symbol_count": len(set(oos["symbol"].to_pylist())), "feature_missingness": feature_missingness, "peak_memory_bytes": None, "prediction_quality_contract": _prediction_quality_contract(), "model_timings_seconds": timings, "model_details": model_details, "smoke_overrides": dict(settings.smoke_overrides), "non_production_smoke": bool(settings.smoke_overrides)}
    _atomic_json(temp_dir / "metrics.json", metrics)
    manifest = {**run_identity, "contract_version": CONTRACT_VERSION, "decision_date": decision_date, "training_row_count": training_row_count, "training_row_count_before_cap": training_row_count_before_cap, "training_decision_timestamp_max": training_decision_max, "training_label_available_timestamp_max": training_label_max, "oos_row_count": oos.num_rows, "oos_symbol_count": metrics["oos_symbol_count"], "label_availability_cutoff": decision_timestamp, "model_fit_statuses": statuses, "model_details": model_details, "smoke_overrides": dict(settings.smoke_overrides), "non_production_smoke": bool(settings.smoke_overrides), "prediction_checksum": checksum, "completion_status": "complete", "git_commit": _git_commit(), "elapsed_seconds": time.perf_counter() - started}
    _atomic_json(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, settings.output_root / f"date={decision_date}")
    for model_id in settings.model_allowlist:
        (settings.output_root / f"date={decision_date}.model={model_id}.failure.json").unlink(missing_ok=True)
    return {"decision_date": decision_date, "status": "complete", **metrics, "elapsed_seconds": manifest["elapsed_seconds"]}


def _valid_completed_date(path: Path, decision_date: str, identity: Mapping[str, Any], rows_path: Path) -> bool:
    try:
        manifest = _read_json(path / "manifest.json")
        if manifest.get("completion_status") != "complete" or manifest.get("decision_date") != decision_date:
            return False
        if not manifest.get("training_decision_timestamp_max") or not manifest.get("training_label_available_timestamp_max"):
            return False
        if any(manifest.get(key) != value for key, value in identity.items()): return False
        predictions_path = path / "predictions.parquet"
        if _sha256(predictions_path) != manifest.get("prediction_checksum"): return False
        table = pq.read_table(predictions_path)
        candidates = tuple(f"stock_level_predicted_forward_return_10d_{name}" for name in identity["model_allowlist"]) + tuple(BASELINE_CANDIDATES[name] for name in identity["baseline_allowlist"])
        _validate_predictions(table, decision_date, candidates)
        for model_id in identity["model_allowlist"]:
            _prediction_quality(table[f"stock_level_predicted_forward_return_10d_{model_id}"].to_numpy(zero_copy_only=False), table.num_rows, require_dispersion=True)
        for baseline_id in identity["baseline_allowlist"]:
            _prediction_quality(table[BASELINE_CANDIDATES[baseline_id]].to_numpy(zero_copy_only=False), table.num_rows, require_dispersion=False)
        source_ids = set(ds.dataset(rows_path, format="parquet").to_table(columns=["row_id"], filter=ds.field("decision_session_date") == decision_date)["row_id"].to_pylist())
        return set(table["row_id"].to_pylist()) <= source_ids
    except Exception:
        return False


def _select_dates(rows_path: Path, settings: BoundedSelectorSettings) -> list[str]:
    filt = None
    if settings.oos_start_date: filt = ds.field("decision_session_date") >= settings.oos_start_date
    if settings.oos_end_date:
        end = ds.field("decision_session_date") <= settings.oos_end_date
        filt = end if filt is None else filt & end
    table = ds.dataset(rows_path, format="parquet").to_table(columns=["decision_session_date"], filter=filt)
    dates = sorted(set(table["decision_session_date"].to_pylist()))
    if settings.max_oos_dates is not None: dates = dates[:settings.max_oos_dates] if settings.oos_start_date else dates[-settings.max_oos_dates:]
    if not dates: raise RuntimeError("Bounded selector date selection is empty")
    return dates


def _resolve_features(rows_path: Path, include_engineered: bool, feature_schema_path: Path | None = None) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    parquet_schema = pq.ParquetFile(rows_path).schema_arrow
    names = set(parquet_schema.names)
    if feature_schema_path is not None:
        schema = load_feature_schema(feature_schema_path)
        features = tuple(row["name"] for row in schema["features"])
        missing = sorted(set(features) - names - set(DETERMINISTIC_SIGNAL_COLUMNS))
        if missing:
            raise RuntimeError(f"Requested selector features are missing: {missing}")
        type_mismatches = []
        for row in schema["features"]:
            name = row["name"]
            if name in names and str(parquet_schema.field(name).type) != row["data_type"]:
                type_mismatches.append(f"{name}: expected={row['data_type']} actual={parquet_schema.field(name).type}")
        if type_mismatches:
            raise RuntimeError(f"Selector feature schema type mismatch: {type_mismatches}")
        return features, {"path": str(feature_schema_path), "contract_version": schema["contract_version"], "schema_hash": schema["schema_hash"]}
    engineered = tuple(name for name in ENGINEERED_FEATURE_COLUMNS if name in names) if include_engineered else ()
    features = (*DETERMINISTIC_SIGNAL_COLUMNS, *engineered)
    missing = sorted(set(engineered) - names)
    if missing: raise RuntimeError(f"Requested engineered features are missing: {missing}")
    if any(name.startswith(OUTCOME_PREFIX) for name in features): raise RuntimeError("Outcome columns cannot be selector features")
    return features, None


def _source_columns(features):
    base = ["row_id", "asset_id", "symbol", "rebalance_date", "decision_timestamp", "decision_session_date", "label_available_timestamp", "selector_eligible", "target_status", "actual_forward_return_10d", *TARGET_OUTPUT_COLUMNS]
    return list(dict.fromkeys([*base, *(name for name in features if name not in DETERMINISTIC_SIGNAL_COLUMNS)]))


def _matrix(table: pa.Table, features):
    import numpy as np
    return np.column_stack([pc.cast(table[name], pa.float64()).to_numpy(zero_copy_only=False) for name in features])


def _float_array(column): return pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)


def _validate_features(table, features):
    for name in features:
        if name not in table.schema.names or table[name].null_count == table.num_rows:
            raise RuntimeError(f"Requested selector feature is missing or entirely null: {name}")


def _validate_predictions(table, decision_date, candidates):
    ids = table["row_id"].to_pylist()
    if len(ids) != len(set(ids)): raise RuntimeError("Prediction row IDs are not unique")
    if set(table["decision_session_date"].to_pylist()) != {decision_date}: raise RuntimeError("Prediction rows contain an unexpected decision date")
    for name in candidates:
        if name not in table.schema.names: raise RuntimeError(f"Missing candidate prediction: {name}")
        if any(value is None or not math.isfinite(float(value)) for value in table[name].to_pylist()): raise RuntimeError(f"Candidate contains non-finite predictions: {name}")


def _prediction_quality(values: Any, expected_count: int, *, require_dispersion: bool) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(values)
    reasons: list[str] = []
    if array.ndim != 1:
        reasons.append(f"prediction_output_must_be_one_dimensional: ndim={array.ndim}")
        flat = array.reshape(-1)
    else:
        flat = array
    try:
        numeric = flat.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise PredictionQualityError([f"predictions_are_not_numeric: {exc}"], {"prediction_count": int(flat.size), "expected_count": expected_count}) from exc
    count = int(numeric.size)
    finite_mask = np.isfinite(numeric)
    finite = numeric[finite_mask]
    finite_count = int(finite.size)
    coverage = finite_count / expected_count if expected_count else 0.0
    unique_count = int(np.unique(finite).size)
    minimum = float(np.min(finite)) if finite_count else None
    maximum = float(np.max(finite)) if finite_count else None
    mean = float(np.mean(finite)) if finite_count else None
    standard_deviation = float(np.std(finite, ddof=0)) if finite_count else None
    value_range = maximum - minimum if finite_count else None
    quantiles = {f"q{int(q * 100):03d}": float(np.quantile(finite, q)) for q in PREDICTION_QUANTILES} if finite_count else {}
    metrics = {
        "contract_version": PREDICTION_QUALITY_CONTRACT_VERSION,
        "validation_scope": "fitted_candidate" if require_dispersion else "direct_baseline",
        "prediction_count": count,
        "expected_count": expected_count,
        "finite_count": finite_count,
        "coverage": coverage,
        "unique_finite_value_count": unique_count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "range": value_range,
        "quantiles": quantiles,
        "distinct_rank_count": unique_count,
        "dispersion_requirement_applied": require_dispersion,
    }
    if count != expected_count:
        reasons.append(f"prediction_count_mismatch: expected={expected_count} actual={count}")
    if finite_count != count:
        reasons.append(f"non_finite_predictions: count={count - finite_count}")
    if coverage < 1.0:
        reasons.append(f"prediction_coverage_below_one: coverage={coverage}")
    if require_dispersion:
        if unique_count < 2:
            reasons.append(f"unique_finite_prediction_count_below_two: actual={unique_count}")
        if standard_deviation is None or standard_deviation < MIN_PREDICTION_STANDARD_DEVIATION:
            reasons.append(f"prediction_standard_deviation_below_tolerance: actual={standard_deviation} tolerance={MIN_PREDICTION_STANDARD_DEVIATION}")
        if value_range is None or value_range < MIN_PREDICTION_RANGE:
            reasons.append(f"prediction_range_below_tolerance: actual={value_range} tolerance={MIN_PREDICTION_RANGE}")
    if reasons:
        metrics["status"] = "rejected"; metrics["rejection_reasons"] = reasons
        raise PredictionQualityError(reasons, metrics)
    metrics["status"] = "accepted"
    metrics["values"] = [float(value) for value in numeric]
    return metrics


def _prediction_quality_contract() -> dict[str, Any]:
    return {
        "contract_version": PREDICTION_QUALITY_CONTRACT_VERSION,
        "fitted_candidate_requirements": {"finite": True, "coverage": 1.0, "minimum_unique_finite_values": 2, "minimum_standard_deviation": MIN_PREDICTION_STANDARD_DEVIATION, "minimum_range": MIN_PREDICTION_RANGE},
        "direct_baseline_dispersion_requirement": False,
        "quantiles": list(PREDICTION_QUANTILES),
    }


def _feature_selection_reporting(settings: BoundedSelectorSettings, features, selected_schema) -> dict[str, Any]:
    mode = "explicit_versioned_schema" if selected_schema else "legacy_engineered_flag" if settings.include_engineered_features else "compatibility_signals_only"
    return {"feature_selection_mode": mode, "selected_feature_count": len(features), "selected_feature_schema": selected_schema, "legacy_include_engineered_features_flag": settings.include_engineered_features}


def _run_identity(settings, manifest, features, selected_schema):
    payload = {"dataset_id": manifest["dataset_id"], "source_dataset_checksum": manifest["source_sha256"], "rows_checksum": manifest.get("checksums", {}).get("rows.parquet"), "baseline_checksum": manifest.get("checksums", {}).get("baseline_scores.parquet"), "feature_schema_hash": _sha256(settings.dataset_root / "feature_schema.json"), **_feature_selection_reporting(settings, features, selected_schema), "prediction_quality_contract": _prediction_quality_contract(), "target_field": "actual_forward_return_10d", "model_allowlist": list(settings.model_allowlist), "baseline_allowlist": list(settings.baseline_allowlist), "feature_columns": list(features), "random_seed": settings.random_seed, "sklearn_n_jobs": settings.sklearn_n_jobs, "smoke_overrides": dict(settings.smoke_overrides), "non_production_smoke": bool(settings.smoke_overrides), "runner_contract_version": CONTRACT_VERSION}
    payload["config_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _remove_date_dir(path, root):
    if path.parent.resolve() != root.resolve(): raise RuntimeError("Unsafe bounded date path")
    shutil.rmtree(path)


def _atomic_json(path, payload):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"); os.replace(tmp, path)


def _read_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()
def _git_commit(): return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
def _text(value): return str(value).strip() if value not in (None, "") else None


def _bounded_model(model_id: str, settings: BoundedSelectorSettings):
    model = _build_tabular_model(model_id, settings.random_seed, settings.sklearn_n_jobs)
    override_keys = {
        "random_forest": {"random_forest_n_estimators": "randomforestregressor__n_estimators", "random_forest_max_depth": "randomforestregressor__max_depth", "random_forest_min_samples_leaf": "randomforestregressor__min_samples_leaf"},
        "gradient_boosting": {"gradient_boosting_n_estimators": "gradientboostingregressor__n_estimators", "gradient_boosting_max_depth": "gradientboostingregressor__max_depth", "gradient_boosting_learning_rate": "gradientboostingregressor__learning_rate"},
    }.get(model_id, {})
    parameters = {target: settings.smoke_overrides[source] for source, target in override_keys.items() if source in settings.smoke_overrides}
    if parameters:
        model.set_params(**parameters)
    return model


def _reported_parameters(model_id: str, model: Any) -> dict[str, Any]:
    params = model.get_params()
    if model_id == "random_forest":
        prefix = "randomforestregressor__"
        return {"estimator_count": params[prefix + "n_estimators"], "max_depth": params[prefix + "max_depth"], "min_samples_leaf": params[prefix + "min_samples_leaf"], "max_features": params[prefix + "max_features"], "bootstrap": params[prefix + "bootstrap"], "max_samples": params.get(prefix + "max_samples"), "subsampling": None, "learning_rate": None, "random_seed": params[prefix + "random_state"], "workers": params[prefix + "n_jobs"], "sample_weighting": None, "preprocessing": "SimpleImputer(strategy=median)"}
    if model_id == "gradient_boosting":
        prefix = "gradientboostingregressor__"
        return {"estimator_count": params[prefix + "n_estimators"], "max_depth": params[prefix + "max_depth"], "min_samples_leaf": params[prefix + "min_samples_leaf"], "max_features": params[prefix + "max_features"], "subsampling": params[prefix + "subsample"], "learning_rate": params[prefix + "learning_rate"], "loss": params[prefix + "loss"], "random_seed": params[prefix + "random_state"], "workers": 1, "sample_weighting": None, "preprocessing": "SimpleImputer(strategy=median)"}
    return {"random_seed": params.get("regressor__random_state", params.get("random_state", settings_random_seed(model))), "workers": 1, "sample_weighting": None, "preprocessing": "embedded sklearn imputer/scaler pipeline"}


def settings_random_seed(model: Any) -> Any:
    return next((value for key, value in model.get_params().items() if key.endswith("random_state") and value is not None), None)
