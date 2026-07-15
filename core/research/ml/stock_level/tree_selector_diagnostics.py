from __future__ import annotations

import json
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from core.research.ml.stock_level.bounded_selector_runner import (
    _bounded_model, _float_array, _matrix, _prediction_quality, _resolve_features,
)
from core.research.ml.stock_level.selector_dataset import DETERMINISTIC_SIGNAL_COLUMNS
from core.research.ml.stock_level.selector_feature_schema import load_feature_schema

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
DIAGNOSTIC_CONTRACT = "tree_selector_routing_diagnostic_v1"
ROLLING_WINDOW_CONTRACT = "selector_date_session_window_v1"


def window_identity(*, requested_start_date: str | None, trailing_sessions: int | None, resolved_start_date: str | None, decision_date: str) -> dict[str, Any]:
    mode = "trailing_sessions" if trailing_sessions is not None else "explicit_start_date" if requested_start_date else "expanding"
    return {"contract_version": ROLLING_WINDOW_CONTRACT, "mode": mode, "requested_start_date": requested_start_date, "trailing_sessions": trailing_sessions, "resolved_start_date": resolved_start_date, "exclusive_end_date": decision_date}


def filter_legal_training_window(table, *, decision_timestamp: str, start_date: str | None = None):
    mask = pc.and_(pc.less(table["decision_timestamp"], decision_timestamp), pc.less_equal(table["label_available_timestamp"], decision_timestamp))
    if start_date is not None:
        mask = pc.and_(mask, pc.greater_equal(table["decision_session_date"], start_date))
    return table.filter(mask)


def temporal_legality_report(*, decision_timestamp: str, training_decision_max: str, training_label_available_max: str) -> dict[str, Any]:
    return {"decision_timestamp_cutoff": decision_timestamp, "training_decision_timestamp_max": training_decision_max, "training_label_available_timestamp_max": training_label_available_max, "decision_timestamp_guard_passed": training_decision_max < decision_timestamp, "label_availability_guard_passed": training_label_available_max <= decision_timestamp}


def feature_variation(values: np.ndarray, *, missing_count: int = 0) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    unique = int(np.unique(finite).size)
    std = float(np.std(finite, ddof=0)) if finite.size else None
    value_range = float(np.max(finite) - np.min(finite)) if finite.size else None
    if not finite.size:
        category = "missing_or_imputed"
    elif unique == 1:
        category = "common_to_every_stock"
    elif missing_count:
        category = "partly_stock_specific"
    else:
        category = "stock_specific"
    return {"finite_count": int(finite.size), "unique_value_count": unique, "minimum": float(np.min(finite)) if finite.size else None, "maximum": float(np.max(finite)) if finite.size else None, "standard_deviation": std, "range": value_range, "missing_count": missing_count, "missingness": missing_count / len(values), "constant": unique <= 1, "near_constant": bool(std is not None and std < 1e-12 or value_range is not None and value_range < 1e-12), "classification": category}


def forest_routing(estimator, x_oos: np.ndarray, features: tuple[str, ...]) -> list[dict[str, Any]]:
    reports = []
    for tree_index, tree in enumerate(estimator.estimators_):
        structure = tree.tree_; leaves = tree.apply(x_oos); unique_leaves, counts = np.unique(leaves, return_counts=True)
        split_nodes = np.flatnonzero(structure.feature >= 0)
        reports.append({
            "tree_index": tree_index, "actual_depth": int(structure.max_depth), "node_count": int(structure.node_count),
            "leaf_count": int(np.sum(structure.children_left == -1)),
            "splits": [{"node_id": int(node), "feature_index": int(structure.feature[node]), "feature_name": features[structure.feature[node]], "threshold": float(structure.threshold[node])} for node in split_nodes],
            "feature_importances": {features[index]: float(value) for index, value in enumerate(tree.feature_importances_) if value > 0},
            "distinct_oos_leaf_ids": int(unique_leaves.size),
            "oos_leaf_routing": [{"leaf_id": int(leaf), "oos_row_count": int(count), "leaf_prediction": float(structure.value[leaf, 0, 0])} for leaf, count in zip(unique_leaves, counts)],
            "every_oos_row_reaches_same_leaf": unique_leaves.size == 1,
        })
    return reports


def boosting_routing(estimator, x_oos: np.ndarray, features: tuple[str, ...]) -> list[dict[str, Any]]:
    cumulative = np.asarray(estimator.init_.predict(x_oos), dtype=float)
    reports = []
    for stage_index, row in enumerate(estimator.estimators_):
        tree = row[0]; structure = tree.tree_; root_feature = int(structure.feature[0]); threshold = float(structure.threshold[0])
        stage_output = tree.predict(x_oos); cumulative = cumulative + estimator.learning_rate * stage_output
        left = int(np.sum(x_oos[:, root_feature] <= threshold)) if root_feature >= 0 else len(x_oos)
        quality = _prediction_quality(cumulative, len(cumulative), require_dispersion=False); quality.pop("values")
        reports.append({"stage_index": stage_index, "split_feature": features[root_feature] if root_feature >= 0 else None, "feature_index": root_feature, "threshold": threshold if root_feature >= 0 else None, "oos_left_count": left, "oos_right_count": len(x_oos) - left, "distinct_stage_output_count": int(np.unique(stage_output).size), "stage_outputs": [float(value) for value in np.unique(stage_output)], "stage_contributes_same_value_to_every_oos_row": np.unique(stage_output).size == 1, "cumulative_prediction_quality": quality})
    return reports


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {"missingness": 1 - len(finite) / len(values), "median_imputed_value": float(np.nanmedian(values)) if len(finite) else None, "quantiles": {f"q{int(q*100):03d}": float(np.nanquantile(values, q)) for q in QUANTILES} if len(finite) else {}}


def diagnostic_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = {key: payload[key] for key in ("contract_version", "selector_completion_status", "decision_date", "model_id", "selected_feature_schema", "training_window", "training_row_count", "training_date_min", "training_date_max", "temporal_legality", "prediction_quality_accepted", "prediction_quality", "elapsed_seconds")}
    manifest["artifact_kind"] = "research_diagnostic"
    manifest["eligible_as_completed_selector_partition"] = False
    return manifest


def run_tree_diagnostic(*, dataset_root: Path, feature_schema_path: Path, output_root: Path, decision_date: str, model_id: str, settings: Any, requested_start_date: str | None = None, trailing_sessions: int | None = None) -> dict[str, Any]:
    started = time.perf_counter(); rows_path = dataset_root / "rows.parquet"; scores_path = dataset_root / "baseline_scores.parquet"
    features, selected_schema = _resolve_features(rows_path, False, feature_schema_path)
    schema = load_feature_schema(feature_schema_path)
    if tuple(row["name"] for row in schema["features"]) != features: raise RuntimeError("Feature order does not match versioned schema")
    rows_ds, scores_ds = ds.dataset(rows_path, format="parquet"), ds.dataset(scores_path, format="parquet")
    oos_filter = (ds.field("decision_session_date") == decision_date) & (ds.field("selector_eligible") == True)
    source_features = [name for name in features if name not in DETERMINISTIC_SIGNAL_COLUMNS]
    columns = ["row_id", "symbol", "decision_timestamp", "decision_session_date", "label_available_timestamp", "selector_eligible", "target_status", "actual_forward_return_10d", *source_features]
    oos_source = rows_ds.to_table(columns=columns, filter=oos_filter); decision_timestamp = str(oos_source["decision_timestamp"][0].as_py())
    resolved_start = requested_start_date
    if trailing_sessions is not None:
        dates = rows_ds.to_table(columns=["decision_session_date"], filter=(ds.field("decision_session_date") < decision_date) & (ds.field("selector_eligible") == True))["decision_session_date"].to_pylist()
        unique_dates = sorted(set(dates)); resolved_start = unique_dates[-trailing_sessions] if len(unique_dates) >= trailing_sessions else unique_dates[0]
    train_filter = (ds.field("decision_timestamp") < decision_timestamp) & (ds.field("label_available_timestamp") <= decision_timestamp) & (ds.field("selector_eligible") == True) & (ds.field("target_status") == "realized")
    if resolved_start: train_filter = train_filter & (ds.field("decision_session_date") >= resolved_start)
    train_source = rows_ds.to_table(columns=columns, filter=train_filter)
    score_columns = ["row_id", *DETERMINISTIC_SIGNAL_COLUMNS]
    train_scores = scores_ds.to_table(columns=score_columns, filter=(ds.field("decision_timestamp") < decision_timestamp) & ((ds.field("decision_timestamp") >= f"{resolved_start} 00:00:00+00:00") if resolved_start else (ds.field("decision_timestamp") < decision_timestamp)))
    oos_scores = scores_ds.to_table(columns=score_columns, filter=ds.field("decision_timestamp") == decision_timestamp)
    train = train_source.join(train_scores, keys="row_id", join_type="inner"); oos = oos_source.join(oos_scores, keys="row_id", join_type="inner")
    if train.num_rows != train_source.num_rows or oos.num_rows != oos_source.num_rows: raise RuntimeError("Diagnostic sidecar population mismatch")
    legality = temporal_legality_report(decision_timestamp=decision_timestamp, training_decision_max=str(pc.max(train["decision_timestamp"]).as_py()), training_label_available_max=str(pc.max(train["label_available_timestamp"]).as_py()))
    if not legality["decision_timestamp_guard_passed"] or not legality["label_availability_guard_passed"]: raise RuntimeError("Diagnostic temporal legality guard failed")
    training_row_count = train.num_rows; training_date_min = str(pc.min(train["decision_session_date"]).as_py()); training_date_max = str(pc.max(train["decision_session_date"]).as_py())
    training_decision_max = str(pc.max(train["decision_timestamp"]).as_py()); training_label_max = str(pc.max(train["label_available_timestamp"]).as_py())
    x_train = _matrix(train, features); y_train = _float_array(train["actual_forward_return_10d"]); x_oos = _matrix(oos, features)
    recent_mask = pc.greater_equal(train["decision_session_date"], "2021-06-25").to_numpy(zero_copy_only=False)
    raw_oos = {name: np.asarray(oos[name].to_numpy(zero_copy_only=False), dtype=float) for name in features}
    oos_row_count = oos.num_rows
    del train, train_source, train_scores, oos_source, oos_scores, oos
    gc.collect()
    model = __import__("core.research.ml.stock_level.bounded_selector_runner", fromlist=["_bounded_model"])._bounded_model(model_id, settings)
    fit_started = time.perf_counter(); model.fit(x_train, y_train); fit_seconds = time.perf_counter() - fit_started
    imputer = model.named_steps["simpleimputer"]; x_oos_imputed = imputer.transform(x_oos)
    predict_started = time.perf_counter(); predictions = model.predict(x_oos); prediction_seconds = time.perf_counter() - predict_started
    try: quality = _prediction_quality(predictions, len(x_oos), require_dispersion=True); quality.pop("values"); accepted = True
    except Exception as exc: quality = getattr(exc, "metrics", {"status": "rejected", "rejection_reasons": [str(exc)]}); accepted = False
    feature_report = []
    for index, name in enumerate(features):
        before = feature_variation(raw_oos[name], missing_count=int(np.isnan(raw_oos[name]).sum())); after = feature_variation(x_oos_imputed[:, index])
        feature_report.append({"feature_index": index, "feature_name": name, "before_imputation": before, "after_training_median_imputation": after, "training_median": float(imputer.statistics_[index])})
    estimator = model.named_steps["randomforestregressor" if model_id == "random_forest" else "gradientboostingregressor"]
    routing = forest_routing(estimator, x_oos_imputed, features) if model_id == "random_forest" else boosting_routing(estimator, x_oos_imputed, features)
    used = sorted({split["feature_name"] for tree in routing for split in tree.get("splits", [])} if model_id == "random_forest" else {stage["split_feature"] for stage in routing if stage["split_feature"]})
    distribution = {}
    thresholds = {}
    if model_id == "random_forest":
        for tree in routing:
            for split in tree["splits"]: thresholds.setdefault(split["feature_name"], []).append(split["threshold"])
    else:
        for stage in routing: thresholds.setdefault(stage["split_feature"], []).append(stage["threshold"])
    for name in used:
        index = features.index(name); distribution[name] = {"expanding_or_selected_training": _numeric_summary(x_train[:, index]), "recent_five_year_training": _numeric_summary(x_train[recent_mask, index]), "oos": _numeric_summary(x_oos[:, index]), "split_thresholds": [{"threshold": threshold, "oos_left_fraction": float(np.mean(x_oos_imputed[:, index] <= threshold)), "oos_right_fraction": float(np.mean(x_oos_imputed[:, index] > threshold))} for threshold in sorted(set(thresholds[name]))]}
    payload = {"contract_version": DIAGNOSTIC_CONTRACT, "selector_completion_status": "not_published_diagnostic_only", "decision_date": decision_date, "model_id": model_id, "selected_feature_schema": selected_schema, "feature_order_verified": True, "feature_columns": list(features), "training_window": window_identity(requested_start_date=requested_start_date, trailing_sessions=trailing_sessions, resolved_start_date=resolved_start, decision_date=decision_date), "training_row_count": training_row_count, "training_date_min": training_date_min, "training_date_max": training_date_max, "training_decision_timestamp_max": training_decision_max, "training_label_available_timestamp_max": training_label_max, "oos_row_count": oos_row_count, "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds, "prediction_quality_accepted": accepted, "prediction_quality": quality, "oos_feature_variation": feature_report, "routing": routing, "split_feature_distributions": distribution, "elapsed_seconds": time.perf_counter() - started}
    payload["temporal_legality"] = legality
    output_root.mkdir(parents=True, exist_ok=True); (output_root / "diagnostic.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); (output_root / "diagnostic_manifest.json").write_text(json.dumps(diagnostic_manifest_payload(payload), indent=2, sort_keys=True), encoding="utf-8")
    return payload
