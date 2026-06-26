from __future__ import annotations

import csv
import ctypes.util
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
from statistics import mean
from typing import Any

from core.research.ml.calibration import (
    build_probability_calibration,
    compare_calibration_methods,
)
from core.research.ml.evaluation import classification_metrics
from core.research.ml.leaderboard import write_leaderboard
from core.research.ml.overlay import overlay_decision_rule, should_reduce_exposure


_MAX_ABS_PERIOD_RETURN = 5.0


@dataclass(frozen=True)
class MetaEnsembleResult:
    output_dir: Path
    meta_dataset_path: Path
    audit_path: Path
    metrics_path: Path
    walk_forward_metrics_path: Path
    probability_calibration_path: Path
    calibrated_probability_calibration_path: Path
    holdout_shadow_overlay_path: Path
    threshold_sweep_path: Path
    meta_model_comparison_path: Path
    promotion_gates_path: Path
    overlay_model_comparison_path: Path
    leaderboard_path: Path
    leaderboard_markdown_path: Path


@dataclass
class MetaLearnerModel:
    model_type: str
    feature_names: list[str]
    estimator: Any = None
    constant_probability: float | None = None

    def predict_proba(self, features: list[dict[str, float]]) -> list[float]:
        if self.constant_probability is not None:
            return [self.constant_probability for _ in features]
        matrix = _feature_matrix(features, self.feature_names)
        probabilities = self.estimator.predict_proba(matrix)[:, 1].tolist()
        return [float(value) for value in probabilities]


def run_meta_ensemble(config: dict[str, Any]) -> MetaEnsembleResult:
    ml_config = config.get("ml", {})
    output_dir = Path(ml_config.get("output_dir", "reports/ml/regime_transformer_meta_ensemble_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(ml_config.get("meta_dataset_path", "cache/ml/meta_ensemble_dataset.csv"))
    expanded_dataset_path = Path(
        ml_config.get("expanded_rebalance_dataset_path", "cache/ml/expanded_rebalance_dataset.csv")
    )
    source_dirs = [Path(path) for path in ml_config.get("source_prediction_dirs", [])]
    source_predictions, warnings = _load_source_predictions(source_dirs)
    source_models = sorted(source_predictions)
    if len(source_models) < 2:
        raise RuntimeError(
            "Meta ensemble requires at least two available source prediction artifacts"
        )

    expanded_rows = _read_csv(expanded_dataset_path)
    meta_rows, audit = build_meta_dataset_rows(expanded_rows, source_predictions)
    audit.update({
        "warnings": warnings,
        "source_prediction_dirs": [str(path) for path in source_dirs],
        "research_only": True,
        "trading_impact": "none",
    })
    _write_csv(dataset_path, meta_rows)
    audit_path = output_dir / "meta_dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    train_rows = [row for row in meta_rows if row["split"] == "out_of_fold"]
    holdout_rows = [row for row in meta_rows if row["split"] == "holdout"]
    if not train_rows or not holdout_rows:
        raise RuntimeError("Meta ensemble requires out-of-fold and holdout rows")

    features = [_feature_values(row) for row in train_rows]
    labels = [int(row["actual_label"]) for row in train_rows]
    holdout_features = [_feature_values(row) for row in holdout_rows]
    holdout_labels = [int(row["actual_label"]) for row in holdout_rows]

    random_seed = int(ml_config.get("random_seed", 42))
    selected_meta_model_type = str(
        ml_config.get("meta_model_type", "logistic_regression")
    )
    model = _fit_meta_model(
        selected_meta_model_type,
        features,
        labels,
        random_seed=random_seed,
    )
    train_probabilities = model.predict_proba(features)
    holdout_probabilities = model.predict_proba(holdout_features)
    threshold = float(ml_config.get("decision_threshold", 0.5))
    predictions = [int(probability >= threshold) for probability in holdout_probabilities]
    label_type = ml_config.get("label_type", "should_reduce_exposure")
    reduce_when, decision_rule = overlay_decision_rule(str(label_type))
    reduced_exposure = float(ml_config.get("promotion_reduced_exposure", 0.7))
    metrics_path = output_dir / "metrics.json"
    metrics = classification_metrics(holdout_labels, predictions)
    metrics_path.write_text(json.dumps({
        "mode": "research",
        "ensemble_name": ml_config.get("ensemble_name", "regime_transformer_meta_ensemble_v1"),
        "model_type": _meta_ensemble_name(selected_meta_model_type),
        "meta_model_type": selected_meta_model_type,
        "label_type": label_type,
        "feature_set": ml_config.get("feature_set", "expanded_rebalance_v1"),
        "train_sample_count": len(train_rows),
        "test_sample_count": len(holdout_rows),
        "metrics": metrics,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }, indent=2), encoding="utf-8")

    calibration_path = output_dir / "probability_calibration.json"
    calibration = build_probability_calibration(holdout_labels, holdout_probabilities)
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    calibrated_probability_calibration_path = (
        output_dir / "calibrated_probability_calibration.json"
    )
    calibration_comparison = compare_calibration_methods(
        labels,
        train_probabilities,
        holdout_labels,
        holdout_probabilities,
        bin_count=int(ml_config.get("calibration_bin_count", 10)),
    )
    calibrated_probability_calibration_path.write_text(
        json.dumps(calibration_comparison, indent=2),
        encoding="utf-8",
    )
    leaderboard_calibration = _best_calibration_summary(calibration_comparison)

    walk_forward_path = output_dir / "walk_forward_metrics.json"
    walk_forward = _walk_forward_meta_evaluation(
        meta_rows,
        model_type=selected_meta_model_type,
        fold_count=int(ml_config.get("walk_forward_folds", 3)),
        threshold=threshold,
        reduced_exposure=reduced_exposure,
        reduce_when=reduce_when,
        random_seed=random_seed,
        calibration_bin_count=int(ml_config.get("calibration_bin_count", 10)),
    )
    walk_forward_path.write_text(
        json.dumps(walk_forward, indent=2),
        encoding="utf-8",
    )

    holdout_overlay_path = output_dir / "holdout_shadow_overlay.json"
    overlay = _overlay_summary(
        holdout_rows,
        holdout_probabilities,
        threshold,
        reduced_exposure,
        reduce_when=reduce_when,
    )
    holdout_overlay_path.write_text(json.dumps({
        "mode": "meta_ensemble_holdout_shadow_research_only",
        "decision_threshold": threshold,
        "reduced_exposure": reduced_exposure,
        "overlay_decision_rule": decision_rule,
        "result": overlay,
        "research_only": True,
        "trading_impact": "none",
    }, indent=2), encoding="utf-8")

    threshold_sweep_path = output_dir / "threshold_sweep.json"
    threshold_sweep = _threshold_sweep(
        holdout_rows,
        holdout_labels,
        holdout_probabilities,
        thresholds=ml_config.get("decision_thresholds", [0.5, 0.6, 0.7]),
        reduced_exposures=ml_config.get("reduced_exposures", [0.7, 0.8, 0.9]),
        reduce_when=reduce_when,
    )
    threshold_sweep_path.write_text(
        json.dumps(threshold_sweep, indent=2),
        encoding="utf-8",
    )

    meta_model_comparison_path = output_dir / "meta_model_comparison.json"
    meta_model_comparison = _compare_meta_learners(
        train_rows,
        holdout_rows,
        model_types=_meta_model_types(ml_config, selected_meta_model_type),
        threshold=threshold,
        reduced_exposure=reduced_exposure,
        reduce_when=reduce_when,
        random_seed=random_seed,
        calibration_bin_count=int(ml_config.get("calibration_bin_count", 10)),
        all_rows=meta_rows,
        walk_forward_folds=int(ml_config.get("walk_forward_folds", 3)),
        promotion_config=ml_config,
    )
    meta_model_comparison_path.write_text(
        json.dumps(meta_model_comparison, indent=2),
        encoding="utf-8",
    )

    promotion_gates_path = output_dir / "promotion_gates.json"
    promotion_gates = _promotion_gate_report(
        metrics=metrics,
        calibration=leaderboard_calibration,
        overlay=overlay,
        walk_forward=walk_forward,
        config=ml_config,
    )
    promotion_gates_path.write_text(
        json.dumps(promotion_gates, indent=2),
        encoding="utf-8",
    )

    overlay_comparison_path = output_dir / "overlay_model_comparison.json"
    scenarios = []
    for decision_threshold in ml_config.get("decision_thresholds", [0.5, 0.6, 0.7]):
        for scenario_reduced_exposure in ml_config.get("reduced_exposures", [0.7, 0.8, 0.9]):
            scenarios.append({
                "decision_threshold": float(decision_threshold),
                "reduced_exposure": float(scenario_reduced_exposure),
                "summary": _overlay_summary(
                    holdout_rows,
                    holdout_probabilities,
                    float(decision_threshold),
                    float(scenario_reduced_exposure),
                    reduce_when=reduce_when,
                ),
            })
    overlay_comparison_path.write_text(json.dumps({
        "mode": "meta_ensemble_overlay_model_comparison_research_only",
        "overlay_decision_rule": decision_rule,
        "models": [
            {
                "model_type": _meta_ensemble_name(selected_meta_model_type),
                "scenarios": scenarios,
            }
        ],
        "research_only": True,
        "trading_impact": "none",
    }, indent=2), encoding="utf-8")

    leaderboard_path = output_dir / "leaderboard.json"
    leaderboard_markdown_path = output_dir / "leaderboard.md"
    write_leaderboard(
        leaderboard_path,
        leaderboard_markdown_path,
        source_dirs,
        metrics,
        leaderboard_calibration,
        overlay,
        meta_model_name=_meta_ensemble_name(selected_meta_model_type),
        meta_walk_forward=walk_forward.get("summary", {}),
        promotion_gates=promotion_gates,
        meta_selections=meta_model_comparison.get("selections", {}),
    )
    return MetaEnsembleResult(
        output_dir=output_dir,
        meta_dataset_path=dataset_path,
        audit_path=audit_path,
        metrics_path=metrics_path,
        walk_forward_metrics_path=walk_forward_path,
        probability_calibration_path=calibration_path,
        calibrated_probability_calibration_path=calibrated_probability_calibration_path,
        holdout_shadow_overlay_path=holdout_overlay_path,
        threshold_sweep_path=threshold_sweep_path,
        meta_model_comparison_path=meta_model_comparison_path,
        promotion_gates_path=promotion_gates_path,
        overlay_model_comparison_path=overlay_comparison_path,
        leaderboard_path=leaderboard_path,
        leaderboard_markdown_path=leaderboard_markdown_path,
    )


def build_meta_dataset_rows(
    expanded_rows: list[dict[str, str]],
    source_predictions: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = []
    missing_counts = {model: 0 for model in source_predictions}
    auxiliary_prediction_columns_by_model: dict[str, list[str]] = {}
    ignored_leakage_columns_by_model: dict[str, list[str]] = {}
    duplicate_feature_ids = len(expanded_rows) - len({row["feature_id"] for row in expanded_rows})
    expanded_by_id = {row["feature_id"]: row for row in expanded_rows}
    all_feature_ids = sorted(
        set.intersection(
            set(expanded_by_id),
            *(set(predictions) for predictions in source_predictions.values()),
        )
    )
    for feature_id in all_feature_ids:
        expanded = expanded_by_id[feature_id]
        split_values = {
            predictions[feature_id].get("split", "")
            for predictions in source_predictions.values()
        }
        split = "holdout" if "holdout" in split_values else "out_of_fold"
        row = {
            "feature_id": feature_id,
            "rebalance_date": expanded.get("rebalance_date", expanded.get("feature_date", "")),
            "variant_id": expanded.get("variant_id", ""),
            "split": split,
            "fold": next(iter(source_predictions.values()))[feature_id].get("fold", ""),
            "actual_label": next(iter(source_predictions.values()))[feature_id]["actual_label"],
        }
        for model, predictions in source_predictions.items():
            prediction = predictions.get(feature_id)
            if prediction is None:
                missing_counts[model] += 1
                continue
            row[f"{model}_raw_probability"] = prediction["raw_probability"]
            row[f"{model}_calibrated_probability"] = (
                prediction.get("calibrated_probability") or prediction["raw_probability"]
            )
            auxiliary_features, ignored_columns = _source_prediction_feature_values(
                model,
                prediction,
            )
            auxiliary_prediction_columns_by_model.setdefault(model, [])
            ignored_leakage_columns_by_model.setdefault(model, [])
            for name, value in auxiliary_features.items():
                row[name] = value
                if name not in auxiliary_prediction_columns_by_model[model]:
                    auxiliary_prediction_columns_by_model[model].append(name)
            for name in ignored_columns:
                if name not in ignored_leakage_columns_by_model[model]:
                    ignored_leakage_columns_by_model[model].append(name)
        for name in (
            "variant_top_n",
            "variant_universe_symbol_count",
            "breadth_above_sma_200",
            "spy_realized_volatility_21d",
            "spy_max_drawdown_63d",
            "recent_champion_excess_return",
            "replacements",
            "champion_return_next_period",
        ):
            row[name] = expanded.get(name, "0")
        rows.append(row)
    audit = {
        "row_count": len(rows),
        "feature_count": len(_feature_values(rows[0])) if rows else 0,
        "date_range": [rows[0]["rebalance_date"], rows[-1]["rebalance_date"]] if rows else None,
        "class_balance": _class_balance(rows),
        "source_model_count": len(source_predictions),
        "source_dataset_hash": _single_source_dataset_hash(source_predictions),
        "source_dataset_row_counts_by_model": _source_prediction_field_values(
            source_predictions,
            "source_dataset_row_count",
        ),
        "source_artifact_generated_at_by_model": _source_prediction_field_values(
            source_predictions,
            "artifact_generated_at",
        ),
        "missing_prediction_counts_by_model": missing_counts,
        "auxiliary_prediction_columns_by_model": {
            model: sorted(columns)
            for model, columns in auxiliary_prediction_columns_by_model.items()
        },
        "ignored_leakage_columns_by_model": {
            model: sorted(columns)
            for model, columns in ignored_leakage_columns_by_model.items()
        },
        "duplicate_feature_id_count": duplicate_feature_ids,
        "same_date_leakage_check": _same_date_leakage_check(rows),
        "meta_training_uses_in_sample_base_predictions": False,
    }
    return rows, audit


def _load_source_predictions(source_dirs: list[Path]) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    sources = {}
    warnings = []
    dataset_hashes: dict[str, str] = {}
    for source_dir in source_dirs:
        path = source_dir / "prediction_artifacts.csv"
        if not path.exists():
            warnings.append(f"missing_prediction_artifact:{path}")
            continue
        metadata_path = source_dir / "prediction_artifacts.json"
        metadata = _read_prediction_artifact_metadata(metadata_path)
        dataset_hash = str(metadata.get("dataset_hash") or metadata.get("data_hash") or "")
        if not dataset_hash:
            raise RuntimeError(
                "Prediction artifact metadata is missing dataset_hash: "
                f"{metadata_path}. Rerun ml-research for {source_dir}."
            )
        rows = _read_csv(path)
        if not rows:
            continue
        model_type = rows[0]["model_type"]
        row_hashes = {
            row.get("dataset_hash", "")
            for row in rows
            if row.get("split") in {"out_of_fold", "holdout"}
        }
        if "" in row_hashes:
            raise RuntimeError(
                "Prediction artifact CSV is missing dataset_hash values: "
                f"{path}. Rerun ml-research so prediction_artifacts.csv is regenerated."
            )
        if row_hashes and row_hashes != {dataset_hash}:
            raise RuntimeError(
                "Prediction artifact CSV dataset_hash does not match metadata "
                f"for {source_dir}: csv={sorted(row_hashes)} metadata={dataset_hash}"
            )
        dataset_hashes[model_type] = dataset_hash
        sources[model_type] = {
            row["feature_id"]: {
                **row,
                "dataset_hash": dataset_hash,
                "source_dataset_row_count": str(
                    metadata.get("source_dataset_row_count", row.get("source_dataset_row_count", ""))
                ),
                "artifact_generated_at": str(
                    metadata.get("generated_at", row.get("generated_at", ""))
                ),
                "artifact_train_sample_count": str(
                    metadata.get("train_sample_count", row.get("train_sample_count", ""))
                ),
                "artifact_test_sample_count": str(
                    metadata.get("test_sample_count", row.get("test_sample_count", ""))
                ),
            }
            for row in rows
            if row.get("split") in {"out_of_fold", "holdout"}
        }
    _validate_source_dataset_hashes(dataset_hashes)
    return sources, warnings


def _read_prediction_artifact_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing prediction artifact metadata: {path}. "
            "Rerun ml-research so prediction_artifacts.json is regenerated."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source_dataset_hashes(dataset_hashes: dict[str, str]) -> None:
    unique_hashes = {value for value in dataset_hashes.values() if value}
    if len(unique_hashes) > 1:
        details = ", ".join(
            f"{model}={dataset_hash}"
            for model, dataset_hash in sorted(dataset_hashes.items())
        )
        raise RuntimeError(
            "Meta ensemble prediction artifacts were generated from different "
            f"dataset hashes. Rerun ml-research for all source models. {details}"
        )


def _single_source_dataset_hash(
    source_predictions: dict[str, dict[str, dict[str, str]]],
) -> str | None:
    hashes = {
        row.get("dataset_hash", "")
        for predictions in source_predictions.values()
        for row in predictions.values()
        if row.get("dataset_hash")
    }
    return next(iter(hashes)) if len(hashes) == 1 else None


def _source_prediction_field_values(
    source_predictions: dict[str, dict[str, dict[str, str]]],
    field_name: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for model, predictions in source_predictions.items():
        for row in predictions.values():
            value = row.get(field_name)
            if value:
                values[model] = value
                break
    return values


def _fit_meta_model(
    model_type: str,
    features: list[dict[str, float]],
    labels: list[int],
    random_seed: int,
) -> MetaLearnerModel:
    normalized = _normalize_meta_model_type(model_type)
    feature_names = sorted(features[0]) if features else []
    if not features:
        return MetaLearnerModel(normalized, feature_names, constant_probability=0.5)
    if len(set(labels)) < 2:
        probability = sum(labels) / len(labels) if labels else 0.5
        return MetaLearnerModel(
            normalized,
            feature_names,
            constant_probability=float(probability),
        )

    matrix = _feature_matrix(features, feature_names)
    if normalized in {"logistic_regression", "ridge_logistic"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=100.0 if normalized == "logistic_regression" else 1.0,
                max_iter=5_000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_seed,
            ),
        )
    elif normalized == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=random_seed,
        )
    elif normalized == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=12,
            random_state=random_seed,
        )
    elif normalized == "lightgbm":
        _ensure_lightgbm_runtime_available()
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM meta learner requested but lightgbm is not installed") from exc
        estimator = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            verbose=-1,
        )
    else:
        raise RuntimeError(f"Unsupported ml.meta_model_type '{model_type}'")

    estimator.fit(matrix, labels)
    return MetaLearnerModel(normalized, feature_names, estimator=estimator)


def _ensure_lightgbm_runtime_available() -> None:
    if platform.system() != "Darwin":
        return
    if ctypes.util.find_library("omp") or ctypes.util.find_library("libomp"):
        return
    raise RuntimeError(
        "LightGBM meta learner requested but macOS libomp is not available. "
        "Install it with 'brew install libomp' or remove lightgbm from "
        "ml.meta_model_types."
    )


def _feature_matrix(
    features: list[dict[str, float]],
    feature_names: list[str],
) -> list[list[float]]:
    return [[float(row.get(name, 0.0)) for name in feature_names] for row in features]


def _normalize_meta_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    aliases = {
        "logistic": "logistic_regression",
        "meta_ensemble_logistic": "logistic_regression",
        "ridge": "ridge_logistic",
        "gbm": "gradient_boosting",
        "light_gradient_boosting": "lightgbm",
    }
    return aliases.get(normalized, normalized)


def _meta_ensemble_name(model_type: str) -> str:
    normalized = _normalize_meta_model_type(model_type)
    if normalized == "logistic_regression":
        return "meta_ensemble_logistic"
    return f"meta_ensemble_{normalized}"


def _meta_model_types(
    ml_config: dict[str, Any],
    selected_meta_model_type: str,
) -> list[str]:
    configured = ml_config.get(
        "meta_model_types",
        [
            "logistic_regression",
            "ridge_logistic",
            "random_forest",
            "gradient_boosting",
            "lightgbm",
        ],
    )
    model_types = [_normalize_meta_model_type(value) for value in configured]
    selected = _normalize_meta_model_type(selected_meta_model_type)
    if selected not in model_types:
        model_types.insert(0, selected)
    return list(dict.fromkeys(model_types))


def _feature_values(row: dict[str, str]) -> dict[str, float]:
    ignored = {"feature_id", "rebalance_date", "variant_id", "split", "fold", "actual_label"}
    values = {}
    for name, value in row.items():
        if name in ignored:
            continue
        if _is_leakage_column(name):
            continue
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _source_prediction_feature_values(
    model: str,
    prediction: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    ignored_columns: list[str] = []
    for name, value in prediction.items():
        if name.startswith("predicted_"):
            if _is_allowed_source_prediction_feature(name):
                values[f"{model}_{name}"] = value
            else:
                ignored_columns.append(name)
            continue
        if name.startswith("actual_") or _is_leakage_column(name):
            ignored_columns.append(name)
    return values, ignored_columns


def _is_allowed_source_prediction_feature(name: str) -> bool:
    return (
        name.startswith("predicted_")
        and not _is_leakage_column(name)
        and name not in {"predicted_class", "predicted_label"}
    )


def _is_leakage_column(name: str) -> bool:
    normalized = name.lower()
    if normalized.startswith("actual_"):
        return True
    if normalized.startswith("predicted_") or "_predicted_" in normalized:
        return False
    return any(
        token in normalized
        for token in (
            "future_",
            "forward_return_",
            "max_adverse_excursion",
            "max_favourable_excursion",
            "label_start",
            "label_end",
        )
    ) and not normalized.startswith("predicted_")


def _compare_meta_learners(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    model_types: list[str],
    threshold: float,
    reduced_exposure: float,
    reduce_when: str,
    random_seed: int,
    calibration_bin_count: int,
    all_rows: list[dict[str, str]] | None = None,
    walk_forward_folds: int = 3,
    promotion_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_features = [_feature_values(row) for row in train_rows]
    train_labels = [int(row["actual_label"]) for row in train_rows]
    holdout_features = [_feature_values(row) for row in holdout_rows]
    holdout_labels = [int(row["actual_label"]) for row in holdout_rows]
    models = []
    for model_type in model_types:
        normalized = _normalize_meta_model_type(model_type)
        try:
            model = _fit_meta_model(
                normalized,
                train_features,
                train_labels,
                random_seed=random_seed,
            )
            train_probabilities = model.predict_proba(train_features)
            probabilities = model.predict_proba(holdout_features)
            predictions = [int(value >= threshold) for value in probabilities]
            calibration_comparison = compare_calibration_methods(
                train_labels,
                train_probabilities,
                holdout_labels,
                probabilities,
                bin_count=calibration_bin_count,
            )
            calibration = _best_calibration_summary(calibration_comparison)
            overlay = _overlay_summary(
                holdout_rows,
                probabilities,
                threshold,
                reduced_exposure,
                reduce_when=reduce_when,
            )
            walk_forward = _walk_forward_meta_evaluation(
                all_rows or train_rows + holdout_rows,
                model_type=normalized,
                fold_count=walk_forward_folds,
                threshold=threshold,
                reduced_exposure=reduced_exposure,
                reduce_when=reduce_when,
                random_seed=random_seed,
                calibration_bin_count=calibration_bin_count,
            )
            promotion_gates = _promotion_gate_report(
                metrics=classification_metrics(holdout_labels, predictions),
                calibration=calibration,
                overlay=overlay,
                walk_forward=walk_forward,
                config=promotion_config or {},
            )
            models.append({
                "model_type": normalized,
                "leaderboard_model": _meta_ensemble_name(normalized),
                "available": True,
                "metrics": classification_metrics(holdout_labels, predictions),
                "calibration": calibration,
                "best_calibration_method": calibration.get("method"),
                "overlay": overlay,
                "walk_forward_summary": walk_forward.get("summary", {}),
                "promotion_gates": promotion_gates,
                "promotion_gate_score": _promotion_gate_score(
                    metrics=classification_metrics(holdout_labels, predictions),
                    calibration=calibration,
                    overlay=overlay,
                    walk_forward=walk_forward.get("summary", {}),
                    promotion_gates=promotion_gates,
                ),
            })
        except Exception as exc:
            if normalized == "lightgbm":
                models.append({
                    "model_type": normalized,
                    "leaderboard_model": _meta_ensemble_name(normalized),
                    "available": False,
                    "reason": str(exc),
                })
                continue
            raise
    available_models = [row for row in models if row.get("available")]
    classifier_ranking = sorted(
        available_models,
        key=lambda row: _classifier_rank_key(row),
    )
    calibration_ranking = sorted(
        available_models,
        key=lambda row: _calibration_rank_key(row),
    )
    overlay_ranking = sorted(
        available_models,
        key=lambda row: _overlay_rank_key(row),
    )
    selections = _meta_selection_summary(
        classifier_ranking,
        calibration_ranking,
        overlay_ranking,
    )
    return {
        "mode": "meta_learner_comparison_research_only",
        "models": models,
        "ranked_model_types": [row["model_type"] for row in classifier_ranking],
        "classifier_ranking": _selection_ranking_rows(classifier_ranking),
        "calibration_ranking": _selection_ranking_rows(calibration_ranking),
        "promotion_gate_ranking": _selection_ranking_rows(overlay_ranking),
        "selections": selections,
        "research_only": True,
        "trading_impact": "none",
    }


def _classifier_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(row.get("metrics", {}).get("balanced_accuracy") or 0.0),
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
        -float(row.get("overlay", {}).get("return_delta") or 0.0),
    )


def _calibration_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
        float(row.get("calibration", {}).get("expected_calibration_error") or float("inf")),
        -float(row.get("metrics", {}).get("balanced_accuracy") or 0.0),
    )


def _overlay_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(row.get("promotion_gate_score") or 0.0),
        -float(row.get("walk_forward_summary", {}).get("balanced_accuracy") or 0.0),
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
    )


def _meta_selection_summary(
    classifier_ranking: list[dict[str, Any]],
    calibration_ranking: list[dict[str, Any]],
    overlay_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    selections = {}
    if classifier_ranking:
        selections["selected_classifier"] = _selection_payload(
            classifier_ranking[0],
            role="selected_classifier",
            reason=(
                "highest holdout balanced accuracy; not automatically selected "
                "as trading overlay"
            ),
        )
    if calibration_ranking:
        selections["selected_calibrated"] = _selection_payload(
            calibration_ranking[0],
            role="selected_calibrated",
            reason="lowest Brier score with ECE as tie-breaker",
        )
    if overlay_ranking:
        selections["selected_overlay"] = _selection_payload(
            overlay_ranking[0],
            role="selected_overlay",
            reason=(
                "highest promotion-gate utility balancing walk-forward accuracy, "
                "Brier/ECE, overlay return delta, drawdown impact, turnover, "
                "and reduced-exposure days"
            ),
        )
    return selections


def _selection_payload(
    row: dict[str, Any],
    role: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "selection_role": role,
        "selected_model": row.get("leaderboard_model"),
        "model_type": row.get("model_type"),
        "selection_reason": reason,
        "metrics": row.get("metrics", {}),
        "calibration": row.get("calibration", {}),
        "overlay": row.get("overlay", {}),
        "walk_forward_summary": row.get("walk_forward_summary", {}),
        "promotion_gates": row.get("promotion_gates", {}),
        "promotion_gate_score": row.get("promotion_gate_score"),
    }


def _selection_ranking_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "model_type": row.get("model_type"),
            "leaderboard_model": row.get("leaderboard_model"),
            "holdout_balanced_accuracy": row.get("metrics", {}).get(
                "balanced_accuracy"
            ),
            "walk_forward_balanced_accuracy": row.get(
                "walk_forward_summary", {}
            ).get("balanced_accuracy"),
            "brier_score": row.get("calibration", {}).get("brier_score"),
            "expected_calibration_error": row.get("calibration", {}).get(
                "expected_calibration_error"
            ),
            "overlay_return_delta": row.get("overlay", {}).get("return_delta"),
            "max_drawdown_delta": row.get("overlay", {}).get("max_drawdown_delta"),
            "turnover": row.get("overlay", {}).get("overlay_turnover"),
            "reduced_exposure_days": row.get("overlay", {}).get(
                "reduced_exposure_days"
            ),
            "promotion_gate_score": row.get("promotion_gate_score"),
            "promotion_candidate": row.get("promotion_gates", {}).get(
                "promotion_candidate"
            ),
        }
        for index, row in enumerate(rows)
    ]


def _promotion_gate_score(
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    overlay: dict[str, Any],
    walk_forward: dict[str, Any],
    promotion_gates: dict[str, Any],
) -> float:
    gate_checks = promotion_gates.get("checks", {})
    passed_gate_count = sum(
        1 for check in gate_checks.values()
        if isinstance(check, dict) and check.get("passed")
    )
    finite_bonus = 0.25 if gate_checks.get("finite_sanity_check", {}).get("passed") else -1.0
    return (
        2.0 * float(walk_forward.get("balanced_accuracy") or 0.0)
        + 0.5 * float(metrics.get("balanced_accuracy") or 0.0)
        - 1.5 * float(calibration.get("brier_score") or 1.0)
        - 1.0 * float(calibration.get("expected_calibration_error") or 1.0)
        + 2.0 * float(overlay.get("return_delta") or 0.0)
        + 1.0 * float(overlay.get("max_drawdown_delta") or 0.0)
        - 0.01 * float(overlay.get("overlay_turnover") or 0.0)
        - 0.001 * float(overlay.get("reduced_exposure_days") or 0.0)
        + 0.05 * passed_gate_count
        + finite_bonus
    )


def _walk_forward_meta_evaluation(
    rows: list[dict[str, str]],
    model_type: str,
    fold_count: int,
    threshold: float,
    reduced_exposure: float,
    reduce_when: str,
    random_seed: int,
    calibration_bin_count: int,
) -> dict[str, Any]:
    unique_dates = sorted({row["rebalance_date"] for row in rows if row.get("rebalance_date")})
    if len(unique_dates) < 2:
        return {
            "validation": "chronological_meta_walk_forward_grouped_by_rebalance_date",
            "fold_count": 0,
            "folds": [],
            "summary": {},
            "research_only": True,
            "trading_impact": "none",
        }
    effective_fold_count = min(max(1, int(fold_count)), len(unique_dates) - 1)
    candidate_test_dates = unique_dates[1:]
    chunk_size = max(1, math.ceil(len(candidate_test_dates) / effective_fold_count))
    folds = []
    for fold_index in range(effective_fold_count):
        test_dates = candidate_test_dates[
            fold_index * chunk_size : (fold_index + 1) * chunk_size
        ]
        if not test_dates:
            continue
        first_test_date = test_dates[0]
        train_dates = [date for date in unique_dates if date < first_test_date]
        train_rows = [row for row in rows if row.get("rebalance_date") in train_dates]
        test_rows = [row for row in rows if row.get("rebalance_date") in set(test_dates)]
        if not train_rows or not test_rows:
            continue
        train_features = [_feature_values(row) for row in train_rows]
        train_labels = [int(row["actual_label"]) for row in train_rows]
        test_features = [_feature_values(row) for row in test_rows]
        test_labels = [int(row["actual_label"]) for row in test_rows]
        model = _fit_meta_model(
            model_type,
            train_features,
            train_labels,
            random_seed=random_seed + fold_index,
        )
        probabilities = model.predict_proba(test_features)
        predictions = [int(value >= threshold) for value in probabilities]
        calibration = build_probability_calibration(
            test_labels,
            probabilities,
            bin_count=calibration_bin_count,
        )
        overlay = _overlay_summary(
            _with_split(test_rows, "test"),
            probabilities,
            threshold,
            reduced_exposure,
            reduce_when=reduce_when,
        )
        folds.append({
            "fold": fold_index + 1,
            "train_start_date": min(train_dates),
            "train_end_date": max(train_dates),
            "test_start_date": min(test_dates),
            "test_end_date": max(test_dates),
            "train_sample_count": len(train_rows),
            "test_sample_count": len(test_rows),
            "metrics": classification_metrics(test_labels, predictions),
            "calibration": calibration,
            "overlay": overlay,
        })
    summary = _walk_forward_summary(folds)
    return {
        "validation": "chronological_meta_walk_forward_grouped_by_rebalance_date",
        "model_type": _normalize_meta_model_type(model_type),
        "fold_count": len(folds),
        "folds": folds,
        "summary": summary,
        "research_only": True,
        "trading_impact": "none",
    }


def _walk_forward_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    def average(path: tuple[str, ...]) -> float | None:
        values = []
        for fold in folds:
            value: Any = fold
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return sum(values) / len(values) if values else None

    return {
        "fold_count": len(folds),
        "balanced_accuracy": average(("metrics", "balanced_accuracy")),
        "accuracy": average(("metrics", "accuracy")),
        "brier_score": average(("calibration", "brier_score")),
        "expected_calibration_error": average(
            ("calibration", "expected_calibration_error")
        ),
        "overlay_return_delta": average(("overlay", "return_delta")),
        "overlay_max_drawdown_improvement": average(("overlay", "max_drawdown_delta")),
        "overlay_turnover": average(("overlay", "overlay_turnover")),
        "reduced_exposure_days": average(("overlay", "reduced_exposure_days")),
    }


def _threshold_sweep(
    rows: list[dict[str, str]],
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
    reduced_exposures: list[float],
    reduce_when: str,
) -> dict[str, Any]:
    scenarios = []
    for threshold in thresholds:
        for reduced_exposure in reduced_exposures:
            threshold_value = float(threshold)
            predictions = [int(value >= threshold_value) for value in probabilities]
            overlay = _overlay_summary(
                rows,
                probabilities,
                threshold_value,
                float(reduced_exposure),
                reduce_when=reduce_when,
            )
            scenarios.append({
                "decision_threshold": threshold_value,
                "reduced_exposure": float(reduced_exposure),
                "metrics": classification_metrics(labels, predictions),
                "overlay": overlay,
                "finite_sanity_check": _finite_sanity_check(overlay),
            })
    ranked = sorted(
        scenarios,
        key=lambda row: (
            -float(row["metrics"].get("balanced_accuracy") or 0.0),
            -float(row["overlay"].get("return_delta") or 0.0),
        ),
    )
    return {
        "mode": "meta_ensemble_threshold_sweep_research_only",
        "scenarios": scenarios,
        "best": ranked[0] if ranked else None,
        "research_only": True,
        "trading_impact": "none",
    }


def _promotion_gate_report(
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    overlay: dict[str, Any],
    walk_forward: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = walk_forward.get("summary", {})
    thresholds = {
        "min_walk_forward_balanced_accuracy": float(
            config.get("promotion_min_walk_forward_balanced_accuracy", 0.50)
        ),
        "max_brier_score": float(config.get("promotion_max_brier_score", 0.25)),
        "max_expected_calibration_error": float(
            config.get("promotion_max_expected_calibration_error", 0.10)
        ),
        "min_overlay_return_delta": float(
            config.get("promotion_min_overlay_return_delta", 0.0)
        ),
        "min_overlay_sample_count": int(
            config.get("promotion_min_overlay_sample_count", 50)
        ),
        "min_max_drawdown_delta": float(
            config.get("promotion_min_max_drawdown_delta", 0.0)
        ),
    }
    checks = {
        "finite_sanity_check": _finite_sanity_check(overlay),
        "walk_forward_balanced_accuracy": _passes_minimum(
            summary.get("balanced_accuracy"),
            thresholds["min_walk_forward_balanced_accuracy"],
        ),
        "brier_score": _passes_maximum(
            calibration.get("brier_score"),
            thresholds["max_brier_score"],
        ),
        "expected_calibration_error": _passes_maximum(
            calibration.get("expected_calibration_error"),
            thresholds["max_expected_calibration_error"],
        ),
        "overlay_return_delta": _passes_minimum(
            overlay.get("return_delta"),
            thresholds["min_overlay_return_delta"],
        ),
        "overlay_sample_count": _passes_minimum(
            overlay.get("overlay_sample_count"),
            thresholds["min_overlay_sample_count"],
        ),
        "max_drawdown_delta": _passes_minimum(
            overlay.get("max_drawdown_delta"),
            thresholds["min_max_drawdown_delta"],
        ),
    }
    passed = all(item.get("passed") for item in checks.values())
    return {
        "promotion_candidate": passed,
        "checks": checks,
        "thresholds": thresholds,
        "observed": {
            "holdout_balanced_accuracy": metrics.get("balanced_accuracy"),
            "walk_forward_balanced_accuracy": summary.get("balanced_accuracy"),
            "brier_score": calibration.get("brier_score"),
            "expected_calibration_error": calibration.get(
                "expected_calibration_error"
            ),
            "overlay_return_delta": overlay.get("return_delta"),
            "overlay_max_drawdown_improvement": overlay.get("max_drawdown_delta"),
            "turnover": overlay.get("overlay_turnover"),
            "reduced_exposure_days": overlay.get("reduced_exposure_days"),
        },
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }


def _passes_minimum(value: Any, minimum: float) -> dict[str, Any]:
    numeric = _finite_or_none(value)
    return {
        "value": numeric,
        "minimum": minimum,
        "passed": numeric is not None and numeric >= minimum,
    }


def _passes_maximum(value: Any, maximum: float) -> dict[str, Any]:
    numeric = _finite_or_none(value)
    return {
        "value": numeric,
        "maximum": maximum,
        "passed": numeric is not None and numeric <= maximum,
    }


def _finite_sanity_check(payload: dict[str, Any]) -> dict[str, Any]:
    checked_fields = [
        "overlay_baseline_return",
        "overlay_adjusted_return",
        "return_delta",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "max_drawdown_delta",
        "overlay_turnover",
    ]
    invalid = [
        name for name in checked_fields
        if payload.get(name) is not None and _finite_or_none(payload.get(name)) is None
    ]
    return {"passed": not invalid, "invalid_fields": invalid}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _with_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [{**row, "split": split} for row in rows]


def _best_calibration_summary(calibration_comparison: dict[str, Any]) -> dict[str, Any]:
    method = calibration_comparison.get("best_method_by_brier", "raw")
    payload = calibration_comparison.get("methods", {}).get(method, {})
    calibration = dict(payload.get("calibration", {}))
    calibration["method"] = method
    return calibration


def _overlay_summary(
    rows: list[dict[str, str]],
    probabilities: list[float],
    threshold: float,
    reduced_exposure: float,
    reduce_when: str = "above_or_equal_threshold",
) -> dict[str, float | int]:
    if len(rows) != len(probabilities):
        raise ValueError("Overlay rows and probabilities must have the same length")
    if not math.isfinite(float(threshold)):
        raise ValueError("Overlay threshold must be finite")
    if not math.isfinite(float(reduced_exposure)) or not 0 <= reduced_exposure <= 1:
        raise ValueError("Reduced exposure must be a finite decimal between 0 and 1")

    pairs = _holdout_overlay_pairs(rows, probabilities)
    if not pairs:
        return {
            "overlay_start_date": None,
            "overlay_end_date": None,
            "overlay_sample_count": 0,
            "overlay_evaluated_dates": 0,
            "base_total_return": 0.0,
            "overlay_total_return": 0.0,
            "overlay_baseline_return": 0.0,
            "overlay_adjusted_return": 0.0,
            "return_delta": 0.0,
            "base_compounded_return": 0.0,
            "overlay_compounded_return": 0.0,
            "base_max_drawdown": 0.0,
            "overlay_max_drawdown": 0.0,
            "max_drawdown_delta": 0.0,
            "reduced_exposure_days": 0,
            "overlay_turnover": 0.0,
            "aggregation": "mean_by_rebalance_date_not_compounded",
        }

    by_date: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, probability in pairs:
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Overlay rows must include rebalance_date or date")
        by_date.setdefault(date, []).append((row, probability))

    date_baseline_returns = []
    date_adjusted_returns = []
    reduced_days = 0
    overlay_turnover = 0.0
    active_reduced = False
    for date in sorted(by_date):
        base_returns = []
        adjusted_returns = []
        date_reduced = False
        for row, probability in by_date[date]:
            _validate_probability(probability)
            base_return = _period_return(row)
            multiplier = (
                reduced_exposure
                if should_reduce_exposure(probability, threshold, reduce_when)
                else 1.0
            )
            date_reduced = date_reduced or multiplier < 1.0
            base_returns.append(base_return)
            adjusted_returns.append(base_return * multiplier)
        date_baseline_returns.append(mean(base_returns))
        date_adjusted_returns.append(mean(adjusted_returns))
        reduced_days += int(date_reduced)
        if date_reduced != active_reduced:
            overlay_turnover += abs((reduced_exposure if date_reduced else 1.0) - (reduced_exposure if active_reduced else 1.0))
            active_reduced = date_reduced

    baseline_return = mean(date_baseline_returns)
    adjusted_return = mean(date_adjusted_returns)
    _validate_finite("overlay_baseline_return", baseline_return)
    _validate_finite("overlay_adjusted_return", adjusted_return)
    return_delta = adjusted_return - baseline_return
    _validate_finite("overlay_return_delta", return_delta)
    base_curve = _equity_curve(date_baseline_returns)
    overlay_curve = _equity_curve(date_adjusted_returns)
    base_max_drawdown = _max_drawdown(base_curve)
    overlay_max_drawdown = _max_drawdown(overlay_curve)
    max_drawdown_delta = overlay_max_drawdown - base_max_drawdown
    for name, value in (
        ("base_compounded_return", base_curve[-1] - 1.0),
        ("overlay_compounded_return", overlay_curve[-1] - 1.0),
        ("base_max_drawdown", base_max_drawdown),
        ("overlay_max_drawdown", overlay_max_drawdown),
        ("max_drawdown_delta", max_drawdown_delta),
    ):
        _validate_finite(name, value)

    return {
        "overlay_start_date": min(by_date),
        "overlay_end_date": max(by_date),
        "overlay_sample_count": len(pairs),
        "overlay_evaluated_dates": len(by_date),
        "base_total_return": baseline_return,
        "overlay_total_return": adjusted_return,
        "overlay_baseline_return": baseline_return,
        "overlay_adjusted_return": adjusted_return,
        "return_delta": return_delta,
        "base_compounded_return": base_curve[-1] - 1.0,
        "overlay_compounded_return": overlay_curve[-1] - 1.0,
        "base_max_drawdown": base_max_drawdown,
        "overlay_max_drawdown": overlay_max_drawdown,
        "max_drawdown_delta": max_drawdown_delta,
        "reduced_exposure_days": reduced_days,
        "overlay_turnover": float(overlay_turnover),
        "aggregation": "mean_by_rebalance_date_not_compounded",
    }


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    values = [equity]
    for value in returns:
        _validate_finite("period return", value)
        if value <= -1.0:
            raise ValueError("period return would zero or invert equity")
        equity *= 1.0 + value
        _validate_finite("overlay equity", equity)
        values.append(equity)
    return values


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            raise ValueError("baseline equity denominator must be positive")
        drawdown = min(drawdown, (value / peak) - 1.0)
    return drawdown


def _holdout_overlay_pairs(
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> list[tuple[dict[str, str], float]]:
    pairs = list(zip(rows, probabilities))
    has_split = any(row.get("split") for row, _ in pairs)
    if not has_split:
        return pairs
    return [
        (row, probability)
        for row, probability in pairs
        if row.get("split") in {"holdout", "test"}
    ]


def _period_return(row: dict[str, str]) -> float:
    value = float(row.get("champion_return_next_period", 0.0) or 0.0)
    _validate_finite("champion_return_next_period", value)
    if abs(value) > _MAX_ABS_PERIOD_RETURN:
        raise ValueError(
            "champion_return_next_period must be a decimal return, not a percent"
        )
    if value <= -1.0:
        raise ValueError("champion_return_next_period would zero or invert equity")
    return value


def _validate_probability(probability: float) -> None:
    value = float(probability)
    _validate_finite("overlay probability", value)
    if value < 0 or value > 1:
        raise ValueError("Overlay probabilities must be between 0 and 1")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _same_date_leakage_check(rows: list[dict[str, str]]) -> dict[str, Any]:
    split_by_date: dict[str, set[str]] = {}
    for row in rows:
        split_by_date.setdefault(row["rebalance_date"], set()).add(row["split"])
    leaked = sorted(date for date, splits in split_by_date.items() if len(splits) > 1)
    return {"passed": not leaked, "leaked_dates": leaked}


def _class_balance(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    positives = sum(int(row["actual_label"]) for row in rows)
    total = len(rows)
    return {
        "positive": positives,
        "negative": total - positives,
        "positive_rate": positives / total if total else None,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["feature_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
