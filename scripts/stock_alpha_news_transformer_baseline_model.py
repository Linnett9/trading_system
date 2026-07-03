from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


TARGET_COLUMN = "reduce_exposure_label"
RETURN_COLUMN = "future_return_20d"
REQUIRED_LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    TARGET_COLUMN,
)
REPORT_FILENAME = "news_transformer_baseline_model_report.json"
PREDICTIONS_FILENAME = "news_transformer_baseline_model_predictions.csv"


def build_baseline_model_report_only(
    *,
    labeled_dataset_path: str | Path,
    split_report_path: str | Path,
    split_assignments_path: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    split_report = _read_json(Path(split_report_path))
    _validate_split_report(split_report)

    labeled_rows = _read_csv(Path(labeled_dataset_path))
    rows_by_event_key = _rows_by_event_key(labeled_rows)
    assignments = _read_csv(Path(split_assignments_path))
    folds = _fold_assignments(assignments, rows_by_event_key)
    leakage_violation_count = _leakage_violation_count(assignments)
    if leakage_violation_count:
        raise ValueError("split assignments contain future leakage")

    fold_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, str]] = []
    for fold_id in sorted(folds):
        fold = folds[fold_id]
        _require_fold_splits(fold_id, fold)
        models = _fit_models(fold["train"])
        model_reports = []
        for model_name, predictor in models.items():
            validation_metrics = _evaluate(fold["validation"], predictor)
            test_metrics = _evaluate(fold["test"], predictor)
            model_reports.append(
                {
                    "model_name": model_name,
                    "validation": validation_metrics,
                    "test": test_metrics,
                }
            )
            prediction_rows.extend(_prediction_rows(fold_id, model_name, "validation", fold["validation"], predictor))
            prediction_rows.extend(_prediction_rows(fold_id, model_name, "test", fold["test"], predictor))
        selected = min(model_reports, key=lambda item: item["validation"]["log_loss"])
        fold_reports.append(
            {
                "fold_id": fold_id,
                "train_rows": len(fold["train"]),
                "validation_rows": len(fold["validation"]),
                "test_rows": len(fold["test"]),
                "train_date_min": _date_min(fold["train"]),
                "train_date_max": _date_max(fold["train"]),
                "validation_date_min": _date_min(fold["validation"]),
                "validation_date_max": _date_max(fold["validation"]),
                "test_date_min": _date_min(fold["test"]),
                "test_date_max": _date_max(fold["test"]),
                "selected_model": selected["model_name"],
                "models": model_reports,
            }
        )

    report = _report(
        labeled_rows=labeled_rows,
        assignments=assignments,
        split_report=split_report,
        folds=fold_reports,
        leakage_violation_count=leakage_violation_count,
    )
    _write_outputs(output_dir_path, report, prediction_rows)
    return report


def _validate_split_report(report: Mapping[str, Any]) -> None:
    guardrails = {
        "next_allowed_step": report.get("next_allowed_step") == "build_news_transformer_baseline_model_report_only",
        "fold_count": int(report.get("fold_count", 0)) > 0,
        "leakage_violation_count": int(report.get("leakage_violation_count", 0)) == 0,
        "overlap_violation_count": int(report.get("overlap_violation_count", 0)) == 0,
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"walk-forward split report is not approved for baseline evaluation: {', '.join(failed)}")


def _rows_by_event_key(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        event_key = str(row.get("event_key", ""))
        if event_key and _has_required_labels(row):
            keyed[event_key] = dict(row)
    return keyed


def _fold_assignments(
    assignments: Sequence[Mapping[str, str]],
    rows_by_event_key: Mapping[str, dict[str, str]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    folds: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: {"train": [], "validation": [], "test": []})
    for assignment in assignments:
        split = str(assignment.get("split", ""))
        if split not in {"train", "validation", "test"}:
            continue
        event_key = str(assignment.get("event_key", ""))
        row = rows_by_event_key.get(event_key)
        if row is None:
            continue
        folds[str(assignment.get("fold_id", ""))][split].append(row)
    return dict(folds)


def _require_fold_splits(fold_id: str, fold: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    missing = [split for split in ("train", "validation", "test") if not fold.get(split)]
    if missing:
        raise ValueError(f"{fold_id} is missing rows for: {', '.join(missing)}")


def _fit_models(rows: Sequence[Mapping[str, str]]) -> dict[str, Callable[[Mapping[str, str]], float]]:
    y = [_target(row) for row in rows]
    global_probability = _smoothed_probability(sum(y), len(y), 0.5, strength=2.0)
    by_symbol = _fit_group_rates(rows, lambda row: str(row.get("symbol", "")), global_probability)
    by_symbol_form = _fit_group_rates(
        rows,
        lambda row: f"{row.get('symbol', '')}|{row.get('form_type', '') or row.get('source_type', '')}",
        global_probability,
    )

    def global_prior(_: Mapping[str, str]) -> float:
        return global_probability

    def symbol_prior(row: Mapping[str, str]) -> float:
        return by_symbol.get(str(row.get("symbol", "")), global_probability)

    def symbol_form_prior(row: Mapping[str, str]) -> float:
        key = f"{row.get('symbol', '')}|{row.get('form_type', '') or row.get('source_type', '')}"
        return by_symbol_form.get(key, symbol_prior(row))

    return {
        "global_prior": global_prior,
        "symbol_prior": symbol_prior,
        "symbol_form_prior": symbol_form_prior,
    }


def _fit_group_rates(
    rows: Sequence[Mapping[str, str]],
    key_fn: Callable[[Mapping[str, str]], str],
    prior: float,
    *,
    strength: float = 20.0,
) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = key_fn(row)
        if not key:
            continue
        counts[key][0] += _target(row)
        counts[key][1] += 1
    return {key: _smoothed_probability(pos, total, prior, strength=strength) for key, (pos, total) in counts.items()}


def _smoothed_probability(positive_count: int, total_count: int, prior: float, *, strength: float) -> float:
    return _clip_probability((positive_count + prior * strength) / (total_count + strength))


def _evaluate(rows: Sequence[Mapping[str, str]], predictor: Callable[[Mapping[str, str]], float]) -> dict[str, Any]:
    y_true = [_target(row) for row in rows]
    probabilities = [_clip_probability(predictor(row)) for row in rows]
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    true_positive = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 1 and predicted == 1)
    true_negative = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 0 and predicted == 0)
    false_positive = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 0 and predicted == 1)
    false_negative = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 1 and predicted == 0)
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    specificity = _safe_div(true_negative, true_negative + false_positive)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    predicted_reduce_returns = [_float(row[RETURN_COLUMN]) for row, predicted in zip(rows, predictions) if predicted == 1]
    predicted_keep_returns = [_float(row[RETURN_COLUMN]) for row, predicted in zip(rows, predictions) if predicted == 0]
    return {
        "rows": len(rows),
        "positive_rate": _safe_div(positive_count, len(y_true)),
        "predicted_positive_rate": _safe_div(sum(predictions), len(predictions)),
        "accuracy": _safe_div(true_positive + true_negative, len(y_true)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": sum((probability - actual) ** 2 for probability, actual in zip(probabilities, y_true)) / len(y_true),
        "log_loss": -sum(
            actual * math.log(probability) + (1 - actual) * math.log(1 - probability)
            for probability, actual in zip(probabilities, y_true)
        ) / len(y_true),
        "mean_future_return_20d_predicted_reduce": _mean(predicted_reduce_returns),
        "mean_future_return_20d_predicted_keep": _mean(predicted_keep_returns),
    }


def _prediction_rows(
    fold_id: str,
    model_name: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    predictor: Callable[[Mapping[str, str]], float],
) -> list[dict[str, str]]:
    return [
        {
            "fold_id": fold_id,
            "model_name": model_name,
            "split": split,
            "event_key": str(row.get("event_key", "")),
            "symbol": str(row.get("symbol", "")),
            "label_date": str(row.get("label_date", "")),
            "target_reduce_exposure": str(_target(row)),
            "predicted_reduce_exposure_probability": f"{_clip_probability(predictor(row)):.10f}",
            "predicted_reduce_exposure_label": str(_clip_probability(predictor(row)) >= 0.5).lower(),
            "future_return_20d": str(row.get(RETURN_COLUMN, "")),
        }
        for row in rows
    ]


def _report(
    *,
    labeled_rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    split_report: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    leakage_violation_count: int,
) -> dict[str, Any]:
    selected = Counter(str(fold["selected_model"]) for fold in folds)
    selected_test_metrics = [next(model["test"] for model in fold["models"] if model["model_name"] == fold["selected_model"]) for fold in folds]
    return {
        "mode": "news_transformer_baseline_model_report_only",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "transformer_training_started": False,
        "rows_total": len(labeled_rows),
        "rows_with_required_labels": sum(1 for row in labeled_rows if _has_required_labels(row)),
        "assignment_rows": len(assignments),
        "fold_count": len(folds),
        "source_split_fold_count": split_report.get("fold_count"),
        "leakage_violation_count": leakage_violation_count,
        "overlap_violation_count": int(split_report.get("overlap_violation_count", 0)),
        "models_evaluated": ["global_prior", "symbol_prior", "symbol_form_prior"],
        "target_column": TARGET_COLUMN,
        "selected_model_counts": dict(sorted(selected.items())),
        "selected_model_average_test_metrics": _average_metrics(selected_test_metrics),
        "folds": list(folds),
        "next_allowed_step": "review_news_transformer_baseline_model_report",
    }


def _average_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = [
        "positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "log_loss",
    ]
    return {key: _mean([float(row[key]) for row in metrics]) for key in keys}


def _leakage_violation_count(assignments: Sequence[Mapping[str, str]]) -> int:
    violations = 0
    for row in assignments:
        available_at = _parse_timestamp(row.get("available_at_timestamp", ""))
        label_date = str(row.get("label_date", ""))[:10]
        if available_at and label_date and label_date < available_at.date().isoformat():
            violations += 1
    return violations


def _has_required_labels(row: Mapping[str, str]) -> bool:
    return all(row.get(column) for column in REQUIRED_LABEL_COLUMNS)


def _target(row: Mapping[str, str]) -> int:
    return 1 if str(row.get(TARGET_COLUMN, "")).lower() == "true" else 0


def _date_min(rows: Sequence[Mapping[str, str]]) -> str:
    return min(str(row["label_date"]) for row in rows)


def _date_max(rows: Sequence[Mapping[str, str]]) -> str:
    return max(str(row["label_date"]) for row in rows)


def _clip_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _float(value: str) -> float:
    return float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(output_dir: Path, report: Mapping[str, Any], predictions: Sequence[Mapping[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / PREDICTIONS_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "fold_id",
            "model_name",
            "split",
            "event_key",
            "symbol",
            "label_date",
            "target_reduce_exposure",
            "predicted_reduce_exposure_probability",
            "predicted_reduce_exposure_label",
            "future_return_20d",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report-only baseline model benchmark for labelled news events.")
    parser.add_argument("--labeled-dataset", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--split-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)

    report = build_baseline_model_report_only(
        labeled_dataset_path=args.labeled_dataset,
        split_report_path=args.split_report,
        split_assignments_path=args.split_assignments,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
