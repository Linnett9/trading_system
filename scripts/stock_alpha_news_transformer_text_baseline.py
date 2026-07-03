from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


TARGET_COLUMN = "reduce_exposure_label"
RETURN_COLUMN = "future_return_20d"
TEXT_COLUMNS = ("title", "summary_or_text")
REQUIRED_LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    TARGET_COLUMN,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
REPORT_FILENAME = "news_transformer_text_baseline_report.json"
PREDICTIONS_FILENAME = "news_transformer_text_baseline_predictions.csv"


class TextModel(Protocol):
    model_name: str

    def predict_probability(self, row: Mapping[str, str]) -> float:
        ...


def build_text_baseline_report_only(
    *,
    labeled_dataset_path: str | Path,
    text_readiness_report_path: str | Path,
    split_assignments_path: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    readiness_report = _read_json(Path(text_readiness_report_path))
    _validate_readiness_report(readiness_report)

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
        for model in models:
            validation_metrics = _evaluate(fold["validation"], model)
            test_metrics = _evaluate(fold["test"], model)
            model_reports.append(
                {
                    "model_name": model.model_name,
                    "validation": validation_metrics,
                    "test": test_metrics,
                }
            )
            prediction_rows.extend(_prediction_rows(fold_id, "validation", fold["validation"], model))
            prediction_rows.extend(_prediction_rows(fold_id, "test", fold["test"], model))
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
        readiness_report=readiness_report,
        folds=fold_reports,
        leakage_violation_count=leakage_violation_count,
    )
    _write_outputs(output_dir_path, report, prediction_rows)
    return report


def _validate_readiness_report(report: Mapping[str, Any]) -> None:
    guardrails = {
        "mode": report.get("mode") == "news_transformer_text_feature_readiness_report_only",
        "research_only": report.get("research_only") is True,
        "trading_impact": report.get("trading_impact") == "none",
        "transformer_training_started": report.get("transformer_training_started") is False,
        "readiness_available": report.get("readiness_available") is True,
        "blocking_reasons": not list(report.get("blocking_reasons") or []),
        "next_allowed_step": report.get("next_allowed_step") == "build_news_transformer_text_baseline_report_only",
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"text readiness report is not approved for text baseline: {', '.join(failed)}")


def _rows_by_event_key(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        event_key = str(row.get("event_key", ""))
        if event_key and _has_required_labels(row) and _combined_text(row):
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
        row = rows_by_event_key.get(str(assignment.get("event_key", "")))
        if row is None:
            continue
        folds[str(assignment.get("fold_id", ""))][split].append(row)
    return dict(folds)


def _require_fold_splits(fold_id: str, fold: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    missing = [split for split in ("train", "validation", "test") if not fold.get(split)]
    if missing:
        raise ValueError(f"{fold_id} is missing rows for: {', '.join(missing)}")


def _fit_models(rows: Sequence[Mapping[str, str]]) -> list[TextModel]:
    y = [_target(row) for row in rows]
    global_probability = _smoothed_probability(sum(y), len(y), 0.5, strength=2.0)
    return [
        _GlobalPriorTextModel(global_probability),
        _TokenNaiveBayesTextModel.fit(rows, min_token_count=1, alpha=1.0),
        _TokenNaiveBayesTextModel.fit(rows, min_token_count=5, alpha=1.0),
    ]


class _GlobalPriorTextModel:
    model_name = "text_global_prior"

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_probability(self, row: Mapping[str, str]) -> float:
        return self._probability


class _TokenNaiveBayesTextModel:
    def __init__(
        self,
        *,
        model_name: str,
        class_log_prior: dict[int, float],
        token_log_probability: dict[int, dict[str, float]],
        unknown_log_probability: dict[int, float],
    ) -> None:
        self.model_name = model_name
        self._class_log_prior = class_log_prior
        self._token_log_probability = token_log_probability
        self._unknown_log_probability = unknown_log_probability

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, str]], *, min_token_count: int, alpha: float) -> "_TokenNaiveBayesTextModel":
        class_counts = Counter(_target(row) for row in rows)
        raw_counts: Counter[str] = Counter(token for row in rows for token in _tokens(_combined_text(row)))
        vocabulary = {token for token, count in raw_counts.items() if count >= min_token_count}
        token_counts = {0: Counter(), 1: Counter()}
        total_tokens = {0: 0, 1: 0}
        for row in rows:
            target = _target(row)
            for token in _tokens(_combined_text(row)):
                if token not in vocabulary:
                    continue
                token_counts[target][token] += 1
                total_tokens[target] += 1

        class_log_prior = {
            target: math.log(_smoothed_probability(class_counts[target], len(rows), 0.5, strength=2.0))
            for target in (0, 1)
        }
        class_log_prior[0] = math.log(1.0 - math.exp(class_log_prior[1]))
        denominator = {
            target: total_tokens[target] + alpha * max(len(vocabulary), 1)
            for target in (0, 1)
        }
        token_log_probability = {
            target: {
                token: math.log((token_counts[target][token] + alpha) / denominator[target])
                for token in vocabulary
            }
            for target in (0, 1)
        }
        unknown_log_probability = {
            target: math.log(alpha / denominator[target])
            for target in (0, 1)
        }
        return cls(
            model_name=f"text_token_naive_bayes_min{min_token_count}",
            class_log_prior=class_log_prior,
            token_log_probability=token_log_probability,
            unknown_log_probability=unknown_log_probability,
        )

    def predict_probability(self, row: Mapping[str, str]) -> float:
        scores = dict(self._class_log_prior)
        for target in (0, 1):
            for token in _tokens(_combined_text(row)):
                scores[target] += self._token_log_probability[target].get(token, self._unknown_log_probability[target])
        max_score = max(scores.values())
        negative = math.exp(scores[0] - max_score)
        positive = math.exp(scores[1] - max_score)
        return _clip_probability(positive / (negative + positive))


def _evaluate(rows: Sequence[Mapping[str, str]], model: TextModel) -> dict[str, Any]:
    y_true = [_target(row) for row in rows]
    probabilities = [_clip_probability(model.predict_probability(row)) for row in rows]
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    true_positive = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 1 and predicted == 1)
    true_negative = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 0 and predicted == 0)
    false_positive = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 0 and predicted == 1)
    false_negative = sum(1 for actual, predicted in zip(y_true, predictions) if actual == 1 and predicted == 0)
    positive_count = sum(y_true)
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
    split: str,
    rows: Sequence[Mapping[str, str]],
    model: TextModel,
) -> list[dict[str, str]]:
    return [
        {
            "fold_id": fold_id,
            "model_name": model.model_name,
            "split": split,
            "event_key": str(row.get("event_key", "")),
            "symbol": str(row.get("symbol", "")),
            "label_date": str(row.get("label_date", "")),
            "target_reduce_exposure": str(_target(row)),
            "predicted_reduce_exposure_probability": f"{_clip_probability(model.predict_probability(row)):.10f}",
            "predicted_reduce_exposure_label": str(_clip_probability(model.predict_probability(row)) >= 0.5).lower(),
            "future_return_20d": str(row.get(RETURN_COLUMN, "")),
        }
        for row in rows
    ]


def _report(
    *,
    labeled_rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    readiness_report: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    leakage_violation_count: int,
) -> dict[str, Any]:
    selected = Counter(str(fold["selected_model"]) for fold in folds)
    selected_test_metrics = [next(model["test"] for model in fold["models"] if model["model_name"] == fold["selected_model"]) for fold in folds]
    metadata_metrics = readiness_report.get("baseline_review", {}).get("selected_model_average_test_metrics", {}) or {}
    text_metrics = _average_metrics(selected_test_metrics)
    metadata_balanced_accuracy = float(metadata_metrics.get("balanced_accuracy", 0.0))
    text_balanced_accuracy = float(text_metrics["balanced_accuracy"])
    return {
        "mode": "news_transformer_text_baseline_report_only",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "transformer_training_started": False,
        "rows_total": len(labeled_rows),
        "rows_with_required_labels_and_text": sum(1 for row in labeled_rows if _has_required_labels(row) and _combined_text(row)),
        "assignment_rows": len(assignments),
        "fold_count": len(folds),
        "leakage_violation_count": leakage_violation_count,
        "models_evaluated": ["text_global_prior", "text_token_naive_bayes_min1", "text_token_naive_bayes_min5"],
        "selected_model_counts": dict(sorted(selected.items())),
        "selected_model_average_test_metrics": text_metrics,
        "metadata_baseline_average_test_metrics": dict(metadata_metrics),
        "text_vs_metadata_baseline": {
            "balanced_accuracy_delta": text_balanced_accuracy - metadata_balanced_accuracy,
            "beats_metadata_on_balanced_accuracy": text_balanced_accuracy > metadata_balanced_accuracy,
        },
        "text_surface": {
            "summary_or_text_coverage": readiness_report.get("text_columns", {}).get("summary_or_text", {}).get("coverage"),
            "title_coverage": readiness_report.get("text_columns", {}).get("title", {}).get("coverage"),
            "combined_text_vocabulary_size": readiness_report.get("text_columns", {}).get("combined_text", {}).get("vocabulary_size"),
            "warnings": readiness_report.get("warnings", []),
        },
        "folds": list(folds),
        "next_allowed_step": "review_news_transformer_text_baseline_report",
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


def _combined_text(row: Mapping[str, str]) -> str:
    return " ".join(str(row.get(column, "")).strip() for column in TEXT_COLUMNS if str(row.get(column, "")).strip())


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(value)]


def _target(row: Mapping[str, str]) -> int:
    return 1 if str(row.get(TARGET_COLUMN, "")).lower() == "true" else 0


def _date_min(rows: Sequence[Mapping[str, str]]) -> str:
    return min(str(row["label_date"]) for row in rows)


def _date_max(rows: Sequence[Mapping[str, str]]) -> str:
    return max(str(row["label_date"]) for row in rows)


def _smoothed_probability(positive_count: int, total_count: int, prior: float, *, strength: float) -> float:
    return _clip_probability((positive_count + prior * strength) / (total_count + strength))


def _clip_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


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
    parser = argparse.ArgumentParser(description="Build a report-only text baseline for labelled news transformer events.")
    parser.add_argument("--labeled-dataset", required=True)
    parser.add_argument("--text-readiness-report", required=True)
    parser.add_argument("--split-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)

    report = build_text_baseline_report_only(
        labeled_dataset_path=args.labeled_dataset,
        text_readiness_report_path=args.text_readiness_report,
        split_assignments_path=args.split_assignments,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
