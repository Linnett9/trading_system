from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Mapping


def walk_forward_logistic(
    rows: list[Mapping[str, Any]],
    feature_columns: list[str],
    splits: list[tuple[list[int], list[int]]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    max_train_rows: int,
) -> tuple[dict[str, Any], dict[int, float]]:
    probabilities: dict[int, float] = {}
    fold_reports = []
    for fold_id, (train_index, test_index) in enumerate(splits, start=1):
        if max_train_rows > 0:
            train_index = train_index[-max_train_rows:]
        train = [rows[index] for index in train_index]
        test = [rows[index] for index in test_index]
        labels = [int(row["news_risk_label"]) for row in train]
        if len(set(labels)) < 2:
            continue
        model = fit_logistic(train, feature_columns, learning_rate=learning_rate, epochs=epochs, l2=l2)
        fold_probs = [predict_logistic(model, row) for row in test]
        for index, probability in zip(test_index, fold_probs):
            probabilities[index] = probability
        fold_reports.append(
            {
                "fold_id": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                **classification_metrics([int(row["news_risk_label"]) for row in test], fold_probs),
            }
        )
    if not probabilities:
        raise ValueError("walk-forward logistic regression produced no out-of-sample predictions")
    y_true = [int(rows[index]["news_risk_label"]) for index in probabilities]
    y_prob = [probabilities[index] for index in probabilities]
    return {
        "oos_rows": len(probabilities),
        "folds_completed": len(fold_reports),
        "folds": fold_reports,
        **classification_metrics(y_true, y_prob),
    }, probabilities


def fit_logistic(
    rows: list[Mapping[str, Any]],
    feature_columns: list[str],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, Any]:
    matrix = [[_number(row.get(column)) or 0.0 for column in feature_columns] for row in rows]
    labels = [float(row["news_risk_label"]) for row in rows]
    means = [mean(column) for column in zip(*matrix)]
    stdevs = [pstdev(column) or 1.0 for column in zip(*matrix)]
    weights = [0.0 for _ in feature_columns]
    intercept = 0.0
    for _ in range(max(1, epochs)):
        grad = [0.0 for _ in weights]
        intercept_grad = 0.0
        for features, label in zip(matrix, labels):
            scaled = [(value - means[i]) / stdevs[i] for i, value in enumerate(features)]
            prediction = _sigmoid(intercept + sum(w * x for w, x in zip(weights, scaled)))
            error = prediction - label
            intercept_grad += error
            for i, value in enumerate(scaled):
                grad[i] += error * value
        n = max(len(matrix), 1)
        intercept -= learning_rate * intercept_grad / n
        for i in range(len(weights)):
            weights[i] -= learning_rate * ((grad[i] / n) + l2 * weights[i])
    return {"columns": feature_columns, "means": means, "stdevs": stdevs, "weights": weights, "intercept": intercept}


def predict_logistic(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    total = float(model["intercept"])
    for column, avg, scale, weight in zip(model["columns"], model["means"], model["stdevs"], model["weights"]):
        total += float(weight) * (((_number(row.get(column)) or 0.0) - float(avg)) / float(scale))
    return _sigmoid(total)


def classification_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "brier": 0.0, "roc_auc": 0.0}
    predictions = [1 if value >= 0.5 else 0 for value in y_prob]
    tp = sum(1 for y, p in zip(y_true, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, predictions) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, predictions) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(y_true, predictions) if y == 1 and p == 0)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "brier": mean([(p - y) ** 2 for y, p in zip(y_true, y_prob)]),
        "roc_auc": roc_auc(y_true, y_prob),
        "positive_rate": sum(y_true) / len(y_true),
    }


def roc_auc(y_true: list[int], y_prob: list[float]) -> float:
    positive_count = sum(1 for label in y_true if label == 1)
    negative_count = len(y_true) - positive_count
    if not positive_count or not negative_count:
        return 0.0
    ranked = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (rank + end) / 2.0
        rank_sum += average_rank * sum(1 for _, label in ranked[index:end] if label == 1)
        rank = end + 1
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))
