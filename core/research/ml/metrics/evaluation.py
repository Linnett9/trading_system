from __future__ import annotations


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | None]:
    if not y_true:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "samples": 0,
        }

    true_positive = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred == 1)
    true_negative = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred == 0)
    false_positive = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 1)
    false_negative = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 0)
    samples = len(y_true)
    accuracy = (true_positive + true_negative) / samples if samples else None
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    true_positive_rate = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    true_negative_rate = true_negative / (true_negative + false_positive) if true_negative + false_positive else None
    balanced_accuracy = (
        (true_positive_rate + true_negative_rate) / 2
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "samples": samples,
    }
