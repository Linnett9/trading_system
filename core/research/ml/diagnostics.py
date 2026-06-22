from __future__ import annotations

from core.research.ml.evaluation import classification_metrics


def rolling_base_rate_probabilities(
    train_labels: list[int],
    train_label_end_dates: list[str],
    test_feature_dates: list[str],
    test_labels: list[int],
    test_label_end_dates: list[str],
    lookback_samples: int,
) -> list[float]:
    """Forecast each test row from labels known before its feature date."""
    if len(train_labels) != len(train_label_end_dates):
        raise ValueError("Training labels and end dates must have the same length")
    if len(test_feature_dates) != len(test_labels) or len(test_labels) != len(test_label_end_dates):
        raise ValueError("Test labels and dates must have the same length")
    if lookback_samples < 1:
        raise ValueError("Rolling base-rate lookback must be at least one sample")

    known_labels = list(zip(train_label_end_dates, train_labels))
    probabilities = []
    for feature_date in test_feature_dates:
        eligible = [
            (end_date, label)
            for end_date, label in [*known_labels, *zip(test_label_end_dates, test_labels)]
            if end_date < feature_date
        ]
        eligible.sort(key=lambda item: item[0])
        labels = [label for _, label in eligible[-lookback_samples:]]
        if not labels:
            raise ValueError(
                "Rolling base-rate forecast has no labels known before a test date"
            )
        probabilities.append(sum(labels) / len(labels))
    return probabilities


def probability_summary(
    labels: list[int],
    probabilities: list[float],
    decision_threshold: float,
    reference_brier_score: float | None = None,
) -> dict:
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same length")
    if not 0 < decision_threshold < 1:
        raise ValueError("Decision threshold must be between zero and one")
    if not labels:
        return {
            "sample_count": 0,
            "brier_score": None,
            "brier_skill_vs_reference": None,
            "roc_auc": None,
            "positive_prediction_rate": None,
            "classification_metrics": classification_metrics([], []),
        }
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("Probabilities must be between zero and one")

    brier_score = sum(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities)
    ) / len(labels)
    predictions = [int(probability >= decision_threshold) for probability in probabilities]
    return {
        "sample_count": len(labels),
        "brier_score": brier_score,
        "brier_skill_vs_reference": (
            1 - brier_score / reference_brier_score
            if reference_brier_score is not None and reference_brier_score > 0
            else None
        ),
        "roc_auc": _roc_auc(labels, probabilities),
        "positive_prediction_rate": sum(predictions) / len(predictions),
        "classification_metrics": classification_metrics(labels, predictions),
    }


def build_ranking_diagnostics(
    labels: list[int],
    probabilities: list[float],
    outcomes: list[dict[str, float | None]],
    quantile_count: int = 5,
) -> dict:
    """Measure whether higher model rank corresponds to better realized outcomes."""
    if len(labels) != len(probabilities) or len(labels) != len(outcomes):
        raise ValueError("Labels, probabilities, and outcomes must have the same length")
    if quantile_count < 2:
        raise ValueError("Ranking diagnostics require at least two quantiles")
    if not labels:
        return {"sample_count": 0, "quantiles": [], "top_minus_bottom": {}}

    ordered = sorted(
        zip(labels, probabilities, outcomes),
        key=lambda item: item[1],
    )
    bucket_count = min(quantile_count, len(ordered))
    quantiles = []
    for index in range(bucket_count):
        start = index * len(ordered) // bucket_count
        end = (index + 1) * len(ordered) // bucket_count
        bucket = ordered[start:end]
        quantiles.append({
            "prediction_quantile": index + 1,
            "quantile_direction": "lowest" if index == 0 else (
                "highest" if index == bucket_count - 1 else "middle"
            ),
            "sample_count": len(bucket),
            "mean_predicted_probability": _mean([item[1] for item in bucket]),
            "success_rate": _mean([float(item[0]) for item in bucket]),
            "average_forward_strategy_return": _outcome_mean(
                bucket, "strategy_return"
            ),
            "average_forward_excess_spy": _outcome_mean(
                bucket, "excess_spy_return"
            ),
            "drawdown_frequency": _outcome_mean(bucket, "drawdown_event"),
        })
    return {
        "sample_count": len(labels),
        "quantile_count": bucket_count,
        "quantiles": quantiles,
        "top_minus_bottom": {
            "success_rate": _difference(quantiles, "success_rate"),
            "average_forward_strategy_return": _difference(
                quantiles, "average_forward_strategy_return"
            ),
            "average_forward_excess_spy": _difference(
                quantiles, "average_forward_excess_spy"
            ),
            "drawdown_frequency": _difference(quantiles, "drawdown_frequency"),
        },
    }


def _roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _outcome_mean(
    bucket: list[tuple[int, float, dict[str, float | None]]],
    name: str,
) -> float | None:
    return _mean([item[2].get(name) for item in bucket])


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _difference(quantiles: list[dict], name: str) -> float | None:
    bottom = quantiles[0].get(name)
    top = quantiles[-1].get(name)
    return top - bottom if top is not None and bottom is not None else None
