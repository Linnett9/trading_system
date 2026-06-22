from __future__ import annotations


def build_probability_calibration(
    labels: list[int],
    probabilities: list[float],
    bin_count: int = 10,
) -> dict:
    """Summarise whether model probabilities match observed outcomes."""
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same length")
    if bin_count < 2:
        raise ValueError("Probability calibration requires at least two bins")
    if not labels:
        return {
            "sample_count": 0,
            "positive_rate": None,
            "brier_score": None,
            "base_rate_brier_score": None,
            "brier_skill_score": None,
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
            "bins": [],
        }

    _validate_probabilities(probabilities)
    positive_rate = sum(labels) / len(labels)
    brier_score = sum(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities)
    ) / len(labels)
    base_rate_brier_score = sum(
        (positive_rate - label) ** 2 for label in labels
    ) / len(labels)
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bin_count)]
    for label, probability in zip(labels, probabilities):
        index = min(int(probability * bin_count), bin_count - 1)
        buckets[index].append((label, probability))

    bins = []
    weighted_error = 0.0
    maximum_error = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        observed_rate = sum(label for label, _ in bucket) / len(bucket)
        mean_probability = sum(value for _, value in bucket) / len(bucket)
        calibration_error = observed_rate - mean_probability
        absolute_error = abs(calibration_error)
        weighted_error += absolute_error * len(bucket) / len(labels)
        maximum_error = max(maximum_error, absolute_error)
        bins.append({
            "lower_bound": index / bin_count,
            "upper_bound": (index + 1) / bin_count,
            "sample_count": len(bucket),
            "mean_predicted_probability": mean_probability,
            "observed_positive_rate": observed_rate,
            "calibration_error": calibration_error,
        })

    return {
        "sample_count": len(labels),
        "positive_rate": positive_rate,
        "brier_score": brier_score,
        "base_rate_brier_score": base_rate_brier_score,
        "brier_skill_score": (
            1 - brier_score / base_rate_brier_score
            if base_rate_brier_score > 0 else None
        ),
        "expected_calibration_error": weighted_error,
        "maximum_calibration_error": maximum_error,
        "bins": bins,
    }


def _validate_probabilities(probabilities: list[float]) -> None:
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("Probabilities must be between zero and one")
