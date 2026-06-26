from __future__ import annotations

import math


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


def compare_calibration_methods(
    train_labels: list[int],
    train_probabilities: list[float],
    test_labels: list[int],
    test_probabilities: list[float],
    bin_count: int = 10,
) -> dict:
    if len(train_labels) != len(train_probabilities):
        raise ValueError("Training labels and probabilities must have the same length")
    if len(test_labels) != len(test_probabilities):
        raise ValueError("Test labels and probabilities must have the same length")
    _validate_probabilities(train_probabilities)
    _validate_probabilities(test_probabilities)

    calibrated = {
        "raw": list(test_probabilities),
        "platt": _platt_probabilities(
            train_labels,
            train_probabilities,
            test_probabilities,
        ),
        "isotonic": _isotonic_probabilities(
            train_labels,
            train_probabilities,
            test_probabilities,
        ),
        "temperature_scaling": _temperature_scaled_probabilities(
            train_labels,
            train_probabilities,
            test_probabilities,
        ),
    }
    methods = {
        name: {
            "probabilities": probabilities,
            "calibration": build_probability_calibration(
                test_labels,
                probabilities,
                bin_count=bin_count,
            ),
        }
        for name, probabilities in calibrated.items()
    }
    ranked = sorted(
        (
            {
                "method": name,
                "brier_score": payload["calibration"].get("brier_score"),
                "expected_calibration_error": payload["calibration"].get(
                    "expected_calibration_error"
                ),
            }
            for name, payload in methods.items()
        ),
        key=lambda row: (
            row["brier_score"] is None,
            row["brier_score"] if row["brier_score"] is not None else float("inf"),
            row["expected_calibration_error"]
            if row["expected_calibration_error"] is not None else float("inf"),
        ),
    )
    return {
        "methods": methods,
        "ranked_methods": ranked,
        "best_method_by_brier": ranked[0]["method"] if ranked else None,
    }


def _validate_probabilities(probabilities: list[float]) -> None:
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("Probabilities must be between zero and one")


def _platt_probabilities(
    labels: list[int],
    train_probabilities: list[float],
    test_probabilities: list[float],
) -> list[float]:
    if len(set(labels)) < 2 or not labels:
        return list(test_probabilities)
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return list(test_probabilities)
    model = LogisticRegression(solver="lbfgs")
    x_train = [[_logit(probability)] for probability in train_probabilities]
    model.fit(x_train, labels)
    return model.predict_proba([[_logit(value)] for value in test_probabilities])[
        :, 1
    ].tolist()


def _isotonic_probabilities(
    labels: list[int],
    train_probabilities: list[float],
    test_probabilities: list[float],
) -> list[float]:
    if len(set(labels)) < 2 or not labels:
        return list(test_probabilities)
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return list(test_probabilities)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(train_probabilities, labels)
    return [float(value) for value in model.predict(test_probabilities)]


def _temperature_scaled_probabilities(
    labels: list[int],
    train_probabilities: list[float],
    test_probabilities: list[float],
) -> list[float]:
    if not labels:
        return list(test_probabilities)
    candidates = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_temperature = min(
        candidates,
        key=lambda temperature: _brier_score(
            labels,
            [_sigmoid(_logit(value) / temperature) for value in train_probabilities],
        ),
    )
    return [
        _sigmoid(_logit(value) / best_temperature)
        for value in test_probabilities
    ]


def _brier_score(labels: list[int], probabilities: list[float]) -> float:
    return sum(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities)
    ) / len(labels)


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
