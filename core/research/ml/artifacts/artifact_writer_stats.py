from __future__ import annotations


class MLArtifactStatsMixin:
    @staticmethod
    def standard_deviation(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        average = sum(values) / len(values)
        return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5
    @staticmethod
    def correlation(left: list[float], right: list[float]) -> float:
        if len(left) < 2 or len(left) != len(right):
            return 0.0
        left_average = sum(left) / len(left)
        right_average = sum(right) / len(right)
        numerator = sum(
            (left_value - left_average) * (right_value - right_average)
            for left_value, right_value in zip(left, right)
        )
        left_scale = sum((value - left_average) ** 2 for value in left) ** 0.5
        right_scale = sum((value - right_average) ** 2 for value in right) ** 0.5
        if left_scale == 0 or right_scale == 0:
            return 0.0
        return numerator / (left_scale * right_scale)
    @staticmethod
    def is_numeric_column(
        rows: list[dict[str, float | str]],
        name: str,
    ) -> bool:
        for row in rows:
            try:
                float(row[name])
            except (KeyError, TypeError, ValueError):
                return False
        return True
