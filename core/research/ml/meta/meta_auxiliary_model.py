from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from core.research.ml.meta.meta_auxiliary_features import _feature_matrix


@dataclass
class _AuxiliaryRegressor:
    feature_names: list[str]
    estimator: Any = None
    constant: float | None = None

    def predict(self, rows: list[dict[str, str]]) -> list[float]:
        if self.constant is not None:
            return [self.constant for _ in rows]
        return [
            float(value)
            for value in self.estimator.predict(_feature_matrix(rows, self.feature_names))
        ]
def _fit_regressor(
    rows: list[dict[str, str]],
    actual_name: str,
    feature_names: list[str],
) -> _AuxiliaryRegressor:
    targets = [float(row[actual_name]) for row in rows]
    if len(rows) < 2 or max(targets) == min(targets):
        return _AuxiliaryRegressor(feature_names, constant=mean(targets))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(_feature_matrix(rows, feature_names), targets)
    return _AuxiliaryRegressor(feature_names, estimator=estimator)
