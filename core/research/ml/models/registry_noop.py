from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.models.registry_protocol import IMLModel


class NoOpMLModel(IMLModel):
    """Neutral research-only model used to wire ML safely."""

    model_type = "noop"

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        return None

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [0 for _ in x]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [0.5 for _ in x]

    def feature_importances(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model_type": self.model_type}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "NoOpMLModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model payload: {payload}")
        return cls()
