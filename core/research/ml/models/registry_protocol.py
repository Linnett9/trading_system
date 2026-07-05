from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class IMLModel(ABC):
    @abstractmethod
    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[dict[str, float]]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def feature_importances(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "IMLModel":
        raise NotImplementedError
