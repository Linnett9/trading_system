from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, TypeVar


Row = dict[str, Any]
T = TypeVar("T")


class FeatureGenerator(Protocol):
    def generate(self, rows: Sequence[Row]) -> list[Row]: ...


class TargetGenerator(Protocol):
    def generate(self, rows: Sequence[Row]) -> list[float | int | None]: ...


class WalkForwardSplitter(Protocol):
    def split(self, rows: Sequence[Row]) -> Sequence[Any]: ...


class PredictionWriter(Protocol):
    def write_predictions(self, path: Path, rows: Sequence[Row]) -> Path: ...


class RankingEvaluator(Protocol):
    def evaluate(
        self,
        rows: Sequence[Row],
        *,
        signal_column: str,
        target_column: str,
    ) -> dict[str, Any]: ...


class ReplayEngine(Protocol):
    def replay(self, rows: Sequence[Row]) -> dict[str, Any]: ...


class ValidationGate(Protocol):
    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class BenchmarkRunner(Protocol):
    def run(self) -> dict[str, Any]: ...


class ReportWriter(Protocol):
    def write_json(self, path: Path, payload: Any) -> Path: ...

    def write_markdown(self, path: Path, content: str) -> Path: ...


class ComponentFactory(Protocol[T]):
    def __call__(self) -> T: ...
