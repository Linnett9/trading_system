from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ScoreCandidate:
    symbol: str
    timestamp: object
    momentum_score: float | None


@dataclass(frozen=True)
class OOSScoreMetadata:
    fold_id: str | None = None


@dataclass(frozen=True)
class OOSArtifactDiagnostics:
    path: str
    signal_column: str
    row_count: int
    date_count: int
    symbol_count: int
    fold_count: int
    min_date: str | None
    max_date: str | None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "signal_column": self.signal_column,
            "row_count": self.row_count,
            "date_count": self.date_count,
            "symbol_count": self.symbol_count,
            "fold_count": self.fold_count,
            "min_date": self.min_date,
            "max_date": self.max_date,
        }


class DualMomentumScoreProvider(Protocol):
    name: str

    def score_candidates(
        self,
        timestamp: object,
        candidates: list[ScoreCandidate],
    ) -> dict[str, float]:
        ...

    def diagnostics(self) -> dict:
        ...


class MomentumScoreProvider:
    name = "dual_momentum"

    def score_candidates(
        self,
        timestamp: object,
        candidates: list[ScoreCandidate],
    ) -> dict[str, float]:
        return {
            candidate.symbol: candidate.momentum_score
            for candidate in candidates
            if candidate.momentum_score is not None
            and math.isfinite(candidate.momentum_score)
        }

    def diagnostics(self) -> dict:
        return {"provider": self.name}


class OOSArtifactScoreProvider:
    def __init__(
        self,
        path: str | Path,
        signal_column: str,
        date_column: str = "rebalance_date",
        symbol_column: str = "symbol",
        fold_column: str = "fold_id",
        oos_flag_column: str | None = None,
        allow_target_columns: bool = False,
        rank_normalize: bool = True,
        name: str | None = None,
    ):
        self.path = Path(path)
        self.signal_column = signal_column
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.fold_column = fold_column
        self.oos_flag_column = oos_flag_column
        self.rank_normalize = rank_normalize
        self.name = name or signal_column
        self._scores: dict[tuple[date, str], float] = {}
        self._metadata: dict[tuple[date, str], OOSScoreMetadata] = {}
        self._dates: set[date] = set()
        self._symbols: set[str] = set()
        self._folds: set[str] = set()

        self._load(allow_target_columns=allow_target_columns)

    @property
    def dates(self) -> set[date]:
        return set(self._dates)

    @property
    def symbols(self) -> set[str]:
        return set(self._symbols)

    def score_candidates(
        self,
        timestamp: object,
        candidates: list[ScoreCandidate],
    ) -> dict[str, float]:
        row_date = _to_date(timestamp)
        raw_scores = {}

        for candidate in candidates:
            key = (row_date, candidate.symbol)
            score = self._scores.get(key)
            if score is not None:
                raw_scores[candidate.symbol] = score

        if self.rank_normalize:
            return rank_normalize_scores(raw_scores)

        return raw_scores

    def missing_symbols(
        self,
        timestamp: object,
        symbols: list[str],
    ) -> list[str]:
        row_date = _to_date(timestamp)
        return [
            symbol
            for symbol in symbols
            if (row_date, symbol) not in self._scores
        ]

    def fold_id(self, timestamp: object, symbol: str) -> str | None:
        metadata = self._metadata.get((_to_date(timestamp), symbol))
        return metadata.fold_id if metadata is not None else None

    def diagnostics(self) -> dict:
        dates = sorted(self._dates)
        payload = OOSArtifactDiagnostics(
            path=str(self.path),
            signal_column=self.signal_column,
            row_count=len(self._scores),
            date_count=len(self._dates),
            symbol_count=len(self._symbols),
            fold_count=len(self._folds),
            min_date=dates[0].isoformat() if dates else None,
            max_date=dates[-1].isoformat() if dates else None,
        )
        return payload.to_dict()

    def _load(self, allow_target_columns: bool) -> None:
        if _is_target_like_column(self.signal_column) and not allow_target_columns:
            raise ValueError(
                f"Refusing target-like score column: {self.signal_column}"
            )

        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            required = {self.date_column, self.symbol_column, self.signal_column}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(
                    f"Missing required OOS score artifact columns: {missing}"
                )

            for row_number, row in enumerate(reader, start=2):
                if self.oos_flag_column and self.oos_flag_column in columns:
                    if not _truthy(row.get(self.oos_flag_column)):
                        continue

                row_date = _parse_date(row[self.date_column])
                symbol = str(row[self.symbol_column]).strip()
                key = (row_date, symbol)
                if key in self._scores:
                    raise ValueError(
                        "Duplicate OOS score row for "
                        f"{self.date_column}={row_date.isoformat()} "
                        f"{self.symbol_column}={symbol}"
                    )

                score = _parse_finite_float(
                    row[self.signal_column],
                    row_number=row_number,
                    column=self.signal_column,
                )
                fold_id = row.get(self.fold_column) if self.fold_column in columns else None
                self._scores[key] = score
                self._metadata[key] = OOSScoreMetadata(fold_id=fold_id)
                self._dates.add(row_date)
                self._symbols.add(symbol)
                if fold_id:
                    self._folds.add(fold_id)


class HybridScoreProvider:
    def __init__(
        self,
        artifact_provider: OOSArtifactScoreProvider,
        momentum_weight: float = 0.5,
        artifact_weight: float = 0.5,
        name: str | None = None,
    ):
        if momentum_weight < 0 or artifact_weight < 0:
            raise ValueError("Hybrid weights must be non-negative")
        if momentum_weight + artifact_weight <= 0:
            raise ValueError("Hybrid weights must have positive total weight")
        self.artifact_provider = artifact_provider
        self.momentum_weight = momentum_weight
        self.artifact_weight = artifact_weight
        self.name = name or f"hybrid_{artifact_provider.name}"

    def score_candidates(
        self,
        timestamp: object,
        candidates: list[ScoreCandidate],
    ) -> dict[str, float]:
        momentum = {
            candidate.symbol: candidate.momentum_score
            for candidate in candidates
            if candidate.momentum_score is not None
            and math.isfinite(candidate.momentum_score)
        }
        ml_scores = self.artifact_provider.score_candidates(timestamp, candidates)
        momentum_ranks = rank_normalize_scores(momentum)
        common_symbols = sorted(set(momentum_ranks) & set(ml_scores))
        total_weight = self.momentum_weight + self.artifact_weight

        return {
            symbol: (
                self.momentum_weight * momentum_ranks[symbol]
                + self.artifact_weight * ml_scores[symbol]
            )
            / total_weight
            for symbol in common_symbols
        }

    def diagnostics(self) -> dict:
        return {
            "provider": self.name,
            "momentum_weight": self.momentum_weight,
            "artifact_weight": self.artifact_weight,
            "artifact": self.artifact_provider.diagnostics(),
        }


class RankWeightedEnsembleScoreProvider:
    def __init__(
        self,
        name: str,
        members: dict[str, tuple[DualMomentumScoreProvider, float]],
    ):
        if not members:
            raise ValueError("Ensemble score provider requires at least one member")
        if any(weight < 0 for _, weight in members.values()):
            raise ValueError("Ensemble weights must be non-negative")
        total_weight = sum(weight for _, weight in members.values())
        if total_weight <= 0:
            raise ValueError("Ensemble weights must have positive total weight")
        self.name = name
        self.members = dict(members)
        self.total_weight = total_weight

    def score_candidates(
        self,
        timestamp: object,
        candidates: list[ScoreCandidate],
    ) -> dict[str, float]:
        member_scores = {
            name: provider.score_candidates(timestamp, candidates)
            for name, (provider, _) in self.members.items()
        }
        common_symbols = set.intersection(
            *[
                set(scores)
                for scores in member_scores.values()
            ]
        )
        return {
            symbol: sum(
                weight * member_scores[name][symbol]
                for name, (_, weight) in self.members.items()
            )
            / self.total_weight
            for symbol in sorted(common_symbols)
        }

    def diagnostics(self) -> dict:
        return {
            "provider": self.name,
            "members": {
                name: {
                    "weight": weight,
                    "provider": provider.name,
                    "diagnostics": provider.diagnostics(),
                }
                for name, (provider, weight) in self.members.items()
            },
        }


def rank_normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    finite_scores = {
        symbol: score
        for symbol, score in scores.items()
        if score is not None and math.isfinite(score)
    }
    if not finite_scores:
        return {}
    if len(finite_scores) == 1:
        symbol = next(iter(finite_scores))
        return {symbol: 1.0}

    sorted_items = sorted(
        finite_scores.items(),
        key=lambda item: (item[1], item[0]),
    )
    ranks: dict[str, float] = {}
    index = 0
    denominator = len(sorted_items)

    while index < len(sorted_items):
        score = sorted_items[index][1]
        end = index
        while end + 1 < len(sorted_items) and sorted_items[end + 1][1] == score:
            end += 1
        average_rank = (index + end) / 2
        normalized = (average_rank + 1) / denominator
        for item_index in range(index, end + 1):
            ranks[sorted_items[item_index][0]] = normalized
        index = end + 1

    return ranks


def _is_target_like_column(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered.startswith("actual_")
        or lowered.startswith("target_")
        or lowered.endswith("_label")
        or "_label_" in lowered
    )


def _parse_date(value: str) -> date:
    return _to_date(datetime.fromisoformat(value.strip()))


def _to_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return _parse_date(value)
    raise TypeError(f"Unsupported score timestamp type: {type(value)!r}")


def _parse_finite_float(value: str, row_number: int, column: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric score in row {row_number}, column {column}"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            f"Non-finite score in row {row_number}, column {column}"
        )
    return score


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
