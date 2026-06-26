from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.research.ml.datasets import MLDataset


@dataclass(frozen=True)
class SequenceMLDataset:
    """Research-only rolling-window view over an existing tabular MLDataset.

    Each sequence ends on the sample's feature_date. The label belongs to the
    final row in that sequence, so the sequence never includes data from the
    future label window.
    """

    sequences: list[list[list[float]]]
    labels: list[int]
    feature_dates: list[str]
    label_start_dates: list[str]
    label_end_dates: list[str]
    feature_names: list[str]
    sequence_length: int

    @property
    def sample_count(self) -> int:
        return min(len(self.sequences), len(self.labels))

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def build_sequence_dataset(
    dataset: MLDataset,
    sequence_length: int = 63,
    feature_names: Sequence[str] | None = None,
) -> SequenceMLDataset:
    """Convert a chronological tabular dataset into rolling sequences.

    The first emitted sample uses rows [0:sequence_length] and inherits the
    label/date metadata from row sequence_length - 1. This keeps labels aligned
    with the last observable feature row.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    if dataset.sample_count == 0:
        names = list(feature_names or [])
        return SequenceMLDataset([], [], [], [], [], names, sequence_length)

    names = list(feature_names or sorted(dataset.features[0]))
    rows = [
        [float(row.get(name, 0.0) or 0.0) for name in names]
        for row in dataset.features
    ]

    sequences: list[list[list[float]]] = []
    labels: list[int] = []
    feature_dates: list[str] = []
    label_start_dates: list[str] = []
    label_end_dates: list[str] = []

    for end_index in range(sequence_length - 1, dataset.sample_count):
        start_index = end_index - sequence_length + 1
        sequences.append(rows[start_index : end_index + 1])
        labels.append(int(dataset.labels[end_index]))
        feature_dates.append(dataset.feature_dates[end_index])
        label_start_dates.append(dataset.label_start_dates[end_index])
        label_end_dates.append(dataset.label_end_dates[end_index])

    return SequenceMLDataset(
        sequences=sequences,
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=label_start_dates,
        label_end_dates=label_end_dates,
        feature_names=names,
        sequence_length=sequence_length,
    )
