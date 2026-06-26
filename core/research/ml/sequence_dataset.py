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
    sequence_group_ids: list[str]

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
        return SequenceMLDataset([], [], [], [], [], names, sequence_length, [])

    names = list(feature_names or sorted(dataset.features[0]))
    rows = [
        [float(row.get(name, 0.0) or 0.0) for name in names]
        for row in dataset.features
    ]
    group_ids = sequence_group_ids_from_metadata(
        dataset.metadata,
        dataset.sample_count,
    )

    sequences: list[list[list[float]]] = []
    labels: list[int] = []
    feature_dates: list[str] = []
    label_start_dates: list[str] = []
    label_end_dates: list[str] = []
    sequence_group_ids: list[str] = []

    for indices in build_sequence_indices(group_ids, sequence_length):
        end_index = indices[-1]
        sequences.append([rows[index] for index in indices])
        labels.append(int(dataset.labels[end_index]))
        feature_dates.append(dataset.feature_dates[end_index])
        label_start_dates.append(dataset.label_start_dates[end_index])
        label_end_dates.append(dataset.label_end_dates[end_index])
        sequence_group_ids.append(group_ids[end_index])

    return SequenceMLDataset(
        sequences=sequences,
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=label_start_dates,
        label_end_dates=label_end_dates,
        feature_names=names,
        sequence_length=sequence_length,
        sequence_group_ids=sequence_group_ids,
    )


def sequence_group_ids_from_metadata(
    metadata: Sequence[dict[str, str]] | None,
    sample_count: int,
) -> list[str]:
    """Return stable sequence-group IDs for row-adjacent sequence models.

    Expanded rebalance research datasets contain multiple universes/variants on
    the same rebalance date. A sequence model should learn chronology within a
    variant, not a synthetic path formed by the global CSV sort order.
    """

    if not metadata:
        return ["global" for _ in range(sample_count)]
    group_ids: list[str] = []
    for index in range(sample_count):
        row = metadata[index] if index < len(metadata) else {}
        group_ids.append(str(row.get("variant_id") or row.get("sequence_group_id") or "global"))
    return group_ids


def build_sequence_indices(
    group_ids: Sequence[str],
    sequence_length: int,
) -> list[list[int]]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    grouped_indices: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(str(group_id), []).append(index)

    sequences: list[list[int]] = []
    for indices in grouped_indices.values():
        if len(indices) < sequence_length:
            continue
        for end_offset in range(sequence_length - 1, len(indices)):
            sequences.append(indices[end_offset - sequence_length + 1 : end_offset + 1])
    sequences.sort(key=lambda item: item[-1])
    return sequences
