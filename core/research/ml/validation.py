from __future__ import annotations

from dataclasses import dataclass

from core.research.ml.datasets import MLDataset


@dataclass(frozen=True)
class ChronologicalSplit:
    train: MLDataset
    test: MLDataset
    test_start_date: str | None
    purged_train_samples: int


@dataclass(frozen=True)
class WalkForwardFold:
    fold_number: int
    split: ChronologicalSplit


def chronological_holdout(
    dataset: MLDataset,
    test_fraction: float = 0.20,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> ChronologicalSplit:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if not dataset.sample_count:
        empty = _slice_dataset(dataset, [])
        return ChronologicalSplit(empty, empty, None, 0)

    all_indices = list(range(dataset.sample_count))
    if test_start is None:
        test_size = max(1, int(dataset.sample_count * test_fraction))
        test_indices = all_indices[-test_size:]
    else:
        test_indices = [
            index for index in all_indices
            if dataset.feature_dates[index] >= test_start
            and (test_end is None or dataset.feature_dates[index] <= test_end)
        ]
    if not test_indices:
        raise ValueError("Chronological split has no test samples")

    resolved_test_start = dataset.feature_dates[test_indices[0]]
    train_indices = [
        index for index in all_indices
        if dataset.feature_dates[index] < resolved_test_start
        and (train_start is None or dataset.feature_dates[index] >= train_start)
        and (train_end is None or dataset.feature_dates[index] <= train_end)
    ]
    purged_indices = [
        index for index in train_indices
        if dataset.label_end_dates[index] >= resolved_test_start
    ]
    retained_train_indices = [
        index for index in train_indices
        if index not in set(purged_indices)
    ]
    if not retained_train_indices:
        raise ValueError("Chronological split has no training samples after purging")

    return ChronologicalSplit(
        train=_slice_dataset(dataset, retained_train_indices),
        test=_slice_dataset(dataset, test_indices),
        test_start_date=resolved_test_start,
        purged_train_samples=len(purged_indices),
    )


def _slice_dataset(dataset: MLDataset, indices: list[int]) -> MLDataset:
    return MLDataset(
        features=[dataset.features[index] for index in indices],
        labels=[dataset.labels[index] for index in indices],
        feature_dates=[dataset.feature_dates[index] for index in indices],
        label_start_dates=[dataset.label_start_dates[index] for index in indices],
        label_end_dates=[dataset.label_end_dates[index] for index in indices],
    )


def rolling_walk_forward(
    dataset: MLDataset,
    fold_count: int = 3,
) -> list[WalkForwardFold]:
    if fold_count < 1:
        raise ValueError("fold_count must be at least one")
    if dataset.sample_count < fold_count * 2:
        return []

    test_size = dataset.sample_count // (fold_count + 2)
    folds: list[WalkForwardFold] = []
    for fold_number in range(1, fold_count + 1):
        test_start_index = test_size * (fold_number + 1)
        test_end_index = test_start_index + test_size
        if test_end_index > dataset.sample_count:
            break
        try:
            split = chronological_holdout(
                dataset,
                test_start=dataset.feature_dates[test_start_index],
                test_end=dataset.feature_dates[test_end_index - 1],
            )
        except ValueError:
            continue
        folds.append(WalkForwardFold(fold_number=fold_number, split=split))
    return folds
