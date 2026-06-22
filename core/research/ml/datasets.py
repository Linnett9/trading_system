from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MLDataset:
    features: list[dict[str, float]]
    labels: list[int]
    feature_dates: list[str]
    label_start_dates: list[str]
    label_end_dates: list[str]

    @property
    def sample_count(self) -> int:
        return min(len(self.features), len(self.labels))

    @property
    def feature_count(self) -> int:
        if not self.features:
            return 0
        return len(self.features[0])


def build_dataset(
    feature_rows: list[dict[str, float | str]],
    label_rows: list[dict[str, Any]],
    label_name: str = "risk_regime",
) -> MLDataset:
    labels_by_date = {str(row["feature_date"]): row for row in label_rows}
    features: list[dict[str, float]] = []
    labels: list[int] = []
    feature_dates: list[str] = []
    label_start_dates: list[str] = []
    label_end_dates: list[str] = []

    for feature_row in feature_rows:
        feature_date = str(feature_row["feature_date"])
        label_row = labels_by_date.get(feature_date)
        if label_row is None:
            continue

        label_start = str(label_row["label_start_date"])
        label_end = str(label_row["label_end_date"])
        if not feature_date < label_start <= label_end:
            raise ValueError("Dataset contains a label that leaks future information")

        features.append({
            name: float(value)
            for name, value in feature_row.items()
            if name != "feature_date"
        })
        labels.append(int(label_row[label_name]))
        feature_dates.append(feature_date)
        label_start_dates.append(label_start)
        label_end_dates.append(label_end)

    return MLDataset(
        features=features,
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=label_start_dates,
        label_end_dates=label_end_dates,
    )


def write_dataset(path: Path, dataset: MLDataset, label_name: str = "risk_regime") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_date",
        "label_start_date",
        "label_end_date",
        label_name,
        *list(dataset.features[0] if dataset.features else {}),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, features in enumerate(dataset.features):
            writer.writerow({
                "feature_date": dataset.feature_dates[index],
                "label_start_date": dataset.label_start_dates[index],
                "label_end_date": dataset.label_end_dates[index],
                label_name: dataset.labels[index],
                **features,
            })
