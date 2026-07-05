from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from core.research.ml.data.datasets import MLDataset


class MLArtifactHashingMixin:
    def dataset_hash(self, dataset: MLDataset) -> str:
        return self.source_dataset_hash(dataset)
    def source_dataset_hash(self, dataset: MLDataset) -> str:
        return self.hash_payload(self.source_dataset_identity(dataset))
    def source_dataset_identity(self, dataset: MLDataset) -> dict[str, Any]:
        rows = []
        for index in range(dataset.sample_count):
            metadata = dataset.metadata[index] if index < len(dataset.metadata) else {}
            rows.append({
                "feature_date": dataset.feature_dates[index],
                "label_start_date": dataset.label_start_dates[index],
                "label_end_date": dataset.label_end_dates[index],
                "label": dataset.labels[index],
                "rebalance_date": metadata.get(
                    "rebalance_date",
                    dataset.feature_dates[index],
                ),
                "variant_id": metadata.get("variant_id", ""),
                "symbol": metadata.get("symbol", ""),
                "selected_symbols": metadata.get("selected_symbols", ""),
                "variant_universe": metadata.get("variant_universe", ""),
                "variant_rebalance_frequency": metadata.get(
                    "variant_rebalance_frequency",
                    "",
                ),
                "variant_weighting": metadata.get("variant_weighting", ""),
            })
        rows.sort(key=lambda row: tuple(str(value) for value in row.values()))
        return {
            "label_type": self._experiment_config.label_type,
            "rows": rows,
        }
    def model_input_hash(self, dataset: MLDataset) -> str:
        return self.hash_payload({
            "features": dataset.features,
            "labels": dataset.labels,
            "feature_ids": dataset.feature_ids,
            "feature_dates": dataset.feature_dates,
            "label_start_dates": dataset.label_start_dates,
            "label_end_dates": dataset.label_end_dates,
            "auxiliary_targets": dataset.auxiliary_targets,
        })
    @staticmethod
    def hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    @staticmethod
    def git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
