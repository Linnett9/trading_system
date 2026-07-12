from __future__ import annotations

# Compatibility re-export for older internal imports. The canonical
# implementation lives in core.research.ml.data.datasets.
from core.research.ml.data.datasets import (
    MLDataset,
    MODEL_INPUT_CONTRACT_VERSION,
    MULTITASK_AUXILIARY_TARGET_COLUMNS,
    TARGET_DERIVED_COLUMNS,
    build_dataset,
    dataset_leakage_audit,
    forbidden_predictor_columns,
    write_dataset,
)

__all__ = [
    "MLDataset",
    "MODEL_INPUT_CONTRACT_VERSION",
    "MULTITASK_AUXILIARY_TARGET_COLUMNS",
    "TARGET_DERIVED_COLUMNS",
    "build_dataset",
    "dataset_leakage_audit",
    "forbidden_predictor_columns",
    "write_dataset",
]
