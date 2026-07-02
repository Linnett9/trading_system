from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUXILIARY_TARGETS = {
    "actual_forward_return_5d": "meta_predicted_forward_return_5d",
    "actual_forward_return_10d": "meta_predicted_forward_return_10d",
    "actual_future_volatility": "meta_predicted_future_volatility",
    "actual_future_drawdown": "meta_predicted_future_drawdown",
    "actual_max_adverse_excursion": "meta_predicted_max_adverse_excursion",
    "actual_max_favourable_excursion": "meta_predicted_max_favourable_excursion",
}
AUXILIARY_PREDICTION_COLUMNS = tuple(
    prediction_name.removeprefix("meta_")
    for prediction_name in AUXILIARY_TARGETS.values()
)
@dataclass(frozen=True)
class MetaAuxiliaryResult:
    train_rows: list[dict[str, str]]
    holdout_rows: list[dict[str, str]]
    selection_train_indexes: tuple[int, ...]
    predictions_path: Path
    metrics_json_path: Path
    metrics_markdown_path: Path
    metrics: dict[str, Any]
