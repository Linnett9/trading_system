from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StockLevelResearchConfig:
    output_dir: Path
    base_artifact_path: Path
    artifact_path: Path
    benchmark_path: Path
    oos_predictions_path: Path
    parquet_dir: Path
    min_train_dates: int
    test_window_dates: int
    embargo_dates: int
    random_seed: int
    sklearn_n_jobs: int
    model_n_jobs: int
    include_sequence_models: bool
    include_engineered_features: bool
    sequence_length: int
    sequence_epochs: int
    sequence_batch_size: int
    sequence_device: str
    permutation_repeats: int
    spy_symbol: str

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "StockLevelResearchConfig":
        ml = dict(config.get("ml", {}) or {})
        reports = dict(config.get("reports", {}) or {})
        output_dir = Path(
            ml.get(
                "output_dir",
                Path(reports.get("ml_dir", "reports/ml"))
                / "regime_transformer_meta_ensemble_v1",
            )
        )
        base_artifact_path = Path(
            ml.get(
                "stock_level_base_prediction_artifacts_path",
                output_dir / "stock_level_prediction_artifacts.csv",
            )
        )
        artifact_path = Path(
            ml.get(
                "stock_level_prediction_artifacts_path",
                output_dir / "stock_level_prediction_artifacts.csv",
            )
        )
        instance = cls(
            output_dir=output_dir,
            base_artifact_path=base_artifact_path,
            artifact_path=artifact_path,
            benchmark_path=Path(
                ml.get(
                    "stock_level_model_ranking_benchmark_path",
                    output_dir / "stock_level_model_ranking_benchmark.json",
                )
            ),
            oos_predictions_path=Path(
                ml.get(
                    "stock_level_model_oos_predictions_path",
                    output_dir / "stock_level_model_oos_predictions.csv",
                )
            ),
            parquet_dir=Path(
                ml.get("stooq_parquet_dir", "data/processed/stooq_parquet")
            ),
            min_train_dates=int(ml.get("stock_ranker_min_train_dates", 52)),
            test_window_dates=int(ml.get("stock_ranker_test_window_dates", 13)),
            embargo_dates=int(ml.get("stock_ranker_embargo_dates", 2)),
            random_seed=int(ml.get("random_seed", 42)),
            sklearn_n_jobs=int(ml.get("sklearn_n_jobs", 1)),
            model_n_jobs=int(ml.get("stock_ranker_model_n_jobs", 1)),
            include_sequence_models=bool(
                ml.get("stock_ranker_include_sequence_models", True)
            ),
            include_engineered_features=bool(
                ml.get("stock_ranker_include_engineered_features", False)
            ),
            sequence_length=int(ml.get("stock_ranker_sequence_length", 13)),
            sequence_epochs=int(ml.get("stock_ranker_sequence_epochs", 5)),
            sequence_batch_size=int(
                ml.get("stock_ranker_sequence_batch_size", 256)
            ),
            sequence_device=str(ml.get("stock_ranker_sequence_device", "cpu")),
            permutation_repeats=int(
                ml.get("stock_ranker_permutation_importance_repeats", 3)
            ),
            spy_symbol=str(ml.get("stock_ranker_spy_symbol", "SPY")).upper(),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        positive = {
            "stock_ranker_min_train_dates": self.min_train_dates,
            "stock_ranker_test_window_dates": self.test_window_dates,
            "sklearn_n_jobs": self.sklearn_n_jobs,
            "stock_ranker_model_n_jobs": self.model_n_jobs,
            "stock_ranker_sequence_length": self.sequence_length,
            "stock_ranker_sequence_epochs": self.sequence_epochs,
            "stock_ranker_sequence_batch_size": self.sequence_batch_size,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"ml.{name} must be at least one")
        if self.embargo_dates < 0:
            raise ValueError("ml.stock_ranker_embargo_dates cannot be negative")
        if self.permutation_repeats < 0:
            raise ValueError(
                "ml.stock_ranker_permutation_importance_repeats cannot be negative"
            )
