from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


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
    alpha_feature_n_jobs: int
    overnight_stage_n_jobs: int
    include_sequence_models: bool
    include_engineered_features: bool
    sequence_length: int
    sequence_epochs: int
    sequence_batch_size: int
    sequence_device: str
    permutation_repeats: int
    spy_symbol: str
    target_column: str
    target_comparison_enabled: bool
    target_columns: tuple[str, ...]
    overnight_run_attribution: bool
    run_size: str
    dev_max_dates: int
    dev_max_symbols: int
    dev_recent_dates_only: bool
    resume_existing_outputs: bool
    force_refresh: bool
    target_comparison_n_jobs: int
    attribution_max_models: int
    attribution_max_features: int
    attribution_permutation_repeats: int
    portfolio_replay_enabled: bool
    portfolio_top_n: int
    portfolio_cost_bps: float
    portfolio_slippage_bps: float
    portfolio_max_position_weight: float
    portfolio_min_position_weight: float
    portfolio_allow_short: bool
    portfolio_signal_columns: tuple[str, ...]
    overnight_run_portfolio_replay: bool

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "StockLevelResearchConfig":
        ml = dict(config.get("ml", {}) or {})
        run_size = str(ml.get("stock_alpha_run_size", "benchmark")).lower()
        stock_alpha_default_workers = 1 if run_size == "dev" else 4
        reports = dict(config.get("reports", {}) or {})
        output_dir = stock_alpha_output_dir(config)
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
            model_n_jobs=int(ml.get("stock_ranker_model_n_jobs", stock_alpha_default_workers)),
            alpha_feature_n_jobs=int(ml.get("stock_alpha_feature_n_jobs", stock_alpha_default_workers)),
            overnight_stage_n_jobs=int(
                ml.get("stock_alpha_overnight_stage_n_jobs", 1)
            ),
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
            target_column=str(ml.get("stock_ranker_target_column", "actual_forward_return_10d")),
            target_comparison_enabled=bool(ml.get("stock_ranker_target_comparison_enabled", True)),
            target_columns=tuple(ml.get("stock_ranker_target_columns", ["actual_forward_return_10d"])),
            overnight_run_attribution=bool(ml.get("stock_alpha_overnight_run_attribution", False)),
            run_size=run_size,
            dev_max_dates=int(ml.get("stock_alpha_dev_max_dates", 80)),
            dev_max_symbols=int(ml.get("stock_alpha_dev_max_symbols", 120)),
            dev_recent_dates_only=bool(ml.get("stock_alpha_dev_recent_dates_only", True)),
            resume_existing_outputs=bool(ml.get("stock_alpha_resume_existing_outputs", True)),
            force_refresh=bool(ml.get("stock_alpha_force_refresh", False)),
            target_comparison_n_jobs=int(ml.get("stock_target_comparison_n_jobs", stock_alpha_default_workers)),
            attribution_max_models=int(ml.get("stock_feature_attribution_max_models", 4)),
            attribution_max_features=int(ml.get("stock_feature_attribution_max_features", 12)),
            attribution_permutation_repeats=int(ml.get("stock_feature_attribution_permutation_repeats", 1)),
            portfolio_replay_enabled=bool(ml.get("stock_portfolio_replay_enabled", True)),
            portfolio_top_n=int(ml.get("stock_portfolio_replay_top_n", 25)),
            portfolio_cost_bps=float(ml.get("stock_portfolio_replay_cost_bps", 10)),
            portfolio_slippage_bps=float(ml.get("stock_portfolio_replay_slippage_bps", 5)),
            portfolio_max_position_weight=float(ml.get("stock_portfolio_replay_max_position_weight", 0.05)),
            portfolio_min_position_weight=float(ml.get("stock_portfolio_replay_min_position_weight", 0.0)),
            portfolio_allow_short=bool(ml.get("stock_portfolio_replay_allow_short", False)),
            portfolio_signal_columns=tuple(ml.get("stock_portfolio_replay_signal_columns", ["stock_level_predicted_forward_return_10d_elastic_net", "stock_level_predicted_forward_return_10d_random_forest", "predicted_momentum_120d"])),
            overnight_run_portfolio_replay=bool(ml.get("stock_alpha_overnight_run_portfolio_replay", True)),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        positive = {
            "stock_ranker_min_train_dates": self.min_train_dates,
            "stock_ranker_test_window_dates": self.test_window_dates,
            "sklearn_n_jobs": self.sklearn_n_jobs,
            "stock_ranker_model_n_jobs": self.model_n_jobs,
            "stock_alpha_feature_n_jobs": self.alpha_feature_n_jobs,
            "stock_alpha_overnight_stage_n_jobs": self.overnight_stage_n_jobs,
            "stock_ranker_sequence_length": self.sequence_length,
            "stock_ranker_sequence_epochs": self.sequence_epochs,
            "stock_ranker_sequence_batch_size": self.sequence_batch_size,
            "stock_alpha_dev_max_dates": self.dev_max_dates,
            "stock_alpha_dev_max_symbols": self.dev_max_symbols,
            "stock_target_comparison_n_jobs": self.target_comparison_n_jobs,
            "stock_feature_attribution_max_models": self.attribution_max_models,
            "stock_feature_attribution_max_features": self.attribution_max_features,
            "stock_feature_attribution_permutation_repeats": self.attribution_permutation_repeats,
            "stock_portfolio_replay_top_n": self.portfolio_top_n,
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
        if self.run_size not in {"dev", "benchmark", "full"}:
            raise ValueError("ml.stock_alpha_run_size must be dev, benchmark, or full")
        if not 0.0 < self.portfolio_max_position_weight <= 1.0:
            raise ValueError("ml.stock_portfolio_replay_max_position_weight must be in (0, 1]")
