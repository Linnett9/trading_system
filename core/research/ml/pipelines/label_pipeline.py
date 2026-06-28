from __future__ import annotations

from typing import Any, Mapping

from core.research.ml.config import MLExperimentConfig
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.labels import (
    ChampionSuccessLabelBuilder,
    DrawdownRiskLabelBuilder,
    MLLabelBuildResult,
    RiskRegimeLabelBuilder,
    ShouldReduceExposureLabelBuilder,
)


class MLLabelPipeline:
    """Build research labels for an ML experiment."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        champion_equity_curve: list[Any],
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._champion_equity_curve = champion_equity_curve

    def build(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLLabelBuildResult:
        if not candles_by_symbol:
            return MLLabelBuildResult(
                rows=[],
                dropped_rows_insufficient_horizon=0,
                label_name=self._experiment_config.label_type,
            )

        benchmark_symbol = self._benchmark_symbol()
        if self._experiment_config.label_type == "risk_regime":
            builder = RiskRegimeLabelBuilder(
                horizon_days=self._experiment_config.label_horizon_days,
            )
        elif self._experiment_config.label_type == "drawdown_risk":
            builder = DrawdownRiskLabelBuilder(
                horizon_days=self._experiment_config.label_horizon_days,
                threshold=self._experiment_config.drawdown_risk_threshold,
            )
        elif self._experiment_config.label_type == "champion_success":
            builder = ChampionSuccessLabelBuilder(
                horizon_days=self._experiment_config.label_horizon_days,
            )
            return builder.build(
                feature_result.rows,
                candles_by_symbol[benchmark_symbol],
                self._champion_equity_curve,
            )
        elif self._experiment_config.label_type == "should_reduce_exposure":
            return ShouldReduceExposureLabelBuilder().build(feature_result.rows)
        else:
            raise ValueError(
                f"Unsupported ML label type: {self._experiment_config.label_type}"
            )
        return builder.build(feature_result.rows, candles_by_symbol[benchmark_symbol])

    def _benchmark_symbol(self) -> str:
        return str(
            self._config.get("ml", {}).get("benchmark_symbols", ["SPY", "QQQ"])[0]
        )
