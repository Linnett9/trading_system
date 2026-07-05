from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.artifacts import MLFeatureCache
from core.research.ml.features.features import MLFeatureBuildResult
from core.research.ml.features.labels import MLLabelBuildResult
from core.research.ml.pipelines import (
    MLFeaturePipelineResult,
    MLLabelPipeline,
    MLRebalancePipeline,
)


class MLExperimentRunnerFeatureMixin:
    def _build_features(
        self,
    ) -> tuple[MLFeatureBuildResult, dict[str, list[Any]]]:
        result = self._feature_pipeline().build()
        self._apply_feature_pipeline_result(result)
        return result.feature_result, result.candles_by_symbol

    def _apply_feature_pipeline_result(
        self,
        result: MLFeaturePipelineResult,
    ) -> None:
        if result.champion_state_updated:
            self._champion_equity_curve = result.champion_equity_curve
            self._champion_selections = result.champion_selections
            self._champion_rebalance_dates = result.champion_rebalance_dates
        if result.history_data_metadata_updated:
            self._history_data_metadata = result.history_data_metadata

    def _validate_history_coverage(
        self,
        candles_by_symbol: dict[str, list[Any]],
        ml_config: dict[str, Any],
    ) -> None:
        self._feature_pipeline().validate_history_coverage(
            candles_by_symbol,
            ml_config,
            self._history_data_metadata,
        )

    def _build_labels(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLLabelBuildResult:
        return MLLabelPipeline(
            self.config,
            self.experiment_config,
            self._champion_equity_curve,
        ).build(feature_result, candles_by_symbol)

    def _feature_symbols(self) -> list[str]:
        return self._feature_pipeline().feature_symbols()

    def _expanded_rebalance_universe_symbols(self, dual_momentum: dict) -> list[str]:
        return self._feature_pipeline().expanded_rebalance_universe_symbols(
            dual_momentum,
        )

    def _features_path(self) -> Path:
        return self._experiment_path_builder().features_path()

    def _labels_path(self) -> Path:
        return self._experiment_path_builder().labels_path()

    def _dataset_path(self) -> Path:
        return self._experiment_path_builder().dataset_path()

    def _rebalance_dataset_path(self) -> Path:
        return self._experiment_path_builder().rebalance_dataset_path()

    def _build_expanded_rebalance_features(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLFeatureBuildResult:
        return self._rebalance_pipeline().build_expanded_rebalance_features(
            feature_result,
            candles_by_symbol,
        )

    def _load_cached_feature_rows(
        self,
        path: Path,
        cache_key: str,
    ) -> MLFeatureBuildResult | None:
        return self._feature_cache().load_feature_rows(path, cache_key)

    def _write_cached_feature_rows(
        self,
        path: Path,
        feature_result: MLFeatureBuildResult,
        cache_key: str,
    ) -> None:
        self._feature_cache().write_feature_rows(path, feature_result, cache_key)

    def _load_cached_expanded_rebalance_rows(
        self,
        path: Path,
        cache_key: str,
        dropped_rows: int,
    ) -> MLFeatureBuildResult | None:
        return self._feature_cache().load_expanded_rebalance_rows(
            path,
            cache_key,
            dropped_rows,
        )

    def _feature_cache_key(
        self,
        symbols: list[str],
        benchmark_symbols: tuple[Any, ...],
        lookback_days: int,
        candles_by_symbol: dict[str, list[Any]],
    ) -> str:
        return self._feature_cache().feature_cache_key(
            symbols,
            benchmark_symbols,
            lookback_days,
            candles_by_symbol,
        )

    def _expanded_rebalance_cache_key(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> str:
        return self._rebalance_pipeline().expanded_rebalance_cache_key(
            feature_result,
            candles_by_symbol,
        )

    def _candles_cache_summary(
        self,
        candles_by_symbol: dict[str, list[Any]],
    ) -> dict[str, dict[str, Any]]:
        return MLFeatureCache.candles_cache_summary(candles_by_symbol)

    def _read_cache_metadata(self, path: Path) -> dict[str, Any]:
        return self._feature_cache().read_metadata(path)

    def _write_cache_metadata(
        self,
        path: Path,
        cache_key: str,
        metadata: dict[str, Any],
    ) -> None:
        self._feature_cache().write_metadata(path, cache_key, metadata)

    @staticmethod
    def _cache_metadata_path(path: Path) -> Path:
        return MLFeatureCache.metadata_path(path)

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        return MLFeatureCache.read_csv_rows(path)

    def _rows_hash(self, rows: list[dict[str, Any]]) -> str:
        return MLFeatureCache.rows_hash(rows)

    def _write_rebalance_dataset(
        self,
        path: Path,
        audit_path: Path,
        feature_rows: list[dict[str, float | str]],
        candles_by_symbol: dict[str, list[Any]],
        rule_study_path: Path,
    ) -> list[dict[str, float | str]]:
        return self._rebalance_pipeline().write_rebalance_dataset(
            path,
            audit_path,
            feature_rows,
            candles_by_symbol,
            rule_study_path,
        )

    def _sector_by_symbol(self) -> dict[str, str]:
        return self._rebalance_pipeline().sector_by_symbol()

    def _row_rate(self, rows: list[dict[str, float | str]], key: str) -> float | None:
        return MLRebalancePipeline.row_rate(rows, key)
