from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.ml.artifacts import (
    MLCoreArtifactWriter,
    MLExperimentPathBuilder,
    MLExperimentPaths,
    MLFeatureCache,
)
from core.research.ml.artifacts.report_annotation import annotate_report_artifacts
from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset, write_dataset
from core.research.ml.drawdown_review import write_drawdown_event_review
from core.research.ml.pipelines import (
    MLDatasetPipeline,
    MLFeaturePipeline,
    MLFeaturePipelineResult,
    MLLabelPipeline,
    MLModelPipeline,
    MLRebalancePipeline,
)
from core.research.ml.reports import (
    MLCalibrationReportWriter,
    MLDiagnosticReportWriter,
    MLOverlayReportWriter,
)
from core.research.ml.reports.ranking_outcomes import (
    future_drawdown_event,
    outcomes_by_feature_date,
    period_return,
)
from core.research.ml.html_report import write_research_html_report
from core.research.ml.features import (
    MLFeatureBuildResult,
    write_feature_rows,
)
from core.research.ml.labels import (
    MLLabelBuildResult,
    write_label_rows,
)
from core.research.ml.validation import ChronologicalSplit


@dataclass(frozen=True)
class MLExperimentResult:
    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    confusion_matrix_path: Path
    metadata_path: Path
    model_path: Path
    features_path: Path
    feature_summary_path: Path
    labels_path: Path
    dataset_path: Path
    dataset_audit_path: Path
    walk_forward_metrics_path: Path
    threshold_sweep_path: Path
    model_comparison_path: Path
    shadow_overlay_path: Path
    holdout_shadow_overlay_path: Path
    rebalance_dataset_path: Path
    rebalance_dataset_audit_path: Path
    history_coverage_path: Path
    drawdown_event_review_path: Path
    rule_exposure_study_path: Path
    probability_calibration_path: Path
    walk_forward_probability_calibration_path: Path
    baseline_model_comparison_path: Path
    ranking_diagnostics_path: Path
    calibrated_probability_calibration_path: Path
    overlay_model_comparison_path: Path
    prediction_artifacts_path: Path
    prediction_artifacts_metadata_path: Path
    html_report_path: Path


class MLExperimentRunner:
    """Research-only ML runner. It does not affect trading decisions."""

    def __init__(self, config: dict[str, Any], feed: Any = None):
        self.config = config
        self.feed = feed
        self.experiment_config = MLExperimentConfig.from_config(config)
        self.research_label = str(
            config.get("ml", {}).get("research_label", "UNSPECIFIED_RESEARCH")
        )
        self._champion_equity_curve = []
        self._champion_rebalance_dates: set[str] = set()
        self._champion_selections: list[Any] = []
        self._history_data_metadata: dict[str, dict] = {}

    def build_expanded_rebalance_dataset(self) -> tuple[Path, Path, int]:
        feature_result, candles_by_symbol = self._build_features()
        expanded = self._build_expanded_rebalance_features(
            feature_result,
            candles_by_symbol,
        )
        dataset_path = self._rebalance_dataset_path()
        audit_path = Path(
            self.config.get("ml", {}).get(
                "expanded_rebalance_audit_path",
                "reports/ml/expanded_rebalance_dataset_audit.json",
            )
        )
        return dataset_path, audit_path, len(expanded.rows)

    def run(self) -> MLExperimentResult:
        feature_result, candles_by_symbol = self._build_features()
        if self.experiment_config.label_type == "should_reduce_exposure":
            feature_result = self._build_expanded_rebalance_features(
                feature_result,
                candles_by_symbol,
            )
        label_result = self._build_labels(feature_result, candles_by_symbol)
        prepared_dataset = self._dataset_pipeline().prepare(
            feature_result,
            label_result,
        )
        dataset = prepared_dataset.dataset
        split = prepared_dataset.split
        model = self._model_pipeline().build_model()
        self._fit_research_model(model, split.train)
        probabilities, auxiliary_predictions = self._predict_research_model(
            model,
            split.test,
            prediction_context=self._prediction_context(split),
        )
        predictions = self._predictions_from_probabilities(probabilities)

        paths = self._experiment_paths()

        write_feature_rows(paths.features_path, feature_result.rows)
        write_label_rows(paths.labels_path, label_result.rows, label_result.label_name)
        write_dataset(paths.dataset_path, dataset, label_name=label_result.label_name)
        self._write_feature_summary(paths.feature_summary_path, feature_result)
        self._write_dataset_audit(paths.dataset_audit_path, dataset, label_result)
        self._write_walk_forward_metrics(paths.walk_forward_metrics_path, dataset)
        self._write_probability_calibration(
            paths.probability_calibration_path,
            split.test.labels,
            probabilities,
        )
        self._write_calibrated_probability_calibration(
            paths.calibrated_probability_calibration_path,
            split,
            probabilities,
        )
        self._write_walk_forward_probability_calibration(
            paths.walk_forward_probability_calibration_path,
            dataset,
        )
        self._write_baseline_model_comparison(
            paths.baseline_model_comparison_path,
            dataset,
        )
        self._write_ranking_diagnostics(
            paths.ranking_diagnostics_path,
            dataset,
            self._outcomes_by_feature_date(label_result, candles_by_symbol),
        )
        self._write_threshold_sweep(
            paths.threshold_sweep_path,
            split.test,
            probabilities,
        )
        self._write_model_comparison(paths.model_comparison_path, dataset)
        self._write_shadow_overlay(paths.shadow_overlay_path, dataset)
        self._write_overlay_model_comparison(
            paths.overlay_model_comparison_path,
            dataset,
        )
        self._write_holdout_shadow_overlay(
            paths.holdout_shadow_overlay_path,
            split,
        )
        rebalance_rows = self._write_rebalance_dataset(
            paths.rebalance_dataset_path,
            paths.rebalance_dataset_audit_path,
            feature_result.rows,
            candles_by_symbol,
            paths.rule_exposure_study_path,
        )
        write_drawdown_event_review(
            paths.drawdown_event_review_path,
            rebalance_rows,
        )

        prediction_artifact_provenance = self._prediction_artifact_provenance(
            dataset,
            split,
        )
        self._write_metrics(paths.metrics_path, dataset, split, predictions)
        self._write_predictions(
            paths.predictions_path,
            split.test,
            predictions,
            probabilities,
        )
        self._write_feature_importance(
            paths.feature_importance_path,
            model.feature_importances(),
        )
        self._write_confusion_matrix(
            paths.confusion_matrix_path,
            split.test,
            predictions,
        )
        self._write_metadata(paths.metadata_path, dataset, split)
        self._write_prediction_artifacts(
            paths.prediction_artifacts_path,
            paths.prediction_artifacts_metadata_path,
            dataset,
            split,
            probabilities,
            auxiliary_predictions,
            dataset_hash=str(prediction_artifact_provenance["dataset_hash"]),
            source_dataset_row_count=int(
                prediction_artifact_provenance["source_dataset_row_count"]
            ),
            train_sample_count=int(prediction_artifact_provenance["train_sample_count"]),
            test_sample_count=int(prediction_artifact_provenance["test_sample_count"]),
            generated_at=str(prediction_artifact_provenance["generated_at"]),
        )
        model.save(paths.model_path)
        self._annotate_report_artifacts(paths.output_dir)
        write_research_html_report(paths.html_report_path, paths.output_dir)

        return MLExperimentResult(**paths.result_kwargs())

    def _experiment_path_builder(self) -> MLExperimentPathBuilder:
        return MLExperimentPathBuilder(self.config, self.experiment_config)

    def _experiment_paths(self) -> MLExperimentPaths:
        return self._experiment_path_builder().build()

    def _feature_cache(self) -> MLFeatureCache:
        return MLFeatureCache(self.config)

    def _feature_pipeline(self) -> MLFeaturePipeline:
        return MLFeaturePipeline(
            self.config,
            self.experiment_config,
            feed=self.feed,
            research_label=self.research_label,
            feature_cache=self._feature_cache(),
            path_builder=self._experiment_path_builder(),
        )

    def _dataset_pipeline(self) -> MLDatasetPipeline:
        return MLDatasetPipeline(self.experiment_config)

    def _model_pipeline(self) -> MLModelPipeline:
        return MLModelPipeline(self.config, self.experiment_config)

    def _rebalance_pipeline(self) -> MLRebalancePipeline:
        return MLRebalancePipeline(
            self.config,
            self.experiment_config,
            champion_equity_curve=self._champion_equity_curve,
            champion_selections=self._champion_selections,
            feature_cache=self._feature_cache(),
        )

    def _artifact_writer(self) -> MLCoreArtifactWriter:
        return MLCoreArtifactWriter(
            self.config,
            self.experiment_config,
            research_label=self.research_label,
            model_pipeline=self._model_pipeline(),
        )

    def _calibration_report_writer(self) -> MLCalibrationReportWriter:
        return MLCalibrationReportWriter(
            self.config,
            self.experiment_config,
            model_pipeline=self._model_pipeline(),
        )

    def _diagnostic_report_writer(self) -> MLDiagnosticReportWriter:
        return MLDiagnosticReportWriter(
            self.config,
            self.experiment_config,
            model_pipeline=self._model_pipeline(),
        )

    def _overlay_report_writer(self) -> MLOverlayReportWriter:
        return MLOverlayReportWriter(
            self.config,
            self.experiment_config,
            self._champion_equity_curve,
            self._champion_rebalance_dates,
            model_pipeline=self._model_pipeline(),
        )

    def _split_dataset(self, dataset: MLDataset) -> ChronologicalSplit:
        return self._dataset_pipeline().split(dataset)

    def _set_model_sequence_context(self, model: Any, dataset: MLDataset) -> None:
        self._model_pipeline().set_sequence_context(model, dataset)

    def _fit_research_model(self, model: Any, dataset: MLDataset) -> None:
        self._model_pipeline().fit(model, dataset)

    def _predict_research_model(
        self,
        model: Any,
        dataset: MLDataset,
        *,
        prediction_context: MLDataset | None = None,
    ) -> tuple[list[float], list[dict[str, float]]]:
        prediction = self._model_pipeline().predict(
            model,
            dataset,
            prediction_context=prediction_context,
        )
        return prediction.probabilities, prediction.auxiliary_predictions

    def _prediction_context(self, split: ChronologicalSplit) -> MLDataset:
        return self._model_pipeline().prediction_context(split)

    @staticmethod
    def _tail_rows(rows: list[Any], sample_count: int) -> list[Any]:
        return MLModelPipeline.tail_rows(rows, sample_count)

    @staticmethod
    def _concat_datasets(left: MLDataset, right: MLDataset) -> MLDataset:
        return MLModelPipeline.concat_datasets(left, right)

    def _model_component_predictions(
        self,
        model: Any,
        features: list[dict[str, float]],
    ) -> list[dict[str, float]] | None:
        return MLModelPipeline.model_component_predictions(model, features)

    def _component_probability(self, row: dict[str, float]) -> float:
        return MLModelPipeline.component_probability(row)

    def _safe_component_auxiliary_predictions(
        self,
        row: dict[str, float],
    ) -> dict[str, float]:
        return MLModelPipeline.safe_component_auxiliary_predictions(row)

    def _multitask_enabled(self) -> bool:
        return self._model_pipeline().multitask_enabled()

    def _multitask_regression_targets(self) -> list[str]:
        return self._model_pipeline().multitask_regression_targets()

    def _auxiliary_targets_for_dataset(
        self,
        dataset: MLDataset,
    ) -> dict[str, list[float | None]]:
        return self._model_pipeline().auxiliary_targets_for_dataset(dataset)

    def _model_filename(self) -> str:
        return self._experiment_path_builder().model_filename()

    def _class_weight(self) -> str | None:
        return self._model_pipeline().class_weight()

    def _predictions_from_probabilities(self, probabilities: list[float]) -> list[int]:
        return self._model_pipeline().predictions_from_probabilities(probabilities)

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

    def _annotate_report_artifacts(self, output_dir: Path) -> None:
        annotate_report_artifacts(output_dir, self.research_label)

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

    def _write_feature_summary(
        self,
        path: Path,
        feature_result: MLFeatureBuildResult,
    ) -> None:
        self._artifact_writer().write_feature_summary(path, feature_result)

    def _write_dataset_audit(
        self,
        path: Path,
        dataset: MLDataset,
        label_result: MLLabelBuildResult,
    ) -> None:
        self._artifact_writer().write_dataset_audit(path, dataset, label_result)

    def _write_walk_forward_metrics(self, path: Path, dataset: MLDataset) -> None:
        self._diagnostic_report_writer().write_walk_forward_metrics(path, dataset)

    def _write_probability_calibration(
        self,
        path: Path,
        labels: list[int],
        probabilities: list[float],
    ) -> None:
        self._calibration_report_writer().write_probability_calibration(
            path,
            labels,
            probabilities,
        )

    def _write_calibrated_probability_calibration(
        self,
        path: Path,
        split: ChronologicalSplit,
        raw_probabilities: list[float],
    ) -> None:
        self._calibration_report_writer().write_calibrated_probability_calibration(
            path,
            split,
            raw_probabilities,
        )

    def _quantile_calibrated_probabilities(
        self,
        train_labels: list[int],
        train_probabilities: list[float],
        probabilities: list[float],
        bin_count: int,
    ) -> list[float]:
        if not train_labels or not train_probabilities:
            return [0.5 for _ in probabilities]
        if len(train_labels) != len(train_probabilities):
            raise ValueError("Calibration labels and probabilities must align")
        pairs = sorted(
            (float(probability), int(label))
            for probability, label in zip(train_probabilities, train_labels)
        )
        resolved_bin_count = max(1, min(int(bin_count), len(pairs)))
        bins = []
        for index in range(resolved_bin_count):
            start = index * len(pairs) // resolved_bin_count
            end = (index + 1) * len(pairs) // resolved_bin_count
            chunk = pairs[start:end]
            if not chunk:
                continue
            observed_rate = sum(label for _, label in chunk) / len(chunk)
            bins.append({
                "lower": chunk[0][0],
                "upper": chunk[-1][0],
                "observed_rate": observed_rate,
            })
        if not bins:
            base_rate = sum(train_labels) / len(train_labels)
            return [base_rate for _ in probabilities]
        calibrated = []
        for probability in probabilities:
            value = float(probability)
            selected = bins[-1]
            for bin_payload in bins:
                if value <= bin_payload["upper"]:
                    selected = bin_payload
                    break
            calibrated.append(float(selected["observed_rate"]))
        return calibrated

    def _write_walk_forward_probability_calibration(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        self._calibration_report_writer().write_walk_forward_probability_calibration(
            path,
            dataset,
        )

    def _write_baseline_model_comparison(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        self._diagnostic_report_writer().write_baseline_model_comparison(
            path,
            dataset,
        )

    def _write_ranking_diagnostics(
        self,
        path: Path,
        dataset: MLDataset,
        outcomes_by_feature_date: dict[str, dict[str, float | None]],
    ) -> None:
        self._diagnostic_report_writer().write_ranking_diagnostics(
            path,
            dataset,
            outcomes_by_feature_date,
        )

    def _outcomes_by_feature_date(
        self,
        label_result: MLLabelBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> dict[str, dict[str, float | None]]:
        return outcomes_by_feature_date(
            self.config,
            label_result,
            candles_by_symbol,
            self._champion_equity_curve,
        )

    def _period_return(
        self,
        values_by_date: dict[str, float],
        start_date: str,
        end_date: str,
    ) -> float | None:
        return period_return(values_by_date, start_date, end_date)

    def _future_drawdown_event(
        self,
        dates: list[str],
        values_by_date: dict[str, float],
        index_by_date: dict[str, int],
        start_date: str,
        end_date: str,
    ) -> float | None:
        return future_drawdown_event(
            dates,
            values_by_date,
            index_by_date,
            start_date,
            end_date,
        )

    def _mean_probability_summary(self, summaries: list[dict]) -> dict[str, float | None]:
        return self._diagnostic_report_writer().mean_probability_summary(summaries)

    def _rolling_base_rate_lookback_samples(self) -> int:
        return self._diagnostic_report_writer().rolling_base_rate_lookback_samples()

    def _ranking_quantile_count(self) -> int:
        return self._diagnostic_report_writer().ranking_quantile_count()

    def _calibration_bin_count(self) -> int:
        return self._calibration_report_writer().calibration_bin_count()

    def _write_threshold_sweep(
        self,
        path: Path,
        dataset: MLDataset,
        probabilities: list[float],
    ) -> None:
        self._diagnostic_report_writer().write_threshold_sweep(
            path,
            dataset,
            probabilities,
        )

    def _write_model_comparison(self, path: Path, dataset: MLDataset) -> None:
        self._diagnostic_report_writer().write_model_comparison(path, dataset)

    def _write_overlay_model_comparison(self, path: Path, dataset: MLDataset) -> None:
        self._overlay_report_writer().write_overlay_model_comparison(
            path,
            dataset,
        )

    def _overlay_probabilities(self, probabilities: list[float]) -> list[float]:
        return [float(probability) for probability in probabilities]

    def _overlay_probability_label(self) -> str:
        return self._overlay_report_writer().overlay_probability_label()

    def _overlay_fold_summary(self, folds: list[dict]) -> dict[str, float | int | None]:
        return self._overlay_report_writer().overlay_fold_summary(folds)

    def _unique_strings(self, values: list[Any]) -> list[str]:
        return MLOverlayReportWriter.unique_strings(values)

    def _write_shadow_overlay(self, path: Path, dataset: MLDataset) -> None:
        self._overlay_report_writer().write_shadow_overlay(path, dataset)

    def _write_holdout_shadow_overlay(
        self,
        path: Path,
        split: ChronologicalSplit,
    ) -> None:
        self._overlay_report_writer().write_holdout_shadow_overlay(path, split)

    def _mean_metric(self, metrics: list[dict], key: str) -> float | None:
        values = [item[key] for item in metrics if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    def _standard_deviation(self, values: list[float]) -> float:
        return MLCoreArtifactWriter.standard_deviation(values)

    def _correlation(self, left: list[float], right: list[float]) -> float:
        return MLCoreArtifactWriter.correlation(left, right)

    def _is_numeric_column(
        self,
        rows: list[dict[str, float | str]],
        name: str,
    ) -> bool:
        return MLCoreArtifactWriter.is_numeric_column(rows, name)

    def _write_metrics(
        self,
        path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
        predictions: list[int],
    ) -> None:
        self._artifact_writer().write_metrics(path, dataset, split, predictions)

    def _baseline_metrics(self, split: ChronologicalSplit) -> dict[str, dict]:
        return self._artifact_writer().baseline_metrics(split)

    def _write_predictions(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
        probabilities: list[float],
    ) -> None:
        self._artifact_writer().write_predictions(
            path,
            dataset,
            predictions,
            probabilities,
        )

    def _write_prediction_artifacts(
        self,
        csv_path: Path,
        metadata_path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
        holdout_probabilities: list[float],
        holdout_auxiliary_predictions: list[dict[str, float]] | None = None,
        *,
        dataset_hash: str | None = None,
        source_dataset_row_count: int | None = None,
        train_sample_count: int | None = None,
        test_sample_count: int | None = None,
        generated_at: str | None = None,
    ) -> None:
        self._artifact_writer().write_prediction_artifacts(
            csv_path,
            metadata_path,
            dataset,
            split,
            holdout_probabilities,
            holdout_auxiliary_predictions,
            dataset_hash=dataset_hash,
            source_dataset_row_count=source_dataset_row_count,
            train_sample_count=train_sample_count,
            test_sample_count=test_sample_count,
            generated_at=generated_at,
        )

    def _prediction_artifact_provenance(
        self,
        dataset: MLDataset,
        split: ChronologicalSplit,
        *,
        dataset_hash: str | None = None,
        source_dataset_row_count: int | None = None,
        train_sample_count: int | None = None,
        test_sample_count: int | None = None,
        generated_at: str | None = None,
    ) -> dict[str, str | int]:
        return self._artifact_writer().prediction_artifact_provenance(
            dataset,
            split,
            dataset_hash=dataset_hash,
            source_dataset_row_count=source_dataset_row_count,
            train_sample_count=train_sample_count,
            test_sample_count=test_sample_count,
            generated_at=generated_at,
        )

    def _prediction_artifact_rows(
        self,
        dataset: MLDataset,
        probabilities: list[float],
        auxiliary_predictions: list[dict[str, float]] | None,
        split_name: str,
        fold: int | str,
        provenance: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._artifact_writer().prediction_artifact_rows(
            dataset,
            probabilities,
            auxiliary_predictions,
            split_name,
            fold,
            provenance,
        )

    def _prediction_artifact_model_name(self) -> str:
        return self._artifact_writer().prediction_artifact_model_name()

    def _prediction_artifact_auxiliary_values(
        self,
        dataset: MLDataset,
        index: int,
        auxiliary_prediction: dict[str, float],
    ) -> dict[str, float | str]:
        return self._artifact_writer().prediction_artifact_auxiliary_values(
            dataset,
            index,
            auxiliary_prediction,
        )

    @staticmethod
    def _prediction_artifact_auxiliary_fieldnames(
        rows: list[dict[str, Any]],
    ) -> list[str]:
        return MLCoreArtifactWriter.prediction_artifact_auxiliary_fieldnames(rows)

    def _write_feature_importance(
        self,
        path: Path,
        feature_importances: dict[str, float],
    ) -> None:
        self._artifact_writer().write_feature_importance(path, feature_importances)

    def _write_confusion_matrix(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
    ) -> None:
        self._artifact_writer().write_confusion_matrix(path, dataset, predictions)

    def _write_metadata(
        self,
        path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
    ) -> None:
        self._artifact_writer().write_metadata(path, dataset, split)

    def _dataset_hash(self, dataset: MLDataset) -> str:
        return self._artifact_writer().dataset_hash(dataset)

    def _source_dataset_hash(self, dataset: MLDataset) -> str:
        return self._artifact_writer().source_dataset_hash(dataset)

    def _source_dataset_identity(self, dataset: MLDataset) -> dict[str, Any]:
        return self._artifact_writer().source_dataset_identity(dataset)

    def _model_input_hash(self, dataset: MLDataset) -> str:
        return self._artifact_writer().model_input_hash(dataset)

    def _hash_payload(self, payload: Any) -> str:
        return MLCoreArtifactWriter.hash_payload(payload)

    def _git_commit(self) -> str | None:
        return MLCoreArtifactWriter.git_commit()
