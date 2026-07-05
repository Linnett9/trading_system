from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.artifacts import MLCoreArtifactWriter
from core.research.ml.artifacts.report_annotation import annotate_report_artifacts
from core.research.ml.data.datasets import MLDataset
from core.research.ml.features.features import MLFeatureBuildResult
from core.research.ml.features.labels import MLLabelBuildResult
from core.research.ml.reports import MLOverlayReportWriter
from core.research.ml.reports.ranking_outcomes import (
    future_drawdown_event,
    outcomes_by_feature_date,
    period_return,
)
from core.research.ml.validation import ChronologicalSplit


class MLExperimentRunnerReportingMixin:
    def _annotate_report_artifacts(self, output_dir: Path) -> None:
        annotate_report_artifacts(output_dir, self.research_label)

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
