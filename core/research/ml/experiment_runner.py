from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from core.research.ml.config import MLExperimentConfig
from core.research.ml.calibration import build_probability_calibration
from core.research.ml.diagnostics import (
    build_ranking_diagnostics,
    probability_summary,
    rolling_base_rate_probabilities,
)
from core.research.ml.datasets import MLDataset, build_dataset, write_dataset
from core.research.ml.drawdown_review import write_drawdown_event_review
from core.research.ml.history_coverage import (
    assess_history_coverage,
    write_history_coverage_report,
)
from core.research.ml.evaluation import classification_metrics
from application.services.market_data_loader import load_candles_with_metadata
from core.research.ml.features import (
    HistoricalFeatureBuilder,
    MLFeatureBuildResult,
    add_champion_state_features,
    write_feature_rows,
)
from core.research.ml.labels import (
    ChampionSuccessLabelBuilder,
    DrawdownRiskLabelBuilder,
    MLLabelBuildResult,
    RiskRegimeLabelBuilder,
    write_label_rows,
)
from core.research.ml.models import build_ml_model
from core.research.ml.overlay import simulate_shadow_overlay
from core.research.ml.rebalance_dataset import (
    build_champion_rebalance_rows,
    write_rebalance_dataset,
)
from core.research.ml.rule_overlay import (
    run_drawdown_risk_diagnostics,
    run_rule_exposure_study,
    run_volatility_managed_walk_forward,
)
from core.research.ml.sector_reference import load_sector_by_symbol
from core.research.ml.validation import (
    ChronologicalSplit,
    chronological_holdout,
    rolling_walk_forward,
)


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

    def run(self) -> MLExperimentResult:
        feature_result, candles_by_symbol = self._build_features()
        label_result = self._build_labels(feature_result, candles_by_symbol)
        dataset = build_dataset(
            feature_result.rows,
            label_result.rows,
            label_name=label_result.label_name,
        )
        split = self._split_dataset(dataset)
        model = build_ml_model(
            self.experiment_config.model_type,
            random_seed=self.experiment_config.random_seed,
            class_weight=self._class_weight(),
        )
        model.fit(split.train.features, split.train.labels)
        probabilities = model.predict_proba(split.test.features)
        predictions = self._predictions_from_probabilities(probabilities)

        output_dir = Path(self.experiment_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        predictions_path = output_dir / "predictions.csv"
        feature_importance_path = output_dir / "feature_importance.csv"
        confusion_matrix_path = output_dir / "confusion_matrix.csv"
        metadata_path = output_dir / "metadata.json"
        model_path = output_dir / self._model_filename()
        features_path = self._features_path()
        feature_summary_path = output_dir / "feature_summary.json"
        labels_path = self._labels_path()
        dataset_path = self._dataset_path()
        dataset_audit_path = output_dir / "dataset_audit.json"
        walk_forward_metrics_path = output_dir / "walk_forward_metrics.json"
        threshold_sweep_path = output_dir / "threshold_sweep.json"
        model_comparison_path = output_dir / "model_comparison.json"
        shadow_overlay_path = output_dir / "shadow_overlay.json"
        holdout_shadow_overlay_path = output_dir / "holdout_shadow_overlay.json"
        rebalance_dataset_path = self._rebalance_dataset_path()
        rebalance_dataset_audit_path = output_dir / "rebalance_dataset_audit.json"
        history_coverage_path = output_dir / "history_coverage.json"
        drawdown_event_review_path = output_dir / "drawdown_event_review.json"
        rule_exposure_study_path = output_dir / "rule_exposure_study.json"
        probability_calibration_path = output_dir / "probability_calibration.json"
        walk_forward_probability_calibration_path = (
            output_dir / "walk_forward_probability_calibration.json"
        )
        baseline_model_comparison_path = output_dir / "baseline_model_comparison.json"
        ranking_diagnostics_path = output_dir / "ranking_diagnostics.json"

        write_feature_rows(features_path, feature_result.rows)
        write_label_rows(labels_path, label_result.rows, label_result.label_name)
        write_dataset(dataset_path, dataset, label_name=label_result.label_name)
        self._write_feature_summary(feature_summary_path, feature_result)
        self._write_dataset_audit(dataset_audit_path, dataset, label_result)
        self._write_walk_forward_metrics(walk_forward_metrics_path, dataset)
        self._write_probability_calibration(
            probability_calibration_path,
            split.test.labels,
            probabilities,
        )
        self._write_walk_forward_probability_calibration(
            walk_forward_probability_calibration_path,
            dataset,
        )
        self._write_baseline_model_comparison(
            baseline_model_comparison_path,
            dataset,
        )
        self._write_ranking_diagnostics(
            ranking_diagnostics_path,
            dataset,
            self._outcomes_by_feature_date(label_result, candles_by_symbol),
        )
        self._write_threshold_sweep(threshold_sweep_path, split.test, probabilities)
        self._write_model_comparison(model_comparison_path, dataset)
        self._write_shadow_overlay(shadow_overlay_path, dataset)
        self._write_holdout_shadow_overlay(
            holdout_shadow_overlay_path,
            split,
        )
        rebalance_rows = self._write_rebalance_dataset(
            rebalance_dataset_path,
            rebalance_dataset_audit_path,
            feature_result.rows,
            candles_by_symbol,
            rule_exposure_study_path,
        )
        write_drawdown_event_review(drawdown_event_review_path, rebalance_rows)

        self._write_metrics(metrics_path, split, predictions)
        self._write_predictions(predictions_path, split.test, predictions, probabilities)
        self._write_feature_importance(feature_importance_path, model.feature_importances())
        self._write_confusion_matrix(confusion_matrix_path, split.test, predictions)
        self._write_metadata(metadata_path, dataset, split)
        model.save(model_path)
        self._annotate_report_artifacts(output_dir)

        return MLExperimentResult(
            output_dir=output_dir,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            feature_importance_path=feature_importance_path,
            confusion_matrix_path=confusion_matrix_path,
            metadata_path=metadata_path,
            model_path=model_path,
            features_path=features_path,
            feature_summary_path=feature_summary_path,
            labels_path=labels_path,
            dataset_path=dataset_path,
            dataset_audit_path=dataset_audit_path,
            walk_forward_metrics_path=walk_forward_metrics_path,
            threshold_sweep_path=threshold_sweep_path,
            model_comparison_path=model_comparison_path,
            shadow_overlay_path=shadow_overlay_path,
            holdout_shadow_overlay_path=holdout_shadow_overlay_path,
            rebalance_dataset_path=rebalance_dataset_path,
            rebalance_dataset_audit_path=rebalance_dataset_audit_path,
            history_coverage_path=history_coverage_path,
            drawdown_event_review_path=drawdown_event_review_path,
            rule_exposure_study_path=rule_exposure_study_path,
            probability_calibration_path=probability_calibration_path,
            walk_forward_probability_calibration_path=(
                walk_forward_probability_calibration_path
            ),
            baseline_model_comparison_path=baseline_model_comparison_path,
            ranking_diagnostics_path=ranking_diagnostics_path,
        )

    def _split_dataset(self, dataset: MLDataset) -> ChronologicalSplit:
        return chronological_holdout(
            dataset,
            test_fraction=self.experiment_config.test_fraction,
            train_start=self.experiment_config.train_start,
            train_end=self.experiment_config.train_end,
            test_start=self.experiment_config.test_start,
            test_end=self.experiment_config.test_end,
        )

    def _model_filename(self) -> str:
        if self.experiment_config.model_type == "noop":
            return "model.json"
        return "model.joblib"

    def _class_weight(self) -> str | None:
        return "balanced" if self.experiment_config.class_weight_balanced else None

    def _predictions_from_probabilities(self, probabilities: list[float]) -> list[int]:
        return [
            int(probability >= self.experiment_config.decision_threshold)
            for probability in probabilities
        ]

    def _build_features(
        self,
    ) -> tuple[MLFeatureBuildResult, dict[str, list[Any]]]:
        if self.feed is None:
            return MLFeatureBuildResult(rows=[], dropped_rows=0, date_range=None), {}

        symbols = self._feature_symbols()
        loaded_candles = {
            symbol: load_candles_with_metadata(symbol, self.config, self.feed)
            for symbol in symbols
        }
        candles_by_symbol = {
            symbol: result.candles for symbol, result in loaded_candles.items()
        }
        self._history_data_metadata = {
            symbol: result.metadata for symbol, result in loaded_candles.items()
        }
        ml_config = self.config.get("ml", {})
        self._validate_history_coverage(candles_by_symbol, ml_config)
        benchmark_symbols = tuple(ml_config.get("benchmark_symbols", ["SPY", "QQQ"]))
        if len(benchmark_symbols) != 2:
            raise ValueError("ml.benchmark_symbols must contain exactly SPY and QQQ")

        builder = HistoricalFeatureBuilder(
            benchmark_symbols=(str(benchmark_symbols[0]), str(benchmark_symbols[1])),
            lookback_days=int(ml_config.get("feature_lookback_days", 252)),
        )
        feature_result = builder.build(candles_by_symbol)
        if (
            ml_config.get("include_champion_state_features", True)
            and self.config.get("research", {}).get("dual_momentum")
        ):
            from application.services.dual_momentum_config import (
                active_dual_momentum_config,
            )
            from core.research.dual_momentum_factory import build_dual_momentum_tester

            champion_config = active_dual_momentum_config(self.config)
            champion_result = build_dual_momentum_tester(
                self.config,
                champion_config,
            ).run(candles_by_symbol)
            self._champion_equity_curve = champion_result.result.equity_curve
            self._champion_selections = champion_result.selections
            self._champion_rebalance_dates = {
                selection.timestamp.date().isoformat()
                for selection in champion_result.selections
            }
            feature_result = MLFeatureBuildResult(
                rows=add_champion_state_features(
                    feature_result.rows,
                    champion_result.selections,
                ),
                dropped_rows=feature_result.dropped_rows,
                date_range=feature_result.date_range,
            )
        return feature_result, candles_by_symbol

    def _validate_history_coverage(
        self,
        candles_by_symbol: dict[str, list[Any]],
        ml_config: dict[str, Any],
    ) -> None:
        required_years = ml_config.get("minimum_history_years")
        if required_years is None:
            return
        report = assess_history_coverage(
            candles_by_symbol,
            required_years=int(required_years),
            tolerance_days=int(ml_config.get("history_coverage_tolerance_days", 10)),
            source_metadata=self._history_data_metadata,
        )
        output_dir = Path(self.experiment_config.output_dir)
        report_path = output_dir / "history_coverage.json"
        allow_short_history = bool(
            ml_config.get("allow_short_history_for_smoke_test", False)
        )
        report["research_label"] = self.research_label
        report["short_history_allowed_for_smoke_test"] = allow_short_history
        write_history_coverage_report(report_path, report)
        if not report["coverage_sufficient"] and not allow_short_history:
            raise RuntimeError(
                "ML research stopped: historical coverage is insufficient. "
                f"Required {report['required_years']} years, but the common range is "
                f"{report['common_start_date']} to {report['common_end_date']}. "
                f"See {report_path}."
            )

    def _annotate_report_artifacts(self, output_dir: Path) -> None:
        warning = (
            "Short-history ML smoke test only. Not valid for production conclusions."
            if self.research_label == "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
            else None
        )
        for path in output_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["research_label"] = self.research_label
                payload["production_validated"] = False
                if warning:
                    payload["warning"] = warning
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for path in output_dir.glob("*.csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            if "research_label" not in fieldnames:
                fieldnames.append("research_label")
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row["research_label"] = self.research_label
                    writer.writerow(row)

    def _build_labels(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLLabelBuildResult:
        if not candles_by_symbol:
            return MLLabelBuildResult(
                rows=[],
                dropped_rows_insufficient_horizon=0,
                label_name=self.experiment_config.label_type,
            )

        benchmark_symbol = str(
            self.config.get("ml", {}).get("benchmark_symbols", ["SPY", "QQQ"])[0]
        )
        if self.experiment_config.label_type == "risk_regime":
            builder = RiskRegimeLabelBuilder(
                horizon_days=self.experiment_config.label_horizon_days,
            )
        elif self.experiment_config.label_type == "drawdown_risk":
            builder = DrawdownRiskLabelBuilder(
                horizon_days=self.experiment_config.label_horizon_days,
                threshold=self.experiment_config.drawdown_risk_threshold,
            )
        elif self.experiment_config.label_type == "champion_success":
            builder = ChampionSuccessLabelBuilder(
                horizon_days=self.experiment_config.label_horizon_days,
            )
            return builder.build(
                feature_result.rows,
                candles_by_symbol[benchmark_symbol],
                self._champion_equity_curve,
            )
        else:
            raise ValueError(
                f"Unsupported ML label type: {self.experiment_config.label_type}"
            )
        return builder.build(feature_result.rows, candles_by_symbol[benchmark_symbol])

    def _feature_symbols(self) -> list[str]:
        dual_momentum = self.config.get("research", {}).get("dual_momentum", {})
        symbols = dual_momentum.get(
            "symbols",
            self.config.get("backtest", {}).get("symbols", []),
        )
        benchmarks = self.config.get("ml", {}).get("benchmark_symbols", ["SPY", "QQQ"])
        return list(dict.fromkeys([*symbols, *benchmarks]))

    def _features_path(self) -> Path:
        cache_config = self.config.get("cache", {})
        return Path(cache_config.get("ml_dir", "cache/ml")) / "features.csv"

    def _labels_path(self) -> Path:
        cache_config = self.config.get("cache", {})
        filename = f"labels_{self.experiment_config.label_type}.csv"
        return Path(cache_config.get("ml_dir", "cache/ml")) / filename

    def _dataset_path(self) -> Path:
        cache_config = self.config.get("cache", {})
        filename = f"dataset_{self.experiment_config.label_type}.csv"
        return Path(cache_config.get("ml_dir", "cache/ml")) / filename

    def _rebalance_dataset_path(self) -> Path:
        cache_config = self.config.get("cache", {})
        return Path(cache_config.get("ml_dir", "cache/ml")) / "champion_rebalance_dataset.csv"

    def _write_rebalance_dataset(
        self,
        path: Path,
        audit_path: Path,
        feature_rows: list[dict[str, float | str]],
        candles_by_symbol: dict[str, list[Any]],
        rule_study_path: Path,
    ) -> list[dict[str, float | str]]:
        benchmark = str(self.config.get("ml", {}).get("benchmark_symbols", ["SPY"])[0])
        rows = build_champion_rebalance_rows(
            feature_rows,
            self._champion_selections,
            self._champion_equity_curve,
            candles_by_symbol.get(benchmark, []),
            self.experiment_config.label_horizon_days,
            candles_by_symbol=candles_by_symbol,
            sector_by_symbol=self._sector_by_symbol(),
        )
        write_rebalance_dataset(path, rows)
        audit_path.write_text(json.dumps({
            "row_count": len(rows),
            "good_period_rate": self._row_rate(rows, "good_period"),
            "bad_period_rate": self._row_rate(rows, "bad_period"),
            "underperforms_spy_rate": self._row_rate(rows, "underperforms_spy"),
            "drawdown_event_rate": self._row_rate(rows, "drawdown_event"),
            "history_years": self.config.get("backtest", {}).get("years"),
            "recommended_generalization_years": self.config.get("ml", {}).get(
                "research_years", 10,
            ),
            "minimum_history_years": self.config.get("ml", {}).get(
                "minimum_history_years",
            ),
            "sector_reference_path": self.config.get("ml", {}).get(
                "sector_reference_path",
            ),
            "research_only": True,
        }, indent=2), encoding="utf-8")
        rule_study_path.write_text(json.dumps({
            "mode": "rule_based_research_only",
            "rules": run_rule_exposure_study(rows, transaction_cost_bps=5.0),
            "volatility_managed_walk_forward": run_volatility_managed_walk_forward(
                rows,
                transaction_cost_bps=5.0,
            ),
            "drawdown_risk_diagnostics": run_drawdown_risk_diagnostics(rows),
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")
        return rows

    def _sector_by_symbol(self) -> dict[str, str]:
        ml_config = self.config.get("ml", {})
        return load_sector_by_symbol(
            ml_config.get("sector_reference_path"),
            inline_mapping=dict(ml_config.get("sector_by_symbol", {})),
        )

    def _row_rate(self, rows: list[dict[str, float | str]], key: str) -> float | None:
        return sum(int(row[key]) for row in rows) / len(rows) if rows else None

    def _write_feature_summary(
        self,
        path: Path,
        feature_result: MLFeatureBuildResult,
    ) -> None:
        rows = feature_result.rows
        numeric_columns = [
            name for name in (rows[0] if rows else {}) if name != "feature_date"
        ]
        summary = {
            "row_count": len(rows),
            "dropped_rows_insufficient_lookback": feature_result.dropped_rows,
            "date_range": feature_result.date_range,
            "missing_values": {
                name: sum(row.get(name) is None for row in rows)
                for name in numeric_columns
            },
            "means": {
                name: sum(float(row[name]) for row in rows) / len(rows)
                for name in numeric_columns
            } if rows else {},
            "standard_deviations": {
                name: self._standard_deviation([float(row[name]) for row in rows])
                for name in numeric_columns
            } if rows else {},
            "correlation_matrix": {
                left: {
                    right: self._correlation(
                        [float(row[left]) for row in rows],
                        [float(row[right]) for row in rows],
                    )
                    for right in numeric_columns
                }
                for left in numeric_columns
            } if rows else {},
        }
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_dataset_audit(
        self,
        path: Path,
        dataset: MLDataset,
        label_result: MLLabelBuildResult,
    ) -> None:
        positive_labels = sum(dataset.labels)
        sample_count = dataset.sample_count
        payload = {
            "sample_count": sample_count,
            "feature_count": dataset.feature_count,
            "date_coverage": (
                [dataset.feature_dates[0], dataset.feature_dates[-1]]
                if dataset.feature_dates
                else None
            ),
            "class_balance": {
                "positive": positive_labels,
                "negative": sample_count - positive_labels,
                "positive_rate": positive_labels / sample_count if sample_count else None,
            },
            "dropped_rows_insufficient_label_horizon": (
                label_result.dropped_rows_insufficient_horizon
            ),
            "leakage_check_passed": all(
                feature_date < label_start <= label_end
                for feature_date, label_start, label_end in zip(
                    dataset.feature_dates,
                    dataset.label_start_dates,
                    dataset.label_end_dates,
                )
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_walk_forward_metrics(self, path: Path, dataset: MLDataset) -> None:
        folds = rolling_walk_forward(
            dataset,
            fold_count=self.experiment_config.walk_forward_folds,
        )
        payload_folds = []
        for fold in folds:
            model = build_ml_model(
                self.experiment_config.model_type,
                random_seed=self.experiment_config.random_seed,
                class_weight=self._class_weight(),
            )
            model.fit(fold.split.train.features, fold.split.train.labels)
            predictions = self._predictions_from_probabilities(
                model.predict_proba(fold.split.test.features)
            )
            payload_folds.append({
                "fold": fold.fold_number,
                "train_sample_count": fold.split.train.sample_count,
                "test_sample_count": fold.split.test.sample_count,
                "test_start_date": fold.split.test_start_date,
                "purged_train_samples": fold.split.purged_train_samples,
                "metrics": classification_metrics(fold.split.test.labels, predictions),
                "baselines": self._baseline_metrics(fold.split),
            })
        path.write_text(json.dumps({
            "model_type": self.experiment_config.model_type,
            "fold_count": len(payload_folds),
            "folds": payload_folds,
            "research_only": True,
        }, indent=2), encoding="utf-8")

    def _write_probability_calibration(
        self,
        path: Path,
        labels: list[int],
        probabilities: list[float],
    ) -> None:
        path.write_text(json.dumps({
            "evaluation": "chronological_holdout",
            "model_type": self.experiment_config.model_type,
            "calibration": build_probability_calibration(
                labels,
                probabilities,
                bin_count=self._calibration_bin_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def _write_walk_forward_probability_calibration(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        fold_payloads = []
        all_labels: list[int] = []
        all_probabilities: list[float] = []
        for fold in rolling_walk_forward(
            dataset,
            fold_count=self.experiment_config.walk_forward_folds,
        ):
            model = build_ml_model(
                self.experiment_config.model_type,
                random_seed=self.experiment_config.random_seed,
                class_weight=self._class_weight(),
            )
            model.fit(fold.split.train.features, fold.split.train.labels)
            probabilities = model.predict_proba(fold.split.test.features)
            all_labels.extend(fold.split.test.labels)
            all_probabilities.extend(probabilities)
            fold_payloads.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "calibration": build_probability_calibration(
                    fold.split.test.labels,
                    probabilities,
                    bin_count=self._calibration_bin_count(),
                ),
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "model_type": self.experiment_config.model_type,
            "fold_count": len(fold_payloads),
            "folds": fold_payloads,
            "pooled_out_of_sample_calibration": build_probability_calibration(
                all_labels,
                all_probabilities,
                bin_count=self._calibration_bin_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def _write_baseline_model_comparison(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        model_types = list(self.config.get("ml", {}).get(
            "comparison_models",
            ["logistic_regression", "random_forest", "gradient_boosting"],
        ))
        fold_payloads = []
        summaries_by_name: dict[str, list[dict]] = {}
        for fold in rolling_walk_forward(
            dataset,
            fold_count=self.experiment_config.walk_forward_folds,
        ):
            static_probability = (
                sum(fold.split.train.labels) / fold.split.train.sample_count
            )
            static_probabilities = [static_probability] * fold.split.test.sample_count
            static_summary = probability_summary(
                fold.split.test.labels,
                static_probabilities,
                decision_threshold=self.experiment_config.decision_threshold,
            )
            rolling_probabilities = rolling_base_rate_probabilities(
                fold.split.train.labels,
                fold.split.train.label_end_dates,
                fold.split.test.feature_dates,
                fold.split.test.labels,
                fold.split.test.label_end_dates,
                lookback_samples=self._rolling_base_rate_lookback_samples(),
            )
            baseline_summaries = {
                "static_base_rate": static_summary,
                "rolling_base_rate": probability_summary(
                    fold.split.test.labels,
                    rolling_probabilities,
                    decision_threshold=self.experiment_config.decision_threshold,
                    reference_brier_score=static_summary["brier_score"],
                ),
                "always_positive": probability_summary(
                    fold.split.test.labels,
                    [1.0] * fold.split.test.sample_count,
                    decision_threshold=self.experiment_config.decision_threshold,
                    reference_brier_score=static_summary["brier_score"],
                ),
            }
            model_summaries = []
            for model_type in model_types:
                model = build_ml_model(
                    model_type,
                    random_seed=self.experiment_config.random_seed,
                    class_weight=self._class_weight(),
                )
                model.fit(fold.split.train.features, fold.split.train.labels)
                summary = probability_summary(
                    fold.split.test.labels,
                    model.predict_proba(fold.split.test.features),
                    decision_threshold=self.experiment_config.decision_threshold,
                    reference_brier_score=baseline_summaries["rolling_base_rate"][
                        "brier_score"
                    ],
                )
                summary["brier_skill_vs_static_base_rate"] = (
                    1 - summary["brier_score"] / static_summary["brier_score"]
                    if static_summary["brier_score"] else None
                )
                summaries_by_name.setdefault(model_type, []).append(summary)
                model_summaries.append({"model_type": model_type, **summary})
            for name, summary in baseline_summaries.items():
                summaries_by_name.setdefault(name, []).append(summary)
            fold_payloads.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "baselines": baseline_summaries,
                "models": model_summaries,
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "rolling_base_rate_lookback_samples": (
                self._rolling_base_rate_lookback_samples()
            ),
            "fold_count": len(fold_payloads),
            "folds": fold_payloads,
            "mean_metrics_by_predictor": {
                name: self._mean_probability_summary(summaries)
                for name, summaries in summaries_by_name.items()
            },
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def _write_ranking_diagnostics(
        self,
        path: Path,
        dataset: MLDataset,
        outcomes_by_feature_date: dict[str, dict[str, float | None]],
    ) -> None:
        folds = []
        all_labels: list[int] = []
        all_probabilities: list[float] = []
        all_outcomes: list[dict[str, float | None]] = []
        for fold in rolling_walk_forward(
            dataset,
            fold_count=self.experiment_config.walk_forward_folds,
        ):
            model = build_ml_model(
                self.experiment_config.model_type,
                random_seed=self.experiment_config.random_seed,
                class_weight=self._class_weight(),
            )
            model.fit(fold.split.train.features, fold.split.train.labels)
            probabilities = model.predict_proba(fold.split.test.features)
            outcomes = [
                outcomes_by_feature_date.get(feature_date, {})
                for feature_date in fold.split.test.feature_dates
            ]
            all_labels.extend(fold.split.test.labels)
            all_probabilities.extend(probabilities)
            all_outcomes.extend(outcomes)
            folds.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "diagnostics": build_ranking_diagnostics(
                    fold.split.test.labels,
                    probabilities,
                    outcomes,
                    quantile_count=self._ranking_quantile_count(),
                ),
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "model_type": self.experiment_config.model_type,
            "quantile_count": self._ranking_quantile_count(),
            "fold_count": len(folds),
            "folds": folds,
            "pooled_out_of_sample_diagnostics": build_ranking_diagnostics(
                all_labels,
                all_probabilities,
                all_outcomes,
                quantile_count=self._ranking_quantile_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def _outcomes_by_feature_date(
        self,
        label_result: MLLabelBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> dict[str, dict[str, float | None]]:
        benchmark_symbol = str(
            self.config.get("ml", {}).get("benchmark_symbols", ["SPY"])[0]
        )
        benchmark_closes = {
            candle.timestamp.date().isoformat(): candle.close
            for candle in candles_by_symbol.get(benchmark_symbol, [])
            if candle.close > 0
        }
        equity_by_date = {
            point.timestamp.date().isoformat(): point.equity
            for point in self._champion_equity_curve
            if point.equity > 0
        }
        equity_dates = sorted(equity_by_date)
        index_by_date = {value: index for index, value in enumerate(equity_dates)}
        outcomes = {}
        for row in label_result.rows:
            feature_date = str(row["feature_date"])
            label_end_date = str(row["label_end_date"])
            strategy_return = self._period_return(
                equity_by_date, feature_date, label_end_date
            )
            benchmark_return = self._period_return(
                benchmark_closes, feature_date, label_end_date
            )
            outcomes[feature_date] = {
                "strategy_return": strategy_return,
                "excess_spy_return": (
                    strategy_return - benchmark_return
                    if strategy_return is not None and benchmark_return is not None
                    else None
                ),
                "drawdown_event": self._future_drawdown_event(
                    equity_dates,
                    equity_by_date,
                    index_by_date,
                    feature_date,
                    label_end_date,
                ),
            }
        return outcomes

    def _period_return(
        self,
        values_by_date: dict[str, float],
        start_date: str,
        end_date: str,
    ) -> float | None:
        start = values_by_date.get(start_date)
        end = values_by_date.get(end_date)
        return (end / start) - 1.0 if start and end else None

    def _future_drawdown_event(
        self,
        dates: list[str],
        values_by_date: dict[str, float],
        index_by_date: dict[str, int],
        start_date: str,
        end_date: str,
    ) -> float | None:
        start_index = index_by_date.get(start_date)
        end_index = index_by_date.get(end_date)
        if start_index is None or end_index is None:
            return None
        peak = values_by_date[dates[start_index]]
        maximum_drawdown = 0.0
        for date in dates[start_index:end_index + 1]:
            value = values_by_date[date]
            peak = max(peak, value)
            maximum_drawdown = min(maximum_drawdown, (value / peak) - 1.0)
        return float(maximum_drawdown <= -0.10)

    def _mean_probability_summary(self, summaries: list[dict]) -> dict[str, float | None]:
        keys = (
            "brier_score",
            "brier_skill_vs_reference",
            "brier_skill_vs_static_base_rate",
            "roc_auc",
            "positive_prediction_rate",
        )
        return {
            key: self._mean_metric(summaries, key)
            for key in keys
        }

    def _rolling_base_rate_lookback_samples(self) -> int:
        return int(
            self.config.get("ml", {}).get(
                "rolling_base_rate_lookback_samples", 252
            )
        )

    def _ranking_quantile_count(self) -> int:
        return int(self.config.get("ml", {}).get("ranking_quantile_count", 5))

    def _calibration_bin_count(self) -> int:
        return int(self.config.get("ml", {}).get("calibration_bin_count", 10))

    def _write_threshold_sweep(
        self,
        path: Path,
        dataset: MLDataset,
        probabilities: list[float],
    ) -> None:
        thresholds = [round(value / 100, 2) for value in range(20, 85, 5)]
        path.write_text(json.dumps({
            "evaluation": "holdout_only",
            "thresholds": [
                {
                    "threshold": threshold,
                    "metrics": classification_metrics(
                        dataset.labels,
                        [int(value >= threshold) for value in probabilities],
                    ),
                }
                for threshold in thresholds
            ],
        }, indent=2), encoding="utf-8")

    def _write_model_comparison(self, path: Path, dataset: MLDataset) -> None:
        folds = rolling_walk_forward(dataset, self.experiment_config.walk_forward_folds)
        thresholds = [round(value / 100, 2) for value in range(20, 85, 5)]
        model_types = self.config.get("ml", {}).get(
            "comparison_models",
            ["logistic_regression", "random_forest", "gradient_boosting"],
        )
        models = []
        for model_type in model_types:
            threshold_metrics = {threshold: [] for threshold in thresholds}
            for fold in folds:
                model = build_ml_model(
                    model_type,
                    random_seed=self.experiment_config.random_seed,
                    class_weight=self._class_weight(),
                )
                model.fit(fold.split.train.features, fold.split.train.labels)
                probabilities = model.predict_proba(fold.split.test.features)
                for threshold in thresholds:
                    threshold_metrics[threshold].append(classification_metrics(
                        fold.split.test.labels,
                        [int(value >= threshold) for value in probabilities],
                    ))
            models.append({
                "model_type": model_type,
                "thresholds": [
                    {
                        "threshold": threshold,
                        "mean_balanced_accuracy": self._mean_metric(
                            threshold_metrics[threshold], "balanced_accuracy"
                        ),
                        "mean_precision": self._mean_metric(
                            threshold_metrics[threshold], "precision"
                        ),
                        "mean_recall": self._mean_metric(
                            threshold_metrics[threshold], "recall"
                        ),
                    }
                    for threshold in thresholds
                ],
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "fold_count": len(folds),
            "models": models,
            "research_only": True,
        }, indent=2), encoding="utf-8")

    def _write_shadow_overlay(self, path: Path, dataset: MLDataset) -> None:
        config = self.config.get("ml", {})
        model_type = str(config.get("shadow_model_type", "gradient_boosting"))
        thresholds = config.get("shadow_thresholds", [0.10, 0.15, 0.20, 0.25])
        reduced_exposures = config.get("shadow_reduced_exposures", [0.70, 0.80, 0.90])
        transaction_cost_bps = float(config.get("shadow_transaction_cost_bps", 5.0))
        equity_by_date = {
            point.timestamp.date().isoformat(): point.equity
            for point in self._champion_equity_curve
        }
        scenarios = [
            {"threshold": float(threshold), "reduced_exposure": float(exposure), "folds": []}
            for threshold in thresholds
            for exposure in reduced_exposures
        ]
        for fold in rolling_walk_forward(dataset, self.experiment_config.walk_forward_folds):
            model = build_ml_model(model_type, random_seed=self.experiment_config.random_seed)
            model.fit(fold.split.train.features, fold.split.train.labels)
            probabilities = model.predict_proba(fold.split.test.features)
            probabilities_by_date = dict(zip(fold.split.test.feature_dates, probabilities))
            for scenario in scenarios:
                result = simulate_shadow_overlay(
                    equity_by_date,
                    probabilities_by_date,
                    scenario["threshold"],
                    scenario["reduced_exposure"],
                    rebalance_dates=self._champion_rebalance_dates,
                    transaction_cost_bps=transaction_cost_bps,
                )
                if result is not None:
                    scenario["folds"].append({"fold": fold.fold_number, **result.__dict__})
        path.write_text(json.dumps({
            "mode": "shadow_research_only",
            "model_type": model_type,
            "rebalance_only": True,
            "transaction_cost_bps": transaction_cost_bps,
            "scenarios": scenarios,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def _write_holdout_shadow_overlay(
        self,
        path: Path,
        split: ChronologicalSplit,
    ) -> None:
        config = self.config.get("ml", {})
        model_type = str(config.get("shadow_model_type", "gradient_boosting"))
        threshold = float(config.get("shadow_holdout_threshold", 0.20))
        reduced_exposure = float(
            config.get("shadow_holdout_reduced_exposure", 0.70)
        )
        transaction_cost_bps = float(config.get("shadow_transaction_cost_bps", 5.0))
        model = build_ml_model(model_type, random_seed=self.experiment_config.random_seed)
        model.fit(split.train.features, split.train.labels)
        probabilities = model.predict_proba(split.test.features)
        equity_by_date = {
            point.timestamp.date().isoformat(): point.equity
            for point in self._champion_equity_curve
        }
        result = simulate_shadow_overlay(
            equity_by_date,
            dict(zip(split.test.feature_dates, probabilities)),
            threshold,
            reduced_exposure,
            rebalance_dates=self._champion_rebalance_dates,
            transaction_cost_bps=transaction_cost_bps,
        )
        path.write_text(json.dumps({
            "mode": "final_holdout_shadow_research_only",
            "model_type": model_type,
            "success_threshold": threshold,
            "reduced_exposure": reduced_exposure,
            "rebalance_only": True,
            "transaction_cost_bps": transaction_cost_bps,
            "test_start_date": split.test_start_date,
            "result": result.__dict__ if result is not None else None,
            "trading_impact": "none",
            "candidate_frozen_before_holdout": True,
        }, indent=2), encoding="utf-8")

    def _mean_metric(self, metrics: list[dict], key: str) -> float | None:
        values = [item[key] for item in metrics if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    def _standard_deviation(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        average = sum(values) / len(values)
        return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5

    def _correlation(self, left: list[float], right: list[float]) -> float:
        if len(left) < 2 or len(left) != len(right):
            return 0.0
        left_average = sum(left) / len(left)
        right_average = sum(right) / len(right)
        numerator = sum(
            (left_value - left_average) * (right_value - right_average)
            for left_value, right_value in zip(left, right)
        )
        left_scale = sum((value - left_average) ** 2 for value in left) ** 0.5
        right_scale = sum((value - right_average) ** 2 for value in right) ** 0.5
        if left_scale == 0 or right_scale == 0:
            return 0.0
        return numerator / (left_scale * right_scale)

    def _write_metrics(
        self,
        path: Path,
        split: ChronologicalSplit,
        predictions: list[int],
    ) -> None:
        metrics = classification_metrics(split.test.labels, predictions)
        payload = {
            "mode": "research",
            "model_type": self.experiment_config.model_type,
            "feature_set": self.experiment_config.feature_set,
            "label_type": self.experiment_config.label_type,
            "decision_threshold": self.experiment_config.decision_threshold,
            "class_weight": self._class_weight(),
            "train_sample_count": split.train.sample_count,
            "test_sample_count": split.test.sample_count,
            "feature_count": split.train.feature_count,
            "test_start_date": split.test_start_date,
            "purged_train_samples": split.purged_train_samples,
            "metrics": metrics,
            "baselines": self._baseline_metrics(split),
            "note": (
                "Research-only out-of-sample evaluation; ML does not affect "
                "trading decisions."
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _baseline_metrics(self, split: ChronologicalSplit) -> dict[str, dict]:
        no_op_predictions = [0] * split.test.sample_count
        majority_class = int(
            sum(split.train.labels) >= (split.train.sample_count / 2)
        ) if split.train.sample_count else 0
        majority_predictions = [majority_class] * split.test.sample_count
        return {
            "noop": classification_metrics(split.test.labels, no_op_predictions),
            "majority_class": {
                "predicted_class": majority_class,
                "metrics": classification_metrics(
                    split.test.labels,
                    majority_predictions,
                ),
            },
        }

    def _write_predictions(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
        probabilities: list[float],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "row",
                    "feature_date",
                    "label_start_date",
                    "label_end_date",
                    "prediction",
                    "probability",
                    "label",
                ],
            )
            writer.writeheader()
            for index, prediction in enumerate(predictions):
                writer.writerow({
                    "row": index,
                    "feature_date": dataset.feature_dates[index],
                    "label_start_date": dataset.label_start_dates[index],
                    "label_end_date": dataset.label_end_dates[index],
                    "prediction": prediction,
                    "probability": probabilities[index],
                    "label": dataset.labels[index],
                })

    def _write_feature_importance(
        self,
        path: Path,
        feature_importances: dict[str, float],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
            writer.writeheader()
            for feature, importance in sorted(
                feature_importances.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                writer.writerow({"feature": feature, "importance": importance})

    def _write_confusion_matrix(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
    ) -> None:
        counts = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for actual, prediction in zip(dataset.labels, predictions):
            if actual == prediction == 1:
                counts["true_positive"] += 1
            elif actual == prediction == 0:
                counts["true_negative"] += 1
            elif actual == 0 and prediction == 1:
                counts["false_positive"] += 1
            elif actual == 1 and prediction == 0:
                counts["false_negative"] += 1

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bucket", "count"])
            writer.writeheader()
            for bucket, count in counts.items():
                writer.writerow({"bucket": bucket, "count": count})

    def _write_metadata(
        self,
        path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
    ) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self._hash_payload(self.config),
            "data_hash": self._hash_payload({
                "features": dataset.features,
                "labels": dataset.labels,
            }),
            "git_commit": self._git_commit(),
            "model_type": self.experiment_config.model_type,
            "feature_set": self.experiment_config.feature_set,
            "label_type": self.experiment_config.label_type,
            "random_seed": self.experiment_config.random_seed,
            "experiment_config": self.experiment_config.to_dict(),
            "validation": {
                "method": "purged_chronological_holdout",
                "train_sample_count": split.train.sample_count,
                "test_sample_count": split.test.sample_count,
                "test_start_date": split.test_start_date,
                "purged_train_samples": split.purged_train_samples,
            },
            "research_only": True,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _hash_payload(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _git_commit(self) -> str | None:
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
