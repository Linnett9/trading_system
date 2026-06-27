from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import inspect

import pytest

from application.services import ml_commands
from application.services.ml_commands import (
    MLResearchBatchResult,
    _artifact_source_dirs,
    incomplete_ml_run_dirs,
    _update_source_leaderboard,
    run_ml_validate_artifacts,
    run_ml_research,
    run_ml_research_batch,
    validate_ml_research_batch_config,
)
from core.research.ml.artifact_schema import ARTIFACT_SCHEMA_VERSION


def test_ml_research_batch_config_validation(tmp_path):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)

    items = validate_ml_research_batch_config(
        _batch_config(shared_cache, [first, second])
    )

    assert [item.config_path for item in items] == [first, second]
    assert items[0].output_dir.name == "first"
    assert items[1].output_dir.name == "second"


def test_ml_research_batch_rejects_duplicate_output_dirs(tmp_path):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    first = _research_config(tmp_path, "first", "reports/same", shared_cache)
    second = _research_config(tmp_path, "second", "reports/same", shared_cache)

    with pytest.raises(RuntimeError, match="Duplicate ml.output_dir"):
        validate_ml_research_batch_config(_batch_config(shared_cache, [first, second]))


def test_ml_research_batch_runs_two_dummy_configs_in_parallel(tmp_path):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)
    submitted = []

    def fake_worker(
        config_path: str,
        model_threads: int,
        expanded_dataset_path: str,
        profile_name: str = "",
    ):
        submitted.append((config_path, model_threads, expanded_dataset_path, profile_name))
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(Path(config_path).with_suffix("")),
            success=True,
            metrics_path="metrics.json",
            prediction_artifacts_path="prediction_artifacts.csv",
        )

    results = run_ml_research_batch(
        _batch_config(shared_cache, [first, second], max_workers=2, model_threads=3),
        executor_cls=ThreadPoolExecutor,
        worker_fn=fake_worker,
    )

    assert len(results) == 2
    assert len(submitted) == 2
    assert {item[1] for item in submitted} == {3}
    assert all(item[2].endswith("expanded_rebalance_dataset.csv") for item in submitted)


def test_ml_research_applies_runtime_parallelism_settings(monkeypatch, tmp_path):
    captured = {}

    class FakeRunner:
        def __init__(self, config, feed=None):
            captured["config"] = config
            captured["feed"] = feed

        def run(self):
            output_dir = tmp_path / "reports" / "model"
            output_dir.mkdir(parents=True)
            return _fake_ml_result(output_dir)

    monkeypatch.setattr(ml_commands, "MLExperimentRunner", FakeRunner)
    monkeypatch.setattr(
        ml_commands,
        "_update_source_leaderboard",
        lambda config, output_dir: (
            tmp_path / "leaderboard.md",
            tmp_path / "leaderboard.json",
        ),
    )

    run_ml_research({
        "ml": {
            "research_label": "TEST",
            "model_threads": 2,
            "torch_num_threads": 2,
            "sklearn_n_jobs": 2,
            "feature_workers": 1,
        }
    })

    assert captured["config"]["ml"]["model_threads"] == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_ml_research_batch_reports_failures_clearly(tmp_path):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    config_path = _research_config(tmp_path, "first", "reports/first", shared_cache)

    def failing_worker(
        config_path: str,
        model_threads: int,
        expanded_dataset_path: str,
        profile_name: str = "",
    ):
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir="reports/first",
            success=False,
            error="boom",
        )

    with pytest.raises(RuntimeError, match="boom"):
        run_ml_research_batch(
            _batch_config(shared_cache, [config_path], fail_fast=True),
            executor_cls=ThreadPoolExecutor,
            worker_fn=failing_worker,
        )


def test_ml_research_batch_service_does_not_import_operational_modules():
    source = inspect.getsource(ml_commands)

    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "broker" not in source
    assert "execution" not in source


def test_artifact_source_dirs_discovers_report_child_run_dirs(tmp_path):
    report_dir = tmp_path / "reports" / "ml"
    first = report_dir / "dlinear"
    second = report_dir / "patchtst"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    source_dirs = _artifact_source_dirs(
        {
            "reports": {"ml_dir": str(report_dir)},
            "ml": {"output_dir": str(report_dir)},
        },
        require_exists=False,
    )

    assert source_dirs == [first, second]


def test_artifact_source_dirs_skips_meta_ensemble_output(tmp_path):
    report_dir = tmp_path / "reports" / "ml"
    source = report_dir / "dlinear"
    meta_output = report_dir / "regime_transformer_meta_ensemble_v1"
    source.mkdir(parents=True)
    meta_output.mkdir(parents=True)

    source_dirs = _artifact_source_dirs(
        {"reports": {"ml_dir": str(report_dir)}},
        require_exists=False,
    )

    assert source_dirs == [source]


def test_validate_artifacts_reports_meta_ensemble_not_run_yet(tmp_path, capsys):
    report_dir = tmp_path / "reports" / "ml"
    source_dir = report_dir / "dlinear"
    source_dir.mkdir(parents=True)
    _write_valid_source_artifacts(source_dir, "dlinear")

    run_ml_validate_artifacts({"reports": {"ml_dir": str(report_dir)}})

    output = capsys.readouterr().out
    assert f"ok: {source_dir}" in output
    assert (
        f"not run yet: {report_dir / 'regime_transformer_meta_ensemble_v1'}"
        in output
    )


def test_research_batch_includes_traditional_baseline_configs():
    batch_path = Path("configs/research/ml_research_batch.yaml")
    text = batch_path.read_text(encoding="utf-8")

    assert "configs/research/logistic_regression_should_reduce_exposure.yaml" in text
    assert "configs/research/random_forest_should_reduce_exposure.yaml" in text
    assert "configs/research/gradient_boosting_should_reduce_exposure.yaml" in text


def test_update_source_leaderboard_uses_profile_report_dir(tmp_path):
    report_dir = tmp_path / "reports" / "ml" / "development"
    source_dir = report_dir / "patchtst_should_reduce_exposure"
    source_dir.mkdir(parents=True)
    _write_valid_source_artifacts(source_dir, "patchtst")

    markdown_path, json_path = _update_source_leaderboard(
        {"reports": {"ml_dir": str(report_dir)}},
        source_dir,
    )

    assert markdown_path == report_dir / "regime_transformer_meta_ensemble_v1" / "leaderboard.md"
    assert json_path.exists()
    assert "patchtst" in markdown_path.read_text(encoding="utf-8")


def test_incomplete_run_detection_skips_complete_and_empty_dirs(tmp_path):
    report_dir = tmp_path / "reports" / "ml" / "development"
    complete = report_dir / "complete_model"
    partial = report_dir / "partial_model"
    history_only = report_dir / "history_only_model"
    metrics_only = report_dir / "metrics_only_model"
    empty = report_dir / "empty_model"
    complete.mkdir(parents=True)
    partial.mkdir()
    history_only.mkdir()
    metrics_only.mkdir()
    empty.mkdir()
    _write_valid_source_artifacts(complete, "complete_model")
    (partial / "metrics.json").write_text("{}", encoding="utf-8")
    (partial / "prediction_artifacts.csv").write_text("feature_id\n", encoding="utf-8")
    (history_only / "history_coverage.json").write_text("{}", encoding="utf-8")
    (metrics_only / "metrics.json").write_text("{}", encoding="utf-8")

    incomplete = incomplete_ml_run_dirs(report_dir)

    assert incomplete == [history_only, metrics_only, partial]


def _batch_config(
    shared_cache: Path,
    config_paths: list[Path],
    max_workers: int = 2,
    model_threads: int = 1,
    fail_fast: bool = False,
) -> dict:
    return {
        "cache": {"ml_dir": str(shared_cache)},
        "ml_research_batch": {
            "config_paths": [str(path) for path in config_paths],
            "max_workers": max_workers,
            "model_threads": model_threads,
            "fail_fast": fail_fast,
        },
    }


def _fake_ml_result(output_dir: Path):
    class Result:
        pass

    result = Result()
    result.output_dir = output_dir
    result.metrics_path = output_dir / "metrics.json"
    result.predictions_path = output_dir / "predictions.csv"
    result.feature_importance_path = output_dir / "feature_importance.csv"
    result.confusion_matrix_path = output_dir / "confusion_matrix.csv"
    result.metadata_path = output_dir / "metadata.json"
    result.model_path = output_dir / "model.joblib"
    result.features_path = output_dir / "features.csv"
    result.feature_summary_path = output_dir / "feature_summary.json"
    result.labels_path = output_dir / "labels.csv"
    result.dataset_path = output_dir / "dataset.csv"
    result.dataset_audit_path = output_dir / "dataset_audit.json"
    result.walk_forward_metrics_path = output_dir / "walk_forward_metrics.json"
    result.threshold_sweep_path = output_dir / "threshold_sweep.json"
    result.model_comparison_path = output_dir / "model_comparison.json"
    result.shadow_overlay_path = output_dir / "shadow_overlay.json"
    result.holdout_shadow_overlay_path = output_dir / "holdout_shadow_overlay.json"
    result.rebalance_dataset_path = output_dir / "expanded_rebalance_dataset.csv"
    result.rebalance_dataset_audit_path = output_dir / "rebalance_dataset_audit.json"
    result.history_coverage_path = output_dir / "history_coverage.json"
    result.drawdown_event_review_path = output_dir / "drawdown_event_review.json"
    result.rule_exposure_study_path = output_dir / "rule_exposure_study.json"
    result.probability_calibration_path = output_dir / "probability_calibration.json"
    result.walk_forward_probability_calibration_path = (
        output_dir / "walk_forward_probability_calibration.json"
    )
    result.baseline_model_comparison_path = output_dir / "baseline_model_comparison.json"
    result.ranking_diagnostics_path = output_dir / "ranking_diagnostics.json"
    return result


def _research_config(
    tmp_path: Path,
    name: str,
    output_dir: str,
    shared_cache: Path,
) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        "\n".join([
            "backtest:",
            "  provider: stooq_parquet",
            "  data_dir: data/processed/stooq_parquet",
            "cache:",
            f"  ml_dir: {shared_cache}",
            "ml:",
            "  mode: research",
            "  model_type: noop",
            "  label_type: champion_success",
            f"  output_dir: {tmp_path / output_dir}",
        ]),
        encoding="utf-8",
    )
    return path


def _write_valid_source_artifacts(path: Path, model_type: str) -> None:
    dataset_hash = "dataset-hash"
    row = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "profile": "development",
        "model_name": model_type,
        "model_type": model_type,
        "config_path": f"configs/research/{model_type}.yaml",
        "dataset_hash": dataset_hash,
        "source_dataset_row_count": "1",
        "train_sample_count": "1",
        "prediction_date": "2024-01-01",
        "symbol": "",
        "rebalance_date": "2024-01-01",
        "actual_label": "0",
        "predicted_probability": "0.5",
        "feature_id": "feature-a",
        "split": "holdout",
    }
    (path / "prediction_artifacts.csv").write_text(
        ",".join(row) + "\n" + ",".join(row.values()) + "\n",
        encoding="utf-8",
    )
    (path / "prediction_artifacts.json").write_text(
        '{"artifact_schema_version":"ml_prediction_artifact_v1","dataset_hash":"dataset-hash"}',
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        '{"model_type":"' + model_type + '","metrics":{"balanced_accuracy":0.6}}',
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        '{"model_type":"' + model_type + '"}',
        encoding="utf-8",
    )
    (path / "dataset_audit.json").write_text(
        '{"dataset_hash":"dataset-hash"}',
        encoding="utf-8",
    )
    (path / "probability_calibration.json").write_text("{}", encoding="utf-8")
    (path / "calibrated_probability_calibration.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (path / "holdout_shadow_overlay.json").write_text("{}", encoding="utf-8")
