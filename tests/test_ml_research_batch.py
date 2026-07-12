from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import inspect
import threading
import time

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
from application.services.ml_commands_batch import (
    _completed_ml_research_output_check,
    _current_model_input_identity,
)
from core.research.ml.artifacts.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MODEL_INPUT_CONTRACT_VERSION
from core.research.ml.immutable_runs import deterministic_run_id, preserve_immutable_run


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


def test_ml_research_batch_max_workers_one_uses_serial_order(tmp_path, monkeypatch):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)
    invoked = []
    captured_leaderboard_dirs = []
    monkeypatch.setattr(
        "application.services.ml_commands_batch._update_source_leaderboard",
        lambda config, first_dir, rest=None: (
            captured_leaderboard_dirs.append([first_dir, *(rest or [])])
            or (tmp_path / "leaderboard.md", tmp_path / "leaderboard.json")
        ),
    )

    class ForbiddenExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers):
            raise AssertionError("serial max_workers=1 path should not create executor")

    def fake_worker(config_path, model_threads, expanded_dataset_path, profile_name=""):
        invoked.append(Path(config_path).stem)
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(Path(config_path).with_suffix("")),
            success=True,
        )

    results = run_ml_research_batch(
        _batch_config(shared_cache, [first, second], max_workers=1),
        executor_cls=ForbiddenExecutor,
        worker_fn=fake_worker,
    )

    assert invoked == ["first", "second"]
    assert [Path(result.config_path).stem for result in results] == ["first", "second"]
    assert [path.name for path in captured_leaderboard_dirs[0]] == ["first", "second"]


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

    captured_leaderboard_dirs = []
    from application.services import ml_commands_batch
    original_update = ml_commands_batch._update_source_leaderboard
    ml_commands_batch._update_source_leaderboard = lambda config, first, rest=None: (
        captured_leaderboard_dirs.append([first, *(rest or [])])
        or (tmp_path / "leaderboard.md", tmp_path / "leaderboard.json")
    )
    try:
        results = run_ml_research_batch(
        _batch_config(shared_cache, [first, second], max_workers=2, model_threads=3),
        executor_cls=ThreadPoolExecutor,
        worker_fn=fake_worker,
        )
    finally:
        ml_commands_batch._update_source_leaderboard = original_update

    assert len(results) == 2
    assert len(submitted) == 2
    assert {item[1] for item in submitted} == {3}
    assert all(item[2].endswith("expanded_rebalance_dataset.csv") for item in submitted)
    assert len(captured_leaderboard_dirs) == 1
    assert len(captured_leaderboard_dirs[0]) == 2


def test_ml_research_batch_uses_bounded_executor_when_max_workers_exceeds_one(
    tmp_path,
    monkeypatch,
):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n", encoding="utf-8"
    )
    config_path = _research_config(tmp_path, "first", "reports/first", shared_cache)
    captured = {}
    monkeypatch.setattr(
        "application.services.ml_commands_batch._update_source_leaderboard",
        lambda config, output_dir, additional=None: (
            tmp_path / "leaderboard.md", tmp_path / "leaderboard.json"
        ),
    )

    class CapturingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers)

    def fake_worker(config_path, model_threads, expanded_dataset_path, profile_name=""):
        return MLResearchBatchResult(config_path, "reports/first", True)

    run_ml_research_batch(
        _batch_config(shared_cache, [config_path], max_workers=4),
        executor_cls=CapturingExecutor,
        worker_fn=fake_worker,
    )

    assert captured["max_workers"] == 4


def test_ml_research_batch_never_exceeds_configured_concurrency(
    tmp_path,
    monkeypatch,
):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n", encoding="utf-8"
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)
    third = _research_config(tmp_path, "third", "reports/third", shared_cache)
    lock = threading.Lock()
    active = 0
    peak_active = 0
    invoked = []
    monkeypatch.setattr(
        "application.services.ml_commands_batch._update_source_leaderboard",
        lambda config, output_dir, additional=None: (
            tmp_path / "leaderboard.md", tmp_path / "leaderboard.json"
        ),
    )

    def fake_worker(config_path, model_threads, expanded_dataset_path, profile_name=""):
        nonlocal active, peak_active
        with lock:
            invoked.append(Path(config_path).stem)
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return MLResearchBatchResult(
            config_path,
            str(Path(config_path).with_suffix("")),
            True,
        )

    run_ml_research_batch(
        _batch_config(shared_cache, [first, second, third], max_workers=2),
        executor_cls=ThreadPoolExecutor,
        worker_fn=fake_worker,
    )

    assert peak_active <= 2
    assert sorted(invoked) == ["first", "second", "third"]


def test_ml_research_batch_publishes_outputs_in_config_order(tmp_path, monkeypatch):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n", encoding="utf-8"
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)
    captured_leaderboard_dirs = []
    monkeypatch.setattr(
        "application.services.ml_commands_batch._update_source_leaderboard",
        lambda config, first_dir, rest=None: (
            captured_leaderboard_dirs.append([first_dir, *(rest or [])])
            or (tmp_path / "leaderboard.md", tmp_path / "leaderboard.json")
        ),
    )

    def fake_worker(config_path, model_threads, expanded_dataset_path, profile_name=""):
        if Path(config_path).stem == "first":
            time.sleep(0.05)
        return MLResearchBatchResult(
            config_path,
            str(Path(config_path).with_suffix("")),
            True,
        )

    results = run_ml_research_batch(
        _batch_config(shared_cache, [first, second], max_workers=2),
        executor_cls=ThreadPoolExecutor,
        worker_fn=fake_worker,
    )

    assert [Path(result.config_path).stem for result in results] == ["first", "second"]
    assert [path.name for path in captured_leaderboard_dirs[0]] == ["first", "second"]


def test_ml_research_batch_surfaces_worker_exception(tmp_path, monkeypatch):
    shared_cache = tmp_path / "cache" / "ml"
    shared_cache.mkdir(parents=True)
    (shared_cache / "expanded_rebalance_dataset.csv").write_text(
        "feature_id,feature_date,should_reduce_exposure\n",
        encoding="utf-8",
    )
    first = _research_config(tmp_path, "first", "reports/first", shared_cache)
    second = _research_config(tmp_path, "second", "reports/second", shared_cache)
    monkeypatch.setattr(
        "application.services.ml_commands_batch._update_source_leaderboard",
        lambda config, output_dir, additional=None: (
            tmp_path / "leaderboard.md", tmp_path / "leaderboard.json"
        ),
    )

    def fake_worker(config_path, model_threads, expanded_dataset_path, profile_name=""):
        if Path(config_path).stem == "second":
            raise RuntimeError("boom")
        return MLResearchBatchResult(
            config_path,
            str(Path(config_path).with_suffix("")),
            True,
        )

    with pytest.raises(RuntimeError, match="second.yaml: boom"):
        run_ml_research_batch(
            _batch_config(shared_cache, [first, second], max_workers=2, fail_fast=False),
            executor_cls=ThreadPoolExecutor,
            worker_fn=fake_worker,
        )


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

    from importlib import import_module

    research_module = import_module(run_ml_research.__module__)

    monkeypatch.setattr(
        research_module,
        "MLExperimentRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        research_module,
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


def test_completed_ml_research_output_skips_compatible_run(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is True
    assert check.reason == ""


def test_completed_ml_research_output_skips_compatible_immutable_run(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    identity = _current_model_input_identity(
        config,
        MLExperimentConfig.from_config(config),
    )
    assert identity is not None
    run_identity = {
        **identity.values,
        "model_name": "noop",
        "feature_set": "expanded_rebalance_v1",
    }
    run_id = deterministic_run_id("exposure_ml", run_identity)
    preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="exposure_ml",
        identity=run_identity,
        artifact_paths=(
            output_dir / "metrics.json",
            output_dir / "metadata.json",
            output_dir / "predictions.csv",
            output_dir / "model.json",
            output_dir / "prediction_artifacts.csv",
            output_dir / "prediction_artifacts.json",
            output_dir / "dataset_audit.json",
        ),
    )
    for name in (
        "metrics.json",
        "metadata.json",
        "predictions.csv",
        "model.json",
        "prediction_artifacts.csv",
        "prediction_artifacts.json",
        "dataset_audit.json",
    ):
        (output_dir / name).unlink()

    check = _completed_ml_research_output_check(config)

    assert check.reusable is True
    assert check.reason == ""


def test_completed_ml_research_output_rejects_changed_dataset_hash(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["dataset_hash"] = "different-dataset"
    metadata["data_hash"] = "different-dataset"
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "metrics.json_dataset_hash_mismatch"


def test_completed_ml_research_output_rejects_prediction_dataset_hash_mismatch(
    tmp_path,
):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    updated_dataset_hash = "different-dataset"
    metadata = _read_json(output_dir / "metadata.json")
    metadata["dataset_hash"] = updated_dataset_hash
    metadata["data_hash"] = updated_dataset_hash
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    metrics = _read_json(output_dir / "metrics.json")
    metrics["dataset_hash"] = updated_dataset_hash
    (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    dataset_audit = _read_json(output_dir / "dataset_audit.json")
    dataset_audit["dataset_hash"] = updated_dataset_hash
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(dataset_audit),
        encoding="utf-8",
    )

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "prediction_dataset_hash_mismatch"


def test_completed_ml_research_output_rejects_changed_feature_columns(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["feature_columns"] = ["alpha", "gamma"]
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "prediction_feature_columns_mismatch"


def test_completed_ml_research_output_rejects_changed_feature_order(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["feature_columns"] = ["beta", "alpha"]
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "prediction_feature_columns_mismatch"


def test_completed_ml_research_output_rejects_changed_target(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["target_label_name"] = "other_target"
    metadata["label_type"] = "other_target"
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "label_type_mismatch"


def test_completed_ml_research_output_rejects_missing_or_malformed_metadata(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    (output_dir / "metadata.json").write_text("{not-json", encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "malformed_metadata"


def test_completed_ml_research_output_rejects_partial_artifact_set(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    (output_dir / "model.json").unlink()

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "missing_or_empty_required_artifacts:model.json"


def test_completed_ml_research_output_rejects_not_complete_status(tmp_path):
    config = _completed_run_config(tmp_path)
    output_dir = _write_completed_batch_output(config)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["run_status"] = "partial"
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "run_status_not_complete"


def test_completed_ml_research_output_rejects_added_current_input_row(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    _write_current_input_rows(config, [*_current_input_rows(), _current_input_row("c")])

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_sample_count_mismatch"


def test_completed_ml_research_output_rejects_removed_current_input_row(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    _write_current_input_rows(config, [_current_input_row("a")])

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_sample_count_mismatch"


def test_completed_ml_research_output_rejects_current_predictor_value_change(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    rows = _current_input_rows()
    rows[0]["alpha"] = "99.0"
    _write_current_input_rows(config, rows)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_model_input_hash_mismatch"


def test_completed_ml_research_output_rejects_current_target_value_change(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    rows = _current_input_rows()
    rows[0]["should_reduce_exposure"] = "0"
    _write_current_input_rows(config, rows)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_dataset_hash_mismatch"


def test_completed_ml_research_output_rejects_current_feature_membership_change(
    tmp_path,
):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    rows = _current_input_rows()
    for row in rows:
        row["gamma"] = "1.0"
    _write_current_input_rows(config, rows, fieldnames=[*_current_input_fieldnames(), "gamma"])

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_feature_columns_mismatch"


def test_completed_ml_research_output_rejects_current_feature_order_change(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    fieldnames = _current_input_fieldnames()
    alpha_index = fieldnames.index("alpha")
    beta_index = fieldnames.index("beta")
    fieldnames[alpha_index], fieldnames[beta_index] = (
        fieldnames[beta_index],
        fieldnames[alpha_index],
    )
    _write_current_input_rows(config, _current_input_rows(), fieldnames=fieldnames)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_feature_order_mismatch"


def test_completed_ml_research_output_rejects_current_date_coverage_change(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    rows = _current_input_rows()
    rows[1]["feature_date"] = "2024-01-03"
    rows[1]["rebalance_date"] = "2024-01-03"
    rows[1]["label_start_date"] = "2024-01-04"
    rows[1]["label_end_date"] = "2024-01-08"
    _write_current_input_rows(config, rows)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_date_coverage_mismatch"


def test_completed_ml_research_output_rejects_changed_current_input_file(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    new_input_path = tmp_path / "cache" / "ml" / "other_rebalance_dataset.csv"
    config["ml"]["expanded_rebalance_dataset_path"] = str(new_input_path)
    _write_current_input_rows(config, _current_input_rows())
    output_dir = Path(config["ml"]["output_dir"])
    metadata = _read_json(output_dir / "metadata.json")
    metadata["config_hash"] = MLCoreArtifactWriter.hash_payload(config)
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_input_source_mismatch"


def test_completed_ml_research_output_rejects_internally_consistent_stale_artifacts(
    tmp_path,
):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    rows = _current_input_rows()
    rows[0]["alpha"] = "42.0"
    _write_current_input_rows(config, rows)

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_model_input_hash_mismatch"


def test_completed_ml_research_output_rejects_unavailable_current_identity(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_completed_batch_output(config)
    Path(config["ml"]["expanded_rebalance_dataset_path"]).unlink()

    check = _completed_ml_research_output_check(config)

    assert check.reusable is False
    assert check.reason == "current_input_identity_unavailable"


def test_current_model_input_identity_is_deterministic(tmp_path):
    config = _completed_run_config(tmp_path)
    _write_current_input_rows(config, _current_input_rows())

    first = _current_model_input_identity(
        config,
        MLExperimentConfig.from_config(config),
    )
    second = _current_model_input_identity(
        config,
        MLExperimentConfig.from_config(config),
    )

    assert first is not None
    assert second is not None
    assert first.values == second.values


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


def _completed_run_config(tmp_path: Path) -> dict:
    return {
        "ml": {
            "model_type": "noop",
            "feature_set": "expanded_rebalance_v1",
            "label_type": "should_reduce_exposure",
            "output_dir": str(tmp_path / "reports" / "noop"),
            "read_existing_expanded_rebalance_dataset": True,
            "expanded_rebalance_dataset_path": str(
                tmp_path / "cache" / "ml" / "expanded_rebalance_dataset.csv"
            ),
        }
    }


def _write_completed_batch_output(
    config: dict,
    *,
    feature_columns: list[str] | None = None,
) -> Path:
    _write_current_input_rows(config, _current_input_rows())
    identity = _current_model_input_identity(
        config,
        MLExperimentConfig.from_config(config),
    )
    assert identity is not None
    values = identity.values
    dataset_hash = values["dataset_hash"]
    model_input_hash = values["model_input_hash"]
    feature_columns = feature_columns or values["feature_columns"]
    output_dir = Path(config["ml"]["output_dir"])
    output_dir.mkdir(parents=True)
    metadata = {
        "config_hash": MLCoreArtifactWriter.hash_payload(config),
        "data_hash": dataset_hash,
        "dataset_hash": dataset_hash,
        "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "model_input_hash": model_input_hash,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "sample_count": 2,
        "source_dataset_row_count": 2,
        "feature_date_min": "2024-01-01",
        "feature_date_max": "2024-01-02",
        "training_date_min": "2024-01-01",
        "training_date_max": "2024-01-02",
        "model_input_source_path": values["model_input_source_path"],
        "model_name": "noop",
        "model_type": "noop",
        "feature_set": "expanded_rebalance_v1",
        "label_type": "should_reduce_exposure",
        "target_label_name": "should_reduce_exposure",
        "run_status": "complete",
        "research_only": True,
    }
    prediction_metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "data_hash": dataset_hash,
        "dataset_hash": dataset_hash,
        "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "model_input_hash": model_input_hash,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "sample_count": 2,
        "model_type": "noop",
        "label_type": "should_reduce_exposure",
        "target_label_name": "should_reduce_exposure",
        "model_input_source_path": values["model_input_source_path"],
    }
    row = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "profile": "",
        "model_name": "noop",
        "model_type": "noop",
        "config_path": "",
        "dataset_hash": dataset_hash,
        "source_dataset_row_count": "2",
        "train_sample_count": "1",
        "prediction_date": "2024-01-02",
        "symbol": "",
        "rebalance_date": "2024-01-02",
        "actual_label": "0",
        "predicted_probability": "0.5",
        "feature_id": "feature-a",
        "split": "holdout",
    }
    for name, payload in {
        "metadata.json": metadata,
        "metrics.json": {
            "dataset_hash": dataset_hash,
            "model_type": "noop",
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
        },
        "dataset_audit.json": {"sample_count": 2, "feature_count": 2},
        "prediction_artifacts.json": prediction_metadata,
    }.items():
        (output_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (output_dir / "prediction_artifacts.csv").write_text(
        ",".join(row) + "\n" + ",".join(row.values()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.csv").write_text("prediction\n0\n", encoding="utf-8")
    (output_dir / "model.json").write_text("{}", encoding="utf-8")
    return output_dir


def _current_input_fieldnames() -> list[str]:
    return [
        "feature_id",
        "feature_date",
        "rebalance_date",
        "label_start_date",
        "label_end_date",
        "symbol",
        "variant_id",
        "alpha",
        "beta",
        "should_reduce_exposure",
        "future_drawdown",
        "future_max_drawdown",
        "forward_return_5d",
        "forward_return_10d",
        "future_volatility",
        "max_adverse_excursion",
        "max_favourable_excursion",
        "champion_excess_return",
        "volatility_adjusted_excess_return",
    ]


def _current_input_rows() -> list[dict[str, str]]:
    return [_current_input_row("a"), _current_input_row("b")]


def _current_input_row(suffix: str) -> dict[str, str]:
    day = {"a": "01", "b": "02", "c": "03"}.get(suffix, "04")
    label_start_day = f"{int(day) + 1:02d}"
    label_end_day = f"{int(day) + 5:02d}"
    return {
        "feature_id": f"feature-{suffix}",
        "feature_date": f"2024-01-{day}",
        "rebalance_date": f"2024-01-{day}",
        "label_start_date": f"2024-01-{label_start_day}",
        "label_end_date": f"2024-01-{label_end_day}",
        "symbol": "SPY",
        "variant_id": "variant-a",
        "alpha": "1.0" if suffix != "b" else "2.0",
        "beta": "3.0" if suffix != "b" else "4.0",
        "should_reduce_exposure": "1" if suffix == "a" else "0",
        "future_drawdown": "-0.10" if suffix == "a" else "-0.02",
        "future_max_drawdown": "-0.10" if suffix == "a" else "-0.02",
        "forward_return_5d": "0.01",
        "forward_return_10d": "0.02",
        "future_volatility": "0.12",
        "max_adverse_excursion": "-0.05",
        "max_favourable_excursion": "0.04",
        "champion_excess_return": "-0.01" if suffix == "a" else "0.03",
        "volatility_adjusted_excess_return": "-0.10" if suffix == "a" else "0.20",
    }


def _write_current_input_rows(
    config: dict,
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(config["ml"]["expanded_rebalance_dataset_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or _current_input_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
