from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import inspect

import pytest

from application.services import ml_commands
from application.services.ml_commands import (
    MLResearchBatchResult,
    run_ml_research_batch,
    validate_ml_research_batch_config,
)


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
