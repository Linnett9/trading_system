from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from config.config_loader import load_config
from application.services.research_profiles import apply_research_profile
from application.services.ml_commands_types import MLResearchBatchItem, MLResearchBatchResult
from application.services.ml_commands_artifacts import _update_source_leaderboard
from core.research.ml.artifacts import MLExperimentPathBuilder
from core.research.ml.artifacts.artifact_validator import validate_prediction_artifacts
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MODEL_INPUT_CONTRACT_VERSION
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.features.features import MLFeatureBuildResult
from core.research.ml.features.labels import ShouldReduceExposureLabelBuilder
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    immutable_run_dir,
    read_run_manifest,
    run_dir_from_latest_completed,
)
from core.research.ml.pipelines import MLDatasetPipeline
from core.research.ml.runtime_parallelism import (
    apply_runtime_parallelism,
    apply_worker_thread_environment,
    format_runtime_settings,
)


@dataclass(frozen=True)
class CompletedMLResearchOutputCheck:
    reusable: bool
    reason: str = ""


@dataclass(frozen=True)
class CurrentModelInputIdentity:
    values: dict[str, Any]


def run_ml_research_batch(
    config: dict[str, Any],
    *,
    executor_cls: type[ProcessPoolExecutor] = ProcessPoolExecutor,
    worker_fn: Callable[[str, int, str, str, dict[str, Any] | None], MLResearchBatchResult] | None = None,
) -> list[MLResearchBatchResult]:
    batch_config = config.get("ml_research_batch", {})
    runtime_settings = apply_runtime_parallelism(config)
    items = validate_ml_research_batch_config(config)
    max_workers = int(batch_config.get("max_workers", runtime_settings.num_workers))
    model_threads = int(batch_config.get("model_threads", runtime_settings.model_threads))
    fail_fast = bool(batch_config.get("fail_fast", True))
    shared_dataset_path = str(_expanded_rebalance_dataset_path(config))
    profile_name = str(config.get("research_profile", {}).get("name", "") or "")
    parent_overrides = _batch_parent_overrides(config)
    resume_completed = bool(batch_config.get("resume_completed", True))
    worker = worker_fn or _run_ml_research_batch_worker
    include_parent_overrides = _worker_accepts_parent_overrides(worker)

    print("\nML RESEARCH BATCH")
    print("mode=research | trading_impact=none")
    print(f"Configs: {len(items)}")
    print(f"Workers: {max_workers}")
    if max_workers > 1:
        print(f"Executor max_workers: {max_workers}")
    print(f"Model threads: {model_threads}")
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Shared expanded dataset: {shared_dataset_path}")

    if max_workers <= 1:
        results = []
        for item in items:
            result = _run_batch_item(
                item,
                worker=worker,
                model_threads=model_threads,
                shared_dataset_path=shared_dataset_path,
                profile_name=profile_name,
                parent_overrides=parent_overrides,
                resume_completed=resume_completed,
                include_parent_overrides=include_parent_overrides,
            )
            _print_batch_result(item, result)
            results.append(result)
            if fail_fast and not result.success:
                break
    else:
        indexed_results: list[tuple[int, MLResearchBatchResult]] = []
        with executor_cls(max_workers=max_workers) as executor:
            future_by_index = {
                executor.submit(
                    _run_batch_item,
                    item,
                    worker=worker,
                    model_threads=model_threads,
                    shared_dataset_path=shared_dataset_path,
                    profile_name=profile_name,
                    parent_overrides=parent_overrides,
                    resume_completed=resume_completed,
                    include_parent_overrides=include_parent_overrides,
                ): index
                for index, item in enumerate(items)
            }
            for future in as_completed(future_by_index):
                index = future_by_index[future]
                result = future.result()
                indexed_results.append((index, result))
                _print_batch_result(items[index], result)
                if fail_fast and not result.success:
                    for pending in future_by_index:
                        pending.cancel()
                    break
        results = [result for _, result in sorted(indexed_results)]

    failures = [result for result in results if not result.success]
    if failures:
        details = "; ".join(
            f"{Path(result.config_path).name}: {result.error}" for result in failures
        )
        raise RuntimeError(f"ML research batch failed: {details}")
    successful_output_dirs = [Path(result.output_dir) for result in results]
    if successful_output_dirs:
        leaderboard_markdown_path, _ = _update_source_leaderboard(
            config,
            successful_output_dirs[0],
            successful_output_dirs[1:],
        )
        print(f"Leaderboard: {leaderboard_markdown_path}")
    return results


def _run_batch_item(
    item: MLResearchBatchItem,
    *,
    worker: Callable[..., MLResearchBatchResult],
    model_threads: int,
    shared_dataset_path: str,
    profile_name: str,
    parent_overrides: dict[str, Any],
    resume_completed: bool,
    include_parent_overrides: bool,
) -> MLResearchBatchResult:
    try:
        if include_parent_overrides:
            return worker(
                str(item.config_path),
                model_threads,
                shared_dataset_path,
                profile_name,
                parent_overrides,
                resume_completed,
            )
        return worker(
            str(item.config_path),
            model_threads,
            shared_dataset_path,
            profile_name,
        )
    except Exception as exc:  # pragma: no cover - defensive process boundary
        return MLResearchBatchResult(
            config_path=str(item.config_path),
            output_dir=str(item.output_dir),
            success=False,
            error=str(exc),
        )


def _print_batch_result(item: MLResearchBatchItem, result: MLResearchBatchResult) -> None:
    status = "ok" if result.success else "failed"
    print(f"{status}: {item.config_path} -> {result.output_dir}")
    if result.error:
        print(f"  error: {result.error}")


def _worker_accepts_parent_overrides(worker: Callable[..., MLResearchBatchResult]) -> bool:
    signature = inspect.signature(worker)
    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return len(positional) >= 5

def validate_ml_research_batch_config(config: dict[str, Any]) -> list[MLResearchBatchItem]:
    batch_config = config.get("ml_research_batch")
    if not isinstance(batch_config, dict):
        raise RuntimeError("ml-research-batch requires ml_research_batch config")
    config_paths = batch_config.get("config_paths", [])
    if not config_paths:
        raise RuntimeError("ml_research_batch.config_paths must contain at least one config")
    max_workers = int(batch_config.get("max_workers", 1))
    model_threads = int(batch_config.get("model_threads", 1))
    if max_workers < 1:
        raise RuntimeError("ml_research_batch.max_workers must be at least one")
    if model_threads < 1:
        raise RuntimeError("ml_research_batch.model_threads must be at least one")

    expanded_dataset_path = _expanded_rebalance_dataset_path(config)
    if not expanded_dataset_path.exists():
        raise RuntimeError(
            "ml-research-batch requires existing expanded rebalance dataset: "
            f"{expanded_dataset_path}"
        )

    items = []
    output_dirs: dict[Path, Path] = {}
    shared_cache_dir = Path(config.get("cache", {}).get("ml_dir", "cache/ml")).resolve()
    profile_name = str(config.get("research_profile", {}).get("name", "") or "")
    for raw_path in config_paths:
        config_path = Path(str(raw_path))
        if not config_path.exists():
            raise RuntimeError(f"Batch research config does not exist: {config_path}")
        child_config = apply_research_profile(
            load_config(str(config_path), overlay_project_config=True),
            profile_name or None,
        )
        child_cache_dir = Path(
            child_config.get("cache", {}).get("ml_dir", "cache/ml")
        ).resolve()
        if child_cache_dir != shared_cache_dir:
            raise RuntimeError(
                "All ml-research-batch configs must use the shared cache ml_dir "
                f"{shared_cache_dir}; {config_path} uses {child_cache_dir}"
            )
        output_dir = _batch_output_dir(config, child_config, config_path).resolve()
        if output_dir in output_dirs:
            raise RuntimeError(
                "Duplicate ml.output_dir in batch configs: "
                f"{output_dir} used by {output_dirs[output_dir]} and {config_path}"
            )
        output_dirs[output_dir] = config_path
        items.append(MLResearchBatchItem(config_path=config_path, output_dir=output_dir))
    return items

def _expanded_rebalance_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )

def _run_ml_research_batch_worker(
    config_path: str,
    model_threads: int,
    expanded_dataset_path: str,
    profile_name: str = "",
    parent_overrides: dict[str, Any] | None = None,
    resume_completed: bool = True,
) -> MLResearchBatchResult:
    apply_worker_thread_environment(model_threads)

    config = apply_research_profile(
        load_config(config_path, overlay_project_config=True),
        profile_name or None,
    )
    ml_config = config.setdefault("ml", {})
    ml_config.setdefault("model_threads", model_threads)
    ml_config.setdefault("torch_num_threads", model_threads)
    ml_config.setdefault("sklearn_n_jobs", model_threads)
    output_root = (parent_overrides or {}).get("ml_research_batch", {}).get("output_dir_root")
    if output_root:
        ml_config["output_dir"] = str(Path(str(output_root)) / Path(config_path).stem)
    apply_runtime_parallelism(config)
    output_dir = Path(config.get("ml", {}).get("output_dir", "reports/ml"))
    worker_config = _batch_worker_config(config, expanded_dataset_path, parent_overrides)
    completed_check = (
        _completed_ml_research_output_check(worker_config)
        if resume_completed
        else CompletedMLResearchOutputCheck(False, "resume_completed_disabled")
    )
    if completed_check.reusable:
        paths = MLExperimentPathBuilder(
            worker_config,
            MLExperimentConfig.from_config(worker_config),
        ).build()
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(paths.output_dir),
            success=True,
            metrics_path=str(paths.metrics_path),
            prediction_artifacts_path=str(paths.prediction_artifacts_path),
        )
    if resume_completed and completed_check.reason:
        print(
            "ml-research-batch resume rejected for "
            f"{Path(config_path).name}: {completed_check.reason}"
        )
    try:
        result = MLExperimentRunner(
            worker_config,
            feed=_build_research_feed(worker_config),
        ).run()
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(result.output_dir),
            success=True,
            metrics_path=str(result.metrics_path),
            prediction_artifacts_path=str(result.prediction_artifacts_path),
        )
    except Exception as exc:
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(output_dir),
            success=False,
            error=str(exc),
        )

def _batch_worker_config(
    config: dict[str, Any],
    expanded_dataset_path: str,
    parent_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_config = deepcopy(config)
    if parent_overrides:
        for section, values in parent_overrides.items():
            if isinstance(values, dict):
                worker_config.setdefault(section, {}).update(values)
    ml_config = worker_config.setdefault("ml", {})
    output_dir = Path(ml_config.get("output_dir", "reports/ml"))
    worker_config.setdefault("cache", {})["enabled"] = False
    worker_config.setdefault("cache", {})["ml_dir"] = str(output_dir / "_batch_cache")
    ml_config["read_existing_expanded_rebalance_dataset"] = True
    ml_config["expanded_rebalance_dataset_path"] = expanded_dataset_path
    return worker_config


def _batch_output_dir(
    parent_config: dict[str, Any],
    child_config: dict[str, Any],
    config_path: Path,
) -> Path:
    output_root = (parent_config.get("ml_research_batch", {}) or {}).get("output_dir_root")
    if output_root:
        return Path(str(output_root)) / config_path.stem
    return Path(child_config.get("ml", {}).get("output_dir", "reports/ml"))


def _completed_ml_research_output(config: dict[str, Any]) -> bool:
    return _completed_ml_research_output_check(config).reusable


def _completed_ml_research_output_check(
    config: dict[str, Any],
) -> CompletedMLResearchOutputCheck:
    experiment_config = MLExperimentConfig.from_config(config)
    builder = MLExperimentPathBuilder(
        config,
        experiment_config,
    )
    output_dir = _completed_ml_research_artifact_dir(config, experiment_config)
    required = (
        output_dir / "metrics.json",
        output_dir / "metadata.json",
        output_dir / "predictions.csv",
        output_dir / builder.model_filename(),
        output_dir / "prediction_artifacts.csv",
        output_dir / "prediction_artifacts.json",
        output_dir / "dataset_audit.json",
    )
    missing = [path.name for path in required if not path.exists() or path.stat().st_size <= 0]
    if missing:
        return CompletedMLResearchOutputCheck(
            False,
            "missing_or_empty_required_artifacts:" + ",".join(missing),
        )
    try:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return CompletedMLResearchOutputCheck(False, "malformed_metadata")
    if not isinstance(metadata, dict):
        return CompletedMLResearchOutputCheck(False, "malformed_metadata")

    expected_config_hash = MLCoreArtifactWriter.hash_payload(config)
    if metadata.get("config_hash") != expected_config_hash:
        return CompletedMLResearchOutputCheck(False, "config_hash_mismatch")
    if metadata.get("run_status") != "complete":
        return CompletedMLResearchOutputCheck(False, "run_status_not_complete")
    if metadata.get("model_input_contract_version") != MODEL_INPUT_CONTRACT_VERSION:
        return CompletedMLResearchOutputCheck(
            False,
            "model_input_contract_version_mismatch",
        )
    if metadata.get("model_type") != experiment_config.model_type:
        return CompletedMLResearchOutputCheck(False, "model_type_mismatch")
    if metadata.get("model_name") not in {None, experiment_config.model_type}:
        return CompletedMLResearchOutputCheck(False, "model_name_mismatch")
    if metadata.get("label_type") != experiment_config.label_type:
        return CompletedMLResearchOutputCheck(False, "label_type_mismatch")
    if metadata.get("target_label_name") != experiment_config.label_type:
        return CompletedMLResearchOutputCheck(False, "target_label_mismatch")
    if metadata.get("feature_set") != experiment_config.feature_set:
        return CompletedMLResearchOutputCheck(False, "feature_set_mismatch")
    if not metadata.get("dataset_hash"):
        return CompletedMLResearchOutputCheck(False, "missing_dataset_hash")
    if not metadata.get("model_input_hash"):
        return CompletedMLResearchOutputCheck(False, "missing_model_input_hash")
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        return CompletedMLResearchOutputCheck(False, "missing_or_invalid_feature_columns")
    if metadata.get("feature_count") != len(feature_columns):
        return CompletedMLResearchOutputCheck(False, "feature_count_mismatch")
    if metadata.get("sample_count") != metadata.get("source_dataset_row_count"):
        return CompletedMLResearchOutputCheck(False, "sample_count_mismatch")

    metrics_check = _json_artifact_matches_metadata(
        output_dir / "metrics.json",
        metadata,
        keys=("dataset_hash", "model_type", "label_type", "feature_set"),
    )
    if metrics_check:
        return CompletedMLResearchOutputCheck(False, metrics_check)
    dataset_audit_check = _json_artifact_matches_metadata(
        output_dir / "dataset_audit.json",
        metadata,
        keys=("sample_count", "feature_count"),
    )
    if dataset_audit_check:
        return CompletedMLResearchOutputCheck(False, dataset_audit_check)

    try:
        prediction_validation = validate_prediction_artifacts(
            output_dir / "prediction_artifacts.csv",
            output_dir / "prediction_artifacts.json",
        )
        prediction_metadata = json.loads(
            (output_dir / "prediction_artifacts.json").read_text(encoding="utf-8")
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        return CompletedMLResearchOutputCheck(False, "invalid_prediction_artifacts")
    if prediction_validation.dataset_hash != metadata.get("dataset_hash"):
        return CompletedMLResearchOutputCheck(False, "prediction_dataset_hash_mismatch")
    prediction_checks = {
        "model_input_contract_version": "prediction_model_input_contract_version_mismatch",
        "model_input_hash": "prediction_model_input_hash_mismatch",
        "feature_columns": "prediction_feature_columns_mismatch",
        "feature_count": "prediction_feature_count_mismatch",
        "sample_count": "prediction_sample_count_mismatch",
        "target_label_name": "prediction_target_label_mismatch",
        "label_type": "prediction_label_type_mismatch",
        "model_type": "prediction_model_type_mismatch",
        "model_input_source_path": "prediction_model_input_source_mismatch",
    }
    for key, reason in prediction_checks.items():
        if prediction_metadata.get(key) != metadata.get(key):
            return CompletedMLResearchOutputCheck(False, reason)

    current_identity = _current_model_input_identity(config, experiment_config)
    if current_identity is None:
        return CompletedMLResearchOutputCheck(
            False,
            "current_input_identity_unavailable",
        )
    current_check = _current_model_input_matches_metadata(
        current_identity.values,
        metadata,
    )
    if current_check:
        return CompletedMLResearchOutputCheck(False, current_check)

    return CompletedMLResearchOutputCheck(True)


def _completed_ml_research_artifact_dir(
    config: dict[str, Any],
    experiment_config: MLExperimentConfig,
) -> Path:
    fixed_output_dir = Path(experiment_config.output_dir)
    current_identity = _current_model_input_identity(config, experiment_config)
    if current_identity is None:
        return fixed_output_dir
    identity = {
        **current_identity.values,
        "model_name": experiment_config.model_type,
        "feature_set": experiment_config.feature_set,
    }
    expected_run_id = deterministic_run_id("exposure_ml", identity)
    expected_run_dir = immutable_run_dir(fixed_output_dir, expected_run_id)
    if expected_run_dir.exists():
        return expected_run_dir
    latest_run_dir = run_dir_from_latest_completed(fixed_output_dir)
    if latest_run_dir is None:
        return fixed_output_dir
    manifest = read_run_manifest(latest_run_dir)
    if not manifest:
        return fixed_output_dir
    if manifest.get("run_id") == expected_run_id and manifest.get("identity") == identity:
        return latest_run_dir
    return fixed_output_dir


def _current_model_input_identity(
    config: dict[str, Any],
    experiment_config: MLExperimentConfig,
) -> CurrentModelInputIdentity | None:
    if experiment_config.label_type != "should_reduce_exposure":
        return None
    ml_config = config.get("ml", {}) or {}
    if not bool(ml_config.get("read_existing_expanded_rebalance_dataset", False)):
        return None
    source_path = Path(
        ml_config.get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )
    try:
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    feature_result = MLFeatureBuildResult(
        rows=rows,
        dropped_rows=0,
        date_range=(
            str(rows[0].get("feature_date", "")),
            str(rows[-1].get("feature_date", "")),
        ),
    )
    try:
        label_result = ShouldReduceExposureLabelBuilder().build(feature_result.rows)
        dataset = MLDatasetPipeline(experiment_config).build_dataset(
            feature_result,
            label_result,
        )
    except (KeyError, TypeError, ValueError):
        return None
    writer = MLCoreArtifactWriter(
        config,
        experiment_config,
        research_label=str(ml_config.get("research_label", "UNSPECIFIED_RESEARCH")),
    )
    feature_columns = writer.model_input_feature_columns(dataset)
    feature_date_min = min(dataset.feature_dates) if dataset.feature_dates else None
    feature_date_max = max(dataset.feature_dates) if dataset.feature_dates else None
    return CurrentModelInputIdentity({
        "config_hash": MLCoreArtifactWriter.hash_payload(config),
        "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "dataset_hash": writer.source_dataset_hash(dataset),
        "model_input_hash": writer.model_input_hash(dataset),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "sample_count": dataset.sample_count,
        "feature_date_min": feature_date_min,
        "feature_date_max": feature_date_max,
        "training_date_min": feature_date_min,
        "training_date_max": feature_date_max,
        "target_label_name": experiment_config.label_type,
        "label_type": experiment_config.label_type,
        "model_type": experiment_config.model_type,
        "model_input_source_path": str(source_path.resolve()),
    })


def _current_model_input_matches_metadata(
    current: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    checks = (
        ("model_input_contract_version", "current_contract_version_mismatch"),
        ("config_hash", "current_config_hash_mismatch"),
        ("model_input_source_path", "current_input_source_mismatch"),
        ("target_label_name", "current_target_mismatch"),
        ("label_type", "current_target_mismatch"),
        ("model_type", "current_model_type_mismatch"),
        ("feature_columns", "current_feature_columns_mismatch"),
        ("feature_count", "current_feature_count_mismatch"),
        ("sample_count", "current_sample_count_mismatch"),
        ("feature_date_min", "current_date_coverage_mismatch"),
        ("feature_date_max", "current_date_coverage_mismatch"),
        ("training_date_min", "current_date_coverage_mismatch"),
        ("training_date_max", "current_date_coverage_mismatch"),
        ("dataset_hash", "current_dataset_hash_mismatch"),
        ("model_input_hash", "current_model_input_hash_mismatch"),
    )
    for key, reason in checks:
        if current.get(key) != metadata.get(key):
            if key == "feature_columns" and isinstance(metadata.get(key), list):
                if set(current.get(key, [])) == set(metadata.get(key, [])):
                    return "current_feature_order_mismatch"
            return reason
    return ""


def _json_artifact_matches_metadata(
    path: Path,
    metadata: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return f"malformed_{path.name}"
    if not isinstance(payload, dict):
        return f"malformed_{path.name}"
    for key in keys:
        if key in payload and payload.get(key) != metadata.get(key):
            return f"{path.name}_{key}_mismatch"
    return ""


def _batch_parent_overrides(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = dict(config.get("ml", {}) or {})
    backtest_config = dict(config.get("backtest", {}) or {})
    overrides: dict[str, Any] = {
        "backtest": {
            key: backtest_config[key]
            for key in ("provider", "data_dir", "timeframe", "years")
            if key in backtest_config
        },
        "ml": {
            key: ml_config[key]
            for key in (
                "historical_data_provider",
                "data_dir",
                "parquet_dir",
                "stooq_parquet_dir",
                "market_data",
                "benchmark_symbols",
                "num_workers",
                "feature_workers",
                "model_threads",
                "torch_num_threads",
                "sklearn_n_jobs",
            )
            if key in ml_config
        },
        "cache": dict(config.get("cache", {}) or {}),
        "ml_research_batch": {
            key: config.get("ml_research_batch", {})[key]
            for key in ("output_dir_root",)
            if key in config.get("ml_research_batch", {})
        },
    }
    return {section: values for section, values in overrides.items() if values}

def _build_research_feed(config: dict[str, Any]):
    provider = config.get("backtest", {}).get("provider", "alpaca").lower()
    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed

        return StooqParquetDataFeed(
            data_dir=config.get("backtest", {}).get(
                "data_dir", "data/processed/stooq_parquet"
            )
        )
    if provider == "market_parquet":
        from infrastructure.data.market_parquet import MarketParquetDataFeed

        return MarketParquetDataFeed(
            data_root=config.get("backtest", {}).get("data_dir", "data/processed")
        )
    if provider == "stooq_csv":
        from infrastructure.data.stooq_csv_data_feed import StooqCsvDataFeed

        return StooqCsvDataFeed(
            data_dir=config.get("backtest", {}).get("data_dir", "data/raw/stooq")
        )
    if provider == "stooq":
        from infrastructure.data.stooq_data_feed import StooqDataFeed

        return StooqDataFeed()
    raise RuntimeError(
        "ml-research-batch supports local research data providers only; "
        f"unsupported provider '{provider}'"
    )
