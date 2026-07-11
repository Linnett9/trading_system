from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any, Callable

from config.config_loader import load_config
from application.services.research_profiles import apply_research_profile
from application.services.ml_commands_types import MLResearchBatchItem, MLResearchBatchResult
from application.services.ml_commands_artifacts import _update_source_leaderboard
from core.research.ml.artifacts import MLExperimentPathBuilder
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MODEL_INPUT_CONTRACT_VERSION
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.runtime_parallelism import (
    apply_runtime_parallelism,
    apply_worker_thread_environment,
    format_runtime_settings,
)


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
    if resume_completed and _completed_ml_research_output(worker_config):
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
    experiment_config = MLExperimentConfig.from_config(config)
    builder = MLExperimentPathBuilder(
        config,
        experiment_config,
    )
    output_dir = Path(experiment_config.output_dir)
    required = (
        output_dir / "metrics.json",
        output_dir / "metadata.json",
        output_dir / "predictions.csv",
        output_dir / builder.model_filename(),
        output_dir / "prediction_artifacts.csv",
        output_dir / "prediction_artifacts.json",
        output_dir / "dataset_audit.json",
    )
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        import json

        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        metadata.get("config_hash") == MLCoreArtifactWriter.hash_payload(config)
        and metadata.get("model_input_contract_version")
        == MODEL_INPUT_CONTRACT_VERSION
    )


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
