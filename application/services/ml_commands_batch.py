from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from config.config_loader import load_config
from application.services.research_profiles import apply_research_profile
from application.services.ml_commands_types import MLResearchBatchItem, MLResearchBatchResult
from application.services.ml_commands_artifacts import _update_source_leaderboard
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.runtime_parallelism import apply_runtime_parallelism, apply_worker_thread_environment, format_runtime_settings


def run_ml_research_batch(
    config: dict[str, Any],
    *,
    executor_cls: type[ProcessPoolExecutor] = ProcessPoolExecutor,
    worker_fn: Callable[[str, int, str], MLResearchBatchResult] | None = None,
) -> list[MLResearchBatchResult]:
    batch_config = config.get("ml_research_batch", {})
    runtime_settings = apply_runtime_parallelism(config)
    items = validate_ml_research_batch_config(config)
    max_workers = int(batch_config.get("max_workers", runtime_settings.num_workers))
    model_threads = int(batch_config.get("model_threads", runtime_settings.model_threads))
    fail_fast = bool(batch_config.get("fail_fast", True))
    shared_dataset_path = str(_expanded_rebalance_dataset_path(config))
    profile_name = str(config.get("research_profile", {}).get("name", "") or "")
    worker = worker_fn or _run_ml_research_batch_worker

    print("\nML RESEARCH BATCH")
    print("mode=research | trading_impact=none")
    print(f"Configs: {len(items)}")
    print(f"Workers: {max_workers}")
    print(f"Model threads: {model_threads}")
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Shared expanded dataset: {shared_dataset_path}")

    results: list[MLResearchBatchResult] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker,
                str(item.config_path),
                model_threads,
                shared_dataset_path,
                profile_name,
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive process boundary
                result = MLResearchBatchResult(
                    config_path=str(item.config_path),
                    output_dir=str(item.output_dir),
                    success=False,
                    error=str(exc),
                )
            results.append(result)
            status = "ok" if result.success else "failed"
            print(f"{status}: {result.config_path} -> {result.output_dir}")
            if result.error:
                print(f"  error: {result.error}")
            if result.success:
                leaderboard_markdown_path, _ = _update_source_leaderboard(
                    config,
                    Path(result.output_dir),
                )
                print(f"  leaderboard: {leaderboard_markdown_path}")
            if fail_fast and not result.success:
                for pending in futures:
                    pending.cancel()
                break

    failures = [result for result in results if not result.success]
    if failures:
        details = "; ".join(
            f"{Path(result.config_path).name}: {result.error}" for result in failures
        )
        raise RuntimeError(f"ML research batch failed: {details}")
    return results

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
        output_dir = Path(
            child_config.get("ml", {}).get("output_dir", "reports/ml")
        ).resolve()
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
    apply_runtime_parallelism(config)
    output_dir = Path(config.get("ml", {}).get("output_dir", "reports/ml"))
    worker_config = _batch_worker_config(config, expanded_dataset_path)
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
) -> dict[str, Any]:
    worker_config = deepcopy(config)
    ml_config = worker_config.setdefault("ml", {})
    output_dir = Path(ml_config.get("output_dir", "reports/ml"))
    worker_config.setdefault("cache", {})["enabled"] = False
    worker_config.setdefault("cache", {})["ml_dir"] = str(output_dir / "_batch_cache")
    ml_config["read_existing_expanded_rebalance_dataset"] = True
    ml_config["expanded_rebalance_dataset_path"] = expanded_dataset_path
    return worker_config

def _build_research_feed(config: dict[str, Any]):
    provider = config.get("backtest", {}).get("provider", "alpaca").lower()
    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed

        return StooqParquetDataFeed(
            data_dir=config.get("backtest", {}).get(
                "data_dir", "data/processed/stooq_parquet"
            )
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
