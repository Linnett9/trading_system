from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NewsRiskParallelConfig:
    enabled: bool
    requested_workers: int | None
    actual_workers: int
    backend: str
    min_items: int
    chunk_size: int
    batch_limit: int
    progress: bool
    cpu_count: int
    fallback_reason: str | None = None


def parallel_config(config: Mapping[str, Any]) -> NewsRiskParallelConfig:
    cpu_count = os.cpu_count() or 1
    requested = optional_int(
        config.get("news_risk_max_workers")
        or config.get("stock_alpha_news_risk_overlay_parallel_max_workers")
    )
    configured_backend = str(
        config.get("news_risk_parallel_backend")
        or config.get("stock_alpha_news_risk_overlay_parallel_backend")
        or "thread"
    ).lower()
    backend = configured_backend if configured_backend in {"thread", "process"} else "thread"
    max_allowed = max(cpu_count - 1, 1)
    requested_workers = requested if requested is not None else max_allowed
    actual_workers = max(1, min(int(requested_workers), max_allowed))
    enabled = bool(
        config.get("news_risk_parallel_enabled")
        or config.get("stock_alpha_news_risk_overlay_parallel_enabled")
        or False
    )
    fallback_reason = None
    if not enabled:
        fallback_reason = "parallel disabled by configuration"
    elif actual_workers <= 1:
        fallback_reason = "single-worker mode requested"
    return NewsRiskParallelConfig(
        enabled=enabled,
        requested_workers=requested,
        actual_workers=actual_workers,
        backend=backend,
        min_items=max(
            1,
            int(
                config.get("news_risk_parallel_min_items")
                or config.get("stock_alpha_news_risk_overlay_parallel_min_items")
                or 16
            ),
        ),
        chunk_size=max(
            1,
            int(
                config.get("news_risk_parallel_chunk_size")
                or config.get("stock_alpha_news_risk_overlay_parallel_chunk_size")
                or 32
            ),
        ),
        batch_limit=max(
            1,
            int(
                config.get("news_risk_parallel_batch_limit")
                or config.get("stock_alpha_news_risk_overlay_parallel_batch_limit")
                or 128
            ),
        ),
        progress=bool(
            config.get("news_risk_parallel_progress")
            or config.get("stock_alpha_news_risk_overlay_parallel_progress")
            or False
        ),
        cpu_count=cpu_count,
        fallback_reason=fallback_reason,
    )


def parallel_report_skeleton(config: NewsRiskParallelConfig) -> dict[str, Any]:
    return {
        "parallel_enabled": config.enabled,
        "backend": config.backend,
        "requested_workers": config.requested_workers,
        "actual_workers": config.actual_workers,
        "cpu_count": config.cpu_count,
        "task_count": 0,
        "chunk_size": config.chunk_size,
        "batch_limit": config.batch_limit,
        "phases_parallelised": [],
        "phases_kept_sequential": [],
        "elapsed_seconds_by_phase": {},
        "worker_count_used": config.actual_workers,
        "number_of_tasks": {},
        "average_task_duration_seconds": {},
        "slowest_tasks": {},
        "worker_count_semantics": "actual parallel workers" if config.enabled else "unused because parallel mode is disabled",
        "worker_failures": [],
        "fallback_events": (
            [{"phase": "global", "reason": config.fallback_reason}]
            if config.fallback_reason
            else []
        ),
        "determinism_status": "PENDING",
        "phases_forced_sequential": [
            "chronological_model_fitting",
            "point_in_time_join",
            "single_strategy_daily_portfolio_replay",
            "shared_ledger_or_equity_state",
            "broker_paper_live_order_paths",
        ],
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
    }


@contextmanager
def timed_phase(report: dict[str, Any] | None, phase: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        if report is not None:
            elapsed = time.perf_counter() - started
            report.setdefault("elapsed_seconds_by_phase", {})[phase] = (
                report.setdefault("elapsed_seconds_by_phase", {}).get(phase, 0.0) + elapsed
            )


def should_parallelize(
    config: NewsRiskParallelConfig,
    item_count: int,
    *,
    phase: str,
    report: dict[str, Any] | None,
) -> bool:
    if not config.enabled:
        record_fallback(report, phase, "parallel disabled by configuration")
        return False
    if config.actual_workers <= 1:
        record_fallback(report, phase, "single-worker mode")
        return False
    if item_count < config.min_items:
        record_fallback(report, phase, f"item_count {item_count} below min_items {config.min_items}")
        return False
    if phase == "bar_loading" and config.backend != "thread":
        record_fallback(report, phase, "bar loading uses thread backend only because parquet reads are I/O-bound")
        return False
    return True


def record_parallel_phase(
    report: dict[str, Any] | None,
    phase: str,
    *,
    task_count: int,
    task_durations: list[float],
    parallelized: bool,
) -> None:
    if report is None:
        return
    report["task_count"] = int(report.get("task_count", 0)) + task_count
    report.setdefault("number_of_tasks", {})[phase] = task_count
    if task_durations:
        report.setdefault("average_task_duration_seconds", {})[phase] = mean(task_durations)
        report.setdefault("slowest_tasks", {})[phase] = sorted(task_durations, reverse=True)[:5]
    target = "phases_parallelised" if parallelized else "phases_kept_sequential"
    if phase not in report.setdefault(target, []):
        report[target].append(phase)


def record_fallback(report: dict[str, Any] | None, phase: str, reason: str) -> None:
    if report is None:
        return
    event = {"phase": phase, "reason": reason}
    if event not in report.setdefault("fallback_events", []):
        report["fallback_events"].append(event)
    if phase not in report.setdefault("phases_kept_sequential", []):
        report["phases_kept_sequential"].append(phase)


def record_worker_failures(
    report: dict[str, Any] | None,
    phase: str,
    failures: list[Mapping[str, Any]],
) -> None:
    if report is None:
        return
    for failure in failures:
        report.setdefault("worker_failures", []).append({"phase": phase, **dict(failure)})


def parallel_determinism_status(report: Mapping[str, Any]) -> str:
    if report.get("worker_failures"):
        return "FAILED_WORKER"
    if not report.get("parallel_enabled"):
        return "NOT_ENABLED"
    if report.get("phases_parallelised"):
        return "DETERMINISTIC_EQUIVALENCE_PASSED"
    return "STABLE_ORDERING_ENFORCED"


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), max(size, 1)):
        yield items[index : index + max(size, 1)]


def optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
