from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar
import time


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ParallelExecutionResult(Generic[ResultT]):
    results: dict[str, ResultT]
    errors: dict[str, str]
    timings: dict[str, float]


class ParallelTaskExecutor(Generic[TaskT, ResultT]):
    """Execute independent tasks while preserving registry/input ordering."""

    def execute(
        self,
        tasks: Sequence[TaskT],
        worker: Callable[[TaskT], ResultT],
        *,
        key: Callable[[TaskT], str],
        max_workers: int = 1,
        executor_cls: type[Executor] = ProcessPoolExecutor,
        initializer: Callable[..., None] | None = None,
        initargs: tuple = (),
    ) -> ParallelExecutionResult[ResultT]:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if not tasks:
            return ParallelExecutionResult({}, {}, {})
        completed: dict[str, ResultT] = {}
        errors: dict[str, str] = {}
        timings: dict[str, float] = {}
        if max_workers == 1:
            for task in tasks:
                name = key(task)
                started = time.perf_counter()
                try:
                    completed[name] = worker(task)
                except Exception as exc:  # isolated task boundary
                    errors[name] = f"{type(exc).__name__}: {exc}"
                timings[name] = time.perf_counter() - started
                print(f"[parallel] completed task={name} elapsed={timings[name]:.3f}s")
            return ParallelExecutionResult(completed, errors, timings)

        kwargs = {"max_workers": min(max_workers, len(tasks))}
        if initializer is not None:
            kwargs.update({"initializer": initializer, "initargs": initargs})
        with executor_cls(**kwargs) as executor:
            futures = {executor.submit(worker, task): (task, time.perf_counter()) for task in tasks}
            for future in as_completed(futures):
                task, started = futures[future]
                name = key(task)
                try:
                    completed[name] = future.result()
                except Exception as exc:  # isolated executor boundary
                    errors[name] = f"{type(exc).__name__}: {exc}"
                timings[name] = time.perf_counter() - started
                print(f"[parallel] completed task={name} elapsed={timings[name]:.3f}s")
        ordered_results = {
            key(task): completed[key(task)]
            for task in tasks
            if key(task) in completed
        }
        ordered_errors = {
            key(task): errors[key(task)] for task in tasks if key(task) in errors
        }
        ordered_timings = {key(task): timings[key(task)] for task in tasks if key(task) in timings}
        return ParallelExecutionResult(ordered_results, ordered_errors, ordered_timings)
