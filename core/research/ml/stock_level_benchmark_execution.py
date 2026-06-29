from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from core.research.framework.parallel import ParallelTaskExecutor
from core.research.framework.walk_forward import (
    ExpandingWindowSpec,
    ExpandingWindowSplitter,
)
from core.research.ml.stock_level_benchmark_types import (
    AUXILIARY_TARGET_COLUMNS,
    ModelRunSpec,
    TARGET_COLUMN,
)


_MODEL_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _execute_model_runs(
    specs: list[ModelRunSpec],
    *,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
    model_n_jobs: int,
    executor_cls: type,
) -> tuple[
    dict[str, dict[tuple[str, str], float]],
    dict[str, str],
    dict[str, float],
]:
    arguments = (
        prepared_rows,
        dates,
        first_test_index,
        test_window_dates,
        embargo_dates,
        sequence_length,
        1 if model_n_jobs > 1 else None,
    )
    if model_n_jobs == 1:
        worker = lambda spec: _run_model_walk_forward(spec, *arguments)
        initializer = None
        initargs = ()
    else:
        worker = _run_initialized_model
        initializer = _initialize_model_worker
        initargs = (arguments,)
    execution = ParallelTaskExecutor[ModelRunSpec, dict[tuple[str, str], float]]()
    result = execution.execute(
        specs,
        worker,
        key=lambda spec: spec.name,
        max_workers=model_n_jobs,
        executor_cls=executor_cls,
        initializer=initializer,
        initargs=initargs,
    )
    return result.results, result.errors, result.timings


def _initialize_model_worker(arguments: tuple[Any, ...]) -> None:
    global _MODEL_WORKER_CONTEXT
    _MODEL_WORKER_CONTEXT = arguments


def _run_initialized_model(
    spec: ModelRunSpec,
) -> dict[tuple[str, str], float]:
    if _MODEL_WORKER_CONTEXT is None:
        raise RuntimeError("Model worker context was not initialized")
    return _run_model_walk_forward(spec, *_MODEL_WORKER_CONTEXT)


def _run_model_walk_forward(
    spec: ModelRunSpec,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
    native_thread_limit: int | None,
) -> dict[tuple[str, str], float]:
    thread_context: Any = nullcontext()
    if native_thread_limit is not None:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # no native sklearn/BLAS pools in this environment
            pass
        else:
            thread_context = threadpool_limits(limits=native_thread_limit)
    with thread_context:
        return _run_model_walk_forward_unlimited(
            spec,
            prepared_rows,
            dates,
            first_test_index,
            test_window_dates,
            embargo_dates,
            sequence_length,
        )


def _run_model_walk_forward_unlimited(
    spec: ModelRunSpec,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
) -> dict[tuple[str, str], float]:
    predictions: dict[tuple[str, str], float] = {}
    for _, train_rows, test_rows, _, _, _ in _walk_forward_partitions(
        prepared_rows,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    ):
        model = spec.factory()
        if spec.kind == "tabular":
            x_train = [
                [row[column] for column in spec.feature_columns]
                for row in train_rows
            ]
            x_test = [
                [row[column] for column in spec.feature_columns]
                for row in test_rows
            ]
            model.fit(x_train, [row[TARGET_COLUMN] for row in train_rows])
            values = [float(value) for value in model.predict(x_test)]
            prediction_rows = test_rows
        else:
            train_sequences, sequence_train_rows = _build_sequences(
                prepared_rows,
                train_rows,
                spec.feature_columns,
                sequence_length,
            )
            test_sequences, prediction_rows = _build_sequences(
                prepared_rows,
                test_rows,
                spec.feature_columns,
                sequence_length,
            )
            if len(prediction_rows) != len(test_rows):
                raise ValueError(
                    f"{spec.name} cannot produce one prediction per OOS row; "
                    "increase min_train_dates or reduce sequence_length"
                )
            auxiliary_targets = (
                [
                    [row[column] for column in AUXILIARY_TARGET_COLUMNS]
                    for row in sequence_train_rows
                ]
                if spec.name == "multitask_transformer"
                else None
            )
            model.fit(
                train_sequences,
                [row[TARGET_COLUMN] for row in sequence_train_rows],
                auxiliary_targets,
            )
            values = [float(value) for value in model.predict(test_sequences)]
        if len(values) != len(prediction_rows):
            raise ValueError(
                f"{spec.name} returned {len(values)} predictions for "
                f"{len(prediction_rows)} OOS rows"
            )
        predictions.update(
            {
                (row["rebalance_date"], row["symbol"]): value
                for row, value in zip(prediction_rows, values)
            }
        )
    return predictions


def _walk_forward_partitions(
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    *,
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
):
    splitter = ExpandingWindowSplitter(
        ExpandingWindowSpec(
            min_train_dates=first_test_index - embargo_dates,
            test_window_dates=test_window_dates,
            embargo_dates=embargo_dates,
        )
    )
    for fold in splitter.split(prepared_rows, dates=dates):
        yield (
            fold.fold_id,
            list(fold.train_rows),
            list(fold.test_rows),
            list(fold.train_dates),
            list(fold.test_dates),
            list(fold.embargoed_dates),
        )


def _build_sequences(
    all_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    sequence_length: int,
) -> tuple[list[list[list[float]]], list[dict[str, Any]]]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    target_keys = {
        (row["rebalance_date"], row["symbol"]): row for row in target_rows
    }
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
    keyed_sequences: list[tuple[tuple[str, str], list[list[float]], dict[str, Any]]] = []
    for symbol, symbol_rows in by_symbol.items():
        ordered = sorted(symbol_rows, key=lambda row: row["rebalance_date"])
        for end_index in range(sequence_length - 1, len(ordered)):
            end_row = ordered[end_index]
            key = (end_row["rebalance_date"], symbol)
            target_row = target_keys.get(key)
            if target_row is None:
                continue
            window = ordered[end_index - sequence_length + 1 : end_index + 1]
            keyed_sequences.append(
                (
                    key,
                    [
                        [float(row.get(column, 0.0)) for column in feature_columns]
                        for row in window
                    ],
                    target_row,
                )
            )
    keyed_sequences.sort(key=lambda item: item[0])
    return (
        [item[1] for item in keyed_sequences],
        [item[2] for item in keyed_sequences],
    )
