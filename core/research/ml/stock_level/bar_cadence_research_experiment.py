from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_model_sets import (
    FULL_SEQUENCE_MODELS,
    TABULAR_MODELS,
    resolve_stock_alpha_model_set,
)
from core.research.ml.stock_level.bar_cadence_portfolio_replay import (
    BarCadenceReplayResult,
    build_bar_cadence_portfolio_replay,
)
from core.research.ml.stock_level.bar_targets import (
    add_forward_return_targets,
    label_maturity_column_name,
    target_column_name,
)
from core.research.ml.stock_level.feature_bank_adapter import (
    FeatureBankSlice,
    load_feature_bank_slice,
)
from core.research.ml.stock_level_benchmark_models import (
    _sequence_model_factories,
    stock_ranker_model_registry,
)


GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}


@dataclass(frozen=True)
class BarCadenceExperimentPaths:
    predictions_path: Path
    summary_json_path: Path
    summary_markdown_path: Path
    equity_curve_path: Path
    decisions_path: Path


@dataclass(frozen=True)
class BarCadenceExperimentResult:
    predictions: list[dict[str, Any]]
    replay: BarCadenceReplayResult
    payload: dict[str, Any]
    paths: BarCadenceExperimentPaths | None = None


def build_bar_cadence_research_experiment(
    config: Mapping[str, Any],
) -> BarCadenceExperimentResult:
    ml = dict(config.get("ml", {}) or {})
    timeframe = str(ml.get("stock_bar_cadence_replay_timeframe", "1Day"))
    horizon_bars = int(ml.get("stock_bar_cadence_target_horizon_bars", 1))
    model_names, model_set_metadata = _resolve_models(ml)
    feature_slice = load_feature_bank_slice(
        timeframe,
        path=ml.get("stock_bar_cadence_feature_bank_path"),
        symbols=ml.get("stock_bar_cadence_symbols"),
        start=ml.get("stock_bar_cadence_start"),
        end=ml.get("stock_bar_cadence_end"),
        columns=ml.get("stock_bar_cadence_feature_columns"),
        max_rows=ml.get("stock_bar_cadence_max_rows"),
    )
    requested_workers = max(1, int(ml.get("stock_bar_cadence_replay_n_jobs", 1)))
    independent_workers = min(requested_workers, max(1, len(model_names)))
    jobs = [
        (
            model_name,
            feature_slice,
            horizon_bars,
            int(ml.get("stock_bar_cadence_refit_frequency_bars", 20)),
            int(ml.get("stock_bar_cadence_min_train_rows", 50)),
            int(ml.get("stock_bar_cadence_replay_nested_sklearn_n_jobs", 1)),
            int(ml.get("stock_bar_cadence_replay_nested_torch_num_threads", 1)),
            int(ml.get("stock_bar_cadence_sequence_length", 13)),
            int(ml.get("stock_bar_cadence_sequence_epochs", 1)),
            int(ml.get("stock_bar_cadence_sequence_batch_size", 64)),
            int(ml.get("stock_bar_cadence_random_seed", 1729)),
            _coalesce_bool(ml.get("stock_bar_cadence_allow_cross_session_horizon"), timeframe == "1Day"),
            _expected_bar_seconds(timeframe, ml),
            _coalesce_bool(ml.get("stock_bar_cadence_allow_missing_intermediate_bars"), timeframe == "1Day"),
        )
        for model_name in model_names
    ]
    if independent_workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=independent_workers) as executor:
            prediction_results = list(executor.map(lambda args: _build_model_predictions(*args), jobs))
    else:
        prediction_results = [_build_model_predictions(*args) for args in jobs]
    predictions = [row for rows, _ in prediction_results for row in rows]
    prediction_payload_by_model = {
        model_name: payload
        for model_name, (_, payload) in zip(model_names, prediction_results)
    }
    primary_model = model_names[0]
    replays = {
        model_name: _build_replay_for_model(
            predictions,
            model_name=model_name,
            timeframe=timeframe,
            ml=ml,
        )
        for model_name in model_names
    }
    replay = replays[primary_model]
    grid_payload = build_replay_grid_from_predictions(
        [row for row in predictions if row["model"] == primary_model],
        timeframe=timeframe,
        scenarios=_replay_grid_scenarios(ml),
        max_workers=independent_workers,
    )
    payload = {
        "mode": "stock_bar_cadence_research_experiment",
        "timeframe": timeframe,
        "target_horizon_bars": horizon_bars,
        "model": primary_model,
        "models": list(model_names),
        "model_set": model_set_metadata,
        "feature_bank": feature_slice.metadata,
        "prediction_report": prediction_payload_by_model[primary_model],
        "prediction_report_by_model": prediction_payload_by_model,
        "replay_summary": replay.summary,
        "replay_summary_by_model": {
            model_name: item.summary for model_name, item in replays.items()
        },
        "replay_parallelism": replay.payload["parallelism"],
        "independent_unit_parallelism": {
            "requested_workers": requested_workers,
            "effective_workers": independent_workers,
            "backend": "thread",
            "work_unit": "independent model and replay-grid scenario",
            "full_dataset_copy_per_worker": False,
            "nested_sklearn_n_jobs": int(ml.get("stock_bar_cadence_replay_nested_sklearn_n_jobs", 1)),
            "nested_torch_num_threads": int(ml.get("stock_bar_cadence_replay_nested_torch_num_threads", 1)),
        },
        "replay_grid": grid_payload,
        **GUARDRAILS,
    }
    return BarCadenceExperimentResult(predictions, replay, payload)


def write_bar_cadence_research_experiment(
    config: Mapping[str, Any],
) -> BarCadenceExperimentPaths:
    result = build_bar_cadence_research_experiment(config)
    output = Path(
        (config.get("ml", {}) or {}).get(
            "stock_bar_cadence_output_root",
            "reports/ml/stock_bar_cadence",
        )
    )
    paths = BarCadenceExperimentPaths(
        output / "every_bar_oos_predictions.csv",
        output / "bar_cadence_replay_summary.json",
        output / "bar_cadence_replay_summary.md",
        output / "bar_cadence_equity_curve.csv",
        output / "bar_cadence_decisions.csv",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(
        paths.predictions_path,
        [_serializable(row) for row in result.predictions],
        fieldnames=list(_prediction_fieldnames()),
    )
    writer.write_json(paths.summary_json_path, _serializable(result.payload))
    writer.write_markdown(paths.summary_markdown_path, _markdown(result.payload))
    writer.write_csv(
        paths.equity_curve_path,
        result.replay.periods,
        fieldnames=list(result.replay.periods[0]) if result.replay.periods else ["decision_timestamp"],
    )
    writer.write_csv(
        paths.decisions_path,
        result.replay.decisions,
        fieldnames=list(result.replay.decisions[0]) if result.replay.decisions else ["decision_timestamp"],
    )
    return paths


def build_every_bar_oos_predictions(
    feature_slice: FeatureBankSlice,
    *,
    horizon_bars: int,
    model_name: str,
    refit_frequency_bars: int,
    min_train_rows: int,
    sklearn_n_jobs: int = 1,
    torch_num_threads: int = 1,
    sequence_length: int = 13,
    sequence_epochs: int = 1,
    sequence_batch_size: int = 64,
    random_seed: int = 1729,
    allow_cross_session_horizon: bool = True,
    expected_bar_seconds: int | None = None,
    allow_missing_intermediate_bars: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if refit_frequency_bars < 1:
        raise ValueError("refit_frequency_bars must be at least one")
    targeted_rows, target_metadata = add_forward_return_targets(
        feature_slice.rows,
        horizon_bars=horizon_bars,
        allow_cross_session_horizon=allow_cross_session_horizon,
        expected_bar_seconds=expected_bar_seconds,
        allow_missing_intermediate_bars=allow_missing_intermediate_bars,
    )
    target_column = target_column_name(horizon_bars)
    maturity_column = label_maturity_column_name(horizon_bars)
    feature_columns = _usable_feature_columns(targeted_rows, feature_slice.feature_columns)
    by_timestamp: dict[datetime, list[dict[str, Any]]] = {}
    for row in targeted_rows:
        by_timestamp.setdefault(row["timestamp"], []).append(row)
    timestamps = sorted(by_timestamp)
    model: Any | None = None
    fit_cutoff: datetime | None = None
    refit_id = 0
    scoring_since_refit = refit_frequency_bars
    predictions: list[dict[str, Any]] = []
    fit_events: list[dict[str, Any]] = []
    for timestamp in timestamps:
        should_refit = model is None or scoring_since_refit >= refit_frequency_bars
        if should_refit:
            train_rows = [
                row
                for row in targeted_rows
                if row["timestamp"] < timestamp
                and isinstance(row.get(maturity_column), datetime)
                and row[maturity_column] <= timestamp
                and finite_number(row.get(target_column)) is not None
            ]
            matrix, targets, used_columns = _matrix(train_rows, feature_columns, target_column=target_column)
            if len(targets) >= min_train_rows and used_columns:
                model = _new_model(
                    model_name,
                    random_seed=random_seed,
                    sklearn_n_jobs=sklearn_n_jobs,
                    torch_num_threads=torch_num_threads,
                    sequence_length=sequence_length,
                    sequence_epochs=sequence_epochs,
                    sequence_batch_size=sequence_batch_size,
                )
                if _is_sequence_model(model_name):
                    sequences, sequence_targets, used_columns = _sequence_training_matrix(
                        train_rows,
                        targeted_rows,
                        used_columns,
                        target_column=target_column,
                        sequence_length=sequence_length,
                    )
                    if len(sequence_targets) < min_train_rows:
                        model = None
                        continue
                    model.fit(sequences, sequence_targets)
                else:
                    model.fit(matrix, targets)
                fit_cutoff = timestamp
                refit_id += 1
                scoring_since_refit = 0
                fit_events.append(
                    {
                        "refit_id": refit_id,
                        "fit_cutoff_timestamp": timestamp,
                        "train_row_count": len(targets),
                        "feature_count": len(used_columns),
                        "model": model_name,
                    }
                )
                feature_columns = used_columns
        if model is None or fit_cutoff is None:
            continue
        score_rows = by_timestamp[timestamp]
        if _is_sequence_model(model_name):
            matrix, valid_rows = _sequence_score_matrix(
                score_rows,
                targeted_rows,
                feature_columns,
                sequence_length=sequence_length,
            )
        else:
            matrix, valid_rows = _score_matrix(score_rows, feature_columns)
        if not valid_rows:
            continue
        values = model.predict(matrix)
        ranks = _ranks_descending(values)
        execution_timestamp = _next_timestamp(timestamps, timestamp)
        for row, prediction, rank in zip(valid_rows, values, ranks):
            predictions.append(
                {
                    "timeframe": row["timeframe"],
                    "timestamp": row["timestamp"],
                    "symbol": row["symbol"],
                    "model": model_name,
                    "prediction": float(prediction),
                    "cross_sectional_rank": rank,
                    "target_horizon_bars": horizon_bars,
                    "target_value": row.get(target_column),
                    "label_maturity_timestamp": row.get(maturity_column),
                    "fit_cutoff_timestamp": fit_cutoff,
                    "refit_id": refit_id,
                    "feature_timestamp": row["timestamp"],
                    "intended_execution_timestamp": execution_timestamp,
                    "open": row["open"],
                    "close": row["close"],
                }
            )
        scoring_since_refit += 1
    duplicate_count = _prediction_duplicate_count(predictions)
    if duplicate_count:
        raise ValueError(f"duplicate prediction keys detected: {duplicate_count}")
    return predictions, {
        "prediction_count": len(predictions),
        "unique_prediction_key_count": len(predictions),
        "duplicate_prediction_key_count": 0,
        "fit_count": len(fit_events),
        "fit_events": fit_events,
        "target_metadata": target_metadata,
        "feature_columns": list(feature_columns),
        "refit_frequency_bars": refit_frequency_bars,
        "min_train_rows": min_train_rows,
        "score_cadence": "every_completed_eligible_bar",
        "model_frozen_between_refits": True,
        "model_registry_family": "sequence_regressor" if _is_sequence_model(model_name) else "tabular_regressor",
    }


def _build_model_predictions(
    model_name: str,
    feature_slice: FeatureBankSlice,
    horizon_bars: int,
    refit_frequency_bars: int,
    min_train_rows: int,
    sklearn_n_jobs: int,
    torch_num_threads: int,
    sequence_length: int,
    sequence_epochs: int,
    sequence_batch_size: int,
    random_seed: int,
    allow_cross_session_horizon: bool,
    expected_bar_seconds: int | None,
    allow_missing_intermediate_bars: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_every_bar_oos_predictions(
        feature_slice,
        horizon_bars=horizon_bars,
        model_name=model_name,
        refit_frequency_bars=refit_frequency_bars,
        min_train_rows=min_train_rows,
        sklearn_n_jobs=sklearn_n_jobs,
        torch_num_threads=torch_num_threads,
        sequence_length=sequence_length,
        sequence_epochs=sequence_epochs,
        sequence_batch_size=sequence_batch_size,
        random_seed=random_seed,
        allow_cross_session_horizon=allow_cross_session_horizon,
        expected_bar_seconds=expected_bar_seconds,
        allow_missing_intermediate_bars=allow_missing_intermediate_bars,
    )


def build_replay_grid_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
    scenarios: Sequence[Mapping[str, Any]],
    max_workers: int = 1,
) -> dict[str, Any]:
    if not scenarios:
        return {
            "enabled": False,
            "scenario_count": 0,
            "results": [],
            "parallelism": {
                "requested_workers": max(1, int(max_workers)),
                "effective_workers": 1,
                "work_unit": "replay_grid_scenario",
                "full_dataset_copy_per_worker": False,
            },
        }

    def run(scenario: Mapping[str, Any]) -> dict[str, Any]:
        replay = build_bar_cadence_portfolio_replay(
            predictions,
            signal_column="prediction",
            timeframe=timeframe,
            top_n=int(scenario.get("top_n", 25)),
            max_position_weight=float(scenario.get("max_position_weight", 0.05)),
            min_position_weight=float(scenario.get("min_position_weight", 0.0)),
            cost_bps=float(scenario.get("cost_bps", 10)),
            slippage_bps=float(scenario.get("slippage_bps", 5)),
            decision_frequency_bars=int(scenario.get("decision_frequency_bars", 1)),
            max_workers=1,
        )
        return {
            "scenario": dict(scenario),
            "summary": replay.summary,
        }

    requested = max(1, int(max_workers))
    effective = min(requested, len(scenarios))
    if effective > 1:
        with ThreadPoolExecutor(max_workers=effective) as executor:
            results = list(executor.map(run, scenarios))
    else:
        results = [run(scenario) for scenario in scenarios]
    return {
        "enabled": True,
        "scenario_count": len(results),
        "results": results,
        "parallelism": {
            "requested_workers": requested,
            "effective_workers": effective,
            "work_unit": "replay_grid_scenario",
            "full_dataset_copy_per_worker": False,
        },
    }


def _resolve_models(ml: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    explicit = ml.get("stock_bar_cadence_models")
    if explicit:
        names = tuple(str(name).strip().lower() for name in explicit if str(name).strip())
        return names, {
            "requested_model_set": "explicit",
            "effective_model_set": "explicit",
            "included_models": list(names),
            "excluded_models": [],
        }
    if ml.get("stock_bar_cadence_model_set"):
        model_set = resolve_stock_alpha_model_set(
            str(ml["stock_bar_cadence_model_set"]),
            include_sequence_models=bool(ml.get("stock_bar_cadence_include_sequence_models", False)),
        )
        return tuple(model_set.included_models), model_set.metadata()
    name = str(ml.get("stock_bar_cadence_model", "ridge")).strip().lower()
    return (name,), {
        "requested_model_set": "single",
        "effective_model_set": "single",
        "included_models": [name],
        "excluded_models": [],
    }


def _build_replay_for_model(
    predictions: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    timeframe: str,
    ml: Mapping[str, Any],
) -> BarCadenceReplayResult:
    return build_bar_cadence_portfolio_replay(
        [row for row in predictions if row["model"] == model_name],
        signal_column="prediction",
        timeframe=timeframe,
        top_n=int(ml.get("stock_bar_cadence_replay_top_n", 25)),
        max_position_weight=float(ml.get("stock_bar_cadence_replay_max_position_weight", 0.05)),
        min_position_weight=float(ml.get("stock_bar_cadence_replay_min_position_weight", 0.0)),
        cost_bps=float(ml.get("stock_bar_cadence_replay_cost_bps", 10)),
        slippage_bps=float(ml.get("stock_bar_cadence_replay_slippage_bps", 5)),
        decision_frequency_bars=int(ml.get("stock_bar_cadence_replay_decision_frequency_bars", 1)),
        retraining_cadence=f"every_{int(ml.get('stock_bar_cadence_refit_frequency_bars', 20))}_eligible_scoring_bars",
        max_workers=int(ml.get("stock_bar_cadence_replay_n_jobs", 1)),
    )


def _replay_grid_scenarios(ml: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if not ml.get("stock_bar_cadence_replay_grid_enabled", False):
        return ()
    top_n_values = _as_list(ml.get("stock_bar_cadence_replay_grid_top_n", [ml.get("stock_bar_cadence_replay_top_n", 25)]))
    cost_values = _as_list(ml.get("stock_bar_cadence_replay_grid_cost_bps", [ml.get("stock_bar_cadence_replay_cost_bps", 10)]))
    slippage_values = _as_list(ml.get("stock_bar_cadence_replay_grid_slippage_bps", [ml.get("stock_bar_cadence_replay_slippage_bps", 5)]))
    scenarios = []
    for top_n in top_n_values:
        for cost_bps in cost_values:
            for slippage_bps in slippage_values:
                scenarios.append(
                    {
                        "top_n": int(top_n),
                        "cost_bps": float(cost_bps),
                        "slippage_bps": float(slippage_bps),
                        "max_position_weight": float(ml.get("stock_bar_cadence_replay_max_position_weight", 0.05)),
                        "min_position_weight": float(ml.get("stock_bar_cadence_replay_min_position_weight", 0.0)),
                        "decision_frequency_bars": int(ml.get("stock_bar_cadence_replay_decision_frequency_bars", 1)),
                        "weight_policy": "equal_weight_capped",
                    }
                )
    return tuple(scenarios)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _expected_bar_seconds(timeframe: str, ml: Mapping[str, Any]) -> int | None:
    if ml.get("stock_bar_cadence_expected_bar_seconds") is not None:
        return int(ml["stock_bar_cadence_expected_bar_seconds"])
    if timeframe == "5m":
        return 300
    if timeframe == "1h":
        return 3600
    return None


def _coalesce_bool(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _new_model(
    model_name: str,
    *,
    random_seed: int,
    sklearn_n_jobs: int,
    torch_num_threads: int,
    sequence_length: int,
    sequence_epochs: int,
    sequence_batch_size: int,
) -> Any:
    if model_name in TABULAR_MODELS:
        registry = stock_ranker_model_registry(
            random_seed=random_seed,
            sklearn_n_jobs=sklearn_n_jobs,
        )
        return registry.get(model_name)()
    if _is_sequence_model(model_name):
        factories = _sequence_model_factories(
            sequence_length=sequence_length,
            epochs=sequence_epochs,
            batch_size=sequence_batch_size,
            random_seed=random_seed,
            device="cpu",
            torch_num_threads=torch_num_threads,
        )
        return factories[model_name]()
    raise ValueError(f"Unsupported stock bar-cadence model: {model_name}")


def _is_sequence_model(model_name: str) -> bool:
    return model_name in FULL_SEQUENCE_MODELS


def _usable_feature_columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> tuple[str, ...]:
    blocked = {"open", "high", "low", "close", "volume"}
    usable = []
    for column in columns:
        if column in blocked:
            continue
        if any(finite_number(row.get(column)) is not None for row in rows):
            usable.append(column)
    return tuple(usable)


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    used = tuple(
        column
        for column in columns
        if any(finite_number(row.get(column)) is not None for row in rows)
    )
    x_rows = []
    y = []
    for row in rows:
        target = finite_number(row.get(target_column))
        values = [finite_number(row.get(column)) for column in used]
        if target is None or any(value is None for value in values):
            continue
        x_rows.append([float(value) for value in values if value is not None])
        y.append(target)
    return np.asarray(x_rows, dtype=float), np.asarray(y, dtype=float), used


def _score_matrix(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x_rows = []
    valid = []
    for row in rows:
        values = [finite_number(row.get(column)) for column in columns]
        if any(value is None for value in values):
            continue
        x_rows.append([float(value) for value in values if value is not None])
        valid.append(dict(row))
    return np.asarray(x_rows, dtype=float), valid


def _sequence_training_matrix(
    train_rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    target_column: str,
    sequence_length: int,
) -> tuple[list[list[list[float]]], list[float], tuple[str, ...]]:
    by_key = {
        (row["symbol"], row["timestamp"]): row
        for row in train_rows
    }
    sequences = []
    targets = []
    for row in train_rows:
        target = finite_number(row.get(target_column))
        sequence = _row_sequence(row, all_rows, columns, sequence_length=sequence_length)
        if target is None or sequence is None:
            continue
        if (row["symbol"], row["timestamp"]) not in by_key:
            continue
        sequences.append(sequence)
        targets.append(float(target))
    return sequences, targets, tuple(columns)


def _sequence_score_matrix(
    rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    sequence_length: int,
) -> tuple[list[list[list[float]]], list[dict[str, Any]]]:
    sequences = []
    valid = []
    for row in rows:
        sequence = _row_sequence(row, all_rows, columns, sequence_length=sequence_length)
        if sequence is None:
            continue
        sequences.append(sequence)
        valid.append(dict(row))
    return sequences, valid


def _row_sequence(
    row: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    sequence_length: int,
) -> list[list[float]] | None:
    symbol = row["symbol"]
    timestamp = row["timestamp"]
    history = [
        candidate
        for candidate in all_rows
        if candidate["symbol"] == symbol and candidate["timestamp"] <= timestamp
    ]
    history.sort(key=lambda item: item["timestamp"])
    if len(history) < sequence_length:
        return None
    window = history[-sequence_length:]
    sequence = []
    for item in window:
        values = [finite_number(item.get(column)) for column in columns]
        if any(value is None for value in values):
            return None
        sequence.append([float(value) for value in values if value is not None])
    return sequence


def _ranks_descending(values: Sequence[float]) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: (-float(item[1]), item[0]))
    ranks = [0] * len(values)
    for rank, (index, _) in enumerate(indexed, start=1):
        ranks[index] = rank
    return ranks


def _prediction_duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [
        (row["timeframe"], row["model"], row["timestamp"], row["symbol"])
        for row in rows
    ]
    return len(keys) - len(set(keys))


def _next_timestamp(timestamps: Sequence[datetime], timestamp: datetime) -> datetime | None:
    for candidate in timestamps:
        if candidate > timestamp:
            return candidate
    return None


def _prediction_fieldnames() -> tuple[str, ...]:
    return (
        "timeframe",
        "timestamp",
        "symbol",
        "model",
        "prediction",
        "cross_sectional_rank",
        "target_horizon_bars",
        "target_value",
        "label_maturity_timestamp",
        "fit_cutoff_timestamp",
        "refit_id",
        "feature_timestamp",
        "intended_execution_timestamp",
        "open",
        "close",
    )


def _serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    return value


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["replay_summary"]
    prediction = payload["prediction_report"]
    return "\n".join(
        [
            "# Stock Bar-Cadence Research Experiment",
            "",
            "Research only. Trading impact: none. Production validated: false.",
            "",
            f"- Timeframe: `{payload['timeframe']}`",
            f"- Model: `{payload['model']}`",
            f"- Prediction rows: {prediction['prediction_count']}",
            f"- Fit count: {prediction['fit_count']}",
            f"- Total return: {summary['total_return']}",
            f"- Net return: {summary['net_return']}",
            f"- Cost drag: {summary['transaction_cost_drag']}",
            f"- Max drawdown: {summary['max_drawdown']}",
            f"- Average turnover: {summary['average_turnover']}",
            "",
        ]
    )
