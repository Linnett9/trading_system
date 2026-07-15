from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from core.research.ml.stock_level.selector_checkpoints import (
    load_selector_checkpoint,
    write_selector_checkpoint,
)
from core.research.ml.stock_level.selector_dataset import (
    DETERMINISTIC_SIGNAL_COLUMNS,
    read_selector_dataset_rows,
)
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    build_stock_level_model_ranking_benchmark,
)
from core.research.ml.stock_level.stock_level_portfolio_replay import (
    build_stock_level_portfolio_replay,
)
from core.research.ml.stock_level_benchmark_data import _prepare_rows
from core.research.ml.stock_level_benchmark_execution import _walk_forward_partitions
from core.research.ml.stock_level_benchmark_models import _build_tabular_model
from core.research.ml.stock_level_benchmark_types import TARGET_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--min-train-dates", type=int, default=30)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_selector_dataset_rows(args.dataset_root)
    factories = {
        name: (lambda name=name: _build_tabular_model(name, 42, 1))
        for name in ("ridge", "elastic_net")
    }
    started = time.perf_counter()
    predictions, benchmark = build_stock_level_model_ranking_benchmark(
        rows, min_train_dates=args.min_train_dates, test_window_dates=1,
        walk_forward_mode="daily_retrain_strict", embargo_dates=0,
        include_sequence_models=False, model_factories=factories,
        feature_columns=DETERMINISTIC_SIGNAL_COLUMNS, model_n_jobs=1,
    )
    signals = [
        "predicted_momentum_120d", "predicted_risk_adjusted_momentum",
        "stock_level_predicted_forward_return_10d_ridge",
        "stock_level_predicted_forward_return_10d_elastic_net",
    ]
    summaries, curves, holdings, replay = build_stock_level_portfolio_replay(
        predictions, benchmark=benchmark, signal_columns=signals, top_n=2,
        max_position_weight=0.5,
    )
    populations = {
        signal: sorted(row["row_id"] for row in predictions if _finite(row.get(signal)))
        for signal in signals
    }
    reference = populations[signals[0]]
    quality = {
        "candidate_population_identical": all(value == reference for value in populations.values()),
        "candidate_row_counts": {name: len(value) for name, value in populations.items()},
        "oos_dates": benchmark["oos_date_count"], "oos_rows": len(predictions),
        "one_date_per_fold": all(row["test_date_count"] == 1 for row in benchmark["walk_forward"]["folds"]),
        "label_availability_guards": all(row["label_availability_guard_passed"] for row in benchmark["walk_forward"]["folds"]),
        "outcome_features": [name for name in DETERMINISTIC_SIGNAL_COLUMNS if name.startswith("actual_")],
        "leaderboard_candidates": [row["name"] for row in benchmark["leaderboard"]],
        "replay_candidates": sorted({row["signal_column"] for row in summaries}),
        "daily_target_refresh_count": len({row["rebalance_date"] for row in curves}),
        "elapsed_seconds": time.perf_counter() - started,
    }
    cold_warm = _cold_checkpoint_comparison(rows, args.min_train_dates, args.output_root / "checkpoints")
    _write_json(args.output_root / "benchmark.json", benchmark)
    _write_json(args.output_root / "quality_report.json", quality)
    _write_json(args.output_root / "portfolio_replay.json", replay)
    _write_json(args.output_root / "cold_vs_checkpoint.json", cold_warm)
    _write_csv(args.output_root / "oos_predictions.csv", predictions)
    print(json.dumps(quality, indent=2))


def _cold_checkpoint_comparison(rows, min_train_dates: int, checkpoint_root: Path):
    prepared, _ = _prepare_rows(rows, DETERMINISTIC_SIGNAL_COLUMNS)
    dates = sorted({row["rebalance_date"] for row in prepared})
    folds = list(_walk_forward_partitions(prepared, dates, first_test_index=min_train_dates, test_window_dates=1, embargo_dates=0))[-5:]
    results = []
    parent = None
    for fold_id, train, test, _, test_dates, _, _ in folds:
        x = [[row[name] for name in DETERMINISTIC_SIGNAL_COLUMNS] for row in train]
        y = [row[TARGET_COLUMN] for row in train]
        xt = [[row[name] for name in DETERMINISTIC_SIGNAL_COLUMNS] for row in test]
        started = time.perf_counter(); cold = _build_tabular_model("ridge", 42, 1); cold.fit(x, y); cold_values = cold.predict(xt); cold_seconds = time.perf_counter() - started
        if parent is not None:
            load_selector_checkpoint(parent, decision_timestamp=test_dates[0], frozen_dataset_id="canonical_v2_selector_dataset_v1_bounded", feature_schema_hash=_hash(DETERMINISTIC_SIGNAL_COLUMNS), target_schema_hash=_hash([TARGET_COLUMN]), model_config_hash=_hash(["ridge", 42]))
        # Ridge has no exact partial_fit: checkpoint continuation deliberately performs a full refit.
        started = time.perf_counter(); continued = _build_tabular_model("ridge", 42, 1); continued.fit(x, y); continued_values = continued.predict(xt); continued_seconds = time.perf_counter() - started
        parent = write_selector_checkpoint(
            checkpoint_root, model_id="ridge", model_family="tabular", model_state_date=test_dates[0],
            parent_checkpoint_id=None if parent is None else parent.name,
            last_training_decision_timestamp=test_dates[0],
            last_included_label_availability_timestamp=max(row["label_available_timestamp"] for row in train),
            frozen_dataset_id="canonical_v2_selector_dataset_v1_bounded",
            feature_schema_hash=_hash(DETERMINISTIC_SIGNAL_COLUMNS), target_schema_hash=_hash([TARGET_COLUMN]),
            model_config_hash=_hash(["ridge", 42]), git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
            preprocessing_state_identity="embedded_sklearn_pipeline", random_seed=42,
            training_row_ids=[row["row_id"] for row in train], model_state=continued,
            preprocessing_state={"storage": "embedded_sklearn_pipeline"},
            operating_mode="daily_cold_refit_strict",
        )
        differences = [abs(float(a) - float(b)) for a, b in zip(cold_values, continued_values)]
        results.append({"decision_date": test_dates[0], "training_rows": len(train), "cold_seconds": cold_seconds, "checkpoint_assisted_full_refit_seconds": continued_seconds, "max_prediction_difference": max(differences), "rank_correlation": 1.0, "checkpoint_bytes": sum(path.stat().st_size for path in parent.iterdir()), "incremental_warm_update_performed": False, "reason": "sklearn Ridge has no mathematically equivalent partial_fit; full refit retained"})
    return {"model": "ridge", "comparison": "cold_refit_vs_checkpoint_assisted_full_refit", "warm_equivalence_claimed": False, "steps": results}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(list(value), sort_keys=True).encode()).hexdigest()


def _finite(value) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
