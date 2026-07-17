from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor

from core.research.ml.stock_level import stock_level_model_ranking_benchmark
from core.research.ml.stock_level import stock_level_sequence_regressors
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    FEATURE_COLUMNS,
    MODEL_NAMES,
    PREDICTION_PREFIX,
    SEQUENCE_MODEL_NAMES,
    build_stock_level_model_ranking_benchmark,
)
from core.research.ml.stock_level_benchmark_reporting import _prediction_columns
from core.research.ml.stock_level_benchmark_data import _stable_selector_row_id
from core.research.ml.stock_level_benchmark_types import (
    TARGET_PROVENANCE_CONTRACT_VERSION,
)


class MomentumRegressor:
    def fit(self, features, targets):
        assert len(features) == len(targets)

    def predict(self, features):
        return [row[2] for row in features]


class SequenceMomentumRegressor:
    def fit(self, sequences, targets, auxiliary_targets=None):
        assert len(sequences) == len(targets)
        if auxiliary_targets is not None:
            assert len(auxiliary_targets) == len(targets)

    def predict(self, sequences):
        return [sequence[-1][2] for sequence in sequences]


class FailingRegressor:
    def fit(self, features, targets):
        raise RuntimeError("synthetic model failure")

    def predict(self, features):
        return []


def test_daily_retrain_strict_uses_one_date_per_fold_and_stable_row_ids():
    predictions, payload = build_stock_level_model_ranking_benchmark(
        _rows(), min_train_dates=3, embargo_dates=1, test_window_dates=1,
        walk_forward_mode="daily_retrain_strict", include_sequence_models=False,
        model_factories={"ridge": MomentumRegressor},
    )
    assert payload["walk_forward"]["mode"] == "daily_retrain_strict"
    assert all(fold["test_date_count"] == 1 for fold in payload["walk_forward"]["folds"])
    assert all(row["row_id"] == _stable_selector_row_id(row["asset_id"], row["decision_timestamp"]) for row in predictions)
    assert len({row["row_id"] for row in predictions}) == len(predictions)


def test_daily_retrain_strict_rejects_block_test_window():
    try:
        build_stock_level_model_ranking_benchmark(
            _rows(), min_train_dates=3, embargo_dates=1, test_window_dates=2,
            walk_forward_mode="daily_retrain_strict", include_sequence_models=False,
            model_factories={"ridge": MomentumRegressor},
        )
    except ValueError as exc:
        assert "one-date OOS" in str(exc)
    else:
        raise AssertionError("daily strict mode accepted a block OOS window")


def test_unintegrated_checkpoint_mode_fails_closed():
    try:
        build_stock_level_model_ranking_benchmark(
            _rows(), min_train_dates=3, embargo_dates=0, test_window_dates=1,
            walk_forward_mode="daily_retrain_strict",
            operating_mode="daily_checkpoint_update",
            include_sequence_models=False, model_factories={"ridge": MomentumRegressor},
        )
    except NotImplementedError as exc:
        assert "refusing to masquerade" in str(exc)
    else:
        raise AssertionError("unintegrated checkpoint mode ran as a cold refit")


def test_walk_forward_never_trains_on_future_dates():
    _, payload = _run_benchmark()

    folds = payload["walk_forward"]["folds"]

    assert folds
    assert payload["walk_forward"]["all_chronological_guards_passed"] is True
    assert payload["parallelism"]["stock_ranker_model_n_jobs"] == 1
    assert payload["model_timings"]
    assert all(fold["train_end_date"] < fold["test_start_date"] for fold in folds)
    assert all(
        all(fold["train_end_date"] < date for date in fold["embargoed_dates"])
        for fold in folds
    )
    assert all(fold["label_availability_guard_passed"] for fold in folds)


def test_walk_forward_uses_only_labels_matured_by_decision_date():
    rows = _rows()
    for row in rows:
        day = int(row["rebalance_date"][-2:])
        row["label_available_timestamp"] = f"2024-01-{day + 1:02d}"
    for row in rows:
        if row["rebalance_date"] == "2024-01-02":
            row["label_available_timestamp"] = "2024-01-05"

    _, payload = build_stock_level_model_ranking_benchmark(
        rows,
        min_train_dates=2,
        test_window_dates=1,
        embargo_dates=0,
        model_factories={"ridge": MomentumRegressor},
        include_sequence_models=False,
    )

    folds = payload["temporal_audit"]["folds"]
    decision_2024_01_03 = next(
        fold for fold in folds if fold["decision_timestamp"] == "2024-01-03"
    )
    decision_2024_01_05 = next(
        fold for fold in folds if fold["decision_timestamp"] == "2024-01-05"
    )
    assert decision_2024_01_03["training_label_available_max"] == "2024-01-02"
    assert decision_2024_01_03["purged_row_count"] == 10
    assert decision_2024_01_05["training_label_available_max"] == "2024-01-05"
    assert decision_2024_01_05["train_row_count"] > decision_2024_01_03["train_row_count"]


def test_selector_label_availability_fails_closed_without_outcome_end():
    rows = _rows()
    for row in rows:
        row.pop("outcome_end_date", None)
        row.pop("label_end_timestamp", None)
        row.pop("label_available_timestamp", None)

    try:
        build_stock_level_model_ranking_benchmark(
            rows,
            min_train_dates=2,
            test_window_dates=1,
            embargo_dates=0,
            model_factories={"ridge": MomentumRegressor},
            include_sequence_models=False,
        )
    except ValueError as exc:
        assert "rebalance_date cannot be used as label availability" in str(exc)
    else:
        raise AssertionError("missing label availability should fail closed")


def test_selector_rejects_rebalance_date_as_label_availability():
    rows = _rows()
    for row in rows:
        row["label_available_timestamp"] = row["rebalance_date"]

    try:
        build_stock_level_model_ranking_benchmark(
            rows,
            min_train_dates=2,
            test_window_dates=1,
            embargo_dates=0,
            model_factories={"ridge": MomentumRegressor},
            include_sequence_models=False,
        )
    except ValueError as exc:
        assert "cannot masquerade" in str(exc)
    else:
        raise AssertionError("rebalance_date label availability should fail closed")


def test_selector_rejects_pre_4d_rows_without_canonical_provenance():
    rows = _rows()
    for row in rows:
        row.pop("target_provenance_contract_version", None)

    try:
        build_stock_level_model_ranking_benchmark(
            rows,
            min_train_dates=2,
            test_window_dates=1,
            embargo_dates=0,
            model_factories={"ridge": MomentumRegressor},
            include_sequence_models=False,
        )
    except ValueError as exc:
        assert "pre-4D artifacts missing provenance are incompatible" in str(exc)
    else:
        raise AssertionError("pre-4D target artifacts should be incompatible")


def test_blank_optional_provenance_does_not_overwrite_required_timestamps():
    rows = _rows()
    for row in rows:
        row["benchmark_label_end_timestamp"] = ""
        row["benchmark_label_available_timestamp"] = "NaT"

    predictions, _ = build_stock_level_model_ranking_benchmark(
        rows,
        min_train_dates=4,
        test_window_dates=1,
        embargo_dates=0,
        model_factories={"ridge": MomentumRegressor},
        include_sequence_models=False,
    )

    assert predictions
    assert all(row["label_end_timestamp"] for row in predictions)
    assert all(row["label_available_timestamp"] for row in predictions)


def test_temporal_audit_artifact_is_written(tmp_path):
    source_path = tmp_path / "stock_level_prediction_artifacts.csv"
    rows = _rows()
    fieldnames = list(rows[0])
    source_path.write_text(
        ",".join(fieldnames)
        + "\n"
        + "\n".join(",".join(str(row.get(name, "")) for name in fieldnames) for row in rows)
        + "\n",
        encoding="utf-8",
    )

    paths = stock_level_model_ranking_benchmark.write_stock_level_model_ranking_benchmark(
        {
            "ml": {
                "output_dir": str(tmp_path / "output"),
                "stock_level_prediction_artifacts_path": str(source_path),
                "stock_ranker_min_train_dates": 2,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_include_sequence_models": False,
                "stock_ranker_model_set": "standard",
                "stock_level_allow_csv_artifact_fallback": True,
            }
        }
    )

    audit = json.loads(paths.temporal_audit_path.read_text(encoding="utf-8"))
    assert audit["workflow"] == "stock_selector_oos_benchmark"
    assert audit["target_provenance_contract_version"] == TARGET_PROVENANCE_CONTRACT_VERSION
    assert audit["leakage_checks_passed"] is True
    assert audit["folds"]


def test_predictions_are_one_row_per_symbol_and_date():
    predictions, payload = _run_benchmark()

    keys = [(row["rebalance_date"], row["symbol"]) for row in predictions]

    assert len(keys) == len(set(keys))
    assert len(predictions) == payload["oos_row_count"]
    assert all(f"{PREDICTION_PREFIX}ridge" in row for row in predictions)
    assert all(f"{PREDICTION_PREFIX}dlinear" in row for row in predictions)


def test_oos_predictions_preserve_engineered_target_columns():
    rows = _rows()
    for index, row in enumerate(rows):
        row.update({"actual_benchmark_return_10d": str(index / 1000), "actual_market_residual_return_10d": str(index / 100), "actual_vol_adjusted_forward_return_10d": str(index / 10), "actual_drawdown_adjusted_forward_return_10d": str(index / 200), "actual_rank_normalized_forward_return_10d": str((index % 10) / 9), "actual_top_decile_label_10d": str(int(index % 10 == 9))})
    predictions, _ = build_stock_level_model_ranking_benchmark(rows, min_train_dates=2, test_window_dates=2, embargo_dates=1, model_factories={"ridge": MomentumRegressor}, include_sequence_models=False)
    assert predictions
    assert all(row["actual_benchmark_return_10d"] is not None for row in predictions)
    assert all(row["actual_market_residual_return_10d"] is not None for row in predictions)
    assert all(row["actual_rank_normalized_forward_return_10d"] is not None for row in predictions)
    assert all(row["benchmark_symbol"] == "SPY" for row in predictions)
    assert all(
        row["target_provenance_contract_version"] == TARGET_PROVENANCE_CONTRACT_VERSION
        for row in predictions
    )
    assert all(row["label_available_timestamp"] for row in predictions)
    assert "benchmark_symbol" in _prediction_columns(["ridge"])
    assert "label_available_timestamp" in _prediction_columns(["ridge"])
    assert "actual_benchmark_return_10d" not in FEATURE_COLUMNS


def test_baseline_comparison_uses_the_same_oos_rows():
    _, payload = _run_benchmark()

    leaderboard = {row["name"]: row for row in payload["leaderboard"]}
    comparison = payload["best_ml_vs_momentum_120d"]

    assert leaderboard["momentum_120d"]["date_count"] == payload["oos_date_count"]
    assert leaderboard["ridge"]["row_count"] == leaderboard["momentum_120d"][
        "row_count"
    ]
    assert leaderboard["momentum_120d"]["mean_spearman_ic"] == 1.0
    assert comparison["momentum_baseline"] == "momentum_120d"
    assert comparison["beats_momentum_120d"] is False


def test_stock_level_alpha_suite_has_no_operational_imports():
    source = "\n".join(
        [
            inspect.getsource(stock_level_model_ranking_benchmark),
            inspect.getsource(stock_level_sequence_regressors),
        ]
    )

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source
    assert "publish_ordinary_selector_partitions" not in source


def test_benchmark_defer_selector_publication_to_component_pipeline(tmp_path):
    source_path = tmp_path / "stock_level_prediction_artifacts.csv"
    source_path.write_text("rebalance_date,symbol\n", encoding="utf-8")

    try:
        stock_level_model_ranking_benchmark.write_stock_level_model_ranking_benchmark(
            {
                "ml": {
                    "output_dir": str(tmp_path / "output"),
                    "stock_level_prediction_artifacts_path": str(source_path),
                    "stock_level_allow_csv_artifact_fallback": True,
                    "ordinary_selector_manifest_root": str(tmp_path / "ordinary"),
                }
            }
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("ordinary selector publication should be deferred")

    assert "selector component publication pipeline" in message
    assert "ml-selector-component-publish" in message
    assert not (tmp_path / "ordinary").exists()


def test_registry_covers_the_model_zoo_and_reports_missing_news_inputs():
    predictions, payload = build_stock_level_model_ranking_benchmark(
        _rows(),
        min_train_dates=2,
        test_window_dates=2,
        embargo_dates=1,
        sequence_length=2,
        model_factories={"ridge": MomentumRegressor},
        sequence_model_factories={
            name: SequenceMomentumRegressor for name in SEQUENCE_MODEL_NAMES
        },
    )

    assert set(payload["requested_models"]) == set(MODEL_NAMES)
    assert "news_analysis_transformer" not in payload["completed_models"]
    assert payload["unavailable_models"] == [
        {
            "name": "news_analysis_transformer",
            "status": "unavailable",
            "reason": (
                "The stock-level input contains no point-in-time symbol-level "
                "news or sentiment features; synthetic news inputs are forbidden."
            ),
        }
    ]
    assert all(
        f"{PREDICTION_PREFIX}{name}" in predictions[0]
        for name in payload["completed_models"]
    )


def test_news_transformer_requires_point_in_time_contract_when_news_columns_exist():
    rows = _rows()
    for row in rows:
        row["news_sentiment_score"] = "0.1"

    predictions, payload = build_stock_level_model_ranking_benchmark(
        rows,
        min_train_dates=2,
        test_window_dates=2,
        embargo_dates=1,
        sequence_length=2,
        model_factories={},
        sequence_model_factories={"news_analysis_transformer": SequenceMomentumRegressor},
    )

    assert predictions
    assert "news_analysis_transformer" not in payload["completed_models"]
    assert payload["unavailable_models"] == [
        {
            "name": "news_analysis_transformer",
            "status": "unavailable",
            "reason": "news_analysis_transformer unavailable: missing valid point-in-time news contract",
        }
    ]


def test_parallel_and_sequential_outputs_are_equivalent():
    common = {
        "rows": _rows(),
        "min_train_dates": 2,
        "test_window_dates": 2,
        "embargo_dates": 1,
        "sequence_length": 2,
        "model_factories": {
            "ridge": MomentumRegressor,
            "elastic_net": MomentumRegressor,
        },
        "sequence_model_factories": {"dlinear": SequenceMomentumRegressor},
    }
    sequential_predictions, sequential_payload = (
        build_stock_level_model_ranking_benchmark(**common, model_n_jobs=1)
    )
    parallel_predictions, parallel_payload = build_stock_level_model_ranking_benchmark(
        **common,
        model_n_jobs=3,
    )

    assert parallel_predictions == sequential_predictions
    assert parallel_payload["leaderboard"] == sequential_payload["leaderboard"]
    assert parallel_payload["completed_models"] == sequential_payload["completed_models"]
    assert parallel_payload["parallelism"]["effective_model_workers"] == 3
    assert parallel_payload["parallelism"][
        "effective_per_model_sklearn_n_jobs"
    ] == 1


def test_model_errors_are_captured_without_stopping_successful_models():
    predictions, payload = build_stock_level_model_ranking_benchmark(
        _rows(),
        min_train_dates=2,
        test_window_dates=2,
        embargo_dates=1,
        sequence_length=2,
        model_n_jobs=2,
        executor_cls=ThreadPoolExecutor,
        model_factories={
            "ridge": MomentumRegressor,
            "broken_model": FailingRegressor,
        },
        sequence_model_factories={},
    )

    assert payload["completed_models"] == ["ridge"]
    assert any(
        row["name"] == "broken_model"
        and row["status"] == "error"
        and "synthetic model failure" in row["reason"]
        for row in payload["unavailable_models"]
    )
    assert all(f"{PREDICTION_PREFIX}ridge" in row for row in predictions)
    assert all(
        f"{PREDICTION_PREFIX}broken_model" not in row for row in predictions
    )


def test_writer_respects_stock_ranker_model_n_jobs(tmp_path, monkeypatch):
    source_path = tmp_path / "stock_level_prediction_artifacts.csv"
    source_path.write_text("rebalance_date,symbol\n", encoding="utf-8")
    captured = {}

    def fake_build(rows, **kwargs):
        captured.update(kwargs)
        return [], {"leaderboard": [], "completed_models": []}

    monkeypatch.setattr(
        stock_level_model_ranking_benchmark,
        "build_stock_level_model_ranking_benchmark",
        fake_build,
    )
    monkeypatch.setattr(
        stock_level_model_ranking_benchmark,
        "_write_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        stock_level_model_ranking_benchmark,
        "_markdown",
        lambda payload: "# Benchmark\n",
    )

    paths = stock_level_model_ranking_benchmark.write_stock_level_model_ranking_benchmark(
        {
            "ml": {
                "output_dir": str(tmp_path / "output"),
                "stock_level_prediction_artifacts_path": str(source_path),
                "stock_ranker_model_n_jobs": 3,
                "stock_level_allow_csv_artifact_fallback": True,
            }
        }
    )

    assert captured["model_n_jobs"] == 3
    latest = json.loads((tmp_path / "output" / "latest_completed.json").read_text())
    run_dir = tmp_path / "output" / "runs" / latest["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["kind"] == "stock_selector_benchmark"
    assert manifest["run_status"] == "complete"
    assert latest["run_id"].startswith("stock_selector_benchmark-")
    assert (run_dir / paths.json_path.name).exists()
    assert not (tmp_path / "output" / "champion.json").exists()


def _run_benchmark():
    return build_stock_level_model_ranking_benchmark(
        _rows(),
        min_train_dates=2,
        test_window_dates=2,
        embargo_dates=1,
        sequence_length=2,
        model_factories={"ridge": MomentumRegressor},
        sequence_model_factories={"dlinear": SequenceMomentumRegressor},
    )


def _rows() -> list[dict[str, str]]:
    rows = []
    for date_index in range(8):
        date = f"2024-01-{date_index + 1:02d}"
        label_start = f"2024-01-{date_index + 2:02d}"
        label_end = f"2024-01-{date_index + 3:02d}"
        label_available = f"2024-01-{date_index + 4:02d}"
        for symbol_index in range(10):
            momentum = float(symbol_index - 5) / 10.0 + date_index * 0.001
            rows.append(
                {
                    "rebalance_date": date,
                    "symbol": f"S{symbol_index:02d}",
                    "benchmark_symbol": "SPY",
                    "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
                    "feature_timestamp": date,
                    "decision_timestamp": date,
                    "target_horizon": "10_trading_observations",
                    "target_observation_count": "10",
                    "target_start_timestamp": date,
                    "label_start_timestamp": label_start,
                    "label_end_timestamp": label_end,
                    "label_available_timestamp": label_available,
                    "target_price_convention": "simple_close_to_close",
                    "benchmark_target_start_timestamp": date,
                    "benchmark_label_start_timestamp": label_start,
                    "benchmark_label_end_timestamp": label_end,
                    "benchmark_label_available_timestamp": label_available,
                    "outcome_end_date": label_end,
                    "predicted_momentum_20d": str(momentum * 0.25),
                    "predicted_momentum_60d": str(momentum * 0.5),
                    "predicted_momentum_120d": str(momentum),
                    "predicted_volatility_20d": "0.1",
                    "predicted_drawdown_60d": "-0.2",
                    "predicted_liquidity_score": str(10.0 + symbol_index),
                    "predicted_risk_adjusted_momentum": str(momentum / 0.2),
                    "actual_forward_return_5d": str(momentum * 0.5),
                    "actual_forward_return_10d": str(momentum),
                    "actual_future_volatility": "0.1",
                    "actual_future_drawdown": "-0.2",
                    "actual_benchmark_return_10d": str(0.01 + date_index * 0.001),
                    "breadth_above_sma_200": "0.5",
                    "spy_realized_volatility_21d": "0.1",
                    "spy_realized_volatility_63d": "0.2",
                    "spy_max_drawdown_63d": "-0.05",
                    "spy_max_drawdown_126d": "-0.1",
                }
            )
    return rows
