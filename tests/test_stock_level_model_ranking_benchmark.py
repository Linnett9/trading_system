from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor

from core.research.ml.stock_level import stock_level_model_ranking_benchmark
from core.research.ml.stock_level import stock_level_sequence_regressors
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    MODEL_NAMES,
    PREDICTION_PREFIX,
    SEQUENCE_MODEL_NAMES,
    build_stock_level_model_ranking_benchmark,
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


def test_walk_forward_never_trains_on_future_dates():
    _, payload = _run_benchmark()

    folds = payload["walk_forward"]["folds"]

    assert folds
    assert payload["walk_forward"]["all_chronological_guards_passed"] is True
    assert payload["parallelism"]["stock_ranker_model_n_jobs"] == 1
    assert all(fold["train_end_date"] < fold["test_start_date"] for fold in folds)
    assert all(
        all(fold["train_end_date"] < date for date in fold["embargoed_dates"])
        for fold in folds
    )


def test_predictions_are_one_row_per_symbol_and_date():
    predictions, payload = _run_benchmark()

    keys = [(row["rebalance_date"], row["symbol"]) for row in predictions]

    assert len(keys) == len(set(keys))
    assert len(predictions) == payload["oos_row_count"]
    assert all(f"{PREDICTION_PREFIX}ridge" in row for row in predictions)
    assert all(f"{PREDICTION_PREFIX}dlinear" in row for row in predictions)


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
        lambda payload: "",
    )

    stock_level_model_ranking_benchmark.write_stock_level_model_ranking_benchmark(
        {
            "ml": {
                "output_dir": str(tmp_path / "output"),
                "stock_level_prediction_artifacts_path": str(source_path),
                "stock_ranker_model_n_jobs": 3,
            }
        }
    )

    assert captured["model_n_jobs"] == 3


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
        for symbol_index in range(10):
            momentum = float(symbol_index - 5) / 10.0 + date_index * 0.001
            rows.append(
                {
                    "rebalance_date": date,
                    "symbol": f"S{symbol_index:02d}",
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
                    "breadth_above_sma_200": "0.5",
                    "spy_realized_volatility_21d": "0.1",
                    "spy_realized_volatility_63d": "0.2",
                    "spy_max_drawdown_63d": "-0.05",
                    "spy_max_drawdown_126d": "-0.1",
                }
            )
    return rows
