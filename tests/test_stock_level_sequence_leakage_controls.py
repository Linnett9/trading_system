import pytest

from core.research.ml.stock_level_benchmark_execution import _build_sequences
from core.research.ml.stock_level.stock_level_sequence_regressors import (
    SequenceRegressorConfig,
    TorchSequenceReturnRegressor,
)


def test_sequence_windows_are_symbol_isolated_and_chronological():
    rows = [
        _row("2025-01-01", "AAA", 1.0),
        _row("2025-01-02", "AAA", 2.0),
        _row("2025-01-03", "AAA", 3.0),
        _row("2025-01-01", "BBB", 10.0),
        _row("2025-01-02", "BBB", 20.0),
        _row("2025-01-03", "BBB", 30.0),
    ]

    sequences, prediction_rows = _build_sequences(
        rows,
        [rows[2], rows[5]],
        ("feature",),
        sequence_length=3,
    )

    assert prediction_rows == [rows[2], rows[5]]
    assert sequences == [
        [[1.0], [2.0], [3.0]],
        [[10.0], [20.0], [30.0]],
    ]


def test_sequence_windows_do_not_use_future_bars():
    rows = [
        _row("2025-01-01", "AAA", 1.0),
        _row("2025-01-02", "AAA", 2.0),
        _row("2025-01-03", "AAA", 3.0),
        _row("2025-01-04", "AAA", 999.0),
    ]

    sequences, prediction_rows = _build_sequences(
        rows,
        [rows[2]],
        ("feature",),
        sequence_length=2,
    )

    assert prediction_rows == [rows[2]]
    assert sequences == [[[2.0], [3.0]]]


def test_sequence_regressor_normalization_is_fit_on_training_data_only():
    pytest.importorskip("torch")
    model = TorchSequenceReturnRegressor(
        SequenceRegressorConfig(
            architecture="dlinear",
            sequence_length=2,
            epochs=1,
            batch_size=2,
            torch_num_threads=1,
        )
    )
    model.fit(
        sequences=[
            [[1.0], [2.0]],
            [[2.0], [3.0]],
        ],
        targets=[0.1, 0.2],
    )
    train_mean = float(model.feature_means.reshape(-1)[0])

    model.predict([[[1000.0], [2000.0]]])

    assert float(model.feature_means.reshape(-1)[0]) == train_mean


def _row(date, symbol, feature):
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "feature": feature,
        "actual_forward_return_10d": 0.0,
    }
