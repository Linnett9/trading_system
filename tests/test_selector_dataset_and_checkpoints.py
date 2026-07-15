from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.selector_checkpoints import (
    load_selector_checkpoint,
    newly_matured_rows,
    write_selector_checkpoint,
)
from core.research.ml.stock_level.selector_dataset import (
    BASELINE_CONTRACT_VERSION,
    DETERMINISTIC_SIGNAL_COLUMNS,
    deterministic_baseline_scores,
)


def test_missing_input_disposition_is_exact_and_deterministic():
    assert DETERMINISTIC_SIGNAL_COLUMNS == (
        "predicted_momentum_20d", "predicted_momentum_60d",
        "predicted_momentum_120d", "predicted_volatility_20d",
        "predicted_drawdown_60d", "predicted_liquidity_score",
        "predicted_risk_adjusted_momentum",
    )


def test_baseline_scores_use_only_observations_before_decision():
    dates = [f"2024-01-{day:02d}" for day in range(1, 32)] + [f"2024-02-{day:02d}" for day in range(1, 30)] + [f"2024-03-{day:02d}" for day in range(1, 32)] + [f"2024-04-{day:02d}" for day in range(1, 31)] + ["2024-05-01"]
    closes = [100.0 + index for index in range(len(dates))]
    base = deterministic_baseline_scores(
        asset_id="asset-a", decision_timestamp="2024-05-01T20:05:00Z",
        decision_date="2024-05-01", close_dates=dates, close_values=closes,
        dollar_volume_dates=dates, dollar_volume_values=[1000.0] * len(dates),
    )
    changed_future = deterministic_baseline_scores(
        asset_id="asset-a", decision_timestamp="2024-05-01T20:05:00Z",
        decision_date="2024-05-01", close_dates=dates + ["2024-05-02"],
        close_values=closes + [999999.0], dollar_volume_dates=dates + ["2024-05-02"],
        dollar_volume_values=[1000.0] * len(dates) + [999999.0],
    )
    assert base == changed_future
    assert base["baseline_contract_version"] == BASELINE_CONTRACT_VERSION
    assert base["predicted_momentum_120d"] == pytest.approx(closes[-2] / closes[-122] - 1.0)
    assert base["predicted_risk_adjusted_momentum"] is not None


def _checkpoint(tmp_path: Path, **overrides):
    kwargs = dict(
        model_id="ridge", model_family="tabular", model_state_date="2026-01-05",
        parent_checkpoint_id=None, last_training_decision_timestamp="2026-01-05",
        last_included_label_availability_timestamp="2026-01-05",
        frozen_dataset_id="dataset-a", feature_schema_hash="features-a",
        target_schema_hash="target-a", model_config_hash="config-a", git_commit="abc",
        preprocessing_state_identity="prep-a", random_seed=42,
        training_row_ids=["r2", "r1"], model_state={"coef": [1.0]},
        preprocessing_state={"mean": [0.0]}, optimizer_state={"step": 1},
        scheduler_state={"epoch": 1}, rng_state={"seed": 42},
        operating_mode="daily_checkpoint_update",
    )
    kwargs.update(overrides)
    return write_selector_checkpoint(tmp_path, **kwargs)


def test_checkpoint_round_trip_restores_all_state(tmp_path: Path):
    path = _checkpoint(tmp_path)
    loaded = load_selector_checkpoint(
        path, decision_timestamp="2026-01-06", frozen_dataset_id="dataset-a",
        feature_schema_hash="features-a", target_schema_hash="target-a",
        model_config_hash="config-a", require_optimizer_state=True,
        require_scheduler_state=True,
    )
    assert loaded.model_state == {"coef": [1.0]}
    assert loaded.preprocessing_state == {"mean": [0.0]}
    assert loaded.optimizer_state == {"step": 1}
    required = {"model_id", "model_family", "model_state_date", "parent_checkpoint_id", "last_training_decision_timestamp", "last_included_label_availability_timestamp", "frozen_dataset_id", "feature_schema_hash", "target_schema_hash", "model_config_hash", "git_commit", "preprocessing_state_identity", "random_seed", "training_row_count", "training_row_id_checksum", "checkpoint_checksum", "completion_status"}
    assert required <= loaded.manifest.keys()


@pytest.mark.parametrize("field,value", [
    ("frozen_dataset_id", "dataset-b"), ("feature_schema_hash", "features-b"),
    ("target_schema_hash", "target-b"), ("model_config_hash", "config-b"),
])
def test_checkpoint_identity_mismatch_fails_closed(tmp_path: Path, field: str, value: str):
    path = _checkpoint(tmp_path)
    kwargs = dict(decision_timestamp="2026-01-06", frozen_dataset_id="dataset-a", feature_schema_hash="features-a", target_schema_hash="target-a", model_config_hash="config-a")
    kwargs[field] = value
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_selector_checkpoint(path, **kwargs)


def test_future_and_incomplete_checkpoints_are_rejected(tmp_path: Path):
    future = _checkpoint(tmp_path / "future")
    with pytest.raises(RuntimeError, match="future-dated"):
        load_selector_checkpoint(future, decision_timestamp="2026-01-04", frozen_dataset_id="dataset-a", feature_schema_hash="features-a", target_schema_hash="target-a", model_config_hash="config-a")
    incomplete = _checkpoint(tmp_path / "incomplete", completion_status="interrupted")
    with pytest.raises(RuntimeError, match="incomplete"):
        load_selector_checkpoint(incomplete, decision_timestamp="2026-01-06", frozen_dataset_id="dataset-a", feature_schema_hash="features-a", target_schema_hash="target-a", model_config_hash="config-a")


def test_newly_matured_labels_are_strictly_chained():
    rows = [
        {"row_id": "a", "decision_timestamp": "2025-12-01", "label_available_timestamp": "2026-01-04"},
        {"row_id": "b", "decision_timestamp": "2025-12-02", "label_available_timestamp": "2026-01-05"},
        {"row_id": "c", "decision_timestamp": "2025-12-03", "label_available_timestamp": "2026-01-06"},
    ]
    assert [row["row_id"] for row in newly_matured_rows(rows, previous_label_timestamp="2026-01-04", current_decision_timestamp="2026-01-05")] == ["b"]
