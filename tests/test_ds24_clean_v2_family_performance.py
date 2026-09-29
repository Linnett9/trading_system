from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from core.research.ml.ds24.clean_v2_data import DataAssemblyProfiler
from core.research.ml.ds24.clean_v2_package_cache import (
    PackageCacheError,
    PanelPackageKey,
    SharedPanelPackageCache,
)
from core.research.ml import ds24_metrics_only_evaluator as evaluator
from core.research.ml.stock_level.stock_level_sequence_regressors import (
    SequenceRegressorConfig,
    TorchSequenceReturnRegressor,
)
from scripts.local.ds24_clean_v2_family_worker import (
    CleanV2WorkerError,
    _family_contract,
    _fit_model,
    _model_artifact_payload,
    _restore_model_artifact,
    _score_model,
    _scoring_timestamp_batches,
    _tabular_estimator,
)


TABULAR_FAMILIES = (
    "random_forest",
    "ridge_C5",
    "elastic_net_C5",
    "elastic_net_C6",
    "huber",
    "gradient_boosting_C0",
    "gradient_boosting_C0_W20",
    "gradient_boosting_C0_W40",
    "gradient_boosting_C0_W80",
)


def _cache_key(*, session_date: str = "2024-01-02") -> PanelPackageKey:
    return PanelPackageKey(
        feature_authority_hash="feature",
        target_authority_hash="target",
        static_authority_bundle_sha256="bundle",
        predictor_order_hash="predictors",
        assembly_source_hash="source",
        session_dates=(session_date,),
        maximum_assets=None,
    )


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(240803)
    rows = 80
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "target_value": rng.normal(size=rows),
        }
    )
    frame.loc[::11, "x2"] = np.nan
    return frame


def _score_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in pd.to_datetime(
        ["2024-01-02T14:35:00Z", "2024-01-02T14:40:00Z"], utc=True
    ):
        for asset_index in range(6):
            rows.append(
                {
                    "decision_timestamp": timestamp,
                    "asset_id": f"A{asset_index}",
                    "x1": float(asset_index),
                    "x2": np.nan if asset_index == 1 else float(asset_index * 2),
                    "x3": float(asset_index % 3),
                }
            )
    return pd.DataFrame(rows)


def test_data_assembly_profiler_accumulates_bounded_stage_evidence() -> None:
    profiler = DataAssemblyProfiler()
    with profiler.measure("feature_reads"):
        pass
    with profiler.measure("feature_reads"):
        pass
    profiler.record_rows("feature_reads", 7)
    profiler.record_rows("feature_reads", 5)

    payload = profiler.payload()

    assert payload["calls"] == {"feature_reads": 2}
    assert payload["output_rows"] == {"feature_reads": 12}
    assert payload["elapsed_seconds"]["feature_reads"] >= 0.0


def test_shared_panel_package_cache_reuses_exact_immutable_frame(
    tmp_path: Path,
) -> None:
    cache = SharedPanelPackageCache(tmp_path / "cache", maximum_bytes=16 * 1024**2)
    frame = pd.DataFrame(
        {
            "asset_id": ["A", "B"],
            "decision_timestamp": pd.to_datetime(
                ["2024-01-02T14:35:00Z", "2024-01-02T14:35:00Z"], utc=True
            ),
            "session_date": ["2024-01-02", "2024-01-02"],
            "predictor": [1.0, np.nan],
            "target_is_trainable": [True, False],
        }
    )
    calls = 0

    def build() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return frame.copy()

    first = cache.get_or_build(_cache_key(), build)
    second = cache.get_or_build(_cache_key(), build)

    assert calls == 1
    assert first.disposition == "MISS_PUBLISHED"
    assert second.disposition == "HIT"
    assert first.cache_key == second.cache_key
    assert_frame_equal(second.frame, frame, check_exact=True)


def test_shared_panel_package_cache_derives_exact_slice_from_smallest_superset(
    tmp_path: Path,
) -> None:
    cache = SharedPanelPackageCache(tmp_path / "cache", maximum_bytes=16 * 1024**2)
    superset_key = PanelPackageKey(
        feature_authority_hash="feature",
        target_authority_hash="target",
        static_authority_bundle_sha256="bundle",
        predictor_order_hash="predictors",
        assembly_source_hash="source",
        session_dates=("2024-01-02", "2024-01-03", "2024-01-04"),
        maximum_assets=None,
    )
    frame = pd.DataFrame(
        {
            "session_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "asset_id": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    cache.get_or_build(superset_key, lambda: frame.copy())
    builder_called = False

    def unexpected_builder() -> pd.DataFrame:
        nonlocal builder_called
        builder_called = True
        return frame.iloc[[1]].copy()

    result = cache.get_or_build(
        _cache_key(session_date="2024-01-03"), unexpected_builder
    )

    assert not builder_called
    assert result.disposition == "HIT_SUPERSET"
    assert result.cache_key == superset_key.digest
    assert_frame_equal(
        result.frame.reset_index(drop=True),
        frame.iloc[[1]].reset_index(drop=True),
        check_exact=True,
    )


def test_shared_panel_package_cache_fails_closed_on_byte_corruption(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache = SharedPanelPackageCache(cache_root, maximum_bytes=16 * 1024**2)
    key = _cache_key()
    cache.get_or_build(key, lambda: pd.DataFrame({"value": [1.0, 2.0]}))
    with (
        cache_root / f"{SharedPanelPackageCache._artifact_id(key.digest)}.parquet"
    ).open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(PackageCacheError, match="size mismatch"):
        cache.get_or_build(key, lambda: pd.DataFrame({"value": [3.0]}))


def test_transformer_portable_checkpoint_round_trip_preserves_predictions() -> None:
    rng = np.random.default_rng(63)
    sequences = rng.normal(size=(16, 4, 3)).tolist()
    targets = rng.normal(size=16).tolist()
    model = TorchSequenceReturnRegressor(
        SequenceRegressorConfig(
            architecture="transformer",
            sequence_length=4,
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0001,
            random_seed=63,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            device="cpu",
            torch_num_threads=1,
        )
    )
    model.fit(sequences, targets)
    expected = model.predict(sequences[:5])
    identity = {"family": "transformer", "refit": "bounded-test"}

    artifact = _model_artifact_payload(model=model, identity=identity)
    restored = _restore_model_artifact(
        artifact,
        expected_identity=identity,
    )

    assert artifact["model_serialization"] == model.CHECKPOINT_SCHEMA
    assert "model" not in artifact
    np.testing.assert_array_equal(restored.predict(sequences[:5]), expected)


def test_transformer_float32_array_input_matches_legacy_python_windows() -> None:
    rng = np.random.default_rng(6301)
    array_sequences = rng.normal(size=(12, 4, 3)).astype(np.float32)
    legacy_sequences = array_sequences.astype(float).tolist()
    targets = rng.normal(size=12).tolist()
    config = SequenceRegressorConfig(
        architecture="transformer",
        sequence_length=4,
        epochs=1,
        batch_size=4,
        random_seed=63,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        device="cpu",
        torch_num_threads=1,
    )
    legacy = TorchSequenceReturnRegressor(config)
    compact = TorchSequenceReturnRegressor(config)

    legacy.fit(legacy_sequences, targets)
    compact.fit(array_sequences, targets)

    np.testing.assert_array_equal(
        compact.predict(array_sequences),
        legacy.predict(legacy_sequences),
    )


def test_transformer_checkpoint_identity_mismatch_fails_closed() -> None:
    model = TorchSequenceReturnRegressor(
        SequenceRegressorConfig(architecture="transformer")
    )
    artifact = _model_artifact_payload(model=model, identity={"family": "transformer"})

    with pytest.raises(CleanV2WorkerError, match="identity mismatch"):
        _restore_model_artifact(
            artifact,
            expected_identity={"family": "ridge_C5"},
        )


@pytest.mark.parametrize("family", TABULAR_FAMILIES)
def test_contiguous_tabular_boundary_matches_legacy_dataframe_fit(
    family: str,
) -> None:
    predictors = ("x1", "x2", "x3")
    config, _lane = _family_contract(family)
    training = _training_frame()
    scoring = _score_frame()
    legacy = _tabular_estimator(family, config)
    legacy.fit(
        training.loc[:, list(predictors)],
        training["target_value"].astype(float),
    )
    candidate, _metadata = _fit_model(
        family, config, training, predictors
    )

    expected = legacy.predict(scoring.loc[:, list(predictors)])
    actual = candidate.predict(
        np.ascontiguousarray(
            scoring.loc[:, list(predictors)].to_numpy(dtype=np.float64)
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("family", TABULAR_FAMILIES)
def test_all_tabular_family_session_batches_match_timestamp_scoring(
    family: str,
) -> None:
    predictors = ("x1", "x2", "x3")
    config, _lane = _family_contract(family)
    model, metadata = _fit_model(family, config, _training_frame(), predictors)
    panel = _score_frame()

    sequential = pd.concat(
        [
            _score_model(
                family,
                model,
                rows.copy(),
                panel,
                predictors,
                metadata,
            )
            for _timestamp, rows in panel.groupby(
                "decision_timestamp", sort=False
            )
        ],
        ignore_index=True,
    )
    batched = _score_model(
        family,
        model,
        panel.copy(),
        panel,
        predictors,
        metadata,
    )

    assert_frame_equal(
        sequential.drop(columns="prediction").reset_index(drop=True),
        batched.drop(columns="prediction").reset_index(drop=True),
        check_exact=True,
    )
    np.testing.assert_allclose(
        sequential["prediction"].to_numpy(),
        batched["prediction"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    for predictions in (sequential, batched):
        predictions["rank"] = predictions.groupby("decision_timestamp")[
            "prediction"
        ].rank(method="first", ascending=False)
    assert sequential["rank"].tolist() == batched["rank"].tolist()


@pytest.mark.parametrize("family", [*TABULAR_FAMILIES, "momentum", "equal_weight_no_model"])
def test_non_sequence_family_uses_one_bounded_session_batch(family: str) -> None:
    timestamps = list(
        pd.date_range("2024-01-02T14:35:00Z", periods=12, freq="5min", tz="UTC")
    )

    assert _scoring_timestamp_batches(family, timestamps, set()) == [
        tuple(timestamps)
    ]


@pytest.mark.parametrize(
    "family",
    [
        "ridge_C5",
        "elastic_net_C5",
        "huber",
        "gradient_boosting_C0",
        "transformer",
        "momentum",
        "equal_weight_no_model",
    ],
)
def test_incremental_evaluator_hot_commit_is_shared_by_every_family_kind(
    tmp_path: Path,
    family: str,
) -> None:
    def predictions(timestamp: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "family": family,
                "decision_timestamp": [timestamp] * 4,
                "asset_id": ["A", "B", "C", "D"],
                "prediction": [0.4, 0.3, 0.2, 0.1],
            }
        )

    def target_loader(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        targets = keys.copy()
        targets["decision_timestamp"] = pd.to_datetime(
            targets["decision_timestamp"], utc=True
        )
        targets["target_available_timestamp"] = (
            targets["decision_timestamp"] + pd.Timedelta(minutes=60)
        )
        targets["target_is_trainable"] = True
        targets["target_value"] = np.arange(len(targets), dtype=float)
        targets["target_id"] = evaluator.TARGET_ID
        return targets, {"target_rows_loaded": len(targets)}

    metadata = {
        "model_hash": "model",
        "model_vintage_id": "vintage",
        "preprocessing_hash": "preprocessing",
        "policy_hash": "policy",
        "training_cutoff": "2024-01-02T14:30:00Z",
        "prediction_timestamp": "2024-01-02T14:35:01Z",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "target_contract_hash": "target-contract",
        "static_authority_bundle_sha256": "bundle",
        "clean_source_hash": "source",
        "attempt_generation": "1",
    }
    writer = evaluator.ResolvedPerformanceV3Writer(
        tmp_path / family,
        family=family,
        target_loader=target_loader,
    )
    writer.commit_predictions(
        predictions("2024-01-02T14:35:00Z"), metadata=metadata
    )
    result = writer.commit_predictions(
        predictions("2024-01-02T15:35:00Z"), metadata=metadata
    )

    assert (
        result["checkpoint"]["incremental_hot_path_historical_rows_read"] == 0
    )
    state = evaluator._read_json_mapping(writer.incremental_state_path)
    assert state["schema_version"] == "DS24_INCREMENTAL_EVALUATOR_STATE_V1"


def _evaluator_predictions(family: str, timestamp: pd.Timestamp) -> pd.DataFrame:
    assets = [f"A{index:03d}" for index in range(24)]
    return pd.DataFrame(
        {
            "family": family,
            "decision_timestamp": timestamp,
            "asset_id": assets,
            "prediction": np.linspace(1.0, 0.0, len(assets)),
        }
    )


def _evaluator_targets(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    targets = keys.copy()
    targets["decision_timestamp"] = pd.to_datetime(
        targets["decision_timestamp"], utc=True
    )
    asset_number = targets["asset_id"].str.removeprefix("A").astype(int)
    targets["target_available_timestamp"] = (
        targets["decision_timestamp"] + pd.Timedelta(minutes=60)
    )
    targets["target_is_trainable"] = True
    targets["target_value"] = asset_number.astype(float) / 100.0
    targets["target_id"] = evaluator.TARGET_ID
    return targets, {"target_rows_loaded": len(targets)}


EVALUATOR_METADATA = {
    "model_hash": "model-v1",
    "model_vintage_id": "vintage-v1",
    "preprocessing_hash": "preprocessing-v1",
    "policy_hash": "policy-v1",
    "training_cutoff": "2024-01-02T14:30:00Z",
    "prediction_timestamp": "2024-01-02T14:35:01Z",
    "feature_authority_hash": "feature-authority",
    "target_authority_hash": "target-authority",
    "target_contract_hash": "target-contract",
    "static_authority_bundle_sha256": "authority-bundle",
    "clean_source_hash": "source-v1",
    "attempt_generation": "7",
}


def _sorted_log(root: Path, stem: str) -> pd.DataFrame:
    frame = evaluator.read_parquet_log(root, stem)
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in (
            "family",
            "decision_timestamp",
            "session_date",
            "side",
            "sleeve_id",
            "rank",
            "asset_id",
            "audit_sample_ordinal",
            "resolution_timestamp",
        )
        if column in frame
    ]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def test_batched_evaluator_publication_is_row_and_hash_equivalent(
    tmp_path: Path,
) -> None:
    timestamps = list(
        pd.date_range("2024-01-02T14:35:00Z", periods=4, freq="60min", tz="UTC")
    )
    batches = [_evaluator_predictions("ridge_C5", timestamp) for timestamp in timestamps]
    target_batches = [_evaluator_targets(frame[["asset_id", "decision_timestamp"]])[0] for frame in batches]
    expected = {
        timestamp.isoformat(): tuple(frame["asset_id"].astype(str))
        for timestamp, frame in zip(timestamps, batches, strict=True)
    }
    sequential_root = tmp_path / "sequential"
    batched_root = tmp_path / "batched"
    sequential = evaluator.MetricsOnlyEvidenceWriter(
        sequential_root,
        family="ridge_C5",
        enable_resolved_performance_v3=True,
        target_loader=_evaluator_targets,
        terminal_timestamp=timestamps[-1].isoformat(),
    )
    batched = evaluator.MetricsOnlyEvidenceWriter(
        batched_root,
        family="ridge_C5",
        enable_resolved_performance_v3=True,
        target_loader=_evaluator_targets,
        terminal_timestamp=timestamps[-1].isoformat(),
    )
    for timestamp, predictions, targets in zip(
        timestamps, batches, target_batches, strict=True
    ):
        sequential.commit_predictions(
            predictions,
            targets=targets,
            expected_assets=expected[timestamp.isoformat()],
            metadata=EVALUATOR_METADATA,
        )
    result = batched.commit_prediction_batches(
        batches,
        target_batches=target_batches,
        expected_assets_by_timestamp=expected,
        metadata=EVALUATOR_METADATA,
    )

    assert result["publication_disposition"] == "BATCHED_TIMESTAMP_EQUIVALENT"
    for stem in (
        "rank_ic_v3",
        "decision_trace_v3",
        "sleeve_maturity_ledger_v3",
        "transaction_costs_v3",
        "rank_ic_audit_sample_v3",
        "terminal_censored_v3",
        "daily_portfolio_returns_v3",
        "refit_events_v3",
        "pending_outcome_ledger_v3",
        "per_t_metrics",
        "decision_trace",
    ):
        assert_frame_equal(
            _sorted_log(sequential_root, stem),
            _sorted_log(batched_root, stem),
            check_exact=True,
            check_dtype=True,
        )
    assert_frame_equal(
        pd.read_parquet(sequential_root / "pending_scores_v3.parquet"),
        pd.read_parquet(batched_root / "pending_scores_v3.parquet"),
        check_exact=True,
    )
    sequential_pending = pd.read_parquet(
        sequential_root / "pending_buffer.parquet"
    ).drop(columns="committed_at_utc")
    batched_pending = pd.read_parquet(
        batched_root / "pending_buffer.parquet"
    ).drop(columns="committed_at_utc")
    assert_frame_equal(
        sequential_pending,
        batched_pending,
        check_exact=True,
    )
    assert evaluator.parquet_log_rows(batched_root, "rank_ic_v3") == evaluator.parquet_log_rows(
        sequential_root, "rank_ic_v3"
    )
    assert len(evaluator.parquet_log_part_paths(batched_root, "decision_trace_v3")) == 1
    assert len(evaluator.parquet_log_part_paths(sequential_root, "decision_trace_v3")) > 1


def test_batched_evaluator_recovers_after_side_evidence_before_rank_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = list(
        pd.date_range("2024-01-02T14:35:00Z", periods=3, freq="60min", tz="UTC")
    )
    batches = [_evaluator_predictions("ridge_C5", timestamp) for timestamp in timestamps]
    reference_root = tmp_path / "reference"
    interrupted_root = tmp_path / "interrupted"
    reference = evaluator.ResolvedPerformanceV3Writer(
        reference_root,
        family="ridge_C5",
        target_loader=_evaluator_targets,
    )
    reference.commit_prediction_batches(batches, metadata=EVALUATOR_METADATA)

    interrupted = evaluator.ResolvedPerformanceV3Writer(
        interrupted_root,
        family="ridge_C5",
        target_loader=_evaluator_targets,
    )
    original_append = evaluator.append_parquet_log
    rank_failure_injected = False

    def fail_before_rank_marker(
        root: Path, stem: str, frame: pd.DataFrame, **kwargs: object
    ) -> dict[str, object]:
        nonlocal rank_failure_injected
        if stem == "rank_ic_v3" and not rank_failure_injected:
            rank_failure_injected = True
            raise RuntimeError("SYNTHETIC_BATCH_CRASH_BEFORE_RANK_MARKER")
        return original_append(root, stem, frame, **kwargs)

    monkeypatch.setattr(evaluator, "append_parquet_log", fail_before_rank_marker)
    with pytest.raises(RuntimeError, match="SYNTHETIC_BATCH_CRASH"):
        interrupted.commit_prediction_batches(
            batches, metadata=EVALUATOR_METADATA
        )
    assert evaluator.read_parquet_log(interrupted_root, "rank_ic_v3").empty
    assert not evaluator.read_parquet_log(
        interrupted_root, "sleeve_maturity_ledger_v3"
    ).empty

    monkeypatch.setattr(evaluator, "append_parquet_log", original_append)
    resumed = evaluator.ResolvedPerformanceV3Writer(
        interrupted_root,
        family="ridge_C5",
        target_loader=_evaluator_targets,
    )
    resumed.commit_prediction_batches(batches, metadata=EVALUATOR_METADATA)

    for stem in (
        "rank_ic_v3",
        "decision_trace_v3",
        "sleeve_maturity_ledger_v3",
        "transaction_costs_v3",
        "rank_ic_audit_sample_v3",
        "daily_portfolio_returns_v3",
        "refit_events_v3",
        "pending_outcome_ledger_v3",
    ):
        assert_frame_equal(
            _sorted_log(reference_root, stem),
            _sorted_log(interrupted_root, stem),
            check_exact=True,
            check_dtype=True,
        )

