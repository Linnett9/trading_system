from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
)
from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.prediction_ledger_contract import (
    PredictionProvenance,
    build_ranked_prediction_rows,
)
from core.research.ml.ds24.comparable_policy import RefitPackageSpec
from core.research.ml.ds24_metrics_only_evaluator import append_parquet_log
from scripts.local.ds24_clean_v2_family_worker import (
    DETERMINISTIC_RANDOM_FOREST_PREDICT_N_JOBS,
    QUALIFIED_RANDOM_FOREST_N_JOBS,
    CleanV2WorkerError,
    _apply_production_execution_overrides,
    _control_model_identity,
    _package_assembly_dates,
    _recover_or_score_prediction_batches,
    _score_model,
    _scoring_timestamp_batches,
    _tabular_estimator,
)
from infrastructure.data.ds24_prediction_ledger import PredictionLedgerStore
from scripts.local import ds24_clean_v2_family_worker as worker


def _score_panel() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        ["2024-01-02T14:35:00Z", "2024-01-02T14:40:00Z"], utc=True
    )
    rows = []
    for timestamp in timestamps:
        for asset_index in range(4):
            rows.append(
                {
                    "decision_timestamp": timestamp,
                    "asset_id": f"A{asset_index}",
                    "x1": float(asset_index),
                    "x2": np.nan if asset_index == 1 else float(asset_index * 2),
                }
            )
    return pd.DataFrame(rows)


def test_random_forest_session_batch_is_exactly_equal_to_sequential_scoring() -> None:
    predictors = ["x1", "x2"]
    training = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [0.0, np.nan, 4.0, 6.0, 8.0, 10.0],
            "target_value": [0.1, 0.2, -0.1, 0.3, -0.2, 0.4],
        }
    )
    config = {
        "parameters": {
            "n_estimators": 12,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "n_jobs": 1,
            "random_state": 23,
        }
    }
    model = _tabular_estimator("random_forest", config)
    model.fit(training[predictors], training["target_value"])
    panel = _score_panel()

    sequential = pd.concat(
        [
            _score_model(
                "random_forest",
                model,
                rows.copy(),
                panel,
                predictors,
                {},
            )
            for _timestamp, rows in panel.groupby(
                "decision_timestamp", sort=False
            )
        ],
        ignore_index=True,
    )
    batched = _score_model(
        "random_forest",
        model,
        panel.copy(),
        panel,
        predictors,
        {},
    )

    assert_frame_equal(
        sequential.reset_index(drop=True),
        batched.reset_index(drop=True),
        check_exact=True,
    )
    sequential_ranks = sequential.groupby("decision_timestamp")[
        "prediction"
    ].rank(method="first", ascending=False)
    batched_ranks = batched.groupby("decision_timestamp")["prediction"].rank(
        method="first", ascending=False
    )
    assert sequential_ranks.tolist() == batched_ranks.tolist()


def test_random_forest_prediction_is_bitwise_stable_and_restores_fit_parallelism(
    tmp_path: Path,
) -> None:
    predictors = ["x1", "x2"]
    training = pd.DataFrame(
        {
            "x1": np.arange(48, dtype=float),
            "x2": np.arange(48, dtype=float) % 7,
            "target_value": np.sin(np.arange(48, dtype=float)),
        }
    )
    config = {
        "parameters": {
            "n_estimators": 32,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "n_jobs": QUALIFIED_RANDOM_FOREST_N_JOBS,
            "random_state": 23,
        }
    }
    model = _tabular_estimator("random_forest", config)
    model.fit(training[predictors], training["target_value"])
    panel = _score_panel()

    baseline = _score_model(
        "random_forest", model, panel.copy(), panel, predictors, {}
    )
    baseline_bytes = baseline["prediction"].to_numpy(dtype=np.float64).tobytes()
    for _ in range(100):
        observed = _score_model(
            "random_forest", model, panel.copy(), panel, predictors, {}
        )
        assert observed["prediction"].to_numpy(dtype=np.float64).tobytes() == baseline_bytes
        assert model.steps[-1][1].n_jobs == QUALIFIED_RANDOM_FOREST_N_JOBS

    model_path = tmp_path / "rf.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    for _ in range(50):
        with model_path.open("rb") as handle:
            reloaded = pickle.load(handle)
        observed = _score_model(
            "random_forest", reloaded, panel.copy(), panel, predictors, {}
        )
        assert observed["prediction"].to_numpy(dtype=np.float64).tobytes() == baseline_bytes
        assert reloaded.steps[-1][1].n_jobs == QUALIFIED_RANDOM_FOREST_N_JOBS

    assert DETERMINISTIC_RANDOM_FOREST_PREDICT_N_JOBS == 1


def test_rf_fit_parallelism_is_operational_and_does_not_mutate_scientific_config() -> None:
    scientific = {
        "parameters": {"n_estimators": 8, "random_state": 23},
        "training": {"refit_policy": "REFIT_EVERY_5_TRADING_SESSIONS_V1"},
    }
    before = stable_hash(scientific)

    effective = _apply_production_execution_overrides("random_forest", scientific)

    assert effective["parameters"]["n_jobs"] == QUALIFIED_RANDOM_FOREST_N_JOBS
    assert stable_hash(scientific) == before
    assert "n_jobs" not in scientific["parameters"]


def test_complete_ledger_population_skips_model_prediction(tmp_path: Path) -> None:
    timestamp = pd.Timestamp("2024-01-02T14:35:00Z")
    provenance = PredictionProvenance(
        source_hash="source-before-restart",
        feature_authority_hash="feature",
        target_authority_hash="target",
        static_bundle_hash="static",
        run_id="run",
    )
    durable = build_ranked_prediction_rows(
        pd.DataFrame(
            {
                "decision_timestamp": [timestamp, timestamp],
                "asset_id": ["A", "B"],
                "raw_prediction": [0.2, 0.1],
            }
        ),
        family="random_forest",
        refit_id="model-vintage",
        refit_timestamp="2024-01-01T14:35:00Z",
        evaluation_eligible=True,
        provenance=provenance,
    )
    store = PredictionLedgerStore(tmp_path)
    store.append(durable)
    publisher = DS24PredictionLedgerPublisher(
        store=store,
        provenance=PredictionProvenance(
            source_hash="repair-source",
            feature_authority_hash="feature",
            target_authority_hash="target",
            static_bundle_hash="static",
            run_id="run",
        ),
    )
    score_batch = pd.DataFrame(
        {
            "decision_timestamp": [timestamp, timestamp],
            "asset_id": ["A", "B"],
            "x1": [1.0, 2.0],
        }
    )

    batches, recovery = _recover_or_score_prediction_batches(
        publisher=publisher,
        family="random_forest",
        model=object(),  # would fail immediately if prediction were attempted
        score_batch=score_batch,
        full_panel=score_batch,
        predictors=["x1"],
        metadata={},
        timestamp_batch=(timestamp,),
        expected_assets_by_timestamp={timestamp.isoformat(): ("A", "B")},
        refit_id="model-vintage",
        refit_timestamp=pd.Timestamp("2024-01-01T14:35:00Z"),
    )

    assert recovery.recovered_row_count == 2
    assert batches[0]["prediction"].tolist() == [0.2, 0.1]


def test_mixed_batch_scores_only_absent_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_timestamp = pd.Timestamp("2024-01-02T14:35:00Z")
    absent_timestamp = pd.Timestamp("2024-01-02T14:40:00Z")
    provenance = PredictionProvenance(
        source_hash="source",
        feature_authority_hash="feature",
        target_authority_hash="target",
        static_bundle_hash="static",
        run_id="run",
    )
    store = PredictionLedgerStore(tmp_path)
    store.append(
        build_ranked_prediction_rows(
            pd.DataFrame(
                {
                    "decision_timestamp": [durable_timestamp] * 2,
                    "asset_id": ["A", "B"],
                    "raw_prediction": [0.2, 0.1],
                }
            ),
            family="ridge",
            refit_id="model-vintage",
            refit_timestamp="2024-01-01T14:35:00Z",
            evaluation_eligible=True,
            provenance=provenance,
        )
    )
    publisher = DS24PredictionLedgerPublisher(store=store, provenance=provenance)
    score_batch = pd.DataFrame(
        {
            "decision_timestamp": [
                durable_timestamp,
                durable_timestamp,
                absent_timestamp,
                absent_timestamp,
            ],
            "asset_id": ["A", "B", "A", "B"],
            "x1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    observed_score_timestamps: list[pd.Timestamp] = []

    def fake_score(
        family: str,
        model: object,
        score_panel: pd.DataFrame,
        full_panel: pd.DataFrame,
        predictors: list[str],
        metadata: dict[str, object],
        **kwargs: object,
    ) -> pd.DataFrame:
        del model, full_panel, predictors, metadata, kwargs
        normalized = pd.to_datetime(
            score_panel["decision_timestamp"], utc=True
        ).reset_index(drop=True)
        observed_score_timestamps.extend(normalized.unique())
        return pd.DataFrame(
            {
                "family": family,
                "decision_timestamp": normalized.map(
                    lambda value: value.isoformat()
                ),
                "asset_id": score_panel["asset_id"].astype(str).reset_index(drop=True),
                "prediction": [0.4, 0.3],
                "eligible": pd.Series([True, True], dtype=bool),
            }
        )

    monkeypatch.setattr(worker, "_score_model", fake_score)
    batches, recovery = worker._recover_or_score_prediction_batches(
        publisher=publisher,
        family="ridge",
        model=object(),
        score_batch=score_batch,
        full_panel=score_batch,
        predictors=["x1"],
        metadata={},
        timestamp_batch=(durable_timestamp, absent_timestamp),
        expected_assets_by_timestamp={
            durable_timestamp.isoformat(): ("A", "B"),
            absent_timestamp.isoformat(): ("A", "B"),
        },
        refit_id="model-vintage",
        refit_timestamp=pd.Timestamp("2024-01-01T14:35:00Z"),
    )

    assert observed_score_timestamps == [absent_timestamp]
    assert recovery.recovered_row_count == 2
    assert [len(batch) for batch in batches] == [2, 2]


def test_scoring_batches_stay_within_session_and_skip_completed() -> None:
    timestamps = list(
        pd.to_datetime(
            [
                "2024-01-02T14:35:00Z",
                "2024-01-02T14:40:00Z",
                "2024-01-03T14:35:00Z",
            ],
            utc=True,
        )
    )
    batches = _scoring_timestamp_batches(
        "random_forest",
        timestamps,
        {timestamps[0].isoformat()},
    )

    assert batches == [(timestamps[1],), (timestamps[2],)]
    assert _scoring_timestamp_batches("transformer", timestamps, set()) == [
        (timestamp,) for timestamp in timestamps
    ]


def test_no_model_controls_assemble_only_pending_tail_and_score_sessions() -> None:
    package = RefitPackageSpec(
        ordinal=3,
        refit_T=pd.Timestamp("2024-02-01T14:35:00Z"),
        training_session_dates=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-31",
        ],
        score_session_dates=["2024-02-01", "2024-02-02"],
        policy_hash="policy",
    )

    assert _package_assembly_dates("equal_weight_no_model", package) == [
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
    ]
    assert _package_assembly_dates("momentum", package) == [
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
    ]
    assert _package_assembly_dates("random_forest", package) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
    ]


def test_no_model_control_requires_preceding_target_resolution_session() -> None:
    package = RefitPackageSpec(
        ordinal=0,
        refit_T=pd.Timestamp("2024-02-01T14:35:00Z"),
        training_session_dates=[],
        score_session_dates=["2024-02-01"],
        policy_hash="policy",
    )

    with pytest.raises(CleanV2WorkerError, match="no preceding session"):
        _package_assembly_dates("equal_weight_no_model", package)


def _control_package() -> RefitPackageSpec:
    return RefitPackageSpec(
        ordinal=0,
        refit_T=pd.Timestamp("2024-02-01T14:35:00Z"),
        training_session_dates=["2024-01-31"],
        score_session_dates=["2024-02-01"],
        policy_hash="policy",
    )


def test_control_model_identity_is_independent_of_operational_identity(
    tmp_path: Path,
) -> None:
    package = _control_package()
    scientific_identity = {"family": "equal_weight_no_model", "package": 0}
    timestamps = [pd.Timestamp("2024-02-01T14:35:00Z")]

    first = _control_model_identity(
        tmp_path,
        family="equal_weight_no_model",
        scientific_identity=scientific_identity,
        package=package,
        score_timestamps=timestamps,
    )
    second = _control_model_identity(
        tmp_path,
        family="equal_weight_no_model",
        scientific_identity=dict(scientific_identity),
        package=package,
        score_timestamps=timestamps,
    )

    assert first == (stable_hash(scientific_identity), "SCIENTIFIC_PACKAGE_IDENTITY")
    assert second == first


def test_control_model_identity_reuses_persisted_legacy_package_hash(
    tmp_path: Path,
) -> None:
    package = _control_package()
    legacy_hash = "a" * 64
    append_parquet_log(
        tmp_path,
        "refit_events_v3",
        pd.DataFrame(
            [
                {
                    "family": "equal_weight_no_model",
                    "training_cutoff": package.refit_T.isoformat(),
                    "model_hash": legacy_hash,
                }
            ]
        ),
    )

    observed = _control_model_identity(
        tmp_path,
        family="equal_weight_no_model",
        scientific_identity={"new": "scientific identity"},
        package=package,
        score_timestamps=[pd.Timestamp("2024-02-01T14:35:00Z")],
    )

    assert observed == (legacy_hash, "PERSISTED_REFIT_EVENT_COMPATIBILITY")


def test_control_model_identity_recovers_orphaned_partial_evidence(
    tmp_path: Path,
) -> None:
    package = _control_package()
    legacy_hash = "c" * 64
    append_parquet_log(
        tmp_path,
        "sleeve_maturity_ledger_v3",
        pd.DataFrame(
            [
                {
                    "family": "equal_weight_no_model",
                    "decision_timestamp": "2024-02-01T14:35:00+00:00",
                    "model_hash": legacy_hash,
                }
            ]
        ),
    )

    observed = _control_model_identity(
        tmp_path,
        family="equal_weight_no_model",
        scientific_identity={"new": "scientific identity"},
        package=package,
        score_timestamps=[pd.Timestamp("2024-02-01T14:35:00Z")],
    )

    assert observed == (legacy_hash, "PERSISTED_PARTIAL_EVIDENCE_COMPATIBILITY")


def test_control_model_identity_fails_closed_on_conflicting_package_hashes(
    tmp_path: Path,
) -> None:
    package = _control_package()
    append_parquet_log(
        tmp_path,
        "refit_events_v3",
        pd.DataFrame(
            [
                {
                    "family": "equal_weight_no_model",
                    "training_cutoff": package.refit_T.isoformat(),
                    "model_hash": "a" * 64,
                },
                {
                    "family": "equal_weight_no_model",
                    "training_cutoff": package.refit_T.isoformat(),
                    "model_hash": "b" * 64,
                },
            ]
        ),
    )

    with pytest.raises(CleanV2WorkerError, match="conflicting persisted"):
        _control_model_identity(
            tmp_path,
            family="equal_weight_no_model",
            scientific_identity={"new": "scientific identity"},
            package=package,
            score_timestamps=[pd.Timestamp("2024-02-01T14:35:00Z")],
        )
