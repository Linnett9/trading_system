from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from core.research.ml.ds24.clean_v2_checkpoint_compatibility import (
    CHECKPOINT_IDENTITY_VERSION,
    CLEAN_V2_FEATURE_AUTHORITY_HASH,
    CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH,
    CLEAN_V2_TARGET_AUTHORITY_HASH,
    SCIENTIFIC_IDENTITY_VERSION,
    CheckpointCompatibilityError,
    assess_checkpoint_compatibility,
    make_checkpoint_identity,
)
from core.research.ml.ds24.clean_v2_contracts import file_sha256, stable_hash
from scripts.local import ds24_clean_v2_family_worker as worker


def _scientific() -> dict[str, object]:
    return {
        "scientific_identity_version": SCIENTIFIC_IDENTITY_VERSION,
        "family": "random_forest",
        "model_config_hash": "model-config",
        "feature_authority_hash": CLEAN_V2_FEATURE_AUTHORITY_HASH,
        "predictor_order_hash": "predictor-order",
        "predictor_manifest_hash": "predictor-manifest",
        "target_authority_hash": CLEAN_V2_TARGET_AUTHORITY_HASH,
        "target_contract_hash": "target-contract",
        "training_lookback_sessions": 20,
        "training_sample_contract": {
            "sample_cap": None,
            "max_training_examples": None,
            "max_training_rows": None,
        },
        "refit_policy_id": "REFIT_EVERY_5_TRADING_SESSIONS_V1",
        "refit_policy_hash": "refit-policy",
        "schedule_contract": {
            "lookback_sessions": 20,
            "score_sessions_per_refit": 5,
            "qualifier_years": None,
            "package_contract_hashes": {
                "2016-02-02T14:35:00+00:00": stable_hash(
                    {
                        "ordinal": 0,
                        "refit_T": "2016-02-02T14:35:00+00:00",
                        "training_session_dates": ["2016-01-04"],
                        "score_session_dates": ["2016-02-02"],
                        "policy_hash": "package-policy",
                    }
                ),
                "2016-02-09T14:35:00+00:00": stable_hash(
                    {
                        "ordinal": 1,
                        "refit_T": "2016-02-09T14:35:00+00:00",
                        "training_session_dates": ["2016-01-11"],
                        "score_session_dates": ["2016-02-09"],
                        "policy_hash": "package-policy",
                    }
                ),
            },
        },
        "schedule_hash": "schedule",
        "estimator_contract": {
            "estimator": "RandomForestRegressor",
            "parameters": {"n_estimators": 100, "random_state": 23},
        },
        "preprocessing_semantics": ["SimpleImputer(strategy='median')"],
        "static_authority_bundle_sha256": (
            CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH
        ),
        "package_contract": {
            "ordinal": 0,
            "refit_T": "2016-02-02T14:35:00+00:00",
            "training_session_dates": ["2016-01-04"],
            "score_session_dates": ["2016-02-02"],
            "policy_hash": "package-policy",
        },
    }


def _operational(
    *, source: str = "source-current", policy: str = "policy-current", attempt: int = 2
) -> dict[str, object]:
    return {
        "clean_source_hash": source,
        "resource_policy_id": policy,
        "resource_policy_hash": stable_hash(policy),
        "maximum_model_workers": 3,
        "attempt_generation": attempt,
        "attempt_id": f"attempt-{attempt}",
        "host": "dell",
        "pid": 123,
        "process_creation_time_utc": "2026-09-27T12:00:00+00:00",
    }


def _checkpoint_identity() -> dict[str, object]:
    return make_checkpoint_identity(
        scientific_identity=_scientific(),
        operational_identity=_operational(
            source="source-checkpoint", policy="policy-old", attempt=1
        ),
    )


def test_operational_only_change_can_resume_scientifically_identical_checkpoint() -> None:
    assessment = assess_checkpoint_compatibility(
        checkpoint_identity=_checkpoint_identity(),
        expected_scientific_identity=_scientific(),
        current_operational_identity=_operational(),
    )

    assert assessment.reusable is True
    assert assessment.classification == (
        "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE"
    )
    assert assessment.checkpoint_source_hash == "source-checkpoint"
    assert assessment.current_source_hash == "source-current"
    assert set(assessment.operational_differences) >= {
        "clean_source_hash",
        "resource_policy_id",
        "attempt_generation",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_authority_hash", "changed-feature"),
        ("target_authority_hash", "changed-target"),
        ("model_config_hash", "changed-model"),
        ("training_lookback_sessions", 40),
        ("schedule_hash", "changed-schedule"),
    ],
)
def test_scientific_changes_fail_closed(field: str, value: object) -> None:
    expected = {**_scientific(), field: value}

    with pytest.raises(CheckpointCompatibilityError):
        assess_checkpoint_compatibility(
            checkpoint_identity=_checkpoint_identity(),
            expected_scientific_identity=expected,
            current_operational_identity=_operational(),
        )


def test_changed_model_parameters_fail_even_if_display_hash_is_unchanged() -> None:
    expected = _scientific()
    expected["estimator_contract"] = {
        "estimator": "RandomForestRegressor",
        "parameters": {"n_estimators": 101, "random_state": 23},
    }

    with pytest.raises(CheckpointCompatibilityError, match="SCIENTIFIC"):
        assess_checkpoint_compatibility(
            checkpoint_identity=_checkpoint_identity(),
            expected_scientific_identity=expected,
            current_operational_identity=_operational(),
        )


def test_contaminated_v1_checkpoint_is_not_accepted() -> None:
    contaminated = {
        "clean_source_hash": "r40",
        "family": "random_forest",
        "feature_authority_hash": "v1-feature",
        "model_config_hash": "model-config",
        "policy_hash": "package-policy",
        "refit_T": "2016-02-02T14:35:00+00:00",
        "refit_policy_hash": "refit-policy",
        "static_authority_bundle_sha256": "v1-bundle",
        "target_authority_hash": "v1-target",
        "target_contract_hash": "target-contract",
    }

    with pytest.raises(CheckpointCompatibilityError):
        assess_checkpoint_compatibility(
            checkpoint_identity=contaminated,
            expected_scientific_identity=_scientific(),
            current_operational_identity=_operational(),
        )


def test_legacy_clean_v2_checkpoint_uses_bundle_backed_scientific_projection() -> None:
    scientific = _scientific()
    package = dict(scientific["package_contract"])
    legacy = {
        "clean_source_hash": "source-checkpoint",
        "family": scientific["family"],
        "feature_authority_hash": scientific["feature_authority_hash"],
        "model_config_hash": scientific["model_config_hash"],
        "policy_hash": package["policy_hash"],
        "refit_T": package["refit_T"],
        "refit_policy_hash": scientific["refit_policy_hash"],
        "static_authority_bundle_sha256": scientific[
            "static_authority_bundle_sha256"
        ],
        "target_authority_hash": scientific["target_authority_hash"],
        "target_contract_hash": scientific["target_contract_hash"],
    }

    assessment = assess_checkpoint_compatibility(
        checkpoint_identity=legacy,
        expected_scientific_identity=scientific,
        current_operational_identity=_operational(),
    )

    assert assessment.reusable is True
    assert assessment.legacy_clean_v2_checkpoint is True
    assert assessment.classification == (
        "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE"
    )


def test_prior_completed_package_checkpoint_is_safe_but_not_reused() -> None:
    prior = _scientific()
    prior["package_contract"] = {
        **dict(prior["package_contract"]),
        "refit_T": "2016-02-02T14:35:00+00:00",
    }
    expected = _scientific()
    expected["package_contract"] = {
        **dict(expected["package_contract"]),
        "ordinal": 1,
        "refit_T": "2016-02-09T14:35:00+00:00",
        "training_session_dates": ["2016-01-11"],
        "score_session_dates": ["2016-02-09"],
    }
    identity = make_checkpoint_identity(
        scientific_identity=prior,
        operational_identity=_operational(source="old"),
    )

    assessment = assess_checkpoint_compatibility(
        checkpoint_identity=identity,
        expected_scientific_identity=expected,
        current_operational_identity=_operational(),
        completed_refits=["2016-02-02T14:35:00+00:00"],
    )

    assert assessment.reusable is False
    assert assessment.prior_completed_package is True
    assert assessment.classification == (
        "PRIOR_COMPLETED_PACKAGE_CHECKPOINT_NOT_REUSED"
    )


def test_loader_records_provenance_before_loading_compatible_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    model_path = tmp_path / "model.pkl"
    identity = _checkpoint_identity()
    with model_path.open("wb") as handle:
        pickle.dump({"identity": identity, "model": "tiny-model"}, handle)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "identity": identity,
                "model_path": "model.pkl",
                "model_hash": file_sha256(model_path),
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "compatibility.json"

    loaded = worker._load_resume_model(
        checkpoint_path,
        expected_scientific_identity=_scientific(),
        current_operational_identity=_operational(),
        completed_refits=[],
        compatibility_evidence_path=evidence_path,
    )

    assert loaded is not None and loaded[0] == "tiny-model"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["classification"] == (
        "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE"
    )
    assert evidence["artifact_identity_verification"] == "VERIFIED"
    assert evidence["checkpoint_source_hash"] == "source-checkpoint"
    assert evidence["current_source_hash"] == "source-current"


def test_completed_package_and_metric_keys_are_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = list(pd.date_range("2016-02-02T14:35:00Z", periods=3, freq="5min"))
    completed = {timestamp.isoformat() for timestamp in timestamps}
    assert worker._package_score_is_complete(timestamps, completed) is True
    monkeypatch.setattr(
        worker,
        "read_parquet_log",
        lambda *_args: pd.DataFrame(
            {
                "family": ["random_forest", "random_forest"],
                "decision_timestamp": [timestamps[0], timestamps[0]],
            }
        ),
    )
    assert worker._completed_timestamps(Path("unused"), "random_forest") == {
        timestamps[0].isoformat()
    }


def test_stale_attempt_state_cannot_override_current_attempt() -> None:
    prior = {
        "attempt_generation": 10,
        "clean_source_hash": "old",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "model_config_hash": "model",
        "completed_refits": ["one"],
        "metrics_cursor": 3,
    }
    authority = {
        "clean_source_hash": "new",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "model_config_hash": "model",
    }

    with pytest.raises(worker.CleanV2WorkerError, match="newer attempt"):
        worker._validate_resume_authority(
            prior=prior,
            authority=authority,
            attempt_generation=10,
        )


def test_state_records_current_attempt_and_checkpoint_source_provenance() -> None:
    compatibility = {
        "classification": "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE",
        "checkpoint_source_hash": "source-checkpoint",
        "current_source_hash": "source-current",
    }
    state = worker._state_payload(
        family="random_forest",
        host="dell",
        authority={"clean_source_hash": "source-current"},
        expected_refits=2,
        completed_refits=["2016-02-02T14:35:00+00:00"],
        completed_timestamps={"2016-02-02T14:35:00+00:00"},
        terminal_state="RUNNING",
        attempt_generation=22,
        attempt_id="attempt-22",
        operational_identity=_operational(attempt=22),
        checkpoint_compatibility=compatibility,
    )

    assert state["attempt_generation"] == 22
    assert state["clean_source_hash"] == "source-current"
    assert state["current_operational_identity"]["attempt_id"] == "attempt-22"
    assert state["checkpoint_compatibility"] == compatibility


def test_checkpoint_identity_schema_is_explicit() -> None:
    identity = _checkpoint_identity()

    assert identity["checkpoint_identity_version"] == CHECKPOINT_IDENTITY_VERSION
    assert identity["scientific_identity_hash"] == stable_hash(_scientific())
