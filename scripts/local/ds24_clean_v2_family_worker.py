from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (
    authority_bundle,
    clean_source_hash,
    file_sha256,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_checkpoint_compatibility import (
    SCIENTIFIC_IDENTITY_VERSION,
    CheckpointCompatibilityError,
    assess_checkpoint_compatibility,
    make_checkpoint_identity,
    make_no_model_control_resume_identity,
    no_model_control_config,
    validate_no_model_control_resume_identity,
)
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData
from core.research.ml.ds24.clean_v2_runtime import (
    MAX_DELL_MODEL_WORKERS,
    QUALIFIER_YEARS,
    REFIT_POLICY_ID,
    RUN_ID,
    validate_target_use,
)
from core.research.ml.ds24.clean_v2_resources import (
    DELL_WORKER_THREAD_ENVIRONMENT,
    RECOVERY_RESOURCE_POLICY_ID,
    RESOURCE_CAPACITY_DEFERRED_EXIT_CODE,
    RESOURCE_PRESSURE_EXIT_CODE,
    AllocationEstimate,
    CleanV2ResourcePressure,
    FileReservationClient,
    ResourceReservationUnavailable,
    WorkerReservationOwner,
    allocation_failure_decision,
    is_resource_allocation_failure,
    panel_allocation_estimate,
    process_identity,
    read_json_object_with_retry,
    recovery_policy_payload,
    reservation_payload_is_local_capacity_deferral,
    require_memory,
    sequence_allocation_estimate,
    tabular_fit_allocation_estimate,
    write_json_object_atomic,
)
from core.research.ml.ds24.comparable_policy import RefitPackageSpec
from core.research.ml.stock_level.stock_level_sequence_regressors import (
    SequenceRegressorConfig,
    TorchSequenceReturnRegressor,
)
from core.research.ml.ds24_metrics_only_evaluator import (
    MetricsOnlyEvidenceWriter,
    read_parquet_log,
    resolved_performance_contract_v3_hash,
)


SEQUENCE_FAMILIES = {
    "transformer",
    "momentum_transformer",
    "market_context_encoder",
    "temporal_fusion_transformer",
}
CONTROL_FAMILIES = {"momentum", "equal_weight_no_model"}
SHORT_LANES = {"SHORT_REQUALIFICATION", "UNSCORED_DISCOVERY"}


class CleanV2WorkerError(RuntimeError):
    """Raised when a clean V2 worker cannot safely continue."""


def _validate_worker_admission(host: str) -> dict[str, Any]:
    admission_path = (
        ROOT
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / f"manual_admission_{host}.json"
    )
    if not admission_path.is_file():
        raise CleanV2WorkerError("Manual supervisor admission is missing")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    token = admission.pop("admission_token", None)
    environment_token = os.environ.get("DS24_CLEAN_V2_ADMISSION_TOKEN")
    if not token or token != stable_hash(admission) or token != environment_token:
        raise CleanV2WorkerError("Manual supervisor admission token is invalid")
    if admission.get("run_id") != RUN_ID or admission.get("host_role") != host:
        raise CleanV2WorkerError("Manual supervisor admission scope mismatch")
    if admission.get("refit_policy") != REFIT_POLICY_ID:
        raise CleanV2WorkerError("Manual supervisor admission refit mismatch")
    if admission.get("static_authority_bundle_sha256") != authority_bundle()[
        "bundle_sha256"
    ]:
        raise CleanV2WorkerError("Manual supervisor admission bundle is stale")
    if admission.get("clean_source_hash") != clean_source_hash():
        raise CleanV2WorkerError("Manual supervisor admission source hash is stale")
    expected_workers = MAX_DELL_MODEL_WORKERS if host == "dell" else 1
    if admission.get("maximum_model_workers") != expected_workers:
        raise CleanV2WorkerError("Manual supervisor admission worker limit is stale")
    resource_policy = admission.get("resource_policy") or {}
    if resource_policy != recovery_policy_payload():
        raise CleanV2WorkerError("Manual supervisor resource policy is stale")
    if os.environ.get("DS24_CLEAN_V2_RESOURCE_POLICY_ID") != (
        RECOVERY_RESOURCE_POLICY_ID
    ):
        raise CleanV2WorkerError("Supervisor resource policy propagation is missing")
    return {**admission, "admission_token": token}


def _utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def _release_disposable_caches() -> None:
    """Release runtime caches only at an already committed package boundary."""

    gc.collect()
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            cuda.empty_cache()


@contextmanager
def _allocation_guard(
    client: FileReservationClient | None,
    *,
    stage: str,
    estimate: AllocationEstimate,
) -> Iterable[None]:
    """Hold supervisor authorization for the complete allocation lifetime."""

    if client is None:
        require_memory(
            stage=stage,
            estimated_allocation_bytes=estimate.physical_bytes,
        )
        yield
        return
    with client.reserve(stage=stage, estimate=estimate):
        yield


def _reservation_client(
    *, family: str, host: str, attempt_generation: int
) -> FileReservationClient:
    reservation_root = os.environ.get("DS24_CLEAN_V2_RESERVATION_ROOT", "")
    if not reservation_root:
        raise CleanV2WorkerError("Supervisor reservation controller is missing")
    identity = process_identity(os.getpid())
    if not identity.alive or not identity.creation_time_utc:
        raise CleanV2WorkerError("Worker reservation identity is unavailable")
    if host == "dell":
        invalid = [
            name
            for name, expected in DELL_WORKER_THREAD_ENVIRONMENT.items()
            if os.environ.get(name) != expected
        ]
        if invalid:
            raise CleanV2WorkerError(
                "Dell worker thread-cap propagation is invalid:" + ",".join(invalid)
            )
    return FileReservationClient(
        Path(reservation_root),
        WorkerReservationOwner(
            family=family,
            attempt_generation=attempt_generation,
            pid=os.getpid(),
            process_creation_time_utc=identity.creation_time_utc,
        ),
    )


def _log_runtime_file_retry(event: Mapping[str, Any]) -> None:
    print(
        json.dumps({**dict(event), "component": "worker"}, sort_keys=True),
        flush=True,
    )


def _read_runtime_json(path: Path) -> dict[str, Any]:
    return read_json_object_with_retry(path, on_retry=_log_runtime_file_retry)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_object_atomic(path, payload, on_retry=_log_runtime_file_retry)


def _session_dates(spine: Iterable[pd.Timestamp]) -> list[str]:
    return sorted({pd.Timestamp(value).date().isoformat() for value in spine})


def build_clean_refit_schedule(
    spine: Sequence[pd.Timestamp],
    *,
    lookback_sessions: int,
    qualifier_years: Sequence[int] | None,
    maximum_refits: int | None = None,
) -> list[RefitPackageSpec]:
    """Build fixed five-session refits, optionally restarted within frozen qualifier years."""

    if lookback_sessions <= 0:
        raise CleanV2WorkerError("Training lookback must be positive")
    timestamps = [pd.Timestamp(value).tz_convert("UTC") for value in spine]
    sessions = _session_dates(timestamps)
    positions = {session: index for index, session in enumerate(sessions)}
    first_timestamp_by_session: dict[str, pd.Timestamp] = {}
    for timestamp in timestamps:
        first_timestamp_by_session.setdefault(
            timestamp.date().isoformat(), timestamp
        )
    groups: list[list[str]] = []
    if qualifier_years is None:
        score_sessions = sessions[lookback_sessions:]
        groups = [
            score_sessions[index : index + 5]
            for index in range(0, len(score_sessions), 5)
            if score_sessions[index : index + 5]
        ]
    else:
        for year in qualifier_years:
            year_sessions = [
                session
                for session in sessions
                if int(session[:4]) == int(year)
                and positions[session] >= lookback_sessions
            ]
            groups.extend(
                year_sessions[index : index + 5]
                for index in range(0, len(year_sessions), 5)
                if year_sessions[index : index + 5]
            )

    policy_hash = stable_hash(
        {
            "id": REFIT_POLICY_ID,
            "lookback_sessions": lookback_sessions,
            "score_sessions_per_refit": 5,
            "score_cadence": "registered five-minute decision spine",
            "qualifier_years": list(qualifier_years) if qualifier_years else None,
        }
    )
    out: list[RefitPackageSpec] = []
    for score_sessions in groups:
        first_position = positions[score_sessions[0]]
        training_sessions = sessions[
            first_position - lookback_sessions : first_position
        ]
        if len(training_sessions) != lookback_sessions:
            continue
        refit_t = first_timestamp_by_session[score_sessions[0]]
        out.append(
            RefitPackageSpec(
                ordinal=len(out),
                refit_T=refit_t,
                training_session_dates=training_sessions,
                score_session_dates=list(score_sessions),
                policy_hash=policy_hash,
            )
        )
        if maximum_refits is not None and len(out) >= maximum_refits:
            break
    return out


def _schedule_identity(
    *,
    schedule: Sequence[RefitPackageSpec],
    lane: str,
    lookback_sessions: int,
    qualifier_years: Sequence[int] | None,
) -> tuple[dict[str, Any], str]:
    packages = [
        {
            "ordinal": package.ordinal,
            "refit_T": package.refit_T.isoformat(),
            "training_session_dates": list(package.training_session_dates),
            "score_session_dates": list(package.score_session_dates),
            "policy_hash": package.policy_hash,
        }
        for package in schedule
    ]
    contract = {
        "run_id": RUN_ID,
        "lane": lane,
        "refit_policy_id": REFIT_POLICY_ID,
        "lookback_sessions": int(lookback_sessions),
        "score_sessions_per_refit": 5,
        "score_cadence": "registered five-minute decision spine",
        "qualifier_years": (
            list(map(int, qualifier_years)) if qualifier_years else None
        ),
        "expected_refits": len(schedule),
        "package_contract_hashes": {
            str(package["refit_T"]): stable_hash(package) for package in packages
        },
    }
    return contract, stable_hash({"contract": contract, "packages": packages})


def _scientific_checkpoint_identity(
    *,
    family: str,
    config: Mapping[str, Any],
    predictors: Sequence[str],
    feature_authority_hash: str,
    target_authority_hash: str,
    target_contract_hash: str,
    static_authority_bundle_sha256: str,
    schedule_contract: Mapping[str, Any],
    schedule_hash: str,
    package: RefitPackageSpec,
) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    predictor_order = list(map(str, predictors))
    predictor_manifest = load_contract("predictor_manifest.json")
    return {
        "scientific_identity_version": SCIENTIFIC_IDENTITY_VERSION,
        "family": family,
        "model_config_hash": stable_hash(config),
        "feature_authority_hash": feature_authority_hash,
        "predictor_order_hash": stable_hash(predictor_order),
        "predictor_manifest_hash": stable_hash(predictor_manifest),
        "target_authority_hash": target_authority_hash,
        "target_contract_hash": target_contract_hash,
        "training_lookback_sessions": int(training["lookback_sessions"]),
        "training_sample_contract": {
            "sample_cap": training.get("sample_cap"),
            "max_training_examples": training.get("max_training_examples"),
            "max_training_rows": training.get("max_training_rows"),
        },
        "refit_policy_id": REFIT_POLICY_ID,
        "refit_policy_hash": stable_hash(
            {
                "id": REFIT_POLICY_ID,
                "lookback_sessions": training["lookback_sessions"],
            }
        ),
        "schedule_contract": dict(schedule_contract),
        "schedule_hash": schedule_hash,
        "estimator_contract": {
            key: config.get(key)
            for key in (
                "status",
                "implementation",
                "estimator",
                "parameters",
                "execution",
            )
        },
        "preprocessing_semantics": config.get("preprocessing", []),
        "static_authority_bundle_sha256": static_authority_bundle_sha256,
        "package_contract": {
            "ordinal": package.ordinal,
            "refit_T": package.refit_T.isoformat(),
            "training_session_dates": list(package.training_session_dates),
            "score_session_dates": list(package.score_session_dates),
            "policy_hash": package.policy_hash,
        },
    }


def _operational_checkpoint_identity(
    *,
    admission: Mapping[str, Any],
    host: str,
    attempt_generation: int,
    attempt_id: str,
) -> dict[str, Any]:
    identity = process_identity(os.getpid())
    if not identity.alive or not identity.creation_time_utc:
        raise CleanV2WorkerError("Worker operational identity is unavailable")
    resource_policy = admission.get("resource_policy")
    if not isinstance(resource_policy, dict):
        raise CleanV2WorkerError("Worker resource policy provenance is missing")
    return {
        "clean_source_hash": clean_source_hash(),
        "resource_policy_id": str(resource_policy["resource_policy_id"]),
        "resource_policy_hash": stable_hash(resource_policy),
        "maximum_model_workers": int(admission["maximum_model_workers"]),
        "attempt_generation": int(attempt_generation),
        "attempt_id": attempt_id,
        "host": host,
        "pid": os.getpid(),
        "process_creation_time_utc": identity.creation_time_utc,
    }


def _package_score_is_complete(
    score_timestamps: Sequence[pd.Timestamp], completed: set[str]
) -> bool:
    score_keys = {
        pd.Timestamp(timestamp).tz_convert("UTC").isoformat()
        for timestamp in score_timestamps
    }
    return bool(score_keys and score_keys <= completed)


def _family_contract(family: str) -> tuple[dict[str, Any], str]:
    models = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    if family in models["families"]:
        config = dict(models["families"][family])
    elif family in models["controls"]:
        config = no_model_control_config(
            family=family,
            model_registry=models,
            refit_policy_id=REFIT_POLICY_ID,
        )
    else:
        raise CleanV2WorkerError(f"Family is not admitted: {family}")
    lane_matches = [
        lane for lane, families in tournament["lanes"].items() if family in families
    ]
    if lane_matches != [config["lane"]]:
        raise CleanV2WorkerError(
            f"Registry/tournament lane mismatch for {family}: {lane_matches}"
        )
    if config["training"]["refit_policy"] != REFIT_POLICY_ID:
        raise CleanV2WorkerError(f"Forbidden refit policy for {family}")
    return config, str(config["lane"])


def _control_resume_identity(
    *,
    family: str,
    config: Mapping[str, Any],
    feature_authority_hash: str,
    target_authority_hash: str,
    target_contract_hash: str,
    static_authority_bundle_sha256: str,
) -> dict[str, Any] | None:
    if family not in CONTROL_FAMILIES:
        return None
    return make_no_model_control_resume_identity(
        family=family,
        model_config=config,
        feature_authority_hash=feature_authority_hash,
        target_authority_hash=target_authority_hash,
        target_contract_hash=target_contract_hash,
        static_authority_bundle_sha256=static_authority_bundle_sha256,
        predictor_manifest=load_contract("predictor_manifest.json"),
        eligibility_contract=load_contract("eligibility_contract.json"),
        tournament_contract=load_contract("tournament_contract.json"),
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )


def _tabular_estimator(family: str, config: Mapping[str, Any]) -> Any:
    parameters = dict(config.get("parameters") or {})
    execution = dict(config.get("execution_parameters") or {})
    if family == "random_forest":
        model = RandomForestRegressor(**parameters)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family.startswith("gradient_boosting_C0"):
        model = GradientBoostingRegressor(**parameters)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family == "huber":
        model = HuberRegressor(**parameters)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if family == "ridge_C5":
        model = Ridge(**parameters)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if family in {"elastic_net_C5", "elastic_net_C6"}:
        model = ElasticNet(**parameters, **execution)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    raise CleanV2WorkerError(f"No tabular estimator for {family}")


def _sequence_examples(
    panel: pd.DataFrame,
    predictors: Sequence[str],
    *,
    sequence_length: int,
    eligible_mask: pd.Series,
    include_targets: bool,
    maximum_examples: int | None = None,
    reservation_client: FileReservationClient | None = None,
) -> tuple[list[list[list[float]]], list[float], pd.DataFrame]:
    if maximum_examples is not None and maximum_examples <= 0:
        raise CleanV2WorkerError("Sequence example cap must be positive")
    with _allocation_guard(
        reservation_client,
        stage="sequence_candidate_index",
        estimate=tabular_fit_allocation_estimate(
            row_count=len(panel), predictor_count=len(predictors)
        ),
    ):
        work = panel.reset_index(drop=True).copy()
        if len(eligible_mask) != len(work):
            raise CleanV2WorkerError("Sequence eligible-mask length mismatch")
        work["sequence_eligible"] = (
            eligible_mask.reset_index(drop=True).fillna(False).astype(bool).to_numpy()
        )
        work = work.sort_values(["asset_id", "decision_timestamp"], kind="mergesort")
        candidates: list[tuple[str, int]] | deque[tuple[str, int]]
        if maximum_examples is None:
            candidates = []
        else:
            candidates = deque(maxlen=maximum_examples)
        for asset_id, asset in work.groupby("asset_id", sort=False):
            asset = asset.sort_values("decision_timestamp", kind="mergesort")
            eligible = asset["sequence_eligible"].to_numpy(dtype=bool, copy=False)
            for offset, is_eligible in enumerate(eligible):
                if not is_eligible or offset + 1 < sequence_length:
                    continue
                candidates.append((str(asset_id), offset))

    selected = list(candidates)
    with _allocation_guard(
        reservation_client,
        stage="sequence_window_and_tensor_materialization",
        estimate=sequence_allocation_estimate(
            example_count=len(selected),
            sequence_length=sequence_length,
            predictor_count=len(predictors),
        ),
    ):
        selected_offsets: dict[str, list[int]] = {}
        for asset_id, offset in selected:
            selected_offsets.setdefault(asset_id, []).append(offset)

        examples: list[list[list[float]]] = []
        targets: list[float] = []
        metadata: list[dict[str, str]] = []
        for asset_id, asset in work.groupby("asset_id", sort=False):
            offsets = selected_offsets.get(str(asset_id), [])
            if not offsets:
                continue
            asset = asset.sort_values("decision_timestamp", kind="mergesort")
            values = asset[list(predictors)].to_numpy(dtype=np.float32, copy=True)
            for offset in offsets:
                row = asset.iloc[offset]
                window = values[offset - sequence_length + 1 : offset + 1]
                examples.append(window.astype(float).tolist())
                if include_targets:
                    targets.append(float(row["target_value"]))
                metadata.append(
                    {
                        "asset_id": str(row["asset_id"]),
                        "decision_timestamp": pd.Timestamp(
                            row["decision_timestamp"]
                        ).isoformat(),
                    }
                )
    return examples, targets, pd.DataFrame(metadata)


def _rank_labels(train: pd.DataFrame) -> pd.Series:
    ranks = train.groupby("decision_timestamp", sort=False)["target_value"].rank(
        method="first", pct=True
    )
    labels = np.floor(ranks.to_numpy(dtype=float) * 4.999).astype(int)
    return pd.Series(labels, index=train.index, dtype=int)


def _prepare_xendcg_training_inputs(
    train: pd.DataFrame,
    predictors: Sequence[str],
    *,
    maximum_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Reproduce the frozen Mac producer's ranker inputs exactly."""

    if maximum_rows <= 0:
        raise CleanV2WorkerError("XENDCG training-row cap must be positive")
    ordered = train.sort_values(
        ["decision_timestamp", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ordered) > maximum_rows:
        # The recovered producer applies tail(max_training_rows) to rows, not
        # complete query groups. Its retained artifacts prove that the oldest
        # query can therefore be partial; changing this changes the fit.
        ordered = ordered.tail(maximum_rows).reset_index(drop=True)
    matrix = (
        ordered.loc[:, list(predictors)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    labels = _rank_labels(ordered).to_numpy(dtype=int)
    groups = (
        ordered.groupby("decision_timestamp", sort=False).size().astype(int).tolist()
    )
    return matrix, labels, groups


def _prepare_xendcg_scoring_inputs(
    score_panel: pd.DataFrame, predictors: Sequence[str]
) -> pd.DataFrame:
    return (
        score_panel.loc[:, list(predictors)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _fit_xendcg(
    config: Mapping[str, Any],
    train: pd.DataFrame,
    predictors: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise CleanV2WorkerError("LightGBM dependency is unavailable") from exc
    matrix, labels, groups = _prepare_xendcg_training_inputs(
        train,
        predictors,
        maximum_rows=int(config["training"]["max_training_rows"]),
    )
    model = lgb.LGBMRanker(**dict(config["parameters"]))
    model.fit(matrix, labels, group=groups)
    return model, {
        "kind": "ranker",
        "training_rows": len(matrix),
        "training_groups": len(groups),
    }


def _fit_model(
    family: str,
    config: Mapping[str, Any],
    train: pd.DataFrame,
    predictors: Sequence[str],
    reservation_client: FileReservationClient | None = None,
) -> tuple[Any, dict[str, Any]]:
    if family in SEQUENCE_FAMILIES:
        parameters = dict(config["parameters"])
        sequence_config = SequenceRegressorConfig(**parameters)
        mask = pd.Series(True, index=train.index)
        sequences, targets, _ = _sequence_examples(
            train,
            predictors,
            sequence_length=sequence_config.sequence_length,
            eligible_mask=mask,
            include_targets=True,
            maximum_examples=int(config["training"]["max_training_examples"]),
            reservation_client=reservation_client,
        )
        if not sequences:
            raise CleanV2WorkerError(f"No sequence training examples for {family}")
        with _allocation_guard(
            reservation_client,
            stage=f"{family}_fit",
            estimate=sequence_allocation_estimate(
                example_count=len(sequences),
                sequence_length=sequence_config.sequence_length,
                predictor_count=len(predictors),
            ),
        ):
            model = TorchSequenceReturnRegressor(sequence_config)
            model.fit(sequences, targets)
        return model, {
            "kind": "sequence",
            "sequence_length": sequence_config.sequence_length,
            "training_examples": len(sequences),
        }
    if family == "lightgbm_rank_xendcg":
        with _allocation_guard(
            reservation_client,
            stage=f"{family}_fit",
            estimate=tabular_fit_allocation_estimate(
                row_count=min(
                    len(train), int(config["training"]["max_training_rows"])
                ),
                predictor_count=len(predictors),
            ),
        ):
            return _fit_xendcg(config, train, predictors)
    with _allocation_guard(
        reservation_client,
        stage=f"{family}_fit",
        estimate=tabular_fit_allocation_estimate(
            row_count=len(train), predictor_count=len(predictors)
        ),
    ):
        estimator = _tabular_estimator(family, config)
        estimator.fit(train[list(predictors)], train["target_value"].astype(float))
    return estimator, {"kind": "tabular", "training_rows": len(train)}


def _score_model(
    family: str,
    model: Any,
    score_panel: pd.DataFrame,
    full_panel: pd.DataFrame,
    predictors: Sequence[str],
    metadata: Mapping[str, Any],
    reservation_client: FileReservationClient | None = None,
) -> pd.DataFrame:
    if family == "momentum":
        values = pd.to_numeric(score_panel["ret_60m"], errors="coerce").fillna(0.0)
        asset_ids = score_panel["asset_id"].astype(str).reset_index(drop=True)
    elif family == "equal_weight_no_model":
        values = pd.Series(np.zeros(len(score_panel), dtype=float))
        asset_ids = score_panel["asset_id"].astype(str).reset_index(drop=True)
    elif family in SEQUENCE_FAMILIES:
        mask = full_panel["decision_timestamp"].isin(score_panel["decision_timestamp"])
        sequences, _targets, rows = _sequence_examples(
            full_panel,
            predictors,
            sequence_length=int(metadata["sequence_length"]),
            eligible_mask=mask,
            include_targets=False,
            reservation_client=reservation_client,
        )
        with _allocation_guard(
            reservation_client,
            stage=f"{family}_score_predict",
            estimate=sequence_allocation_estimate(
                example_count=len(sequences),
                sequence_length=int(metadata["sequence_length"]),
                predictor_count=len(predictors),
            ),
        ):
            values = pd.Series(model.predict(sequences), dtype=float)
        asset_ids = rows["asset_id"].astype(str)
    elif family == "lightgbm_rank_xendcg":
        with _allocation_guard(
            reservation_client,
            stage=f"{family}_score_matrix",
            estimate=tabular_fit_allocation_estimate(
                row_count=len(score_panel), predictor_count=len(predictors)
            ),
        ):
            values = pd.Series(
                model.predict(_prepare_xendcg_scoring_inputs(score_panel, predictors)),
                dtype=float,
            )
        asset_ids = score_panel["asset_id"].astype(str).reset_index(drop=True)
    else:
        with _allocation_guard(
            reservation_client,
            stage=f"{family}_score_matrix",
            estimate=tabular_fit_allocation_estimate(
                row_count=len(score_panel), predictor_count=len(predictors)
            ),
        ):
            values = pd.Series(
                model.predict(score_panel[list(predictors)]), dtype=float
            )
        asset_ids = score_panel["asset_id"].astype(str).reset_index(drop=True)
    return pd.DataFrame(
        {
            "family": family,
            "decision_timestamp": [
                pd.Timestamp(score_panel["decision_timestamp"].iloc[0]).isoformat()
            ]
            * len(values),
            "asset_id": asset_ids.to_numpy(),
            "prediction": values.to_numpy(dtype=float),
            "eligible": True,
        }
    )


def _save_model(
    family_root: Path,
    *,
    family: str,
    refit_t: pd.Timestamp,
    model: Any,
    identity: Mapping[str, Any],
) -> tuple[Path, str]:
    model_root = family_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    path = model_root / f"{family}_{refit_t.strftime('%Y%m%dT%H%M%SZ')}.pkl"
    temporary = path.with_suffix(".pkl.partial")
    with temporary.open("wb") as handle:
        pickle.dump({"identity": dict(identity), "model": model}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    candidates = sorted(
        model_root.glob(f"{family}_*.pkl"),
        key=lambda candidate: candidate.name,
        reverse=True,
    )
    for stale in candidates[2:]:
        stale.unlink()
    return path, file_sha256(path)


def _load_resume_model(
    checkpoint_path: Path,
    *,
    expected_scientific_identity: Mapping[str, Any],
    current_operational_identity: Mapping[str, Any],
    completed_refits: Sequence[str],
    compatibility_evidence_path: Path,
) -> tuple[Any | None, Path | None, str, dict[str, Any]] | None:
    if not checkpoint_path.is_file():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_identity = checkpoint.get("identity")
    if not isinstance(checkpoint_identity, dict):
        raise CleanV2WorkerError("Resume checkpoint identity is missing")
    try:
        assessment = assess_checkpoint_compatibility(
            checkpoint_identity=checkpoint_identity,
            expected_scientific_identity=expected_scientific_identity,
            current_operational_identity=current_operational_identity,
            completed_refits=completed_refits,
        )
    except CheckpointCompatibilityError as exc:
        raise CleanV2WorkerError(
            f"Resume checkpoint scientific compatibility failed: {exc}"
        ) from exc
    evidence = {
        **assessment.payload(),
        "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
        "model_path": checkpoint.get("model_path"),
        "model_hash": checkpoint.get("model_hash"),
        "artifact_identity_verification": (
            "NOT_REQUIRED_PRIOR_COMPLETED_PACKAGE"
            if assessment.prior_completed_package
            else "PENDING_BEFORE_ARTIFACT_LOAD"
        ),
        "assessed_at_utc": _utc_now(),
    }
    _write_json_atomic(compatibility_evidence_path, evidence)
    if not assessment.reusable:
        return None, None, "", evidence
    path = ROOT / str(checkpoint["model_path"])
    if not path.is_file() or file_sha256(path) != checkpoint.get("model_hash"):
        raise CleanV2WorkerError("Resume model artifact is missing or changed")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if artifact.get("identity") != checkpoint_identity:
        raise CleanV2WorkerError("Resume model artifact identity mismatch")
    verified_evidence = {
        **evidence,
        "artifact_identity_verification": "VERIFIED",
        "verified_at_utc": _utc_now(),
    }
    _write_json_atomic(compatibility_evidence_path, verified_evidence)
    return (
        artifact["model"],
        path,
        str(checkpoint["model_hash"]),
        verified_evidence,
    )


def _resume_model_for_family(
    family: str,
    checkpoint_path: Path,
    *,
    expected_scientific_identity: Mapping[str, Any],
    current_operational_identity: Mapping[str, Any],
    completed_refits: Sequence[str],
    compatibility_evidence_path: Path,
) -> tuple[Any | None, Path | None, str, dict[str, Any]] | None:
    """Load fitted state only for families whose scientific method has a fit."""

    if family in CONTROL_FAMILIES:
        return None
    return _load_resume_model(
        checkpoint_path,
        expected_scientific_identity=expected_scientific_identity,
        current_operational_identity=current_operational_identity,
        completed_refits=completed_refits,
        compatibility_evidence_path=compatibility_evidence_path,
    )


def _completed_timestamps(metrics_root: Path, family: str) -> set[str]:
    metrics = read_parquet_log(metrics_root, "per_t_metrics")
    if metrics.empty:
        return set()
    rows = metrics[metrics["family"].astype(str) == family]
    return set(
        pd.to_datetime(rows["decision_timestamp"], utc=True).map(
            lambda value: value.isoformat()
        )
    )


def _estimated_panel_row_upper_bound(
    *,
    data: CleanV2CompositeData,
    spine: Sequence[pd.Timestamp],
    session_dates: Sequence[str],
    maximum_assets: int | None,
) -> int:
    years = {int(value[:4]) for value in session_dates}
    asset_count = len(
        {partition.asset_id for partition in data.partitions if partition.year in years}
    )
    if maximum_assets is not None:
        asset_count = min(asset_count, max(1, int(maximum_assets)))
    selected_dates = set(session_dates)
    timestamp_count = sum(
        pd.Timestamp(timestamp).date().isoformat() in selected_dates
        for timestamp in spine
    )
    return asset_count * timestamp_count


def _validate_resume_authority(
    *,
    prior: Mapping[str, Any],
    authority: Mapping[str, Any],
    attempt_generation: int,
    control_scientific_identity: Mapping[str, Any] | None = None,
) -> None:
    prior_generation = int(prior.get("attempt_generation", 0) or 0)
    for key, value in authority.items():
        if key == "clean_source_hash" and prior.get(key) != value:
            if attempt_generation <= prior_generation:
                raise CleanV2WorkerError(
                    "Resume source transition requires a newer attempt generation"
                )
            continue
        if (
            key == "clean_source_hash"
            and prior.get(key) is None
            and not prior.get("completed_refits")
            and int(prior.get("metrics_cursor", 0) or 0) == 0
        ):
            continue
        if prior.get(key) != value:
            raise CleanV2WorkerError(f"Resume state authority mismatch: {key}")
    if control_scientific_identity is not None:
        observed = prior.get("control_scientific_identity")
        if not isinstance(observed, Mapping):
            raise CleanV2WorkerError(
                "Resume control scientific identity is missing"
            )
        try:
            validate_no_model_control_resume_identity(
                observed, control_scientific_identity
            )
        except CheckpointCompatibilityError as exc:
            raise CleanV2WorkerError(
                f"Resume control scientific identity mismatch: {exc}"
            ) from exc


def _state_payload(
    *,
    family: str,
    host: str,
    authority: Mapping[str, Any],
    expected_refits: int,
    completed_refits: Sequence[str],
    completed_timestamps: set[str],
    terminal_state: str,
    attempt_generation: int,
    attempt_id: str,
    operational_identity: Mapping[str, Any] | None = None,
    checkpoint_compatibility: Mapping[str, Any] | None = None,
    control_scientific_identity: Mapping[str, Any] | None = None,
    completed_scoring_packages: Sequence[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
    is_control = family in CONTROL_FAMILIES
    payload = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
        **dict(authority),
        "refit_policy_id": REFIT_POLICY_ID,
        "expected_refits": 0 if is_control else expected_refits,
        "completed_refits": [] if is_control else list(completed_refits),
        "latest_completed_refit": (
            None
            if is_control or not completed_refits
            else completed_refits[-1]
        ),
        "latest_scored_decision": max(completed_timestamps) if completed_timestamps else None,
        "metrics_cursor": len(completed_timestamps),
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "current_operational_identity": dict(operational_identity or {}),
        "checkpoint_compatibility": dict(checkpoint_compatibility or {}),
        "terminal_state": terminal_state,
        "error": error,
        "heartbeat_utc": _utc_now(),
        "paper_orders": 0,
        "live_orders": 0,
    }
    if is_control:
        payload.update(
            {
                "expected_scoring_packages": expected_refits,
                "completed_scoring_packages": list(completed_scoring_packages),
                "latest_completed_scoring_package": (
                    completed_scoring_packages[-1]
                    if completed_scoring_packages
                    else None
                ),
                "control_scientific_identity": dict(
                    control_scientific_identity or {}
                ),
                "control_scientific_identity_hash": stable_hash(
                    dict(control_scientific_identity or {})
                ),
            }
        )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    admission = _validate_worker_admission(args.host)
    require_memory(
        stage=f"{args.family}_worker_start",
        estimated_allocation_bytes=0,
    )
    attempt_generation = max(1, int(args.resume_generation))
    reservation_client = _reservation_client(
        family=args.family,
        host=args.host,
        attempt_generation=attempt_generation,
    )
    attempt_id = stable_hash(
        {
            "run_id": RUN_ID,
            "family": args.family,
            "host": args.host,
            "attempt_generation": attempt_generation,
        }
    )
    operational_identity = _operational_checkpoint_identity(
        admission=admission,
        host=args.host,
        attempt_generation=attempt_generation,
        attempt_id=attempt_id,
    )
    config, lane = _family_contract(args.family)
    ownership = load_contract("cross_host_ownership.json")
    if args.family not in ownership["hosts"][args.host]:
        raise CleanV2WorkerError(
            f"{args.family} is not owned by host role {args.host}"
        )
    if args.family not in admission.get("launch_families", []):
        raise CleanV2WorkerError(
            f"{args.family} is outside the admitted {args.host} queue"
        )
    data = CleanV2CompositeData(ROOT)
    feature_hash = str(data.sidecar_manifest["logical_sha256"])
    target_hash = str(data.delta_target_manifest["logical_sha256"])
    model_config_hash = stable_hash(config)
    authority = {
        "feature_authority_hash": feature_hash,
        "target_authority_hash": target_hash,
        "target_contract_hash": data.target_contract["resolved_contract_sha256"],
        "model_config_hash": model_config_hash,
        "clean_source_hash": clean_source_hash(),
        "static_authority_bundle_sha256": authority_bundle()["bundle_sha256"],
        "refit_policy_hash": stable_hash(
            {
                "id": REFIT_POLICY_ID,
                "lookback_sessions": config["training"]["lookback_sessions"],
            }
        ),
    }
    control_scientific_identity = _control_resume_identity(
        family=args.family,
        config=config,
        feature_authority_hash=feature_hash,
        target_authority_hash=target_hash,
        target_contract_hash=str(
            data.target_contract["resolved_contract_sha256"]
        ),
        static_authority_bundle_sha256=str(
            authority["static_authority_bundle_sha256"]
        ),
    )
    spine = data.decision_spine()
    qualifier_years = QUALIFIER_YEARS if lane in SHORT_LANES else None
    full_schedule = build_clean_refit_schedule(
        spine,
        lookback_sessions=int(config["training"]["lookback_sessions"]),
        qualifier_years=qualifier_years,
    )
    schedule = (
        full_schedule[: args.maximum_refits]
        if args.maximum_refits is not None
        else full_schedule
    )
    schedule_contract, schedule_hash = _schedule_identity(
        schedule=full_schedule,
        lane=lane,
        lookback_sessions=int(config["training"]["lookback_sessions"]),
        qualifier_years=qualifier_years,
    )
    family_root = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID / f"family={args.family}"
    metrics_root = family_root / "metrics"
    state_path = family_root / "resume_state.json"
    checkpoint_path = family_root / "model_checkpoint.json"
    completed = _completed_timestamps(metrics_root, args.family)
    completed_refits: list[str] = []
    completed_scoring_packages: list[str] = []
    latest_checkpoint_compatibility: dict[str, Any] = (
        {
            "classification": (
                "NO_MODEL_CONTROL_FITTED_CHECKPOINT_NOT_APPLICABLE"
            ),
            "fitted_model_required": False,
            "checkpoint_required": False,
        }
        if args.family in CONTROL_FAMILIES
        else {}
    )
    if state_path.is_file():
        prior = _read_runtime_json(state_path)
        _validate_resume_authority(
            prior=prior,
            authority=authority,
            attempt_generation=attempt_generation,
            control_scientific_identity=control_scientific_identity,
        )
        completed_refits = list(dict.fromkeys(prior.get("completed_refits", [])))
        completed_scoring_packages = list(
            dict.fromkeys(prior.get("completed_scoring_packages", []))
        )
        latest_checkpoint_compatibility = dict(
            prior.get("checkpoint_compatibility")
            or latest_checkpoint_compatibility
        )
    _write_json_atomic(
        state_path,
        _state_payload(
            family=args.family,
            host=args.host,
            authority=authority,
            expected_refits=len(schedule),
            completed_refits=completed_refits,
            completed_timestamps=completed,
            terminal_state="RUNNING",
            attempt_generation=attempt_generation,
            attempt_id=attempt_id,
            operational_identity=operational_identity,
            checkpoint_compatibility=latest_checkpoint_compatibility,
            control_scientific_identity=control_scientific_identity,
            completed_scoring_packages=completed_scoring_packages,
        ),
    )

    for package in schedule:
        score_timestamps = [
            timestamp
            for timestamp in spine
            if timestamp.date().isoformat() in package.score_session_dates
        ]
        if _package_score_is_complete(score_timestamps, completed):
            refit_iso = package.refit_T.isoformat()
            if (
                args.family in CONTROL_FAMILIES
                and refit_iso not in completed_scoring_packages
            ):
                completed_scoring_packages.append(refit_iso)
            elif args.family not in CONTROL_FAMILIES and refit_iso not in completed_refits:
                completed_refits.append(refit_iso)
            continue
        dates = [*package.training_session_dates, *package.score_session_dates]
        estimated_panel_rows = _estimated_panel_row_upper_bound(
            data=data,
            spine=spine,
            session_dates=dates,
            maximum_assets=args.maximum_assets,
        )
        with _allocation_guard(
            reservation_client,
            stage=f"{args.family}_panel_assembly",
            estimate=panel_allocation_estimate(
                row_count=estimated_panel_rows,
                column_count=len(data.predictors) + 12,
            ),
        ):
            panel = data.assemble_sessions(dates, maximum_assets=args.maximum_assets)
        if panel.empty:
            raise CleanV2WorkerError(f"Empty refit package: {package.refit_T}")
        with _allocation_guard(
            reservation_client,
            stage=f"{args.family}_training_and_scoring_population_slices",
            estimate=panel_allocation_estimate(
                row_count=len(panel),
                column_count=len(panel.columns),
            ),
        ):
            training = panel[
                panel["session_date"].astype(str).isin(package.training_session_dates)
                & (panel["decision_timestamp"] < package.refit_T)
                & panel["target_is_trainable"].fillna(False).astype(bool)
                & (panel["target_available_timestamp"] <= package.refit_T)
                & np.isfinite(pd.to_numeric(panel["target_value"], errors="coerce"))
            ].copy()
            scoring = panel[
                panel["session_date"].astype(str).isin(package.score_session_dates)
            ].copy()
        validate_target_use(
            predictor_names=data.predictors,
            training_rows=training,
            scoring_rows=scoring,
            fit_timestamp=package.refit_T,
        )
        if training.empty or scoring.empty:
            raise CleanV2WorkerError(f"Empty train/score population: {package.refit_T}")

        scientific_identity = _scientific_checkpoint_identity(
            family=args.family,
            config=config,
            predictors=data.predictors,
            feature_authority_hash=feature_hash,
            target_authority_hash=target_hash,
            target_contract_hash=data.target_contract[
                "resolved_contract_sha256"
            ],
            static_authority_bundle_sha256=authority[
                "static_authority_bundle_sha256"
            ],
            schedule_contract=schedule_contract,
            schedule_hash=schedule_hash,
            package=package,
        )
        checkpoint_identity = make_checkpoint_identity(
            scientific_identity=scientific_identity,
            operational_identity=operational_identity,
        )
        compatibility_evidence_path = family_root / (
            "checkpoint_compatibility_"
            f"attempt={attempt_generation}_"
            f"refit={package.refit_T.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        resumed = _resume_model_for_family(
            args.family,
            checkpoint_path,
            expected_scientific_identity=scientific_identity,
            current_operational_identity=operational_identity,
            completed_refits=completed_refits,
            compatibility_evidence_path=compatibility_evidence_path,
        )
        fit_metadata: dict[str, Any]
        if args.family in CONTROL_FAMILIES:
            model, model_path, model_hash = (
                None,
                None,
                stable_hash(checkpoint_identity),
            )
            fit_metadata = {"kind": "control", "training_rows": 0}
        elif resumed is not None and resumed[0] is not None:
            model, model_path, model_hash, latest_checkpoint_compatibility = resumed
            fit_metadata = {
                "kind": "resumed_model",
                "checkpoint_compatibility_classification": (
                    latest_checkpoint_compatibility["classification"]
                ),
            }
        else:
            if resumed is not None:
                latest_checkpoint_compatibility = resumed[3]
            started = time.perf_counter()
            model, fit_metadata = _fit_model(
                args.family,
                config,
                training,
                data.predictors,
                reservation_client=reservation_client,
            )
            fit_metadata["fit_wall_seconds"] = round(
                time.perf_counter() - started, 6
            )
            model_path, model_hash = _save_model(
                family_root,
                family=args.family,
                refit_t=package.refit_T,
                model=model,
                identity=checkpoint_identity,
            )
            _write_json_atomic(
                checkpoint_path,
                {
                    "identity": checkpoint_identity,
                    "model_path": model_path.relative_to(ROOT).as_posix(),
                    "model_hash": model_hash,
                    "fit_metadata": fit_metadata,
                    "published_at_utc": _utc_now(),
                },
            )

        require_memory(
            stage=f"{args.family}_post_fit_committed_checkpoint_boundary",
            estimated_allocation_bytes=0,
        )

        writer = MetricsOnlyEvidenceWriter(
            metrics_root,
            family=args.family,
            top_n=10_000 if args.family == "equal_weight_no_model" else 20,
            enable_resolved_performance_v3=True,
            target_loader=lambda request, source=panel: data.target_loader_from_panel(
                source, request
            ),
            terminal_timestamp=max(spine).isoformat(),
            namespace_lease_enabled=False,
            resume_generation=args.resume_generation,
            command_hash=stable_hash(" ".join(sys.argv)),
            configuration_hash=model_config_hash,
            evaluation_contract_hash=resolved_performance_contract_v3_hash(),
        )
        preprocessing_hash = stable_hash(
            {
                "family": args.family,
                "predictors": list(data.predictors),
                "config": config.get("preprocessing", []),
                "kind": fit_metadata["kind"],
            }
        )
        for timestamp in score_timestamps:
            timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
            if timestamp.isoformat() in completed:
                continue
            score_at_t = scoring[scoring["decision_timestamp"] == timestamp].copy()
            if score_at_t.empty:
                raise CleanV2WorkerError(f"Empty score timestamp: {timestamp}")
            predictions = _score_model(
                args.family,
                model,
                score_at_t,
                panel,
                data.predictors,
                fit_metadata,
                reservation_client=reservation_client,
            )
            targets = score_at_t[
                [
                    "asset_id",
                    "decision_timestamp",
                    "target_value",
                    "target_available_timestamp",
                ]
            ]
            writer.commit_predictions(
                predictions,
                targets=targets,
                expected_assets=sorted(score_at_t["asset_id"].astype(str).unique()),
                metadata={
                    "model_hash": model_hash,
                    "model_vintage_id": stable_hash(
                        {
                            "family": args.family,
                            "refit_T": package.refit_T.isoformat(),
                            "model_hash": model_hash,
                        }
                    ),
                    "preprocessing_hash": preprocessing_hash,
                    "policy_hash": package.policy_hash,
                    "training_cutoff": package.refit_T.isoformat(),
                    "prediction_timestamp": _utc_now(),
                    "refit_policy": REFIT_POLICY_ID,
                },
            )
            completed.add(timestamp.isoformat())
            _write_json_atomic(
                state_path,
                _state_payload(
                    family=args.family,
                    host=args.host,
                    authority=authority,
                    expected_refits=len(schedule),
                    completed_refits=completed_refits,
                    completed_timestamps=completed,
                    terminal_state="RUNNING",
                    attempt_generation=attempt_generation,
                    attempt_id=attempt_id,
                    operational_identity=operational_identity,
                    checkpoint_compatibility=latest_checkpoint_compatibility,
                    control_scientific_identity=control_scientific_identity,
                    completed_scoring_packages=completed_scoring_packages,
                ),
            )
        refit_iso = package.refit_T.isoformat()
        if (
            args.family in CONTROL_FAMILIES
            and refit_iso not in completed_scoring_packages
        ):
            completed_scoring_packages.append(refit_iso)
        elif args.family not in CONTROL_FAMILIES and refit_iso not in completed_refits:
            completed_refits.append(refit_iso)
        _write_json_atomic(
            state_path,
            _state_payload(
                family=args.family,
                host=args.host,
                authority=authority,
                expected_refits=len(schedule),
                completed_refits=completed_refits,
                completed_timestamps=completed,
                terminal_state="RUNNING",
                attempt_generation=attempt_generation,
                attempt_id=attempt_id,
                operational_identity=operational_identity,
                checkpoint_compatibility=latest_checkpoint_compatibility,
                control_scientific_identity=control_scientific_identity,
                completed_scoring_packages=completed_scoring_packages,
            ),
        )
        model = None
        writer = None
        training = pd.DataFrame()
        scoring = pd.DataFrame()
        panel = pd.DataFrame()
        predictions = None
        targets = None
        score_at_t = None
        _release_disposable_caches()

    final = _state_payload(
        family=args.family,
        host=args.host,
        authority=authority,
        expected_refits=len(schedule),
        completed_refits=completed_refits,
        completed_timestamps=completed,
        terminal_state="COMPLETE",
        attempt_generation=attempt_generation,
        attempt_id=attempt_id,
        operational_identity=operational_identity,
        checkpoint_compatibility=latest_checkpoint_compatibility,
        control_scientific_identity=control_scientific_identity,
        completed_scoring_packages=completed_scoring_packages,
    )
    _write_json_atomic(state_path, final)
    return final


def _record_worker_failure(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    family_root = (
        ROOT
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / f"family={args.family}"
    )
    state_path = family_root / "resume_state.json"
    prior = (
        _read_runtime_json(state_path)
        if state_path.is_file()
        else {}
    )
    attempt_generation = max(1, int(getattr(args, "resume_generation", 1)))
    attempt_id = stable_hash(
        {
            "run_id": RUN_ID,
            "family": args.family,
            "host": args.host,
            "attempt_generation": attempt_generation,
        }
    )
    failed_at = _utc_now()
    error = f"{type(exc).__name__}:{exc}"
    log_path = (
        ROOT
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / "logs"
        / args.host
        / f"{args.family}.log"
    )
    relative_log_path = log_path.relative_to(ROOT).as_posix()
    failure = {
        "run_id": RUN_ID,
        "family": args.family,
        "host": args.host,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "classification": "DS24_CLEAN_V2_WORKER_FAILED_CLOSED",
        "terminal_state": "FAILED_CLOSED",
        "error": error,
        "failed_at_utc": failed_at,
        "log_path": relative_log_path,
        "completed_refits": list(prior.get("completed_refits", [])),
        "metrics_cursor": int(prior.get("metrics_cursor", 0) or 0),
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_json_atomic(
        family_root / f"worker_failure_attempt={attempt_generation}.json",
        failure,
    )

    state_identity_matches = not prior or (
        prior.get("run_id") == RUN_ID
        and prior.get("family") == args.family
        and prior.get("owner_host") == args.host
    )
    prior_generation = int(prior.get("attempt_generation", 0) or 0)
    stale_attempt = prior_generation > attempt_generation or (
        prior_generation >= attempt_generation
        and prior.get("terminal_state") == "COMPLETE"
    )
    if state_identity_matches and not stale_attempt:
        _write_json_atomic(family_root / "worker_failure.json", failure)
        failed_state = {
            **prior,
            "run_id": RUN_ID,
            "family": args.family,
            "owner_host": args.host,
            "attempt_generation": attempt_generation,
            "attempt_id": attempt_id,
            "terminal_state": "FAILED_CLOSED",
            "terminal_failure_classification": failure["classification"],
            "error": error,
            "failure_timestamp": failed_at,
            "failure_log_path": relative_log_path,
            "completed_refits": list(prior.get("completed_refits", [])),
            "metrics_cursor": int(prior.get("metrics_cursor", 0) or 0),
            "heartbeat_utc": failed_at,
            "paper_orders": 0,
            "live_orders": 0,
        }
        _write_json_atomic(state_path, failed_state)
        failure["current_failure_pointer_updated"] = True
    else:
        failure["current_failure_pointer_updated"] = False
    return failure


def _record_resource_pause(
    args: argparse.Namespace,
    exc: CleanV2ResourcePressure | ResourceReservationUnavailable,
    *,
    classification: str = "DS24_CLEAN_V2_PAUSED_RESOURCE_PRESSURE",
) -> dict[str, Any]:
    family_root = (
        ROOT
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / f"family={args.family}"
    )
    state_path = family_root / "resume_state.json"
    prior = (
        _read_runtime_json(state_path)
        if state_path.is_file()
        else {}
    )
    attempt_generation = max(1, int(getattr(args, "resume_generation", 1)))
    paused_at = _utc_now()
    evidence = {
        "run_id": RUN_ID,
        "family": args.family,
        "host": args.host,
        "attempt_generation": attempt_generation,
        "terminal_state": "PAUSED_RESOURCE_PRESSURE",
        "classification": classification,
        "paused_at_utc": paused_at,
        "resource_decision": (
            exc.decision.payload()
            if isinstance(exc, CleanV2ResourcePressure)
            else {
                "stage": exc.stage,
                **exc.response,
                "resource_policy": recovery_policy_payload(),
            }
        ),
        "completed_refits": list(prior.get("completed_refits", [])),
        "metrics_cursor": int(prior.get("metrics_cursor", 0) or 0),
        "automatic_retry": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_json_atomic(
        family_root / f"resource_pause_attempt={attempt_generation}.json",
        evidence,
    )
    prior_generation = int(prior.get("attempt_generation", 0) or 0)
    if prior_generation <= attempt_generation and prior.get("terminal_state") != "COMPLETE":
        _write_json_atomic(
            state_path,
            {
                **prior,
                "run_id": RUN_ID,
                "family": args.family,
                "owner_host": args.host,
                "attempt_generation": attempt_generation,
                "terminal_state": "PAUSED_RESOURCE_PRESSURE",
                "resource_pause_classification": evidence["classification"],
                "resource_pause_timestamp": paused_at,
                "resource_decision": evidence["resource_decision"],
                "completed_refits": evidence["completed_refits"],
                "metrics_cursor": evidence["metrics_cursor"],
                "heartbeat_utc": paused_at,
                "paper_orders": 0,
                "live_orders": 0,
            },
        )
    return evidence


def _record_capacity_deferral(
    args: argparse.Namespace,
    exc: ResourceReservationUnavailable,
) -> dict[str, Any]:
    """Persist a scheduling deferral without manufacturing a worker failure."""

    resource_decision = {
        "stage": exc.stage,
        **exc.response,
        "resource_policy": recovery_policy_payload(),
    }
    if not reservation_payload_is_local_capacity_deferral(resource_decision):
        raise CleanV2WorkerError(
            "Non-capacity reservation denial cannot be recorded as a deferral"
        )
    family_root = (
        ROOT
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / f"family={args.family}"
    )
    state_path = family_root / "resume_state.json"
    prior = (
        _read_runtime_json(state_path)
        if state_path.is_file()
        else {}
    )
    attempt_generation = max(1, int(getattr(args, "resume_generation", 1)))
    deferred_at = _utc_now()
    evidence = {
        "run_id": RUN_ID,
        "family": args.family,
        "host": args.host,
        "attempt_generation": attempt_generation,
        "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
        "classification": "DS24_CLEAN_V2_DEFERRED_RESOURCE_CAPACITY",
        "deferred_at_utc": deferred_at,
        "resource_decision": resource_decision,
        "completed_refits": list(prior.get("completed_refits", [])),
        "metrics_cursor": int(prior.get("metrics_cursor", 0) or 0),
        "automatic_retry": False,
        "scheduler_readmission_requires_material_capacity_change": True,
        "paper_orders": 0,
        "live_orders": 0,
    }
    event_path = family_root / (
        f"resource_deferral_attempt={attempt_generation}_event={time.time_ns()}.json"
    )
    _write_json_atomic(event_path, evidence)
    _write_json_atomic(family_root / "resource_deferral.json", evidence)
    prior_generation = int(prior.get("attempt_generation", 0) or 0)
    if prior_generation <= attempt_generation and prior.get("terminal_state") != "COMPLETE":
        _write_json_atomic(
            state_path,
            {
                **prior,
                "run_id": RUN_ID,
                "family": args.family,
                "owner_host": args.host,
                "attempt_generation": attempt_generation,
                "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
                "resource_deferral_classification": evidence["classification"],
                "resource_deferral_timestamp": deferred_at,
                "resource_decision": resource_decision,
                "completed_refits": evidence["completed_refits"],
                "metrics_cursor": evidence["metrics_cursor"],
                "heartbeat_utc": deferred_at,
                "paper_orders": 0,
                "live_orders": 0,
            },
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DS24 clean-V2 metrics-only family worker."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--host", choices=("dell", "mac"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-generation", type=int, default=1)
    parser.add_argument("--maximum-refits", type=int)
    parser.add_argument("--maximum-assets", type=int)
    args = parser.parse_args()
    try:
        result = run(args)
    except ResourceReservationUnavailable as exc:
        _release_disposable_caches()
        if reservation_payload_is_local_capacity_deferral(exc.response):
            result = _record_capacity_deferral(args, exc)
            print(json.dumps(result, indent=2, sort_keys=True))
            return RESOURCE_CAPACITY_DEFERRED_EXIT_CODE
        result = _record_resource_pause(args, exc)
        print(json.dumps(result, indent=2, sort_keys=True))
        return RESOURCE_PRESSURE_EXIT_CODE
    except CleanV2ResourcePressure as exc:
        _release_disposable_caches()
        result = _record_resource_pause(args, exc)
        print(json.dumps(result, indent=2, sort_keys=True))
        return RESOURCE_PRESSURE_EXIT_CODE
    except Exception as exc:
        if is_resource_allocation_failure(exc):
            _release_disposable_caches()
            pressure = CleanV2ResourcePressure(
                allocation_failure_decision(
                    stage=f"{args.family}_allocation_failure",
                    reason=(
                        "WINDOWS_JOB_OR_SYSTEM_COMMITTED_MEMORY_ALLOCATION_FAILED"
                    ),
                )
            )
            result = _record_resource_pause(
                args,
                pressure,
                classification="DS24_CLEAN_V2_PAUSED_RESOURCE_LIMIT",
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return RESOURCE_PRESSURE_EXIT_CODE
        _record_worker_failure(args, exc)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
