from __future__ import annotations

import argparse
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
    file_sha256,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData
from core.research.ml.ds24.clean_v2_runtime import (
    QUALIFIER_YEARS,
    REFIT_POLICY_ID,
    RUN_ID,
    validate_target_use,
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
    return {**admission, "admission_token": token}


def _utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
        refit_t = min(
            timestamp
            for timestamp in timestamps
            if timestamp.date().isoformat() == score_sessions[0]
        )
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


def _family_contract(family: str) -> tuple[dict[str, Any], str]:
    models = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    if family in models["families"]:
        config = dict(models["families"][family])
    elif family in models["controls"]:
        config = {
            "lane": "CONTROLS",
            "status": "FIXED_NO_FIT_CONTROL",
            "training": {
                "lookback_sessions": 20,
                "refit_policy": REFIT_POLICY_ID,
            },
            "parameters": {},
        }
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
) -> tuple[list[list[list[float]]], list[float], pd.DataFrame]:
    work = panel.reset_index(drop=True).copy()
    if len(eligible_mask) != len(work):
        raise CleanV2WorkerError("Sequence eligible-mask length mismatch")
    work["sequence_eligible"] = (
        eligible_mask.reset_index(drop=True).fillna(False).astype(bool).to_numpy()
    )
    work = work.sort_values(["asset_id", "decision_timestamp"], kind="mergesort")
    examples: list[list[list[float]]] = []
    targets: list[float] = []
    metadata: list[dict[str, str]] = []
    for _asset_id, asset in work.groupby("asset_id", sort=False):
        asset = asset.sort_values("decision_timestamp", kind="mergesort")
        values = asset[list(predictors)].to_numpy(dtype=np.float32, copy=True)
        rows = list(asset.itertuples(index=False))
        for offset, row in enumerate(rows):
            if not bool(getattr(row, "sequence_eligible")) or offset + 1 < sequence_length:
                continue
            window = values[offset - sequence_length + 1 : offset + 1]
            examples.append(window.astype(float).tolist())
            if include_targets:
                targets.append(float(getattr(row, "target_value")))
            metadata.append(
                {
                    "asset_id": str(getattr(row, "asset_id")),
                    "decision_timestamp": pd.Timestamp(
                        getattr(row, "decision_timestamp")
                    ).isoformat(),
                }
            )
    if maximum_examples and len(examples) > maximum_examples:
        examples = examples[-maximum_examples:]
        targets = targets[-maximum_examples:] if include_targets else targets
        metadata = metadata[-maximum_examples:]
    return examples, targets, pd.DataFrame(metadata)


def _rank_labels(train: pd.DataFrame) -> pd.Series:
    ranks = train.groupby("decision_timestamp", sort=False)["target_value"].rank(
        method="first", pct=True
    )
    return np.minimum(np.ceil(ranks * 5.0).astype(int) - 1, 4).clip(lower=0)


def _fit_model(
    family: str,
    config: Mapping[str, Any],
    train: pd.DataFrame,
    predictors: Sequence[str],
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
        )
        if not sequences:
            raise CleanV2WorkerError(f"No sequence training examples for {family}")
        model = TorchSequenceReturnRegressor(sequence_config)
        model.fit(sequences, targets)
        return model, {
            "kind": "sequence",
            "sequence_length": sequence_config.sequence_length,
            "training_examples": len(sequences),
        }
    if family == "lightgbm_rank_xendcg":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise CleanV2WorkerError("LightGBM dependency is unavailable") from exc
        ordered = train.sort_values(
            ["decision_timestamp", "asset_id"], kind="mergesort"
        ).copy()
        cap = int(config["training"]["max_training_rows"])
        if len(ordered) > cap:
            keep_timestamps: list[pd.Timestamp] = []
            row_count = 0
            for timestamp, group in reversed(
                list(ordered.groupby("decision_timestamp", sort=True))
            ):
                if keep_timestamps and row_count + len(group) > cap:
                    break
                keep_timestamps.append(pd.Timestamp(timestamp))
                row_count += len(group)
            ordered = ordered[
                ordered["decision_timestamp"].isin(keep_timestamps)
            ].copy()
        imputer = SimpleImputer(strategy="median")
        matrix = imputer.fit_transform(ordered[list(predictors)])
        labels = _rank_labels(ordered).to_numpy(dtype=int)
        groups = (
            ordered.groupby("decision_timestamp", sort=False).size().astype(int).tolist()
        )
        model = lgb.LGBMRanker(**dict(config["parameters"]))
        model.fit(matrix, labels, group=groups, eval_at=config["parameters"]["eval_at"])
        return (imputer, model), {
            "kind": "ranker",
            "training_rows": len(ordered),
            "training_groups": len(groups),
        }
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
        )
        values = pd.Series(model.predict(sequences), dtype=float)
        asset_ids = rows["asset_id"].astype(str)
    elif family == "lightgbm_rank_xendcg":
        imputer, ranker = model
        values = pd.Series(
            ranker.predict(imputer.transform(score_panel[list(predictors)])),
            dtype=float,
        )
        asset_ids = score_panel["asset_id"].astype(str).reset_index(drop=True)
    else:
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
    checkpoint_path: Path, expected_identity: Mapping[str, Any]
) -> tuple[Any, Path, str] | None:
    if not checkpoint_path.is_file():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("identity") != dict(expected_identity):
        raise CleanV2WorkerError("Resume checkpoint authority identity mismatch")
    path = ROOT / str(checkpoint["model_path"])
    if not path.is_file() or file_sha256(path) != checkpoint.get("model_hash"):
        raise CleanV2WorkerError("Resume model artifact is missing or changed")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if artifact.get("identity") != dict(expected_identity):
        raise CleanV2WorkerError("Resume model artifact identity mismatch")
    return artifact["model"], path, str(checkpoint["model_hash"])


def _completed_timestamps(metrics_root: Path, family: str) -> set[str]:
    metrics = read_parquet_log(metrics_root, "per_t_metrics")
    if metrics.empty:
        return set()
    rows = metrics[metrics["family"].astype(str) == family]
    return set(pd.to_datetime(rows["decision_timestamp"], utc=True).map(lambda x: x.isoformat()))


def _state_payload(
    *,
    family: str,
    host: str,
    authority: Mapping[str, Any],
    expected_refits: int,
    completed_refits: Sequence[str],
    completed_timestamps: set[str],
    terminal_state: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
        **dict(authority),
        "refit_policy_id": REFIT_POLICY_ID,
        "expected_refits": expected_refits,
        "completed_refits": list(completed_refits),
        "latest_completed_refit": completed_refits[-1] if completed_refits else None,
        "latest_scored_decision": max(completed_timestamps) if completed_timestamps else None,
        "metrics_cursor": len(completed_timestamps),
        "terminal_state": terminal_state,
        "error": error,
        "heartbeat_utc": _utc_now(),
        "paper_orders": 0,
        "live_orders": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    admission = _validate_worker_admission(args.host)
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
        "static_authority_bundle_sha256": authority_bundle()["bundle_sha256"],
        "refit_policy_hash": stable_hash(
            {
                "id": REFIT_POLICY_ID,
                "lookback_sessions": config["training"]["lookback_sessions"],
            }
        ),
    }
    spine = data.decision_spine()
    qualifier_years = QUALIFIER_YEARS if lane in SHORT_LANES else None
    schedule = build_clean_refit_schedule(
        spine,
        lookback_sessions=int(config["training"]["lookback_sessions"]),
        qualifier_years=qualifier_years,
        maximum_refits=args.maximum_refits,
    )
    family_root = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID / f"family={args.family}"
    metrics_root = family_root / "metrics"
    state_path = family_root / "resume_state.json"
    checkpoint_path = family_root / "model_checkpoint.json"
    completed = _completed_timestamps(metrics_root, args.family)
    completed_refits: list[str] = []
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in authority.items():
            if prior.get(key) != value:
                raise CleanV2WorkerError(f"Resume state authority mismatch: {key}")
        completed_refits = list(prior.get("completed_refits", []))
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
        ),
    )

    for package in schedule:
        score_timestamps = [
            timestamp
            for timestamp in spine
            if timestamp.date().isoformat() in package.score_session_dates
        ]
        score_keys = {timestamp.isoformat() for timestamp in score_timestamps}
        if score_keys and score_keys <= completed:
            refit_iso = package.refit_T.isoformat()
            if refit_iso not in completed_refits:
                completed_refits.append(refit_iso)
            continue
        dates = [*package.training_session_dates, *package.score_session_dates]
        panel = data.assemble_sessions(dates, maximum_assets=args.maximum_assets)
        if panel.empty:
            raise CleanV2WorkerError(f"Empty refit package: {package.refit_T}")
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

        identity = {
            **authority,
            "family": args.family,
            "refit_T": package.refit_T.isoformat(),
            "policy_hash": package.policy_hash,
        }
        resumed = _load_resume_model(checkpoint_path, identity)
        fit_metadata: dict[str, Any]
        if args.family in CONTROL_FAMILIES:
            model, model_path, model_hash = None, None, stable_hash(identity)
            fit_metadata = {"kind": "control", "training_rows": 0}
        elif resumed is not None:
            model, model_path, model_hash = resumed
            fit_metadata = {"kind": "resumed_model"}
        else:
            started = time.perf_counter()
            model, fit_metadata = _fit_model(
                args.family, config, training, data.predictors
            )
            fit_metadata["fit_wall_seconds"] = round(
                time.perf_counter() - started, 6
            )
            model_path, model_hash = _save_model(
                family_root,
                family=args.family,
                refit_t=package.refit_T,
                model=model,
                identity=identity,
            )
            _write_json_atomic(
                checkpoint_path,
                {
                    "identity": identity,
                    "model_path": model_path.relative_to(ROOT).as_posix(),
                    "model_hash": model_hash,
                    "fit_metadata": fit_metadata,
                    "published_at_utc": _utc_now(),
                },
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
                ),
            )
        refit_iso = package.refit_T.isoformat()
        if refit_iso not in completed_refits:
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
            ),
        )

    final = _state_payload(
        family=args.family,
        host=args.host,
        authority=authority,
        expected_refits=len(schedule),
        completed_refits=completed_refits,
        completed_timestamps=completed,
        terminal_state="COMPLETE",
    )
    _write_json_atomic(state_path, final)
    return final


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
    except Exception as exc:
        family_root = (
            ROOT
            / "research_runs"
            / "ds24_clean_v2"
            / RUN_ID
            / f"family={args.family}"
        )
        _write_json_atomic(
            family_root / "worker_failure.json",
            {
                "run_id": RUN_ID,
                "family": args.family,
                "host": args.host,
                "classification": "DS24_CLEAN_V2_WORKER_FAILED_CLOSED",
                "error": f"{type(exc).__name__}:{exc}",
                "failed_at_utc": _utc_now(),
                "paper_orders": 0,
                "live_orders": 0,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
