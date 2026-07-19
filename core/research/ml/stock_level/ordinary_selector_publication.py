from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.artifact_lineage import VERIFIED_STRICT_OOS, build_artifact_link, verify_selector_artifact
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash
from core.research.ml.experiment_ledger import require_selector_experiment
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level_benchmark_models import _build_tabular_model
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    publish_selector_sklearn_model_package,
)


PUBLICATION_CONTRACT_VERSION = "ordinary_selector_publication.v1"
COMPONENT_SCHEMA_VERSION = "authoritative_selector_component_v1"
SUPPORTED_MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")


def publish_planned_ordinary_component(
    *,
    job: Mapping[str, Any],
    parent_gate_path: Path,
    training_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    ledger_path: Path,
    random_seed: int = 42,
    sklearn_n_jobs: int = 1,
    campaign_identity: str | None = None,
    plan_job_identity: str | None = None,
    declared_component_runner: str | None = None,
    resolved_runtime_owner: str | None = None,
    operational_input_identity: str | None = None,
) -> dict[str, Any]:
    if job.get("experiment_id"):
        require_selector_experiment(ledger_path, job, required_status="SUCCEEDED")
    gate = _json(parent_gate_path)
    model_id = str(job.get("model_id", ""))
    prediction_date = str(job.get("prediction_date", ""))
    owner = Path(str(job.get("authoritative_output_root", "")))
    _validate_job(job, gate, model_id, prediction_date)
    bundle = load_registry_bundle(); resolver = RegistryResolver(bundle)
    model_resolution = resolver.resolve("selector_models", model_id, role="selector")
    model_payload = model_resolution.entry.payload
    if not model_payload.get("ordinary_runner_support"):
        raise ValueError(f"Registry ordinary capability is disabled for {model_id}")
    target = resolver.resolve("target_contracts", str(job["target_contract"]), role="selector")
    model_target = resolver.resolve(
        "target_contracts", str(model_payload.get("target_contract")), role="selector"
    )
    if target.canonical_id != model_target.canonical_id:
        raise ValueError("Target contract mismatch")
    expected_feature = str(model_payload["feature_schema"])
    if str(job["feature_schema"]) != expected_feature:
        raise ValueError("Feature-schema mismatch")
    if model_payload.get("ranking_problem_contract") != job.get("ranking_contract"):
        raise ValueError("Ranking-contract mismatch")

    runner_evidence = {
        "campaign_identity": campaign_identity or "legacy_direct_ordinary.v1",
        "plan_job_identity": plan_job_identity or str(job["job_id"]),
        "declared_component_runner": declared_component_runner or (
            "core.research.ml.stock_level_benchmark_models:"
            "TabularModelFactory/SequenceModelFactory"
        ),
        "resolved_runtime_owner": resolved_runtime_owner or (
            "core.research.ml.stock_level.ordinary_selector_publication:"
            "publish_planned_ordinary_component"
        ),
        "operational_input_identity": operational_input_identity or (
            f"legacy-plan-job:{job['logical_checksum']}"
        ),
    }
    identity = _identity(
        job, gate, model_resolution.entry.entry_hash,
        target.entry.entry_hash, runner_evidence,
    )
    existing = _compatible(owner, identity)
    run_id = f"ordinary-{canonical_hash(identity)[:20].lower()}"
    spec_hash = experiment_spec_hash(identity)
    if existing:
        _ledger(ledger_path, spec_hash, run_id, model_id, model_resolution.entry.entry_hash,
                target.entry.entry_hash, "SKIPPED_COMPLETE", (str(owner),), metadata={"result_status": "SKIPPED_COMPATIBLE"})
        return {"status": "SKIPPED_COMPATIBLE", "manifest_path": str(owner / "manifest.json")}
    if owner.exists():
        prior = None
        try:
            prior = _json(owner / "manifest.json")
        except (OSError, ValueError):
            prior = None
        if (
            prior
            and prior.get("publication_status") == "complete"
            and job.get("overwrite_policy") == "never_overwrite_complete_component"
        ):
            raise FileExistsError(f"Incompatible complete component: {owner}")
        if job.get("resume_policy") != "resume_only_incomplete_owned_component":
            raise FileExistsError(f"Incomplete component replacement is not permitted: {owner}")
        shutil.rmtree(owner)

    _ledger(ledger_path, spec_hash, run_id, model_id, model_resolution.entry.entry_hash,
            target.entry.entry_hash, "STARTED", ())
    temp: Path | None = None
    try:
        feature_names = _feature_names(Path(expected_feature))
        train, predict = _validate_rows(
            training_rows, prediction_rows, prediction_date, feature_names
        )
        x_train = [[float(row[name]) for name in feature_names] for row in train]
        x_predict = [[float(row[name]) for name in feature_names] for row in predict]
        y_train = [float(row["actual_forward_return_10d"]) for row in train]
        model = _build_tabular_model(model_id, random_seed, sklearn_n_jobs)
        if model_id == "ordered_logit_ranker":
            model.fit(
                x_train, y_train,
                groups=[str(row["decision_timestamp"]) for row in train],
                row_ids=[str(row["row_id"]) for row in train],
            )
        else:
            model.fit(x_train, y_train)
        scores = [float(value) for value in model.predict(x_predict)]
        if len(scores) != len(predict) or not all(math.isfinite(value) for value in scores):
            raise ValueError("Incomplete or nonfinite prediction population")
        probabilities = model.predict_proba(x_predict) if model_id == "ordered_logit_ranker" else None
        output_rows = _prediction_output(
            predict, scores, probabilities, model_id, prediction_date,
            gate, identity,
        )
        temp = owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp")
        temp.mkdir(parents=True, exist_ok=False)
        prediction_path = temp / "predictions.csv"
        _write_csv(prediction_path, output_rows)
        prediction_checksum = _sha256(prediction_path)
        population_checksum = canonical_hash([row["row_id"] for row in output_rows])
        training_start = min(str(row["decision_timestamp"]) for row in train)
        training_cutoff = max(str(row["decision_timestamp"]) for row in train)
        label_max = max(str(row["label_available_timestamp"]) for row in train)
        fold_identity = canonical_hash({
            "training_start": training_start, "training_cutoff": training_cutoff,
            "maximum_label_available_timestamp": label_max,
            "prediction_date": prediction_date,
            "dataset_checksum": gate["selector_dataset_artifact_checksum"],
            "row_population_hash": population_checksum,
        })
        final_prediction_path = owner / "predictions.csv"
        link = build_artifact_link(
            artifact_kind="ORDINARY_SELECTOR_PREDICTION",
            artifact_id=f"ordinary-selector:{canonical_hash(identity)}",
            artifact_manifest_path=owner / "manifest.json",
            artifact_path=final_prediction_path,
            artifact_checksum=prediction_checksum,
            experiment_spec_hash=spec_hash, experiment_run_id=run_id,
            source_commit=_git_commit(),
            canonical_model_or_policy_id=model_id,
            model_or_policy_entry_hash=model_resolution.entry.entry_hash,
            dataset_id=gate["selector_dataset_id"],
            dataset_checksum=gate["selector_dataset_artifact_checksum"],
            row_population_hash=population_checksum,
            feature_schema_hash=gate["selector_feature_schema_checksum"],
            target_contract_hash=target.entry.entry_hash,
            decision_start=prediction_date, decision_end=prediction_date,
            training_cutoff=training_cutoff,
            maximum_label_available_timestamp=label_max,
            strict_oos_claim=True,
            strict_oos_evidence={
                "prediction_quality_passed": True,
                "row_population_verified": True,
                "temporal_legality_checked": True,
            },
            completion_status="complete",
        )
        verification = verify_selector_artifact(link); link.update(verification.to_dict())
        if verification.status != VERIFIED_STRICT_OOS:
            raise ValueError(f"Artifact link unverified: {verification.reason_codes}")
        diagnostics = getattr(model, "diagnostics", {})
        _write_json(temp / "metrics.json", {
            "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
            "model_id": model_id, "prediction_date": prediction_date,
            "model_parameters": model.get_params(),
            "model_diagnostics": diagnostics,
        })
        shared_model_artifact = publish_selector_sklearn_model_package(
            component_root=temp,
            published_component_root=owner,
            estimator=model,
            preprocessing=None,
            feature_order=feature_names,
            model_id=model_id,
            model_family=(
                "ordered_logit"
                if model_id == "ordered_logit_ranker" else model_id
            ),
            model_configuration=model.get_params(),
            random_seed=random_seed,
            training_boundary={
                "training_start": training_start,
                "training_cutoff": training_cutoff,
                "maximum_label_available_timestamp": label_max,
                "fold_identity": fold_identity,
            },
            training_population_checksum=canonical_hash(
                [str(row["row_id"]) for row in train]
            ),
            target_horizon_identity=target.canonical_id,
            prediction_path=prediction_path,
            prediction_schema=list(output_rows[0]),
            prediction_count=len(output_rows),
            input_population_checksum=str(
                job.get("expected_dataset_checksum")
                or gate["selector_dataset_artifact_checksum"]
            ),
            output_population_checksum=population_checksum,
            campaign_identity=runner_evidence["campaign_identity"],
            plan_job_identity=runner_evidence["plan_job_identity"],
            component_identity=canonical_hash(identity),
            component_runner=runner_evidence["declared_component_runner"],
            runtime_owner=runner_evidence["resolved_runtime_owner"],
            implementation_owner=(
                f"{type(model).__module__}.{type(model).__qualname__}"
            ),
            decision_date=prediction_date,
            fold_identity=fold_identity,
            training_row_artifact_identity=(
                f"{runner_evidence['operational_input_identity']}:training"
            ),
            prediction_row_artifact_identity=(
                f"{runner_evidence['operational_input_identity']}:prediction"
            ),
            source_schema_guarantee_identity=gate[
                "selector_feature_schema_checksum"
            ],
            input_package_identity=runner_evidence[
                "operational_input_identity"
            ],
            source_git_commit=_git_commit(),
        )
        manifest = {
            "component_schema_version": COMPONENT_SCHEMA_VERSION,
            "selector_model_identity": model_id,
            "selector_model_version": model_resolution.entry.entry_hash,
            "prediction_date": prediction_date,
            "training_start": training_start, "training_cutoff": training_cutoff,
            "training_label_available_timestamp_max": label_max,
            "fold_identity": fold_identity,
            "frozen_selector_dataset_identity": {
                "dataset_id": gate["selector_dataset_id"],
                "dataset_checksum": gate["selector_dataset_artifact_checksum"],
                "manifest_path": str(Path(job["selector_dataset_root"]) / "manifest.json"),
            },
            "symbol_registry_identity": gate["canonical_registry_id"],
            "daily_stock_spine_identity": gate["daily_spine_id"],
            "feature_contract_version": (
                "canonical_v2_daily_tree_cross_sectional_features_v1"
                if model_id == "ordered_logit_ranker"
                else "canonical_v2_daily_tabular_features_v1"
            ),
            "target_contract_version": target.canonical_id,
            "ranking_contract_version": model_payload.get("ranking_problem_contract") or "ranking_metric_contract_v1",
            "relevance_contract_version": model_payload.get("relevance_contract"),
            "prediction_row_count": len(output_rows),
            "prediction_population_checksum": population_checksum,
            "prediction_artifact_path": str(final_prediction_path),
            "prediction_checksum": prediction_checksum,
            "artifact_link": link, "publication_status": "complete",
            "validation_status": VERIFIED_STRICT_OOS,
            "git_commit": _git_commit(), "non_production_smoke": False,
            "parent_gate_logical_checksum": gate["logical_checksum"],
            "production_plan_job_checksum": job["logical_checksum"],
            **runner_evidence,
            "metrics_path": str(owner / "metrics.json"),
            "shared_model_artifact": shared_model_artifact,
        }
        manifest["manifest_checksum"] = canonical_hash(manifest)
        _write_json(temp / "manifest.json", manifest)
        owner.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, owner)
        if _sha256(owner / "predictions.csv") != prediction_checksum:
            raise ValueError("Artifact checksum mismatch after publication")
        _ledger(ledger_path, spec_hash, run_id, model_id, model_resolution.entry.entry_hash,
                target.entry.entry_hash, "COMPLETED", (str(owner / "manifest.json"), str(owner / "predictions.csv")),
                metadata={"metrics_path": str(owner / "metrics.json"), "prediction_date": prediction_date})
        return {"status": "COMPLETED", "manifest_path": str(owner / "manifest.json")}
    except (ValueError, FileExistsError) as exc:
        if temp is not None and temp.exists():
            shutil.rmtree(temp)
        _ledger(ledger_path, spec_hash, run_id, model_id, model_resolution.entry.entry_hash,
                target.entry.entry_hash, "REJECTED", (), rejection_summary=str(exc))
        raise
    except BaseException as exc:
        if temp is not None and temp.exists():
            shutil.rmtree(temp)
        _ledger(ledger_path, spec_hash, run_id, model_id, model_resolution.entry.entry_hash,
                target.entry.entry_hash, "FAILED", (), error_summary=f"{type(exc).__name__}: {exc}")
        raise


def _validate_job(job, gate, model_id, prediction_date):
    if gate.get("gate_contract_version") != "selector_parent_publication_gate.v1" or gate.get("status") != "READY":
        raise ValueError("Parent gate not ready")
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported ordinary selector model: {model_id}")
    expected = canonical_hash({key: value for key, value in job.items() if key != "logical_checksum"})
    if job.get("logical_checksum") != expected:
        raise ValueError("Production-plan checksum mismatch")
    if job.get("expected_parent_gate_checksum") != gate.get("logical_checksum"):
        raise ValueError("Parent-gate checksum mismatch")
    if job.get("expected_dataset_checksum") != gate.get("selector_dataset_artifact_checksum"):
        raise ValueError("Dataset mismatch")
    if not prediction_date:
        raise ValueError("Prediction date is required")


def _validate_rows(training, prediction, prediction_date, features):
    train = sorted((dict(row) for row in training), key=lambda row: (str(row["decision_timestamp"]), str(row["row_id"])))
    predict = sorted((dict(row) for row in prediction), key=lambda row: str(row["asset_id"]))
    ids = [str(row["row_id"]) for row in predict]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate prediction row IDs")
    if not train or not predict: raise ValueError("Incomplete prediction population")
    for row in train:
        if str(row["decision_timestamp"]) >= prediction_date: raise ValueError("Training/prediction overlap")
        if str(row["label_available_timestamp"]) > prediction_date: raise ValueError("Immature training label")
    for row in [*train, *predict]:
        if any(not math.isfinite(float(row[name])) for name in features):
            raise ValueError("Nonfinite selector feature")
    return train, predict


def _prediction_output(rows, scores, probabilities, model_id, date, gate, identity):
    import numpy as np
    asset_ids = np.asarray([str(row["asset_id"]) for row in rows])
    order = np.lexsort((asset_ids, -np.asarray(scores)))
    ranks = np.empty(len(rows), dtype=int); ranks[order] = np.arange(1, len(rows) + 1)
    output = []
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(rows), 5) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("Invalid ordered-logit probabilities")
    for index, (row, score) in enumerate(zip(rows, scores)):
        result = {
            "row_id": str(row["row_id"]), "asset_id": str(row["asset_id"]),
            "symbol": str(row["symbol"]), "prediction_date": date,
            "model_id": model_id, "selector_score": score,
            "deterministic_rank": int(ranks[index]),
            "dataset_identity": gate["selector_dataset_id"],
            "feature_contract_identity": identity["feature_schema"],
            "target_contract_identity": identity["target_contract"],
            "fold_identity": identity["job_id"],
        }
        if probabilities is not None:
            for cls in range(5): result[f"ordered_logit_probability_{cls}"] = float(probabilities[index, cls])
            result["ordered_logit_predicted_relevance_class"] = int(np.argmax(probabilities[index]))
            result["ordered_logit_expected_relevance"] = score
            result["ordered_logit_cross_sectional_rank"] = int(ranks[index])
            result["ordered_logit_rank_percentile"] = (int(ranks[index]) - 1) / max(len(rows) - 1, 1)
        output.append(result)
    return output


def _identity(job, gate, model_hash, target_hash, runner_evidence):
    return {
        "job_id": job["job_id"], "job_checksum": job["logical_checksum"],
        "model_id": job["model_id"], "model_entry_hash": model_hash,
        "prediction_date": job["prediction_date"],
        "dataset_id": gate["selector_dataset_id"],
        "dataset_checksum": gate["selector_dataset_artifact_checksum"],
        "feature_schema": job["feature_schema"],
        "target_contract": job["target_contract"], "target_contract_hash": target_hash,
        "ranking_contract": job.get("ranking_contract"),
        "parent_gate_checksum": gate["logical_checksum"],
        **runner_evidence,
    }


def _compatible(owner, identity):
    try:
        manifest = _json(owner / "manifest.json")
        return (
            manifest.get("publication_status") == "complete"
            and manifest.get("validation_status") == VERIFIED_STRICT_OOS
            and manifest.get("production_plan_job_checksum") == identity["job_checksum"]
            and all(
                manifest.get(field) == identity.get(field)
                for field in (
                    "campaign_identity", "plan_job_identity",
                    "declared_component_runner", "resolved_runtime_owner",
                    "operational_input_identity",
                )
            )
            and _sha256(owner / "predictions.csv") == manifest.get("prediction_checksum")
        )
    except (OSError, ValueError):
        return False


def _feature_names(path):
    payload = _json(path)
    names = [str(row["name"]) for row in payload.get("features", [])]
    if not names: raise ValueError("Feature schema is empty")
    return names


def _ledger(path, spec, run, model, model_hash, target_hash, status, artifacts, **kwargs):
    append_ledger_event(
        path, experiment_spec_hash_value=spec, experiment_run_id=run,
        event_status=status, artifact_kind="SELECTOR_PREDICTION_PARTITION",
        canonical_model_id=model, requested_model_id=model,
        registry_hashes={"model_entry_hash": model_hash, "target_entry_hash": target_hash},
        source_commit=_git_commit(), artifact_paths=artifacts, **kwargs,
    )


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
