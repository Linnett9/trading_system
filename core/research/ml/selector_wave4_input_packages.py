from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.ranking_labels import grouped_ranking_dataset
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_component_rows import (
    PREDICTION_ROWS_CONTRACT,
    TRAINING_ROWS_CONTRACT,
    prediction_row,
    training_row,
)
from core.research.ml.selector_operational_plan import (
    PLAN_CONTRACT,
    WAVE4_RUNNER,
    validate_selector_operational_plan,
)
from core.research.ml.selector_research_campaign import (
    validate_selector_campaign,
)
from core.research.ml.stock_level.contextual_elastic_net_selector import (
    contextual_elastic_net_input,
    contextual_interaction_contract,
)
from core.research.ml.stock_level.huber_selector import huber_selector_input
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    multi_horizon_linear_input,
    multi_horizon_target_contract,
)


PACKAGE_CONTRACT = "selector_component_input_package.v2"
FIT_INPUT_CONTRACT = "selector_wave4_fit_input.v1"
SOURCE_GUARANTEE_CONTRACT = "selector_source_schema_guarantees.v1"
RESULT_CONTRACT = "selector_operational_package_publication.v2"
PATH_FIELDS = {
    "training_rows_path", "prediction_rows_path",
    "fit_validation_rows_path", "wave4_fit_input_path",
    "package_manifest_path",
}


def source_schema_guarantee_manifest(
    *,
    dataset_identity: str,
    schema_checksum: str,
    guarantees: Sequence[str],
    field_identities: Mapping[str, str],
    source_commit: str,
) -> dict[str, Any]:
    logical = {
        "contract_version": SOURCE_GUARANTEE_CONTRACT,
        "dataset_identity": str(dataset_identity),
        "schema_checksum": str(schema_checksum),
        "guarantees": sorted({str(value) for value in guarantees}),
        "field_identities": dict(sorted(field_identities.items())),
        "source_commit": str(source_commit),
    }
    if not all(
        logical[field] for field in (
            "dataset_identity", "schema_checksum", "source_commit"
        )
    ):
        raise ValueError("Source-guarantee ancestry is incomplete")
    logical["guarantee_identity"] = canonical_hash(logical)
    logical["logical_checksum"] = canonical_hash(logical)
    return logical


def publish_selector_operational_packages_v2(
    *,
    plan: Mapping[str, Any],
    campaign: Mapping[str, Any],
    source_guarantees: Mapping[str, Any],
    parent_identities: Mapping[str, str],
    rows_by_job: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    validate_selector_campaign(campaign)
    validate_selector_operational_plan(plan, campaign=campaign)
    _validate_source_guarantees(source_guarantees)
    results = []
    for job in plan["jobs"]:
        results.append(
            _publish_job(
                job=job, plan=plan, campaign=campaign,
                source_guarantees=source_guarantees,
                parent_identities=parent_identities,
                supplied=rows_by_job.get(str(job["job_id"])),
                output_root=output_root,
            )
        )
    counts = {
        status: sum(row["status"] == status for row in results)
        for status in (
            "PACKAGE_PUBLISHED", "SKIPPED_COMPATIBLE",
            "BLOCKED_SOURCE_SCHEMA", "BLOCKED_PARENT_IDENTITIES",
            "INVALID_INPUT",
        )
    }
    complete = counts["PACKAGE_PUBLISHED"] + counts["SKIPPED_COMPATIBLE"]
    logical = {
        "result_contract_version": RESULT_CONTRACT,
        "operational_plan_identity": plan["plan_identity"],
        "operational_plan_checksum": plan["logical_checksum"],
        "campaign_identity": campaign["campaign_identity"],
        "source_guarantee_identity": source_guarantees[
            "guarantee_identity"
        ],
        "expected_jobs": len(plan["jobs"]),
        "packages_published": counts["PACKAGE_PUBLISHED"],
        "packages_skipped_compatible": counts["SKIPPED_COMPATIBLE"],
        "packages_blocked_source_schema": counts[
            "BLOCKED_SOURCE_SCHEMA"
        ],
        "packages_blocked_parent_identities": counts[
            "BLOCKED_PARENT_IDENTITIES"
        ],
        "invalid_jobs": counts["INVALID_INPUT"],
        "per_job_results": results,
        "all_packages_complete": complete == len(plan["jobs"]),
        "ordinary_packages_complete": all(
            row["status"] in {"PACKAGE_PUBLISHED", "SKIPPED_COMPATIBLE"}
            for row, job in zip(results, plan["jobs"])
            if not job["package_publication_requirements"][
                "wave4_fit_input_required"
            ]
        ),
        "wave4_packages_complete": all(
            row["status"] in {"PACKAGE_PUBLISHED", "SKIPPED_COMPATIBLE"}
            for row, job in zip(results, plan["jobs"])
            if job["package_publication_requirements"][
                "wave4_fit_input_required"
            ]
        ),
        "fitting_performed": False,
        "prediction_performed": False,
        "evaluation_performed": False,
        "production_dataset_read": False,
    }
    logical["logical_checksum"] = canonical_hash(logical)
    return logical


def validate_v2_package(manifest_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if manifest.get("package_contract_version") != PACKAGE_CONTRACT:
        raise ValueError("V2 package contract mismatch")
    if manifest.get("logical_checksum") != _package_logical(manifest):
        raise ValueError("V2 package logical checksum mismatch")
    for prefix in ("training", "prediction"):
        path = Path(str(manifest[f"{prefix}_rows_path"]))
        if _sha(path) != manifest[f"{prefix}_artifact_sha256"]:
            raise ValueError(f"V2 package {prefix} artifact was tampered")
        rows = _json(path)
        if canonical_hash(rows) != manifest[
            f"{prefix}_rows_logical_checksum"
        ]:
            raise ValueError(f"V2 package {prefix} row checksum mismatch")
        _validated_rows(rows, training_role=prefix == "training")
    fit_path = manifest.get("wave4_fit_input_path")
    if manifest["wave4_fit_input_applicable"]:
        if not fit_path:
            raise ValueError("Wave-4 package fit input is missing")
        fit = _json(Path(str(fit_path)))
        if _sha(Path(str(fit_path))) != manifest[
            "wave4_fit_input_artifact_sha256"
        ]:
            raise ValueError("Wave-4 fit-input artifact was tampered")
        if fit.get("wave4_fit_input_logical_checksum") != _fit_logical(fit):
            raise ValueError("Wave-4 fit-input logical checksum mismatch")
        if fit["wave4_fit_input_logical_checksum"] != manifest[
            "wave4_fit_input_logical_checksum"
        ]:
            raise ValueError("Wave-4 fit-input manifest identity mismatch")
    elif fit_path:
        raise ValueError("Ordinary package cannot own a Wave-4 fit input")
    return manifest


def _publish_job(
    *,
    job, plan, campaign, source_guarantees, parent_identities,
    supplied, output_root,
):
    job_id = str(job["job_id"])
    missing_parents = [
        key for key, expected in plan["parent_identity_requirements"].items()
        if not parent_identities.get(key)
        or parent_identities.get(key) != expected
    ]
    if missing_parents:
        return _result(
            job_id, "BLOCKED_PARENT_IDENTITIES",
            [f"PARENT_IDENTITY_MISSING_OR_CHANGED:{key}" for key in missing_parents],
        )
    available = set(source_guarantees["guarantees"])
    if source_guarantees.get("dataset_identity") != parent_identities.get(
        "selector_dataset_identity"
    ):
        return _result(
            job_id, "INVALID_INPUT",
            ["SOURCE_GUARANTEE_DATASET_IDENTITY_MISMATCH"],
        )
    missing = sorted(
        set(job["required_source_guarantees"]) - available
    )
    if missing:
        return _result(
            job_id, "BLOCKED_SOURCE_SCHEMA",
            [f"SOURCE_CONTRACT_MISSING:{field}" for field in missing],
        )
    if supplied is None:
        return _result(job_id, "INVALID_INPUT", ["FROZEN_ROWS_MISSING"])
    try:
        training = _validated_rows(
            list(supplied.get("training_rows") or ()), training_role=True
        )
        prediction = _validated_rows(
            list(supplied.get("prediction_rows") or ()), training_role=False
        )
        _validate_populations(training, prediction, job)
        package_id = canonical_hash({
            "plan_identity": plan["plan_identity"],
            "plan_job_identity": job["plan_job_identity"],
            "source_guarantee_identity": source_guarantees[
                "guarantee_identity"
            ],
            "training_rows": training,
            "prediction_rows": prediction,
            "translation_identity": {
                key: value for key, value in supplied.items()
                if key not in {"training_rows", "prediction_rows"}
            },
        })
        fit_input = (
            _fit_input(
                job, training, prediction, supplied, package_id,
                plan, campaign, source_guarantees, parent_identities,
            )
            if job["package_publication_requirements"][
                "wave4_fit_input_required"
            ] else None
        )
        manifest_path, status = _publish_directory(
            job=job, plan=plan, campaign=campaign,
            source_guarantees=source_guarantees,
            parent_identities=parent_identities,
            training=training, prediction=prediction,
            fit_input=fit_input, package_id=package_id,
            supplied=supplied, output_root=output_root,
        )
        return _result(
            job_id, status, [], manifest_path=str(manifest_path),
            package_identity=package_id,
        )
    except (KeyError, TypeError, ValueError, FileExistsError) as exc:
        return _result(
            job_id, "INVALID_INPUT",
            [f"{type(exc).__name__}:{exc}"],
        )


def _validated_rows(rows, *, training_role):
    if not rows:
        raise ValueError("Frozen row population is empty")
    validated = []
    for value in rows:
        payload = {
            key: item for key, item in value.items()
            if key not in {
                "logical_row_checksum", "contract_version", "role"
            }
        }
        rebuilt = training_row(payload) if training_role else prediction_row(
            payload
        )
        expected_contract = (
            TRAINING_ROWS_CONTRACT
            if training_role else PREDICTION_ROWS_CONTRACT
        )
        if (
            value.get("contract_version") != expected_contract
            or value.get("logical_row_checksum")
            != rebuilt["logical_row_checksum"]
        ):
            raise ValueError("Strengthened row checksum/contract mismatch")
        validated.append(dict(value))
    ordered = sorted(
        validated,
        key=lambda row: (
            str(row["decision_date"]), str(row["symbol_identity"]),
            str(row["dataset_row_identity"]),
        ),
    )
    if validated != ordered:
        raise ValueError("Frozen rows are not canonically ordered")
    return validated


def _validate_populations(training, prediction, job):
    train_ids = {row["dataset_row_identity"] for row in training}
    prediction_ids = {row["dataset_row_identity"] for row in prediction}
    if train_ids & prediction_ids:
        raise ValueError("Training and prediction populations overlap")
    for row in [*training, *prediction]:
        expectations = {
            "campaign_identity": job["campaign_identity"],
            "plan_job_identity": job["plan_job_identity"],
            "model_id": job["model_id"],
            "target_horizon": job.get("horizon_id") or "return_10s",
        }
        if any(row.get(key) != value for key, value in expectations.items()):
            raise ValueError("Frozen row ancestry differs from plan job")
    if {row["prediction_date"] for row in prediction} != {
        job["prediction_date"]
    }:
        raise ValueError("Prediction-row date differs from plan job")


def _fit_input(
    job, training, prediction, supplied, package_id, plan, campaign,
    source_guarantees, parent_identities,
):
    profile = job["operational_input_profile"]
    rows = [
        *_translated_rows(training, "TRAINING", profile),
        *_translated_rows(prediction, "PREDICTION", profile),
    ]
    fold = str(training[0]["fold_identity"])
    population = canonical_hash([
        row["dataset_row_identity"] for row in [*training, *prediction]
    ])
    if profile == "WAVE4_TABULAR":
        value = huber_selector_input(
            rows, target_horizon=job.get("horizon_id") or "return_10s",
            target_contract_identity=job["target_contract"],
            feature_schema_identity=job["required_feature_profile"],
            training_fold_identity=fold,
            validation_fold_identity=str(prediction[0]["fold_identity"]),
            dataset_identity=parent_identities["selector_dataset_identity"],
            source_population_checksum=population,
        )
    elif profile == "WAVE4_CONTEXTUAL":
        interactions = contextual_interaction_contract(
            rows[0]["stock_feature_ids"], rows[0]["market_context_ids"],
            interactions=supplied.get("interactions"),
        )
        value = contextual_elastic_net_input(
            rows, target_horizon=job.get("horizon_id") or "return_10s",
            stock_feature_schema_identity=supplied[
                "stock_feature_schema_identity"
            ],
            market_context_schema_identity=supplied[
                "market_context_schema_identity"
            ],
            interaction_contract_identity=interactions["contract_version"],
            training_fold_identity=fold,
            validation_fold_identity=str(prediction[0]["fold_identity"]),
            dataset_identity=parent_identities["selector_dataset_identity"],
            source_population_checksum=population,
        )
        value["interaction_contract"] = interactions
    elif profile == "WAVE4_MULTI_HORIZON":
        value = multi_horizon_linear_input(
            rows, target_contract=multi_horizon_target_contract(),
            feature_schema_identity=job["required_feature_profile"],
            dataset_identity=parent_identities["selector_dataset_identity"],
            fold_identity=fold, source_population_checksum=population,
        )
    elif profile == "WAVE4_GROUPED_RANKING":
        value = grouped_ranking_dataset(
            rows, label_type="quintile_integer",
            feature_schema_identity=job["required_feature_profile"],
            target_contract_identity=job["target_contract"],
            ranking_label_contract_identity=job[
                "required_relevance_label_fields"
            ][1],
            split_identity=fold,
            allowed_cutoff=training[0]["training_boundary_identity"],
            minimum_group_size=int(supplied.get("minimum_group_size", 1)),
        )
        if not value.get("valid"):
            raise ValueError(
                "Grouped ranking input invalid: "
                + ",".join(value.get("blocking_reasons") or ())
            )
    else:
        raise ValueError(f"Unsupported Wave-4 profile: {profile}")
    value.update({
        "wave4_fit_input_contract_version": FIT_INPUT_CONTRACT,
        "source_rows_embedded": True,
        "operational_input_identity": package_id,
        "operational_plan_identity": plan["plan_identity"],
        "campaign_identity": campaign["campaign_identity"],
        "plan_job_identity": job["plan_job_identity"],
        "component_identity": job["component_identity"],
        "model_id": job["model_id"],
        "prediction_date": job["prediction_date"],
        "horizon_id": job.get("horizon_id"),
        "operational_input_profile": profile,
        "declared_component_runner": job["component_runner"],
        "resolved_runtime_owner": job["runtime_publication_owner"],
        "source_guarantee_identity": source_guarantees[
            "guarantee_identity"
        ],
        "fitting_performed": False,
        "prediction_performed": False,
        "evaluation_performed": False,
    })
    if job.get("label_gain_policy"):
        value["label_gain_policy"] = job["label_gain_policy"]
    value["wave4_fit_input_logical_checksum"] = _fit_logical(value)
    return value


def _translated_rows(rows, role, profile):
    output = []
    for row in rows:
        common = {
            "row_id": row["dataset_row_identity"],
            "asset_id": row["symbol_identity"],
            "decision_timestamp": row["decision_date"],
            "feature_availability_timestamp": row.get(
                "feature_availability_timestamp", row["decision_date"]
            ),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "split": role,
        }
        if profile in {"WAVE4_TABULAR", "WAVE4_MULTI_HORIZON"}:
            common.update(
                feature_ids=list(row["ordered_feature_ids"]),
                feature_values=list(row["ordered_feature_values"]),
            )
        if role == "TRAINING":
            if profile in {"WAVE4_TABULAR", "WAVE4_CONTEXTUAL"}:
                common.update(
                    target_value=float(row["target_value"]),
                    target_maturity_timestamp=row[
                        "target_maturity_timestamp"
                    ],
                )
            elif profile == "WAVE4_MULTI_HORIZON":
                common.update(
                    target_values=dict(row["horizon_target_values"]),
                    target_maturity_timestamps=dict(
                        row["horizon_target_maturity_timestamps"]
                    ),
                    target_availability_state=dict(
                        row["horizon_target_availability_states"]
                    ),
                )
        if profile == "WAVE4_CONTEXTUAL":
            common.update(
                stock_feature_ids=list(row["stock_feature_ids"]),
                stock_feature_values=list(row["stock_feature_values"]),
                market_context_ids=list(row["market_context_ids"]),
                market_context_values=list(row["market_context_values"]),
            )
        elif profile == "WAVE4_GROUPED_RANKING":
            common = {
                "row_id": row["dataset_row_identity"],
                "asset_id": row["symbol_identity"],
                "decision_date": row["decision_date"],
                "feature_names": list(row["ordered_feature_ids"]),
                "feature_values": list(row["ordered_feature_values"]),
                "feature_availability_timestamp": row.get(
                    "feature_availability_timestamp", row["decision_date"]
                ),
                "split_role": role,
            }
            if role == "TRAINING":
                common.update(
                    label=int(row["relevance_label"]),
                    target_maturity_timestamp=row[
                        "target_maturity_timestamp"
                    ],
                )
        output.append(common)
    return output


def _publish_directory(
    *,
    job, plan, campaign, source_guarantees, parent_identities,
    training, prediction, fit_input, package_id, supplied, output_root,
):
    owner = output_root / "component_inputs_v2" / job["job_id"].replace(
        ":", "__"
    )
    if owner.exists():
        manifest = validate_v2_package(owner / "manifest.json")
        if manifest["package_id"] == package_id:
            return owner / "manifest.json", "SKIPPED_COMPATIBLE"
        raise FileExistsError(f"Incompatible V2 package: {owner}")
    temp = owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp")
    temp.mkdir(parents=True, exist_ok=False)
    try:
        _write(temp / "training_rows.json", training)
        _write(temp / "prediction_rows.json", prediction)
        if fit_input is not None:
            _write(temp / "wave4_fit_input.json", fit_input)
        paths = {
            "training_rows_path": str(owner / "training_rows.json"),
            "prediction_rows_path": str(owner / "prediction_rows.json"),
            "package_manifest_path": str(owner / "manifest.json"),
            "wave4_fit_input_path": (
                str(owner / "wave4_fit_input.json")
                if fit_input is not None else None
            ),
        }
        manifest = {
            "package_contract_version": PACKAGE_CONTRACT,
            "package_id": package_id,
            "operational_plan_identity": plan["plan_identity"],
            "operational_plan_checksum": plan["logical_checksum"],
            "campaign_identity": campaign["campaign_identity"],
            "campaign_version": campaign["campaign_version"],
            "protocol_identity": plan["protocol_identity"],
            "production_plan_job_id": job["job_id"],
            "production_plan_job_checksum": job["plan_job_identity"],
            "plan_job_identity": job["plan_job_identity"],
            "component_identity": job["component_identity"],
            "model_id": job["model_id"],
            "prediction_date": job["prediction_date"],
            "horizon_id": job.get("horizon_id"),
            "component_runner": job["component_runner"],
            "declared_component_runner": job["component_runner"],
            "resolved_runtime_owner": job["runtime_publication_owner"],
            "operational_input_profile": job[
                "operational_input_profile"
            ],
            **dict(parent_identities),
            "model_registry_entry_checksum": job[
                "model_registry_entry_checksum"
            ],
            "source_guarantee_identity": source_guarantees[
                "guarantee_identity"
            ],
            "source_guarantee_checksum": source_guarantees[
                "logical_checksum"
            ],
            "training_row_contract": TRAINING_ROWS_CONTRACT,
            "training_row_count": len(training),
            "training_rows_logical_checksum": canonical_hash(training),
            "training_artifact_sha256": _sha(
                temp / "training_rows.json"
            ),
            "prediction_row_contract": PREDICTION_ROWS_CONTRACT,
            "prediction_row_count": len(prediction),
            "prediction_rows_logical_checksum": canonical_hash(prediction),
            "prediction_artifact_sha256": _sha(
                temp / "prediction_rows.json"
            ),
            "fit_validation_applicable": False,
            "evaluation_outcome_applicable": True,
            "wave4_fit_input_applicable": fit_input is not None,
            "wave4_fit_input_logical_checksum": (
                fit_input["wave4_fit_input_logical_checksum"]
                if fit_input else None
            ),
            "wave4_fit_input_artifact_sha256": (
                _sha(temp / "wave4_fit_input.json") if fit_input else None
            ),
            "feature_profile": job["required_feature_profile"],
            "ordered_feature_checksum": training[0][
                "feature_order_checksum"
            ],
            "target_contract_id": job["target_contract"],
            "fold_identity": training[0]["fold_identity"],
            "split_identity": supplied.get(
                "split_identity", training[0]["fold_identity"]
            ),
            "training_boundary": training[0][
                "training_boundary_identity"
            ],
            "training_cutoff": training[0][
                "training_boundary_identity"
            ],
            "outcome_maturity_cutoff": max(
                row["target_maturity_timestamp"] for row in training
            ),
            "purge_sessions": training[0]["purge_sessions"],
            "embargo_sessions": training[0]["embargo_sessions"],
            "source_git_commit": job["source_git_commit"],
            "package_status": "COMPLETE",
            "publication_status": "complete",
            "validation_status": "VERIFIED",
            "blockers": [],
            "fitting_performed": False,
            "prediction_performed": False,
            "evaluation_performed": False,
            **paths,
        }
        manifest["logical_checksum"] = _package_logical(manifest)
        _write(temp / "manifest.json", manifest)
        owner.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, owner)
        validate_v2_package(owner / "manifest.json")
        return owner / "manifest.json", "PACKAGE_PUBLISHED"
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def _validate_source_guarantees(value):
    expected = canonical_hash({
        key: item for key, item in value.items()
        if key != "logical_checksum"
    })
    if (
        value.get("contract_version") != SOURCE_GUARANTEE_CONTRACT
        or value.get("logical_checksum") != expected
        or value.get("guarantee_identity") != canonical_hash({
            key: item for key, item in value.items()
            if key not in {"guarantee_identity", "logical_checksum"}
        })
    ):
        raise ValueError("Source-schema guarantee manifest is invalid")


def _package_logical(value):
    return canonical_hash({
        key: item for key, item in value.items()
        if key != "logical_checksum" and key not in PATH_FIELDS
    })


def _fit_logical(value):
    return canonical_hash({
        key: item for key, item in value.items()
        if key != "wave4_fit_input_logical_checksum"
    })


def _result(job_id, status, blockers, **extra):
    return {
        "job_id": job_id, "status": status,
        "blockers": list(blockers), **extra,
    }


def _write(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()
