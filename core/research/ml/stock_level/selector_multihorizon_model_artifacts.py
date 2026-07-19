from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.compute.artifact_contracts import (
    FITTED_MODEL_CONTRACT,
    PREDICTION_BINDING_CONTRACT,
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
    canonical_checksum,
    validate_prediction_binding,
)
from core.research.compute.artifact_storage import (
    publish_artifact_package,
    validate_artifact_package,
)
from core.research.compute.model_artifacts import (
    validate_model_artifact_manifest,
)
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    FittedMultiHorizonMember,
    HORIZON_IDS,
)
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    publish_selector_sklearn_model_package,
)


def publish_selector_multihorizon_package(
    *,
    component_root: Path,
    published_component_root: Path,
    fitted_members: Sequence[FittedMultiHorizonMember],
    fit_result: Mapping[str, Any],
    selected_component_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    campaign_identity: str,
    plan_job_identity: str,
    component_identity: str,
    component_runner: str,
    runtime_owner: str,
    decision_date: str,
    training_row_artifact_identity: str,
    prediction_row_artifact_identity: str,
    input_package_identity: str,
    source_schema_guarantee_identity: str,
    input_population_checksum: str,
    source_git_commit: str,
) -> dict[str, Any]:
    family = {
        "multi_horizon_ridge": "ridge",
        "multi_horizon_elastic_net": "elastic_net",
    }.get(model_id)
    if family is None:
        raise ValueError("MEMBER_HORIZON_UNEXPECTED")
    members = [row for row in fitted_members if row.model_family == family]
    ordered = sorted(members, key=lambda row: row.horizon_order)
    horizons = [row.horizon_id for row in ordered]
    if len(horizons) != len(set(horizons)):
        raise ValueError("MEMBER_HORIZON_DUPLICATE")
    if any(value not in HORIZON_IDS for value in horizons):
        raise ValueError("MEMBER_HORIZON_UNEXPECTED")
    if horizons != list(HORIZON_IDS):
        raise ValueError("ENSEMBLE_MEMBER_COUNT_MISMATCH")
    if any(
        not row.target_identity
        or not row.training_population.get("training_checksum")
        or not row.training_population.get(
            "maximum_label_maturity_timestamp"
        )
        or not row.preprocessing
        or not row.ordered_feature_ids
        for row in ordered
    ):
        raise ValueError("MEMBER_MATURITY_EVIDENCE_MISSING")
    feature_order = list(ordered[0].ordered_feature_ids)
    if any(list(row.ordered_feature_ids) != feature_order for row in ordered):
        raise ValueError("MEMBER_FEATURE_ORDER_MISSING")

    wide_rows = _wide_prediction_rows(
        fit_result=fit_result,
        selected_component_rows=selected_component_rows,
        family=family,
    )
    prediction_schema = list(wide_rows[0])
    prediction_bytes = _csv_bytes(wide_rows, prediction_schema)
    prediction_checksum = hashlib.sha256(prediction_bytes).hexdigest()
    output_population_checksum = canonical_checksum(
        [row["row_id"] for row in wide_rows]
    )
    run_id = os.environ.get("SELECTOR_COMPUTE_RUN_ID") or component_identity
    attempt_id = (
        os.environ.get("SELECTOR_COMPUTE_ATTEMPT_ID")
        or canonical_checksum(
            {"component": component_identity, "job": plan_job_identity}
        )
    )

    member_packages = []
    member_publication_statuses = []
    for row in ordered:
        member_root = (
            component_root
            / "shared_model_artifact"
            / "members"
            / row.horizon_id
        )
        member = publish_selector_sklearn_model_package(
            component_root=component_root,
            published_component_root=published_component_root,
            model_artifact_root=member_root,
            publish_prediction_binding=False,
            estimator=row.estimator,
            preprocessing=row.preprocessing,
            feature_order=row.ordered_feature_ids,
            model_id=model_id,
            member_model_identity=(
                f"{family}:{row.horizon_id}:{row.target_identity}"
            ),
            model_family=family,
            model_configuration=row.estimator_configuration,
            random_seed=row.random_state_identity,
            training_boundary={
                "training_cutoff": row.training_cutoff,
                "fold_identity": row.fold_identity,
                "maximum_label_maturity_timestamp": row.training_population[
                    "maximum_label_maturity_timestamp"
                ],
                "target_maturity_policy": (
                    "label_maturity_timestamp_lte_training_cutoff"
                ),
            },
            training_population_checksum=row.training_population[
                "training_checksum"
            ],
            target_horizon_identity=row.target_identity,
            prediction_path=component_root / "predictions.csv",
            prediction_schema=prediction_schema,
            prediction_count=len(wide_rows),
            input_population_checksum=input_population_checksum,
            output_population_checksum=output_population_checksum,
            campaign_identity=campaign_identity,
            plan_job_identity=plan_job_identity,
            component_identity=(
                f"{component_identity}:{row.horizon_id}:{family}"
            ),
            component_runner=component_runner,
            runtime_owner=runtime_owner,
            implementation_owner=(
                f"{type(row.estimator).__module__}."
                f"{type(row.estimator).__qualname__}"
            ),
            decision_date=decision_date,
            fold_identity=row.fold_identity,
            training_row_artifact_identity=(
                f"{training_row_artifact_identity}:{row.horizon_id}"
            ),
            prediction_row_artifact_identity=prediction_row_artifact_identity,
            source_schema_guarantee_identity=(
                source_schema_guarantee_identity
            ),
            input_package_identity=input_package_identity,
            source_git_commit=source_git_commit,
        )
        member_publication_statuses.append(member["compatible_skip_status"])
        member_packages.append({
            "horizon_id": row.horizon_id,
            "horizon_order": row.horizon_order,
            "target_identity": row.target_identity,
            "target_maturity_policy": (
                "label_maturity_timestamp_lte_training_cutoff"
            ),
            "maximum_label_maturity_timestamp": row.training_population[
                "maximum_label_maturity_timestamp"
            ],
            "training_population_checksum": row.training_population[
                "training_checksum"
            ],
            "preprocessing_identity": member["preprocessing_identity"],
            "artifact_identity": member["artifact_identity"],
            "artifact_checksum": member["package_checksum"],
            "artifact_logical_checksum": member["logical_checksum"],
            "package_path": member["model_package_path"],
            "prediction_column": f"selector_score_{row.horizon_id}",
        })

    preprocessing_identities = [
        row["preprocessing_identity"] for row in member_packages
    ]
    ensemble_metadata = {
        "model_id": model_id,
        "implementation_owner": (
            "core.research.ml.stock_level.multi_horizon_linear_selector:"
            "fit_multi_horizon_linear_selector"
        ),
        "model_family": family,
        "model_configuration": dict(fit_result.get("configuration") or {}),
        "model_configuration_checksum": str(
            fit_result.get("configuration_checksum")
            or canonical_checksum(fit_result.get("configuration") or {})
        ),
        "random_seed": (
            0 if family == "elastic_net"
            else "NOT_APPLICABLE_DETERMINISTIC"
        ),
        "training_boundary": {
            "training_cutoff": ordered[0].training_cutoff,
            "fold_identity": ordered[0].fold_identity,
        },
        "training_population_checksum": canonical_checksum([
            row["training_population_checksum"] for row in member_packages
        ]),
        "target_horizon_identity": canonical_checksum([
            row["target_identity"] for row in member_packages
        ]),
        "feature_order": feature_order,
        "feature_order_checksum": canonical_checksum(feature_order),
        "preprocessing_identity": canonical_checksum(
            preprocessing_identities
        ),
        "ordered_preprocessing_identities": preprocessing_identities,
        "ordered_horizons": list(HORIZON_IDS),
        "ordered_members": member_packages,
        "weights": [1.0] * len(member_packages),
        "combination_rule": "ordered_horizon_vector_no_scalar_aggregation",
        "required_prediction_schema": prediction_schema,
        "horizon_to_prediction_column": {
            row["horizon_id"]: row["prediction_column"]
            for row in member_packages
        },
        "member_count": len(member_packages),
        "expected_member_count": len(HORIZON_IDS),
        "campaign_identity": campaign_identity,
        "plan_job_identity": plan_job_identity,
        "component_identity": component_identity,
        "decision_date": decision_date,
        "source_schema_guarantee_identity": (
            source_schema_guarantee_identity
        ),
    }
    ensemble_id = "selector-ensemble:" + canonical_checksum({
        "component": component_identity,
        "members": [
            (row["horizon_id"], row["artifact_checksum"])
            for row in member_packages
        ],
    })
    ensemble_template = build_artifact_manifest(
        artifact_id=ensemble_id,
        artifact_type=ArtifactType.FITTED_MODEL.value,
        artifact_subtype="SELECTOR_MULTI_HORIZON_ENSEMBLE",
        artifact_role=ArtifactRole.RESEARCH_FOLD_MODEL.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[{
            "identity": input_package_identity,
            "checksum": input_population_checksum,
        }],
        source_artifacts=[
            {
                "identity": row["artifact_identity"],
                "checksum": row["artifact_checksum"],
            }
            for row in member_packages
        ],
        configuration_identity=ensemble_metadata[
            "model_configuration_checksum"
        ],
        configuration_checksum=ensemble_metadata[
            "model_configuration_checksum"
        ],
        source_git_commit=source_git_commit,
        serialization_handler="ENSEMBLE_MANIFEST",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=_dependencies(),
        claims={
            "fitting_performed": True,
            "prediction_performed": True,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
        },
        fitted_model_contract_version=FITTED_MODEL_CONTRACT,
        model_metadata=ensemble_metadata,
    )
    ensemble_root = component_root / "shared_model_artifact" / "ensemble"
    ensemble_status, ensemble = publish_artifact_package(
        ensemble_root,
        ensemble_template,
        {"metadata/ensemble.json": _json_bytes(ensemble_metadata)},
    )
    validate_model_artifact_manifest(ensemble)

    binding = {
        "contract_version": PREDICTION_BINDING_CONTRACT,
        "fitted_model_artifact_identity": ensemble["artifact_id"],
        "fitted_model_artifact_checksum": ensemble["logical_checksum"],
        "fitted_model_package_checksum": ensemble["package_checksum"],
        "preprocessing_identity": ensemble_metadata[
            "preprocessing_identity"
        ],
        "ordered_preprocessing_identities": preprocessing_identities,
        "input_population_checksum": input_population_checksum,
        "output_population_checksum": output_population_checksum,
        "prediction_artifact_identity": (
            "selector-multihorizon-prediction:"
            + canonical_checksum({
                "component": component_identity,
                "checksum": prediction_checksum,
            })
        ),
        "prediction_artifact_checksum": prediction_checksum,
        "prediction_schema": prediction_schema,
        "prediction_count": len(wide_rows),
        "ordered_member_identities": [
            row["artifact_identity"] for row in member_packages
        ],
        "ordered_member_checksums": [
            row["artifact_checksum"] for row in member_packages
        ],
        "horizon_to_prediction_column": ensemble_metadata[
            "horizon_to_prediction_column"
        ],
        "campaign_identity": campaign_identity,
        "component_identity": component_identity,
        "plan_job_identity": plan_job_identity,
        "decision_date": decision_date,
        "source_git_commit": source_git_commit,
    }
    prediction_template = build_artifact_manifest(
        artifact_id=binding["prediction_artifact_identity"],
        artifact_type=ArtifactType.PREDICTION_ARTIFACT.value,
        artifact_subtype="SELECTOR_MULTI_HORIZON_PREDICTION_BINDING",
        artifact_role=ArtifactRole.RESEARCH_PREDICTIONS.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[{
            "identity": input_package_identity,
            "checksum": input_population_checksum,
        }],
        source_artifacts=[{
            "identity": ensemble["artifact_id"],
            "checksum": ensemble["package_checksum"],
        }],
        configuration_identity=ensemble["configuration_identity"],
        configuration_checksum=ensemble["configuration_checksum"],
        source_git_commit=source_git_commit,
        serialization_handler="GENERIC_STAGE_FILES",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=_dependencies(),
        claims={
            "fitting_performed": False,
            "prediction_performed": True,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
        },
        prediction_model_binding=binding,
    )
    prediction_root = component_root / "shared_model_artifact" / "prediction"
    prediction_status, prediction = publish_artifact_package(
        prediction_root,
        prediction_template,
        {
            "predictions/multi_horizon_predictions.csv": prediction_bytes,
            "metadata/prediction_binding.json": _json_bytes(binding),
        },
    )
    validate_prediction_binding(prediction, ensemble)
    validate_multihorizon_artifacts(
        component_root=component_root,
        expected_horizons=HORIZON_IDS,
    )
    public_shared = published_component_root / "shared_model_artifact"
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": (
            "SKIPPED_COMPATIBLE"
            if ensemble_status == prediction_status == "SKIPPED_COMPATIBLE"
            and all(
                status == "SKIPPED_COMPATIBLE"
                for status in member_publication_statuses
            )
            else "COMPLETE"
        ),
        "artifact_identity": ensemble["artifact_id"],
        "package_checksum": ensemble["package_checksum"],
        "preprocessing_identity": ensemble_metadata[
            "preprocessing_identity"
        ],
        "ordered_preprocessing_identities": preprocessing_identities,
        "feature_order_identity": ensemble_metadata[
            "feature_order_checksum"
        ],
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction["artifact_id"],
        "ordered_member_identities": binding["ordered_member_identities"],
        "ordered_member_checksums": binding["ordered_member_checksums"],
        "ordered_horizons": list(HORIZON_IDS),
        "model_package_path": str(public_shared / "ensemble"),
        "prediction_package_path": str(public_shared / "prediction"),
        "member_package_paths": [
            str(
                public_shared / "members" / row["horizon_id"]
            )
            for row in member_packages
        ],
    }


def validate_multihorizon_artifacts(
    *, component_root: Path, expected_horizons: Sequence[str]
) -> dict[str, Any]:
    expected = list(expected_horizons)
    if expected != list(HORIZON_IDS):
        raise ValueError("MEMBER_HORIZON_UNEXPECTED")
    member_rows = []
    for horizon in expected:
        root = component_root / "shared_model_artifact" / "members" / horizon
        if not root.exists():
            raise ValueError("MEMBER_MODEL_PACKAGE_MISSING")
        try:
            manifest = validate_artifact_package(root)
        except ValueError as exc:
            raise ValueError("MEMBER_MODEL_PACKAGE_CORRUPT") from exc
        validate_model_artifact_manifest(manifest)
        metadata = manifest["model_metadata"]
        if metadata.get("model_id") not in {
            "multi_horizon_ridge",
            "multi_horizon_elastic_net",
        }:
            raise ValueError("MEMBER_MODEL_PACKAGE_INCOMPATIBLE")
        if metadata.get("member_model_identity", "").split(":")[1:2] != [
            horizon
        ]:
            raise ValueError("MEMBER_TARGET_IDENTITY_MISMATCH")
        if metadata["target_horizon_identity"] in (None, ""):
            raise ValueError("MEMBER_TARGET_IDENTITY_MISMATCH")
        if metadata["preprocessing_identity"] in (None, ""):
            raise ValueError("MEMBER_PREPROCESSING_MISSING")
        member_rows.append((horizon, manifest))
    ensemble = validate_artifact_package(
        component_root / "shared_model_artifact" / "ensemble"
    )
    validate_model_artifact_manifest(ensemble)
    metadata = ensemble["model_metadata"]
    if metadata["ordered_horizons"] != expected:
        raise ValueError("ENSEMBLE_ORDER_MISMATCH")
    actual = [
        (row["horizon_id"], row["artifact_checksum"])
        for row in metadata["ordered_members"]
    ]
    wanted = [
        (horizon, manifest["package_checksum"])
        for horizon, manifest in member_rows
    ]
    if actual != wanted:
        raise ValueError("MEMBER_MODEL_PACKAGE_INCOMPATIBLE")
    prediction = validate_artifact_package(
        component_root / "shared_model_artifact" / "prediction"
    )
    validate_prediction_binding(prediction, ensemble)
    binding = prediction["prediction_model_binding"]
    mapping = binding["horizon_to_prediction_column"]
    if set(mapping) != set(expected) or any(
        mapping[horizon] != f"selector_score_{horizon}"
        for horizon in expected
    ):
        raise ValueError("PREDICTION_HORIZON_MAPPING_MISMATCH")
    prediction_file = (
        component_root
        / "shared_model_artifact"
        / "prediction"
        / "predictions"
        / "multi_horizon_predictions.csv"
    )
    rows, schema = _read_csv(prediction_file)
    if schema != binding["prediction_schema"]:
        raise ValueError("PREDICTION_HORIZON_COLUMN_MISSING")
    if len(rows) != binding["prediction_count"]:
        raise ValueError("Prediction row-count mismatch")
    return {
        "ensemble": ensemble,
        "prediction": prediction,
        "members": [manifest for _, manifest in member_rows],
    }


def resolve_multihorizon_package(
    *,
    component_root: Path,
    job: Mapping[str, Any],
    component_manifest: Mapping[str, Any],
    run_identity: str,
) -> Mapping[str, Any] | None:
    shared = component_root / "shared_model_artifact"
    if not shared.exists():
        return None
    validated = validate_multihorizon_artifacts(
        component_root=component_root,
        expected_horizons=HORIZON_IDS,
    )
    ensemble = validated["ensemble"]
    prediction = validated["prediction"]
    metadata = ensemble["model_metadata"]
    if (
        metadata.get("campaign_identity") != job.get("campaign_identity")
        or metadata.get("plan_job_identity") != job.get("job_id")
        or metadata.get("decision_date") != job.get("prediction_date")
        or component_manifest.get("production_plan_job_checksum")
        != job.get("logical_checksum")
        or (
            metadata.get("run_identity") is not None
            and metadata.get("run_identity") != run_identity
        )
    ):
        raise ValueError("ENSEMBLE_MANIFEST_INCOMPATIBLE")
    binding = prediction["prediction_model_binding"]
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": "SKIPPED_COMPATIBLE",
        "artifact_identity": ensemble["artifact_id"],
        "package_checksum": ensemble["package_checksum"],
        "preprocessing_identity": metadata["preprocessing_identity"],
        "ordered_preprocessing_identities": metadata[
            "ordered_preprocessing_identities"
        ],
        "feature_order_identity": metadata["feature_order_checksum"],
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction["artifact_id"],
        "ordered_member_identities": binding["ordered_member_identities"],
        "ordered_member_checksums": binding["ordered_member_checksums"],
        "ordered_horizons": metadata["ordered_horizons"],
        "model_package_path": str(shared / "ensemble"),
        "prediction_package_path": str(shared / "prediction"),
        "member_package_paths": [
            str(shared / "members" / horizon) for horizon in HORIZON_IDS
        ],
    }


def _wide_prediction_rows(
    *,
    fit_result: Mapping[str, Any],
    selected_component_rows: Sequence[Mapping[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    owners = {str(row["row_id"]): row for row in selected_component_rows}
    if not owners:
        raise ValueError("Prediction row-count mismatch")
    scores: dict[str, dict[str, float]] = {}
    for row in fit_result.get("predictions", ()):
        if row.get("model_family") != family:
            continue
        horizon = str(row.get("horizon_id") or "")
        if horizon not in HORIZON_IDS:
            raise ValueError("MEMBER_HORIZON_UNEXPECTED")
        scores.setdefault(str(row["row_id"]), {})[horizon] = float(
            row["predicted_return"]
        )
    if set(scores) != set(owners):
        raise ValueError("Prediction row-count mismatch")
    output = []
    for row_id in sorted(owners):
        if list(scores[row_id]) != list(HORIZON_IDS):
            if set(scores[row_id]) != set(HORIZON_IDS):
                raise ValueError("PREDICTION_HORIZON_COLUMN_MISSING")
        owner = owners[row_id]
        output.append({
            "row_id": row_id,
            "asset_id": str(owner["asset_id"]),
            "symbol": str(owner["symbol"]),
            "prediction_date": str(owner["prediction_date"]),
            **{
                f"selector_score_{horizon}": scores[row_id][horizon]
                for horizon in HORIZON_IDS
            },
        })
    return output


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], schema: Sequence[str]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(schema))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or ())


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8")


def _dependencies() -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "UNAVAILABLE"

    return {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "joblib": version("joblib"),
    }
