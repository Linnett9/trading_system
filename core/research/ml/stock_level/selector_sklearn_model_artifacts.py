from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import csv
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
    inspect_model_metadata,
    read_trusted_model_bytes,
    validate_model_artifact_manifest,
)


SELECTOR_SKLEARN_PACKAGE_CONTRACT = "selector_sklearn_model_package.v1"
SUPPORTED_MODELS = {
    "ridge",
    "elastic_net",
    "ordered_logit_ranker",
    "huber",
    "contextual_elastic_net",
    "multi_horizon_ridge",
    "multi_horizon_elastic_net",
}


def publish_selector_sklearn_model_package(
    *,
    component_root: Path,
    published_component_root: Path | None = None,
    estimator: Any,
    preprocessing: Mapping[str, Any] | None,
    feature_order: Sequence[str],
    model_id: str,
    model_family: str,
    model_configuration: Mapping[str, Any],
    random_seed: int | str,
    training_boundary: Mapping[str, Any],
    training_population_checksum: str,
    target_horizon_identity: str,
    prediction_path: Path,
    prediction_schema: Sequence[str],
    prediction_count: int,
    input_population_checksum: str,
    output_population_checksum: str,
    campaign_identity: str,
    plan_job_identity: str,
    component_identity: str,
    component_runner: str,
    runtime_owner: str,
    implementation_owner: str,
    decision_date: str,
    fold_identity: str,
    training_row_artifact_identity: str,
    prediction_row_artifact_identity: str,
    source_schema_guarantee_identity: str,
    input_package_identity: str,
    source_git_commit: str,
    contextual_evidence: Mapping[str, Any] | None = None,
    model_artifact_root: Path | None = None,
    publish_prediction_binding: bool = True,
    member_model_identity: str | None = None,
) -> dict[str, Any]:
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported selector sklearn artifact family: {model_id}")
    ordered_features = [str(value) for value in feature_order]
    if not ordered_features or len(ordered_features) != len(set(ordered_features)):
        raise ValueError("FEATURE_ORDER_MISSING")
    if publish_prediction_binding:
        if not prediction_path.is_file():
            raise ValueError("PREDICTION_BINDING_MISSING")
        if prediction_count < 1 or not prediction_schema:
            raise ValueError("PREDICTION_BINDING_MISSING")
        _validate_prediction_file(
            prediction_path,
            expected_schema=[str(value) for value in prediction_schema],
            expected_count=prediction_count,
        )
    if preprocessing is None and not _preprocessing_embedded(estimator):
        raise ValueError("PREPROCESSING_EVIDENCE_MISSING")

    run_id = os.environ.get("SELECTOR_COMPUTE_RUN_ID") or component_identity
    attempt_id = (
        os.environ.get("SELECTOR_COMPUTE_ATTEMPT_ID")
        or canonical_checksum(
            {"component_identity": component_identity, "plan_job": plan_job_identity}
        )
    )
    run_identity = os.environ.get("SELECTOR_COMPUTE_RUN_IDENTITY")
    estimator_bytes = _joblib_bytes(estimator)
    preprocessing_payload = dict(preprocessing or {})
    preprocessing_bytes = _json_bytes(preprocessing_payload)
    preprocessing_identity = canonical_checksum(
        {
            "embedded": _preprocessing_embedded(estimator),
            "payload_checksum": hashlib.sha256(preprocessing_bytes).hexdigest(),
            "feature_order": ordered_features,
        }
    )
    configuration = _json_identity(dict(model_configuration))
    dependencies = _dependency_versions()
    estimator_parameters = _estimator_parameters(estimator)
    model_artifact_id = "selector-model:" + canonical_checksum({
        "campaign": campaign_identity,
        "plan_job": plan_job_identity,
        "component": component_identity,
        "model": model_id,
        "target_horizon_identity": target_horizon_identity,
    })
    model_root = model_artifact_root or (
        component_root / "shared_model_artifact" / "model"
    )
    model_metadata = {
        "model_id": model_id,
        "member_model_identity": member_model_identity,
        "implementation_owner": implementation_owner,
        "model_family": model_family,
        "model_role": ArtifactRole.RESEARCH_FOLD_MODEL.value,
        "model_configuration": configuration,
        "model_configuration_checksum": canonical_checksum(configuration),
        "random_seed": random_seed,
        "training_boundary": dict(training_boundary),
        "training_population_checksum": training_population_checksum,
        "target_horizon_identity": target_horizon_identity,
        "feature_order": ordered_features,
        "feature_order_checksum": canonical_checksum(ordered_features),
        "preprocessing_identity": preprocessing_identity,
        "preprocessing_checksum": hashlib.sha256(preprocessing_bytes).hexdigest(),
        "preprocessing_embedded": _preprocessing_embedded(estimator),
        "preprocessing_file": (
            None
            if _preprocessing_embedded(estimator)
            else "model/preprocessing.json"
        ),
        "fitted_feature_count": len(ordered_features),
        "sklearn_version": dependencies["scikit-learn"],
        "estimator_parameters": estimator_parameters,
        "serialized_python_type": (
            f"{type(estimator).__module__}.{type(estimator).__qualname__}"
        ),
        "trusted_python_serialization": True,
        "serialization_format": "joblib",
        "model_file": "model/estimator.joblib",
        "model_byte_checksum": hashlib.sha256(estimator_bytes).hexdigest(),
        "campaign_identity": campaign_identity,
        "plan_job_identity": plan_job_identity,
        "component_identity": component_identity,
        "component_runner": component_runner,
        "runtime_owner": runtime_owner,
        "decision_date": decision_date,
        "fold_identity": fold_identity,
        "training_row_artifact_identity": training_row_artifact_identity,
        "prediction_row_artifact_identity": prediction_row_artifact_identity,
        "source_schema_guarantee_identity": source_schema_guarantee_identity,
        "input_package_identity": input_package_identity,
        "run_identity": run_identity,
        "contextual_evidence": dict(contextual_evidence or {}),
    }
    if model_id == "contextual_elastic_net":
        context = model_metadata["contextual_evidence"]
        required_context = (
            "ordered_stock_feature_ids",
            "ordered_market_context_ids",
            "interaction_specification",
            "interaction_output_order",
            "context_source_guarantee_identity",
        )
        if any(context.get(field) in (None, "", []) for field in required_context):
            raise ValueError("Contextual model evidence is incomplete")
    if model_id == "ordered_logit_ranker":
        model_metadata["ordered_logit_evidence"] = _ordered_logit_evidence(estimator)
    model_template = build_artifact_manifest(
        artifact_id=model_artifact_id,
        artifact_type=ArtifactType.FITTED_MODEL.value,
        artifact_subtype="SELECTOR_SKLEARN_MODEL",
        artifact_role=ArtifactRole.RESEARCH_FOLD_MODEL.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[
            {
                "identity": training_row_artifact_identity,
                "checksum": training_population_checksum,
            }
        ],
        source_artifacts=[
            {
                "identity": input_package_identity,
                "checksum": input_population_checksum,
            }
        ],
        configuration_identity=canonical_checksum(configuration),
        configuration_checksum=canonical_checksum(configuration),
        source_git_commit=source_git_commit,
        serialization_handler="SKLEARN_PIPELINE",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=dependencies,
        claims={
            "fitting_performed": True,
            "prediction_performed": True,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
        },
        fitted_model_contract_version=FITTED_MODEL_CONTRACT,
        selector_package_contract_version=SELECTOR_SKLEARN_PACKAGE_CONTRACT,
        model_metadata=model_metadata,
    )
    owned = {
        "model/estimator.joblib": estimator_bytes,
        "metadata/feature_schema.json": _json_bytes(
            {
                "ordered_features": ordered_features,
                "feature_order_checksum": canonical_checksum(ordered_features),
            }
        ),
        "metadata/configuration.json": _json_bytes(configuration),
        "metadata/dependency_versions.json": _json_bytes(dependencies),
    }
    if not _preprocessing_embedded(estimator):
        owned["model/preprocessing.json"] = preprocessing_bytes
    model_status, model_manifest = publish_artifact_package(
        model_root, model_template, owned
    )
    validate_model_artifact_manifest(model_manifest)
    public_root = published_component_root or component_root
    if not publish_prediction_binding:
        public_model_root = model_root
        if published_component_root is not None:
            public_model_root = published_component_root / model_root.relative_to(
                component_root
            )
        return {
            "completion_status": "COMPLETE",
            "compatible_skip_status": model_status,
            "artifact_identity": model_manifest["artifact_id"],
            "logical_checksum": model_manifest["logical_checksum"],
            "package_checksum": model_manifest["package_checksum"],
            "preprocessing_identity": preprocessing_identity,
            "feature_order_identity": model_metadata["feature_order_checksum"],
            "model_package_path": str(public_model_root),
            "public_component_root": str(public_root),
        }

    prediction_checksum = _sha256(prediction_path)
    prediction_artifact_id = "selector-prediction:" + canonical_checksum({
        "component": component_identity,
        "checksum": prediction_checksum,
    })
    binding = {
        "contract_version": PREDICTION_BINDING_CONTRACT,
        "fitted_model_artifact_identity": model_manifest["artifact_id"],
        "fitted_model_artifact_checksum": model_manifest["logical_checksum"],
        "fitted_model_package_checksum": model_manifest["package_checksum"],
        "preprocessing_identity": preprocessing_identity,
        "input_population_checksum": input_population_checksum,
        "output_population_checksum": output_population_checksum,
        "prediction_artifact_identity": prediction_artifact_id,
        "prediction_artifact_checksum": prediction_checksum,
        "prediction_schema": list(prediction_schema),
        "prediction_count": prediction_count,
        "campaign_identity": campaign_identity,
        "component_identity": component_identity,
        "plan_job_identity": plan_job_identity,
        "decision_date": decision_date,
        "horizon": target_horizon_identity,
        "source_git_commit": source_git_commit,
    }
    prediction_template = build_artifact_manifest(
        artifact_id=prediction_artifact_id,
        artifact_type=ArtifactType.PREDICTION_ARTIFACT.value,
        artifact_subtype="SELECTOR_COMPONENT_PREDICTION_BINDING",
        artifact_role=ArtifactRole.RESEARCH_PREDICTIONS.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[
            {"identity": input_package_identity, "checksum": input_population_checksum}
        ],
        source_artifacts=[
            {
                "identity": model_manifest["artifact_id"],
                "checksum": model_manifest["package_checksum"],
            },
            {
                "identity": prediction_row_artifact_identity,
                "checksum": prediction_checksum,
            },
        ],
        configuration_identity=model_manifest["configuration_identity"],
        configuration_checksum=model_manifest["configuration_checksum"],
        source_git_commit=source_git_commit,
        serialization_handler="GENERIC_STAGE_FILES",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=dependencies,
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
    prediction_status, prediction_manifest = publish_artifact_package(
        prediction_root,
        prediction_template,
        {"metadata/prediction_binding.json": _json_bytes(binding)},
    )
    validate_prediction_binding(prediction_manifest, model_manifest)
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": (
            "SKIPPED_COMPATIBLE"
            if model_status == prediction_status == "SKIPPED_COMPATIBLE"
            else "COMPLETE"
        ),
        "artifact_identity": model_manifest["artifact_id"],
        "package_checksum": model_manifest["package_checksum"],
        "preprocessing_identity": preprocessing_identity,
        "feature_order_identity": model_metadata["feature_order_checksum"],
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction_manifest["artifact_id"],
        "model_package_path": str(
            public_root / "shared_model_artifact" / "model"
        ),
        "prediction_package_path": str(
            public_root / "shared_model_artifact" / "prediction"
        ),
    }


def resolve_selector_model_package(
    *,
    job: Mapping[str, Any],
    component_result: Mapping[str, Any] | None = None,
    run_identity: str,
) -> Mapping[str, Any] | None:
    if component_result is None:
        manifest_path = Path(
            str(job.get("authoritative_output_root") or "")
        ) / "manifest.json"
    else:
        manifest_path = Path(str(component_result.get("manifest_path") or ""))
    if not manifest_path.is_file():
        return None
    component = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    prediction_path = Path(str(component.get("prediction_artifact_path") or ""))
    if (
        component.get("publication_status") != "complete"
        or component.get("production_plan_job_checksum")
        != job.get("logical_checksum")
        or component.get("selector_model_identity") != job.get("model_id")
        or component.get("prediction_date") != job.get("prediction_date")
        or not prediction_path.is_file()
        or _sha256(prediction_path) != component.get("prediction_checksum")
    ):
        return None
    if job.get("model_id") in {
        "multi_horizon_ridge",
        "multi_horizon_elastic_net",
    }:
        from core.research.ml.stock_level.selector_multihorizon_model_artifacts import (
            resolve_multihorizon_package,
        )

        return resolve_multihorizon_package(
            component_root=root,
            job=job,
            component_manifest=component,
            run_identity=run_identity,
        )
    if job.get("model_id") in {
        "lightgbm_rank_xendcg",
        "lightgbm_lambdarank",
    }:
        from core.research.ml.stock_level.selector_lightgbm_model_artifacts import (
            resolve_lightgbm_package,
        )

        return resolve_lightgbm_package(
            component_root=root,
            job=job,
            component_manifest=component,
            run_identity=run_identity,
        )
    model_root = root / "shared_model_artifact" / "model"
    prediction_root = root / "shared_model_artifact" / "prediction"
    if not model_root.exists() or not prediction_root.exists():
        return None
    model = validate_artifact_package(model_root)
    prediction = validate_artifact_package(prediction_root)
    validate_model_artifact_manifest(model)
    validate_prediction_binding(prediction, model)
    metadata = model["model_metadata"]
    if (
        metadata.get("campaign_identity") != job.get("campaign_identity")
        or metadata.get("plan_job_identity") != job.get("job_id")
        or metadata.get("decision_date") != job.get("prediction_date")
        or metadata.get("run_identity") not in {None, run_identity}
    ):
        raise ValueError("INCOMPATIBLE_MODEL_ARTIFACT")
    binding = prediction["prediction_model_binding"]
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": "SKIPPED_COMPATIBLE",
        "artifact_identity": model["artifact_id"],
        "package_checksum": model["package_checksum"],
        "preprocessing_identity": metadata["preprocessing_identity"],
        "feature_order_identity": metadata["feature_order_checksum"],
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction["artifact_id"],
        "model_package_path": str(model_root),
        "prediction_package_path": str(prediction_root),
    }


def reconstruct_selector_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_order: Sequence[str],
    preprocessing: Mapping[str, Any],
) -> list[list[float]]:
    ordered = list(feature_order)
    if list(preprocessing.get("ordered_feature_ids") or ordered) != ordered:
        raise ValueError("Feature order differs from preprocessing state")
    matrix = [[float(row[name]) for name in ordered] for row in rows]
    location = preprocessing.get("location")
    scale = preprocessing.get("scale")
    if location is None or scale is None:
        return matrix
    if len(location) != len(ordered) or len(scale) != len(ordered):
        raise ValueError("Preprocessing state length mismatch")
    return [
        [
            (value - float(center)) / float(width)
            for value, center, width in zip(row, location, scale)
        ]
        for row in matrix
    ]


def load_trusted_selector_model(
    package_root: Path, *, trusted_artifact: bool
) -> Any:
    payload = read_trusted_model_bytes(
        package_root,
        "model/estimator.joblib",
        trusted_artifact=trusted_artifact,
    )
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("MODEL_RELOAD_FAILED: joblib unavailable") from exc
    return joblib.load(io.BytesIO(payload))


def reconstruct_contextual_selector_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    preprocessing: Mapping[str, Any],
    contextual_evidence: Mapping[str, Any],
) -> list[list[float]]:
    stock_ids = list(preprocessing.get("stock_feature_ids") or ())
    context_ids = list(preprocessing.get("context_feature_ids") or ())
    interactions = list(
        contextual_evidence.get("interaction_specification") or ()
    )
    expected_order = list(
        contextual_evidence.get("interaction_output_order") or ()
    )
    if not stock_ids or not context_ids or not expected_order:
        raise ValueError("Contextual preprocessing evidence is incomplete")
    stock_index = {value: index for index, value in enumerate(stock_ids)}
    context_index = {value: index for index, value in enumerate(context_ids)}
    output: list[list[float]] = []
    actual_order = [f"stock:{value}" for value in stock_ids]
    include_context = any(
        value.startswith("context:") for value in expected_order
    )
    if include_context:
        actual_order.extend(f"context:{value}" for value in context_ids)
    actual_order.extend(
        f"interaction:{row['interaction_id']}" for row in interactions
    )
    if actual_order != expected_order:
        raise ValueError("Contextual interaction output order mismatch")
    for row in rows:
        stock = _scale_row(
            [float(row[value]) for value in stock_ids],
            preprocessing["stock_location"],
            preprocessing["stock_scale"],
            preprocessing.get("stock_lower"),
            preprocessing.get("stock_upper"),
        )
        context = _scale_row(
            [float(row[value]) for value in context_ids],
            preprocessing["context_location"],
            preprocessing["context_scale"],
            preprocessing.get("context_lower"),
            preprocessing.get("context_upper"),
        )
        values = list(stock)
        if include_context:
            values.extend(context)
        values.extend(
            stock[stock_index[item["stock_feature_id"]]]
            * context[context_index[item["market_context_id"]]]
            for item in interactions
        )
        output.append(values)
    return output


def _scale_row(
    values: Sequence[float],
    location: Sequence[float],
    scale: Sequence[float],
    lower: Sequence[float] | None,
    upper: Sequence[float] | None,
) -> list[float]:
    bounded = list(values)
    if lower is not None and upper is not None:
        bounded = [
            min(max(value, float(low)), float(high))
            for value, low, high in zip(bounded, lower, upper)
        ]
    return [
        (value - float(center)) / float(width)
        for value, center, width in zip(bounded, location, scale)
    ]


def _preprocessing_embedded(estimator: Any) -> bool:
    module = type(estimator).__module__
    return (
        (
            module.startswith("sklearn.")
            and (
                hasattr(estimator, "steps")
                or hasattr(estimator, "regressor_")
            )
        )
        or (
            hasattr(estimator, "imputer_")
            and hasattr(estimator, "scaler_")
        )
    )


def _ordered_logit_evidence(estimator: Any) -> dict[str, Any]:
    required = ("beta_", "thresholds_", "classes_", "imputer_", "scaler_")
    if any(not hasattr(estimator, field) for field in required):
        raise ValueError("Ordered Logit fitted state is incomplete")
    return {
        "coefficient_values": estimator.beta_.tolist(),
        "threshold_values": estimator.thresholds_.tolist(),
        "class_order": estimator.classes_.tolist(),
        "link_identity": "cumulative_logit",
        "distribution_identity": "logistic",
        "imputation_statistics": estimator.imputer_.statistics_.tolist(),
        "scaling_mean": estimator.scaler_.mean_.tolist(),
        "scaling_scale": estimator.scaler_.scale_.tolist(),
        "dependency_owner": "core.research.ml.ranking.OrderedLogitRanker",
    }


def _estimator_parameters(estimator: Any) -> Mapping[str, Any]:
    getter = getattr(estimator, "get_params", None)
    return _json_identity(dict(getter(deep=True))) if callable(getter) else {}


def _json_identity(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_identity(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_identity(value.item())
        except (TypeError, ValueError):
            pass
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "parameters": (
            _json_identity(value.get_params(deep=True))
            if callable(getattr(value, "get_params", None))
            else str(value)
        ),
    }


def _joblib_bytes(value: Any) -> bytes:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("MODEL_SERIALIZATION_FAILED: joblib unavailable") from exc
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    return buffer.getvalue()


def _dependency_versions() -> dict[str, str]:
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


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, indent=2, sort_keys=True, default=str
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prediction_file(
    path: Path, *, expected_schema: Sequence[str], expected_count: int
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_schema = list(reader.fieldnames or ())
        count = sum(1 for _ in reader)
    if actual_schema != list(expected_schema):
        raise ValueError("Prediction schema mismatch")
    if count != expected_count:
        raise ValueError("Prediction count mismatch")
