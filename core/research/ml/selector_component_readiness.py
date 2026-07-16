from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.registries import RegistryResolver, load_registry_bundle


READINESS_CONTRACT = "selector_component_readiness.v2"
COMPONENT_SCHEMA = "authoritative_selector_component_v1"
PARENT_GATE_CONTRACT = "selector_parent_publication_gate.v1"
VERIFIED_STRICT_OOS = "VERIFIED_STRICT_OOS"
MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")
DATES = ("2024-03-15", "2024-09-16", "2025-03-17", "2025-09-15", "2026-03-16")
STATE_PRIORITY = (
    "MALFORMED", "NON_AUTHORITATIVE_ROOT", "INCOMPLETE",
    "MODEL_IDENTITY_MISMATCH", "DATASET_IDENTITY_MISMATCH",
    "FEATURE_CONTRACT_MISMATCH", "TARGET_CONTRACT_MISMATCH",
    "RANKING_CONTRACT_MISMATCH", "FOLD_IDENTITY_MISSING",
    "TEMPORAL_LEAKAGE", "POPULATION_MISMATCH",
    "PREDICTION_INCOMPLETE", "ARTIFACT_CHECKSUM_MISMATCH",
    "ARTIFACT_LINK_UNVERIFIED", "SMOKE_OUTPUT_REJECTED",
)


def assess_selector_component_readiness(
    *,
    parent_gate_path: Path,
    authoritative_root: Path,
    selector_dataset_root: Path,
    config_path: Path,
    approved_component_roots: tuple[Path, ...],
) -> dict[str, Any]:
    gate = _read_json(parent_gate_path)
    gate_ready = (
        gate is not None
        and gate.get("gate_contract_version") == PARENT_GATE_CONTRACT
        and gate.get("status") == "READY"
        and bool(gate.get("logical_checksum"))
    )
    root_authoritative = authoritative_root.resolve() in {
        path.resolve() for path in approved_component_roots
    }
    resolver = RegistryResolver(load_registry_bundle())
    expected_models = {
        model: resolver.resolve("selector_models", model, role="selector").entry.payload
        for model in MODELS
    }
    expected_hashes = {
        model: resolver.resolve("selector_models", model, role="selector").entry.entry_hash
        for model in MODELS
    }
    target = resolver.resolve("target_contracts", "forward_return_10d", role="selector")
    matrix: list[dict[str, Any]] = []

    for prediction_date in DATES:
        for model in MODELS:
            owner = authoritative_root / f"model={model}" / f"date={prediction_date}"
            manifest_path = owner / "manifest.json"
            if not root_authoritative:
                row = _base_row(model, prediction_date, owner, manifest_path, "NON_AUTHORITATIVE_ROOT")
            elif not gate_ready:
                row = _base_row(model, prediction_date, owner, manifest_path, "BLOCKED_PARENT_GATE")
            elif not manifest_path.exists():
                row = _base_row(
                    model, prediction_date, owner, manifest_path,
                    "INCOMPLETE" if owner.exists() else "MISSING",
                )
            elif not _within(manifest_path, authoritative_root.resolve()):
                row = _base_row(model, prediction_date, owner, manifest_path, "NON_AUTHORITATIVE_ROOT")
            else:
                payload = _read_json(manifest_path)
                if payload is None:
                    row = _base_row(model, prediction_date, owner, manifest_path, "MALFORMED")
                else:
                    row = _validate_component(
                        payload=payload, manifest_path=manifest_path,
                        model=model, prediction_date=prediction_date,
                        gate=gate, expected_model=expected_models[model],
                        expected_model_hash=expected_hashes[model],
                        target_id=target.canonical_id,
                        target_hash=target.entry.entry_hash,
                    )
            matrix.append(row)

    matched = _matched_populations(matrix)

    plan = [
        _planned_job(
            row=row, gate=gate or {}, dataset_root=selector_dataset_root,
            authoritative_root=authoritative_root, config_path=config_path,
            model_payload=expected_models[row["model_id"]],
        )
        for row in matrix if row["state"] != "READY"
    ]
    ready_count = sum(row["state"] == "READY" for row in matrix)
    missing_count = sum(row["state"] == "MISSING" for row in matrix)
    invalid_count = len(matrix) - ready_count - missing_count
    blockers = sorted({
        row["state"] for row in matrix if row["state"] not in {"READY", "MISSING"}
    })
    if any(row["status"] == "BLOCKED" and row["ready_model_count"] == len(MODELS)
           for row in matched.values()):
        blockers.append("POPULATION_MISMATCH")
    if missing_count:
        blockers.append("MISSING_COMPONENTS")
    status = (
        "BLOCKED" if not gate_ready or not root_authoritative
        else "READY" if ready_count == len(matrix)
        else "PARTIAL"
    )
    result = {
        "readiness_contract_version": READINESS_CONTRACT,
        "required_models": list(MODELS),
        "required_dates": list(DATES),
        "expected_component_count": len(MODELS) * len(DATES),
        "ready_component_count": ready_count,
        "missing_component_count": missing_count,
        "invalid_component_count": invalid_count,
        "component_matrix": matrix,
        "matched_population_results": [matched[date] for date in DATES],
        "parent_gate_contract": gate.get("gate_contract_version") if gate else None,
        "parent_gate_logical_checksum": gate.get("logical_checksum") if gate else None,
        "dataset_identity": gate.get("selector_dataset_id") if gate else None,
        "dataset_checksum": gate.get("selector_dataset_artifact_checksum") if gate else None,
        "production_plan": plan,
        "blockers": sorted(set(blockers)),
        "warnings": [],
        "overall_status": status,
        "fitting_performed": False,
        "commands_executed": False,
    }
    result["logical_checksum"] = _logical_checksum(result)
    return result


def _validate_component(
    *, payload: Mapping[str, Any], manifest_path: Path, model: str,
    prediction_date: str, gate: Mapping[str, Any],
    expected_model: Mapping[str, Any], expected_model_hash: str,
    target_id: str, target_hash: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    link = payload.get("artifact_link")
    link = link if isinstance(link, Mapping) else {}
    frozen = payload.get("frozen_selector_dataset_identity")
    frozen = frozen if isinstance(frozen, Mapping) else {}

    if payload.get("component_schema_version") != COMPONENT_SCHEMA:
        reasons.append("INCOMPLETE")
    if payload.get("selector_model_identity") != model or payload.get("selector_model_version") != expected_model_hash:
        reasons.append("MODEL_IDENTITY_MISMATCH")
    if payload.get("prediction_date") != prediction_date:
        reasons.append("INCOMPLETE")
    if (
        frozen.get("dataset_id") != gate.get("selector_dataset_id")
        or frozen.get("dataset_checksum") != gate.get("selector_dataset_artifact_checksum")
        or payload.get("symbol_registry_identity") != gate.get("canonical_registry_id")
        or payload.get("daily_stock_spine_identity") != gate.get("daily_spine_id")
    ):
        reasons.append("DATASET_IDENTITY_MISMATCH")
    expected_feature = expected_model.get("feature_schema")
    if (
        payload.get("feature_contract_version") not in {
            expected_feature, Path(str(expected_feature)).stem,
            "canonical_v2_daily_tree_cross_sectional_features_v1"
            if model == "ordered_logit_ranker" else
            "canonical_v2_daily_tabular_features_v1",
        }
        or link.get("feature_schema_hash") != gate.get("selector_feature_schema_checksum")
    ):
        reasons.append("FEATURE_CONTRACT_MISMATCH")
    if (
        payload.get("target_contract_version") != target_id
        or link.get("target_contract_hash") != target_hash
    ):
        reasons.append("TARGET_CONTRACT_MISMATCH")
    expected_ranking = expected_model.get("ranking_problem_contract")
    if expected_ranking and payload.get("ranking_contract_version") != expected_ranking:
        reasons.append("RANKING_CONTRACT_MISMATCH")
    if not payload.get("fold_identity"):
        reasons.append("FOLD_IDENTITY_MISSING")
    if not _temporal_legal(payload):
        reasons.append("TEMPORAL_LEAKAGE")
    row_count = payload.get("prediction_row_count")
    population = payload.get("prediction_population_checksum")
    if not isinstance(row_count, int) or row_count <= 0 or not population:
        reasons.append("PREDICTION_INCOMPLETE")
    artifact_path = Path(str(payload.get("prediction_artifact_path", "")))
    if (
        not artifact_path.is_file()
        or not _within(artifact_path, manifest_path.parents[2].resolve())
        or payload.get("prediction_checksum") != _sha256(artifact_path)
        or link.get("artifact_checksum") != payload.get("prediction_checksum")
    ):
        reasons.append("ARTIFACT_CHECKSUM_MISMATCH")
    if (
        link.get("verification_status") != VERIFIED_STRICT_OOS
        or payload.get("validation_status") != VERIFIED_STRICT_OOS
    ):
        reasons.append("ARTIFACT_LINK_UNVERIFIED")
    if payload.get("publication_status") != "complete":
        reasons.append("INCOMPLETE")
    if payload.get("non_production_smoke"):
        reasons.append("SMOKE_OUTPUT_REJECTED")
    if not payload.get("git_commit"):
        reasons.append("INCOMPLETE")
    reasons = sorted(set(reasons), key=lambda value: STATE_PRIORITY.index(value))
    state = reasons[0] if reasons else "READY"
    return {
        **_base_row(model, prediction_date, manifest_path.parent, manifest_path, state),
        "reasons": reasons,
        "prediction_row_count": row_count,
        "prediction_population_checksum": population,
        "dataset_id": frozen.get("dataset_id"),
        "dataset_checksum": frozen.get("dataset_checksum"),
        "feature_contract": payload.get("feature_contract_version"),
        "target_contract": payload.get("target_contract_version"),
        "ranking_contract": payload.get("ranking_contract_version"),
        "fold_identity": payload.get("fold_identity"),
        "source_commit": payload.get("git_commit"),
    }


def _temporal_legal(payload: Mapping[str, Any]) -> bool:
    try:
        return (
            str(payload["training_start"]) < str(payload["training_cutoff"])
            < str(payload["prediction_date"])
            and str(payload["training_label_available_timestamp_max"])
            <= str(payload["prediction_date"])
        )
    except KeyError:
        return False


def _matched_populations(matrix: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {}
    for date in DATES:
        rows = [row for row in matrix if row["prediction_date"] == date]
        ready = [row for row in rows if row["state"] == "READY"]
        populations = {
            (row.get("prediction_row_count"), row.get("prediction_population_checksum"),
             row.get("dataset_id"), row.get("dataset_checksum"),
             row.get("target_contract"))
            for row in ready
        }
        status = "READY" if len(ready) == len(MODELS) and len(populations) == 1 else "BLOCKED"
        results[date] = {
            "prediction_date": date, "status": status,
            "ready_model_count": len(ready),
            "matched_population": next(iter(populations), None) if status == "READY" else None,
        }
    return results


def _planned_job(
    *, row: Mapping[str, Any], gate: Mapping[str, Any], dataset_root: Path,
    authoritative_root: Path, config_path: Path,
    model_payload: Mapping[str, Any],
) -> dict[str, Any]:
    model = str(row["model_id"]); date = str(row["prediction_date"])
    feature = str(model_payload["feature_schema"])
    ranking = model_payload.get("ranking_problem_contract")
    relevance = model_payload.get("relevance_contract")
    owner = authoritative_root / f"model={model}" / f"date={date}"
    command = (
        f'python main.py --mode ml-stock-selector-bounded --config "{config_path}" '
        f'--selector-dataset-root "{dataset_root}" --oos-start-date {date} '
        f'--oos-end-date {date} --max-oos-dates 1 --model-allowlist {model} '
        f'--baseline-allowlist --selector-feature-schema "{feature}" '
        f'--bounded-output-root "{authoritative_root / f"model={model}"}" '
        f'--sklearn-n-jobs 1'
    )
    job = {
        "job_id": f"selector:{date}:{model}",
        "model_id": model, "prediction_date": date,
        "selector_dataset_root": str(dataset_root),
        "authoritative_output_root": str(owner),
        "feature_schema": feature,
        "target_contract": model_payload.get("target_contract"),
        "ranking_contract": ranking,
        "relevance_contract": relevance,
        "expected_parent_gate_checksum": gate.get("logical_checksum"),
        "expected_dataset_checksum": gate.get("selector_dataset_artifact_checksum"),
        "dependency_state": row["state"],
        "command_template": command,
        "overwrite_policy": "never_overwrite_complete_component",
        "resume_policy": "resume_only_incomplete_owned_component",
    }
    job["logical_checksum"] = _logical_checksum(job)
    return job


def _base_row(model, date, owner, manifest, state):
    return {
        "model_id": model, "prediction_date": date,
        "owner": str(owner), "manifest_path": str(manifest),
        "state": state, "reasons": [] if state in {"READY", "MISSING"} else [state],
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _logical_checksum(payload: Mapping[str, Any]) -> str:
    logical = {
        key: value for key, value in payload.items()
        if key not in {"logical_checksum", "creation_timestamp", "generated_at", "report_path"}
    }
    return hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest().upper()
