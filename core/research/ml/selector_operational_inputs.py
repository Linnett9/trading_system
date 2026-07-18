from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash


PACKAGE_CONTRACT = "selector_component_input_package.v1"
OUTCOME_CONTRACT = "selector_mature_outcomes.v1"
INVENTORY_CONTRACT = "selector_operational_input_inventory.v1"
PLAN_CONTRACT = "selector_operational_component_plan.v1"
MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")
DATES = ("2024-03-15", "2024-09-16", "2025-03-17", "2025-09-15", "2026-03-16")
TARGET = "forward_return_10d"
TARGET_PROVENANCE_CONTRACT_VERSION = "stock_level_target_provenance_v2"
PURGE_SESSIONS = 10
EMBARGO_SESSIONS = 10
SELECTOR_RUN_ID = "20260716T091011Z"
OUTCOME_MATURITY_CUTOFF_CONTRACT = "selector_outcome_maturity_cutoff.v1"
OUTCOME_MATURITY_CUTOFF_INTERPRETATION = (
    "ISO-8601 date means the inclusive UTC calendar day; timestamp means the "
    "instant after normalising an explicit timezone to UTC"
)


def resolve_outcome_maturity_cutoff(
    *,
    outcome_maturity_cutoff: str | None = None,
    evaluation_cutoff: str | None = None,
) -> dict[str, str]:
    supplied = [
        value for value in (outcome_maturity_cutoff, evaluation_cutoff)
        if value is not None
    ]
    if not supplied:
        raise ValueError("outcome_maturity_cutoff is required")
    resolved = [_normalise_outcome_maturity_cutoff(value) for value in supplied]
    if len(set(resolved)) != 1:
        raise ValueError("Conflicting outcome maturity cutoff values")
    return {
        "outcome_maturity_cutoff": resolved[0],
        "outcome_maturity_cutoff_contract": OUTCOME_MATURITY_CUTOFF_CONTRACT,
        "outcome_maturity_cutoff_interpretation": OUTCOME_MATURITY_CUTOFF_INTERPRETATION,
    }


def outcome_is_mature(label_available_timestamp: str, cutoff: str) -> bool:
    return _cutoff_instant(label_available_timestamp, field="label_available_timestamp") <= _cutoff_instant(
        cutoff, field="outcome_maturity_cutoff"
    )


def _normalise_outcome_maturity_cutoff(value: str) -> str:
    text = str(value).strip()
    instant = _cutoff_instant(text, field="outcome_maturity_cutoff")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return instant.isoformat().replace("+00:00", "Z")


def _cutoff_instant(value: str, *, field: str) -> datetime:
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.combine(date.fromisoformat(text), time.max, tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Malformed {field}") from exc
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Malformed {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Malformed {field}: timestamp timezone is required")
    return parsed.astimezone(timezone.utc)


def build_selector_component_plan(
    *,
    dataset_manifest: Mapping[str, Any],
    parent_gate: Mapping[str, Any],
    output_root: Path,
    campaign_id: str,
    source_commit: str,
    selector_run_id: str = SELECTOR_RUN_ID,
    model_ids: Sequence[str] = MODELS,
    decision_dates: Sequence[str] = DATES,
    random_seed: int = 42,
    write: bool = True,
    experiment_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic first-campaign selector component plan.

    This function is deliberately plan-only: it resolves registry identities and
    output ownership, but never fits models, reads selector rows, publishes
    components, launches subprocesses, or imports execution owners.
    """
    if not str(source_commit or "").strip():
        raise ValueError("source_commit is required")
    if not str(campaign_id or "").strip():
        raise ValueError("campaign_id is required")
    _validate_plan_parents(
        dataset_manifest=dataset_manifest,
        parent_gate=parent_gate,
        selector_run_id=selector_run_id,
        decision_dates=decision_dates,
    )
    resolver = RegistryResolver(load_registry_bundle())
    target = resolver.resolve(
        "target_contracts",
        str(dataset_manifest.get("target_contract_id") or TARGET),
        role="selector",
    )
    if target.canonical_id != TARGET:
        raise ValueError("Target contract mismatch")
    target_hash = str(dataset_manifest.get("target_contract_hash") or "")
    if target_hash != target.entry.entry_hash:
        raise ValueError("Target contract hash mismatch")
    folds = _folds_by_date(parent_gate)
    rows: list[dict[str, Any]] = []
    for decision_date in decision_dates:
        fold = folds.get(str(decision_date))
        if not fold:
            raise ValueError(f"Strict-OOS fold missing: {decision_date}")
        for model_id in model_ids:
            rows.append(
                _build_component_plan_row(
                    resolver=resolver,
                    dataset_manifest=dataset_manifest,
                    parent_gate=parent_gate,
                    target_hash=target_hash,
                    output_root=output_root,
                    campaign_id=campaign_id,
                    selector_run_id=selector_run_id,
                    model_id=str(model_id),
                    decision_date=str(decision_date),
                    fold=fold,
                    random_seed=random_seed,
                    source_commit=str(source_commit),
                )
            )
    _validate_component_plan_rows(rows)
    manifest = {
        "plan_contract_version": PLAN_CONTRACT,
        "campaign_id": campaign_id,
        "selector_run_id": selector_run_id,
        "component_count": len(rows),
        "fitted_component_count": len(rows),
        "control_models": ["momentum"],
        "fitted_models": list(model_ids),
        "decision_dates": list(decision_dates),
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_checksum": dataset_manifest["dataset_checksum"],
        "dataset_manifest_path": str(dataset_manifest["dataset_manifest_path"]),
        "daily_spine_id": dataset_manifest["daily_spine_id"],
        "daily_spine_manifest_path": str(dataset_manifest["daily_spine_manifest_path"]),
        "symbol_registry_id": dataset_manifest["symbol_registry_id"],
        "symbol_registry_manifest_path": str(dataset_manifest["symbol_registry_manifest_path"]),
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "parent_gate_logical_checksum": parent_gate["logical_checksum"],
        "source_commit": str(source_commit),
        "plans_only": True,
        "model_fitting_performed": False,
        "prediction_generation_performed": False,
        "component_publication_performed": False,
        "components": rows,
    }
    manifest["logical_checksum"] = _logical(manifest)
    if experiment_ledger_path is not None:
        from core.research.ml.experiment_ledger import register_selector_plan
        register_selector_plan(experiment_ledger_path, manifest)
    if write:
        _publish_plan(output_root, manifest)
        manifest = {
            **manifest,
            "component_plan_path": str(output_root / "component_plan.json"),
            "component_plan_markdown_path": str(output_root / "component_plan.md"),
        }
    return manifest


def validate_selector_component_plan(path: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        payload = _json(path)
        if payload.get("plan_contract_version") != PLAN_CONTRACT:
            reasons.append("PLAN_CONTRACT_MISMATCH")
        if payload.get("logical_checksum") != _logical(payload):
            reasons.append("PLAN_CHECKSUM_MISMATCH")
        rows = list(payload.get("components") or [])
        try:
            _validate_component_plan_rows(rows)
        except ValueError as exc:
            reasons.append(str(exc).upper().replace(" ", "_"))
        if payload.get("component_count") != 15 or len(rows) != 15:
            reasons.append("PLAN_COMPONENT_COUNT_MISMATCH")
        if "momentum" in set(payload.get("fitted_models") or []):
            reasons.append("MOMENTUM_IS_NOT_FITTED")
    except (OSError, ValueError, TypeError):
        payload = {}
        reasons.append("PLAN_MISSING_OR_CORRUPT")
    return {
        "status": "READY" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "plan": payload,
    }


def build_operational_inputs(
    *, plan: Mapping[str, Any], dataset_manifest: Mapping[str, Any],
    parent_gate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    output_root: Path, outcome_maturity_cutoff: str | None = None,
    evaluation_cutoff: str | None = None, source_git_commit: str | None = None,
    selector_run_id: str = SELECTOR_RUN_ID,
    max_workers: int = 4,
) -> dict[str, Any]:
    if max_workers < 1 or max_workers > 6:
        raise ValueError("max_workers must be between 1 and 6")
    jobs = list(plan.get("production_plan") or [])
    _validate_plan(jobs)
    _validate_parents(dataset_manifest, parent_gate)
    cutoff = resolve_outcome_maturity_cutoff(
        outcome_maturity_cutoff=outcome_maturity_cutoff,
        evaluation_cutoff=evaluation_cutoff,
    )
    source_commit = source_git_commit or _git_commit()
    canonical_rows = sorted((dict(row) for row in rows), key=_row_key)
    _validate_source_rows(canonical_rows)
    target = RegistryResolver(load_registry_bundle()).resolve("target_contracts", TARGET, role="selector")
    packages = []
    date_populations: dict[str, str] = {}
    def build(job):
        return _build_package(
            job=job, rows=canonical_rows, dataset=dataset_manifest, gate=parent_gate,
            target_hash=target.entry.entry_hash, output_root=output_root,
            source_commit=source_commit,
        )
    ordered_jobs = sorted(jobs, key=lambda row: str(row["job_id"]))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        packages = list(pool.map(build, ordered_jobs))
    for job, package in zip(ordered_jobs, packages):
        prior = date_populations.setdefault(job["prediction_date"], package["prediction_ordered_population_checksum"])
        if prior != package["prediction_ordered_population_checksum"]:
            raise ValueError("Model-specific prediction populations differ")
    outcomes = _build_outcomes(
        rows=canonical_rows, dataset=dataset_manifest, target_hash=target.entry.entry_hash,
        output_root=output_root, outcome_maturity_cutoff=cutoff["outcome_maturity_cutoff"],
        expected_populations=date_populations, source_commit=source_commit,
    )
    inventory = {
        "inventory_contract_version": INVENTORY_CONTRACT,
        "selector_run_id": selector_run_id,
        "operational_panel_id": "selector_operational_panel_v1",
        "required_component_count": 15,
        "required_models": list(MODELS), "required_dates": list(DATES),
        "selector_dataset_id": dataset_manifest["dataset_id"],
        "selector_dataset_checksum": dataset_manifest["dataset_checksum"],
        "parent_gate_checksum": parent_gate["logical_checksum"],
        "production_plan_checksum": plan["logical_checksum"],
        "packages": [{
            "job_id": row["production_plan_job_id"],
            "model_id": row["model_id"], "prediction_date": row["prediction_date"],
            "training_rows_path": row["training_rows_path"],
            "prediction_rows_path": row["prediction_rows_path"],
            "package_manifest_path": row["manifest_path"],
            "package_logical_checksum": row["logical_checksum"],
            "prediction_ordered_population_checksum": row["prediction_ordered_population_checksum"],
        } for row in packages],
        "mature_outcome_path": outcomes["outcome_path"],
        "mature_outcome_checksum": outcomes["artifact_checksum"],
        "mature_outcome_manifest_path": outcomes["manifest_path"],
        **cutoff,
        "selector_dataset_scan_count": 1,
    }
    inventory["logical_checksum"] = _logical(inventory)
    path = output_root / "inventory.json"
    _publish_file(path, inventory)
    return {**inventory, "inventory_path": str(path)}


def validate_inventory(
    path: Path, *, readiness: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None, expected_dataset_id: str | None = None,
    expected_dataset_checksum: str | None = None, expected_parent_gate_checksum: str | None = None,
    expected_outcome_maturity_cutoff: str | None = None,
    expected_evaluation_cutoff: str | None = None,
) -> dict[str, Any]:
    value = _json(path)
    reasons = []
    try:
        cutoff = resolve_outcome_maturity_cutoff(
            outcome_maturity_cutoff=value.get("outcome_maturity_cutoff"),
            evaluation_cutoff=value.get("evaluation_cutoff"),
        )
        expected_cutoff = resolve_outcome_maturity_cutoff(
            outcome_maturity_cutoff=expected_outcome_maturity_cutoff,
            evaluation_cutoff=expected_evaluation_cutoff,
        )["outcome_maturity_cutoff"] if (
            expected_outcome_maturity_cutoff is not None
            or expected_evaluation_cutoff is not None
        ) else None
    except ValueError:
        cutoff = {"outcome_maturity_cutoff": ""}
        expected_cutoff = None
        reasons.append("OUTCOME_MATURITY_CUTOFF_INVALID")
    if value.get("inventory_contract_version") != INVENTORY_CONTRACT:
        reasons.append("INVENTORY_CONTRACT_MISMATCH")
    if value.get("logical_checksum") != _logical(value):
        reasons.append("INVENTORY_CHECKSUM_MISMATCH")
    expectations = {
        "selector_run_id": expected_run_id,
        "selector_dataset_id": expected_dataset_id,
        "selector_dataset_checksum": expected_dataset_checksum,
        "parent_gate_checksum": expected_parent_gate_checksum,
    }
    for key, expected in expectations.items():
        if expected is not None and value.get(key) != expected:
            reasons.append(f"{key.upper()}_MISMATCH")
    if expected_cutoff is not None and cutoff["outcome_maturity_cutoff"] != expected_cutoff:
        reasons.append("OUTCOME_MATURITY_CUTOFF_MISMATCH")
    packages = list(value.get("packages") or [])
    ids = [str(row.get("job_id")) for row in packages]
    expected = {f"selector:{date}:{model}" for date in DATES for model in MODELS}
    if len(ids) != 15 or set(ids) != expected or len(ids) != len(set(ids)):
        reasons.append("INVENTORY_JOB_COVERAGE_MISMATCH")
    if any(row.get("model_id") not in MODELS for row in packages):
        reasons.append("CHALLENGER_INVENTORY")
    if readiness is not None:
        plan = list(readiness.get("production_plan") or [])
        planned_ids = {str(row.get("job_id")) for row in plan}
        ready_ids = {
            f"selector:{row['prediction_date']}:{row['model_id']}"
            for row in readiness.get("component_matrix", []) if row.get("state") == "READY"
        }
        if set(ids) != planned_ids | ready_ids:
            reasons.append("INVENTORY_PLAN_MISMATCH")
        if value.get("production_plan_checksum") != readiness.get("logical_checksum"):
            reasons.append("PLAN_CHECKSUM_MISMATCH")
    for row in packages:
        manifest_path = Path(str(row.get("package_manifest_path", "")))
        training = Path(str(row.get("training_rows_path", "")))
        prediction = Path(str(row.get("prediction_rows_path", "")))
        try:
            manifest = _json(manifest_path)
            if manifest.get("package_contract_version") != PACKAGE_CONTRACT:
                reasons.append(f"PACKAGE_CONTRACT:{row.get('job_id')}")
            if manifest.get("logical_checksum") != row.get("package_logical_checksum") or manifest.get("logical_checksum") != _logical(manifest):
                reasons.append(f"PACKAGE_CHECKSUM:{row.get('job_id')}")
            if _sha(training) != manifest.get("training_artifact_checksum") or _sha(prediction) != manifest.get("prediction_artifact_checksum"):
                reasons.append(f"PACKAGE_ARTIFACT_CHECKSUM:{row.get('job_id')}")
        except (OSError, ValueError):
            reasons.append(f"PACKAGE_MISSING:{row.get('job_id')}")
    try:
        outcome = Path(str(value["mature_outcome_path"]))
        outcome_manifest = _json(Path(str(value["mature_outcome_manifest_path"])))
        if _sha(outcome) != value.get("mature_outcome_checksum") or outcome_manifest.get("outcome_contract_version") != OUTCOME_CONTRACT:
            reasons.append("OUTCOME_CHECKSUM_MISMATCH")
        outcome_cutoff = resolve_outcome_maturity_cutoff(
            outcome_maturity_cutoff=outcome_manifest.get("outcome_maturity_cutoff"),
            evaluation_cutoff=outcome_manifest.get("evaluation_cutoff"),
        )
        if outcome_cutoff["outcome_maturity_cutoff"] != cutoff["outcome_maturity_cutoff"]:
            reasons.append("OUTCOME_MATURITY_CUTOFF_MISMATCH")
        if outcome_manifest.get("logical_checksum") != _logical(outcome_manifest):
            reasons.append("OUTCOME_MANIFEST_CHECKSUM_MISMATCH")
        with outcome.open("r", encoding="utf-8", newline="") as handle:
            outcome_rows = list(csv.DictReader(handle))
        dates = {row.get("prediction_date") for row in outcome_rows}
        if dates != set(DATES): reasons.append("OUTCOME_DATE_COVERAGE_MISMATCH")
        if len(outcome_rows) != int(outcome_manifest.get("row_count", -1)): reasons.append("OUTCOME_ROW_COUNT_MISMATCH")
        if any(row.get("target_contract") != TARGET or row.get("target_horizon") != "10_sessions" for row in outcome_rows): reasons.append("OUTCOME_TARGET_MISMATCH")
        if any(
            row.get("maturity_status") != "MATURE"
            or not outcome_is_mature(
                row.get("label_available_timestamp", ""),
                cutoff["outcome_maturity_cutoff"],
            )
            for row in outcome_rows
        ):
            reasons.append("OUTCOME_IMMATURE")
    except (OSError, KeyError, ValueError):
        reasons.append("OUTCOME_OWNER_MISSING")
    return {"status": "READY" if not reasons else "BLOCKED", "reasons": sorted(set(reasons)), "inventory": value}


def _build_package(*, job, rows, dataset, gate, target_hash, output_root, source_commit):
    model, date = str(job["model_id"]), str(job["prediction_date"])
    resolver = RegistryResolver(load_registry_bundle())
    model_entry = resolver.resolve("selector_models", model, role="selector").entry
    feature_path = Path(str(model_entry.payload["feature_schema"]))
    feature_names = [row["name"] for row in _json(feature_path)["features"]]
    feature_hash = _sha(feature_path)
    dates = sorted({str(row["decision_session_date"]) for row in rows})
    if date not in dates:
        raise ValueError(f"Prediction date absent: {date}")
    prediction_index = dates.index(date)
    cutoff_index = prediction_index - PURGE_SESSIONS
    if cutoff_index < 0:
        raise ValueError(f"Insufficient purge history: {date}")
    training_cutoff = dates[cutoff_index]
    prediction = [dict(row) for row in rows if str(row["decision_session_date"]) == date]
    candidates = [dict(row) for row in rows if str(row["decision_session_date"]) < training_cutoff]
    if any(str(row["label_available_timestamp"]) > training_cutoff for row in candidates):
        raise ValueError(f"Immature training label before cutoff: {date}")
    training = candidates
    _validate_package_rows(training, prediction, date, training_cutoff, feature_names, model)
    if model == "ordered_logit_ranker":
        _add_relevance(training)
    training = sorted(training, key=_row_key)
    prediction = sorted(prediction, key=lambda row: (str(row["asset_id"]), str(row["row_id"])))
    owner = output_root / "component_inputs" / _safe_job(str(job["job_id"]))
    manifest = {
        "package_contract_version": PACKAGE_CONTRACT,
        "package_id": f"selector-input:{job['job_id']}:{job['logical_checksum'][:16]}",
        "production_plan_job_id": job["job_id"],
        "production_plan_job_checksum": job["logical_checksum"],
        "model_id": model, "model_entry_hash": model_entry.entry_hash,
        "prediction_date": date,
        "selector_dataset_id": dataset["dataset_id"],
        "selector_dataset_checksum": dataset["dataset_checksum"],
        "target_contract_id": TARGET, "target_contract_hash": target_hash,
        "feature_contract_id": str(feature_path), "feature_contract_hash": feature_hash,
        "ranking_contract": job.get("ranking_contract"),
        "relevance_contract": job.get("relevance_contract"),
        "training_start": min(str(row["decision_session_date"]) for row in training),
        "training_cutoff": training_cutoff, "prediction_cutoff": date,
        "maximum_label_availability_timestamp": max(str(row["label_available_timestamp"]) for row in training),
        "purge_sessions": PURGE_SESSIONS, "embargo_sessions": EMBARGO_SESSIONS,
        "fold_identity": canonical_hash({
            "dataset": dataset["dataset_checksum"], "date": date,
            "training_cutoff": training_cutoff, "purge": PURGE_SESSIONS, "embargo": EMBARGO_SESSIONS,
        }),
        "training_row_count": len(training), "prediction_row_count": len(prediction),
        "training_ordered_population_checksum": canonical_hash([row["row_id"] for row in training]),
        "prediction_ordered_population_checksum": canonical_hash([row["row_id"] for row in prediction]),
        "source_artifact_checksums": {"selector_dataset": dataset["dataset_checksum"], "parent_gate": gate["logical_checksum"]},
        "source_git_commit": source_commit, "publication_status": "complete", "validation_status": "VERIFIED",
    }
    return _publish_package(owner, training, prediction, manifest)


def _build_outcomes(*, rows, dataset, target_hash, output_root, outcome_maturity_cutoff, expected_populations, source_commit):
    selected = [dict(row) for row in rows if str(row["decision_session_date"]) in DATES]
    by_date = {}
    for row in selected:
        date = str(row["decision_session_date"])
        by_date.setdefault(date, []).append(row)
    if set(by_date) != set(DATES):
        raise ValueError("Outcome date coverage incomplete")
    output = []
    populations = {}
    for date in DATES:
        ordered = sorted(by_date[date], key=lambda row: (str(row["asset_id"]), str(row["row_id"])))
        population = canonical_hash([row["row_id"] for row in ordered])
        if population != expected_populations[date]:
            raise ValueError(f"Outcome population mismatch: {date}")
        populations[date] = population
        for row in ordered:
            available = str(row["label_available_timestamp"])
            if not outcome_is_mature(available, outcome_maturity_cutoff):
                raise ValueError(f"Immature outcome: {row['row_id']}")
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"],
                "symbol": row.get("canonical_symbol", row.get("symbol")),
                "prediction_date": date, "outcome_field": "actual_forward_return_10d",
                "actual_forward_return_10d": row["actual_forward_return_10d"],
                "benchmark_return": row.get("actual_benchmark_return_10d", ""),
                "target_contract": TARGET, "target_contract_hash": target_hash,
                "target_horizon": "10_sessions", "label_available_timestamp": available,
                "maturity_status": "MATURE", "selector_dataset_id": dataset["dataset_id"],
                "selector_dataset_checksum": dataset["dataset_checksum"],
                "outcome_source_identity": dataset["dataset_checksum"],
                "ordered_population_checksum": population,
            })
    root = output_root / "mature_outcomes"
    artifact = root / "outcomes.csv"
    manifest_path = root / "manifest.json"
    temporary = root.with_name(f".{root.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    _write_csv(temporary / "outcomes.csv", output)
    checksum = _sha(temporary / "outcomes.csv")
    manifest = {
        "outcome_contract_version": OUTCOME_CONTRACT,
        "target_contract_id": TARGET, "target_contract_hash": target_hash,
        "target_horizon": "10_sessions",
        **resolve_outcome_maturity_cutoff(
            outcome_maturity_cutoff=outcome_maturity_cutoff
        ),
        "selector_dataset_id": dataset["dataset_id"], "selector_dataset_checksum": dataset["dataset_checksum"],
        "required_dates": list(DATES), "row_count": len(output),
        "date_population_checksums": populations, "artifact_checksum": checksum,
        "outcome_path": str(artifact), "source_git_commit": source_commit,
        "publication_status": "complete", "validation_status": "VERIFIED_MATURE",
    }
    manifest["logical_checksum"] = _logical(manifest)
    _write_json(temporary / "manifest.json", manifest)
    _atomic_directory(temporary, root, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _validate_plan(jobs):
    expected = {f"selector:{date}:{model}" for date in DATES for model in MODELS}
    ids = [str(row.get("job_id")) for row in jobs]
    if len(ids) != 15 or set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError("Production plan must contain exactly the 15 base jobs")
    if any(row.get("model_id") not in MODELS for row in jobs):
        raise ValueError("Challenger model in base plan")
    for row in jobs:
        if row.get("logical_checksum") != _logical(row):
            raise ValueError("Production-plan job checksum mismatch")


def _validate_parents(dataset, gate):
    if dataset.get("publication_status") != "complete" or dataset.get("validation_status") != "VERIFIED":
        raise ValueError("Selector dataset is not authoritative")
    if gate.get("status") != "READY" or gate.get("gate_contract_version") != "selector_parent_publication_gate.v1":
        raise ValueError("Parent gate is not READY")
    if gate.get("selector_dataset_id") != dataset.get("dataset_id") or gate.get("selector_dataset_artifact_checksum") != dataset.get("dataset_checksum"):
        raise ValueError("Dataset identity mismatch")


def _validate_plan_parents(*, dataset_manifest, parent_gate, selector_run_id, decision_dates):
    if parent_gate.get("gate_contract_version") != "selector_parent_publication_gate.v1" or parent_gate.get("status") != "READY":
        raise ValueError("Stage 10 parent gate is not READY")
    if not parent_gate.get("logical_checksum"):
        raise ValueError("Parent gate is incomplete")
    if parent_gate.get("selector_run_id") != selector_run_id:
        raise ValueError("Parent gate belongs to another run")
    if dataset_manifest.get("dataset_contract_version") != "canonical_v2_selector_dataset.v1":
        raise ValueError("Canonical-v2 dataset identity is required")
    if dataset_manifest.get("dataset_id") != parent_gate.get("selector_dataset_id"):
        raise ValueError("Dataset identity mismatch")
    if dataset_manifest.get("dataset_checksum") != parent_gate.get("selector_dataset_artifact_checksum"):
        raise ValueError("Dataset identity mismatch")
    required = {
        "dataset_manifest_path", "daily_spine_id", "daily_spine_manifest_path",
        "symbol_registry_id", "symbol_registry_manifest_path",
        "target_contract_id", "target_contract_hash",
        "target_provenance_contract_version",
    }
    missing = sorted(key for key in required if not dataset_manifest.get(key))
    if missing:
        raise ValueError(f"Dataset lineage missing: {','.join(missing)}")
    if dataset_manifest.get("target_provenance_contract_version") != TARGET_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("Target provenance v2 is required")
    if parent_gate.get("target_provenance_contract_version") != TARGET_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("Target provenance v2 is required")
    if parent_gate.get("daily_spine_id") != dataset_manifest.get("daily_spine_id"):
        raise ValueError("Daily-spine identity mismatch")
    if parent_gate.get("canonical_registry_id") != dataset_manifest.get("symbol_registry_id"):
        raise ValueError("Symbol-registry identity mismatch")
    available = set(parent_gate.get("available_operational_dates") or parent_gate.get("required_operational_dates") or [])
    if not set(map(str, decision_dates)).issubset(available):
        raise ValueError("Required operational dates are absent")
    _folds_by_date(parent_gate)


def _folds_by_date(parent_gate):
    raw = parent_gate.get("strict_oos_folds")
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        items = ((row.get("decision_date"), row) for row in (raw or []) if isinstance(row, Mapping))
    folds = {str(date): dict(row) for date, row in items if date}
    required = {
        "fold_id", "training_start", "training_end", "validation_date",
        "maximum_label_available_timestamp", "purge_sessions", "embargo_sessions",
    }
    for date, fold in folds.items():
        missing = sorted(key for key in required if fold.get(key) in {None, ""})
        if missing:
            raise ValueError(f"Strict-OOS fold incomplete: {date}")
        if int(fold["purge_sessions"]) != PURGE_SESSIONS or int(fold["embargo_sessions"]) != EMBARGO_SESSIONS:
            raise ValueError(f"Strict-OOS fold purge/embargo mismatch: {date}")
        if not (
            str(fold["training_start"]) <= str(fold["training_end"])
            < str(fold["validation_date"]) <= date
        ):
            raise ValueError(f"Strict-OOS fold temporal order invalid: {date}")
        if str(fold["maximum_label_available_timestamp"]) > str(fold["training_end"]):
            raise ValueError(f"Strict-OOS fold label availability invalid: {date}")
    return folds


def _build_component_plan_row(
    *, resolver, dataset_manifest, parent_gate, target_hash, output_root,
    campaign_id, selector_run_id, model_id, decision_date, fold, random_seed,
    source_commit,
):
    model_resolution = resolver.resolve("selector_models", model_id, role="selector")
    model_payload = model_resolution.entry.payload
    if not model_payload.get("ordinary_runner_support"):
        raise ValueError(f"Requested model is not ordinary-fittable: {model_id}")
    feature_schema_id = str(model_payload["feature_schema"])
    feature_schema_hash = _sha(Path(feature_schema_id))
    ranking_contract_id = str(
        model_payload.get("ranking_problem_contract")
        or "pointwise_forward_return_regression_ranking_v1"
    )
    identity = {
        "campaign_id": campaign_id,
        "selector_run_id": selector_run_id,
        "model_id": model_id,
        "decision_date": decision_date,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_checksum": dataset_manifest["dataset_checksum"],
        "daily_spine_id": dataset_manifest["daily_spine_id"],
        "symbol_registry_id": dataset_manifest["symbol_registry_id"],
        "feature_schema_id": feature_schema_id,
        "feature_schema_hash": feature_schema_hash,
        "target_contract_id": TARGET,
        "target_contract_hash": target_hash,
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "ranking_contract_id": ranking_contract_id,
        "fold_id": fold["fold_id"],
        "purge_sessions": int(fold["purge_sessions"]),
        "embargo_sessions": int(fold["embargo_sessions"]),
        "random_seed": int(random_seed),
        "source_commit": source_commit,
    }
    component_id = f"selector-component-{canonical_hash(identity)[:24].lower()}"
    identity = {**identity, "component_id": component_id}
    base_owner = output_root / "components" / campaign_id / f"date={decision_date}" / f"model={model_id}" / component_id
    resume = _resume_evidence(base_owner, identity, campaign_id)
    owner = (
        base_owner
        if resume["compatible"] or resume["state"] == "MISSING"
        else base_owner.with_name(f"{component_id}-planned")
    )
    row = {
        "component_id": component_id,
        "experiment_id": f"selector-experiment-{canonical_hash({'component_id': component_id, 'model_id': model_id})[:24].lower()}",
        "campaign_id": campaign_id,
        "model_id": model_id,
        "decision_date": decision_date,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_checksum": dataset_manifest["dataset_checksum"],
        "dataset_manifest_path": str(dataset_manifest["dataset_manifest_path"]),
        "daily_spine_id": dataset_manifest["daily_spine_id"],
        "daily_spine_manifest_path": str(dataset_manifest["daily_spine_manifest_path"]),
        "symbol_registry_id": dataset_manifest["symbol_registry_id"],
        "symbol_registry_manifest_path": str(dataset_manifest["symbol_registry_manifest_path"]),
        "feature_schema_id": feature_schema_id,
        "feature_schema_hash": feature_schema_hash,
        "target_contract_id": TARGET,
        "target_contract_hash": target_hash,
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "ranking_contract_id": ranking_contract_id,
        "fold_id": fold["fold_id"],
        "training_start": fold["training_start"],
        "training_end": fold["training_end"],
        "validation_date": fold["validation_date"],
        "purge_sessions": int(fold["purge_sessions"]),
        "embargo_sessions": int(fold["embargo_sessions"]),
        "maximum_label_available_timestamp": fold["maximum_label_available_timestamp"],
        "random_seed": int(random_seed),
        "source_commit": source_commit,
        "planned_output_root": str(owner),
        "status": "COMPLETE_COMPATIBLE" if resume["compatible"] else "PLANNED",
        "resume_evidence": resume,
        "parent_gate_checksum": parent_gate["logical_checksum"],
        "model_entry_hash": model_resolution.entry.entry_hash,
    }
    row["plan_checksum"] = _logical({**row, "material_identity": identity})
    return row


def _resume_evidence(owner: Path, identity: Mapping[str, Any], campaign_id: str) -> dict[str, Any]:
    manifest_path = owner / "manifest.json"
    if not manifest_path.exists():
        return {"compatible": False, "state": "MISSING", "manifest_path": str(manifest_path)}
    try:
        manifest = _json(manifest_path)
    except (OSError, ValueError):
        return {"compatible": False, "state": "CORRUPT", "manifest_path": str(manifest_path)}
    manifest_checksum = manifest.get("manifest_checksum") or manifest.get("logical_checksum")
    checksum_ok = manifest_checksum in {
        canonical_hash({key: value for key, value in manifest.items() if key not in {"manifest_checksum", "logical_checksum"}}),
        _logical(manifest),
    }
    prediction_path = Path(str(manifest.get("prediction_artifact_path") or owner / "predictions.csv"))
    compatible = bool(
        checksum_ok
        and manifest.get("component_id") == identity["component_id"]
        and manifest.get("campaign_id") == campaign_id
        and manifest.get("selector_model_identity") == identity["model_id"]
        and manifest.get("prediction_date") == identity["decision_date"]
        and (manifest.get("frozen_selector_dataset_identity") or {}).get("dataset_id") == identity["dataset_id"]
        and (manifest.get("frozen_selector_dataset_identity") or {}).get("dataset_checksum") == identity["dataset_checksum"]
        and manifest.get("feature_contract_version") in {identity["feature_schema_id"], Path(identity["feature_schema_id"]).stem}
        and manifest.get("target_contract_version") == identity["target_contract_id"]
        and manifest.get("target_provenance_contract_version") == TARGET_PROVENANCE_CONTRACT_VERSION
        and manifest.get("fold_identity") == identity["fold_id"]
        and manifest.get("git_commit") == identity["source_commit"]
        and manifest.get("publication_status") in {"complete", "COMPLETE"}
        and manifest.get("validation_status") == "VERIFIED_STRICT_OOS"
        and prediction_path.exists()
    )
    return {
        "compatible": compatible,
        "state": "COMPLETE_COMPATIBLE" if compatible else "INCOMPATIBLE_COMPLETE",
        "manifest_path": str(manifest_path),
        "checksum_valid": bool(checksum_ok),
    }


def _validate_component_plan_rows(rows):
    if len(rows) != 15:
        raise ValueError("component plan must contain exactly 15 fitted components")
    ids = [str(row.get("component_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate component IDs")
    owners = [str(row.get("planned_output_root")) for row in rows]
    if len(owners) != len(set(owners)):
        raise ValueError("duplicate output ownership")
    triples = [(row.get("model_id"), row.get("decision_date"), row.get("planned_output_root")) for row in rows]
    if len(triples) != len(set(triples)):
        raise ValueError("duplicate model/date/output ownership")
    expected = [(date, model) for date in DATES for model in MODELS]
    actual = [(row.get("decision_date"), row.get("model_id")) for row in rows]
    if actual != expected:
        raise ValueError("component plan ordering mismatch")
    required = {
        "component_id", "experiment_id", "campaign_id", "model_id", "decision_date",
        "dataset_id", "dataset_checksum", "dataset_manifest_path", "daily_spine_id",
        "daily_spine_manifest_path", "symbol_registry_id",
        "symbol_registry_manifest_path", "feature_schema_id",
        "feature_schema_hash", "target_contract_id", "target_contract_hash",
        "target_provenance_contract_version", "ranking_contract_id", "fold_id",
        "training_start", "training_end", "validation_date", "purge_sessions",
        "embargo_sessions", "maximum_label_available_timestamp", "random_seed",
        "source_commit", "planned_output_root", "status", "plan_checksum",
    }
    for row in rows:
        missing = sorted(key for key in required if row.get(key) in {None, ""})
        if missing:
            raise ValueError(f"component plan field missing: {','.join(missing)}")
        if row.get("target_provenance_contract_version") != TARGET_PROVENANCE_CONTRACT_VERSION:
            raise ValueError("target provenance v2 is required")


def _publish_plan(output_root: Path, manifest: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "component_plan.json"
    md_path = output_root / "component_plan.md"
    temp = output_root / f".component_plan.{uuid.uuid4().hex}.tmp"
    temp.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(temp / "component_plan.json", manifest)
        (temp / "component_plan.md").write_text(_component_plan_markdown(manifest), encoding="utf-8")
        os.replace(temp / "component_plan.json", json_path)
        os.replace(temp / "component_plan.md", md_path)
        shutil.rmtree(temp)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def _component_plan_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Selector operational component plan",
        "",
        f"- Contract: `{manifest.get('plan_contract_version')}`",
        f"- Campaign: `{manifest.get('campaign_id')}`",
        f"- Components: `{manifest.get('component_count')}`",
        f"- Plans only: `{str(manifest.get('plans_only')).lower()}`",
        "",
        "| decision_date | model_id | status | component_id |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest.get("components", []):
        lines.append(f"| {row['decision_date']} | {row['model_id']} | {row['status']} | {row['component_id']} |")
    return "\n".join(lines) + "\n"


def _validate_source_rows(rows):
    ids = [str(row.get("row_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source row ID")
    required = {"row_id", "asset_id", "decision_session_date", "label_available_timestamp", "actual_forward_return_10d"}
    if any(required - set(row) for row in rows):
        raise ValueError("Selector dataset rows incomplete")


def _validate_package_rows(training, prediction, date, cutoff, features, model):
    if not training or not prediction:
        raise ValueError("Incomplete training or prediction population")
    for name, values in (("training", training), ("prediction", prediction)):
        ids = [str(row["row_id"]) for row in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {name} row ID")
        for row in values:
            if any(name not in row or not math.isfinite(float(row[name])) for name in features):
                raise ValueError("Nonfinite or missing feature")
    if any(str(row["decision_session_date"]) >= cutoff for row in training):
        raise ValueError("Purge or embargo violation")
    if any(str(row["label_available_timestamp"]) > cutoff for row in training):
        raise ValueError("Immature training label")
    if any(str(row["decision_session_date"]) != date for row in prediction):
        raise ValueError("Prediction date population mismatch")
    for row in prediction:
        row.pop("actual_forward_return_10d", None)
        row.pop("actual_benchmark_return_10d", None)


def _add_relevance(rows):
    by_date = {}
    for row in rows:
        by_date.setdefault(str(row["decision_session_date"]), []).append(row)
    for date_rows in by_date.values():
        ordered = sorted(date_rows, key=lambda row: (float(row["actual_forward_return_10d"]), str(row["row_id"])))
        if len(ordered) < 5:
            raise ValueError("Ordered-logit date group lacks complete relevance classes")
        for index, row in enumerate(ordered):
            row["relevance_label"] = min(4, index * 5 // len(ordered))
            row["date_group"] = str(row["decision_session_date"])
        if {row["relevance_label"] for row in ordered} != set(range(5)):
            raise ValueError("Ordered-logit relevance class missing")


def _publish_package(owner, training, prediction, manifest):
    temporary = owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    _write_json(temporary / "training_rows.json", training)
    _write_json(temporary / "prediction_rows.json", prediction)
    manifest["training_artifact_checksum"] = _sha(temporary / "training_rows.json")
    manifest["prediction_artifact_checksum"] = _sha(temporary / "prediction_rows.json")
    manifest["training_rows_path"] = str(owner / "training_rows.json")
    manifest["prediction_rows_path"] = str(owner / "prediction_rows.json")
    manifest["manifest_path"] = str(owner / "manifest.json")
    manifest["logical_checksum"] = _logical(manifest)
    _write_json(temporary / "manifest.json", manifest)
    _atomic_directory(temporary, owner, manifest)
    return manifest


def _atomic_directory(temporary, owner, manifest):
    if owner.exists():
        try:
            existing = _json(owner / "manifest.json")
            if (
                existing.get("logical_checksum") == _logical(existing)
                and existing.get("logical_checksum") == manifest.get("logical_checksum")
            ):
                shutil.rmtree(temporary)
                return
        except (OSError, ValueError):
            pass
        shutil.rmtree(temporary)
        raise FileExistsError(f"Incompatible immutable owner: {owner}")
    owner.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, owner)


def _publish_file(path, payload):
    if path.exists():
        existing = _json(path)
        if existing.get("logical_checksum") == payload.get("logical_checksum"):
            return
        raise FileExistsError(f"Incompatible immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_json(temporary, payload)
    os.replace(temporary, path)


def _safe_job(value): return value.replace(":", "__")
def _row_key(row): return (str(row.get("decision_session_date")), str(row.get("asset_id")), str(row.get("row_id")))
def _logical(value): return canonical_hash({key: item for key, item in value.items() if key != "logical_checksum"})
def _json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def _git_commit(): return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
def _write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
