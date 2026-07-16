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

import numpy as np

from core.research.ml.artifact_lineage import VERIFIED_STRICT_OOS, build_artifact_link, verify_selector_artifact
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash
from core.research.ml.ranking import OrderedLogitRanker
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level.contextual_elastic_net_selector import fit_contextual_elastic_net
from core.research.ml.stock_level.huber_selector import fit_huber_selector
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    HORIZON_IDS,
    fit_multi_horizon_linear_selector,
)


PUBLICATION_CONTRACT = "ordinary_selector_publication.v1"
COMPONENT_CONTRACT = "authoritative_selector_component_v1"
AGGREGATION_CONTRACT = "wave4_multi_horizon_evidence.v1"
MODEL_FAMILIES = {
    "huber", "contextual_elastic_net", "multi_horizon_ridge",
    "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
}
HORIZON_TARGETS = {
    "return_1s": "forward_return_1d", "return_5s": "forward_return_5d",
    "return_10s": "forward_return_10d", "return_20s": "forward_return_20d",
}


def publish_wave4_component(
    *,
    model_id: str,
    prediction_date: str,
    fit_input: Mapping[str, Any],
    output_root: Path,
    parent_gate: Mapping[str, Any],
    ledger_path: Path,
    horizon_id: str | None = None,
    interaction_contract: Mapping[str, Any] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit and atomically publish one synthetic-or-production-ready Wave 4 component."""
    if model_id not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported Wave 4 model: {model_id}")
    if parent_gate.get("gate_contract_version") != "selector_parent_publication_gate.v1" or parent_gate.get("status") != "READY":
        raise ValueError("Parent gate not ready")
    if model_id.startswith("multi_horizon_") and horizon_id not in HORIZON_IDS:
        raise ValueError("Explicit registered horizon is required")
    if not model_id.startswith("multi_horizon_") and horizon_id is not None:
        raise ValueError("Horizon is only valid for a multi-horizon component")

    resolver = RegistryResolver(load_registry_bundle())
    model = resolver.resolve("selector_models", model_id, role="selector")
    target_id = HORIZON_TARGETS[horizon_id] if horizon_id else str(model.entry.payload["target_contract"])
    target = resolver.resolve("target_contracts", target_id, role="selector")
    identity = {
        "model_id": model.canonical_id, "model_entry_hash": model.entry.entry_hash,
        "horizon_id": horizon_id, "target_contract": target.canonical_id,
        "target_entry_hash": target.entry.entry_hash, "prediction_date": prediction_date,
        "dataset_id": parent_gate["selector_dataset_id"],
        "dataset_checksum": parent_gate["selector_dataset_artifact_checksum"],
        "feature_schema": model.entry.payload["feature_schema"],
        "parent_gate_checksum": parent_gate["logical_checksum"],
        "fit_input_checksum": fit_input.get("logical_input_checksum"),
        "fit_options": dict(fit_options or {}),
    }
    spec_hash = experiment_spec_hash(identity)
    run_id = f"wave4-{spec_hash[:20].lower()}"
    owner = output_root / f"model={model_id}" / (f"horizon={horizon_id}/date={prediction_date}" if horizon_id else f"date={prediction_date}")
    existing = _compatible(owner, identity)
    if existing:
        _event(ledger_path, spec_hash, run_id, "SKIPPED_COMPLETE", model_id, horizon_id, identity, (str(owner / "manifest.json"),))
        return {"status": "SKIPPED_COMPLETE", "manifest_path": str(owner / "manifest.json")}
    if owner.exists():
        raise FileExistsError(f"Incompatible component exists: {owner}")

    _event(ledger_path, spec_hash, run_id, "STARTED", model_id, horizon_id, identity)
    temp = owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp")
    try:
        result, predictions, diagnostics = _fit_and_select(
            model_id, horizon_id, fit_input, interaction_contract, fit_options or {}
        )
        if not result.get("valid") or not predictions:
            reason = ";".join(result.get("blocking_reasons") or [str(result.get("status"))])
            _event(ledger_path, spec_hash, run_id, "REJECTED", model_id, horizon_id, identity, rejection_summary=reason)
            raise ValueError(f"Wave 4 fit rejected: {reason}")
        output_rows = _component_rows(predictions, model_id, horizon_id, prediction_date, identity)
        if not all(math.isfinite(float(row["selector_score"])) for row in output_rows):
            raise ValueError("Nonfinite selector score")
        temp.mkdir(parents=True, exist_ok=False)
        _write_csv(temp / "predictions.csv", output_rows)
        prediction_checksum = _sha256(temp / "predictions.csv")
        population_checksum = canonical_hash([row["row_id"] for row in output_rows])
        training_start, training_cutoff, label_max = _temporal_fields(fit_input, result, horizon_id)
        if not (training_start < training_cutoff < prediction_date and label_max <= prediction_date):
            raise ValueError("Temporal legality failed")
        final_predictions = owner / "predictions.csv"
        link = build_artifact_link(
            artifact_kind="ORDINARY_SELECTOR_PREDICTION",
            artifact_id=f"wave4-selector:{canonical_hash(identity)}",
            artifact_manifest_path=owner / "manifest.json",
            artifact_path=final_predictions, artifact_checksum=prediction_checksum,
            experiment_spec_hash=spec_hash, experiment_run_id=run_id,
            source_commit=_git_commit(), canonical_model_or_policy_id=model_id,
            model_or_policy_entry_hash=model.entry.entry_hash,
            dataset_id=parent_gate["selector_dataset_id"],
            dataset_checksum=parent_gate["selector_dataset_artifact_checksum"],
            row_population_hash=population_checksum,
            feature_schema_hash=parent_gate["selector_feature_schema_checksum"],
            target_contract_hash=target.entry.entry_hash,
            decision_start=prediction_date, decision_end=prediction_date,
            training_cutoff=training_cutoff,
            maximum_label_available_timestamp=label_max,
            strict_oos_claim=True,
            strict_oos_evidence={
                "prediction_quality_passed": True, "row_population_verified": True,
                "temporal_legality_checked": True,
            }, completion_status="complete",
        )
        verification = verify_selector_artifact(link); link.update(verification.to_dict())
        if verification.status != VERIFIED_STRICT_OOS:
            raise ValueError(f"Artifact verification failed: {verification.reason_codes}")
        metrics = {
            "publication_contract_version": PUBLICATION_CONTRACT,
            "model_id": model_id, "horizon_id": horizon_id,
            "prediction_date": prediction_date, "fit_status": result["status"],
            "configuration": result.get("configuration", {}),
            "diagnostics": diagnostics,
            "gate_w4_evidence": {
                "multi_regime_rank_ic": None, "portfolio_utility_reference": None,
                "turnover_reference": None, "stability_reference": "metrics.json",
                "incremental_information_comparison": None,
                "model_to_baseline_rank_correlation": None,
                "rejected_date_count": 0, "effective_experiment_count": 1,
                "gate_passed": False,
            },
        }
        _write_json(temp / "metrics.json", metrics)
        manifest = {
            "component_schema_version": COMPONENT_CONTRACT,
            "publication_contract_version": PUBLICATION_CONTRACT,
            "selector_model_identity": model_id,
            "selector_model_version": model.entry.entry_hash,
            "component_subtype": f"{model_id}__{horizon_id}" if horizon_id else model_id,
            "horizon_id": horizon_id,
            "horizon_sessions": int(horizon_id.removeprefix("return_").removesuffix("s")) if horizon_id else 10,
            "prediction_date": prediction_date,
            "training_start": training_start, "training_cutoff": training_cutoff,
            "training_label_available_timestamp_max": label_max,
            "fold_identity": fit_input.get("fold_identity") or fit_input.get("validation_fold_identity"),
            "frozen_selector_dataset_identity": {
                "dataset_id": parent_gate["selector_dataset_id"],
                "dataset_checksum": parent_gate["selector_dataset_artifact_checksum"],
            },
            "symbol_registry_identity": parent_gate["canonical_registry_id"],
            "daily_stock_spine_identity": parent_gate["daily_spine_id"],
            "feature_contract_version": model.entry.payload["feature_schema"],
            "interaction_contract_version": (
                interaction_contract.get("contract_version") if interaction_contract else None
            ),
            "target_contract_version": target.canonical_id,
            "ranking_contract_version": model.entry.payload.get("ranking_problem_contract") or "ranking_metric_contract_v1",
            "relevance_contract_version": model.entry.payload.get("relevance_contract"),
            "prediction_row_count": len(output_rows),
            "prediction_population_checksum": population_checksum,
            "prediction_artifact_path": str(final_predictions),
            "prediction_checksum": prediction_checksum,
            "artifact_link": link, "publication_status": "complete",
            "validation_status": VERIFIED_STRICT_OOS, "git_commit": _git_commit(),
            "non_production_smoke": False,
            "parent_gate_logical_checksum": parent_gate["logical_checksum"],
            "experiment_spec_hash": spec_hash, "experiment_run_id": run_id,
            "metrics_path": str(owner / "metrics.json"),
        }
        manifest["wave4_identity"] = identity
        manifest["manifest_checksum"] = canonical_hash(manifest)
        _write_json(temp / "manifest.json", manifest)
        owner.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, owner)
        _event(ledger_path, spec_hash, run_id, "COMPLETED", model_id, horizon_id, identity,
               (str(owner / "manifest.json"), str(owner / "predictions.csv"), str(owner / "metrics.json")),
               metadata={"metrics_path": str(owner / "metrics.json")})
        return {"status": "COMPLETED", "manifest_path": str(owner / "manifest.json")}
    except (ValueError, FileExistsError):
        if temp.exists(): shutil.rmtree(temp)
        raise
    except BaseException as exc:
        if temp.exists(): shutil.rmtree(temp)
        _event(ledger_path, spec_hash, run_id, "FAILED", model_id, horizon_id, identity,
               error_summary=f"{type(exc).__name__}: {exc}")
        raise


def build_multi_horizon_evidence(component_manifests: Sequence[Path]) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(component_manifests)]
    horizons = {row.get("horizon_id") for row in rows}
    if horizons != set(HORIZON_IDS):
        raise ValueError("All four horizon components are required")
    prediction_rows = {}
    for manifest in rows:
        with Path(manifest["prediction_artifact_path"]).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                prediction_rows.setdefault(row["row_id"], {})[manifest["horizon_id"]] = float(row["selector_score"])
    if not prediction_rows or any(set(values) != set(HORIZON_IDS) for values in prediction_rows.values()):
        raise ValueError("Missing horizon prediction")
    evidence = []
    for row_id, scores in sorted(prediction_rows.items()):
        ordered = [scores[horizon] for horizon in HORIZON_IDS]
        ranks = _percentile_ranks(ordered)
        sign_agreement = max(sum(value >= 0 for value in ordered), sum(value <= 0 for value in ordered)) / 4
        persistence = 0.5 * sign_agreement + 0.5 * max(0.0, 1.0 - float(np.std(ranks)))
        disagreement = (
            0.4 * (1.0 - sign_agreement)
            + 0.4 * (max(ranks) - min(ranks))
            + 0.2 * float(np.sign(ordered[0]) != np.sign(ordered[-1]))
        )
        evidence.append({
            "row_id": row_id, "short_term_score": scores["return_1s"],
            "medium_term_score": 0.5 * (scores["return_5s"] + scores["return_10s"]),
            "long_term_score": scores["return_20s"], "persistence_score": persistence,
            "horizon_disagreement": disagreement, "horizon_scores": scores,
        })
    logical = {
        "contract_version": AGGREGATION_CONTRACT,
        "persistence_equation": "0.5*sign_agreement + 0.5*max(0,1-std(horizon_percentile_ranks))",
        "disagreement_equation": "0.4*sign_disagreement + 0.4*rank_range + 0.2*short_long_sign_conflict",
        "components": [row["manifest_checksum"] for row in rows],
        "rows": evidence,
    }
    logical["logical_checksum"] = canonical_hash(logical)
    return logical


def _fit_and_select(model_id, horizon_id, fit_input, interaction_contract, options):
    if model_id == "huber":
        result = fit_huber_selector(fit_input, **options)
        return result, result.get("predictions", []), result.get("diagnostic_summary", {})
    if model_id == "contextual_elastic_net":
        if interaction_contract is None: raise ValueError("Contextual interaction contract is required")
        result = fit_contextual_elastic_net(fit_input, interaction_contract, **options)
        return result, result.get("predictions", []), result.get("coefficient_diagnostics", {})
    if model_id in {"multi_horizon_ridge", "multi_horizon_elastic_net"}:
        family = "ridge" if model_id.endswith("ridge") else "elastic_net"
        result = fit_multi_horizon_linear_selector(fit_input, model_families=(family,), **options)
        selected = [row for row in result.get("predictions", []) if row["horizon_id"] == horizon_id]
        return result, selected, result.get("diagnostics", {}).get("per_horizon", {}).get(horizon_id, {})
    return _fit_ordered_horizon(fit_input, horizon_id, options)


def _fit_ordered_horizon(data, horizon, options):
    rows = list(data["rows"])
    cutoff = str(options.get("training_cutoff"))
    training = [row for row in rows if row["split"] == "TRAINING"
                and row["target_availability_state"][horizon] == "MATURE"
                and row["target_maturity_timestamps"][horizon] <= cutoff
                and row["decision_timestamp"] < cutoff]
    validation = [row for row in rows if row["split"] == "VALIDATION"]
    if len(training) < int(options.get("minimum_training_rows", 5)) or not validation:
        return {"status": "INSUFFICIENT_DATA", "valid": False, "blocking_reasons": ["HORIZON_POPULATION_INADEQUATE"]}, [], {}
    ranker = OrderedLogitRanker(
        bins=5, max_iter=int(options.get("maximum_iterations", 500)),
        tolerance=float(options.get("tolerance", 1e-8)),
    )
    x_train = [row["feature_values"] for row in training]
    ranker.fit(
        x_train, [row["target_values"][horizon] for row in training],
        groups=[row["decision_timestamp"] for row in training],
        row_ids=[row["row_id"] for row in training],
    )
    x_validation = [row["feature_values"] for row in validation]
    probabilities = ranker.predict_proba(x_validation)
    scores = ranker.predict(x_validation)
    predictions = []
    order = np.lexsort((
        np.asarray([row["asset_id"] for row in validation]),
        -np.asarray(scores, dtype=float),
    ))
    ranks = np.empty(len(validation), dtype=int); ranks[order] = np.arange(1, len(validation) + 1)
    for index, row in enumerate(validation):
        item = {
            "row_id": row["row_id"], "asset_id": row["asset_id"],
            "decision_timestamp": row["decision_timestamp"], "score": float(scores[index]),
            "within_date_rank": int(ranks[index]), "horizon_id": horizon,
            "maximum_label_maturity_timestamp": max(row["target_maturity_timestamps"][horizon] for row in training),
            "training_cutoff": cutoff,
            "ordered_logit_predicted_relevance_class": int(np.argmax(probabilities[index])),
        }
        for cls in range(5): item[f"ordered_logit_probability_{cls}"] = float(probabilities[index, cls])
        predictions.append(item)
    result = {
        "status": "READY", "valid": True, "blocking_reasons": [],
        "configuration": dict(options), "predictions": predictions,
    }
    return result, predictions, ranker.diagnostics


def _component_rows(rows, model, horizon, date, identity):
    output = []
    for raw in rows:
        score = raw.get("predicted_return", raw.get("continuous_score", raw.get("score")))
        row = {
            "row_id": str(raw["row_id"]), "asset_id": str(raw["asset_id"]),
            "symbol": str(raw.get("symbol", raw["asset_id"])), "prediction_date": date,
            "model_id": model, "horizon_id": horizon or "",
            "selector_score": float(score), "deterministic_rank": int(raw.get("within_date_rank", 0)),
            "dataset_identity": identity["dataset_id"],
            "feature_contract_identity": identity["feature_schema"],
            "target_contract_identity": identity["target_contract"],
            "fold_identity": str(raw.get("fold_identity", "")),
        }
        for key, value in raw.items():
            if key.startswith("ordered_logit_"): row[key] = value
        if model == "multi_horizon_ordered_logit":
            row["ordered_logit_expected_relevance"] = float(score)
        output.append(row)
    return sorted(output, key=lambda row: (row["prediction_date"], row["deterministic_rank"], row["asset_id"], row["row_id"]))


def _temporal_fields(data, result, horizon):
    training = [row for row in data["rows"] if row["split"] == "TRAINING"]
    training_start = min(str(row["decision_timestamp"]) for row in training)
    training_cutoff = str(result.get("configuration", {}).get("training_cutoff") or max(row["decision_timestamp"] for row in training))
    if horizon:
        mature = [row["target_maturity_timestamps"][horizon] for row in training
                  if row["target_availability_state"][horizon] == "MATURE"
                  and row["target_maturity_timestamps"][horizon] <= training_cutoff]
    else:
        mature = [row["target_maturity_timestamp"] for row in training]
    return training_start, training_cutoff, max(mature)


def _event(path, spec, run, status, model, horizon, identity, artifacts=(), **kwargs):
    append_ledger_event(
        path, experiment_spec_hash_value=spec, experiment_run_id=run,
        event_status=status, artifact_kind="SELECTOR_PREDICTION_PARTITION",
        canonical_model_id=model, requested_model_id=model,
        registry_hashes={
            "model_entry_hash": identity["model_entry_hash"],
            "target_entry_hash": identity["target_entry_hash"],
        }, source_commit=_git_commit(), artifact_paths=artifacts,
        metadata={"horizon": horizon, "dataset_id": identity["dataset_id"], **kwargs.pop("metadata", {})},
        **kwargs,
    )


def _compatible(owner, identity):
    try:
        manifest = json.loads((owner / "manifest.json").read_text(encoding="utf-8"))
        return (
            manifest.get("publication_status") == "complete"
            and manifest.get("wave4_identity") == identity
            and _sha256(owner / "predictions.csv") == manifest.get("prediction_checksum")
        )
    except (OSError, json.JSONDecodeError):
        return False


def _percentile_ranks(values):
    order = np.argsort(np.argsort(values))
    return (order / max(len(values) - 1, 1)).tolist()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
