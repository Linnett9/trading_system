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
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash
from core.research.ml.ranking import ranking_metrics, relevance_labels
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash


EVALUATION_CONTRACT = "selector_component_evaluation.v1"
COMPONENT_CONTRACT = "authoritative_selector_component_v1"
READINESS_CONTRACT = "selector_component_readiness.v2"
VERIFIED_STRICT_OOS = "VERIFIED_STRICT_OOS"
BASE_MODELS = ("momentum_120d", "ridge", "elastic_net", "ordered_logit_ranker")
HORIZON_TARGETS = {
    "return_1s": ("forward_return_1d", "actual_forward_return_1d", "1_sessions"),
    "return_5s": ("forward_return_5d", "actual_forward_return_5d", "5_sessions"),
    "return_10s": ("forward_return_10d", "actual_forward_return_10d", "10_sessions"),
    "return_20s": ("forward_return_20d", "actual_forward_return_20d", "20_sessions"),
}


def evaluate_selector_components(
    *,
    readiness_path: Path,
    component_manifests: Sequence[Path],
    outcome_path: Path,
    output_root: Path,
    ledger_path: Path,
    panel_id: str,
    evaluation_cutoff: str,
    required_models: Sequence[str] = BASE_MODELS,
    required_dates: Sequence[str] | None = None,
    required_horizons: Sequence[str] | None = None,
    replacement_policy: str = "never_replace_complete",
) -> dict[str, Any]:
    readiness = _json(readiness_path)
    specification = {
        "contract": EVALUATION_CONTRACT, "panel_id": panel_id,
        "readiness_checksum": readiness.get("logical_checksum"),
        "required_models": list(required_models),
        "required_horizons": list(required_horizons or ()),
        "required_dates": sorted(required_dates or readiness.get("required_dates", [])),
        "component_manifests": sorted(str(path) for path in component_manifests),
        "outcome_path": str(outcome_path),
        "evaluation_cutoff": evaluation_cutoff,
    }
    spec_hash = experiment_spec_hash(specification)
    run_id = f"selector-evaluation-{spec_hash[:20].lower()}"
    _event(ledger_path, spec_hash, run_id, "STARTED", panel_id)
    temp: Path | None = None
    try:
        if (
            readiness.get("readiness_contract_version") != READINESS_CONTRACT
            or readiness.get("overall_status") != "READY"
        ):
            raise ValueError("Selector component readiness is not READY")
        dates = tuple(sorted(required_dates or readiness.get("required_dates", [])))
        models = tuple(required_models)
        horizons = tuple(required_horizons or ())
        components, rejected = _load_components(component_manifests)
        expected = {
            (date, model, horizon)
            for date in dates for model in models
            for horizon in (horizons if model.startswith("multi_horizon_") else (None,))
        }
        found = set(components)
        missing = sorted(expected - found)
        if missing:
            rejected.extend({"prediction_date": date, "model_id": model, "horizon_id": horizon, "reasons": ["MISSING_COMPONENT"]} for date, model, horizon in missing)
        outcomes = _load_outcomes(outcome_path, dates)
        blockers = []
        if rejected: blockers.append("COMPONENT_VALIDATION_FAILED")
        matched = _matched_evidence(components, dates, models, horizons)
        if any(row["status"] != "READY" for row in matched):
            blockers.append("UNMATCHED_PANEL")
        maturity = _outcome_maturity(outcomes, dates, evaluation_cutoff, horizons)
        if any(row["status"] != "MATURE" for row in maturity):
            blockers.append("IMMATURE_OUTCOME")
        if not _outcomes_match_components(components, outcomes, dates, models, horizons):
            blockers.append("OUTCOME_POPULATION_MISMATCH")
        per_date, aggregate, ordered = [], {}, {}
        if not blockers:
            per_date, aggregate, ordered = _metrics(components, outcomes, dates, models, horizons)
        status = "READY" if not blockers else "BLOCKED"
        target_resolution = RegistryResolver(load_registry_bundle()).resolve(
            "target_contracts", "forward_return_10d", role="selector"
        )
        result = {
            "evaluation_contract_version": EVALUATION_CONTRACT,
            "panel_id": panel_id,
            "panel_logical_checksum": canonical_hash({
                "readiness": readiness.get("logical_checksum"),
                "components": sorted(row["manifest_checksum"] for row in components.values()),
                "outcomes": _sha256(outcome_path),
            }),
            "dataset_identity": readiness.get("dataset_identity"),
            "dataset_checksum": readiness.get("dataset_checksum"),
            "model_roster": list(models), "date_roster": list(dates),
            "horizon_roster": list(horizons),
            "target_contract": "forward_return_10d",
            "target_contract_identity": {
                "canonical_id": target_resolution.canonical_id,
                "entry_hash": target_resolution.entry.entry_hash,
            },
            "component_identities": [
                row["identity"] for _, row in sorted(components.items())
            ],
            "rejected_components": rejected,
            "matched_population_evidence": matched,
            "outcome_maturity_evidence": maturity,
            "per_date_metrics": per_date,
            "aggregate_metrics": aggregate,
            "model_correlation_matrix": aggregate.get("model_rank_correlations", {}),
            "ordered_logit": ordered,
            "blockers": sorted(set(blockers)), "warnings": [],
            "evaluation_status": status, "source_git_commit": _git_commit(),
        }
        result["logical_checksum"] = _logical_checksum(result)
        existing_path = output_root / "evaluation.json"
        if existing_path.exists():
            existing = _json(existing_path)
            if existing.get("logical_checksum") == result["logical_checksum"]:
                _event(ledger_path, spec_hash, run_id, "SKIPPED_COMPLETE", panel_id,
                       artifact_paths=(str(existing_path),),
                       metadata={"model_count": len(models), "date_count": len(dates)})
                return {**existing, "publication_result": "SKIPPED_COMPLETE"}
            if replacement_policy != "replace_incompatible":
                raise ValueError("Incompatible existing evaluation artifact")
            shutil.rmtree(output_root)
        temp = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.tmp")
        temp.mkdir(parents=True, exist_ok=False)
        _write_json(temp / "evaluation.json", result)
        _write_csv(temp / "metrics.csv", per_date)
        (temp / "report.md").write_text(_markdown(result), encoding="utf-8")
        for required in ("evaluation.json", "metrics.csv", "report.md"):
            if not (temp / required).is_file():
                raise RuntimeError(f"Incomplete atomic evaluation publication: {required}")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output_root)
        if status != "READY":
            _event(ledger_path, spec_hash, run_id, "REJECTED", panel_id,
                   rejection_summary=";".join(result["blockers"]),
                   artifact_paths=(str(output_root / "evaluation.json"),))
        else:
            _event(ledger_path, spec_hash, run_id, "COMPLETED", panel_id,
                   artifact_paths=(
                       str(output_root / "evaluation.json"),
                       str(output_root / "metrics.csv"),
                       str(output_root / "report.md"),
                   ), metadata={"model_count": len(models), "date_count": len(dates)})
        return result
    except ValueError as exc:
        if temp is not None and temp.exists(): shutil.rmtree(temp)
        _event(ledger_path, spec_hash, run_id, "REJECTED", panel_id, rejection_summary=str(exc))
        raise
    except BaseException as exc:
        if temp is not None and temp.exists(): shutil.rmtree(temp)
        _event(ledger_path, spec_hash, run_id, "FAILED", panel_id, error_summary=f"{type(exc).__name__}: {exc}")
        raise


def _load_components(paths: Sequence[Path]):
    resolver = RegistryResolver(load_registry_bundle())
    components, rejected = {}, []
    for path in sorted(paths):
        reasons = []
        try:
            manifest = _json(path)
        except (OSError, json.JSONDecodeError):
            rejected.append({"manifest_path": str(path), "reasons": ["MALFORMED_MANIFEST"]})
            continue
        model = str(manifest.get("selector_model_identity", ""))
        date = str(manifest.get("prediction_date", ""))
        horizon = manifest.get("horizon_id")
        artifact = Path(str(manifest.get("prediction_artifact_path", "")))
        link = manifest.get("artifact_link") if isinstance(manifest.get("artifact_link"), Mapping) else {}
        frozen = manifest.get("frozen_selector_dataset_identity") if isinstance(manifest.get("frozen_selector_dataset_identity"), Mapping) else {}
        if manifest.get("component_schema_version") != COMPONENT_CONTRACT: reasons.append("INCOMPLETE_COMPONENT")
        if manifest.get("publication_status") != "complete": reasons.append("INCOMPLETE_COMPONENT")
        if manifest.get("validation_status") != VERIFIED_STRICT_OOS or link.get("verification_status") != VERIFIED_STRICT_OOS: reasons.append("ARTIFACT_LINK_UNVERIFIED")
        if manifest.get("non_production_smoke"): reasons.append("SMOKE_COMPONENT")
        for field in (
            "git_commit", "training_start", "training_cutoff",
            "training_label_available_timestamp_max", "fold_identity",
            "symbol_registry_identity", "daily_stock_spine_identity",
            "feature_contract_version",
        ):
            if manifest.get(field) in (None, ""): reasons.append("INCOMPLETE_COMPONENT")
        if link.get("feature_schema_hash") in (None, ""):
            reasons.append("FEATURE_CONTRACT_MISMATCH")
        if model in {"ordered_logit_ranker", "multi_horizon_ordered_logit"} and (
            manifest.get("ranking_contract_version") != "daily_cross_sectional_ranking_problem_v1"
            or manifest.get("relevance_contract_version") != "within_date_quintile_relevance_v1"
        ):
            reasons.append("RANKING_CONTRACT_MISMATCH")
        try:
            resolution = resolver.resolve("selector_models", model, role="selector")
            if manifest.get("selector_model_version") != resolution.entry.entry_hash: reasons.append("MODEL_REGISTRY_MISMATCH")
        except KeyError:
            reasons.append("MODEL_REGISTRY_MISMATCH")
        target_id = HORIZON_TARGETS.get(horizon, ("forward_return_10d",))[0]
        target = resolver.resolve("target_contracts", target_id, role="selector")
        if manifest.get("target_contract_version") != target.canonical_id or link.get("target_contract_hash") != target.entry.entry_hash: reasons.append("TARGET_CONTRACT_MISMATCH")
        if not artifact.is_file() or _sha256(artifact) != manifest.get("prediction_checksum") or link.get("artifact_checksum") != manifest.get("prediction_checksum"): reasons.append("PREDICTION_CHECKSUM_MISMATCH")
        expected_manifest = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_checksum"})
        if manifest.get("manifest_checksum") != expected_manifest: reasons.append("MANIFEST_CHECKSUM_MISMATCH")
        rows = _csv(artifact) if artifact.is_file() else []
        row_ids = [row.get("row_id", "") for row in rows]
        if len(rows) != manifest.get("prediction_row_count") or len(row_ids) != len(set(row_ids)) or not all(row_ids): reasons.append("PREDICTION_POPULATION_INVALID")
        if canonical_hash(row_ids) != manifest.get("prediction_population_checksum"): reasons.append("POPULATION_CHECKSUM_MISMATCH")
        scores = [_score(row, model) for row in rows]
        if not scores or not all(math.isfinite(value) for value in scores): reasons.append("NONFINITE_SCORE")
        if len(set(scores)) < 2: reasons.append("DEGENERATE_RANK_POPULATION")
        asset_dates = [(row.get("asset_id"), row.get("prediction_date")) for row in rows]
        if len(asset_dates) != len(set(asset_dates)): reasons.append("DUPLICATE_ASSET_DATE")
        if model in {"ordered_logit_ranker", "multi_horizon_ordered_logit"}:
            for row in rows:
                probabilities = [float(row.get(f"ordered_logit_probability_{index}", "nan")) for index in range(5)]
                if not all(math.isfinite(value) for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7):
                    reasons.append("INVALID_ORDERED_LOGIT_PROBABILITIES"); break
        if reasons:
            rejected.append({"manifest_path": str(path), "prediction_date": date, "model_id": model, "reasons": sorted(set(reasons))})
            continue
        components[(date, model, horizon)] = {
            "manifest": manifest, "rows": rows,
            "manifest_checksum": manifest["manifest_checksum"],
            "identity": {
                "prediction_date": date, "model_id": model,
                "horizon_id": horizon,
                "model_entry_hash": manifest["selector_model_version"],
                "dataset_id": frozen.get("dataset_id"),
                "dataset_checksum": frozen.get("dataset_checksum"),
                "feature_contract": manifest.get("feature_contract_version"),
                "target_contract": manifest.get("target_contract_version"),
                "target_contract_hash": link.get("target_contract_hash"),
                "population_checksum": manifest.get("prediction_population_checksum"),
                "artifact_checksum": manifest.get("prediction_checksum"),
            },
        }
    return components, rejected


def _load_outcomes(path: Path, dates: Sequence[str]):
    rows = _csv(path)
    result = {}
    for row in rows:
        key = (str(row["prediction_date"]), str(row["row_id"]), str(row.get("target_contract", "forward_return_10d")))
        if key in result: raise ValueError("Duplicate outcome row")
        result[key] = row
    if not result or any(date not in {key[0] for key in result} for date in dates):
        raise ValueError("Outcome population is incomplete")
    return result


def _matched_evidence(components, dates, models, horizons=()):
    result = []
    for date in dates:
        panels = horizons or (None,)
        for horizon in panels:
            panel_models = [model for model in models if model.startswith("multi_horizon_") == (horizon is not None)]
            if not panel_models: continue
            rows = [components.get((date, model, horizon)) for model in panel_models]
            identities = {
                (
                    row["identity"]["population_checksum"], row["identity"]["dataset_id"],
                    row["identity"]["dataset_checksum"], row["identity"]["target_contract"],
                    tuple(item["row_id"] for item in row["rows"]),
                )
                for row in rows if row
            }
            result.append({
                "prediction_date": date, "horizon_id": horizon,
                "status": "READY" if len(rows) == len(panel_models) and all(rows) and len(identities) == 1 else "BLOCKED",
                "model_count": sum(row is not None for row in rows),
                "population_checksum": next(iter(identities))[0] if len(identities) == 1 else None,
            })
    return result


def _outcome_maturity(outcomes, dates, evaluation_cutoff, horizons=()):
    evidence = []
    for date in dates:
        for horizon in horizons or (None,):
            target, field, sessions = HORIZON_TARGETS.get(horizon, ("forward_return_10d", "actual_forward_return_10d", "10_sessions"))
            rows = [row for (row_date, _, row_target), row in outcomes.items() if row_date == date and row_target == target]
            foreign_target = horizon is None and any(
                row_date == date and row_target != target
                for row_date, _, row_target in outcomes
            )
            mature = bool(rows) and all(
                row.get("maturity_status") == "MATURE"
                and row.get("target_contract") == target
                and row.get("outcome_field") == field
                and row.get("target_horizon") == sessions
                and bool(row.get("label_available_timestamp"))
                and str(row.get("label_available_timestamp")) <= str(evaluation_cutoff)
                and bool(row.get("outcome_source_identity"))
                and bool(row.get("asset_id"))
                for row in rows
            ) and not foreign_target
            evidence.append({"prediction_date": date, "horizon_id": horizon, "status": "MATURE" if mature else "IMMATURE", "row_count": len(rows)})
    return evidence


def _outcomes_match_components(components, outcomes, dates, models, horizons=()):
    for date in dates:
        for model in models:
            for horizon in (horizons if model.startswith("multi_horizon_") else (None,)):
                target = HORIZON_TARGETS.get(horizon, ("forward_return_10d",))[0]
                outcome_ids = {row_id for (row_date, row_id, row_target) in outcomes if row_date == date and row_target == target}
                component = components.get((date, model, horizon))
                if component is None or {row["row_id"] for row in component["rows"]} != outcome_ids:
                    return False
    return True


def _metrics(components, outcomes, dates, models, horizons=()):
    per_date, rank_maps, top_sets = [], {}, {}
    ordered_summary = {"probability_valid": True, "invalid_probability_count": 0, "class_calibration_inputs": [], "expected_relevance_distribution": {}, "predicted_class_distribution": {}, "average_class_probabilities": {}, "diagnostic_references": []}
    for date in dates:
        for model in models:
          for horizon in (horizons if model.startswith("multi_horizon_") else (None,)):
            target, outcome_field, _ = HORIZON_TARGETS.get(horizon, ("forward_return_10d", "actual_forward_return_10d", "10_sessions"))
            date_outcomes = {row_id: row for (row_date, row_id, row_target), row in outcomes.items() if row_date == date and row_target == target}
            relevance = relevance_labels([
                {"row_id": row_id, "decision_timestamp": date, "actual_forward_return_10d": float(row[outcome_field])}
                for row_id, row in date_outcomes.items()
            ], bins=5)
            component = components[(date, model, horizon)]
            merged = []
            for prediction in component["rows"]:
                outcome = date_outcomes[prediction["row_id"]]
                merged.append({
                    **prediction, "decision_timestamp": date,
                    "score": _score(prediction, model),
                    "actual_forward_return_10d": float(outcome[outcome_field]),
                    "benchmark_return": _optional_float(outcome.get("benchmark_return")),
                    "relevance": relevance["labels_by_row_id"][prediction["row_id"]],
                })
            metric = ranking_metrics(merged, score_field="score")["per_date"][0]
            residual = [row for row in merged if row["benchmark_return"] is not None]
            metric["residual_spearman_rank_ic"] = _corr(
                [_rank(row["score"], merged, "score") for row in residual],
                [_rank(row["actual_forward_return_10d"] - row["benchmark_return"], residual, "residual") for row in residual],
            ) if residual else None
            metric.update({"model_id": model, "prediction_date": date, "horizon_id": horizon})
            per_date.append(metric)
            ordered = sorted(merged, key=lambda row: (-row["score"], str(row["asset_id"])))
            metric_key = f"{model}__{horizon}" if horizon else model
            rank_maps[(date, metric_key)] = {str(row["asset_id"]): index + 1 for index, row in enumerate(ordered)}
            top_sets[(date, metric_key)] = {str(row["asset_id"]) for row in ordered[:10]}
            if model in {"ordered_logit_ranker", "multi_horizon_ordered_logit"}:
                probabilities = [[float(row[f"ordered_logit_probability_{index}"]) for index in range(5)] for row in merged]
                classes = [int(row["ordered_logit_predicted_relevance_class"]) for row in merged]
                scores = [row["score"] for row in merged]
                ordered_summary["class_calibration_inputs"].extend({"prediction_date": date, "probabilities": probs, "actual_relevance": row["relevance"]} for probs, row in zip(probabilities, merged))
                ordered_summary["expected_relevance_distribution"][date] = _distribution(scores)
                ordered_summary["predicted_class_distribution"][date] = {str(value): classes.count(value) for value in range(5)}
                ordered_summary["diagnostic_references"].append(component["manifest"].get("metrics_path"))
    calibration = ordered_summary["class_calibration_inputs"]
    if calibration:
        ordered_summary["average_class_probabilities"] = {
            str(index): mean(row["probabilities"][index] for row in calibration)
            for index in range(5)
        }
    aggregate = {}
    metric_models = sorted({(row["model_id"], row.get("horizon_id")) for row in per_date})
    metric_names = []
    for model, horizon in metric_models:
        name = f"{model}__{horizon}" if horizon else model; metric_names.append(name)
        rows = [row for row in per_date if row["model_id"] == model and row.get("horizon_id") == horizon]
        rank_ics = [row["spearman_rank_ic"] for row in rows if row["spearman_rank_ic"] is not None]
        aggregate[name] = {
            "mean_rank_ic": mean(rank_ics) if rank_ics else None,
            "median_rank_ic": median(rank_ics) if rank_ics else None,
            "rank_ic_standard_deviation": pstdev(rank_ics) if rank_ics else None,
            "rank_ic_information_ratio": mean(rank_ics) / pstdev(rank_ics) if len(rank_ics) > 1 and pstdev(rank_ics) else None,
            "positive_rank_ic_fraction": mean(value > 0 for value in rank_ics) if rank_ics else None,
            "mean_pearson_ic": _mean(row["pearson_ic"] for row in rows),
            "mean_residual_rank_ic": _mean(row["residual_spearman_rank_ic"] for row in rows),
            **{f"mean_ndcg_at_{k}": _mean(row[f"ndcg_at_{k}"] for row in rows) for k in (10,20,40)},
            **{f"mean_top_{k}_return": _mean(row[f"top_{k}_mean_return"] for row in rows) for k in (10,20,40)},
            **{f"mean_top_minus_bottom_{k}": _mean(row[f"top_minus_bottom_{k}"] for row in rows) for k in (10,20,40)},
            "rank_turnover": _rank_turnover(rank_maps, dates, name),
            "top_10_continuity": _top_continuity(top_sets, dates, name),
            "top_20_continuity": None, "top_40_continuity": None,
            "evaluated_date_count": len(rows),
            "missing_date_count": len(dates) - len(rows), "rejected_date_count": 0,
        }
    aggregate["model_rank_correlations"] = _model_correlations(rank_maps, dates, metric_names)
    return per_date, aggregate, ordered_summary


def _score(row, model):
    field = "selector_score" if "selector_score" in row else (
        "predicted_momentum_120d" if model == "momentum_120d" else "score"
    )
    try: return float(row[field])
    except (KeyError, TypeError, ValueError): return float("nan")


def _rank(value, rows, field):
    values = [row["actual_forward_return_10d"] - row["benchmark_return"] if field == "residual" else row[field] for row in rows]
    ordered = sorted(set(values))
    return (ordered.index(value) + 1) if value in ordered else float("nan")


def _corr(left, right):
    if len(left) < 2: return None
    lm, rm = mean(left), mean(right)
    numerator = sum((a-lm)*(b-rm) for a,b in zip(left,right))
    denominator = math.sqrt(sum((a-lm)**2 for a in left)*sum((b-rm)**2 for b in right))
    return numerator / denominator if denominator else None


def _rank_turnover(rank_maps, dates, model):
    values = []
    for left, right in zip(dates, dates[1:]):
        a, b = rank_maps[(left, model)], rank_maps[(right, model)]
        common = sorted(set(a) & set(b))
        if common: values.append(mean(abs(a[key]-b[key]) for key in common))
    return mean(values) if values else None


def _top_continuity(top_sets, dates, model):
    values = []
    for left, right in zip(dates, dates[1:]):
        a, b = top_sets[(left, model)], top_sets[(right, model)]
        if a: values.append(len(a & b) / len(a))
    return mean(values) if values else None


def _top_continuity_for_k(components, dates, model, k):
    sets = {}
    for date in dates:
        rows = components[(date, model)]["rows"]
        ordered = sorted(rows, key=lambda row: (-_score(row, model), str(row["asset_id"])))
        sets[date] = {str(row["asset_id"]) for row in ordered[:k]}
    values = [
        len(sets[left] & sets[right]) / len(sets[left])
        for left, right in zip(dates, dates[1:]) if sets[left]
    ]
    return mean(values) if values else None


def _model_correlations(rank_maps, dates, models):
    result = {}
    for index, left in enumerate(models):
        for right in models[index+1:]:
            values = []
            for date in dates:
                a, b = rank_maps[(date,left)], rank_maps[(date,right)]
                common = sorted(set(a)&set(b))
                value = _corr([a[key] for key in common],[b[key] for key in common])
                if value is not None: values.append(value)
            result[f"{left}|{right}"] = mean(values) if values else None
    return result


def _distribution(values):
    return {"count": len(values), "minimum": min(values), "maximum": max(values), "mean": mean(values), "standard_deviation": pstdev(values)}


def _mean(values):
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def _optional_float(value):
    return None if value in (None, "") else float(value)


def _logical_checksum(payload):
    return canonical_hash({key:value for key,value in payload.items() if key not in {"logical_checksum","generated_at","report_path"}})


def _event(path, spec, run, status, panel_id, artifact_paths=(), error_summary=None, rejection_summary=None, metadata=None):
    append_ledger_event(
        path, experiment_spec_hash_value=spec, experiment_run_id=run,
        event_status=status, artifact_kind="SELECTOR_EVALUATION",
        canonical_model_id=None, requested_model_id=None, registry_hashes={},
        source_commit=_git_commit(), artifact_paths=artifact_paths,
        error_summary=error_summary, rejection_summary=rejection_summary,
        metadata={"panel_id":panel_id, **(metadata or {})},
    )


def _markdown(result):
    lines = ["# Selector Component Evaluation", "", f"- Status: `{result['evaluation_status']}`", f"- Panel: `{result['panel_id']}`", f"- Models: `{', '.join(result['model_roster'])}`", f"- Dates: `{len(result['date_roster'])}`"]
    lines.extend(f"- Blocker: `{value}`" for value in result["blockers"])
    return "\n".join(lines) + "\n"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _json(path): return json.loads(path.read_text(encoding="utf-8"))
def _csv(path):
    with path.open("r",encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest().upper()
def _git_commit(): return subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
