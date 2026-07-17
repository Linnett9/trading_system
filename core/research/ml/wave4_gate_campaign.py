"""Evaluation-only gate for the fixed Wave 4 ordinary-selector campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from core.research.ml.registries.io import canonical_hash
from core.research.ml.experiment_ledger import read_selector_ledger, selector_trial_counts


CONTRACT = "wave4_selector_campaign_gate.v1"
METRIC_CONTRACT = "wave4_selector_metric_contract.v1"
GATE_CONTRACT = "wave4_selector_gate_thresholds.v1"
STATUSES = (
    "READY_FOR_PORTFOLIO_REPLAY", "REJECTED", "BLOCKED_INCOMPLETE",
    "BLOCKED_LINEAGE", "BLOCKED_METRICS",
)
DEFAULT_THRESHOLDS = {
    "minimum_mean_rank_ic": 0.0,
    "minimum_date_coverage": 0.95,
    "maximum_largest_tied_score_fraction": 0.50,
    "minimum_positive_rank_ic_dates": 2,
    "maximum_turnover_increase_vs_momentum": 0.10,
    "minimum_rank_ic_improvement_for_turnover_exception": 0.01,
    "minimum_rank_ic_improvement_vs_momentum": 0.0,
}
REQUIRED_MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")


def evaluate_wave4_campaign(
    *, component_plan_path: Path, component_manifest_paths: Sequence[Path],
    output_root: Path, thresholds: Mapping[str, float] | None = None,
    momentum_manifest_paths: Sequence[Path] = (),
    experiment_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and evaluate existing predictions. Never fits, executes, or publishes."""
    plan = _json(component_plan_path)
    panel = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(panel)
        if unknown:
            raise ValueError(f"Unknown gate thresholds: {sorted(unknown)}")
        panel.update({key: float(value) for key, value in thresholds.items()})
    ledger_path = experiment_ledger_path or component_plan_path.with_name("selector_experiment_ledger.json")
    report = _evaluate(plan, component_manifest_paths, momentum_manifest_paths, panel, ledger_path)
    _publish(output_root, report)
    return report


def _evaluate(plan, manifest_paths, momentum_paths, thresholds, ledger_path):
    blockers: list[str] = []
    blocker_class = "BLOCKED_INCOMPLETE"
    components = list(plan.get("components") or [])
    ledger_rows = {}
    trial_counts = {
        "fitted_model_count": 0, "decision_date_count": 0, "seed_count": 0,
        "hyperparameter_configuration_count": 0, "planned_material_trials": 0,
        "executed_material_trials": 0, "failed_material_trials": 0,
        "rejected_material_trials": 0,
    }
    try:
        ledger = read_selector_ledger(ledger_path)
        ledger_rows = {row["experiment_id"]: row for row in ledger["experiments"]}
        trial_counts = selector_trial_counts(ledger)
    except ValueError:
        blockers.append("EXPERIMENT_LEDGER_EVIDENCE_MISSING_OR_INVALID")
    plan_checksum = str(plan.get("logical_checksum") or "")
    if (
        plan.get("plan_contract_version") != "selector_operational_component_plan.v1"
        or not plan_checksum
        or plan_checksum != _logical(plan)
        or len(components) != 15
    ):
        blockers.append("COMPONENT_PLAN_INVALID")
        components = components[:15]
    expected = {(str(row.get("model_id")), str(row.get("decision_date"))): row for row in components}
    if len(expected) != len(components):
        blockers.append("DUPLICATE_PLANNED_MODEL_DATE")
    for component in components:
        evidence = ledger_rows.get(component.get("experiment_id"))
        if not evidence:
            blockers.append(f"EXPERIMENT_LEDGER_EVIDENCE_MISSING:{component.get('experiment_id')}")
        elif evidence.get("component_id") != component.get("component_id"):
            blockers.append(f"EXPERIMENT_LEDGER_IDENTITY_MISMATCH:{component.get('experiment_id')}")
        elif evidence.get("status") not in {"SUCCEEDED", "REJECTED", "ELIGIBLE_FOR_PORTFOLIO_REPLAY"}:
            blockers.append(f"EXPERIMENT_LEDGER_STATUS_INCOMPLETE:{component.get('experiment_id')}")
    if trial_counts["planned_material_trials"] != 15:
        blockers.append("EXPERIMENT_LEDGER_TRIAL_COUNT_MISMATCH")
    paths = sorted(Path(path) for path in manifest_paths)
    if len(paths) < len(components):
        blockers.append("PLANNED_COMPONENTS_MISSING")
    if len(paths) > len(components):
        blockers.append("EXTRA_COMPONENT_CLAIMS_CAMPAIGN")

    observed, validations, input_checksums = {}, [], {}
    for path in paths:
        validation = _validate_component(path, plan, expected)
        validations.append(validation)
        if validation.get("manifest_checksum"):
            input_checksums[str(path)] = validation["manifest_checksum"]
        key = (validation.get("model_id"), validation.get("decision_date"))
        if key in observed:
            validation["reasons"].append("DUPLICATE_MODEL_DATE_COMPONENT")
        else:
            observed[key] = validation
        if validation["reasons"]:
            blockers.extend(validation["reasons"])
    missing = sorted(set(expected) - set(observed))
    blockers.extend(f"MISSING_COMPONENT:{model}:{date}" for model, date in missing)

    lineage_codes = (
        "STRICT_OOS", "PROVENANCE", "DATASET", "SPINE", "REGISTRY", "FEATURE",
        "TARGET", "RANKING", "FOLD", "SOURCE_COMMIT", "CAMPAIGN_ID_MISMATCH",
        "PLAN_COMPONENT_ID", "COMPONENT_PLAN_INVALID",
        "EXPERIMENT_ID", "EXPERIMENT_LEDGER",
    )
    metric_codes = (
        "NONFINITE", "DUPLICATE_ECONOMIC", "DEGENERATE", "PREDICTION_POPULATION",
    )
    if any(any(token in reason for token in lineage_codes) for reason in blockers):
        blocker_class = "BLOCKED_LINEAGE"
    elif any(any(token in reason for token in metric_codes) for reason in blockers):
        blocker_class = "BLOCKED_METRICS"

    per_date: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    momentum = {"available": False, "reason": "OPTIONAL_MOMENTUM_EVIDENCE_NOT_PROVIDED"}
    if not blockers:
        for key in sorted(expected, key=lambda item: (item[1], item[0])):
            result = _metrics(observed[key]["rows"])
            result.update({"model_id": key[0], "decision_date": key[1]})
            per_date.append(result)
        aggregates = {
            model: _aggregate([row for row in per_date if row["model_id"] == model])
            for model in REQUIRED_MODELS
        }
        if momentum_paths:
            momentum = _momentum(momentum_paths, plan)
            if momentum["available"]:
                control = momentum["aggregate_metrics"]
                for model in REQUIRED_MODELS:
                    aggregates[model]["momentum_comparison"] = {
                        "rank_ic_improvement": aggregates[model]["mean_spearman_rank_ic"] - control["mean_spearman_rank_ic"],
                        "turnover_increase": _difference(aggregates[model]["mean_rank_turnover"], control["mean_rank_turnover"]),
                    }
    eligible, rejected = [], []
    if blockers:
        primary = blocker_class
        rejected = list(REQUIRED_MODELS)
    else:
        for model in REQUIRED_MODELS:
            reasons = _gate_reasons(aggregates[model], thresholds, momentum)
            aggregates[model]["gate_reasons"] = reasons
            (eligible if not reasons else rejected).append(model)
        primary = "READY_FOR_PORTFOLIO_REPLAY" if eligible else "REJECTED"
    transition_requests = []
    if not blockers:
        for component in components:
            transition_requests.append({
                "experiment_id": component["experiment_id"],
                "requested_status": (
                    "ELIGIBLE_FOR_PORTFOLIO_REPLAY"
                    if component["model_id"] in eligible else "REJECTED"
                ),
                "reason": f"Wave 4 gate result for {component['model_id']}",
            })

    material = {
        "component_plan_checksum": plan_checksum,
        "component_manifest_checksums": sorted(input_checksums.values()),
        "dataset_identity": plan.get("dataset_id"),
        "feature_schema_panel": sorted({row.get("feature_schema_hash") for row in components}),
        "target_contract_panel": sorted({(row.get("target_contract_id"), row.get("target_contract_hash")) for row in components}),
        "target_provenance": plan.get("target_provenance_contract_version"),
        "ranking_contract_panel": sorted({row.get("ranking_contract_id") for row in components}),
        "fold_panel": sorted({row.get("fold_id") for row in components}),
        "date_panel": list(plan.get("decision_dates") or []),
        "model_panel": list(plan.get("fitted_models") or []),
        "metric_contract": METRIC_CONTRACT, "gate_contract": GATE_CONTRACT,
        "gate_thresholds": thresholds, "source_commit": plan.get("source_commit"),
    }
    checksum = canonical_hash(material)
    return {
        "report_contract_version": CONTRACT,
        "campaign_id": plan.get("campaign_id"),
        "campaign_checksum": checksum,
        "component_plan_checksum": plan_checksum,
        "expected_component_count": 15,
        "observed_component_count": len(paths),
        "per_component_validation": validations,
        "per_date_metrics": per_date,
        "per_model_aggregate_metrics": aggregates,
        "momentum_control_comparison": momentum,
        "metric_contract": {
            "version": METRIC_CONTRACT,
            "aggregation_rule": "unweighted arithmetic mean across all five fixed decision dates; IC IR is mean divided by population standard deviation",
            "deterministic_tie_break": "score descending, then asset_id ascending",
        },
        "gate_thresholds": {"version": GATE_CONTRACT, **thresholds},
        "failure_blocker_reasons": sorted(set(blockers)),
        "effective_fitted_model_count": len(REQUIRED_MODELS) if not blockers else 0,
        "effective_material_trial_count": len(components),
        "experiment_trial_counts": trial_counts,
        "experiment_ledger_transition_requests": transition_requests,
        "primary_status": primary,
        "models_eligible_for_portfolio_replay": eligible,
        "models_rejected": rejected,
        "source_commit": plan.get("source_commit"),
        "input_manifest_checksums": input_checksums,
        "campaign_identity_material": material,
        "portfolio_replay_performed": False,
    }


def _validate_component(path, plan, expected):
    result = {"manifest_path": str(path), "model_id": None, "decision_date": None, "reasons": []}
    try:
        manifest = _json(path)
    except (OSError, ValueError, TypeError):
        result["reasons"].append("COMPONENT_MANIFEST_MISSING_OR_CORRUPT")
        return result
    model = str(manifest.get("selector_model_identity") or "")
    date = str(manifest.get("prediction_date") or "")
    result.update(model_id=model, decision_date=date, manifest_checksum=manifest.get("manifest_checksum"))
    row = expected.get((model, date))
    if row is None:
        result["reasons"].append("UNPLANNED_COMPONENT")
        return result
    checks = {
        "campaign_id": ("CAMPAIGN_ID_MISMATCH", plan.get("campaign_id")),
        "component_id": ("PLAN_COMPONENT_ID_MISMATCH", row.get("component_id")),
        "experiment_id": ("EXPERIMENT_ID_MISSING_OR_MISMATCH", row.get("experiment_id")),
        "target_provenance_contract_version": ("TARGET_PROVENANCE_V2_REQUIRED", "stock_level_target_provenance_v2"),
        "daily_stock_spine_identity": ("DAILY_SPINE_IDENTITY_MISMATCH", row.get("daily_spine_id")),
        "symbol_registry_identity": ("SYMBOL_REGISTRY_IDENTITY_MISMATCH", row.get("symbol_registry_id")),
        "feature_schema_hash": ("FEATURE_SCHEMA_IDENTITY_MISMATCH", row.get("feature_schema_hash")),
        "target_contract_version": ("TARGET_CONTRACT_IDENTITY_MISMATCH", row.get("target_contract_id")),
        "target_contract_hash": ("TARGET_CONTRACT_IDENTITY_MISMATCH", row.get("target_contract_hash")),
        "ranking_contract_version": ("RANKING_CONTRACT_IDENTITY_MISMATCH", row.get("ranking_contract_id")),
        "fold_identity": ("FOLD_IDENTITY_MISMATCH", row.get("fold_id")),
        "git_commit": ("SOURCE_COMMIT_INCOMPATIBLE", row.get("source_commit")),
    }
    frozen = manifest.get("frozen_selector_dataset_identity") or {}
    if frozen.get("dataset_id") != row.get("dataset_id") or frozen.get("dataset_checksum") != row.get("dataset_checksum"):
        result["reasons"].append("DATASET_IDENTITY_MISMATCH")
    for field, (reason, wanted) in checks.items():
        if manifest.get(field) != wanted:
            result["reasons"].append(reason)
    if manifest.get("publication_status") not in {"complete", "COMPLETE"}:
        result["reasons"].append("COMPONENT_INCOMPLETE")
    if manifest.get("validation_status") != "VERIFIED_STRICT_OOS":
        result["reasons"].append("STRICT_OOS_VERIFICATION_ABSENT")
    checksum = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_checksum"})
    if manifest.get("manifest_checksum") != checksum:
        result["reasons"].append("MANIFEST_CHECKSUM_MISMATCH")
    artifact = Path(str(manifest.get("prediction_artifact_path") or ""))
    if not artifact.is_file():
        result["reasons"].append("PREDICTION_EVIDENCE_ABSENT")
        return result
    if _sha(artifact) != manifest.get("prediction_checksum"):
        result["reasons"].append("PREDICTION_CHECKSUM_MISMATCH")
        return result
    try:
        rows = list(csv.DictReader(artifact.open(encoding="utf-8", newline="")))
        result["rows"] = rows
        ids = [str(item.get("row_id") or "") for item in rows]
        economic = [(item.get("asset_id"), item.get("prediction_date")) for item in rows]
        scores = [float(item["selector_score"]) for item in rows]
        if len(rows) != manifest.get("prediction_row_count") or not rows or any(not value for value in ids):
            result["reasons"].append("PREDICTION_POPULATION_INCOMPLETE")
        population_sizes = {int(item["population_size"]) for item in rows if item.get("population_size")}
        if len(population_sizes) > 1 or (population_sizes and len(rows) != next(iter(population_sizes))):
            result["reasons"].append("PREDICTION_POPULATION_INCOMPLETE")
        if len(economic) != len(set(economic)):
            result["reasons"].append("DUPLICATE_ECONOMIC_ROWS")
        if not all(math.isfinite(value) for value in scores):
            result["reasons"].append("NONFINITE_SCORES")
        if len(set(scores)) == 1:
            result["reasons"].append("DEGENERATE_RANK_OUTPUT")
    except (OSError, ValueError, KeyError):
        result["reasons"].append("PREDICTION_EVIDENCE_MALFORMED")
    result["reasons"] = sorted(set(result["reasons"]))
    return result


def _metrics(rows):
    ordered = sorted(rows, key=lambda row: (-float(row["selector_score"]), str(row["asset_id"])))
    scores = [float(row["selector_score"]) for row in ordered]
    realised = [float(row["actual_forward_return_10d"]) for row in ordered]
    market = [float(row.get("market_return_10d") or 0.0) for row in ordered]
    residual = [value - benchmark for value, benchmark in zip(realised, market)]
    n = len(rows)
    relevance = _relevance(realised)
    largest_tie = max(scores.count(value) for value in set(scores))
    return {
        "spearman_rank_ic": _corr(_ranks(scores), _ranks(realised)),
        "pearson_ic": _corr(scores, realised),
        "market_residual_rank_ic": _corr(_ranks(scores), _ranks(residual)),
        "ndcg_at_10": _ndcg(relevance, 10), "ndcg_at_20": _ndcg(relevance, 20),
        "top_10_realised_return": mean(realised[:min(10, n)]),
        "top_20_realised_return": mean(realised[:min(20, n)]),
        "top_minus_bottom_spread": mean(realised[:min(10, n)]) - mean(realised[-min(10, n):]),
        "rank_turnover": _optional_mean(rows, "rank_turnover"),
        "top_10_continuity": _optional_mean(rows, "top_10_continuity"),
        "top_20_continuity": _optional_mean(rows, "top_20_continuity"),
        "prediction_coverage": _coverage(rows),
        "largest_tied_score_group": largest_tie,
        "largest_tied_score_fraction": largest_tie / n,
        "score_dispersion": pstdev(scores),
        "coefficient_stability": {"available": False, "reason": "COMPONENT_COEFFICIENT_EVIDENCE_NOT_AVAILABLE"},
        "row_count": n,
    }


def _aggregate(rows):
    names = (
        "spearman_rank_ic", "pearson_ic", "market_residual_rank_ic", "ndcg_at_10",
        "ndcg_at_20", "top_10_realised_return", "top_20_realised_return",
        "top_minus_bottom_spread", "rank_turnover", "top_10_continuity",
        "top_20_continuity", "prediction_coverage", "largest_tied_score_group",
        "largest_tied_score_fraction", "score_dispersion",
    )
    result = {f"mean_{name}": _mean_available([row[name] for row in rows]) for name in names}
    values = [row["spearman_rank_ic"] for row in rows]
    result["ic_information_ratio"] = mean(values) / pstdev(values) if pstdev(values) else None
    result["positive_rank_ic_date_count"] = sum(value > 0 for value in values)
    result["date_count"] = len(rows)
    result["coefficient_stability"] = {"available": False, "reason": "COMPONENT_COEFFICIENT_EVIDENCE_NOT_AVAILABLE"}
    return result


def _gate_reasons(metrics, thresholds, momentum):
    reasons = []
    if metrics["mean_spearman_rank_ic"] <= thresholds["minimum_mean_rank_ic"]:
        reasons.append("NONPOSITIVE_AGGREGATE_RANK_IC")
    if metrics["mean_prediction_coverage"] < thresholds["minimum_date_coverage"]:
        reasons.append("CATASTROPHIC_COVERAGE")
    if metrics["mean_largest_tied_score_fraction"] > thresholds["maximum_largest_tied_score_fraction"]:
        reasons.append("INSUFFICIENT_RANK_DIVERSITY")
    if metrics["positive_rank_ic_date_count"] < thresholds["minimum_positive_rank_ic_dates"]:
        reasons.append("INSUFFICIENT_MULTI_DATE_STABILITY")
    if momentum["available"]:
        comparison = metrics["momentum_comparison"]
        if comparison["rank_ic_improvement"] <= thresholds["minimum_rank_ic_improvement_vs_momentum"]:
            reasons.append("NO_INCREMENTAL_INFORMATION_VS_MOMENTUM")
        turnover = comparison["turnover_increase"]
        if (
            turnover is not None
            and turnover > thresholds["maximum_turnover_increase_vs_momentum"]
            and comparison["rank_ic_improvement"] < thresholds["minimum_rank_ic_improvement_for_turnover_exception"]
        ):
            reasons.append("UNCOMPENSATED_TURNOVER_INCREASE")
    return reasons


def _momentum(paths, plan):
    # Optional controls use the same evidence shape but are never fitted components.
    expected = {("momentum", date): {} for date in plan.get("decision_dates") or []}
    validations = [_validate_component(path, plan, expected) for path in paths]
    return {"available": False, "reason": "OPTIONAL_MOMENTUM_EVIDENCE_INCOMPATIBLE", "validation": validations}


def _coverage(rows):
    values = [row.get("population_size") for row in rows if row.get("population_size") not in {None, ""}]
    return len(rows) / int(values[0]) if values and int(values[0]) else 1.0


def _optional_mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return mean(values) if values else None


def _mean_available(values):
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def _difference(left, right):
    return left - right if left is not None and right is not None else None


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            output[index] = rank
        start = end
    return output


def _corr(left, right):
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x-left_mean)*(y-right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x-left_mean)**2 for x in left) * sum((y-right_mean)**2 for y in right))
    if not denominator:
        raise ValueError("Degenerate correlation")
    return numerator / denominator


def _relevance(values):
    ranks = _ranks(values)
    n = max(len(values), 1)
    return [min(4, int((rank - 1) * 5 / n)) for rank in ranks]


def _ndcg(relevance, k):
    actual = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(relevance[:k]))
    ideal = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(sorted(relevance, reverse=True)[:k]))
    return actual / ideal if ideal else 0.0


def _logical(payload):
    return canonical_hash({key: value for key, value in payload.items() if key != "logical_checksum"})


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _publish(root, report):
    root = Path(root)
    temp = root / f".wave4-report-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    try:
        (temp / "wave4_campaign_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        (temp / "wave4_campaign_report.md").write_text(_markdown(report), encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        os.replace(temp / "wave4_campaign_report.json", root / "wave4_campaign_report.json")
        os.replace(temp / "wave4_campaign_report.md", root / "wave4_campaign_report.md")
        shutil.rmtree(temp)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def _markdown(report):
    return (
        "# Wave 4 selector campaign gate\n\n"
        f"- Status: `{report['primary_status']}`\n"
        f"- Campaign: `{report['campaign_id']}`\n"
        f"- Checksum: `{report['campaign_checksum']}`\n"
        f"- Components: `{report['observed_component_count']}/{report['expected_component_count']}`\n"
        f"- Eligible for portfolio replay: `{', '.join(report['models_eligible_for_portfolio_replay']) or 'none'}`\n"
        f"- Blockers: `{', '.join(report['failure_blocker_reasons']) or 'none'}`\n"
        "- This evaluation did not run portfolio replay.\n"
    )
