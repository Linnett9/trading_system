from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping

from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import file_digest
from core.research.ml.stock_level.selector_cost_aware_policy_evaluation import (
    build_selector_cost_aware_policy_evaluation,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    read_stock_level_artifact,
    write_stock_level_artifact,
)


ENSEMBLE_CONTRACT_VERSION = "selector_confidence_ensemble_contract_v1"
CONFIDENCE_CONTRACT_VERSION = "selector_confidence_contract_v1"
ABSTENTION_CONTRACT_VERSION = "selector_abstention_contract_v1"
SCHEMA_VERSION = "selector_confidence_ensemble_v1"


@dataclass(frozen=True)
class SelectorConfidenceEnsemblePaths:
    output_dir: Path
    contract_path: Path
    component_manifest_path: Path
    predictions_path: Path
    diagnostics_path: Path
    forecast_metrics_path: Path
    portfolio_metrics_path: Path
    abstention_metrics_path: Path
    comparison_json_path: Path
    comparison_markdown_path: Path


def write_selector_confidence_ensemble(config: Mapping[str, Any]) -> SelectorConfidenceEnsemblePaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.selector_confidence_ensemble.enabled is false")
    source_path = Path(settings["prediction_artifact_path"])
    rows = read_stock_level_artifact(source_path, required_columns={"candidate_id", "rebalance_date", "symbol", "prediction"})
    payload = build_selector_confidence_ensemble(rows, config=config, settings=settings, source_path=source_path)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SelectorConfidenceEnsemblePaths(
        output_dir=output_dir,
        contract_path=output_dir / "selector_ensemble_contract.json",
        component_manifest_path=output_dir / "selector_ensemble_component_manifest.csv",
        predictions_path=output_dir / "selector_ensemble_oos_predictions.parquet",
        diagnostics_path=output_dir / "selector_ensemble_confidence_diagnostics.parquet",
        forecast_metrics_path=output_dir / "selector_ensemble_forecast_metrics.csv",
        portfolio_metrics_path=output_dir / "selector_ensemble_portfolio_metrics.csv",
        abstention_metrics_path=output_dir / "selector_ensemble_abstention_metrics.csv",
        comparison_json_path=output_dir / "selector_ensemble_comparison.json",
        comparison_markdown_path=output_dir / "selector_ensemble_comparison.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.contract_path, payload["ensemble_contracts"])
    writer.write_csv(paths.component_manifest_path, payload["component_manifest"], fieldnames=_fields(payload["component_manifest"], ["ensemble_id", "candidate_id"]))
    write_stock_level_artifact(paths.predictions_path, payload["ensemble_predictions"], fieldnames=_prediction_fields(payload["ensemble_predictions"]), config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}})
    write_stock_level_artifact(paths.diagnostics_path, payload["confidence_diagnostics"], fieldnames=_diagnostic_fields(payload["confidence_diagnostics"]), config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}})
    writer.write_csv(paths.forecast_metrics_path, payload["forecast_metrics"], fieldnames=_fields(payload["forecast_metrics"], ["signal_id"]))
    writer.write_csv(paths.portfolio_metrics_path, payload["portfolio_metrics"], fieldnames=_fields(payload["portfolio_metrics"], ["signal_id", "policy_id"]))
    writer.write_csv(paths.abstention_metrics_path, payload["abstention_metrics"], fieldnames=_fields(payload["abstention_metrics"], ["ensemble_id"]))
    writer.write_json(paths.comparison_json_path, payload)
    writer.write_markdown(paths.comparison_markdown_path, _markdown(payload))
    return paths


def build_selector_confidence_ensemble(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    contracts = [_contract(cohort, settings) for cohort in settings["cohorts"]]
    ensemble_predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    forecast_metrics: list[dict[str, Any]] = []
    portfolio_metrics: list[dict[str, Any]] = []
    abstention_metrics: list[dict[str, Any]] = []
    blockers: list[str] = []

    for contract in contracts:
        component_rows = [row for row in rows if str(row.get("candidate_id")) in contract["component_candidate_ids"]]
        compatibility = _compatibility_audit(component_rows, contract)
        if compatibility["blockers"]:
            blockers.extend(f"{contract['ensemble_id']}:{blocker}" for blocker in compatibility["blockers"])
            continue
        manifests.extend(_component_manifest(component_rows, contract))
        built = _build_cohort(component_rows, contract, settings)
        ensemble_predictions.extend(built["predictions"])
        diagnostics.extend(built["diagnostics"])
        forecast_metrics.extend(_forecast_metrics(built["signals"], contract))
        abstention_metrics.append(_abstention_metrics(contract, built["predictions"]))
        portfolio_metrics.extend(_portfolio_comparison_metrics(built["portfolio_rows"], contract, settings))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "selector_confidence_ensemble_research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_status": "BOUNDED DIAGNOSTIC ONLY / NOT ENSEMBLE PROMOTION EVIDENCE",
        "source_prediction_artifact_identity": _source_identity(source_path, rows),
        "dataset_identity": _dataset_identity(rows),
        "benchmark_availability": _benchmark_availability(rows),
        "ensemble_contracts": contracts,
        "component_manifest": sorted(manifests, key=lambda row: (row["ensemble_id"], row["candidate_id"])),
        "ensemble_predictions": sorted(ensemble_predictions, key=lambda row: (row["ensemble_id"], row["rebalance_date"], row["symbol"])),
        "confidence_diagnostics": sorted(diagnostics, key=lambda row: (row["ensemble_id"], row["rebalance_date"], row["symbol"])),
        "forecast_metrics": sorted(forecast_metrics, key=lambda row: row["signal_id"]),
        "portfolio_metrics": sorted(portfolio_metrics, key=lambda row: (row["signal_id"], row["policy_id"])),
        "abstention_metrics": sorted(abstention_metrics, key=lambda row: row["ensemble_id"]),
        "matched_comparison": _matched_comparison(portfolio_metrics),
        "lineage": {
            "source_prediction_artifact_path": str(source_path) if source_path else None,
            "source_artifact_sha256": file_digest(source_path) if source_path and source_path.exists() else None,
            "ensemble_contract_identities": {row["ensemble_id"]: row["ensemble_contract_identity"] for row in contracts},
            "confidence_contract_identity": _hash(settings["confidence"]),
            "abstention_contract_identity": _hash(settings["abstention"]),
            "portfolio_policy_identity": _hash(settings["portfolio_policy"]),
            "cost_model_identity": _hash({"cost_bps": settings["cost_bps"], "slippage_bps": settings["slippage_bps"]}),
            "configuration_hash": _hash(settings),
            "code_commit": MLCoreArtifactWriter.git_commit(),
        },
        "warnings": _warnings(rows, settings),
        "blockers": blockers,
        "training_performed": False,
        "final_fit_performed": False,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "promotion_thresholds_changed": False,
    }
    return payload


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("selector_confidence_ensemble", {}) or {})
    output_dir = stock_alpha_output_dir(config) / "selector_confidence_ensemble"
    return {
        "enabled": bool(raw.get("enabled", False)),
        "prediction_artifact_path": str(raw.get("prediction_artifact_path", "")),
        "output_dir": str(raw.get("output_dir", output_dir)),
        "cohorts": list(raw.get("cohorts", [])),
        "confidence": {
            "minimum_confidence": float(dict(raw.get("confidence", {}) or {}).get("minimum_confidence", 0.60)),
            "margin_weight": float(dict(raw.get("confidence", {}) or {}).get("margin_weight", 1.0)),
        },
        "abstention": {
            "enabled": bool(dict(raw.get("abstention", {}) or {}).get("enabled", True)),
            "minimum_confidence": float(dict(raw.get("abstention", {}) or {}).get("minimum_confidence", 0.60)),
            "maximum_disagreement": float(dict(raw.get("abstention", {}) or {}).get("maximum_disagreement", 0.40)),
            "minimum_components": int(dict(raw.get("abstention", {}) or {}).get("minimum_components", 3)),
            "minimum_model_families": int(dict(raw.get("abstention", {}) or {}).get("minimum_model_families", 2)),
            "minimum_entry_margin": dict(raw.get("abstention", {}) or {}).get("minimum_entry_margin"),
        },
        "sizing": dict(raw.get("sizing", {"mode": "tiered", "tiers": [{"minimum_confidence": 0.80, "multiplier": 1.0}, {"minimum_confidence": 0.60, "multiplier": 0.5}]})),
        "portfolio_policy": dict(raw.get("portfolio_policy", {}) or {}),
        "cost_bps": float(raw.get("cost_bps", 10.0)),
        "slippage_bps": float(raw.get("slippage_bps", 5.0)),
        "max_position_weight": float(raw.get("max_position_weight", 0.05)),
        "maximum_decision_dates": raw.get("maximum_decision_dates"),
        "maximum_symbols": raw.get("maximum_symbols"),
        "comparison_requires_benchmark": bool(raw.get("comparison_requires_benchmark", False)),
    }


def _contract(cohort: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    components = list(cohort["component_selection"]["candidate_ids"])
    if len(components) != len(set(components)):
        raise ValueError(f"Duplicate ensemble component candidate_ids: {components}")
    contract = {
        "contract_version": ENSEMBLE_CONTRACT_VERSION,
        "ensemble_id": str(cohort["ensemble_id"]),
        "target_id": str(cohort["target_id"]),
        "component_candidate_ids": components,
        "score_normalisation": str(cohort.get("score_normalisation", "cross_sectional_percentile")),
        "aggregation_method": str(cohort.get("aggregation_method", "mean_rank")),
        "minimum_components_per_row": int(cohort.get("minimum_components_per_row", 2)),
        "missing_component_behavior": str(cohort.get("missing_component_behavior", "reduce_confidence")),
        "require_shared_dataset_identity": bool(cohort.get("require_shared_dataset_identity", True)),
        "require_shared_fold_plan_identity": bool(cohort.get("require_shared_fold_plan_identity", True)),
        "require_shared_target_contract": bool(cohort.get("require_shared_target_contract", True)),
        "confidence_definition": {
            "contract_version": CONFIDENCE_CONTRACT_VERSION,
            "formula": "component_coverage * model_coverage * (1 - disagreement) * margin_component",
            **settings["confidence"],
        },
        "abstention_definition": {"contract_version": ABSTENTION_CONTRACT_VERSION, **settings["abstention"]},
        "sizing_definition": settings["sizing"],
    }
    contract["component_models"] = sorted({_candidate_model(component) for component in components})
    contract["component_seeds"] = sorted({_candidate_seed(component) for component in components if _candidate_seed(component) is not None})
    contract["ensemble_contract_identity"] = _hash(contract)
    return contract


def _compatibility_audit(rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not rows:
        blockers.append("no_component_rows")
    components = {str(row.get("candidate_id")) for row in rows}
    missing = sorted(set(contract["component_candidate_ids"]) - components)
    if missing and len(components) < int(contract["minimum_components_per_row"]):
        blockers.append(f"missing_components={missing}")
    duplicate_keys = [(row.get("candidate_id"), row.get("rebalance_date"), row.get("symbol")) for row in rows]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        blockers.append("duplicate_component_date_symbol_rows")
    if any(str(row.get("strict_oos", True)).lower() in {"false", "0", ""} for row in rows):
        blockers.append("non_strict_oos_component_rows")
    target_ids = {str(row.get("target_id")) for row in rows if str(row.get("target_id", "")).strip()}
    if target_ids and target_ids != {contract["target_id"]}:
        blockers.append(f"mixed_or_unexpected_targets={sorted(target_ids)}")
    if contract["require_shared_dataset_identity"] and len(_nonempty_set(rows, "dataset_identity")) > 1:
        blockers.append("mixed_dataset_identities")
    if contract["require_shared_fold_plan_identity"] and len(_nonempty_set(rows, "fold_plan_identity")) > 1:
        blockers.append("mixed_fold_plan_identities")
    if contract["require_shared_target_contract"] and len(_nonempty_set(rows, "target_contract_identity")) > 1:
        blockers.append("mixed_target_contract_identities")
    return {"blockers": blockers}


def _build_cohort(rows: list[dict[str, Any]], contract: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    bounded = _bounded(rows, settings)
    normalized = _normalized_component_scores(bounded, contract)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bounded:
        by_key[(str(row["rebalance_date"]), str(row["symbol"]).upper())] = row
    model_scores = _model_level_scores(normalized)
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    signals: dict[str, list[dict[str, Any]]] = {"best_component": [], "mean_rank": [], "confidence_rank": [], "confidence_abstention": []}
    for date in sorted({key[0] for key in model_scores}):
        symbols = sorted({key[1] for key in model_scores if key[0] == date})
        raw_scores = {symbol: _aggregate_model_scores(model_scores[(date, symbol)], contract) for symbol in symbols}
        ranks = _score_ranks(raw_scores)
        boundary_margin = _top_n_margin(raw_scores, int(settings.get("portfolio_policy", {}).get("top_n", 3)))
        for symbol in symbols:
            source = by_key[(date, symbol)]
            groups = model_scores[(date, symbol)]
            flat_scores = [score for item in groups.values() for score in item["seed_scores"]]
            model_values = [item["model_score"] for item in groups.values()]
            component_count = len(flat_scores)
            model_count = len(model_values)
            seed_count = component_count
            disagreement = _bounded_range(model_values)
            seed_dispersion = mean([_bounded_range(item["seed_scores"]) for item in groups.values()]) if groups else 1.0
            component_coverage = component_count / len(contract["component_candidate_ids"])
            model_coverage = model_count / len(contract["component_models"])
            margin_component = max(0.0, min(1.0, boundary_margin if boundary_margin is not None else 0.0))
            confidence = max(0.0, min(1.0, component_coverage * model_coverage * (1.0 - disagreement) * max(0.0, margin_component)))
            abstention_status = _abstention_status(confidence, disagreement, component_count, model_count, boundary_margin, contract)
            ensemble_score = raw_scores[symbol]
            sized_score = ensemble_score * _confidence_multiplier(confidence, settings["sizing"]) if abstention_status == "eligible" else None
            row = {
                "ensemble_id": contract["ensemble_id"],
                "target_id": contract["target_id"],
                "rebalance_date": _text(date),
                "decision_date": _text(date),
                "decision_timestamp": _text(source.get("decision_timestamp")),
                "symbol": symbol,
                "ensemble_score": ensemble_score,
                "ensemble_rank": ranks[symbol],
                "confidence": confidence,
                "disagreement": disagreement,
                "component_count": component_count,
                "model_count": model_count,
                "seed_count": seed_count,
                "score_margin": boundary_margin,
                "abstention_status": abstention_status,
                "strict_oos": True,
                "fold_id": _text(source.get("fold_id")),
                "dataset_identity": _text(source.get("dataset_identity")),
                "target_contract_identity": _text(source.get("target_contract_identity")),
                "fold_plan_identity": _text(source.get("fold_plan_identity")),
                "ensemble_contract_identity": contract["ensemble_contract_identity"],
                "actual_investable_return_10d": source.get("actual_investable_return_10d"),
                "actual_benchmark_return_10d": source.get("actual_benchmark_return_10d"),
            }
            predictions.append(row)
            diagnostics.append({**row, "seed_dispersion": seed_dispersion, "model_dispersion": disagreement, "component_coverage": component_coverage, "model_coverage": model_coverage, "margin_component": margin_component, "minimum_confidence": contract["abstention_definition"]["minimum_confidence"]})
            best_component = max((item for item in normalized if item["rebalance_date"] == date and item["symbol"] == symbol), key=lambda item: item["normalized_score"])
            base = _portfolio_row(source, contract["ensemble_id"], best_component["normalized_score"], "best_component")
            signals["best_component"].append(base)
            signals["mean_rank"].append(_portfolio_row(source, contract["ensemble_id"], ensemble_score, "mean_rank"))
            signals["confidence_rank"].append(_portfolio_row(source, contract["ensemble_id"], ensemble_score * confidence, "confidence_rank"))
            if sized_score is not None:
                signals["confidence_abstention"].append(_portfolio_row(source, contract["ensemble_id"], sized_score, "confidence_abstention", confidence=confidence, disagreement=disagreement, abstention_status=abstention_status))
    portfolio_rows = {name: values for name, values in signals.items()}
    return {"predictions": predictions, "diagnostics": diagnostics, "signals": signals, "portfolio_rows": portfolio_rows}


def _normalized_component_scores(rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (date, component), group in _group(rows, lambda row: (str(row["rebalance_date"]), str(row["candidate_id"]))).items():
        finite = [(str(row["symbol"]).upper(), finite_number(row.get("prediction")), row) for row in group]
        finite = [(symbol, value, row) for symbol, value, row in finite if value is not None]
        scores = _percentile_scores(finite)
        for symbol, _, row in finite:
            output.append({
                "rebalance_date": date,
                "symbol": symbol,
                "candidate_id": component,
                "model_id": str(row.get("model_id") or _candidate_model(component)),
                "seed": row.get("seed") if row.get("seed") not in (None, "") else _candidate_seed(component),
                "normalized_score": scores[symbol],
            })
    return output


def _model_level_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    output: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for (date, symbol, model), group in _group(rows, lambda row: (row["rebalance_date"], row["symbol"], row["model_id"])).items():
        scores = [float(row["normalized_score"]) for row in group]
        output.setdefault((date, symbol), {})[model] = {"model_score": mean(scores), "seed_scores": scores}
    return output


def _aggregate_model_scores(groups: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> float:
    values = [float(item["model_score"]) for _, item in sorted(groups.items())]
    method = contract["aggregation_method"]
    if method == "median_rank":
        return float(median(values))
    if method == "trimmed_mean_rank" and len(values) > 2:
        trimmed = sorted(values)[1:-1]
        return mean(trimmed)
    return mean(values) if values else 0.0


def _portfolio_comparison_metrics(signals: Mapping[str, list[dict[str, Any]]], contract: Mapping[str, Any], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for name, rows in signals.items():
        if not rows:
            continue
        payload = build_selector_cost_aware_policy_evaluation(
            rows,
            config={"ml": {}},
            source_path=None,
            settings={
                "enabled": True,
                "prediction_artifact_path": "",
                "candidate_id": f"{contract['ensemble_id']}::{name}",
                "prediction_column": "prediction",
                "prediction_semantics": "rank_score",
                "allow_csv_fallback": False,
                "output_dir": "",
                "top_n": int(settings.get("portfolio_policy", {}).get("top_n", 3)),
                "cost_bps": settings["cost_bps"],
                "slippage_bps": settings["slippage_bps"],
                "max_position_weight": settings["max_position_weight"],
                "min_position_weight": 0.0,
                "maximum_decision_dates": None,
                "maximum_symbols": None,
                "development_period": {},
                "evaluation_period": {},
                "policies": [settings["portfolio_policy"]],
            },
        )
        for row in payload["policy_metrics"]:
            metrics.append({**row, "signal_id": name, "ensemble_id": contract["ensemble_id"]})
    return metrics


def _portfolio_row(source: Mapping[str, Any], ensemble_id: str, score: float, signal_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "candidate_id": f"{ensemble_id}::{signal_id}",
        "rebalance_date": source["rebalance_date"],
        "symbol": source["symbol"],
        "fold_id": source.get("fold_id"),
        "prediction": score,
        "actual_investable_return_10d": source.get("actual_investable_return_10d"),
        "actual_benchmark_return_10d": source.get("actual_benchmark_return_10d"),
        "strict_oos": True,
        **extra,
    }


def _forecast_metrics(signals: Mapping[str, list[dict[str, Any]]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for signal_id, values in signals.items():
        by_date = _group(values, lambda row: str(row["rebalance_date"]))
        spreads = []
        for group in by_date.values():
            ordered = sorted(group, key=lambda row: -float(row["prediction"]))
            if len(ordered) >= 2:
                top = mean(float(row["actual_investable_return_10d"]) for row in ordered[: max(1, len(ordered) // 4)])
                bottom = mean(float(row["actual_investable_return_10d"]) for row in ordered[-max(1, len(ordered) // 4):])
                spreads.append(top - bottom)
        rows.append({
            "ensemble_id": contract["ensemble_id"],
            "signal_id": signal_id,
            "coverage": len(values),
            "date_count": len(by_date),
            "top_minus_bottom_realized_return_spread": mean(spreads) if spreads else None,
            "hit_rate": mean([spread > 0.0 for spread in spreads]) if spreads else None,
        })
    return rows


def _abstention_metrics(contract: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    abstained = [row for row in rows if row["abstention_status"] != "eligible"]
    by_date = _group(rows, lambda row: str(row["rebalance_date"]))
    top_n = 3
    confident_breadth = [sum(1 for row in group if row["abstention_status"] == "eligible") for group in by_date.values()]
    return {
        "ensemble_id": contract["ensemble_id"],
        "stock_level_abstention_rate": len(abstained) / total if total else 0.0,
        "decision_dates_with_abstention": sum(1 for group in by_date.values() if any(row["abstention_status"] != "eligible" for row in group)),
        "average_confident_breadth": mean(confident_breadth) if confident_breadth else 0.0,
        "average_unfilled_portfolio_slots": mean([max(0, top_n - count) for count in confident_breadth]) if confident_breadth else 0.0,
        "low_confidence_count": sum(1 for row in abstained if row["abstention_status"] == "low_confidence"),
        "high_disagreement_count": sum(1 for row in abstained if row["abstention_status"] == "high_disagreement"),
        "insufficient_components_count": sum(1 for row in abstained if row["abstention_status"] == "insufficient_components"),
    }


def _component_manifest(rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for candidate_id in contract["component_candidate_ids"]:
        component_rows = [row for row in rows if str(row.get("candidate_id")) == candidate_id]
        output.append({
            "ensemble_id": contract["ensemble_id"],
            "candidate_id": candidate_id,
            "model_id": _candidate_model(candidate_id),
            "seed": _candidate_seed(candidate_id),
            "row_count": len(component_rows),
            "first_decision_date": min((str(row["rebalance_date"]) for row in component_rows), default=None),
            "last_decision_date": max((str(row["rebalance_date"]) for row in component_rows), default=None),
            "target_contract_identity": _first_present(component_rows, "target_contract_identity"),
            "fold_plan_identity": _first_present(component_rows, "fold_plan_identity"),
            "dataset_identity": _first_present(component_rows, "dataset_identity"),
        })
    return output


def _bounded(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    dates = sorted({str(row["rebalance_date"]) for row in rows})
    if settings.get("maximum_decision_dates"):
        dates = dates[: int(settings["maximum_decision_dates"])]
    symbols = sorted({str(row["symbol"]).upper() for row in rows})
    if settings.get("maximum_symbols"):
        symbols = symbols[: int(settings["maximum_symbols"])]
    return [row for row in rows if str(row["rebalance_date"]) in dates and str(row["symbol"]).upper() in symbols]


def _percentile_scores(items: list[tuple[str, float, Mapping[str, Any]]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: (-float(item[1]), item[0]))
    count = len(ordered)
    return {symbol: 1.0 - ((index - 1) / max(1, count - 1)) for index, (symbol, _, _) in enumerate(ordered, start=1)}


def _score_ranks(scores: Mapping[str, float]) -> dict[str, int]:
    return {symbol: index for index, symbol in enumerate(sorted(scores, key=lambda item: (-scores[item], item)), start=1)}


def _top_n_margin(scores: Mapping[str, float], top_n: int) -> float | None:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) <= top_n or top_n <= 0:
        return None
    return max(0.0, ordered[top_n - 1] - ordered[top_n])


def _abstention_status(confidence: float, disagreement: float, component_count: int, model_count: int, margin: float | None, contract: Mapping[str, Any]) -> str:
    rule = contract["abstention_definition"]
    if not rule.get("enabled", True):
        return "eligible"
    if component_count < int(rule["minimum_components"]) or model_count < int(rule["minimum_model_families"]):
        return "insufficient_components"
    if disagreement > float(rule["maximum_disagreement"]):
        return "high_disagreement"
    if confidence < float(rule["minimum_confidence"]):
        return "low_confidence"
    if rule.get("minimum_entry_margin") is not None and (margin is None or margin < float(rule["minimum_entry_margin"])):
        return "weak_margin"
    return "eligible"


def _confidence_multiplier(confidence: float, sizing: Mapping[str, Any]) -> float:
    mode = str(sizing.get("mode", "binary"))
    if mode == "continuous":
        return max(0.0, min(1.0, confidence))
    if mode == "tiered":
        for tier in sorted(sizing.get("tiers", []), key=lambda row: -float(row["minimum_confidence"])):
            if confidence >= float(tier["minimum_confidence"]):
                return float(tier["multiplier"])
        return 0.0
    return 1.0 if confidence >= float(sizing.get("minimum_confidence", 0.6)) else 0.0


def _bounded_range(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    return max(0.0, min(1.0, max(values) - min(values)))


def _benchmark_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(row.get("rebalance_date", "")) for row in rows})
    present_dates = {str(row.get("rebalance_date", "")) for row in rows if finite_number(row.get("actual_benchmark_return_10d")) is not None}
    non_null = sum(1 for row in rows if finite_number(row.get("actual_benchmark_return_10d")) is not None)
    return {
        "benchmark_return_column": "actual_benchmark_return_10d",
        "benchmark_non_null_count": non_null,
        "benchmark_missing_count": len(rows) - non_null,
        "benchmark_date_coverage": len(present_dates) / len(dates) if dates else 0.0,
        "benchmark_relative_metrics_available": bool(dates) and set(dates) == present_dates,
        "benchmark_source_identity": _hash(sorted(present_dates)),
    }


def _source_identity(path: Path | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if path and path.exists() and path.suffix.lower() == ".parquet" and rows:
        return artifact_identity(path, rows=rows, fieldnames=list(rows[0]))
    return {"path": str(path) if path else None, "sha256": file_digest(path) if path and path.exists() else None, "row_count": len(rows)}


def _dataset_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "decision_date_count": len({str(row.get("rebalance_date")) for row in rows}),
        "symbol_count": len({str(row.get("symbol")).upper() for row in rows}),
        "dataset_identities": sorted(_nonempty_set(rows, "dataset_identity")),
        "decision_grid_identity": _first_present(rows, "decision_grid_identity"),
        "universe_identity": _hash(sorted({str(row.get("symbol")).upper() for row in rows})),
        "logical_content_hash": _hash([(row.get("candidate_id"), row.get("rebalance_date"), row.get("symbol"), row.get("prediction")) for row in rows]),
    }


def _matched_comparison(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in metrics if row["signal_id"] == "best_component"), None)
    if not baseline:
        return []
    return [
        {
            "signal_id": row["signal_id"],
            "policy_id": row["policy_id"],
            "net_return_delta_vs_best_component": _delta(row, baseline, "net_cumulative_return"),
            "turnover_avoided_vs_best_component": _delta(baseline, row, "annualised_turnover"),
            "costs_avoided_vs_best_component": _delta(baseline, row, "transaction_costs"),
        }
        for row in metrics
        if row is not baseline
    ]


def _warnings(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[str]:
    warnings = ["BOUNDED DIAGNOSTIC ONLY", "NOT ENSEMBLE PROMOTION EVIDENCE"]
    benchmark = _benchmark_availability(rows)
    if not benchmark["benchmark_relative_metrics_available"]:
        warnings.append("benchmark_returns_unavailable; benchmark_relative_metrics_disabled")
        if settings["comparison_requires_benchmark"]:
            warnings.append("configured_benchmark_relative_comparison_blocked")
    warnings.append("same_dates_used_for_fixed_threshold_diagnostic; do_not_promote")
    return warnings


def _candidate_model(candidate_id: str) -> str:
    parts = str(candidate_id).split("::")
    return parts[1] if len(parts) > 1 else str(candidate_id)


def _candidate_seed(candidate_id: str) -> int | None:
    for part in str(candidate_id).split("::"):
        if part.startswith("seed_"):
            return int(part.replace("seed_", ""))
    return None


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _nonempty_set(rows: list[dict[str, Any]], column: str) -> set[str]:
    return {str(row.get(column)) for row in rows if str(row.get(column, "")).strip()}


def _group(rows: list[Any], key_fn) -> dict[Any, list[Any]]:
    output: dict[Any, list[Any]] = {}
    for row in rows:
        output.setdefault(key_fn(row), []).append(row)
    return output


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    a, b = finite_number(left.get(key)), finite_number(right.get(key))
    return a - b if a is not None and b is not None else None


def _first_present(rows: list[dict[str, Any]], column: str) -> str | None:
    return next((str(row.get(column)) for row in rows if str(row.get(column, "")).strip()), None)


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    return [*preferred, *[key for key in dict.fromkeys(key for row in rows for key in row) if key not in preferred]] if rows else preferred


def _prediction_fields(rows: list[dict[str, Any]]) -> list[str]:
    return _fields(rows, ["ensemble_id", "target_id", "rebalance_date", "symbol", "ensemble_score", "ensemble_rank", "confidence", "disagreement", "component_count", "model_count", "seed_count", "score_margin", "abstention_status", "strict_oos", "fold_id", "dataset_identity", "target_contract_identity", "fold_plan_identity", "ensemble_contract_identity"])


def _diagnostic_fields(rows: list[dict[str, Any]]) -> list[str]:
    return _fields(rows, ["ensemble_id", "rebalance_date", "symbol", "confidence", "disagreement", "component_coverage", "model_coverage", "seed_dispersion", "model_dispersion", "margin_component", "abstention_status"])


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Confidence Ensemble",
        "",
        "BOUNDED DIAGNOSTIC ONLY. NOT ENSEMBLE PROMOTION EVIDENCE.",
        "",
        f"- Source rows: {payload['dataset_identity']['row_count']}",
        f"- Benchmark relative metrics available: {payload['benchmark_availability']['benchmark_relative_metrics_available']}",
        f"- Ensembles: {', '.join(row['ensemble_id'] for row in payload['ensemble_contracts'])}",
        "",
        "| Signal | Policy | Net Return | Turnover | Costs |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["portfolio_metrics"]:
        lines.append(f"| {row['signal_id']} | {row['policy_id']} | {row['net_cumulative_return']} | {row['annualised_turnover']} | {row['transaction_costs']} |")
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines) + "\n"
