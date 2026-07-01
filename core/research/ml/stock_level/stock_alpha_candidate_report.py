from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir

GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}
MIN_PREDICTION_COVERAGE = 0.95


@dataclass(frozen=True)
class StockAlphaCandidateReportPaths:
    json_path: Path
    markdown_path: Path
    csv_path: Path


def write_stock_alpha_candidate_report(config: Mapping[str, Any]) -> StockAlphaCandidateReportPaths:
    output = stock_alpha_output_dir(config)
    ml = dict(config.get("ml", {}) or {})
    profile = str(ml.get("profile", config.get("research_profile", {}).get("name", ml.get("stock_alpha_run_size", "benchmark"))))
    report = build_stock_alpha_candidate_report(output, resolved_profile=profile, resolved_run_size=str(ml.get("stock_alpha_run_size", "benchmark")))
    paths = StockAlphaCandidateReportPaths(output / "stock_alpha_candidate_report.json", output / "stock_alpha_candidate_report.md", output / "stock_alpha_candidate_table.csv")
    output.mkdir(parents=True, exist_ok=True)
    paths.json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(report), encoding="utf-8")
    rows = report["candidate_table"]
    fields = ["category", "source_file", "model_or_signal", "policy", "target", "metric_name", "metric_value", "prediction_non_null_count", "prediction_coverage_ratio", "valid_prediction_output", "beats_baseline", "beats_momentum_120d", "status", "notes"]
    with paths.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return paths


def build_stock_alpha_candidate_report(output: Path, *, resolved_profile: str | None = None, resolved_run_size: str | None = None) -> dict[str, Any]:
    warnings: list[str] = []; errors: list[str] = []; inspected: list[str] = []
    flat_pair = (output / "stock_level_model_ranking_benchmark.json", output / "stock_level_model_oos_predictions.csv")
    nested_pairs = {group: (output / group / "stock_level_model_ranking_benchmark.json", output / group / "stock_level_model_oos_predictions.csv") for group in ("baseline", "enriched")}
    has_flat = any(path.exists() for path in flat_pair); has_nested = any(path.exists() for pair in nested_pairs.values() for path in pair)
    layout = "mixed" if has_flat and has_nested else ("dev_flat" if has_flat else "overnight_nested")
    if has_nested:
        benchmark_pairs = {group: pair for group, pair in nested_pairs.items()}
        if has_flat:
            warnings.append(
                "Mixed flat and nested benchmark outputs detected; using nested "
                "baseline/enriched outputs and ignoring flat stock-alpha benchmark outputs."
            )
    elif has_flat:
        benchmark_pairs = {"dev": flat_pair}
    else:
        benchmark_pairs = {group: pair for group, pair in nested_pairs.items()}
    critical = [path for pair in benchmark_pairs.values() for path in pair]
    for path in critical:
        if not path.exists(): errors.append(f"Missing critical benchmark output: {path}")
    optional_candidates = [(output / "stock_alpha_run_manifest.json",), (output / "stock_alpha_run_status.json",), (output / "overnight_stock_alpha_summary.json",), (output / "stock_level_target_comparison.json", output / "target_comparison/stock_level_target_comparison.json"), (output / "stock_level_portfolio_replay_summary.json", output / "portfolio_replay/stock_level_portfolio_replay_summary.json"), (output / "stock_level_portfolio_policy_sweep.json", output / "portfolio_policy_sweep/stock_level_portfolio_policy_sweep.json"), (output / "stock_alpha_experiment_report.json",)]
    optional = [next((path for path in choices if path.exists()), choices[0]) for choices in optional_candidates]
    for path in optional:
        if path.exists(): inspected.append(str(path))
        else: warnings.append(f"Missing optional output: {path}")
    groups: dict[str, Any] = {}; table: list[dict[str, Any]] = []
    guardrail_violation = False
    for group, (ranking_path, predictions_path) in benchmark_pairs.items():
        if not ranking_path.exists() or not predictions_path.exists(): continue
        inspected += [str(ranking_path), str(predictions_path)]
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        guardrail_violation |= any(ranking.get(k) != v for k, v in GUARDRAILS.items())
        coverage = _prediction_coverage(predictions_path, ranking)
        completed_names = set(ranking.get("completed_models", []))
        unavailable_names = _unavailable_model_names(ranking.get("unavailable_models", []))
        valid_names = {row["model"] for row in coverage if row["valid_for_leaderboard"] and row["model"] in completed_names and row["model"] not in unavailable_names}
        valid_leaderboard = [row for row in ranking.get("leaderboard", []) if row.get("name") in valid_names]
        winners = {metric: max(valid_leaderboard, key=lambda row: _number(row.get(metric)), default=None) for metric in ("mean_spearman_ic", "top_minus_bottom_spread", "spread_sharpe")}
        invalid = [row for row in coverage if row["model"] in completed_names and row["model"] not in unavailable_names and not row["valid_for_leaderboard"]]
        unavailable = ranking.get("unavailable_models", [])
        groups[group] = {"requested_model_set": ranking.get("requested_model_set"), "effective_model_set": ranking.get("effective_model_set"), "included_models": ranking.get("included_models", []), "excluded_models": ranking.get("excluded_models", []), "stock_ranker_model_set": ranking.get("stock_ranker_model_set"), "task_completed_models": ranking.get("completed_models", []), "valid_completed_models": sorted(valid_names), "invalid_prediction_models": invalid, "unavailable_models": unavailable, "prediction_coverage": coverage, "winners": winners}
        for row in coverage:
            if row["model"] in unavailable_names:
                table.append(_table_row("unavailable_model", predictions_path, row["model"], "unavailable", row))
            elif row["model"] in completed_names and not row["valid_for_leaderboard"]:
                table.append(_table_row("invalid_completed_model", predictions_path, row["model"], row["status"], row))
        for metric, winner in winners.items():
            if winner: table.append(_table_row(f"best_{group}_{metric}", ranking_path, winner.get("name"), "valid", winner, metric))
    if guardrail_violation: errors.append("Guardrail violation detected in benchmark output")
    invalid_count = sum(len(g["invalid_prediction_models"]) for g in groups.values())
    valid_count = sum(len(g["valid_completed_models"]) for g in groups.values())
    if valid_count == 0: errors.append("No valid scored ML models")
    if invalid_count: warnings.append(f"{invalid_count} task-completed model outputs are invalid")
    legacy = [str(p) for p in output.rglob("*.json") if "reports/ml/benchmark/ml" in p.as_posix()] if output.exists() else []
    if legacy: warnings.append("Legacy output paths detected")
    status = "red" if errors else ("yellow" if warnings else "green")
    baseline_best = groups.get("baseline", {}).get("winners", {}).get("mean_spearman_ic")
    enriched_best = groups.get("enriched", {}).get("winners", {}).get("mean_spearman_ic")
    enriched_helped = bool(baseline_best and enriched_best and _number(enriched_best.get("mean_spearman_ic")) > _number(baseline_best.get("mean_spearman_ic")))
    return {"mode": "stock_alpha_candidate_report", **GUARDRAILS, "resolved_profile": resolved_profile, "resolved_run_size": resolved_run_size, "output_dir": str(output), "layout_detected": layout, "candidate_status": status, "verdict": "Worth deeper validation" if status == "green" else "Not ready for promotion", "run_complete": not errors, "guardrails_preserved": not guardrail_violation, "benchmark_outputs": {"present": [str(p) for p in critical if p.exists()], "missing": [str(p) for p in critical if not p.exists()]}, "model_validation": groups, "enriched_features_helped": enriched_helped, "warnings": warnings, "errors": errors, "legacy_paths_detected": legacy, "next_recommended_action": "Fix invalid prediction producers before rerunning benchmarks" if invalid_count else "Review candidate evidence and continue research-only validation", "files_inspected": inspected, "candidate_table": table}


def _prediction_coverage(path: Path, ranking: Mapping[str, Any]) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    expected_keys = {(r.get("rebalance_date", ""), r.get("symbol", "")) for r in rows}; result = []
    for model in dict.fromkeys([*ranking.get("requested_models", []), *ranking.get("completed_models", [])]):
        column = f"stock_level_predicted_forward_return_10d_{model}"
        if column not in (rows[0] if rows else {}):
            status, raw_count, numeric_values, finite_values, keys = "invalid_missing_prediction_column", 0, [], [], set()
        else:
            raw_count = sum(not _is_missing_token(r.get(column)) for r in rows)
            parsed = [(r, _parse_float(r.get(column))) for r in rows]
            numeric_values = [value for _, value in parsed if value is not None]
            finite_rows = [(row, value) for row, value in parsed if value is not None and math.isfinite(value)]
            finite_values = [value for _, value in finite_rows]
            keys = {(r.get("rebalance_date", ""), r.get("symbol", "")) for r, _ in finite_rows}
            if not finite_values: status = "invalid_empty_predictions"
            elif len(finite_values) < len(numeric_values) or raw_count > len(numeric_values): status = "invalid_non_finite_predictions"
            elif len(finite_values) / max(len(rows), 1) < MIN_PREDICTION_COVERAGE or not keys.issubset(expected_keys): status = "invalid_low_coverage"
            else: status = "valid"
        unique = len(set(finite_values))
        coverage_ratio = len(finite_values) / max(len(rows), 1)
        result.append({"model": model, "prediction_column": column, "status": status, "raw_non_null_count": raw_count, "numeric_non_null_count": len(numeric_values), "finite_prediction_count": len(finite_values), "non_null_prediction_count": len(finite_values), "non_null_prediction_date_count": len({k[0] for k in keys}), "non_null_prediction_symbol_count": len({k[1] for k in keys}), "expected_oos_row_count": len(rows), "prediction_coverage_ratio": coverage_ratio, "coverage_ratio": coverage_ratio, "all_predictions_null": not finite_values, "predictions_finite": bool(finite_values) and len(finite_values) == len(numeric_values), "rows_align": keys.issubset(expected_keys), "constant_or_nearly_constant": unique <= 1 if finite_values else None, "valid_for_leaderboard": status == "valid"})
    return result


def _parse_float(value: Any) -> float | None:
    if _is_missing_token(value): return None
    try: return float(value)
    except (TypeError, ValueError): return None


def _is_missing_token(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "none", "null"}


def _unavailable_model_names(unavailable: Any) -> set[str]:
    names: set[str] = set()
    for item in unavailable or []:
        if isinstance(item, str): names.add(item)
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("model")
            if name: names.add(str(name))
    return names


def _number(value: Any) -> float:
    try: return float(value) if value is not None else float("-inf")
    except (TypeError, ValueError): return float("-inf")


def _table_row(category: str, source: Path, model: Any, status: str, row: Mapping[str, Any], metric: str = "") -> dict[str, Any]:
    return {"category": category, "source_file": str(source), "model_or_signal": model or "", "policy": "", "target": "", "metric_name": metric, "metric_value": row.get(metric, "") if metric else "", "prediction_non_null_count": row.get("non_null_prediction_count", ""), "prediction_coverage_ratio": row.get("coverage_ratio", ""), "valid_prediction_output": row.get("valid_for_leaderboard", status == "valid"), "beats_baseline": "", "beats_momentum_120d": "", "status": status, "notes": ""}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Stock Alpha Candidate Report", "", f"- Resolved profile: `{report.get('resolved_profile')}`", f"- Resolved run size: `{report.get('resolved_run_size')}`", f"- Output directory: `{report.get('output_dir')}`", f"- Layout detected: `{report.get('layout_detected')}`", "", "## Verdict", "", str(report["verdict"]), "", "## Candidate status", "", f"**{str(report['candidate_status']).upper()}**", "", "## Guardrails", "", "Research only: true. Trading impact: none. Production validated: false. Promotion thresholds changed: false."]
    for group, validation in report["model_validation"].items():
        lines += ["", f"## {group.title()} model status", "", f"- Valid completed models: {', '.join(validation['valid_completed_models']) or 'none'}", f"- Invalid completed models: {', '.join(row['model'] for row in validation['invalid_prediction_models']) or 'none'}", f"- Unavailable models: {', '.join(sorted(_unavailable_model_names(validation['unavailable_models']))) or 'none'}"]
    for title, key in [("Run completeness", "run_complete"), ("Prediction coverage validation", "model_validation"), ("Warnings and errors", "warnings"), ("Next recommended action", "next_recommended_action"), ("Files inspected", "files_inspected")]: lines += ["", f"## {title}", "", f"```json\n{json.dumps(report[key], indent=2)}\n```"]
    return "\n".join(lines) + "\n"
