from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata

GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}


@dataclass(frozen=True)
class StockAlphaExperimentReportPaths:
    json_path: Path
    markdown_path: Path
    registry_path: Path


def write_stock_alpha_experiment_report(config: dict[str, Any]) -> StockAlphaExperimentReportPaths:
    ml = config.get("ml", {})
    if not bool(ml.get("stock_alpha_experiment_report_enabled", True)):
        raise ValueError("ml.stock_alpha_experiment_report_enabled is false")
    settings = StockLevelResearchConfig.from_mapping(config)
    all_paths = _discover_paths(
        settings.output_dir,
        ml,
        legacy_output_paths_allowed=bool(ml.get("stock_alpha_allow_legacy_output_paths", False)),
    )
    level = str(ml.get("stock_alpha_experiment_report_level", "standard"))
    if level not in {"dev_smoke", "standard", "overnight"}:
        raise ValueError("ml.stock_alpha_experiment_report_level must be dev_smoke, standard, or overnight")
    required_names = {"stock_artifact", "alpha_features", "benchmark", "target_comparison", "portfolio_replay", "portfolio_policy_sweep"}
    if level == "overnight": required_names.add("overnight_summary")
    paths = {name: path for name, path in all_paths.items() if name in required_names}
    payloads, checks = validate_stock_alpha_outputs(
        paths,
        expected_root=settings.output_dir,
        expected_run_size=settings.run_size,
        require_all=bool(ml.get("stock_alpha_experiment_report_require_all_outputs", False)) or level in {"standard", "overnight"},
        max_age_hours=float(ml.get("stock_alpha_experiment_report_max_age_hours", 24)),
        legacy_output_paths_allowed=bool(ml.get("stock_alpha_allow_legacy_output_paths", False)),
    )
    checks["report_level"] = level
    checks["optional_outputs"] = {name: str(path) for name, path in all_paths.items() if name not in required_names}
    row = _registry_row(config, settings, paths, payloads, checks)
    registry_path = Path(ml.get("stock_alpha_experiment_registry_path", "reports/ml/stock_alpha_experiment_registry.csv"))
    _append_registry(registry_path, row)
    report = {"mode": "stock_alpha_experiment_report_research_only", "run_id": row["run_id"], "timestamp": row["timestamp"], "report_level": level, "validation_passed": not checks["errors"], "validation": checks, "artifacts": {name: str(path) for name, path in paths.items()}, "registry_row": row, **stock_alpha_report_metadata(config, settings.output_dir, source_artifact_path=settings.base_artifact_path), **GUARDRAILS}
    output_paths = StockAlphaExperimentReportPaths(settings.output_dir / "stock_alpha_experiment_report.json", settings.output_dir / "stock_alpha_experiment_report.md", registry_path)
    writer = ResearchArtifactWriter()
    writer.write_json(output_paths.json_path, report)
    writer.write_markdown(output_paths.markdown_path, _markdown(report))
    return output_paths


def validate_stock_alpha_outputs(paths: dict[str, Path], *, expected_root: Path, expected_run_size: str, require_all: bool, max_age_hours: float, legacy_output_paths_allowed: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []; payloads = {}; mtimes = []
    root = expected_root.resolve(); unexpected: list[str] = []; legacy: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            (errors if require_all else warnings).append({"check": "file_exists", "artifact": name, "message": f"Missing output: {path}"})
            continue
        try:
            resolved = path.resolve()
            is_legacy = "reports/ml/benchmark/ml" in path.as_posix()
            if is_legacy: legacy.append(str(path))
            if root != resolved and root not in resolved.parents:
                unexpected.append(str(path))
                if not (is_legacy and legacy_output_paths_allowed):
                    errors.append({"check": "output_root", "artifact": name, "message": f"Output is outside configured root: {path}"})
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"check": "json_parse", "artifact": name, "message": str(exc)}); continue
        payloads[name] = payload; mtimes.append((name, path.stat().st_mtime))
        for field, expected in GUARDRAILS.items():
            if payload.get(field) != expected:
                errors.append({"check": "guardrail", "artifact": name, "field": field, "expected": expected, "actual": payload.get(field)})
        run_size = payload.get("run_size")
        if run_size is not None and run_size != expected_run_size:
            errors.append({"check": "run_size", "artifact": name, "expected": expected_run_size, "actual": run_size})
        _validate_winners(name, payload, errors)
        if name in {"benchmark", "portfolio_replay", "portfolio_policy_sweep"} and not _oos_dates(payload):
            errors.append({"check": "oos_dates", "artifact": name, "message": "OOS date count is missing or zero"})
    if len(mtimes) > 1:
        spread_hours = (max(value for _, value in mtimes) - min(value for _, value in mtimes)) / 3600
        if spread_hours > max_age_hours:
            errors.append({"check": "stale_mixed_outputs", "age_spread_hours": spread_hours, "max_age_hours": max_age_hours})
    counts = {name: _counts(payload) for name, payload in payloads.items() if any(_counts(payload))}
    if len(set(counts.values())) > 1:
        warnings.append({"check": "effective_counts", "message": "Counts differ across stages because artifacts, OOS predictions, and dev subsets have different eligible rows", "counts": counts})
    return payloads, {"errors": errors, "warnings": warnings, "checked_artifact_count": len(payloads), "missing_artifact_count": len(paths) - len(payloads), "output_root_validation_passed": not unexpected or (legacy_output_paths_allowed and set(unexpected) == set(legacy)), "unexpected_output_paths": unexpected, "legacy_output_paths_detected": legacy, "legacy_output_paths_allowed": legacy_output_paths_allowed}


def _discover_paths(output: Path, ml: dict[str, Any], *, legacy_output_paths_allowed: bool = False) -> dict[str, Path]:
    candidates = {
        "stock_artifact": [output / "stock_level_prediction_artifacts.json"],
        "alpha_features": [output / "stock_level_alpha_feature_audit.json"],
        "benchmark": [Path(ml.get("stock_level_model_ranking_benchmark_path", output / "enriched" / "stock_level_model_ranking_benchmark.json"))],
        "target_comparison": [output / "target_comparison" / "stock_level_target_comparison.json"],
        "portfolio_replay": [output / "portfolio_replay" / "stock_level_portfolio_replay_summary.json"],
        "portfolio_policy_sweep": [output / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.json"],
        "overnight_summary": [output / "overnight_stock_alpha_summary.json"],
    }
    if legacy_output_paths_allowed:
        candidates["alpha_features"].append(output / "overnight_stock_alpha" / "stock_level_alpha_feature_audit.json")
        candidates["benchmark"].append(output / "overnight_stock_alpha" / "enriched" / "stock_level_model_ranking_benchmark.json")
        candidates["target_comparison"].append(output / "overnight_stock_alpha" / "target_comparison" / "stock_level_target_comparison.json")
        candidates["portfolio_replay"].append(output / "overnight_stock_alpha" / "portfolio_replay" / "stock_level_portfolio_replay_summary.json")
        candidates["portfolio_policy_sweep"].append(output / "overnight_stock_alpha" / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.json")
        candidates["overnight_summary"].append(output / "overnight_stock_alpha" / "overnight_stock_alpha_summary.json")
    return {name: next((path for path in choices if path.exists()), choices[0]) for name, choices in candidates.items()}


def _validate_winners(name: str, payload: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    winners = payload.get("winners", {}) or {}
    for key, winner in winners.items():
        if not isinstance(winner, dict): continue
        if winner.get("status") in {"infeasible", "unavailable", "error"} or winner.get("infeasible_reason"):
            errors.append({"check": "winner_eligibility", "artifact": name, "winner": key})
        if winner.get("date_count") == 0 or winner.get("oos_date_count") == 0:
            errors.append({"check": "winner_date_count", "artifact": name, "winner": key})
    comparisons = json.dumps(payload.get("best_ml_vs_momentum_120d", payload.get("winners", {}).get("best_ml_vs_momentum_120d", {})))
    if comparisons not in ("{}", "null") and "momentum" not in comparisons:
        errors.append({"check": "momentum_baseline", "artifact": name, "message": "ML comparison does not identify momentum_120d"})


def _oos_dates(payload: dict[str, Any]) -> int:
    return int(payload.get("oos_date_count") or payload.get("effective_date_count") or max((row.get("date_count", 0) or 0 for row in payload.get("summary", [])), default=0))


def _counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(payload.get(key) or 0) for key in ("effective_row_count", "effective_date_count", "effective_symbol_count"))


def _registry_row(config: dict[str, Any], settings: StockLevelResearchConfig, paths: dict[str, Path], payloads: dict[str, dict[str, Any]], checks: dict[str, Any]) -> dict[str, Any]:
    benchmark = payloads.get("benchmark", {}); target = payloads.get("target_comparison", {}); portfolio = payloads.get("portfolio_replay", {}); sweep = payloads.get("portfolio_policy_sweep", {})
    rank = benchmark.get("best_ml_model") or {}; best_target = max(target.get("targets", []), key=lambda row: row.get("best_spearman_ic") or float("-inf"), default={}); best_portfolio = portfolio.get("winners", {}).get("best_by_net_return_after_costs") or {}; best_sweep = sweep.get("winners", {}).get("best_by_net_return_after_costs") or {}
    timestamp = datetime.now(timezone.utc).isoformat(); fingerprint = "|".join(f"{name}:{path.stat().st_mtime_ns if path.exists() else 0}" for name, path in paths.items())
    counts = _counts(benchmark)
    if not any(counts):
        counts = _counts(payloads.get("alpha_features", {}))
    comparison = sweep.get("winners", {}).get("best_ml_vs_momentum_120d", portfolio.get("best_ml_vs_momentum_120d", {})) or {}
    return {"run_id": f"stock-alpha-{hashlib.sha256(fingerprint.encode()).hexdigest()[:12]}", "timestamp": timestamp, "profile": str(config.get("research", {}).get("profile", settings.run_size)), "run_size": settings.run_size, "config_path": str(config.get("config_path", "config/config.yaml")), "output_dir": str(settings.output_dir), "source_artifact_path": str(settings.base_artifact_path), "enriched_artifact_path": str(settings.output_dir / "stock_level_prediction_artifacts_enriched.csv"), "benchmark_path": str(paths["benchmark"]), "target_comparison_path": str(paths["target_comparison"]), "portfolio_replay_path": str(paths["portfolio_replay"]), "portfolio_policy_sweep_path": str(paths["portfolio_policy_sweep"]), "row_count": counts[0], "date_count": counts[1], "symbol_count": counts[2], "oos_date_count": benchmark.get("oos_date_count"), "best_rank_model": rank.get("name"), "best_rank_ic": rank.get("mean_spearman_ic"), "best_rank_spread": rank.get("top_minus_bottom_spread"), "best_target": best_target.get("target_column"), "best_target_model": best_target.get("best_model_by_spearman_ic"), "best_portfolio_signal": best_portfolio.get("signal_column"), "best_portfolio_policy": best_portfolio.get("policy"), "best_portfolio_net_return": best_portfolio.get("net_return"), "best_portfolio_sharpe": best_portfolio.get("sharpe"), "best_portfolio_max_drawdown": best_portfolio.get("max_drawdown"), "best_policy_sweep_config": best_sweep.get("config_id"), "best_policy_sweep_net_return": best_sweep.get("net_return"), "best_policy_sweep_sharpe": best_sweep.get("sharpe"), "best_policy_sweep_turnover": best_sweep.get("average_turnover"), "ml_beats_momentum_120d": comparison.get("beats_momentum_120d"), "validation_passed": not checks["errors"], "production_validated": False, "promotion_thresholds_changed": False}


def _append_registry(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle: existing = list(csv.DictReader(handle))
    existing = [item for item in existing if item.get("run_id") != row["run_id"]]; existing.append(row)
    ResearchArtifactWriter().write_csv(path, existing, fieldnames=list(row))


def _markdown(payload: dict[str, Any]) -> str:
    row = payload["registry_row"]
    lines = ["# Stock Alpha Experiment Report", "", "Research only. Trading impact: none. Production validated: false.", "", f"- Run ID: `{payload['run_id']}`", f"- Run size: `{payload['run_size']}`", f"- Validation passed: {payload['validation_passed']}", f"- Errors: {len(payload['validation']['errors'])}", f"- Warnings: {len(payload['validation']['warnings'])}", f"- Best rank model: {row['best_rank_model']}", f"- Best portfolio: {row['best_portfolio_signal']} / {row['best_portfolio_policy']}", f"- Best sweep config: {row['best_policy_sweep_config']}", "- Promotion thresholds changed: false", ""]
    return "\n".join(lines)
