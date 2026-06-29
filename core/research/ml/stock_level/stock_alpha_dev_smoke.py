from __future__ import annotations

import time
import json
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_level_prediction_artifacts import write_stock_level_prediction_artifacts
from core.research.ml.stock_level.stock_level_prediction_artifacts import TARGET_TYPES
from core.research.ml.stock_level.stock_level_alpha_features import write_stock_level_alpha_features
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import write_stock_level_model_ranking_benchmark
from core.research.ml.stock_level.stock_level_target_comparison import write_stock_level_target_comparison
from core.research.ml.stock_level.stock_level_portfolio_replay import write_stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import write_stock_level_portfolio_policy_sweep
from core.research.ml.stock_level.stock_alpha_experiment_report import write_stock_alpha_experiment_report
from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.overnight_stock_alpha_runner import _artifact_status
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata


@dataclass(frozen=True)
class StockAlphaDevSmokeStages:
    artifact: Callable[[dict[str, Any]], Any] = write_stock_level_prediction_artifacts
    features: Callable[[dict[str, Any]], Any] = write_stock_level_alpha_features
    benchmark: Callable[[dict[str, Any]], Any] = write_stock_level_model_ranking_benchmark
    targets: Callable[[dict[str, Any]], Any] = write_stock_level_target_comparison
    portfolio: Callable[[dict[str, Any]], Any] = write_stock_level_portfolio_replay
    sweep: Callable[[dict[str, Any]], Any] = write_stock_level_portfolio_policy_sweep
    report: Callable[[dict[str, Any]], Any] = write_stock_alpha_experiment_report


def write_stock_alpha_dev_smoke(config: dict[str, Any], *, stages: StockAlphaDevSmokeStages | None = None) -> Path:
    stages = stages or StockAlphaDevSmokeStages()
    smoke = deepcopy(config); ml = smoke.setdefault("ml", {})
    # This command is a safety boundary: configured benchmark/full profiles are
    # deliberately overridden so a smoke invocation cannot launch a large run.
    ml.update({"stock_alpha_run_size": "dev", "stock_alpha_dev_max_dates": min(int(ml.get("stock_alpha_dev_max_dates", 24)), 24), "stock_alpha_dev_max_symbols": min(int(ml.get("stock_alpha_dev_max_symbols", 20)), 20), "stock_ranker_min_train_dates": min(int(ml.get("stock_ranker_min_train_dates", 8)), 8), "stock_ranker_test_window_dates": min(int(ml.get("stock_ranker_test_window_dates", 4)), 4), "stock_ranker_embargo_dates": min(int(ml.get("stock_ranker_embargo_dates", 1)), 1), "stock_ranker_include_sequence_models": False, "stock_ranker_model_n_jobs": 1, "sklearn_n_jobs": 1, "stock_target_comparison_n_jobs": 1, "stock_portfolio_policy_sweep_n_jobs": 1, "stock_portfolio_policy_sweep_max_configs_dev": min(int(ml.get("stock_portfolio_policy_sweep_max_configs_dev", 8)), 8), "stock_alpha_overnight_run_attribution": False, "stock_alpha_experiment_report_level": "dev_smoke"})
    settings = StockLevelResearchConfig.from_mapping(smoke); output = settings.output_dir
    ml.update({"output_dir": str(output), "stock_alpha_output_dir_override": True})
    output.mkdir(parents=True, exist_ok=True); timings = {}; output_files: list[str] = []

    def run(name: str, action: Callable[[], Any]) -> Any:
        print(f"[stock-alpha-dev-smoke] START {name}"); started = time.perf_counter(); result = action(); elapsed = time.perf_counter() - started
        timings[name] = {"status": "executed", "seconds": elapsed}; print(f"[stock-alpha-dev-smoke] END {name} elapsed={elapsed:.3f}s"); return result

    base = Path(ml.get("stock_level_base_prediction_artifacts_path", output / "stock_level_prediction_artifacts.csv"))
    artifact_stale = _artifact_status(smoke, StockLevelResearchConfig.from_mapping(smoke))["refresh_required"] or not _has_required_target_schema(base)
    if not base.exists() or artifact_stale: run("stock-level artifact", lambda: stages.artifact(smoke))
    else: timings["stock-level artifact"] = {"status": "skipped_existing", "seconds": 0.0}
    feature = run("alpha features", lambda: stages.features(smoke))
    ml["stock_level_prediction_artifacts_path"] = str(feature.enriched_csv_path)
    benchmark = run("model ranking benchmark", lambda: stages.benchmark(smoke))
    ml["stock_level_model_ranking_benchmark_path"] = str(benchmark.json_path); ml["stock_level_model_oos_predictions_path"] = str(benchmark.predictions_path)
    target_result = run("target comparison", lambda: stages.targets(smoke))
    skipped_targets: list[dict[str, Any]] = []
    target_availability: list[dict[str, Any]] = []
    target_json_path = getattr(target_result, "json_path", None)
    if target_json_path and Path(target_json_path).exists():
        target_payload = json.loads(Path(target_json_path).read_text(encoding="utf-8"))
        skipped_targets = list(target_payload.get("skipped_targets", []))
        target_availability = [{key: row.get(key) for key in ("target_column", "target_column_present", "target_non_null_count", "eligible_row_count", "eligible_date_count", "eligible_symbol_count", "status", "skip_reason_code")} for row in target_payload.get("targets", [])]
        if skipped_targets:
            timings["target comparison"]["status"] = "completed_with_skips"
            timings["target comparison"]["skipped_target_count"] = len(skipped_targets)
    portfolio_result = run("portfolio replay", lambda: stages.portfolio(smoke)); sweep_result = run("portfolio policy sweep", lambda: stages.sweep(smoke)); report = run("experiment report", lambda: stages.report(smoke))
    sweep_coverage: dict[str, Any] = {}
    sweep_winners: dict[str, Any] = {}
    sweep_json_path = getattr(sweep_result, "json_path", None)
    if sweep_json_path and Path(sweep_json_path).exists():
        sweep_payload = json.loads(Path(sweep_json_path).read_text(encoding="utf-8"))
        sweep_coverage = dict(sweep_payload.get("baseline_coverage", {}))
        sweep_coverage["best_baseline_policy"] = sweep_payload.get("winners", {}).get("best_baseline_policy")
        sweep_coverage["best_ml_vs_momentum_120d"] = sweep_payload.get("winners", {}).get("best_ml_vs_momentum_120d")
        sweep_winners = dict(sweep_payload.get("winners", {}))
    portfolio_winners: dict[str, Any] = {}
    portfolio_json_path = getattr(portfolio_result, "json_path", None)
    if portfolio_json_path and Path(portfolio_json_path).exists():
        portfolio_winners = dict(json.loads(Path(portfolio_json_path).read_text(encoding="utf-8")).get("winners", {}))
    validation = {"errors": [], "warnings": []}
    report_json_path = getattr(report, "json_path", None)
    if report_json_path and Path(report_json_path).exists():
        validation = json.loads(Path(report_json_path).read_text(encoding="utf-8")).get("validation", validation)
    for result in (feature, benchmark, target_result, portfolio_result, sweep_result, report):
        output_files.extend(str(value) for value in getattr(result, "__dict__", {}).values() if isinstance(value, Path))
    path = output / "stock_alpha_dev_smoke_report.json"; markdown_path = output / "stock_alpha_dev_smoke_report.md"
    payload = {"mode": "stock_alpha_dev_smoke_research_only", "status": "completed_with_skips" if skipped_targets else "completed", **stock_alpha_report_metadata(smoke, output, source_artifact_path=base, generated_artifact_paths=[Path(value) for value in output_files]), "target_column_availability": target_availability, "target_comparison_summary": {"status": "completed_with_skips" if skipped_targets else "completed", "skipped_target_count": len(skipped_targets)}, "skipped_targets": skipped_targets, "policy_sweep_baseline_coverage": sweep_coverage, "portfolio_replay_winners": portfolio_winners, "portfolio_policy_sweep_winners": sweep_winners, "experiment_validation": validation, "effective_caps": {"max_dates": ml["stock_alpha_dev_max_dates"], "max_symbols": ml["stock_alpha_dev_max_symbols"], "max_policy_configs": ml["stock_portfolio_policy_sweep_max_configs_dev"]}, "attribution_enabled": False, "timings": timings, "experiment_report_path": str(report.json_path), "output_files": output_files, "research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}
    writer = ResearchArtifactWriter(); writer.write_json(path, payload); writer.write_markdown(markdown_path, _markdown(payload))
    return path


def _has_required_target_schema(path: Path) -> bool:
    if not path.exists(): return False
    with path.open(newline="", encoding="utf-8") as handle:
        columns = set(next(csv.reader(handle), []))
    return set(TARGET_TYPES).issubset(columns)


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Stock Alpha Dev Smoke Report", "", "## Overall status", f"- Status: {payload['status']}", f"- Run size: {payload['run_size']}", f"- Output directory: `{payload['output_dir']}`", "", "## Effective caps"]
    lines.extend(f"- {key}: {value}" for key, value in payload["effective_caps"].items())
    lines.extend(["", "## Stage timings"]); lines.extend(f"- {name}: {row['status']} ({row['seconds']:.3f}s)" for name, row in payload["timings"].items())
    lines.extend(["", "## Target availability", "| Target | Present | Non-null | Dates | Symbols | Status |", "|---|---|---:|---:|---:|---|"])
    for row in payload["target_column_availability"]: lines.append(f"| {row['target_column']} | {row['target_column_present']} | {row['target_non_null_count']} | {row['eligible_date_count']} | {row['eligible_symbol_count']} | {row['status']} |")
    lines.extend(["", "## Target comparison summary", f"- {payload['target_comparison_summary']}", "", "## Policy sweep baseline coverage", f"- {payload['policy_sweep_baseline_coverage']}", "", "## Portfolio replay winners", f"- {payload['portfolio_replay_winners']}", "", "## Portfolio policy sweep winners", f"- {payload['portfolio_policy_sweep_winners']}", "", "## Experiment validation errors/warnings", f"- Errors: {payload['experiment_validation'].get('errors', [])}", f"- Warnings: {payload['experiment_validation'].get('warnings', [])}", "", "## Output files"])
    lines.extend(f"- `{path}`" for path in payload["output_files"]); lines.extend(["", "Promotion thresholds changed: false.", ""])
    return "\n".join(lines)
