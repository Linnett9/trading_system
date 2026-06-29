from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import time
from datetime import datetime, timezone

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import build_stock_level_model_ranking_benchmark
from core.research.ml.stock_level_benchmark_data import _available_feature_columns
from core.research.ml.stock_level_benchmark_data import _number
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps


@dataclass(frozen=True)
class StockLevelTargetComparisonPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_stock_level_target_comparison(config: dict[str, Any]) -> StockLevelTargetComparisonPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    thread_caps = apply_stock_alpha_worker_caps(config)
    stage_started_at = datetime.now(timezone.utc).isoformat(); stage_started = time.perf_counter()
    if not settings.target_comparison_enabled:
        raise ValueError("ml.stock_ranker_target_comparison_enabled is false")
    rows = CsvRowRepository().read(settings.artifact_path)
    rows, run_profile = apply_stock_alpha_run_profile(rows, settings)
    features = _available_feature_columns(rows, include_engineered=settings.include_engineered_features)
    effective_workers = min(settings.target_comparison_n_jobs, len(settings.target_columns))
    nested_model_jobs = 1 if effective_workers > 1 else settings.model_n_jobs
    nested_sklearn_jobs = 1 if effective_workers > 1 else settings.sklearn_n_jobs

    def run(target: str) -> dict[str, Any]:
        target_started = time.perf_counter()
        print(f"[stock-alpha] START target comparison target={target}")
        availability = _target_availability(rows, target, features)
        minimum_dates = settings.min_train_dates + settings.embargo_dates + 1
        reason_code, reason = _availability_failure(availability, minimum_dates)
        if reason_code:
            row = _skipped_summary(target, availability, reason, reason_code, input_path=str(settings.artifact_path)); row["elapsed_seconds"] = time.perf_counter() - target_started; return row
        try:
            _, payload = build_stock_level_model_ranking_benchmark(
                rows, target_column=target, feature_columns=features,
                min_train_dates=settings.min_train_dates, test_window_dates=settings.test_window_dates,
                embargo_dates=settings.embargo_dates, random_seed=settings.random_seed,
                sklearn_n_jobs=nested_sklearn_jobs, model_n_jobs=nested_model_jobs,
                include_sequence_models=settings.include_sequence_models,
                sequence_length=settings.sequence_length, sequence_epochs=settings.sequence_epochs,
                sequence_batch_size=settings.sequence_batch_size, sequence_device=settings.sequence_device,
            )
        except Exception as exc:  # isolated research target; recorded fail-soft below
            print(f"[stock-alpha] SKIP target comparison target={target} reason={type(exc).__name__}: {exc}")
            row = _skipped_summary(target, availability, f"{type(exc).__name__}: {exc}", "target_execution_error", input_path=str(settings.artifact_path), status="skipped_target_error"); row["elapsed_seconds"] = time.perf_counter() - target_started; return row
        print(f"[stock-alpha] END target comparison target={target}")
        row = _summary(target, payload, availability, str(settings.artifact_path)); row["elapsed_seconds"] = time.perf_counter() - target_started; return row

    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            summaries = list(executor.map(run, settings.target_columns))
    else:
        summaries = [run(target) for target in settings.target_columns]
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = StockLevelTargetComparisonPaths(output / "stock_level_target_comparison.csv", output / "stock_level_target_comparison.json", output / "stock_level_target_comparison.md")
    writer = ResearchArtifactWriter()
    fieldnames = list(dict.fromkeys(key for row in summaries for key in row)) or ["target_column"]
    writer.write_csv(paths.csv_path, summaries, fieldnames=fieldnames)
    skipped = [row for row in summaries if row["status"] != "completed"]
    payload = {"mode": "stock_level_target_comparison_research_only", "status": "completed_with_skips" if skipped else "completed", "started_at": stage_started_at, "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - stage_started, "skipped_target_count": len(skipped), "skipped_targets": skipped, "targets": summaries, "target_timings": {row["target_column"]: row["elapsed_seconds"] for row in summaries}, **run_profile, **stock_alpha_report_metadata(config, output, source_artifact_path=settings.artifact_path), "parallelism": {"requested_workers": settings.target_comparison_n_jobs, "effective_workers": effective_workers, "nested_stock_ranker_model_n_jobs": nested_model_jobs, "nested_sklearn_n_jobs": nested_sklearn_jobs, "nested_torch_num_threads": 1 if effective_workers > 1 else thread_caps["torch_num_threads"]}, "thread_caps": thread_caps, "promotion_thresholds_changed": False, "research_only": True, "trading_impact": "none", "production_validated": False}
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("kind") == "ml_model" and row.get(metric) is not None]
    return max(eligible, key=lambda row: float(row[metric])) if eligible else {}


def _summary(target: str, payload: dict[str, Any], availability: dict[str, Any], input_path: str) -> dict[str, Any]:
    leaderboard = payload["leaderboard"]
    ic, spread, sharpe, risk = (_best(leaderboard, metric) for metric in ("mean_spearman_ic", "top_minus_bottom_spread", "spread_sharpe", "risk_adjusted_spread"))
    return {
        "status": "completed",
        "target_column": target,
        **availability,
        "input_path": input_path,
        "skip_reason_code": None,
        "skip_reason": None,
        "oos_date_count": payload["oos_date_count"],
        "best_model_by_spearman_ic": ic.get("name"), "best_spearman_ic": ic.get("mean_spearman_ic"),
        "best_model_by_top_minus_bottom_spread": spread.get("name"), "best_top_minus_bottom_spread": spread.get("top_minus_bottom_spread"),
        "best_model_by_spread_sharpe": sharpe.get("name"), "best_spread_sharpe": sharpe.get("spread_sharpe"),
        "best_model_by_risk_adjusted_spread": risk.get("name"), "best_risk_adjusted_spread": risk.get("risk_adjusted_spread"),
        "top_decile_hit_rate": ic.get("top_decile_hit_rate"),
        "beats_momentum_120d_identical_oos_dates": payload["ml_beats_momentum_120d"],
        "momentum_comparison_model": payload["best_ml_vs_momentum_120d"]["model"],
        "production_validated": False,
        "promotion_thresholds_changed": False,
    }


def _target_availability(rows: list[dict[str, Any]], target: str, features: tuple[str, ...]) -> dict[str, Any]:
    present = any(target in row for row in rows)
    target_rows = [row for row in rows if _number(row.get(target)) is not None]
    eligible = [row for row in target_rows if row.get("rebalance_date") and row.get("symbol")]
    return {"target_column_present": present, "target_non_null_count": len(target_rows), "eligible_row_count": len(eligible), "eligible_date_count": len({row["rebalance_date"] for row in eligible}), "eligible_symbol_count": len({str(row["symbol"]).upper() for row in eligible})}


def _availability_failure(availability: dict[str, Any], minimum_dates: int) -> tuple[str | None, str | None]:
    if not availability["target_column_present"]: return "column_missing", "Target column is absent from the comparison input"
    if availability["target_non_null_count"] == 0: return "column_present_all_null", "Target column is present but all values are null"
    if availability["eligible_symbol_count"] < 2: return "insufficient_symbols", f"Need at least 2 eligible symbols; found {availability['eligible_symbol_count']}"
    if availability["eligible_row_count"] < 2: return "insufficient_rows", f"Need at least 2 eligible rows; found {availability['eligible_row_count']}"
    if availability["eligible_date_count"] < minimum_dates: return "insufficient_dates", f"Need at least {minimum_dates} eligible rebalance dates; found {availability['eligible_date_count']}"
    return None, None


def _skipped_summary(target: str, availability: dict[str, Any], reason: str, reason_code: str, *, input_path: str, status: str = "skipped_insufficient_data") -> dict[str, Any]:
    return {"status": status, "target_column": target, **availability, "input_path": input_path, "skip_reason_code": reason_code, "skip_reason": reason, "oos_date_count": 0, "best_model_by_spearman_ic": None, "best_spearman_ic": None, "best_model_by_top_minus_bottom_spread": None, "best_top_minus_bottom_spread": None, "best_model_by_spread_sharpe": None, "best_spread_sharpe": None, "best_model_by_risk_adjusted_spread": None, "best_risk_adjusted_spread": None, "top_decile_hit_rate": None, "beats_momentum_120d_identical_oos_dates": False, "momentum_comparison_model": None, "production_validated": False, "promotion_thresholds_changed": False}


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Stock-Level Target Comparison", "", "Research only. Trading impact: none. Production validated: false.", "", f"- Status: `{payload['status']}`", f"- Skipped targets: {payload['skipped_target_count']}", f"- Run size: `{payload.get('run_size', 'benchmark')}`", f"- Effective workers: {payload.get('parallelism', {}).get('effective_workers', 1)}", "", "| Target | Status | Eligible dates | OOS dates | Best IC | Best spread | Skip reason |", "|---|---|---:|---:|---|---|---|"]
    for row in payload["targets"]:
        lines.append(f"| {row['target_column']} | {row['status']} | {row['eligible_date_count']} | {row['oos_date_count']} | {row['best_model_by_spearman_ic']} | {row['best_model_by_top_minus_bottom_spread']} | {row['skip_reason'] or ''} |")
    lines.extend(["", "Promotion thresholds changed: false.", ""])
    return "\n".join(lines)
