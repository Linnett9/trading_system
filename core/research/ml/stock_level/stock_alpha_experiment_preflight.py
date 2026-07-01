from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


TARGET_COLUMN = "actual_forward_return_10d"
GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}


@dataclass(frozen=True)
class StockAlphaExperimentPreflightPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_experiment_preflight(
    config: Mapping[str, Any],
) -> StockAlphaExperimentPreflightPaths:
    payload = build_stock_alpha_experiment_preflight(config)
    output = stock_alpha_output_dir(config) / "preflight"
    paths = StockAlphaExperimentPreflightPaths(
        json_path=output / "stock_alpha_experiment_preflight.json",
        markdown_path=output / "stock_alpha_experiment_preflight.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_experiment_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    source = _source_predictions_path(ml)
    source_exists = bool(source and source.exists())
    header = _csv_header(source) if source_exists and source is not None else []
    required_columns = _required_columns(ml)
    found_columns = [column for column in required_columns if column in header] if source_exists else []
    missing_columns = [column for column in required_columns if column not in header] if source_exists else []
    columns_not_checked_reason = "" if source_exists else "source_file_missing"
    guardrails = {key: ml.get(key, config.get(key)) for key in GUARDRAILS}
    guardrail_failures = [
        key
        for key, expected in GUARDRAILS.items()
        if guardrails.get(key) != expected
    ]
    warnings = _warnings(ml)
    blocking_issues = _blocking_issues(
        source_exists=source_exists,
        missing_columns=missing_columns,
        guardrail_failures=guardrail_failures,
    )
    safe_to_run = (
        source_exists
        and not missing_columns
        and not guardrail_failures
        and not blocking_issues
        and not warnings
    )
    return {
        "config_path": str(config.get("config_path", "")),
        "experiment_stage": str(ml.get("stock_alpha_portfolio_sweep_experiment_stage", ml.get("stock_alpha_ensemble_experiment_stage", ""))),
        "source_predictions_path": str(source) if source is not None else "",
        "source_file_exists": source_exists,
        "source_exists": source_exists,
        "source_predictions_exists": source_exists,
        "required_columns": required_columns,
        "required_columns_found": found_columns,
        "required_columns_missing": missing_columns,
        "required_columns_not_checked_reason": columns_not_checked_reason,
        "estimated_policy_count": _estimated_policy_count(ml),
        "n_jobs": int(ml.get("stock_alpha_portfolio_sweep_n_jobs", 1) or 1),
        "output_root": str(ml.get("stock_alpha_report_root", "")),
        "guardrails": guardrails,
        "guardrail_failures": guardrail_failures,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "safe_to_run": safe_to_run,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "promotion_thresholds_changed": False,
    }


def _blocking_issues(
    *,
    source_exists: bool,
    missing_columns: list[str],
    guardrail_failures: list[str],
) -> list[str]:
    issues: list[str] = []
    if not source_exists:
        issues.append("configured source predictions file does not exist")
    if missing_columns:
        issues.append("configured source predictions file is missing required columns")
    if guardrail_failures:
        issues.append("research guardrails are not satisfied")
    return issues


def _source_predictions_path(ml: Mapping[str, Any]) -> Path | None:
    for key in (
        "stock_alpha_portfolio_sweep_source_predictions_path",
        "stock_alpha_ensemble_source_predictions_path",
        "stock_level_prediction_artifacts_path",
    ):
        value = ml.get(key)
        if value:
            return Path(str(value))
    return None


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [str(value) for value in next(reader)]
        except StopIteration:
            return []


def _required_columns(ml: Mapping[str, Any]) -> list[str]:
    if "stock_alpha_portfolio_sweep_source_predictions_path" in ml:
        target = str(ml.get("stock_alpha_portfolio_sweep_target_column", TARGET_COLUMN))
        return _dedupe(
            [
                "rebalance_date",
                "symbol",
                target,
                *[str(value) for value in ml.get("stock_alpha_portfolio_sweep_signal_columns", [])],
            ]
        )
    if "stock_alpha_ensemble_source_predictions_path" in ml:
        target = str(ml.get("stock_alpha_ensemble_target_column", TARGET_COLUMN))
        return _dedupe(
            [
                "rebalance_date",
                "symbol",
                target,
                *[str(value) for value in ml.get("stock_alpha_ensemble_component_signal_columns", [])],
            ]
        )
    return []


def _estimated_policy_count(ml: Mapping[str, Any]) -> int | None:
    if "stock_alpha_portfolio_sweep_source_predictions_path" not in ml:
        return None
    return (
        len(ml.get("stock_alpha_portfolio_sweep_signal_columns", []))
        * len(ml.get("stock_alpha_portfolio_sweep_top_n_values", [5, 10, 20, 30]))
        * len(ml.get("stock_alpha_portfolio_sweep_max_position_weights", [0.05, 0.075, 0.10]))
        * len(ml.get("stock_alpha_portfolio_sweep_cash_buffers", [0.0, 0.05, 0.10]))
        * len(ml.get("stock_alpha_portfolio_sweep_minimum_signal_thresholds", [None]))
        * len(ml.get("stock_alpha_portfolio_sweep_turnover_caps", [None, 0.25, 0.50]))
    )


def _warnings(ml: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if str(ml.get("stock_alpha_portfolio_sweep_turnover_mode", "strict_top_n")) != "strict_top_n":
        warnings.append("turnover_mode is not strict_top_n")
    if bool(ml.get("stock_alpha_portfolio_sweep_write_all_holdings", False)):
        warnings.append("all-policy holdings output is enabled")
    if bool(ml.get("stock_alpha_portfolio_sweep_write_all_trades", False)):
        warnings.append("all-policy trades output is enabled")
    if bool(ml.get("stock_alpha_news_enable_transformer", False)):
        warnings.append("news transformer is enabled; preflight requires a validated point-in-time news contract")
    return warnings


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _markdown(payload: Mapping[str, Any]) -> str:
    missing = payload.get("required_columns_missing", [])
    source_file_exists = bool(payload.get("source_file_exists", payload.get("source_predictions_exists", False)))
    blocking_issues = payload.get("blocking_issues", [])
    warnings = payload.get("warnings", [])
    guardrail_failures = payload.get("guardrail_failures", [])
    lines = [
        "# Stock-Alpha Experiment Preflight",
        "",
        f"- Config path: {payload.get('config_path', '')}",
        f"- Experiment stage: {payload.get('experiment_stage', '')}",
        f"- Source predictions: {payload.get('source_predictions_path', '')}",
        f"- Source exists: {source_file_exists}",
        f"- Estimated policy count: {payload.get('estimated_policy_count')}",
        f"- n_jobs: {payload.get('n_jobs')}",
        f"- Output root: {payload.get('output_root')}",
        f"- Safe to run: {payload.get('safe_to_run')}",
        "",
        "## Required Columns",
        "",
        f"- Found: {', '.join(payload.get('required_columns_found', [])) or 'None'}",
        f"- Missing: {', '.join(missing) or 'None'}",
        "",
        "## Guardrails",
        "",
    ]
    guardrails = dict(payload.get("guardrails", {}) or {})
    for key in GUARDRAILS:
        lines.append(f"- {key}: {guardrails.get(key)}")
    lines.extend(
        [
            f"- Guardrail failures: {', '.join(guardrail_failures) or 'None'}",
            "",
            "## Blocking Issues",
            "",
            f"- {', '.join(blocking_issues) or 'None'}",
            "",
            "## Warnings",
            "",
            f"- {', '.join(warnings) or 'None'}",
            "",
        ]
    )
    if not source_file_exists:
        lines.extend(
            [
                "## Source File",
                "",
                "The configured source predictions file does not exist yet.",
                "",
            ]
        )
    return "\n".join(lines)
