from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.overnight_stock_alpha_types import OvernightStockAlphaPaths, SUMMARY_MODELS


def _path_payload(paths: Any) -> dict[str, str]:
    if paths is None:
        return {}
    return {
        key: str(value)
        for key, value in vars(paths).items()
        if isinstance(value, Path)
    }

def _write_summary(paths: OvernightStockAlphaPaths, payload: dict[str, Any]) -> None:
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))

def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Overnight Stock Alpha Summary",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Run size: `{payload.get('run_size', 'benchmark')}`",
        f"- Effective rows/dates/symbols: {payload.get('effective_row_count')}/{payload.get('effective_date_count')}/{payload.get('effective_symbol_count')}",
        "",
        "## Winners",
    ]
    for key, value in payload["winners"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Comparison"])
    lines.append(
        "| model | spearman_ic | top_minus_bottom_spread | spread_sharpe | risk_adjusted_spread | top_decile_hit_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in SUMMARY_MODELS:
        row = payload["comparisons"].get(name, {})
        lines.append(
            "| {name} | {spearman} | {spread} | {sharpe} | {risk} | {hit} |".format(
                name=name,
                spearman=_fmt(row.get("spearman_ic")),
                spread=_fmt(row.get("top_minus_bottom_spread")),
                sharpe=_fmt(row.get("spread_sharpe")),
                risk=_fmt(row.get("risk_adjusted_spread")),
                hit=_fmt(row.get("top_decile_hit_rate")),
            )
        )
    lines.extend(["", "## Stage Timings"])
    for stage, timing in payload["stage_timings"].items():
        lines.append(f"- {stage}: status={timing.get('status', 'executed')} seconds={_fmt(timing.get('seconds'))} skipped={timing.get('skipped', False)} output={timing.get('output_paths', {})}")
    portfolio = payload.get("portfolio_replay", {})
    lines.extend(["", "## Portfolio Replay"])
    for key in ("best_portfolio_signal", "best_portfolio_policy", "net_return_after_costs", "sharpe", "max_drawdown", "turnover", "cost_drag"):
        lines.append(f"- {key}: {portfolio.get(key)}")
    sweep = payload.get("portfolio_policy_sweep", {})
    lines.extend(["", "## Portfolio Policy Sweep"])
    for key, value in sweep.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Daily Selector Decision Grid"])
    lines.append(
        "| layer | decision | cadence | output | target | holding_days | evaluation |"
    )
    lines.append("|---|---|---|---|---|---:|---|")
    for row in payload.get("daily_selector_decision_grid", []):
        lines.append(
            "| {layer} | {decision} | {cadence} | {output} | {target} | {holding} | {evaluation} |".format(
                layer=row.get("layer", ""),
                decision=row.get("decision", ""),
                cadence=row.get("cadence", ""),
                output=row.get("output", ""),
                target=row.get("target", ""),
                holding=_fmt(row.get("intended_holding_period_trading_days")),
                evaluation=row.get("historical_evaluation_policy", ""),
            )
        )
    gate = payload.get("twelve_worker_regeneration_gate", {})
    lines.extend(["", "## Twelve-Worker Regeneration Gate"])
    lines.append(f"- status: {gate.get('status')}")
    lines.append(f"- scope: {gate.get('scope')}")
    lines.append(f"- selector_training_invoked: {gate.get('selector_training_invoked')}")
    lines.append(f"- news_backfill_invoked: {gate.get('news_backfill_invoked')}")
    lines.append(f"- portfolio_replay_invoked: {gate.get('portfolio_replay_invoked')}")
    for check in gate.get("checks", []):
        lines.append(
            "- {name}: passed={passed} actual={actual} expected={expected}".format(
                name=check.get("name"),
                passed=check.get("passed"),
                actual=check.get("actual"),
                expected=check.get("expected"),
            )
        )
    parallelism = payload.get("parallelism", {})
    lines.extend(["", "## Parallelism"])
    for key in (
        "stock_alpha_feature_n_jobs",
        "stock_ranker_model_n_jobs",
        "sklearn_n_jobs",
        "effective_model_workers",
        "stock_alpha_overnight_stage_n_jobs",
        "effective_stage_workers",
        "stages",
    ):
        lines.append(f"- {key}: {parallelism.get(key)}")
    return "\n".join(lines) + "\n"

def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
