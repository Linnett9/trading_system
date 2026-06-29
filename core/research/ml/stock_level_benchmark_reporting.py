from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level_benchmark_types import (
    AUXILIARY_TARGET_COLUMNS,
    TARGET_OUTPUT_COLUMNS,
    NOTICE,
    PREDICTION_PREFIX,
    TARGET_COLUMN,
)


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)


def _leaderboard_columns() -> list[str]:
    return [
        "rank",
        "name",
        "kind",
        "signal_column",
        "mean_pearson_ic",
        "mean_spearman_ic",
        "top_decile_return",
        "bottom_decile_return",
        "top_minus_bottom_spread",
        "top_decile_hit_rate",
        "risk_adjusted_spread",
        "spread_sharpe",
        "date_count",
        "row_count",
    ]


def _prediction_columns(model_names: list[str]) -> list[str]:
    return [
        "rebalance_date",
        "symbol",
        "fold_id",
        TARGET_COLUMN,
        *AUXILIARY_TARGET_COLUMNS,
        *TARGET_OUTPUT_COLUMNS,
        "predicted_momentum_120d",
        "predicted_risk_adjusted_momentum",
        *(f"{PREDICTION_PREFIX}{name}" for name in model_names),
    ]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)


def _markdown(payload: dict[str, Any]) -> str:
    comparison = payload["best_ml_vs_momentum_120d"]
    lines = [
        "# Stock-Level Alpha Benchmark Suite",
        "",
        NOTICE,
        "",
        f"- Target: `{payload['target_column']}`",
        f"- Run size: `{payload.get('run_size', 'benchmark')}`",
        f"- Eligible input rows: {payload['eligible_row_count']}",
        f"- OOS rows: {payload['oos_row_count']}",
        f"- OOS dates: {payload['oos_date_count']}",
        f"- Completed models: {len(payload['completed_models'])}",
        f"- Unavailable models: {len(payload['unavailable_models'])}",
        "- Split: chronological expanding window with "
        f"{payload['walk_forward']['embargo_rebalance_dates']} embargoed rebalance dates",
        "- Promotion thresholds changed: false",
        "",
        "## OOS Leaderboard",
        "",
        "| Rank | Model / baseline | Kind | Dates | Pearson IC | Spearman IC | Top decile | Bottom decile | Spread | Sharpe | Top hit rate | Risk-adjusted spread |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            "| {rank} | {name} | {kind} | {dates} | {pearson} | {ic} | {top} | {bottom} | {spread} | {sharpe} | {hit} | {risk} |".format(
                rank=row["rank"],
                name=row["name"],
                kind=row["kind"],
                dates=row["date_count"],
                pearson=_fmt(row["mean_pearson_ic"]),
                ic=_fmt(row["mean_spearman_ic"]),
                top=_fmt(row["top_decile_return"]),
                bottom=_fmt(row["bottom_decile_return"]),
                spread=_fmt(row["top_minus_bottom_spread"]),
                sharpe=_fmt(row["spread_sharpe"]),
                hit=_fmt(row["top_decile_hit_rate"]),
                risk=_fmt(row["risk_adjusted_spread"]),
            )
        )
    lines.extend(
        [
            "",
            "## Best ML vs Momentum 120d",
            "",
            f"- Best ML model: {comparison['model']}",
            f"- Beats OOS-aligned momentum_120d: {comparison['beats_momentum_120d']}",
            "- Decision rule: higher mean Spearman IC and higher top-minus-bottom spread.",
            "- Spearman IC delta: "
            f"{_fmt(comparison['metric_deltas_ml_minus_momentum']['mean_spearman_ic'])}",
            "- Spread delta: "
            f"{_fmt(comparison['metric_deltas_ml_minus_momentum']['top_minus_bottom_spread'])}",
            "",
            "## Full-Period Baseline Reference",
            "",
            "| Baseline | Dates | Spearman IC | Spread | Top hit rate | Risk-adjusted spread |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["full_period_baselines"]:
        lines.append(
            "| {name} | {dates} | {ic} | {spread} | {hit} | {risk} |".format(
                name=row["name"],
                dates=row["date_count"],
                ic=_fmt(row["mean_spearman_ic"]),
                spread=_fmt(row["top_minus_bottom_spread"]),
                hit=_fmt(row["top_decile_hit_rate"]),
                risk=_fmt(row["risk_adjusted_spread"]),
            )
        )
    if payload["unavailable_models"]:
        lines.extend(["", "## Unavailable Models", ""])
        for row in payload["unavailable_models"]:
            lines.append(f"- {row['name']}: {row['reason']}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
