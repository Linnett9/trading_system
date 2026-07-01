from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.feature_attribution.math import _difference
from core.research.ml.stock_level.feature_attribution.types import METRIC_NAMES, NOTICE


def _output_dir(config: dict[str, Any]) -> Path:
    return StockLevelResearchConfig.from_mapping(config).output_dir
def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "feature",
        "attribution_method",
        "coefficient_mean",
        "coefficient_abs_mean",
        "normalized_coefficient_or_importance_magnitude",
        "feature_importance_mean",
        "permutation_spearman_ic_drop_mean",
        "permutation_observation_count",
        "fold_count",
        *(f"full_{metric}" for metric in METRIC_NAMES),
        *(f"ablated_{metric}" for metric in METRIC_NAMES),
        *(f"ablation_delta_{metric}" for metric in METRIC_NAMES),
    ]
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)
def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Feature Attribution and Ablation",
        "",
        NOTICE,
        "",
        f"- Models completed: {', '.join(payload['models_completed'])}",
        f"- Eligible rows: {payload['eligible_row_count']}",
        f"- Permutation repeats per fold: {payload['permutation_importance']['repeats_per_fold']}",
        "- Promotion thresholds changed: false",
    ]
    for model in payload["models"]:
        rows = model["feature_rows"]
        attribution_ranked = sorted(
            rows,
            key=lambda row: -float(
                row.get("normalized_coefficient_or_importance_magnitude") or 0.0
            ),
        )
        ablation_ranked = sorted(
            rows,
            key=lambda row: float(
                row.get("ablation_delta_mean_spearman_ic") or 0.0
            ),
        )
        lines.extend(
            [
                "",
                f"## {model['model']}",
                "",
                "### Attribution",
                "",
                "| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in attribution_ranked:
            lines.append(
                "| {feature} | {coef} | {magnitude} | {tree} | {permutation} |".format(
                    feature=row["feature"],
                    coef=_fmt(row.get("coefficient_mean")),
                    magnitude=_fmt(
                        row.get("normalized_coefficient_or_importance_magnitude")
                    ),
                    tree=_fmt(row.get("feature_importance_mean")),
                    permutation=_fmt(
                        row.get("permutation_spearman_ic_drop_mean")
                    ),
                )
            )
        lines.extend(
            [
                "",
                "### Leave-One-Feature-Out Ablation",
                "",
                "| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ablation_ranked:
            lines.append(
                "| {feature} | {ic} | {ic_delta} | {spread} | {spread_delta} | {hit} | {risk} | {sharpe} |".format(
                    feature=row["feature"],
                    ic=_fmt(row.get("ablated_mean_spearman_ic")),
                    ic_delta=_fmt(row.get("ablation_delta_mean_spearman_ic")),
                    spread=_fmt(row.get("ablated_top_minus_bottom_spread")),
                    spread_delta=_fmt(
                        row.get("ablation_delta_top_minus_bottom_spread")
                    ),
                    hit=_fmt(row.get("ablated_top_decile_hit_rate")),
                    risk=_fmt(row.get("ablated_risk_adjusted_spread")),
                    sharpe=_fmt(row.get("ablated_spread_sharpe")),
                )
            )
    if payload["model_errors"]:
        lines.extend(["", "## Model Errors", ""])
        for error in payload["model_errors"]:
            lines.append(f"- {error['model']}: {error['reason']}")
    lines.append("")
    return "\n".join(lines)
def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
