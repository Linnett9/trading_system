from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.ml.meta.meta_auxiliary_math import _format_metric
from core.research.ml.meta.meta_auxiliary_types import AUXILIARY_TARGETS


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    prediction_columns = [
        name for name in AUXILIARY_TARGETS.values() if any(name in row for row in rows)
    ]
    fieldnames = [
        "feature_id",
        "rebalance_date",
        "variant_id",
        *AUXILIARY_TARGETS.keys(),
        *prediction_columns,
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{name: row.get(name, "") for name in fieldnames},
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            })
def _write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Meta Auxiliary Metrics",
        "",
        "|target|samples|mae|rmse|pearson|spearman|directional_accuracy|",
        "|---|---|---|---|---|---|---|",
    ]
    for target, payload in metrics["targets"].items():
        lines.append(
            "|{target}|{samples}|{mae}|{rmse}|{pearson}|{spearman}|{directional}|".format(
                target=target,
                samples=payload.get("sample_count", 0),
                mae=_format_metric(payload.get("mae")),
                rmse=_format_metric(payload.get("rmse")),
                pearson=_format_metric(payload.get("pearson_correlation")),
                spearman=_format_metric(payload.get("spearman_correlation")),
                directional=_format_metric(payload.get("directional_accuracy")),
            )
        )
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
