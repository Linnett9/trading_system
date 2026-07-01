from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.ranking import CrossSectionalRankingEvaluator, finite_number, ranks
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level_benchmark_reporting import _leaderboard_columns
from core.research.ml.stock_level_benchmark_types import TARGET_COLUMN


ENSEMBLE_SIGNAL_COLUMN = "stock_level_ensemble_average_rank_score"
MEDIAN_RANK_SIGNAL_COLUMN = "stock_level_ensemble_median_rank_score"
TRIMMED_MEAN_RANK_SIGNAL_COLUMN = "stock_level_ensemble_trimmed_mean_rank_score"
CONSENSUS_SIGNAL_COLUMN = "stock_level_ensemble_consensus_score"
DISAGREEMENT_SIGNAL_COLUMN = "stock_level_ensemble_disagreement_score"
CONFIDENCE_SIGNAL_COLUMN = "stock_level_ensemble_confidence_score"
COMPONENT_COUNT_COLUMN = "stock_level_ensemble_component_count"
COMPONENT_COVERAGE_COLUMN = "stock_level_ensemble_component_coverage_ratio"
DISAGREEMENT_RAW_COLUMN = "stock_level_ensemble_disagreement_raw"
DISAGREEMENT_NORMALIZED_COLUMN = "stock_level_ensemble_disagreement_normalized"
ENSEMBLE_METHOD_COLUMNS = {
    "average_rank": ("average_rank_ensemble", ENSEMBLE_SIGNAL_COLUMN),
    "median_rank": ("median_rank_ensemble", MEDIAN_RANK_SIGNAL_COLUMN),
    "trimmed_mean_rank": ("trimmed_mean_rank_ensemble", TRIMMED_MEAN_RANK_SIGNAL_COLUMN),
    "consensus": ("consensus_ensemble", CONSENSUS_SIGNAL_COLUMN),
    "confidence": ("confidence_ensemble", CONFIDENCE_SIGNAL_COLUMN),
}
GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}


@dataclass(frozen=True)
class StockAlphaEnsemblePaths:
    predictions_path: Path
    json_path: Path
    leaderboard_csv_path: Path
    markdown_path: Path


def write_stock_alpha_ensemble(config: Mapping[str, Any]) -> StockAlphaEnsemblePaths:
    ml = dict(config.get("ml", {}) or {})
    method = str(ml.get("stock_alpha_ensemble_method", "average_rank"))
    if method != "average_rank":
        raise NotImplementedError("Only ml.stock_alpha_ensemble_method=average_rank is implemented; weighted_rank, regime_weighted, and stacked_meta_model are future research stubs.")
    source = Path(str(ml["stock_alpha_ensemble_source_predictions_path"]))
    components = [str(column) for column in ml.get("stock_alpha_ensemble_component_signal_columns", [])]
    if not components:
        raise ValueError("ml.stock_alpha_ensemble_component_signal_columns must list at least one signal column")
    min_count = int(ml.get("stock_alpha_ensemble_min_component_count", 2))
    methods = [str(method) for method in ml.get("stock_alpha_ensemble_methods", ["average_rank"])]
    unknown_methods = [method for method in methods if method not in ENSEMBLE_METHOD_COLUMNS]
    if unknown_methods:
        raise ValueError(f"Unsupported stock-alpha ensemble methods: {', '.join(unknown_methods)}")
    trim_extremes = int(ml.get("stock_alpha_ensemble_trim_extremes", 1))
    trim_min_components = int(ml.get("stock_alpha_ensemble_trim_min_components", 4))
    trim_fallback = str(ml.get("stock_alpha_ensemble_trim_fallback", "average"))
    if trim_fallback not in {"average", "missing"}:
        raise ValueError("ml.stock_alpha_ensemble_trim_fallback must be average or missing")
    method_name = str(ml.get("stock_alpha_ensemble_rank_normalization_method", "percentile"))
    if method_name not in {"percentile", "rank_zscore"}:
        raise ValueError("ml.stock_alpha_ensemble_rank_normalization_method must be percentile or rank_zscore")
    rows = _read_source_predictions(source, components=components, min_component_count=min_count)
    output_dir = stock_alpha_output_dir(config) / "ensemble" / "average_rank"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_rows, availability = build_average_rank_ensemble(
        rows,
        component_columns=components,
        min_component_count=min_count,
        rank_normalization_method=method_name,
        trim_extremes=trim_extremes,
        trim_min_components=trim_min_components,
        trim_fallback=trim_fallback,
    )
    found = [column for column in components if any(finite_number(row.get(column)) is not None for row in rows)]
    missing = [column for column in components if column not in found]
    _validate_component_availability(
        source,
        requested=components,
        found=found,
        missing=missing,
        min_component_count=min_count,
    )
    eligible_rows_by_method = {
        method: sum(
            1
            for row in ensemble_rows
            if finite_number(row.get(ENSEMBLE_METHOD_COLUMNS[method][1])) is not None
        )
        for method in methods
    }
    if not any(count > 0 for count in eligible_rows_by_method.values()):
        raise _validation_error(
            source,
            requested=components,
            found=found,
            missing=missing,
            min_component_count=min_count,
            reason="no source rows satisfied min_component_count",
        )
    evaluator = CrossSectionalRankingEvaluator(target_column=TARGET_COLUMN)
    leaderboard = []
    for rank, method in enumerate(methods, start=1):
        name, signal_column = ENSEMBLE_METHOD_COLUMNS[method]
        evaluation = evaluator.evaluate(
            ensemble_rows,
            name=name,
            signal_column=signal_column,
            kind="ensemble",
        )
        evaluation["rank"] = rank
        leaderboard.append(evaluation)
    if any(int(row.get("date_count") or 0) <= 0 or int(row.get("row_count") or 0) <= 0 for row in leaderboard):
        raise _validation_error(
            source,
            requested=components,
            found=found,
            missing=missing,
            min_component_count=min_count,
            reason="ensemble evaluation produced zero date_count or row_count",
        )
    payload = {
        "mode": "stock_alpha_ensemble",
        "source_predictions_path": str(source),
        "ensemble_methods_enabled": methods,
        "component_signal_columns_requested": components,
        "component_signal_columns_found": found,
        "component_signal_columns_missing": missing,
        "min_component_count": min_count,
        "rank_normalization_method": method_name,
        "trim_extremes": trim_extremes,
        "trim_min_components": trim_min_components,
        "trim_fallback": trim_fallback,
        "per_date_component_availability": availability,
        "eligible_rows_by_method": eligible_rows_by_method,
        "ensemble_signal_columns": {method: ENSEMBLE_METHOD_COLUMNS[method][1] for method in methods},
        "leaderboard": leaderboard,
        **GUARDRAILS,
    }
    paths = StockAlphaEnsemblePaths(
        predictions_path=output_dir / "stock_alpha_ensemble_average_rank_predictions.csv",
        json_path=output_dir / "stock_alpha_ensemble_average_rank.json",
        leaderboard_csv_path=output_dir / "stock_alpha_ensemble_average_rank_leaderboard.csv",
        markdown_path=output_dir / "stock_alpha_ensemble_average_rank.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.predictions_path, ensemble_rows, fieldnames=_prediction_fieldnames(ensemble_rows))
    writer.write_csv(paths.leaderboard_csv_path, leaderboard, fieldnames=_leaderboard_columns())
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def _read_source_predictions(
    source: Path,
    *,
    components: list[str],
    min_component_count: int,
) -> list[dict[str, Any]]:
    if not source.exists():
        raise _validation_error(
            source,
            requested=components,
            found=[],
            missing=components,
            min_component_count=min_component_count,
            reason="source predictions path does not exist",
        )
    rows = CsvRowRepository().read(source)
    if not rows:
        raise _validation_error(
            source,
            requested=components,
            found=[],
            missing=components,
            min_component_count=min_component_count,
            reason="source predictions CSV has no rows",
        )
    return rows


def _validate_component_availability(
    source: Path,
    *,
    requested: list[str],
    found: list[str],
    missing: list[str],
    min_component_count: int,
) -> None:
    if len(found) < min_component_count:
        raise _validation_error(
            source,
            requested=requested,
            found=found,
            missing=missing,
            min_component_count=min_component_count,
            reason="fewer than min_component_count component columns were found globally",
        )


def _validation_error(
    source: Path,
    *,
    requested: list[str],
    found: list[str],
    missing: list[str],
    min_component_count: int,
    reason: str,
) -> ValueError:
    return ValueError(
        "Invalid stock-alpha average-rank ensemble input: "
        f"{reason}; "
        f"source_predictions_path={source}; "
        f"requested_component_columns={requested}; "
        f"found_component_columns={found}; "
        f"missing_component_columns={missing}; "
        f"min_component_count={min_component_count}"
    )


def build_average_rank_ensemble(
    rows: list[dict[str, Any]],
    *,
    component_columns: list[str],
    min_component_count: int,
    rank_normalization_method: str,
    trim_extremes: int = 1,
    trim_min_components: int = 4,
    trim_fallback: str = "average",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_like = [column for column in component_columns if column.startswith("actual_")]
    if target_like:
        raise ValueError(f"Target columns cannot be ensemble components: {', '.join(target_like)}")
    output = [dict(row) for row in rows]
    by_date_rows: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        by_date_rows.setdefault(str(row.get("rebalance_date", "")), []).append(row)
    availability = []
    for date, date_rows in sorted(by_date_rows.items()):
        scores_by_key: dict[tuple[str, str], list[float]] = {
            (str(row.get("rebalance_date", "")), str(row.get("symbol", ""))): []
            for row in date_rows
        }
        component_found = 0
        for column in component_columns:
            values = [(row, finite_number(row.get(column))) for row in date_rows]
            finite = [(row, value) for row, value in values if value is not None]
            if not finite:
                continue
            component_found += 1
            normalized = _normalize_component([value for _, value in finite], rank_normalization_method)
            for (row, _), score in zip(finite, normalized):
                key = (str(row.get("rebalance_date", "")), str(row.get("symbol", "")))
                scores_by_key[key].append(score)
        for row in date_rows:
            key = (str(row.get("rebalance_date", "")), str(row.get("symbol", "")))
            scores = scores_by_key[key]
            component_count = len(scores)
            coverage = component_count / len(component_columns) if component_columns else 0.0
            row[COMPONENT_COUNT_COLUMN] = component_count
            row[COMPONENT_COVERAGE_COLUMN] = coverage
            if component_count >= min_component_count:
                average = sum(scores) / component_count
                median = _median(scores)
                disagreement = _sample_std(scores)
                row[ENSEMBLE_SIGNAL_COLUMN] = average
                row[MEDIAN_RANK_SIGNAL_COLUMN] = median
                row[TRIMMED_MEAN_RANK_SIGNAL_COLUMN] = _trimmed_mean(
                    scores,
                    trim_extremes=trim_extremes,
                    trim_min_components=trim_min_components,
                    fallback=trim_fallback,
                )
                row[DISAGREEMENT_RAW_COLUMN] = disagreement
            else:
                row[ENSEMBLE_SIGNAL_COLUMN] = ""
                row[MEDIAN_RANK_SIGNAL_COLUMN] = ""
                row[TRIMMED_MEAN_RANK_SIGNAL_COLUMN] = ""
                row[DISAGREEMENT_RAW_COLUMN] = ""
        availability.append({"rebalance_date": date, "component_columns_found": component_found, "row_count": len(date_rows)})
        finite_disagreements = [finite_number(row.get(DISAGREEMENT_RAW_COLUMN)) for row in date_rows]
        finite_disagreements = [value for value in finite_disagreements if value is not None]
        max_disagreement = max(finite_disagreements, default=0.0)
        for row in date_rows:
            average = finite_number(row.get(ENSEMBLE_SIGNAL_COLUMN))
            raw_disagreement = finite_number(row.get(DISAGREEMENT_RAW_COLUMN))
            coverage = finite_number(row.get(COMPONENT_COVERAGE_COLUMN)) or 0.0
            if average is None or raw_disagreement is None:
                row[DISAGREEMENT_NORMALIZED_COLUMN] = ""
                row[DISAGREEMENT_SIGNAL_COLUMN] = ""
                row[CONSENSUS_SIGNAL_COLUMN] = ""
                row[CONFIDENCE_SIGNAL_COLUMN] = ""
                continue
            normalized_disagreement = raw_disagreement / max_disagreement if max_disagreement > 0.0 else 0.0
            agreement = max(0.0, 1.0 - normalized_disagreement)
            row[DISAGREEMENT_NORMALIZED_COLUMN] = normalized_disagreement
            row[DISAGREEMENT_SIGNAL_COLUMN] = normalized_disagreement
            row[CONSENSUS_SIGNAL_COLUMN] = average * coverage * agreement
            row[CONFIDENCE_SIGNAL_COLUMN] = coverage * agreement
    return output, availability


def _normalize_component(values: list[float], method: str) -> list[float]:
    ranked = ranks(values)
    if method == "percentile":
        denominator = max(len(values) - 1, 1)
        return [(rank - 1.0) / denominator for rank in ranked]
    mean = sum(ranked) / len(ranked)
    variance = sum((rank - mean) ** 2 for rank in ranked) / max(len(ranked) - 1, 1)
    std = math.sqrt(variance) if variance > 0.0 else 1.0
    return [(rank - mean) / std for rank in ranked]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = sum(values) / len(values)
    variance = sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _trimmed_mean(
    values: list[float],
    *,
    trim_extremes: int,
    trim_min_components: int,
    fallback: str,
) -> float | str:
    if len(values) < trim_min_components or trim_extremes <= 0:
        return sum(values) / len(values) if fallback == "average" else ""
    ordered = sorted(values)
    trimmed = ordered[trim_extremes : len(ordered) - trim_extremes]
    if not trimmed:
        return sum(values) / len(values) if fallback == "average" else ""
    return sum(trimmed) / len(trimmed)


def _prediction_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered = [
        "rebalance_date",
        "symbol",
        ENSEMBLE_SIGNAL_COLUMN,
        MEDIAN_RANK_SIGNAL_COLUMN,
        TRIMMED_MEAN_RANK_SIGNAL_COLUMN,
        CONSENSUS_SIGNAL_COLUMN,
        DISAGREEMENT_SIGNAL_COLUMN,
        CONFIDENCE_SIGNAL_COLUMN,
        COMPONENT_COUNT_COLUMN,
        COMPONENT_COVERAGE_COLUMN,
        DISAGREEMENT_RAW_COLUMN,
        DISAGREEMENT_NORMALIZED_COLUMN,
    ]
    extras = [column for row in rows for column in row if column not in ordered]
    return [*ordered, *dict.fromkeys(extras)]


def _markdown(payload: Mapping[str, Any]) -> str:
    leaderboard_lines = [
        f"- {row['name']}: mean_spearman_ic={row['mean_spearman_ic']}, top_minus_bottom_spread={row['top_minus_bottom_spread']}, spread_sharpe={row['spread_sharpe']}"
        for row in payload["leaderboard"]
    ]
    return "\n".join(
        [
            "# Stock Alpha Rank Ensembles",
            "",
            "Research only. Trading impact: none. Production validated: false.",
            "",
            f"- Source predictions: `{payload['source_predictions_path']}`",
            f"- Methods enabled: {', '.join(payload['ensemble_methods_enabled'])}",
            f"- Components found: {len(payload['component_signal_columns_found'])}",
            f"- Components missing: {len(payload['component_signal_columns_missing'])}",
            f"- Min component count: {payload['min_component_count']}",
            f"- Trim extremes: {payload['trim_extremes']}",
            f"- Trim min components: {payload['trim_min_components']}",
            "",
            "## Leaderboard",
            "",
            *leaderboard_lines,
            "- Promotion thresholds changed: false",
            "",
        ]
    )
