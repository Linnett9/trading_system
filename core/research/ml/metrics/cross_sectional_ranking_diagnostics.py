from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
DEFAULT_TARGET = "actual_forward_return_10d"


@dataclass(frozen=True)
class CrossSectionalRankingDiagnosticsPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class RankingSignal:
    name: str
    columns: tuple[str, ...]
    higher_is_better: bool = True
    description: str = ""


def write_cross_sectional_ranking_diagnostics(
    config: dict[str, Any],
) -> CrossSectionalRankingDiagnosticsPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "stock_level_prediction_artifacts.csv"
    if not source_path.exists():
        source_path = output_dir / "meta_auxiliary_predictions.csv"
    rows = _read_csv(source_path)
    payload = build_cross_sectional_ranking_diagnostics(
        rows,
        source_path=str(source_path),
    )
    paths = CrossSectionalRankingDiagnosticsPaths(
        csv_path=output_dir / "cross_sectional_ranking_diagnostics.csv",
        json_path=output_dir / "cross_sectional_ranking_diagnostics.json",
        markdown_path=output_dir / "cross_sectional_ranking_diagnostics.md",
    )
    _write_csv(paths.csv_path, payload["signals"])
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_cross_sectional_ranking_diagnostics(
    rows: list[dict[str, str]],
    *,
    target_column: str = DEFAULT_TARGET,
    source_path: str | None = None,
) -> dict[str, Any]:
    signals = _available_signals(rows)
    summaries = [
        _evaluate_signal(rows, signal, target_column=target_column)
        for signal in signals
    ]
    summaries = [row for row in summaries if row["available"]]
    ranked = sorted(
        summaries,
        key=lambda row: (
            -abs(float(row.get("top_minus_bottom_spread") or 0.0)),
            -abs(float(row.get("mean_spearman") or 0.0)),
            -int(row.get("evaluated_date_count") or 0),
        ),
    )
    row_entity = _row_entity_type(rows)
    return {
        "mode": "cross_sectional_ranking_diagnostics_research_only",
        "purpose": (
            "Evaluate whether existing saved forecasts rank rows by future return "
            "within each rebalance date. This is diagnostic only and does not "
            "change promotion gates."
        ),
        "row_entity_type": row_entity,
        "stock_level_available": row_entity == "symbol",
        "stock_level_note": (
            "Current benchmark artifacts have no populated symbol column; "
            "diagnostics rank artifact rows/variants within each rebalance date."
            if row_entity != "symbol"
            else "Diagnostics rank symbols within each rebalance date."
        ),
        "target_column": target_column,
        "source_path": source_path,
        "risk_adjusted_target": (
            "actual_forward_return_10d divided by max(abs(actual_future_drawdown), "
            "actual_future_volatility, 1e-6)"
        ),
        "grouping": "rebalance_date",
        "time_series_leakage_guard": (
            "Signals are evaluated only against same-row future-return targets "
            "after grouping by rebalance_date; actual_* columns are not used as "
            "ranking signals."
        ),
        "signals": summaries,
        "best_signal": ranked[0] if ranked else None,
        "any_signal_ranks_future_returns": bool(
            ranked
            and ranked[0].get("mean_spearman") is not None
            and float(ranked[0].get("top_minus_bottom_spread") or 0.0) > 0.0
        ),
        "future_ranking_model_justified": bool(
            ranked
            and float(ranked[0].get("top_minus_bottom_spread") or 0.0) > 0.0
            and int(ranked[0].get("evaluated_date_count") or 0) >= 20
        ),
        "promotion_thresholds_changed": False,
        **RESEARCH_METADATA,
    }


def _available_signals(rows: list[dict[str, str]]) -> list[RankingSignal]:
    fieldnames = list(rows[0]) if rows else []
    signals = [
        RankingSignal(
            "predicted_probability",
            tuple(
                name
                for name in fieldnames
                if name == "predicted_probability"
                or name.endswith("_predicted_probability")
            ),
            higher_is_better=False,
            description=(
                "Lower should-reduce-exposure probability is treated as better "
                "for return ranking."
            ),
        ),
        RankingSignal(
            "meta_predicted_forward_return_10d",
            ("meta_predicted_forward_return_10d",),
            description="Meta auxiliary 10-day forward return forecast.",
        ),
        RankingSignal(
            "meta_predicted_forward_return_5d",
            ("meta_predicted_forward_return_5d",),
            description="Meta auxiliary 5-day forward return forecast.",
        ),
        RankingSignal(
            "risk_adjusted_forecast_score",
            (
                "meta_predicted_forward_return_10d",
                "meta_predicted_future_drawdown",
                "meta_predicted_future_volatility",
            ),
            description=(
                "Meta 10-day return forecast divided by forecast drawdown or "
                "volatility risk."
            ),
        ),
        RankingSignal(
            "momentum_20d",
            ("predicted_momentum_20d",),
            description="Point-in-time 20-trading-day trailing momentum.",
        ),
        RankingSignal(
            "momentum_60d",
            ("predicted_momentum_60d",),
            description="Point-in-time 60-trading-day trailing momentum.",
        ),
        RankingSignal(
            "momentum_120d",
            ("predicted_momentum_120d",),
            description="Point-in-time 120-trading-day trailing momentum.",
        ),
        RankingSignal(
            "risk_adjusted_momentum",
            ("predicted_risk_adjusted_momentum",),
            description="Point-in-time 60-day momentum divided by trailing risk.",
        ),
        RankingSignal(
            "liquidity_score",
            ("predicted_liquidity_score",),
            description="Point-in-time log trailing dollar-volume liquidity score.",
        ),
    ]
    for name in fieldnames:
        normalized = name.lower()
        if "actual_" in normalized:
            continue
        if (
            normalized.endswith("predicted_trend_score")
            or normalized.endswith("predicted_regime_score")
        ):
            signals.append(RankingSignal(name, (name,), description="Existing momentum score."))
        elif normalized.endswith("predicted_context_risk_multiplier"):
            signals.append(
                RankingSignal(
                    name,
                    (name,),
                    higher_is_better=False,
                    description="Existing context risk multiplier; lower risk is better.",
                )
            )
    return [
        signal
        for signal in signals
        if signal.name == "risk_adjusted_forecast_score"
        or any(column in fieldnames for column in signal.columns)
    ]


def _evaluate_signal(
    rows: list[dict[str, str]],
    signal: RankingSignal,
    *,
    target_column: str,
) -> dict[str, Any]:
    by_date: dict[str, list[dict[str, float]]] = {}
    missing_signal_rows = 0
    missing_target_rows = 0
    for row in rows:
        date = row.get("rebalance_date")
        if not date:
            continue
        target = _number(row.get(target_column))
        risk_target = _risk_adjusted_target(row, target)
        value = _signal_value(row, signal)
        if target is None or risk_target is None:
            missing_target_rows += 1
            continue
        if value is None:
            missing_signal_rows += 1
            continue
        score = value if signal.higher_is_better else -value
        by_date.setdefault(date, []).append(
            {"score": score, "target": target, "risk_target": risk_target}
        )
    date_metrics = [
        _date_metrics(date, group)
        for date, group in sorted(by_date.items())
        if len(group) >= 2
    ]
    available = bool(date_metrics)
    return {
        "signal": signal.name,
        "available": available,
        "description": signal.description,
        "columns": list(signal.columns),
        "higher_is_better": signal.higher_is_better,
        "evaluated_date_count": len(date_metrics),
        "evaluated_row_count": sum(row["row_count"] for row in date_metrics),
        "missing_signal_rows": missing_signal_rows,
        "missing_target_rows": missing_target_rows,
        "mean_spearman": _average([row["spearman"] for row in date_metrics]),
        "top_decile_forward_return": _average(
            [row["top_decile_forward_return"] for row in date_metrics]
        ),
        "bottom_decile_forward_return": _average(
            [row["bottom_decile_forward_return"] for row in date_metrics]
        ),
        "top_minus_bottom_spread": _average(
            [row["top_minus_bottom_spread"] for row in date_metrics]
        ),
        "hit_rate_top_decile": _average(
            [row["hit_rate_top_decile"] for row in date_metrics]
        ),
        "risk_adjusted_top_minus_bottom_spread": _average(
            [row["risk_adjusted_top_minus_bottom_spread"] for row in date_metrics]
        ),
        "date_metrics": date_metrics,
    }


def _signal_value(row: dict[str, str], signal: RankingSignal) -> float | None:
    if signal.name == "risk_adjusted_forecast_score":
        forecast = _number(row.get("meta_predicted_forward_return_10d"))
        drawdown = _number(row.get("meta_predicted_future_drawdown"))
        volatility = _number(row.get("meta_predicted_future_volatility"))
        if forecast is None:
            return None
        risk = max(abs(drawdown or 0.0), abs(volatility or 0.0), 1e-6)
        return forecast / risk
    values = [
        value
        for value in (_number(row.get(column)) for column in signal.columns)
        if value is not None
    ]
    return mean(values) if values else None


def _date_metrics(date: str, rows: list[dict[str, float]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    bucket_size = max(1, math.ceil(len(ordered) * 0.10))
    top = ordered[:bucket_size]
    bottom = ordered[-bucket_size:]
    top_return = mean(row["target"] for row in top)
    bottom_return = mean(row["target"] for row in bottom)
    top_risk = mean(row["risk_target"] for row in top)
    bottom_risk = mean(row["risk_target"] for row in bottom)
    return {
        "rebalance_date": date,
        "row_count": len(ordered),
        "bucket_size": bucket_size,
        "spearman": _spearman(
            [row["score"] for row in ordered],
            [row["target"] for row in ordered],
        ),
        "top_decile_forward_return": top_return,
        "bottom_decile_forward_return": bottom_return,
        "top_minus_bottom_spread": top_return - bottom_return,
        "hit_rate_top_decile": mean(1.0 if row["target"] > 0.0 else 0.0 for row in top),
        "risk_adjusted_top_minus_bottom_spread": top_risk - bottom_risk,
    }


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _risk_adjusted_target(row: dict[str, str], target: float | None) -> float | None:
    if target is None:
        return None
    drawdown = _number(row.get("actual_future_drawdown"))
    volatility = _number(row.get("actual_future_volatility"))
    risk = max(abs(drawdown or 0.0), abs(volatility or 0.0), 1e-6)
    return target / risk


def _row_entity_type(rows: list[dict[str, str]]) -> str:
    if any(row.get("symbol") for row in rows):
        return "symbol"
    if any(row.get("variant_id") for row in rows):
        return "artifact_row_or_variant"
    return "artifact_row"


def _average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "signal",
        "available",
        "evaluated_date_count",
        "evaluated_row_count",
        "mean_spearman",
        "top_decile_forward_return",
        "bottom_decile_forward_return",
        "top_minus_bottom_spread",
        "hit_rate_top_decile",
        "risk_adjusted_top_minus_bottom_spread",
        "missing_signal_rows",
        "missing_target_rows",
    ]
    normalized = [{name: row.get(name) for name in fieldnames} for row in rows]
    ResearchArtifactWriter().write_csv(path, normalized, fieldnames=fieldnames)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Sectional Ranking Diagnostics",
        "",
        NOTICE,
        "",
        f"- Grouping: {payload['grouping']}",
        f"- Target: {payload['target_column']}",
        f"- Row entity type: {payload['row_entity_type']}",
        f"- Stock-level available: {payload['stock_level_available']}",
        f"- Any signal ranks future returns: {payload['any_signal_ranks_future_returns']}",
        f"- Future ranking model justified: {payload['future_ranking_model_justified']}",
        "",
        "## Signals",
        "",
        "| Signal | Dates | Spearman | Top decile | Bottom decile | Spread | Top hit rate | Risk-adjusted spread |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["signals"]:
        lines.append(
            "| {signal} | {dates} | {spearman} | {top} | {bottom} | {spread} | {hit} | {risk} |".format(
                signal=row["signal"],
                dates=row["evaluated_date_count"],
                spearman=_fmt(row.get("mean_spearman")),
                top=_fmt(row.get("top_decile_forward_return")),
                bottom=_fmt(row.get("bottom_decile_forward_return")),
                spread=_fmt(row.get("top_minus_bottom_spread")),
                hit=_fmt(row.get("hit_rate_top_decile")),
                risk=_fmt(row.get("risk_adjusted_top_minus_bottom_spread")),
            )
        )
    best = payload.get("best_signal") or {}
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- Best signal: {best.get('signal')}",
        f"- Best top-minus-bottom spread: {_fmt(best.get('top_minus_bottom_spread'))}",
        f"- Note: {payload['stock_level_note']}",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
