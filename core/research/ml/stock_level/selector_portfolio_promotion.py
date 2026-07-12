from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.ranking import CrossSectionalRankingEvaluator, finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import file_digest
from core.research.ml.stock_level.stock_alpha_ensemble import ENSEMBLE_METHOD_COLUMNS
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_portfolio_replay import (
    GUARDRAILS,
    TARGET,
    _metrics,
    _replay,
)
from core.research.ml.stock_level_benchmark_evaluation import _build_leaderboard
from core.research.ml.stock_level_benchmark_types import BASELINE_COLUMNS, PREDICTION_PREFIX


DEFAULT_PROMOTION = {
    "enabled": True,
    "comparison_mode": "intersection",
    "candidate_types": ["single_model", "baseline", "ensemble"],
    "fixed_policy": {
        "policy": "long_only_top_n_equal_weight",
        "top_n": 25,
        "cost_bps": 10.0,
        "slippage_bps": 5.0,
        "max_position_weight": 0.05,
        "min_position_weight": 0.0,
    },
    "ranking": {
        "primary_metric": "net_sharpe",
        "secondary_metrics": ["net_cagr", "max_drawdown", "annualized_turnover"],
        "deterministic_tiebreak": "candidate_id",
    },
    "gates": {
        "minimum_oos_decision_dates": 250,
        "minimum_prediction_coverage": 0.95,
        "minimum_net_cagr": None,
        "minimum_net_sharpe": None,
        "maximum_drawdown": None,
        "maximum_annualized_turnover": None,
        "maximum_cost_drag": None,
        "minimum_positive_calendar_year_fraction": None,
        "require_outperformance_of_baseline": True,
    },
    "baseline_candidate_id": "baseline:momentum_120d",
    "multiple_testing": {"enabled": False, "method": "report_only"},
    "max_warning_candidate_count": 20,
}

HIGHER_IS_BETTER = {
    "net_cagr",
    "net_sharpe",
    "gross_sharpe",
    "sortino",
    "calmar",
    "information_ratio",
    "net_cumulative_return",
    "annualized_excess_return",
}
LOWER_IS_BETTER = {
    "max_drawdown",
    "annualized_turnover",
    "cost_drag",
    "annualized_volatility",
}


@dataclass(frozen=True)
class SelectorPortfolioPromotionPaths:
    output_dir: Path
    forecast_leaderboard_csv_path: Path
    candidate_metrics_csv_path: Path
    subperiod_metrics_csv_path: Path
    regime_metrics_csv_path: Path
    gate_results_csv_path: Path
    json_path: Path
    markdown_path: Path


def write_selector_portfolio_promotion(config: Mapping[str, Any]) -> SelectorPortfolioPromotionPaths:
    ml = dict(config.get("ml", {}) or {})
    promotion = _promotion_config(ml)
    if not promotion["enabled"]:
        raise ValueError("ml.selector_promotion.enabled is false")
    predictions_path = Path(
        ml.get(
            "selector_promotion_predictions_path",
            ml.get(
                "stock_level_model_oos_predictions_path",
                stock_alpha_output_dir(config) / "stock_level_model_oos_predictions.csv",
            ),
        )
    )
    benchmark_path = Path(
        ml.get(
            "selector_promotion_benchmark_path",
            ml.get(
                "stock_level_model_ranking_benchmark_path",
                stock_alpha_output_dir(config) / "stock_level_model_ranking_benchmark.json",
            ),
        )
    )
    output_dir = Path(
        ml.get("selector_promotion_output_dir", stock_alpha_output_dir(config) / "selector_promotion")
    )
    rows = CsvRowRepository().read(predictions_path)
    benchmark = JsonRepository().read(benchmark_path) if benchmark_path.exists() else {}
    if (benchmark.get("walk_forward") or {}).get("out_of_sample_only") is not True:
        raise ValueError("Selector promotion requires strict OOS benchmark metadata")

    payload = build_selector_portfolio_promotion(
        rows,
        benchmark=benchmark,
        promotion_config=promotion,
        predictions_path=predictions_path,
        benchmark_path=benchmark_path,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SelectorPortfolioPromotionPaths(
        output_dir=output_dir,
        forecast_leaderboard_csv_path=output_dir / "selector_forecast_leaderboard.csv",
        candidate_metrics_csv_path=output_dir / "selector_portfolio_candidate_metrics.csv",
        subperiod_metrics_csv_path=output_dir / "selector_portfolio_subperiod_metrics.csv",
        regime_metrics_csv_path=output_dir / "selector_portfolio_regime_metrics.csv",
        gate_results_csv_path=output_dir / "selector_promotion_gate_results.csv",
        json_path=output_dir / "selector_promotion_report.json",
        markdown_path=output_dir / "selector_promotion_report.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.forecast_leaderboard_csv_path, payload["forecast_leaderboard"], fieldnames=_forecast_fields(payload["forecast_leaderboard"]))
    writer.write_csv(paths.candidate_metrics_csv_path, payload["candidate_metrics"], fieldnames=_fields(payload["candidate_metrics"], ["candidate_id"]))
    writer.write_csv(paths.subperiod_metrics_csv_path, payload["subperiod_metrics"], fieldnames=_fields(payload["subperiod_metrics"], ["candidate_id", "period_type", "period"]))
    writer.write_csv(paths.regime_metrics_csv_path, payload["regime_metrics"], fieldnames=_fields(payload["regime_metrics"], ["candidate_id", "regime_type", "regime"]))
    writer.write_csv(paths.gate_results_csv_path, payload["gate_results"], fieldnames=_fields(payload["gate_results"], ["candidate_id", "overall_status"]))
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_selector_portfolio_promotion(
    rows: list[dict[str, Any]],
    *,
    benchmark: Mapping[str, Any],
    promotion_config: Mapping[str, Any],
    predictions_path: Path | None = None,
    benchmark_path: Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _reject_duplicate_rows(rows)
    candidates = discover_selector_portfolio_candidates(rows, benchmark=benchmark, promotion_config=promotion_config, predictions_path=predictions_path)
    valid = [candidate for candidate in candidates if candidate["eligible_for_evaluation"]]
    if not valid:
        return _blocked_report(rows, candidates, promotion_config, predictions_path, benchmark_path, config, "no_valid_candidates")
    mode = str(promotion_config.get("comparison_mode", "intersection"))
    matched_rows, common = _matched_rows(rows, valid, mode)
    forecast_leaderboard, best_forecast = _forecast_leaderboard(rows, benchmark)
    metrics: list[dict[str, Any]] = []
    subperiods: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    for candidate in valid:
        candidate_rows = matched_rows[candidate["candidate_id"]]
        periods, holdings = _replay(
            candidate_rows,
            candidate["prediction_column"],
            promotion_config["fixed_policy"]["policy"],
            int(promotion_config["fixed_policy"]["top_n"]),
            float(promotion_config["fixed_policy"]["cost_bps"]),
            float(promotion_config["fixed_policy"]["slippage_bps"]),
            float(promotion_config["fixed_policy"]["max_position_weight"]),
            float(promotion_config["fixed_policy"].get("min_position_weight", 0.0)),
        )
        base = _metrics(
            candidate["prediction_column"],
            promotion_config["fixed_policy"]["policy"],
            periods,
            holdings,
        )
        expanded = _portfolio_metrics(candidate, base, periods, holdings, common)
        metrics.append(expanded)
        subperiods.extend(_subperiod_metrics(candidate["candidate_id"], periods))
        regimes.extend(_regime_metrics(candidate["candidate_id"], candidate_rows, periods))
    gate_results = _gate_results(metrics, subperiods, promotion_config)
    ranking = _eligible_ranking(metrics, gate_results, promotion_config)
    recommended = ranking[0]["candidate_id"] if ranking else None
    warnings = _warnings(candidates, valid, promotion_config)
    payload = {
        "mode": "selector_portfolio_promotion_research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_identity": _run_identity(rows, promotion_config, predictions_path, benchmark_path, config),
        "source_prediction_artifact_identity": _artifact_identity(predictions_path),
        "dataset_identity": _dataset_identity(rows),
        "target_identity": {"target_column": TARGET, "target_contract_identity": _first_present(rows, "target_provenance_contract_version")},
        "feature_identities": benchmark.get("feature_columns", []),
        "candidate_inventory": candidates,
        "comparison_mode": mode,
        "common_row_statistics": common,
        "fixed_portfolio_policy": promotion_config["fixed_policy"],
        "fixed_portfolio_policy_identity": _hash(promotion_config["fixed_policy"]),
        "cost_assumptions": {
            "cost_bps": promotion_config["fixed_policy"]["cost_bps"],
            "slippage_bps": promotion_config["fixed_policy"]["slippage_bps"],
        },
        "candidate_metrics": sorted(metrics, key=lambda row: row["candidate_id"]),
        "subperiod_metrics": sorted(subperiods, key=lambda row: (row["candidate_id"], row["period_type"], row["period"])),
        "regime_metrics": sorted(regimes, key=lambda row: (row["candidate_id"], row["regime_type"], row["regime"])),
        "gate_definitions": promotion_config["gates"],
        "gate_results": sorted(gate_results, key=lambda row: row["candidate_id"]),
        "eligible_candidate_ranking": ranking,
        "forecast_leaderboard": forecast_leaderboard,
        "best_forecast_model": best_forecast,
        "best_ml_model_semantics": "Forecast-only OOS leaderboard winner ranked by mean Spearman IC, then top-minus-bottom spread; it does not imply portfolio promotion.",
        "recommended_portfolio_candidate": recommended,
        "promotion_status": "eligible_candidate_selected" if recommended else "no_eligible_candidate",
        "multiple_testing": _multiple_testing(candidates, promotion_config),
        "warnings": warnings,
        "blockers": [],
        "training_performed": False,
        **GUARDRAILS,
    }
    return payload


def discover_selector_portfolio_candidates(
    rows: list[dict[str, Any]],
    *,
    benchmark: Mapping[str, Any],
    promotion_config: Mapping[str, Any],
    predictions_path: Path | None = None,
) -> list[dict[str, Any]]:
    allowed = set(promotion_config.get("candidate_types", ["single_model", "baseline", "ensemble"]))
    columns = list(rows[0]) if rows else []
    model_names = [str(name) for name in benchmark.get("completed_models", [])]
    candidates: list[dict[str, Any]] = []
    for name in model_names:
        column = f"{PREDICTION_PREFIX}{name}"
        if "single_model" in allowed:
            candidates.append(_candidate(rows, name, "single_model", column, predictions_path, component_ids=[name]))
    for name, column in BASELINE_COLUMNS.items():
        if "baseline" in allowed:
            candidates.append(_candidate(rows, name, "baseline", column, predictions_path, component_ids=[name]))
    for method, (name, column) in ENSEMBLE_METHOD_COLUMNS.items():
        if "ensemble" in allowed and column in columns:
            candidates.append(_candidate(rows, name, "ensemble", column, predictions_path, component_ids=[method]))
    for column in columns:
        if column.startswith("stock_level_ensemble_") and column.endswith("_score") and "ensemble" in allowed:
            candidate_id = f"ensemble:{column}"
            if not any(item["candidate_id"] == candidate_id for item in candidates):
                candidates.append(_candidate(rows, column, "ensemble", column, predictions_path, component_ids=[column]))
    return sorted(candidates, key=lambda row: row["candidate_id"])


def _candidate(
    rows: list[dict[str, Any]],
    name: str,
    candidate_type: str,
    column: str,
    predictions_path: Path | None,
    *,
    component_ids: list[str],
) -> dict[str, Any]:
    exists = bool(rows and column in rows[0])
    finite_rows = [row for row in rows if exists and str(row.get("fold_id", "")).strip() and finite_number(row.get(column)) is not None]
    reasons = []
    if not exists:
        reasons.append("prediction_column_missing")
    if exists and not finite_rows:
        reasons.append("no_strict_oos_predictions")
    return {
        "candidate_id": f"{candidate_type}:{name}",
        "candidate_type": candidate_type,
        "model_ids": component_ids if candidate_type == "single_model" else [],
        "component_ids": component_ids,
        "prediction_column": column,
        "prediction_artifact_identity": _artifact_identity(predictions_path),
        "dataset_identity": _dataset_identity(rows),
        "feature_contract_identity": "",
        "target_contract_identity": _first_present(rows, "target_provenance_contract_version"),
        "oos_first_decision_date": min((str(row["rebalance_date"]) for row in finite_rows), default=None),
        "oos_last_decision_date": max((str(row["rebalance_date"]) for row in finite_rows), default=None),
        "eligible_row_count": len([row for row in rows if str(row.get("fold_id", "")).strip()]),
        "prediction_count": len(finite_rows),
        "eligible_for_evaluation": not reasons,
        "ineligible_reasons": reasons,
    }


def _matched_rows(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    eligible = [row for row in rows if str(row.get("fold_id", "")).strip() and finite_number(row.get(TARGET)) is not None]
    keys_by_candidate = {
        candidate["candidate_id"]: {
            (str(row["rebalance_date"]), str(row["symbol"]).upper())
            for row in eligible
            if finite_number(row.get(candidate["prediction_column"])) is not None
        }
        for candidate in candidates
    }
    if mode == "intersection":
        common_keys = set.intersection(*keys_by_candidate.values()) if keys_by_candidate else set()
    elif mode == "native_coverage":
        common_keys = set.union(*keys_by_candidate.values()) if keys_by_candidate else set()
    else:
        raise ValueError("selector_promotion.comparison_mode must be intersection or native_coverage")
    output: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        keys = common_keys if mode == "intersection" else keys_by_candidate[candidate["candidate_id"]]
        output[candidate["candidate_id"]] = [
            row
            for row in eligible
            if (str(row["rebalance_date"]), str(row["symbol"]).upper()) in keys
            and finite_number(row.get(candidate["prediction_column"])) is not None
        ]
    dates = sorted({date for date, _ in common_keys})
    symbols = sorted({symbol for _, symbol in common_keys})
    return output, {
        "common_first_decision_date": dates[0] if dates else None,
        "common_last_decision_date": dates[-1] if dates else None,
        "common_decision_date_count": len(dates),
        "common_symbol_count": len(symbols),
        "common_row_count": len(common_keys),
        "rows_excluded_per_candidate": {
            candidate_id: len(keys_by_candidate[candidate_id] - common_keys) if mode == "intersection" else 0
            for candidate_id in keys_by_candidate
        },
        "dates_excluded_per_candidate": {
            candidate_id: len({date for date, _ in keys_by_candidate[candidate_id] - common_keys}) if mode == "intersection" else 0
            for candidate_id in keys_by_candidate
        },
    }


def _portfolio_metrics(
    candidate: Mapping[str, Any],
    base: Mapping[str, Any],
    periods: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    net = [float(row["net_return"]) for row in periods]
    gross = [float(row["gross_return"]) for row in periods]
    bench = [float(row["benchmark_return"]) for row in periods]
    annualization = _annualization([str(row["rebalance_date"]) for row in periods])
    gross_equity = _equity(gross)
    net_equity = _equity(net)
    bench_equity = _equity(bench)
    drawdown, duration = _drawdown(net_equity)
    by_date = {row["rebalance_date"]: [h for h in holdings if h["rebalance_date"] == row["rebalance_date"]] for row in periods}
    gross_exposures = [sum(abs(float(h["weight"])) for h in rows) for rows in by_date.values()]
    cash = [max(0.0, 1.0 - value) for value in gross_exposures]
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate["candidate_type"],
        "prediction_column": candidate["prediction_column"],
        "policy": base.get("policy"),
        "gross_cumulative_return": gross_equity[-1] - 1.0 if gross_equity else None,
        "net_cumulative_return": net_equity[-1] - 1.0 if net_equity else None,
        "gross_cagr": _cagr(gross_equity, annualization, len(gross)),
        "net_cagr": _cagr(net_equity, annualization, len(net)),
        "annualized_mean_return": mean(net) * annualization if net and annualization else None,
        "benchmark_cumulative_return": bench_equity[-1] - 1.0 if bench_equity else None,
        "benchmark_cagr": _cagr(bench_equity, annualization, len(bench)),
        "annualized_excess_return": mean([a - b for a, b in zip(net, bench)]) * annualization if net and bench and annualization else None,
        "gross_sharpe": _sharpe(gross, annualization),
        "net_sharpe": _sharpe(net, annualization),
        "sortino": _sortino(net, annualization),
        "calmar": base.get("calmar_ratio"),
        "information_ratio": _sharpe([a - b for a, b in zip(net, bench)], annualization),
        "max_drawdown": drawdown,
        "drawdown_duration": duration,
        "annualized_volatility": pstdev(net) * math.sqrt(annualization) if len(net) > 1 and annualization else 0.0,
        "downside_volatility": _downside_vol(net, annualization),
        "worst_daily_return": min(net, default=None),
        "worst_rebalance_period_return": min(net, default=None),
        "expected_shortfall_cvar": _cvar(net),
        "gross_turnover": sum(float(row["turnover"]) for row in periods),
        "annualized_turnover": mean([float(row["turnover"]) for row in periods]) * annualization if periods and annualization else None,
        "average_turnover_per_rebalance": mean([float(row["turnover"]) for row in periods]) if periods else None,
        "transaction_costs": sum(float(row["transaction_cost_drag"]) for row in periods),
        "slippage_costs": None,
        "total_implementation_costs": sum(float(row["transaction_cost_drag"]) for row in periods),
        "cost_drag": sum(float(row["transaction_cost_drag"]) for row in periods),
        "number_of_trades": sum(1 for row in periods if float(row["turnover"]) > 0.0),
        "average_holding_period": None,
        "average_number_of_holdings": base.get("average_number_of_positions"),
        "maximum_single_position_weight": base.get("max_position_weight"),
        "average_sector_concentration": None,
        "maximum_sector_concentration": None,
        "average_cash_weight": mean(cash) if cash else None,
        "average_gross_exposure": mean(gross_exposures) if gross_exposures else None,
        "portfolio_concentration_hhi": _average_hhi(by_date),
        "eligible_decision_dates": common.get("common_decision_date_count"),
        "replayed_decision_dates": len(periods),
        "prediction_coverage": len(periods) / common["common_decision_date_count"] if common.get("common_decision_date_count") else 0.0,
        "missing_prediction_dates": max(0, int(common.get("common_decision_date_count") or 0) - len(periods)),
        "missing_price_events": 0,
        "skipped_trades": 0,
    }


def _subperiod_metrics(candidate_id: str, periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in periods:
        by_year.setdefault(str(row["rebalance_date"])[:4], []).append(row)
    for year, rows in by_year.items():
        output.append(_period_summary(candidate_id, "calendar_year", year, rows))
    for window in (12, 24):
        for start in range(0, max(0, len(periods) - window + 1)):
            rows = periods[start : start + window]
            output.append(_period_summary(candidate_id, f"rolling_{window}_period", f"{rows[0]['rebalance_date']}..{rows[-1]['rebalance_date']}", rows))
    return output


def _regime_metrics(candidate_id: str, source_rows: list[dict[str, Any]], periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regime_columns = [
        column
        for column in (source_rows[0] if source_rows else {})
        if "regime" in column.lower()
    ][:3]
    output: list[dict[str, Any]] = []
    period_by_date = {str(row["rebalance_date"]): row for row in periods}
    for column in regime_columns:
        by_regime: dict[str, list[dict[str, Any]]] = {}
        seen_dates: set[tuple[str, str]] = set()
        for row in source_rows:
            date = str(row["rebalance_date"])
            regime = str(row.get(column, "")).strip()
            key = (date, regime)
            if regime and date in period_by_date and key not in seen_dates:
                by_regime.setdefault(regime, []).append(period_by_date[date])
                seen_dates.add(key)
        for regime, rows in by_regime.items():
            summary = _period_summary(candidate_id, column, regime, rows)
            summary["regime_type"] = column
            summary["regime"] = regime
            output.append(summary)
    return output


def _period_summary(candidate_id: str, period_type: str, period: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["net_return"]) for row in rows]
    bench = [float(row["benchmark_return"]) for row in rows]
    equity = _equity(values)
    drawdown, _ = _drawdown(equity)
    annualization = _annualization([str(row["rebalance_date"]) for row in rows])
    return {
        "candidate_id": candidate_id,
        "period_type": period_type,
        "period": period,
        "net_return": equity[-1] - 1.0 if equity else None,
        "sharpe": _sharpe(values, annualization),
        "max_drawdown": drawdown,
        "turnover": sum(float(row["turnover"]) for row in rows),
        "costs": sum(float(row["transaction_cost_drag"]) for row in rows),
        "benchmark_relative_return": (math.prod(1.0 + value for value in values) - math.prod(1.0 + value for value in bench)) if values and bench else None,
        "decision_date_count": len(rows),
    }


def _gate_results(metrics: list[dict[str, Any]], subperiods: list[dict[str, Any]], promotion_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    gates = promotion_config["gates"]
    baseline = next((row for row in metrics if row["candidate_id"] == promotion_config.get("baseline_candidate_id")), None)
    results = []
    positive_year_fraction = {
        candidate: mean([1.0 if float(row["net_return"] or 0.0) > 0.0 else 0.0 for row in rows])
        for candidate, rows in _group_by(subperiods, "candidate_id", where=lambda row: row["period_type"] == "calendar_year").items()
        if rows
    }
    for row in metrics:
        gate = {
            "candidate_id": row["candidate_id"],
            "coverage_gate": _pass(row["prediction_coverage"], gates.get("minimum_prediction_coverage"), higher=True),
            "history_length_gate": _history_gate(row["replayed_decision_dates"], gates.get("minimum_oos_decision_dates")),
            "net_return_gate": _pass(row["net_cagr"], gates.get("minimum_net_cagr"), higher=True),
            "net_sharpe_gate": _pass(row["net_sharpe"], gates.get("minimum_net_sharpe"), higher=True),
            "drawdown_gate": _pass(abs(float(row["max_drawdown"] or 0.0)), gates.get("maximum_drawdown"), higher=False),
            "turnover_gate": _pass(row["annualized_turnover"], gates.get("maximum_annualized_turnover"), higher=False),
            "cost_gate": _pass(row["cost_drag"], gates.get("maximum_cost_drag"), higher=False),
            "stability_gate": _pass(positive_year_fraction.get(row["candidate_id"]), gates.get("minimum_positive_calendar_year_fraction"), higher=True),
        }
        if gates.get("require_outperformance_of_baseline", True):
            gate["baseline_outperformance_gate"] = "pass" if baseline and row["candidate_id"] == baseline["candidate_id"] else (
                "pass" if baseline and row.get("net_cumulative_return") is not None and row["net_cumulative_return"] > baseline["net_cumulative_return"] else "fail"
            )
        else:
            gate["baseline_outperformance_gate"] = "not_required"
        required = [value for key, value in gate.items() if key.endswith("_gate")]
        if "insufficient_evidence" in required:
            status = "insufficient_evidence"
        elif any(value == "fail" for value in required):
            status = "ineligible"
        else:
            status = "eligible"
        gate["overall_status"] = status
        results.append(gate)
    return results


def _eligible_ranking(metrics: list[dict[str, Any]], gates: list[dict[str, Any]], promotion_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    eligible = {row["candidate_id"] for row in gates if row["overall_status"] == "eligible"}
    ranking = promotion_config["ranking"]
    metric_order = [ranking["primary_metric"], *ranking.get("secondary_metrics", [])]
    rows = [row for row in metrics if row["candidate_id"] in eligible]

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for metric in metric_order:
            value = row.get(metric)
            if value is None:
                values.append(math.inf)
            elif metric in LOWER_IS_BETTER:
                values.append(abs(float(value)) if metric == "max_drawdown" else float(value))
            else:
                values.append(-float(value))
        values.append(row["candidate_id"])
        return tuple(values)

    ranked = sorted(rows, key=key)
    return [{"rank": index, "candidate_id": row["candidate_id"], **{metric: row.get(metric) for metric in metric_order}} for index, row in enumerate(ranked, start=1)]


def _forecast_leaderboard(rows: list[dict[str, Any]], benchmark: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    completed = tuple(str(name) for name in benchmark.get("completed_models", []))
    leaderboard = _build_leaderboard(rows, completed) if completed else []
    for row in leaderboard:
        row["best_forecast_model_semantics"] = "forecast_only"
    best = leaderboard[0] if leaderboard else (benchmark.get("best_ml_model") or None)
    return leaderboard, best


def _promotion_config(ml: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(ml.get("selector_promotion", {}) or {})
    merged = json.loads(json.dumps(DEFAULT_PROMOTION))
    _deep_update(merged, configured)
    return merged


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _reject_duplicate_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper())
        if key in seen:
            raise ValueError(f"Duplicate selector prediction row identity: rebalance_date={key[0]}; symbol={key[1]}")
        seen.add(key)


def _blocked_report(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], promotion_config: Mapping[str, Any], predictions_path: Path | None, benchmark_path: Path | None, config: Mapping[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "mode": "selector_portfolio_promotion_research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_identity": _run_identity(rows, promotion_config, predictions_path, benchmark_path, config),
        "candidate_inventory": candidates,
        "comparison_mode": promotion_config.get("comparison_mode"),
        "common_row_statistics": {},
        "fixed_portfolio_policy": promotion_config.get("fixed_policy"),
        "candidate_metrics": [],
        "subperiod_metrics": [],
        "regime_metrics": [],
        "gate_definitions": promotion_config.get("gates"),
        "gate_results": [],
        "eligible_candidate_ranking": [],
        "forecast_leaderboard": [],
        "best_forecast_model": None,
        "recommended_portfolio_candidate": None,
        "promotion_status": "blocked",
        "warnings": [],
        "blockers": [reason],
        "training_performed": False,
        **GUARDRAILS,
    }


def _pass(value: Any, threshold: Any, *, higher: bool) -> str:
    if threshold is None:
        return "not_required"
    number = finite_number(value)
    if number is None:
        return "insufficient_evidence"
    threshold_number = float(threshold)
    return "pass" if (number >= threshold_number if higher else number <= threshold_number) else "fail"


def _history_gate(value: Any, threshold: Any) -> str:
    if threshold is None:
        return "not_required"
    number = finite_number(value)
    if number is None or number < float(threshold):
        return "insufficient_evidence"
    return "pass"


def _warnings(candidates: list[dict[str, Any]], valid: list[dict[str, Any]], promotion_config: Mapping[str, Any]) -> list[str]:
    warnings = []
    if len(valid) > int(promotion_config.get("max_warning_candidate_count", 20)):
        warnings.append(f"many_candidates_compared={len(valid)}; best observed candidate may be overfit")
    if any(not row["eligible_for_evaluation"] for row in candidates):
        warnings.append("one_or_more_candidates_marked_ineligible_before_evaluation")
    return warnings


def _multiple_testing(candidates: list[dict[str, Any]], promotion_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "number_of_compared_candidates": sum(1 for row in candidates if row["eligible_for_evaluation"]),
        "number_of_policies_used_in_primary_comparison": 1,
        "number_of_targets_represented": 1,
        "number_of_feature_sets_represented": 1,
        "number_of_prior_selection_stages_known": 1,
        **dict(promotion_config.get("multiple_testing", {}) or {}),
    }


def _annualization(dates: list[str]) -> float | None:
    if len(dates) < 2:
        return None
    from datetime import date

    parsed = [date.fromisoformat(value) for value in dates]
    gaps = [(right - left).days for left, right in zip(parsed, parsed[1:]) if right > left]
    return 365.25 / mean(gaps) if gaps else None


def _equity(values: list[float]) -> list[float]:
    equity = [1.0]
    for value in values:
        equity.append(equity[-1] * (1.0 + value))
    return equity


def _cagr(equity: list[float], annualization: float | None, period_count: int) -> float | None:
    return equity[-1] ** (annualization / period_count) - 1.0 if equity and period_count and annualization else None


def _sharpe(values: list[float], annualization: float | None) -> float | None:
    if len(values) < 2 or not annualization:
        return None
    vol = pstdev(values)
    return mean(values) / vol * math.sqrt(annualization) if vol > 0.0 else None


def _sortino(values: list[float], annualization: float | None) -> float | None:
    downside = [min(0.0, value) for value in values]
    vol = math.sqrt(sum(value * value for value in downside) / len(downside)) if downside else 0.0
    return mean(values) / vol * math.sqrt(annualization) if vol > 0.0 and annualization else None


def _downside_vol(values: list[float], annualization: float | None) -> float | None:
    downside = [value for value in values if value < 0.0]
    return pstdev(downside) * math.sqrt(annualization) if len(downside) > 1 and annualization else 0.0


def _drawdown(equity: list[float]) -> tuple[float, int]:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    current_duration = 0
    max_duration = 0
    for value in equity:
        if value >= peak:
            peak = value
            current_duration = 0
        else:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
            worst = min(worst, value / peak - 1.0)
    return worst, max_duration


def _cvar(values: list[float], fraction: float = 0.05) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return mean(ordered[:count])


def _average_hhi(by_date: Mapping[str, list[dict[str, Any]]]) -> float | None:
    values = []
    for rows in by_date.values():
        total = sum(abs(float(row["weight"])) for row in rows)
        if total:
            values.append(sum((abs(float(row["weight"])) / total) ** 2 for row in rows))
    return mean(values) if values else None


def _group_by(rows: list[dict[str, Any]], key: str, *, where=lambda row: True) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if where(row):
            grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _dataset_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "decision_date_count": len({str(row.get("rebalance_date", "")) for row in rows}),
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in rows}),
        "first_decision_date": min((str(row.get("rebalance_date", "")) for row in rows), default=None),
        "last_decision_date": max((str(row.get("rebalance_date", "")) for row in rows), default=None),
        "logical_content_hash": _hash([{k: row.get(k) for k in ("rebalance_date", "symbol", "fold_id", TARGET)} for row in rows]),
    }


def _artifact_identity(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path) if path else None,
        "sha256": file_digest(path) if path and path.exists() else None,
    }


def _run_identity(rows: list[dict[str, Any]], promotion_config: Mapping[str, Any], predictions_path: Path | None, benchmark_path: Path | None, config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "prediction_artifact_path": str(predictions_path) if predictions_path else None,
        "prediction_artifact_sha256": file_digest(predictions_path) if predictions_path and predictions_path.exists() else None,
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "benchmark_sha256": file_digest(benchmark_path) if benchmark_path and benchmark_path.exists() else None,
        "policy_configuration_hash": _hash(promotion_config.get("fixed_policy")),
        "promotion_configuration_hash": _hash(promotion_config),
        "dataset_identity": _dataset_identity(rows),
        "code_commit": MLCoreArtifactWriter.git_commit(),
        "config_hash": _hash(config or {}),
    }


def _first_present(rows: list[dict[str, Any]], column: str) -> str | None:
    return next((str(row.get(column)) for row in rows if str(row.get(column, "")).strip()), None)


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    return [*preferred, *[key for key in dict.fromkeys(key for row in rows for key in row) if key not in preferred]] if rows else preferred


def _forecast_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "rank",
        "name",
        "kind",
        "signal_column",
        "mean_spearman_ic",
        "top_minus_bottom_spread",
        "date_count",
        "row_count",
    ]
    return _fields(rows, preferred)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Portfolio Promotion",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Comparison mode: `{payload.get('comparison_mode')}`",
        f"- Candidates discovered: {len(payload.get('candidate_inventory', []))}",
        f"- Common rows: {payload.get('common_row_statistics', {}).get('common_row_count')}",
        f"- Fixed policy: `{payload.get('fixed_portfolio_policy', {}).get('policy')}`",
        f"- Best forecast model: `{(payload.get('best_forecast_model') or {}).get('name')}`",
        f"- Recommended portfolio candidate: `{payload.get('recommended_portfolio_candidate')}`",
        f"- Promotion status: `{payload.get('promotion_status')}`",
        "",
        "## Gates",
        "",
        "| Candidate | Status | Coverage | History | Net Sharpe | Drawdown | Turnover | Baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload.get("gate_results", []):
        lines.append(
            f"| {row['candidate_id']} | {row['overall_status']} | {row['coverage_gate']} | {row['history_length_gate']} | {row['net_sharpe_gate']} | {row['drawdown_gate']} | {row['turnover_gate']} | {row['baseline_outperformance_gate']} |"
        )
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines) + "\n"
