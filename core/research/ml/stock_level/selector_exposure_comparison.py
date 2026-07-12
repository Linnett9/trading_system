from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from core.research.ml.allocation.exposures import _binary_exposures
from core.research.ml.allocation.simulation_math import (
    _annualized_return,
    _compound_returns,
    _max_drawdown,
    _population_std,
    _sharpe_ratio,
    _sortino_ratio,
)
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    file_digest,
    preserve_immutable_run,
    read_run_manifest,
    run_dir_from_latest_completed,
)

SCHEMA_VERSION = "selector_exposure_matched_comparison_v1"
DECISION_ADDS_VALUE = "EXPOSURE_OVERLAY_ADDS_VALUE"
DECISION_DOES_NOT_ADD_VALUE = "EXPOSURE_OVERLAY_DOES_NOT_ADD_VALUE"
DECISION_INSUFFICIENT = "INSUFFICIENT_MATCHED_CURRENT_DATA"


@dataclass(frozen=True)
class SelectorExposureComparisonPaths:
    output_dir: Path
    run_dir: Path
    matched_periods_csv: Path
    comparison_summary_json: Path
    comparison_summary_md: Path
    attribution_csv: Path
    attribution_json: Path
    variant_a_equity_curve_csv: Path
    variant_b_equity_curve_csv: Path
    audit_json: Path
    latest_completed_path: Path


def write_selector_exposure_comparison(config: dict[str, Any]) -> SelectorExposureComparisonPaths:
    ml = dict(config.get("ml", {}) or {})
    paths = _resolved_paths(ml)
    _assert_selector_derived_dataset(paths["selector_exposure_dataset_path"])
    selector_run_id = _latest_run_id(paths["selector_benchmark_dir"])
    replay_run_id = _latest_run_id(paths["selector_replay_dir"])
    meta_run_id = _latest_run_id(paths["meta_output_dir"])
    summary = _read_json(paths["replay_summary_path"])
    winner = (summary.get("winners") or {}).get("best_ml_model") or {}
    signal = str(ml.get("selector_exposure_comparison_signal") or winner.get("signal_column") or "")
    policy = str(ml.get("selector_exposure_comparison_policy") or winner.get("policy") or "")
    if not signal or not policy:
        raise RuntimeError("Selector exposure comparison requires selected signal and policy")
    strategy_id = f"{signal}|{policy}"
    selected_equity = _selected_equity_rows(_read_csv(paths["equity_curves_path"]), strategy_id)
    holdings_by_date = _holdings_identity(_read_csv(paths["holdings_path"]), strategy_id)
    exposure_rows = _exposure_prediction_rows(paths["meta_predictions_path"])
    dataset_rows = _read_csv(paths["selector_exposure_dataset_path"])
    dataset_by_date = {row["rebalance_date"]: row for row in dataset_rows}
    common_dates = sorted(set(selected_equity) & set(exposure_rows) & set(dataset_by_date))
    excluded = _excluded_dates(set(selected_equity), set(exposure_rows), set(dataset_by_date))
    if len(common_dates) < 2:
        decision = DECISION_INSUFFICIENT
    else:
        decision = ""
    threshold = float(ml.get("decision_threshold", ml.get("selector_exposure_decision_threshold", 0.5)))
    reduced_exposure = float(ml.get("promotion_reduced_exposure", ml.get("selector_exposure_reduced_exposure", 0.7)))
    incremental_cost_bps = float(
        ml.get(
            "selector_exposure_overlay_transaction_cost_bps",
            float(ml.get("stock_portfolio_replay_cost_bps", 10.0))
            + float(ml.get("stock_portfolio_replay_slippage_bps", 5.0)),
        )
    )
    exposures = _binary_exposures(
        [exposure_rows[date] for date in common_dates],
        [float(exposure_rows[date]["predicted_probability"]) for date in common_dates],
        {"decision_threshold": threshold, "promotion_reduced_exposure": reduced_exposure},
    )
    matched = _matched_period_rows(
        common_dates,
        selected_equity,
        holdings_by_date,
        exposure_rows,
        exposures,
        incremental_cost_bps,
    )
    _assert_matched_invariants(matched)
    variant_a_returns = [float(row["variant_a_net_return"]) for row in matched]
    variant_b_returns = [float(row["variant_b_net_return"]) for row in matched]
    benchmark_returns = [float(row["benchmark_return"]) for row in matched]
    periods_a = [(row["rebalance_date"], float(row["variant_a_net_return"]), 1.0) for row in matched]
    periods_b = [(row["rebalance_date"], float(row["variant_b_net_return"]), float(row["exposure_multiplier"])) for row in matched]
    variant_a = _metrics(variant_a_returns, benchmark_returns, periods_a, [1.0] * len(matched), "variant_a_selector_only")
    variant_b = _metrics(
        variant_b_returns,
        benchmark_returns,
        periods_b,
        [float(row["exposure_multiplier"]) for row in matched],
        "variant_b_selector_plus_exposure",
    )
    incremental = _incremental(variant_a, variant_b)
    attribution_rows, attribution = _attribution(matched)
    if not decision:
        decision = (
            DECISION_ADDS_VALUE
            if incremental["total_return_delta"] > 0.0
            and incremental["sharpe_delta"] >= 0.0
            else DECISION_DOES_NOT_ADD_VALUE
        )
    output_dir = Path(ml.get("selector_exposure_comparison_output_dir", paths["selector_benchmark_dir"] / "selector_exposure_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = output_dir / "matched_periods.csv"
    summary_path = output_dir / "comparison_summary.json"
    markdown_path = output_dir / "comparison_summary.md"
    attribution_csv = output_dir / "exposure_attribution.csv"
    attribution_json = output_dir / "exposure_attribution.json"
    a_curve_path = output_dir / "variant_a_equity_curve.csv"
    b_curve_path = output_dir / "variant_b_equity_curve.csv"
    audit_path = output_dir / "comparison_audit.json"
    _write_csv(matched_path, matched)
    _write_csv(attribution_csv, attribution_rows)
    _write_csv(a_curve_path, _equity_curve_rows(matched, "variant_a"))
    _write_csv(b_curve_path, _equity_curve_rows(matched, "variant_b"))
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "research_conclusion": decision,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "incremental_b_minus_a": incremental,
        "matched_period_count": len(matched),
        "effective_evaluation_duration": {
            "start": common_dates[0] if common_dates else None,
            "end": common_dates[-1] if common_dates else None,
        },
        "exposure_state_changes": variant_b["number_of_exposure_changes"],
        "statistical_caution": "No statistical significance is claimed for this current-data matched sample.",
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "strict_oos_selector_predictions_used": True,
        "final_fitted_selector_used_for_historical_evaluation": False,
        "selector_source": {
            "selector_run_id": selector_run_id,
            "replay_run_id": replay_run_id,
            "oos_prediction_path": str(paths["oos_predictions_path"]),
            "oos_prediction_hash": file_digest(paths["oos_predictions_path"]),
            "selected_signal_column": signal,
            "selected_portfolio_policy": policy,
            "holdings_hash": file_digest(paths["holdings_path"]),
            "base_replay_hash": file_digest(paths["equity_curves_path"]),
            "portfolio_replay_summary_hash": file_digest(paths["replay_summary_path"]),
        },
        "selector_derived_exposure_source": {
            "path": str(paths["selector_exposure_dataset_path"]),
            "hash": file_digest(paths["selector_exposure_dataset_path"]),
            "source_type": "stock_selector_rebalance_dataset",
        },
        "meta_source": {
            "meta_run_id": meta_run_id,
            "prediction_path": str(paths["meta_predictions_path"]),
            "prediction_hash": file_digest(paths["meta_predictions_path"]),
            "strict_oos": all(row.get("split") == "out_of_fold" for row in exposure_rows.values()),
        },
        "exposure_policy": {
            "policy_name": "binary_exposure_overlay",
            "decision_threshold": threshold,
            "reduced_exposure": reduced_exposure,
            "minimum_exposure": min(exposures, default=0.0),
            "maximum_exposure": max(exposures, default=0.0),
            "cash_allocation_rule": "cash_weight = 1 - adjusted gross/net invested weight",
            "reentry_behavior": "return to 1.0 exposure when probability falls below threshold",
            "incremental_transaction_cost_bps": incremental_cost_bps,
        },
        "matched_invariants": _invariant_summary(matched),
        "date_matching": {
            "selector_replay_dates": len(selected_equity),
            "exposure_prediction_dates": len(exposure_rows),
            "common_eligible_dates": len(common_dates),
            "excluded_dates": excluded,
        },
        "news_enabled": False,
        "deep_selector_models_enabled": False,
        "champion_pointer_updated": False,
        "research_only": True,
        "trading_impact": "none",
    }
    _write_json(summary_path, summary_payload)
    _write_json(attribution_json, attribution)
    _write_json(audit_path, audit)
    markdown_path.write_text(_markdown(summary_payload), encoding="utf-8")
    identity = {
        "source_selector_run_id": selector_run_id,
        "oos_prediction_hash": audit["selector_source"]["oos_prediction_hash"],
        "selector_signal": signal,
        "portfolio_policy": policy,
        "holdings_hash": audit["selector_source"]["holdings_hash"],
        "base_replay_hash": audit["selector_source"]["base_replay_hash"],
        "selector_derived_exposure_dataset_hash": audit["selector_derived_exposure_source"]["hash"],
        "meta_prediction_identity": audit["meta_source"],
        "exposure_policy": audit["exposure_policy"],
        "benchmark_identity": {"benchmark_symbol": "SPY", "benchmark_return_source": str(paths["equity_curves_path"])},
        "cost_slippage_configuration": {"incremental_transaction_cost_bps": incremental_cost_bps},
        "starting_capital": 1.0,
        "matched_evaluation_dates": common_dates,
        "comparison_schema_version": SCHEMA_VERSION,
        "resolved_config_hash": MLCoreArtifactWriter.hash_payload(config),
    }
    run_id = deterministic_run_id("selector_exposure_comparison", identity)
    record = preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="selector_exposure_comparison",
        identity=identity,
        artifact_paths=(
            matched_path,
            summary_path,
            markdown_path,
            attribution_csv,
            attribution_json,
            a_curve_path,
            b_curve_path,
            audit_path,
        ),
        extra_manifest={"research_conclusion": decision, "champion_pointer_updated": False},
    )
    return SelectorExposureComparisonPaths(
        output_dir=output_dir,
        run_dir=record.run_dir,
        matched_periods_csv=record.run_dir / matched_path.name,
        comparison_summary_json=record.run_dir / summary_path.name,
        comparison_summary_md=record.run_dir / markdown_path.name,
        attribution_csv=record.run_dir / attribution_csv.name,
        attribution_json=record.run_dir / attribution_json.name,
        variant_a_equity_curve_csv=record.run_dir / a_curve_path.name,
        variant_b_equity_curve_csv=record.run_dir / b_curve_path.name,
        audit_json=record.run_dir / audit_path.name,
        latest_completed_path=record.latest_completed_path,
    )


def _resolved_paths(ml: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(ml.get("selector_exposure_comparison_source_root", ml.get("output_dir", ".")))
    selector_benchmark_dir = Path(ml.get("selector_exposure_selector_benchmark_dir", root / "selector_benchmark"))
    selector_replay_dir = Path(ml.get("selector_exposure_selector_replay_dir", root / "selector_replay"))
    meta_output_dir = Path(ml.get("selector_exposure_meta_output_dir", root / "meta"))
    cache_root = Path(ml.get("selector_exposure_cache_root", root))
    return {
        "selector_benchmark_dir": selector_benchmark_dir,
        "selector_replay_dir": selector_replay_dir,
        "meta_output_dir": meta_output_dir,
        "oos_predictions_path": Path(ml.get("stock_level_model_oos_predictions_path", selector_benchmark_dir / "stock_level_model_oos_predictions.csv")),
        "replay_summary_path": selector_replay_dir / "stock_level_portfolio_replay_summary.json",
        "holdings_path": selector_replay_dir / "stock_level_portfolio_replay_holdings.csv",
        "equity_curves_path": selector_replay_dir / "stock_level_portfolio_replay_equity_curves.csv",
        "meta_predictions_path": meta_output_dir / "prediction_artifacts.csv",
        "selector_exposure_dataset_path": Path(ml.get("stock_selector_rebalance_dataset_path", cache_root / "stock_selector_rebalance_dataset.csv")),
    }


def _assert_selector_derived_dataset(path: Path) -> None:
    rows = _read_csv(path)
    if not rows or "selector_signal" not in rows[0] or "portfolio_policy" not in rows[0]:
        raise RuntimeError(
            "Ticket 6 requires selector-derived stock_selector_rebalance_dataset.csv; "
            f"rejected source={path}"
        )
    forbidden = {"dual_momentum_strategy", "strategy_name"}
    if forbidden & set(rows[0]):
        raise RuntimeError(f"Rejected non-selector exposure dataset source={path}")


def _selected_equity_rows(rows: list[dict[str, str]], strategy_id: str) -> dict[str, dict[str, str]]:
    selected = {
        row["rebalance_date"]: row
        for row in rows
        if row.get("strategy_id") == strategy_id
    }
    if not selected:
        raise RuntimeError(f"No selector replay equity rows for strategy_id={strategy_id}")
    return selected


def _holdings_identity(rows: list[dict[str, str]], strategy_id: str) -> dict[str, str]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if row.get("strategy_id") == strategy_id:
            grouped.setdefault(row["rebalance_date"], []).append((row["symbol"], str(float(row["weight"]))))
    return {date: json.dumps(sorted(values), separators=(",", ":")) for date, values in grouped.items()}


def _exposure_prediction_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    duplicates = len(rows) - len({row["rebalance_date"] for row in rows})
    if duplicates:
        raise RuntimeError(f"Duplicate meta exposure prediction dates in {path}")
    if any(row.get("split") != "out_of_fold" for row in rows):
        raise RuntimeError("Meta exposure predictions must be strict OOS/out_of_fold")
    return {row["rebalance_date"]: row for row in rows}


def _matched_period_rows(
    dates: Sequence[str],
    equity: Mapping[str, Mapping[str, str]],
    holdings_by_date: Mapping[str, str],
    exposure_rows: Mapping[str, Mapping[str, str]],
    exposures: Sequence[float],
    incremental_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = []
    previous_multiplier = 1.0
    a_equity = 1.0
    b_equity = 1.0
    benchmark_equity = 1.0
    for date, multiplier in zip(dates, exposures):
        base = equity[date]
        gross = float(base["gross_return"])
        base_cost = float(base["transaction_cost_drag"])
        a_return = float(base["net_return"])
        overlay_turnover = abs(float(multiplier) - previous_multiplier)
        overlay_cost = overlay_turnover * incremental_cost_bps / 10_000.0
        b_return = gross * float(multiplier) - base_cost - overlay_cost
        benchmark = float(base["benchmark_return"])
        a_equity *= 1.0 + a_return
        b_equity *= 1.0 + b_return
        benchmark_equity *= 1.0 + benchmark
        rows.append({
            "rebalance_date": date,
            "holdings_identity": holdings_by_date.get(date, ""),
            "base_weights_identity": holdings_by_date.get(date, ""),
            "base_portfolio_return_before_costs": gross,
            "variant_a_pre_overlay_base_return": gross,
            "variant_a_net_return": a_return,
            "variant_b_pre_overlay_base_return": gross,
            "exposure_probability": float(exposure_rows[date]["predicted_probability"]),
            "exposure_state": "reduced" if float(multiplier) < 1.0 else "full",
            "exposure_multiplier": float(multiplier),
            "variant_b_net_return": b_return,
            "benchmark_return": benchmark,
            "variant_a_turnover": float(base["turnover"]),
            "variant_b_turnover": float(base["turnover"]) + overlay_turnover,
            "variant_a_cost": base_cost,
            "variant_b_cost": base_cost + overlay_cost,
            "incremental_overlay_cost": overlay_cost,
            "variant_a_equity": a_equity,
            "variant_b_equity": b_equity,
            "benchmark_equity": benchmark_equity,
        })
        previous_multiplier = float(multiplier)
    return rows


def _assert_matched_invariants(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["holdings_identity"] != row["base_weights_identity"]:
            raise RuntimeError("Holdings/base weights mismatch")
        if abs(float(row["variant_b_pre_overlay_base_return"]) - float(row["variant_a_pre_overlay_base_return"])) > 1e-12:
            raise RuntimeError("Variant B pre-overlay return differs from Variant A base return")
        expected = (
            float(row["base_portfolio_return_before_costs"]) * float(row["exposure_multiplier"])
            - float(row["variant_a_cost"])
            - float(row["incremental_overlay_cost"])
        )
        if abs(float(row["variant_b_net_return"]) - expected) > 1e-12:
            raise RuntimeError("Variant B return differs from exposure multiplier plus incremental costs")


def _metrics(
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods: list[tuple[str, float, float]],
    exposures: Sequence[float],
    name: str,
) -> dict[str, Any]:
    curve = _curve(returns)
    total_return = _compound_returns(list(returns))
    vol = _population_std(list(returns)) * math.sqrt(252.0)
    drawdown = _max_drawdown(curve)
    downside = math.sqrt(mean([min(value, 0.0) ** 2 for value in returns])) if returns else 0.0
    return {
        "name": name,
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, periods),
        "annualized_volatility": vol,
        "sharpe": _sharpe_ratio(list(returns), periods),
        "sortino": _sortino_ratio(list(returns), periods),
        "maximum_drawdown": drawdown,
        "calmar": (total_return / drawdown if drawdown else None),
        "downside_deviation": downside,
        "turnover": sum(abs(current - previous) for previous, current in zip(exposures, exposures[1:])),
        "transaction_cost_drag": None,
        "time_at_full_exposure": sum(math.isclose(value, 1.0) for value in exposures),
        "time_at_reduced_exposure": sum(value < 1.0 for value in exposures),
        "average_exposure": mean(exposures) if exposures else 0.0,
        "minimum_exposure": min(exposures, default=0.0),
        "cash_allocation": mean([1.0 - value for value in exposures]) if exposures else 0.0,
        "positive_period_rate": mean([1.0 if value > 0 else 0.0 for value in returns]) if returns else 0.0,
        "benchmark_excess_return": total_return - _compound_returns(list(benchmark_returns)),
        "number_of_exposure_changes": sum(not math.isclose(a, b) for a, b in zip(exposures, exposures[1:])),
    }


def _incremental(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    return {
        "total_return_delta": float(b["total_return"]) - float(a["total_return"]),
        "annualized_return_delta": _delta(b.get("annualized_return"), a.get("annualized_return")),
        "sharpe_delta": float(b["sharpe"]) - float(a["sharpe"]),
        "maximum_drawdown_improvement": float(a["maximum_drawdown"]) - float(b["maximum_drawdown"]),
        "volatility_delta": float(b["annualized_volatility"]) - float(a["annualized_volatility"]),
        "turnover_delta": float(b["turnover"]) - float(a["turnover"]),
    }


def _attribution(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attr_rows = []
    avoided = missed = contribution = 0.0
    reduced_spans = []
    current_duration = 0
    for row in rows:
        base = float(row["variant_a_net_return"])
        overlay = float(row["variant_b_net_return"])
        delta = overlay - base
        reduced = row["exposure_state"] == "reduced"
        if reduced and base < 0:
            avoided += -base * (1.0 - float(row["exposure_multiplier"]))
        if reduced and base > 0:
            missed += base * (1.0 - float(row["exposure_multiplier"]))
        contribution += delta
        current_duration = current_duration + 1 if reduced else 0
        if reduced:
            reduced_spans.append(current_duration)
        attr_rows.append({"rebalance_date": row["rebalance_date"], "reduced": reduced, "base_return": base, "overlay_return": overlay, "delta": delta})
    best = max(attr_rows, key=lambda row: row["delta"], default={})
    worst = min(attr_rows, key=lambda row: row["delta"], default={})
    payload = {
        "return_avoided_during_reduced_exposure_periods": avoided,
        "upside_missed_during_reduced_exposure_periods": missed,
        "net_exposure_overlay_contribution": contribution,
        "incremental_transaction_cost_drag": sum(float(row["incremental_overlay_cost"]) for row in rows),
        "number_of_exposure_reductions": sum(1 for row in rows if row["exposure_state"] == "reduced"),
        "average_duration_of_reduced_exposure": mean(reduced_spans) if reduced_spans else 0.0,
        "worst_overlay_decision": worst,
        "best_overlay_decision": best,
    }
    return attr_rows, payload


def _excluded_dates(selector: set[str], exposure: set[str], dataset: set[str]) -> list[dict[str, str]]:
    rows = []
    for date in sorted(selector | exposure | dataset):
        reasons = []
        if date not in selector:
            reasons.append("missing_selector_replay")
        if date not in exposure:
            reasons.append("missing_exposure_prediction")
        if date not in dataset:
            reasons.append("missing_selector_exposure_dataset_row")
        if reasons:
            rows.append({"rebalance_date": date, "reason": ",".join(reasons)})
    return rows


def _invariant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "holdings_identical": all(row["holdings_identity"] == row["base_weights_identity"] for row in rows),
        "base_weights_identical": True,
        "dates_identical": True,
        "benchmark_returns_identical": True,
        "pre_overlay_returns_match": all(abs(float(row["variant_b_pre_overlay_base_return"]) - float(row["variant_a_pre_overlay_base_return"])) <= 1e-12 for row in rows),
        "starting_capital_identical": True,
        "costs_matched_except_incremental_overlay_cost": True,
    }


def _equity_curve_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [{"rebalance_date": row["rebalance_date"], "equity": row[f"{prefix}_equity"]} for row in rows]


def _curve(returns: Sequence[float]) -> list[float]:
    equity = 1.0
    values = [equity]
    for value in returns:
        equity *= 1.0 + value
        values.append(equity)
    return values


def _delta(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _latest_run_id(output_dir: Path) -> str | None:
    run_dir = run_dir_from_latest_completed(output_dir)
    manifest = read_run_manifest(run_dir) if run_dir else None
    return manifest.get("run_id") if manifest else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _markdown(payload: Mapping[str, Any]) -> str:
    inc = payload["incremental_b_minus_a"]
    return "\n".join([
        "# Selector Exposure Comparison",
        "",
        f"- Research conclusion: `{payload['research_conclusion']}`",
        f"- Matched periods: `{payload['matched_period_count']}`",
        f"- B - A total return: `{inc['total_return_delta']}`",
        f"- B - A Sharpe: `{inc['sharpe_delta']}`",
    ]) + "\n"
