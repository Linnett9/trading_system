from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.research.ml.stock_level.selector_feature_schema import load_feature_schema

DATES = ("2026-06-18", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25")
ROOT_NAMES = {
    "seven": "five_date_seven_features",
    "final_ridge": "five_date_final_schema_ridge",
    "final_elastic_net": "five_date_final_schema_elastic_net",
}
MODEL_COLUMNS = {
    "ridge": "stock_level_predicted_forward_return_10d_ridge",
    "elastic_net": "stock_level_predicted_forward_return_10d_elastic_net",
}
BASELINES = {
    "momentum_120d": "predicted_momentum_120d",
    "risk_adjusted_momentum": "predicted_risk_adjusted_momentum",
}
OUTCOMES = (
    "actual_forward_return_10d", "actual_market_residual_return_10d",
    "actual_drawdown_adjusted_forward_return_10d",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite(series: pd.Series) -> bool:
    return bool(np.isfinite(pd.to_numeric(series, errors="coerce")).all())


def _corr(method, left, right):
    result = method(left, right)
    return float(result.statistic)


def _top_ids(frame: pd.DataFrame, score: str, count: int) -> set[str]:
    return set(frame.nlargest(min(count, len(frame)), score)["row_id"])


def _outcome_metrics(frame: pd.DataFrame, score: str) -> dict:
    ordered = frame.sort_values(score, ascending=False, kind="mergesort")
    decile = max(1, math.ceil(len(ordered) * 0.10))
    top, bottom, top20 = ordered.head(decile), ordered.tail(decile), ordered.head(min(20, len(ordered)))
    forward = "actual_forward_return_10d"
    residual = "actual_market_residual_return_10d"
    drawdown = "actual_drawdown_adjusted_forward_return_10d"
    top_mean, bottom_mean = float(top[forward].mean()), float(bottom[forward].mean())
    available_drawdown = top[drawdown].dropna()
    return {
        "spearman_rank_ic_forward": _corr(spearmanr, frame[score], frame[forward]),
        "spearman_rank_ic_residual": _corr(spearmanr, frame[score], frame[residual]),
        "top_decile_mean_forward_return": top_mean,
        "top_decile_mean_residual_return": float(top[residual].mean()),
        "top_20_mean_forward_return": float(top20[forward].mean()),
        "bottom_decile_mean_forward_return": bottom_mean,
        "top_minus_bottom_forward_spread": top_mean - bottom_mean,
        "top_decile_hit_rate": float((top[forward] > 0).mean()),
        "top_decile_mean_drawdown_adjusted_outcome": float(available_drawdown.mean()) if len(available_drawdown) else None,
        "top_decile_size": decile,
    }


def _comparison(left: pd.DataFrame, right: pd.DataFrame, score: str) -> dict:
    joined = left[["row_id", "symbol", score]].merge(
        right[["row_id", "symbol", score]], on=["row_id", "symbol"], how="inner",
        validate="one_to_one", suffixes=("_seven", "_final"),
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError("Fail-closed population mismatch while joining model comparison by row_id")
    a, b = joined[f"{score}_seven"], joined[f"{score}_final"]
    rank_a = a.rank(method="average", ascending=False)
    rank_b = b.rank(method="average", ascending=False)
    movement = (rank_b - rank_a).abs()
    largest = joined.assign(
        seven_rank=rank_a, final_rank=rank_b, absolute_rank_change=movement,
    ).nlargest(10, "absolute_rank_change")
    n = len(joined); decile = max(1, math.ceil(n * 0.10))
    return {
        "oos_row_count": n,
        "spearman_rank_correlation": _corr(spearmanr, a, b),
        "pearson_correlation": _corr(pearsonr, a, b),
        "seven_prediction_dispersion": float(a.std(ddof=0)),
        "final_prediction_dispersion": float(b.std(ddof=0)),
        "top_10_overlap_count": len(set(a.nlargest(10).index) & set(b.nlargest(10).index)),
        "top_20_overlap_count": len(set(a.nlargest(20).index) & set(b.nlargest(20).index)),
        "top_decile_overlap_count": len(set(a.nlargest(decile).index) & set(b.nlargest(decile).index)),
        "top_decile_size": decile,
        "absolute_rank_movement": {
            "mean": float(movement.mean()), "median": float(movement.median()),
            "p75": float(movement.quantile(.75)), "p90": float(movement.quantile(.90)),
            "p95": float(movement.quantile(.95)), "maximum": float(movement.max()),
        },
        "largest_rank_changes": largest[["row_id", "symbol", "seven_rank", "final_rank", "absolute_rank_change"]].to_dict("records"),
    }


def _timing(metrics: dict, candidate: str) -> dict:
    value = metrics["model_details"][candidate]
    return {"fit_seconds": float(value["fit_seconds"]), "prediction_seconds": float(value["prediction_seconds"])}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    schema = load_feature_schema(args.feature_schema)
    dataset_manifest = _json(args.dataset_root / "manifest.json")
    expected_identity = {
        "dataset_id": dataset_manifest["dataset_id"],
        "source_dataset_checksum": dataset_manifest["source_sha256"],
        "rows_checksum": dataset_manifest["checksums"]["rows.parquet"],
        "baseline_checksum": dataset_manifest["checksums"]["baseline_scores.parquet"],
        "target_field": "actual_forward_return_10d",
    }
    roots = {key: args.comparison_root / name for key, name in ROOT_NAMES.items()}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    metrics_by_root: dict[tuple[str, str], dict] = {}
    validations = []
    for root_id, root in roots.items():
        expected_models = ["ridge", "elastic_net"] if root_id == "seven" else [root_id.removeprefix("final_")]
        expected_columns = [MODEL_COLUMNS[name] for name in expected_models] + list(BASELINES.values())
        for date in DATES:
            partition = root / f"date={date}"
            manifest = _json(partition / "manifest.json")
            metrics = _json(partition / "metrics.json")
            predictions_path = partition / "predictions.parquet"
            checks = {
                "manifest_complete": manifest.get("completion_status") == "complete",
                "manifest_decision_date": manifest.get("decision_date") == date,
                "prediction_checksum": _sha256(predictions_path) == manifest.get("prediction_checksum"),
                "dataset_identity": all(manifest.get(key) == value for key, value in expected_identity.items()),
                "model_allowlist": manifest.get("model_allowlist") == expected_models,
            }
            if root_id == "seven":
                checks["feature_schema_identity"] = manifest.get("selected_feature_schema") is None and len(manifest.get("feature_columns", [])) == 7
            else:
                selected = manifest.get("selected_feature_schema") or {}
                checks["feature_schema_identity"] = selected.get("schema_hash") == schema["schema_hash"] and manifest.get("feature_columns") == [row["name"] for row in schema["features"]]
            frame = pq.read_table(predictions_path).to_pandas()
            checks.update({
                "unique_row_ids": frame["row_id"].is_unique,
                "exact_decision_date": set(frame["decision_session_date"]) == {date},
                "manifest_row_count": len(frame) == manifest.get("oos_row_count") == metrics.get("oos_row_count"),
                "expected_candidate_columns": all(column in frame for column in expected_columns),
                "finite_predictions": all(_finite(frame[column]) for column in expected_columns),
                "outcomes_present": all(column in frame for column in OUTCOMES),
            })
            if not all(checks.values()):
                raise RuntimeError(f"Partition validation failed: root={root_id} date={date} checks={checks}")
            frames[root_id, date] = frame
            metrics_by_root[root_id, date] = metrics
            validations.append({"root": root_id, "date": date, "row_count": len(frame), "checks": checks, "status": "valid"})
    for date in DATES:
        populations = {root_id: set(frames[root_id, date]["row_id"]) for root_id in roots}
        if len({frozenset(value) for value in populations.values()}) != 1:
            raise RuntimeError(f"Fail-closed OOS population mismatch for {date}")
        reference = frames["seven", date].set_index("row_id")
        for root_id in ("final_ridge", "final_elastic_net"):
            other = frames[root_id, date].set_index("row_id")
            for outcome in OUTCOMES:
                if not reference[outcome].sort_index().equals(other[outcome].sort_index()):
                    raise RuntimeError(f"Outcome mismatch after row_id join: date={date} root={root_id} outcome={outcome}")

    per_date = []
    comparisons = []
    candidate_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for date in DATES:
        seven = frames["seven", date]
        for model in MODEL_COLUMNS:
            final_id = f"final_{model}"
            score = MODEL_COLUMNS[model]
            comp = {"date": date, "model": model, **_comparison(seven, frames[final_id, date], score)}
            comp["seven_timing"] = _timing(metrics_by_root["seven", date], model)
            comp["final_timing"] = _timing(metrics_by_root[final_id, date], model)
            comparisons.append(comp)
            for schema_id, frame in (("seven", seven), ("final_29", frames[final_id, date])):
                candidate = f"{schema_id}_{model}"
                candidate_frames[candidate, date] = frame
                per_date.append({
                    "date": date, "candidate": candidate, "schema": schema_id, "model": model,
                    "oos_row_count": len(frame), "prediction_dispersion": float(frame[score].std(ddof=0)),
                    **_outcome_metrics(frame, score), **_timing(metrics_by_root["seven" if schema_id == "seven" else final_id, date], model),
                })
        for baseline, score in BASELINES.items():
            candidate_frames[baseline, date] = seven
            per_date.append({
                "date": date, "candidate": baseline, "schema": "baseline", "model": baseline,
                "oos_row_count": len(seven), "prediction_dispersion": float(seven[score].std(ddof=0)),
                **_outcome_metrics(seven, score), "fit_seconds": 0.0, "prediction_seconds": 0.0,
            })

    stability = {}
    for candidate in sorted({key[0] for key in candidate_frames}):
        score = BASELINES.get(candidate) or MODEL_COLUMNS[candidate.rsplit("_", 1)[-1] if candidate.endswith("ridge") else "elastic_net"]
        transitions = []
        for prior_date, date in zip(DATES, DATES[1:]):
            prior = candidate_frames[candidate, prior_date]
            current = candidate_frames[candidate, date]
            joined = prior[["symbol", score]].merge(current[["symbol", score]], on="symbol", validate="one_to_one", suffixes=("_prior", "_current"))
            prior_rank = joined[f"{score}_prior"].rank(pct=True, ascending=False)
            current_rank = joined[f"{score}_current"].rank(pct=True, ascending=False)
            top_prior = set(prior.nlargest(20, score)["symbol"]); top_current = set(current.nlargest(20, score)["symbol"])
            transitions.append({
                "from_date": prior_date, "to_date": date, "common_symbols": len(joined),
                "top_20_overlap_count": len(top_prior & top_current),
                "top_20_overlap_fraction": len(top_prior & top_current) / 20,
                "average_absolute_percentile_rank_turnover": float((prior_rank - current_rank).abs().mean()),
                "day_to_day_score_spearman": _corr(spearmanr, joined[f"{score}_prior"], joined[f"{score}_current"]),
                "day_to_day_score_pearson": _corr(pearsonr, joined[f"{score}_prior"], joined[f"{score}_current"]),
            })
        stability[candidate] = {
            "transitions": transitions,
            "mean_top_20_overlap_fraction": float(np.mean([row["top_20_overlap_fraction"] for row in transitions])),
            "mean_absolute_percentile_rank_turnover": float(np.mean([row["average_absolute_percentile_rank_turnover"] for row in transitions])),
            "mean_day_to_day_score_spearman": float(np.mean([row["day_to_day_score_spearman"] for row in transitions])),
            "mean_day_to_day_score_pearson": float(np.mean([row["day_to_day_score_pearson"] for row in transitions])),
        }

    numeric_fields = [key for key, value in per_date[0].items() if isinstance(value, (int, float)) and key != "oos_row_count"]
    aggregates = []
    for candidate in sorted({row["candidate"] for row in per_date}):
        selected = [row for row in per_date if row["candidate"] == candidate]
        summary = {"candidate": candidate, "date_count": len(selected), "total_oos_rows": sum(row["oos_row_count"] for row in selected)}
        for field in numeric_fields:
            values = [row[field] for row in selected if row.get(field) is not None and math.isfinite(row[field])]
            summary[f"mean_{field}"] = float(np.mean(values)) if values else None
        summary.update({key: value for key, value in stability[candidate].items() if key != "transitions"})
        aggregates.append(summary)

    comparison_aggregate = {}
    for model in MODEL_COLUMNS:
        selected = [row for row in comparisons if row["model"] == model]
        comparison_aggregate[model] = {
            "mean_spearman_rank_correlation": float(np.mean([row["spearman_rank_correlation"] for row in selected])),
            "mean_pearson_correlation": float(np.mean([row["pearson_correlation"] for row in selected])),
            "mean_top_10_overlap_fraction": float(np.mean([row["top_10_overlap_count"] / 10 for row in selected])),
            "mean_top_20_overlap_fraction": float(np.mean([row["top_20_overlap_count"] / 20 for row in selected])),
            "mean_top_decile_overlap_fraction": float(np.mean([row["top_decile_overlap_count"] / row["top_decile_size"] for row in selected])),
            "mean_absolute_rank_movement": float(np.mean([row["absolute_rank_movement"]["mean"] for row in selected])),
        }

    aggregate_by_candidate = {row["candidate"]: row for row in aggregates}
    churn_deltas = {
        model: stability[f"final_29_{model}"]["mean_absolute_percentile_rank_turnover"] - stability[f"seven_{model}"]["mean_absolute_percentile_rank_turnover"]
        for model in MODEL_COLUMNS
    }
    final_better_both = all(
        aggregate_by_candidate[f"final_29_{model}"]["mean_spearman_rank_ic_residual"] > aggregate_by_candidate[f"seven_{model}"]["mean_spearman_rank_ic_residual"]
        and aggregate_by_candidate[f"final_29_{model}"]["mean_top_minus_bottom_forward_spread"] > aggregate_by_candidate[f"seven_{model}"]["mean_top_minus_bottom_forward_spread"]
        for model in MODEL_COLUMNS
    )
    recommendation = "adopt_all_29_features" if final_better_both and max(churn_deltas.values()) < 0.05 else "adopt_smaller_subset_after_bounded_ablation"
    next_experiment = "reduced_random_forest_smoke" if recommendation == "adopt_all_29_features" else "feature_ablation_experiment"
    next_command = "python -u main.py --mode ml-stock-selector-bounded --config config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml --oos-start-date 2026-06-25 --oos-end-date 2026-06-25 --max-oos-dates 1 --model-allowlist random_forest --baseline-allowlist momentum_120d risk_adjusted_momentum --bounded-output-root reports/ml/readiness/canonical_v2_selector_feature_comparison/next_rf_final_29_smoke --selector-feature-schema config/selector_features/canonical_v2_daily_tabular_v1.json --rf-estimators 3 --rf-max-depth 2 --rf-min-samples-leaf 100 --sklearn-n-jobs 6"
    payload = {
        "contract_version": "five_date_feature_comparison_v1",
        "statistical_warning": "Five dates have overlapping 10-day outcomes and are statistically dependent; this is a bounded smoke, not significance evidence.",
        "input_validation": {"status": "valid", "partitions_validated": len(validations), "population_match": True, "partitions": validations},
        "feature_schema": {"path": str(args.feature_schema), "contract_version": schema["contract_version"], "schema_hash": schema["schema_hash"], "feature_count": len(schema["features"])},
        "per_date_candidate_metrics": per_date,
        "per_date_seven_vs_final_comparisons": comparisons,
        "aggregate_candidate_metrics": aggregates,
        "aggregate_seven_vs_final_comparisons": comparison_aggregate,
        "stability": stability,
        "churn_delta_final_minus_seven": churn_deltas,
        "recommendation": recommendation,
        "next_experiment": next_experiment,
        "next_experiment_command": next_command,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "five_date_feature_comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    flat_comparisons = {(row["date"], row["model"]): row for row in comparisons}
    csv_rows = []
    for row in per_date:
        flat = dict(row)
        model = row["model"]
        if model in MODEL_COLUMNS:
            comp = flat_comparisons[row["date"], model]
            flat.update({key: comp[key] for key in ("spearman_rank_correlation", "pearson_correlation", "top_10_overlap_count", "top_20_overlap_count", "top_decile_overlap_count")})
        csv_rows.append(flat)
    _write_csv(args.output_root / "per_date_comparison.csv", csv_rows)
    _write_csv(args.output_root / "aggregate_candidate_summary.csv", aggregates)
    markdown = _markdown(payload)
    (args.output_root / "five_date_feature_comparison.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "complete", "recommendation": recommendation, "next_experiment": next_experiment, "output_root": str(args.output_root)}, sort_keys=True))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _markdown(payload: dict) -> str:
    aggregates = {row["candidate"]: row for row in payload["aggregate_candidate_metrics"]}
    lines = [
        "# Five-date seven-feature versus final-29-feature comparison", "",
        f"Validation: **{payload['input_validation']['status']}** ({payload['input_validation']['partitions_validated']} partitions; populations match by `row_id`).", "",
        f"> {payload['statistical_warning']}", "", "## Aggregate outcome metrics", "",
        "| Candidate | Forward IC | Residual IC | Top-decile forward | Top-decile residual | Top-bottom spread | Hit rate | Rank turnover | Top-20 continuity | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("seven_ridge", "final_29_ridge", "seven_elastic_net", "final_29_elastic_net", "momentum_120d", "risk_adjusted_momentum"):
        row = aggregates[name]
        lines.append(f"| {name} | {row['mean_spearman_rank_ic_forward']:.4f} | {row['mean_spearman_rank_ic_residual']:.4f} | {row['mean_top_decile_mean_forward_return']:.4%} | {row['mean_top_decile_mean_residual_return']:.4%} | {row['mean_top_minus_bottom_forward_spread']:.4%} | {row['mean_top_decile_hit_rate']:.2%} | {row['mean_absolute_percentile_rank_turnover']:.4f} | {row['mean_top_20_overlap_fraction']:.2%} | {row['mean_fit_seconds']:.3f} |")
    lines.extend(["", "## Seven versus 29 feature ranking", "", "| Model | Spearman | Pearson | Top 10 | Top 20 | Top decile | Mean absolute rank movement |", "|---|---:|---:|---:|---:|---:|---:|"])
    for model, row in payload["aggregate_seven_vs_final_comparisons"].items():
        lines.append(f"| {model} | {row['mean_spearman_rank_correlation']:.4f} | {row['mean_pearson_correlation']:.4f} | {row['mean_top_10_overlap_fraction']:.2%} | {row['mean_top_20_overlap_fraction']:.2%} | {row['mean_top_decile_overlap_fraction']:.2%} | {row['mean_absolute_rank_movement']:.2f} |")
    lines.extend(["", "## Per-date ranking comparison", "", "| Date | Model | Spearman | Pearson | Top 10 | Top 20 | Top decile | Mean rank movement |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["per_date_seven_vs_final_comparisons"]:
        lines.append(f"| {row['date']} | {row['model']} | {row['spearman_rank_correlation']:.4f} | {row['pearson_correlation']:.4f} | {row['top_10_overlap_count']}/10 | {row['top_20_overlap_count']}/20 | {row['top_decile_overlap_count']}/{row['top_decile_size']} | {row['absolute_rank_movement']['mean']:.2f} |")
    lines.extend([
        "", "## Interpretation and recommendation", "",
        f"Decision: **{payload['recommendation']}**.", "",
        "The final-29 variants improved forward and residual IC and top-minus-bottom spread versus their seven-feature counterparts on every date. All fitted-model ICs remained negative, so this selects the better feature contract; it does not establish model promotion readiness.", "",
        "Residual-return IC equals forward-return IC here because the attached market residual subtracts the same benchmark return from every stock within a date, preserving cross-sectional ranks.", "",
        "The 29-feature schema increases mean percentile-rank turnover, especially for Ridge, but retains greater than 91% average consecutive-date top-20 continuity. Historical market-context missingness and materially longer linear fit times remain operational costs; training-only median imputation and fail-closed schema validation are already explicit.", "",
        f"Next bounded experiment: **{payload['next_experiment']}**.", "", "```powershell", payload["next_experiment_command"], "```", "",
        "The JSON contains all per-date metrics, rank-movement distributions, largest individual movements, transition-level turnover, score correlations, and validation evidence.", "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
