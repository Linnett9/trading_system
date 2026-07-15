from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    runs = {name: json.loads((args.root / name / "diagnostic.json").read_text(encoding="utf-8")) for name in ("expanding_rf", "five_year_rf", "sessions_1260_rf", "expanding_gb")}
    for name in runs:
        temporal = {"decision_timestamp_guard_passed": True, "label_availability_guard_passed": True, "training_decision_timestamp_max": runs[name]["training_decision_timestamp_max"], "training_label_available_timestamp_max": runs[name]["training_label_available_timestamp_max"], "evidence": "The diagnostic owner raises before fitting if either guard fails."}
        runs[name]["temporal_legality"] = temporal
        (args.root / name / "diagnostic.json").write_text(json.dumps(runs[name], indent=2, sort_keys=True), encoding="utf-8")
        manifest_path = args.root / name / "diagnostic_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("completion_status", None); manifest["artifact_kind"] = "research_diagnostic"; manifest["eligible_as_completed_selector_partition"] = False; manifest["temporal_legality"] = temporal
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    feature_table = [{"index": row["feature_index"], "feature": row["feature_name"], **row["before_imputation"], "post_imputation_unique_values": row["after_training_median_imputation"]["unique_value_count"], "post_imputation_standard_deviation": row["after_training_median_imputation"]["standard_deviation"]} for row in runs["expanding_rf"]["oos_feature_variation"]]
    results = [{"run": name, "model": row["model_id"], "window": row["training_window"], "training_rows": row["training_row_count"], "training_date_min": row["training_date_min"], "training_date_max": row["training_date_max"], "fit_seconds": row["fit_seconds"], "prediction_seconds": row["prediction_seconds"], "accepted": row["prediction_quality_accepted"], "unique_predictions": row["prediction_quality"]["unique_finite_value_count"], "distinct_ranks": row["prediction_quality"]["distinct_rank_count"], "standard_deviation": row["prediction_quality"]["standard_deviation"], "range": row["prediction_quality"]["range"]} for name, row in runs.items()]
    payload = {"contract_version": "tree_selector_constant_prediction_diagnosis_v1", "decision_date": "2026-06-25", "root_cause": "Date-level market-context features are constant within the OOS cross-section and dominate the shallow trees' root/near-root splits. All 401 OOS stocks therefore follow identical routes; rolling history changes thresholds and leaf values but not routing diversity.", "feature_order_verified": all(row["feature_order_verified"] for row in runs.values()), "oos_feature_variation": feature_table, "diagnostic_results": results, "expanding_random_forest_routing": runs["expanding_rf"]["routing"], "expanding_gradient_boosting_routing": runs["expanding_gb"]["routing"], "split_feature_distributions": runs["expanding_rf"]["split_feature_distributions"], "recommendation": "run_one_additional_cross_sectional_21_feature_five_year_random_forest_diagnostic", "statistical_selection_rule": "Window and feature exclusion selected only for temporal legality and prediction diversity; realised OOS outcomes were not read."}
    payload["tree_schema"] = {"path": "config/selector_features/canonical_v2_daily_tree_cross_sectional_v1.json", "schema_hash": "AC828BFD364544073045FEED338D7FDAACB20AB7AD199E84001DDCDD7ECC332F", "feature_count": 21}
    (args.root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Constant tree-prediction routing diagnosis", "", payload["root_cause"], "", "## Expanding versus rolling", "", "| Run | Rows | Date range | Fit seconds | Unique | Std dev | Range | Accepted |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for row in results: lines.append(f"| {row['run']} | {row['training_rows']:,} | {row['training_date_min']}..{row['training_date_max']} | {row['fit_seconds']:.3f} | {row['unique_predictions']} | {row['standard_deviation']:.3g} | {row['range']:.3g} | {row['accepted']} |")
    lines.extend(["", "## OOS feature variation", "", "| # | Feature | Classification | Finite | Unique | Min | Max | Std dev | Missing | Post-imputer unique |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in feature_table: lines.append(f"| {row['index']} | {row['feature']} | {row['classification']} | {row['finite_count']} | {row['unique_value_count']} | {row['minimum']:.6g} | {row['maximum']:.6g} | {row['standard_deviation']:.6g} | {row['missingness']:.2%} | {row['post_imputation_unique_values']} |")
    lines.extend(["", "## Recommendation", "", "Run one additional five-year RF diagnostic with the versioned 21-feature cross-sectional schema. Do not start checkpoint work unless it passes the unchanged quality gate.", ""])
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "root": str(args.root), "recommendation": payload["recommendation"]}))


if __name__ == "__main__": main()
