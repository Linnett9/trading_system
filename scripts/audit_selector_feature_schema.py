from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research.ml.stock_level.selector_feature_schema import classify_column, load_feature_schema


def _stats(path: Path, date_column: str, sample_limit: int = 100_000) -> dict[str, dict]:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    nulls = {name: 0 for name in names}
    earliest = {name: None for name in names}
    samples = {name: Counter() for name in names}
    sampled = {name: 0 for name in names}
    for batch in parquet.iter_batches(batch_size=65_536):
        dates = batch.column(batch.schema.get_field_index(date_column))
        for index, name in enumerate(names):
            column = batch.column(index)
            nulls[name] += column.null_count
            usable_date = pc.min(pc.filter(dates, pc.is_valid(column))).as_py()
            if usable_date is not None and (earliest[name] is None or usable_date < earliest[name]):
                earliest[name] = usable_date
            remaining = sample_limit - sampled[name]
            if remaining > 0:
                chosen = column.slice(0, remaining).to_pylist()
                samples[name].update(str(value) for value in chosen if value is not None)
                sampled[name] += len(chosen)
    rows = parquet.metadata.num_rows
    result = {}
    for field in parquet.schema_arrow:
        counts = samples[field.name]
        non_null_sample = sum(counts.values())
        dominant = max(counts.values(), default=0)
        result[field.name] = {
            "data_type": str(field.type),
            "row_count": rows,
            "null_count": nulls[field.name],
            "missingness": nulls[field.name] / rows,
            "earliest_usable_date": earliest[field.name],
            "constant": len(counts) == 1 and nulls[field.name] < rows,
            "near_constant": bool(non_null_sample and dominant / non_null_sample >= 0.99),
            "near_constant_method": f"deterministic first-{sample_limit:,}-row sample; threshold >=99%",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    schema = load_feature_schema(args.feature_schema)
    predictors = tuple(row["name"] for row in schema["features"])
    rows_stats = _stats(args.dataset_root / "rows.parquet", "decision_session_date")
    sidecar_stats = _stats(args.dataset_root / "baseline_scores.parquet", "decision_timestamp")
    records = []
    for source, stats in (("rows.parquet", rows_stats), ("baseline_scores.parquet", sidecar_stats)):
        for name, values in stats.items():
            category, reason = classify_column(name, predictor_names=predictors)
            all_null = values["null_count"] == values["row_count"]
            if all_null and category != "target/outcome":
                category, reason = "unavailable or all-null", "Frozen artifact column is entirely null."
            records.append({
                "source_file": source, "column_name": name, **values,
                "semantic_category": category,
                "point_in_time_availability": "available at decision time" if category in {"safe predictor", "conditionally available", "identity", "timestamp", "provenance/diagnostic"} else "not a predictor at decision time",
                "proposed_predictor_status": "include" if category in {"safe predictor", "conditionally available"} else "exclude",
                "reason": reason,
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {"contract_version": "canonical_v2_selector_column_audit_v1", "feature_schema": {"path": str(args.feature_schema), "schema_hash": schema["schema_hash"]}, "columns": records}
    (args.output_root / "column_inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (args.output_root / "column_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    print(json.dumps({"columns": len(records), "included": sum(row["proposed_predictor_status"] == "include" for row in records), "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
