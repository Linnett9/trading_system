"""Reconstruct causal full ranks from saved DS24 tabular model vintages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.canonical_prequential_engine import load_predictor_manifest


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
WORKERS = STAGE / "r7_r14_policy_workers"
PREDICTORS = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json"
FEATURE_ROOT = ROOT / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1/run=ds24_p8_r2_local_20260821T000000Z"
PARTITIONS = STAGE / "91_r7_r1_extended_model_data_partition_manifest.csv"
TRACE_ROOTS = {
    "huber": "metrics_only_v3_r37_huber_replay",
    "elastic_net": "metrics_only_v3_r40_elastic_net",
    "rff_ridge": "metrics_only_v3_r37_rff_retry",
}


def openable(path: Path) -> str:
    absolute = str(path.resolve())
    return "\\\\?\\" + absolute if os.name == "nt" and len(absolute) >= 240 else absolute


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(openable(path), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_vintages(family: str, start: str, end: str) -> dict[str, tuple[Path, str]]:
    models = {}
    for path in sorted((WORKERS / family / "models").glob(f"{family}_*.pkl")):
        stamp = path.stem.removeprefix(f"{family}_")
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        if not start <= day <= end:
            continue
        if day in models:
            raise ValueError(f"multiple model vintages for {family} {day}")
        models[day] = (path, sha256(path))
    if not models:
        raise ValueError(f"no saved model vintages for {family} in range")
    return models


def _accepted_score_timestamps(family: str, start: str, end: str) -> tuple[set[pd.Timestamp], Path]:
    manifest_path = WORKERS / family / TRACE_ROOTS[family] / "decision_trace_v3_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamps: set[pd.Timestamp] = set()
    for part in manifest["parts"]:
        if part["max_decision_timestamp"][:10] < start or part["min_decision_timestamp"][:10] > end:
            continue
        trace_path = manifest_path.parent / part["path"]
        observed = pd.read_parquet(openable(trace_path), columns=["decision_timestamp"])
        timestamps.update(pd.to_datetime(observed["decision_timestamp"], utc=True).unique())
    timestamps = {stamp for stamp in timestamps if start <= stamp.date().isoformat() <= end}
    if not timestamps:
        raise ValueError(f"accepted producer timestamp spine is empty for {family}")
    return timestamps, manifest_path


def reconstruct(root: Path, family: str, start: str, end: str, label: str | None = None) -> None:
    available_mib = psutil.virtual_memory().available // (1024 * 1024)
    if available_mib < 4096:
        raise RuntimeError(f"DS24 reconstruction resource gate: {available_mib} MiB free < 4096 MiB")
    label = label or family
    if not label.startswith(family) or not label.replace("_", "").isalnum():
        raise ValueError("score authority label must be an alphanumeric family-prefixed identifier")
    predictor_manifest = load_predictor_manifest(PREDICTORS)
    models = _model_vintages(family, start, end)

    @lru_cache(maxsize=64)
    def load_model(day: str) -> object:
        path, _ = models[day]
        with open(openable(path), "rb") as source:
            package = pickle.load(source)
        if package["family"] != family or day not in package["spec"].score_session_dates:
            raise ValueError(f"saved model vintage does not own score date {day}: {path}")
        return package["model"]

    accepted_timestamps, trace_manifest_path = _accepted_score_timestamps(family, start, end)
    rows = pd.read_csv(PARTITIONS, usecols=["asset_id", "year", "feature_partition"])
    rows = rows[rows["year"].between(int(start[:4]), int(end[:4]))]
    raw_root = root / "scores" / f"{label}_raw"
    raw_root.mkdir(parents=True, exist_ok=False)
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = defaultdict(int)
    observed_timestamps: set[pd.Timestamp] = set()
    contexts = {}
    for year in sorted(rows["year"].unique()):
        context_path = FEATURE_ROOT / "shared-context" / f"year={year}" / "context.parquet"
        context = pd.read_parquet(openable(context_path), columns=[
            "decision_timestamp", *predictor_manifest.context_predictors,
        ])
        if context.duplicated("decision_timestamp").any():
            raise ValueError(f"duplicate context timestamps in {year}")
        contexts[year] = context
    try:
        for index, row in enumerate(rows.itertuples(index=False), start=1):
            feature_path = ROOT / row.feature_partition
            features = pd.read_parquet(openable(feature_path), columns=[
                "asset_id", "canonical_symbol", "decision_timestamp", "session_date",
                *predictor_manifest.stock_predictors,
            ])
            features = features[features["session_date"].isin(models)]
            features = features[features["decision_timestamp"].isin(accepted_timestamps)]
            if features.empty:
                continue
            observed_timestamps.update(features["decision_timestamp"].unique())
            if features.duplicated(["asset_id", "decision_timestamp"]).any():
                raise ValueError(f"duplicate causal feature key: {feature_path}")
            frame = features.merge(contexts[row.year], on="decision_timestamp", how="left", validate="many_to_one")
            if len(frame) != len(features):
                raise ValueError("context join changed score population")
            matrix = frame[predictor_manifest.predictors].replace([np.inf, -np.inf], np.nan)
            frame["score"] = np.nan
            for day, positions in frame.groupby("session_date", sort=False).indices.items():
                if day not in models:
                    raise ValueError(f"no saved model for eligible feature session: {day}")
                model = load_model(day)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                    prediction = np.asarray(model.predict(matrix.iloc[positions]), dtype=float)
                if not np.isfinite(prediction).all():
                    raise ValueError(f"non-finite score for {day}")
                frame.iloc[positions, frame.columns.get_loc("score")] = prediction
            output = frame[["decision_timestamp", "asset_id", "canonical_symbol", "session_date", "score"]]
            for month, group in output.groupby(output["session_date"].str.slice(0, 7), sort=False):
                table = pa.Table.from_pandas(group.reset_index(drop=True), preserve_index=False)
                if month not in writers:
                    writers[month] = pq.ParquetWriter(openable(raw_root / f"month={month}.parquet"), table.schema,
                                                      compression="zstd")
                writers[month].write_table(table)
                counts[month] += len(group)
            if index % 100 == 0:
                available_mib = psutil.virtual_memory().available // (1024 * 1024)
                if available_mib < 1536:
                    raise RuntimeError(f"DS24 reconstruction resource gate tripped: {available_mib} MiB free")
                print(json.dumps({"family": family, "feature_partitions_read": index,
                                  "score_rows_written": sum(counts.values())}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    missing_timestamps = accepted_timestamps - observed_timestamps
    if missing_timestamps:
        raise ValueError(f"accepted score timestamps without causal features: {len(missing_timestamps)}")

    full_root = root / "scores" / f"{label}_full_rank"
    full_root.mkdir(exist_ok=False)
    total = 0
    for month in sorted(writers):
        raw = pd.read_parquet(openable(raw_root / f"month={month}.parquet"))
        raw = raw.sort_values(["decision_timestamp", "score", "asset_id"], ascending=[True, False, True])
        raw["rank"] = raw.groupby("decision_timestamp", sort=False).cumcount() + 1
        raw["model_artifact_path"] = raw["session_date"].map(
            {day: str(value[0].relative_to(ROOT)).replace("\\", "/") for day, value in models.items()}
        )
        raw["model_artifact_sha256"] = raw["session_date"].map(
            {day: value[1] for day, value in models.items()}
        )
        raw["feature_order_hash"] = predictor_manifest.manifest_hash
        raw["feature_authority"] = str(FEATURE_ROOT.relative_to(ROOT)).replace("\\", "/")
        path = full_root / f"month={month}.parquet"
        raw.to_parquet(openable(path), index=False, compression="zstd")
        total += len(raw)
        (raw_root / f"month={month}.parquet").unlink()
    raw_root.rmdir()
    changed = [day for day, (path, original_hash) in models.items() if sha256(path) != original_hash]
    if changed:
        raise RuntimeError(f"saved model artifact hashes changed during reconstruction: {changed[:5]}")
    authority = {
        "family": family,
        "authority_label": label,
        "first_session": min(models),
        "last_session": max(models),
        "saved_model_vintages_used": len(models),
        "score_rows": total,
        "decision_timestamps": len(accepted_timestamps),
        "score_months": sorted(writers),
        "predictor_manifest": str(PREDICTORS.relative_to(ROOT)).replace("\\", "/"),
        "predictor_manifest_sha256": sha256(PREDICTORS),
        "feature_order_hash": predictor_manifest.manifest_hash,
        "score_population": "all causal feature rows, independent of future target availability",
        "target_partitions_read": 0,
        "producer_trace_manifest": str(trace_manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "producer_trace_manifest_sha256": sha256(trace_manifest_path),
        "source_partition_manifest": str(PARTITIONS.relative_to(ROOT)).replace("\\", "/"),
        "source_partition_manifest_sha256": sha256(PARTITIONS),
        "fit_calls": 0,
        "partial_fit_calls": 0,
        "model_weights_changed": False,
        "output_root": str(full_root.relative_to(root)).replace("\\", "/"),
    }
    (root / "score_authorities").mkdir(exist_ok=True)
    (root / "score_authorities" / f"{label}.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"family": family, "score_rows": total, "model_vintages": len(models)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--family", choices=["huber", "elastic_net", "rff_ridge"], required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--label")
    args = parser.parse_args()
    reconstruct(args.experiment_root.resolve(), args.family, args.start, args.end, args.label)


if __name__ == "__main__":
    main()
