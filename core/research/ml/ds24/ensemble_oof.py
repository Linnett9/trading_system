from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from core.research.ml.ds24_metrics_only_evaluator import compute_per_t_metrics


SCORE_CONTRACT_ID = "DS24_ENSEMBLE_OOF_SCORE_CONTRACT_V1"
SCORE_CONTRACT_VERSION = "DS24_ENSEMBLE_OOF_SCORE_CONTRACT_V1"
COMPACT_SCORE_CONTRACT_ID = "DS24_ENSEMBLE_OOF_COMPACT_CONTRACT_V2"
COMPACT_SCORE_CONTRACT_VERSION = "DS24_ENSEMBLE_OOF_COMPACT_CONTRACT_V2"
ENSEMBLE_OOF_COLUMNS = [
    "trial_id",
    "run_id",
    "family",
    "decision_timestamp",
    "decision_date",
    "asset_id",
    "long_selection_score",
    "cross_sectional_rank",
    "cross_sectional_percentile",
    "eligible_assets",
    "training_cutoff_timestamp",
    "refit_id",
    "refit_ordinal",
    "model_config_hash",
    "dataset_manifest_hash",
    "predictor_contract_hash",
    "target_contract_hash",
    "evaluation_contract_hash",
    "score_contract_version",
]
COMPACT_OOF_COLUMNS = [
    "decision_timestamp",
    "asset_id",
    "long_selection_score",
    "cross_sectional_rank",
    "cross_sectional_percentile",
    "eligible_assets",
    "refit_ordinal",
]
FORBIDDEN_SCORE_COLUMNS = {
    "target",
    "target_value",
    "target_return",
    "forward_return",
    "future_return",
    "actual_return",
    "paper_order",
    "live_order",
    "broker_order",
    "api_key",
    "token",
    "password",
}
FAMILY_ENFORCEMENT_SCOPE = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "dlinear",
    "patchtst",
    "transformer",
    "itransformer",
    "momentum_transformer",
    "market_context_encoder",
    "temporal_fusion_transformer",
]


class EnsembleOOFError(RuntimeError):
    pass


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    with open(openable_path(path), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(openable_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_contract_payload() -> dict[str, Any]:
    payload = {
        "contract_id": SCORE_CONTRACT_ID,
        "version": 1,
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "output_root_pattern": (
            "remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer/"
            "ensemble_oof_scores_v1/decision_date=YYYY-MM-DD/part-*.parquet"
        ),
        "columns": list(ENSEMBLE_OOF_COLUMNS),
        "score_orientation": {
            "higher_is_better": True,
            "tft_long_selection_score": "1.0 - probability_should_reduce_exposure",
        },
        "types": ensemble_schema_payload()["columns"],
        "compression": "zstd",
        "partitioning": "decision_date",
        "prohibited_content": sorted(FORBIDDEN_SCORE_COLUMNS),
        "oos_rules": [
            "training_cutoff_timestamp < decision_timestamp",
            "no target outcomes",
            "no raw feature columns",
            "no holdout rows",
            "no paper/live/broker fields",
            "unique trial_id, decision_timestamp, asset_id",
        ],
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def ensemble_schema_payload() -> dict[str, Any]:
    return {
        "schema_id": "DS24_ENSEMBLE_OOF_SCORE_SCHEMA_V1",
        "deterministic_column_order": list(ENSEMBLE_OOF_COLUMNS),
        "columns": [
            {"name": "trial_id", "type": "string", "encoding": "dictionary"},
            {"name": "run_id", "type": "string", "encoding": "dictionary"},
            {"name": "family", "type": "string", "encoding": "dictionary"},
            {"name": "decision_timestamp", "type": "timestamp[us, UTC]"},
            {"name": "decision_date", "type": "date32", "encoding": "dictionary"},
            {"name": "asset_id", "type": "string", "encoding": "dictionary"},
            {"name": "long_selection_score", "type": "float32"},
            {"name": "cross_sectional_rank", "type": "int32"},
            {"name": "cross_sectional_percentile", "type": "float32"},
            {"name": "eligible_assets", "type": "int32"},
            {"name": "training_cutoff_timestamp", "type": "timestamp[us, UTC]"},
            {"name": "refit_id", "type": "string", "encoding": "dictionary"},
            {"name": "refit_ordinal", "type": "int32"},
            {"name": "model_config_hash", "type": "string", "encoding": "dictionary"},
            {"name": "dataset_manifest_hash", "type": "string", "encoding": "dictionary"},
            {"name": "predictor_contract_hash", "type": "string", "encoding": "dictionary"},
            {"name": "target_contract_hash", "type": "string", "encoding": "dictionary"},
            {"name": "evaluation_contract_hash", "type": "string", "encoding": "dictionary"},
            {"name": "score_contract_version", "type": "string", "encoding": "dictionary"},
        ],
        "no_row_index_column": True,
        "parquet_compression": "zstd",
    }


def compact_oof_v2_contract_payload() -> dict[str, Any]:
    payload = {
        "contract_id": COMPACT_SCORE_CONTRACT_ID,
        "version": 2,
        "score_contract_version": COMPACT_SCORE_CONTRACT_VERSION,
        "source_contract_id": SCORE_CONTRACT_ID,
        "source_contract_version": SCORE_CONTRACT_VERSION,
        "output_root_pattern": (
            "remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/family=<family>/"
            "ensemble_oof_scores_v2/decision_date=YYYY-MM-DD/part-refit=XXXXXX.parquet"
        ),
        "row_columns": list(COMPACT_OOF_COLUMNS),
        "metadata_moves_static_lineage_out_of_rows": True,
        "manifest_static_lineage_fields": [
            "trial_id",
            "run_id",
            "family",
            "model_config_hash",
            "dataset_manifest_hash",
            "predictor_contract_hash",
            "target_contract_hash",
            "evaluation_contract_hash",
            "source_score_contract_version",
        ],
        "manifest_refit_lineage_fields": [
            "refit_ordinal",
            "refit_id",
            "training_cutoff_timestamp",
        ],
        "score_orientation": {
            "higher_is_better": True,
            "tft_long_selection_score": "1.0 - probability_should_reduce_exposure",
            "ranking_families": "higher raw ranker score is better",
            "sequence_families": "higher predicted forward return/probability score is better",
        },
        "types": compact_oof_v2_schema_payload()["columns"],
        "compression": "zstd",
        "partitioning": "decision_date derived from decision_timestamp, plus refit_ordinal in file name",
        "dictionary_encoding": ["asset_id"],
        "prohibited_content": sorted(FORBIDDEN_SCORE_COLUMNS),
        "oos_rules": [
            "training_cutoff_timestamp < decision_timestamp after metadata expansion",
            "no target outcomes",
            "no raw feature columns",
            "no holdout rows",
            "no paper/live/broker fields",
            "unique decision_timestamp, asset_id inside each family/trial manifest",
        ],
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def compact_oof_v2_schema_payload() -> dict[str, Any]:
    return {
        "schema_id": "DS24_ENSEMBLE_OOF_COMPACT_SCHEMA_V2",
        "deterministic_column_order": list(COMPACT_OOF_COLUMNS),
        "columns": [
            {"name": "decision_timestamp", "type": "timestamp[us, UTC]"},
            {"name": "asset_id", "type": "string", "encoding": "dictionary"},
            {"name": "long_selection_score", "type": "float32"},
            {"name": "cross_sectional_rank", "type": "int32"},
            {"name": "cross_sectional_percentile", "type": "float32"},
            {"name": "eligible_assets", "type": "int32"},
            {"name": "refit_ordinal", "type": "int32"},
        ],
        "no_row_index_column": True,
        "parquet_compression": "zstd",
        "static_lineage_location": "ensemble_oof_scores_manifest_v2.json",
    }


def _reject_forbidden_columns(columns: Iterable[str]) -> None:
    lowered = {str(column).lower() for column in columns}
    forbidden = [
        column
        for column in lowered
        if column in FORBIDDEN_SCORE_COLUMNS
        or column.startswith("feature_")
        or column.startswith("target_")
        or column.startswith("future_")
        or column.startswith("actual_")
        or column.startswith("forward_")
    ]
    if forbidden:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_FORBIDDEN_COLUMNS:" + ",".join(sorted(forbidden)))


def prepare_oof_score_frame(
    predictions: pd.DataFrame,
    *,
    trial_id: str,
    run_id: str,
    family: str,
    training_cutoff_timestamp: str | pd.Timestamp,
    refit_id: str,
    refit_ordinal: int,
    model_config_hash: str,
    dataset_manifest_hash: str,
    predictor_contract_hash: str,
    target_contract_hash: str,
    evaluation_contract_hash: str,
    holdout_start: str = "2025-04-02",
) -> pd.DataFrame:
    _reject_forbidden_columns(predictions.columns)
    required = {"decision_timestamp", "asset_id", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_MISSING_COLUMNS:" + ",".join(missing))
    frame = predictions[["decision_timestamp", "asset_id", "prediction"]].copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    cutoff = pd.to_datetime(training_cutoff_timestamp, utc=True)
    if (frame["decision_timestamp"] <= cutoff).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_TRAINING_CUTOFF_VIOLATION")
    holdout_date = pd.Timestamp(holdout_start).date()
    if (frame["decision_timestamp"].dt.date >= holdout_date).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_HOLDOUT_ROW_REJECTED")
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame["long_selection_score"] = pd.to_numeric(frame["prediction"], errors="coerce").astype("float64")
    if not frame["long_selection_score"].map(math.isfinite).all():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_NONFINITE_SCORE")
    if frame.duplicated(["decision_timestamp", "asset_id"]).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_DUPLICATE_KEY")
    frame = frame.sort_values(["decision_timestamp", "long_selection_score", "asset_id"], ascending=[True, False, True])
    frame["cross_sectional_rank"] = (
        frame.groupby("decision_timestamp", sort=False).cumcount().astype("int32") + 1
    )
    frame["eligible_assets"] = frame.groupby("decision_timestamp")["asset_id"].transform("count").astype("int32")
    frame["cross_sectional_percentile"] = (
        (frame["eligible_assets"] - frame["cross_sectional_rank"] + 1) / frame["eligible_assets"]
    ).astype("float32")
    frame["decision_date"] = frame["decision_timestamp"].dt.date
    out = pd.DataFrame(
        {
            "trial_id": trial_id,
            "run_id": run_id,
            "family": family,
            "decision_timestamp": frame["decision_timestamp"],
            "decision_date": pd.to_datetime(frame["decision_date"]).dt.date,
            "asset_id": frame["asset_id"],
            "long_selection_score": frame["long_selection_score"].astype("float32"),
            "cross_sectional_rank": frame["cross_sectional_rank"].astype("int32"),
            "cross_sectional_percentile": frame["cross_sectional_percentile"].astype("float32"),
            "eligible_assets": frame["eligible_assets"].astype("int32"),
            "training_cutoff_timestamp": cutoff,
            "refit_id": refit_id,
            "refit_ordinal": int(refit_ordinal),
            "model_config_hash": model_config_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "predictor_contract_hash": predictor_contract_hash,
            "target_contract_hash": target_contract_hash,
            "evaluation_contract_hash": evaluation_contract_hash,
            "score_contract_version": SCORE_CONTRACT_VERSION,
        },
        columns=ENSEMBLE_OOF_COLUMNS,
    ).sort_values(["decision_timestamp", "asset_id"]).reset_index(drop=True)
    validate_oof_frame(out)
    return out


def validate_oof_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if list(frame.columns) != ENSEMBLE_OOF_COLUMNS:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_SCHEMA_DRIFT")
    if frame.duplicated(["trial_id", "decision_timestamp", "asset_id"]).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_DUPLICATE_KEY")
    ts = pd.to_datetime(frame["decision_timestamp"], utc=True)
    cutoff = pd.to_datetime(frame["training_cutoff_timestamp"], utc=True)
    if (cutoff >= ts).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_TRAINING_CUTOFF_VIOLATION")
    if not pd.to_numeric(frame["long_selection_score"], errors="coerce").map(math.isfinite).all():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_NONFINITE_SCORE")
    if (pd.to_numeric(frame["cross_sectional_rank"]) < 1).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_RANK_RANGE_VIOLATION")
    if (pd.to_numeric(frame["cross_sectional_rank"]) > pd.to_numeric(frame["eligible_assets"])).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_RANK_RANGE_VIOLATION")
    pct = pd.to_numeric(frame["cross_sectional_percentile"], errors="coerce")
    if ((pct < 0.0) | (pct > 1.0) | ~pct.map(math.isfinite)).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_PERCENTILE_RANGE_VIOLATION")
    return {
        "valid": True,
        "row_count": int(len(frame)),
        "distinct_timestamps": int(ts.nunique()),
        "distinct_assets": int(frame["asset_id"].nunique()),
        "first_timestamp": ts.min().isoformat() if len(ts) else "",
        "last_timestamp": ts.max().isoformat() if len(ts) else "",
    }


def _arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("trial_id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("family", pa.string()),
            pa.field("decision_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("decision_date", pa.date32()),
            pa.field("asset_id", pa.string()),
            pa.field("long_selection_score", pa.float32()),
            pa.field("cross_sectional_rank", pa.int32()),
            pa.field("cross_sectional_percentile", pa.float32()),
            pa.field("eligible_assets", pa.int32()),
            pa.field("training_cutoff_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("refit_id", pa.string()),
            pa.field("refit_ordinal", pa.int32()),
            pa.field("model_config_hash", pa.string()),
            pa.field("dataset_manifest_hash", pa.string()),
            pa.field("predictor_contract_hash", pa.string()),
            pa.field("target_contract_hash", pa.string()),
            pa.field("evaluation_contract_hash", pa.string()),
            pa.field("score_contract_version", pa.string()),
        ]
    )


def write_oof_partitions(
    run_root: Path,
    frame: pd.DataFrame,
    *,
    allow_existing_same_hash: bool = True,
) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_oof_frame(frame)
    ledger: list[dict[str, Any]] = []
    score_root = run_root / "ensemble_oof_scores_v1"
    for decision_date, part in frame.groupby("decision_date", sort=True):
        ordinal = int(part["refit_ordinal"].iloc[0])
        rel = Path(f"ensemble_oof_scores_v1/decision_date={decision_date}/part-refit={ordinal:06d}.parquet")
        path = run_root / rel
        os.makedirs(openable_path(path.parent), exist_ok=True)
        table = pa.Table.from_pandas(part[ENSEMBLE_OOF_COLUMNS], schema=_arrow_schema(), preserve_index=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as handle:
            tmp = Path(handle.name)
        pq.write_table(
            table,
            openable_path(tmp),
            compression="zstd",
            use_dictionary=[
                "trial_id",
                "run_id",
                "family",
                "decision_date",
                "asset_id",
                "refit_id",
                "model_config_hash",
                "dataset_manifest_hash",
                "predictor_contract_hash",
                "target_contract_hash",
                "evaluation_contract_hash",
                "score_contract_version",
            ],
        )
        new_hash = sha256_file(tmp)
        if path.exists():
            old_hash = sha256_file(path)
            if old_hash != new_hash or not allow_existing_same_hash:
                tmp.unlink(missing_ok=True)
                raise EnsembleOOFError("DS24_ENSEMBLE_OOF_IMMUTABLE_PARTITION_HASH_MISMATCH")
            tmp.unlink(missing_ok=True)
        else:
            os.replace(openable_path(tmp), openable_path(path))
        ledger.append(
            {
                "relative_path": rel.as_posix(),
                "decision_date": str(decision_date),
                "refit_ordinal": ordinal,
                "row_count": int(len(part)),
                "first_decision_timestamp": pd.to_datetime(part["decision_timestamp"], utc=True).min().isoformat(),
                "last_decision_timestamp": pd.to_datetime(part["decision_timestamp"], utc=True).max().isoformat(),
                "compressed_bytes": os.stat(openable_path(path)).st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(ledger, key=lambda row: row["relative_path"])


def build_oof_manifest(
    run_root: Path,
    ledger: list[dict[str, Any]],
    *,
    run_id: str,
    trial_id: str,
    family: str,
    model_config_hash: str,
    source_bundle_hash: str,
    data_manifest_hash: str,
    predictor_contract_hash: str,
    target_contract_hash: str,
    evaluation_contract_hash: str,
    terminal_completeness_state: str,
    latest_completed_refit_ordinal: int,
    provisional: bool,
) -> dict[str, Any]:
    frames = [read_oof_partition(run_root / row["relative_path"]) for row in ledger]
    all_scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ENSEMBLE_OOF_COLUMNS)
    audit = validate_oof_frame(all_scores) if not all_scores.empty else {"valid": True, "row_count": 0}
    manifest = {
        "manifest_id": "ensemble_oof_scores_manifest_v1",
        "score_contract_id": SCORE_CONTRACT_ID,
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "run_id": run_id,
        "trial_id": trial_id,
        "family": family,
        "schema_version": 1,
        "row_count": int(len(all_scores)),
        "partition_count": len(ledger),
        "first_decision_timestamp": audit.get("first_timestamp", ""),
        "last_decision_timestamp": audit.get("last_timestamp", ""),
        "distinct_decision_timestamps": int(all_scores["decision_timestamp"].nunique()) if not all_scores.empty else 0,
        "distinct_assets": int(all_scores["asset_id"].nunique()) if not all_scores.empty else 0,
        "source_configuration_hash": model_config_hash,
        "source_bundle_hash": source_bundle_hash,
        "data_manifest_hash": data_manifest_hash,
        "predictor_contract_hash": predictor_contract_hash,
        "target_contract_hash": target_contract_hash,
        "evaluation_contract_hash": evaluation_contract_hash,
        "score_orientation": "higher_long_selection_score_is_better; TFT=1.0-probability_should_reduce_exposure",
        "compression_codec": "zstd",
        "total_compressed_bytes": int(sum(row["compressed_bytes"] for row in ledger)),
        "files": ledger,
        "terminal_completeness_state": terminal_completeness_state,
        "latest_successfully_completed_refit_ordinal": int(latest_completed_refit_ordinal),
        "provisional": bool(provisional),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def read_oof_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(openable_path(path))
    if list(frame.columns) != ENSEMBLE_OOF_COLUMNS:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_SCHEMA_DRIFT")
    return frame


def publish_oof_manifest(run_root: Path, manifest: Mapping[str, Any]) -> None:
    write_json(run_root / "ensemble_oof_scores_manifest_v1.json", dict(manifest))
    digest = stable_hash(dict(manifest))
    os.makedirs(openable_path(run_root), exist_ok=True)
    with open(openable_path(run_root / "ensemble_oof_scores_manifest_v1.sha256"), "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    write_csv(run_root / "ensemble_oof_partition_ledger_v1.csv", list(manifest.get("files", [])))


def validate_oof_manifest(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("score_contract_id") != SCORE_CONTRACT_ID:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_MANIFEST_CONTRACT_MISMATCH")
    allowed_families = set(FAMILY_ENFORCEMENT_SCOPE)
    if manifest.get("family") not in allowed_families:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_UNKNOWN_FAMILY")
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for row in manifest.get("files", []):
        rel = str(row.get("relative_path", ""))
        if not rel or rel in seen:
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_DUPLICATE_PARTITION")
        seen.add(rel)
        path = run_root / rel
        if not os.path.exists(openable_path(path)):
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_MISSING_PARTITION")
        if sha256_file(path) != row.get("sha256"):
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_PARTITION_HASH_MISMATCH")
        frames.append(read_oof_partition(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ENSEMBLE_OOF_COLUMNS)
    audit = validate_oof_frame(frame) if not frame.empty else {"valid": True, "row_count": 0}
    if not frame.empty:
        scalar_expectations = {
            "run_id": manifest.get("run_id", ""),
            "trial_id": manifest.get("trial_id", ""),
            "family": manifest.get("family", ""),
            "model_config_hash": manifest.get("source_configuration_hash", ""),
            "dataset_manifest_hash": manifest.get("data_manifest_hash", ""),
            "predictor_contract_hash": manifest.get("predictor_contract_hash", ""),
            "target_contract_hash": manifest.get("target_contract_hash", ""),
            "evaluation_contract_hash": manifest.get("evaluation_contract_hash", ""),
            "score_contract_version": SCORE_CONTRACT_VERSION,
        }
        for column, expected in scalar_expectations.items():
            if expected and set(frame[column].astype(str)) != {str(expected)}:
                raise EnsembleOOFError(f"DS24_ENSEMBLE_OOF_{column.upper()}_MISMATCH")
    if int(manifest.get("row_count", -1)) != int(audit["row_count"]):
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_MANIFEST_ROW_COUNT_MISMATCH")
    if not frame.empty and str(manifest.get("evaluation_contract_hash", "")) and frame["evaluation_contract_hash"].nunique() > 1:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_MIXED_EVALUATION_CONTRACTS")
    return {
        "valid": True,
        "row_count": int(audit["row_count"]),
        "partition_count": len(seen),
        "total_compressed_bytes": int(sum(int(row.get("compressed_bytes", 0)) for row in manifest.get("files", []))),
    }


def audit_oof_scores(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    frames = [read_oof_partition(run_root / row["relative_path"]) for row in manifest.get("files", [])]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ENSEMBLE_OOF_COLUMNS)
    ts = pd.to_datetime(frame["decision_timestamp"], utc=True) if not frame.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    cutoff = pd.to_datetime(frame["training_cutoff_timestamp"], utc=True) if not frame.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    scores = pd.to_numeric(frame["long_selection_score"], errors="coerce") if not frame.empty else pd.Series([], dtype="float64")
    pct = pd.to_numeric(frame["cross_sectional_percentile"], errors="coerce") if not frame.empty else pd.Series([], dtype="float64")
    ranks = pd.to_numeric(frame["cross_sectional_rank"], errors="coerce") if not frame.empty else pd.Series([], dtype="float64")
    eligible = pd.to_numeric(frame["eligible_assets"], errors="coerce") if not frame.empty else pd.Series([], dtype="float64")
    result = {
        "exported_rows": int(len(frame)),
        "distinct_timestamps": int(ts.nunique()) if not frame.empty else 0,
        "distinct_assets": int(frame["asset_id"].nunique()) if not frame.empty else 0,
        "duplicate_rows": int(frame.duplicated(["trial_id", "decision_timestamp", "asset_id"]).sum()) if not frame.empty else 0,
        "training_cutoff_violations": int((cutoff >= ts).sum()) if not frame.empty else 0,
        "missing_score_rows": int(scores.isna().sum()) if not frame.empty else 0,
        "non_finite_scores": int((~scores.map(math.isfinite)).sum()) if not frame.empty else 0,
        "rank_range_violations": int(((ranks < 1) | (ranks > eligible)).sum()) if not frame.empty else 0,
        "percentile_range_violations": int(((pct < 0.0) | (pct > 1.0) | ~pct.map(math.isfinite)).sum()) if not frame.empty else 0,
        "holdout_rows": int((ts.dt.date >= pd.Timestamp("2025-04-02").date()).sum()) if not frame.empty else 0,
        "first_timestamp": ts.min().isoformat() if not frame.empty else "",
        "last_timestamp": ts.max().isoformat() if not frame.empty else "",
        "partition_count": int(len(manifest.get("files", []))),
        "compressed_bytes": int(sum(int(row.get("compressed_bytes", 0)) for row in manifest.get("files", []))),
    }
    result["status"] = "PASS" if all(
        result[key] == 0
        for key in (
            "duplicate_rows",
            "training_cutoff_violations",
            "missing_score_rows",
            "non_finite_scores",
            "rank_range_violations",
            "percentile_range_violations",
            "holdout_rows",
        )
    ) else "FAIL"
    return result


def create_sync_snapshot(run_root: Path, snapshot_root: Path, *, checkpoint_cursor: str) -> dict[str, Any]:
    os.makedirs(openable_path(snapshot_root), exist_ok=True)
    allowed = [
        "checkpoints/latest.pt",
        "checkpoints/latest.pt.sha256",
        "checkpoints/previous.pt",
        "checkpoints/previous.pt.sha256",
        "metrics_only_v3",
        "ensemble_oof_scores_v1",
        "ensemble_oof_scores_manifest_v1.json",
        "ensemble_oof_partition_ledger_v1.csv",
        "ensemble_oof_scores_manifest_v1.sha256",
        "logs",
        "authority",
    ]
    files: list[dict[str, Any]] = []
    for rel in allowed:
        source = run_root / rel
        if not source.exists():
            continue
        if source.is_dir():
            for path in source.rglob("*"):
                if not path.is_file() or path.name.endswith(".tmp") or "__pycache__" in path.parts:
                    continue
                if "data" in path.parts:
                    continue
                target_rel = path.relative_to(run_root)
                target = snapshot_root / target_rel
                os.makedirs(openable_path(target.parent), exist_ok=True)
                shutil.copy2(openable_path(path), openable_path(target))
                files.append({"relative_path": target_rel.as_posix(), "size_bytes": target.stat().st_size, "sha256": sha256_file(target)})
        elif source.is_file():
            target = snapshot_root / source.relative_to(run_root)
            os.makedirs(openable_path(target.parent), exist_ok=True)
            shutil.copy2(openable_path(source), openable_path(target))
            files.append({"relative_path": target.relative_to(snapshot_root).as_posix(), "size_bytes": target.stat().st_size, "sha256": sha256_file(target)})
    manifest = {
        "snapshot_id": "DS24_R44C_IMMUTABLE_SYNC_SNAPSHOT_V1",
        "checkpoint_cursor": checkpoint_cursor,
        "file_count": len(files),
        "total_bytes": int(sum(row["size_bytes"] for row in files)),
        "files": sorted(files, key=lambda row: row["relative_path"]),
        "excludes": ["raw feature data", "raw target data", "temporary files", "credentials", "environment files"],
    }
    manifest["snapshot_hash"] = stable_hash(manifest)
    write_json(snapshot_root / "sync_snapshot_manifest.json", manifest)
    return manifest


def verify_sync_snapshot(snapshot_root: Path) -> dict[str, Any]:
    manifest_path = snapshot_root / "sync_snapshot_manifest.json"
    if not manifest_path.exists():
        raise EnsembleOOFError("DS24_R44C_SYNC_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    mismatched = []
    for row in manifest.get("files", []):
        path = snapshot_root / row["relative_path"]
        if not path.exists():
            missing.append(row["relative_path"])
        elif sha256_file(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
    if missing or mismatched:
        raise EnsembleOOFError("DS24_R44C_SYNC_VERIFICATION_FAILED")
    return {"status": "PASS", "file_count": len(manifest.get("files", [])), "total_bytes": manifest.get("total_bytes", 0)}


def verify_downloaded_snapshot(
    snapshot_root: Path,
    *,
    expected_manifest_hash: str = "",
) -> dict[str, Any]:
    result = verify_sync_snapshot(snapshot_root)
    manifest = json.loads((snapshot_root / "sync_snapshot_manifest.json").read_text(encoding="utf-8"))
    if expected_manifest_hash and manifest.get("snapshot_hash") != expected_manifest_hash:
        raise EnsembleOOFError("DS24_R44C_SYNC_MANIFEST_HASH_MISMATCH")
    result["snapshot_hash"] = manifest.get("snapshot_hash", "")
    result["expected_manifest_hash"] = expected_manifest_hash
    return result


def import_verified_snapshot(
    snapshot_root: Path,
    import_root: Path,
    *,
    expected_snapshot_hash: str = "",
    free_bytes: int | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    verified = verify_downloaded_snapshot(snapshot_root, expected_manifest_hash=expected_snapshot_hash)
    if free_bytes is not None:
        gate = disk_admission(download_bytes=int(verified["total_bytes"]), free_bytes=int(free_bytes))
        if not gate["accepted"]:
            raise EnsembleOOFError("DS24_R44C_IMPORT_DISK_ADMISSION_REFUSED")
    destination = import_root / "reviewed_remote_vast_runs" / snapshot_root.name
    if publish and destination.exists():
        raise EnsembleOOFError("DS24_R44C_DUPLICATE_IMPORT_REJECTED")
    if publish:
        os.makedirs(openable_path(destination.parent), exist_ok=True)
        shutil.copytree(openable_path(snapshot_root), openable_path(destination))
    return {
        "status": "PASS",
        "publish_requested": publish,
        "publish_namespace": str(destination),
        "snapshot_hash": verified["snapshot_hash"],
        "file_count": verified["file_count"],
        "total_bytes": verified["total_bytes"],
        "live_namespace_modified": False,
        "requires_later_adoption": True,
    }


def resume_download_copy(source: Path, staging: Path) -> dict[str, Any]:
    os.makedirs(openable_path(staging.parent), exist_ok=True)
    source_size = source.stat().st_size
    existing_size = staging.stat().st_size if staging.exists() else 0
    mode = "append" if 0 < existing_size < source_size else "fresh"
    if existing_size > source_size:
        staging.unlink()
        existing_size = 0
        mode = "restart"
    with open(openable_path(source), "rb") as reader, open(openable_path(staging), "ab") as writer:
        reader.seek(existing_size)
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return {
        "status": "PASS",
        "mode": mode,
        "source_bytes": source_size,
        "previous_staging_bytes": existing_size,
        "final_staging_bytes": staging.stat().st_size,
        "sha256": sha256_file(staging),
    }


def disk_admission(
    *,
    download_bytes: int,
    free_bytes: int,
    extraction_or_staging_overhead: int = 0,
    hard_floor_bytes: int = 12 * 1024**3,
    safety_margin_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    required = int(download_bytes) + int(extraction_or_staging_overhead) + int(hard_floor_bytes) + int(safety_margin_bytes)
    accepted = int(free_bytes) >= required
    return {
        "required_free_bytes": required,
        "download_bytes": int(download_bytes),
        "extraction_or_staging_overhead": int(extraction_or_staging_overhead),
        "hard_floor_bytes": int(hard_floor_bytes),
        "safety_margin_bytes": int(safety_margin_bytes),
        "current_free_bytes": int(free_bytes),
        "post_import_free_space_estimate": int(free_bytes) - int(download_bytes) - int(extraction_or_staging_overhead),
        "accepted": accepted,
        "shortfall_bytes": 0 if accepted else required - int(free_bytes),
        "external_destination_supported": True,
    }


def reproduce_v3_metrics_from_scores(scores: pd.DataFrame, targets: pd.DataFrame, *, top_n: int = 20) -> dict[str, Any]:
    predictions = scores[["family", "decision_timestamp", "asset_id", "long_selection_score"]].rename(
        columns={"long_selection_score": "prediction"}
    )
    metrics, decisions = compute_per_t_metrics(predictions, targets, top_n=top_n)
    return {
        "status": "PASS",
        "rank_ic_rows": int(len(metrics)),
        "decision_rows": int(len(decisions)),
        "mean_spearman_rank_ic": float(metrics["spearman_rank_ic"].mean()) if len(metrics) else 0.0,
        "tolerance": "exact for synthetic fixture; production tolerance documented as 1e-9 for rank metrics and 1e-6 for returns",
    }


def future_family_ensemble_admission(family: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if family not in FAMILY_ENFORCEMENT_SCOPE:
        return {"family": family, "admitted": False, "classification": "UNKNOWN_FAMILY"}
    evidence = evidence or {}
    required = [
        "v3_metrics",
        "full_oof_score_contract",
        "score_lineage_hashes",
        "resume_safe_partition_ledgers",
        "importable_compact_result_package",
    ]
    missing = [name for name in required if not evidence.get(name)]
    if missing:
        return {
            "family": family,
            "admitted": False,
            "classification": "ENSEMBLE_OUTPUT_CERTIFICATION_REQUIRED",
            "missing": missing,
        }
    return {"family": family, "admitted": True, "classification": "ENSEMBLE_OUTPUT_CERTIFIED"}


def validate_compact_oof_v2_frame(frame: pd.DataFrame) -> dict[str, Any]:
    _reject_forbidden_columns(frame.columns)
    if list(frame.columns) != COMPACT_OOF_COLUMNS:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_SCHEMA_DRIFT")
    if frame.duplicated(["decision_timestamp", "asset_id"]).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_DUPLICATE_KEY")
    ts = pd.to_datetime(frame["decision_timestamp"], utc=True)
    if ts.isna().any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_BAD_TIMESTAMP")
    if frame["asset_id"].astype(str).str.len().eq(0).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_EMPTY_ASSET")
    if not pd.to_numeric(frame["long_selection_score"], errors="coerce").map(math.isfinite).all():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_NONFINITE_SCORE")
    ranks = pd.to_numeric(frame["cross_sectional_rank"], errors="coerce")
    eligible = pd.to_numeric(frame["eligible_assets"], errors="coerce")
    if ((ranks < 1) | (ranks > eligible)).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_RANK_RANGE_VIOLATION")
    if (eligible < 1).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_ELIGIBLE_RANGE_VIOLATION")
    pct = pd.to_numeric(frame["cross_sectional_percentile"], errors="coerce")
    if ((pct < 0.0) | (pct > 1.0) | ~pct.map(math.isfinite)).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_PERCENTILE_RANGE_VIOLATION")
    refits = pd.to_numeric(frame["refit_ordinal"], errors="coerce")
    if refits.isna().any() or (refits < 0).any():
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_REFIT_RANGE_VIOLATION")
    return {
        "valid": True,
        "contract_id": COMPACT_SCORE_CONTRACT_ID,
        "row_count": int(len(frame)),
        "distinct_timestamps": int(ts.nunique()),
        "distinct_assets": int(frame["asset_id"].nunique()),
        "first_timestamp": ts.min().isoformat() if len(ts) else "",
        "last_timestamp": ts.max().isoformat() if len(ts) else "",
        "distinct_refits": int(refits.nunique()),
    }


def compact_oof_v2_metadata_from_v1(
    frame: pd.DataFrame,
    *,
    source_bundle_hash: str = "",
    terminal_completeness_state: str = "SYNTHETIC_CONTRACT_COMPLETE",
    provisional: bool = True,
) -> dict[str, Any]:
    validate_oof_frame(frame)
    static_columns = [
        "trial_id",
        "run_id",
        "family",
        "model_config_hash",
        "dataset_manifest_hash",
        "predictor_contract_hash",
        "target_contract_hash",
        "evaluation_contract_hash",
        "score_contract_version",
    ]
    static: dict[str, Any] = {}
    for column in static_columns:
        values = sorted({str(value) for value in frame[column].dropna().unique()})
        if len(values) != 1:
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_MIXED_STATIC_LINEAGE")
        static[column] = values[0]
    static["source_score_contract_version"] = static.pop("score_contract_version")
    refit_lineage: list[dict[str, Any]] = []
    for ordinal, part in frame.groupby("refit_ordinal", sort=True):
        refit_ids = sorted({str(value) for value in part["refit_id"].dropna().unique()})
        cutoffs = sorted(
            {
                pd.Timestamp(value).isoformat()
                for value in pd.to_datetime(part["training_cutoff_timestamp"], utc=True).unique()
            }
        )
        if len(refit_ids) != 1 or len(cutoffs) != 1:
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_MIXED_REFIT_LINEAGE")
        refit_lineage.append(
            {
                "refit_ordinal": int(ordinal),
                "refit_id": refit_ids[0],
                "training_cutoff_timestamp": cutoffs[0],
            }
        )
    metadata = {
        "contract_id": COMPACT_SCORE_CONTRACT_ID,
        "source_contract_id": SCORE_CONTRACT_ID,
        "compact_score_contract_version": COMPACT_SCORE_CONTRACT_VERSION,
        "static_lineage": static,
        "refit_lineage": refit_lineage,
        "source_bundle_hash": source_bundle_hash,
        "terminal_completeness_state": terminal_completeness_state,
        "provisional": bool(provisional),
    }
    metadata["metadata_hash"] = stable_hash(metadata)
    return metadata


def to_compact_oof_v2_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_oof_frame(frame)
    compact = frame[COMPACT_OOF_COLUMNS].copy()
    compact["decision_timestamp"] = pd.to_datetime(compact["decision_timestamp"], utc=True)
    compact["asset_id"] = compact["asset_id"].astype(str)
    compact["long_selection_score"] = pd.to_numeric(compact["long_selection_score"], errors="coerce").astype("float32")
    compact["cross_sectional_rank"] = pd.to_numeric(
        compact["cross_sectional_rank"], errors="coerce"
    ).astype("int32")
    compact["cross_sectional_percentile"] = pd.to_numeric(
        compact["cross_sectional_percentile"], errors="coerce"
    ).astype("float32")
    compact["eligible_assets"] = pd.to_numeric(compact["eligible_assets"], errors="coerce").astype("int32")
    compact["refit_ordinal"] = pd.to_numeric(compact["refit_ordinal"], errors="coerce").astype("int32")
    compact = compact.sort_values(["decision_timestamp", "asset_id"]).reset_index(drop=True)
    validate_compact_oof_v2_frame(compact)
    return compact


def _compact_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("decision_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("asset_id", pa.string()),
            pa.field("long_selection_score", pa.float32()),
            pa.field("cross_sectional_rank", pa.int32()),
            pa.field("cross_sectional_percentile", pa.float32()),
            pa.field("eligible_assets", pa.int32()),
            pa.field("refit_ordinal", pa.int32()),
        ]
    )


def write_compact_oof_v2_partitions(
    run_root: Path,
    frame: pd.DataFrame,
    *,
    allow_existing_same_hash: bool = True,
) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    compact = to_compact_oof_v2_frame(frame) if list(frame.columns) != COMPACT_OOF_COLUMNS else frame.copy()
    validate_compact_oof_v2_frame(compact)
    working = compact.copy()
    working["decision_date"] = pd.to_datetime(working["decision_timestamp"], utc=True).dt.date
    ledger: list[dict[str, Any]] = []
    for (decision_date, ordinal), part in working.groupby(["decision_date", "refit_ordinal"], sort=True):
        rel = Path(f"ensemble_oof_scores_v2/decision_date={decision_date}/part-refit={int(ordinal):06d}.parquet")
        path = run_root / rel
        os.makedirs(openable_path(path.parent), exist_ok=True)
        table = pa.Table.from_pandas(part[COMPACT_OOF_COLUMNS], schema=_compact_arrow_schema(), preserve_index=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as handle:
            tmp = Path(handle.name)
        pq.write_table(
            table,
            openable_path(tmp),
            compression="zstd",
            use_dictionary=["asset_id"],
        )
        new_hash = sha256_file(tmp)
        if os.path.exists(openable_path(path)):
            old_hash = sha256_file(path)
            if old_hash != new_hash or not allow_existing_same_hash:
                tmp.unlink(missing_ok=True)
                raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_IMMUTABLE_PARTITION_HASH_MISMATCH")
            tmp.unlink(missing_ok=True)
        else:
            os.replace(openable_path(tmp), openable_path(path))
        ledger.append(
            {
                "relative_path": rel.as_posix(),
                "decision_date": str(decision_date),
                "refit_ordinal": int(ordinal),
                "row_count": int(len(part)),
                "first_decision_timestamp": pd.to_datetime(part["decision_timestamp"], utc=True).min().isoformat(),
                "last_decision_timestamp": pd.to_datetime(part["decision_timestamp"], utc=True).max().isoformat(),
                "compressed_bytes": os.stat(openable_path(path)).st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(ledger, key=lambda row: row["relative_path"])


def read_compact_oof_v2_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(openable_path(path))
    if list(frame.columns) != COMPACT_OOF_COLUMNS:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_SCHEMA_DRIFT")
    return frame


def _compact_static_lineage(payload: Mapping[str, Any]) -> dict[str, Any]:
    static = payload.get("static_lineage", payload)
    return dict(static) if isinstance(static, Mapping) else {}


def _compact_refit_lineage_by_ordinal(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    lineage = payload.get("refit_lineage", [])
    return {
        int(row["refit_ordinal"]): dict(row)
        for row in lineage
        if isinstance(row, Mapping) and str(row.get("refit_ordinal", "")).strip() != ""
    }


def expand_compact_oof_v2_to_v1(frame: pd.DataFrame, metadata_or_manifest: Mapping[str, Any]) -> pd.DataFrame:
    compact = frame.copy()
    validate_compact_oof_v2_frame(compact)
    static = _compact_static_lineage(metadata_or_manifest)
    refit_lineage = _compact_refit_lineage_by_ordinal(metadata_or_manifest)
    missing_refits = sorted(set(int(value) for value in compact["refit_ordinal"].unique()) - set(refit_lineage))
    if missing_refits:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_REFIT_LINEAGE_MISSING")
    expanded = compact.copy()
    expanded["decision_timestamp"] = pd.to_datetime(expanded["decision_timestamp"], utc=True)
    expanded["decision_date"] = expanded["decision_timestamp"].dt.date
    expanded["trial_id"] = static.get("trial_id", "")
    expanded["run_id"] = static.get("run_id", "")
    expanded["family"] = static.get("family", "")
    expanded["training_cutoff_timestamp"] = expanded["refit_ordinal"].map(
        lambda ordinal: refit_lineage[int(ordinal)]["training_cutoff_timestamp"]
    )
    expanded["refit_id"] = expanded["refit_ordinal"].map(lambda ordinal: refit_lineage[int(ordinal)]["refit_id"])
    expanded["model_config_hash"] = static.get("model_config_hash", "")
    expanded["dataset_manifest_hash"] = static.get("dataset_manifest_hash", "")
    expanded["predictor_contract_hash"] = static.get("predictor_contract_hash", "")
    expanded["target_contract_hash"] = static.get("target_contract_hash", "")
    expanded["evaluation_contract_hash"] = static.get("evaluation_contract_hash", "")
    expanded["score_contract_version"] = static.get("source_score_contract_version", SCORE_CONTRACT_VERSION)
    out = expanded[ENSEMBLE_OOF_COLUMNS].sort_values(["decision_timestamp", "asset_id"]).reset_index(drop=True)
    validate_oof_frame(out)
    return out


def build_compact_oof_v2_manifest(
    run_root: Path,
    ledger: list[dict[str, Any]],
    *,
    metadata: Mapping[str, Any],
    terminal_completeness_state: str | None = None,
    latest_completed_refit_ordinal: int | None = None,
    provisional: bool | None = None,
) -> dict[str, Any]:
    frames = [read_compact_oof_v2_partition(run_root / row["relative_path"]) for row in ledger]
    all_scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COMPACT_OOF_COLUMNS)
    audit = validate_compact_oof_v2_frame(all_scores) if not all_scores.empty else {"valid": True, "row_count": 0}
    static = _compact_static_lineage(metadata)
    refit_lineage = list(metadata.get("refit_lineage", []))
    manifest = {
        "manifest_id": "ensemble_oof_scores_manifest_v2",
        "score_contract_id": COMPACT_SCORE_CONTRACT_ID,
        "score_contract_version": COMPACT_SCORE_CONTRACT_VERSION,
        "source_score_contract_id": SCORE_CONTRACT_ID,
        "source_score_contract_version": static.get("source_score_contract_version", SCORE_CONTRACT_VERSION),
        "schema_version": 2,
        "run_id": static.get("run_id", ""),
        "trial_id": static.get("trial_id", ""),
        "family": static.get("family", ""),
        "row_count": int(len(all_scores)),
        "partition_count": len(ledger),
        "first_decision_timestamp": audit.get("first_timestamp", ""),
        "last_decision_timestamp": audit.get("last_timestamp", ""),
        "distinct_decision_timestamps": int(all_scores["decision_timestamp"].nunique()) if not all_scores.empty else 0,
        "distinct_assets": int(all_scores["asset_id"].nunique()) if not all_scores.empty else 0,
        "source_configuration_hash": static.get("model_config_hash", ""),
        "source_bundle_hash": metadata.get("source_bundle_hash", ""),
        "data_manifest_hash": static.get("dataset_manifest_hash", ""),
        "predictor_contract_hash": static.get("predictor_contract_hash", ""),
        "target_contract_hash": static.get("target_contract_hash", ""),
        "evaluation_contract_hash": static.get("evaluation_contract_hash", ""),
        "score_orientation": "higher long_selection_score is better; static lineage stored in manifest",
        "compression_codec": "zstd",
        "row_columns": list(COMPACT_OOF_COLUMNS),
        "static_lineage": static,
        "refit_lineage": refit_lineage,
        "metadata_hash": metadata.get("metadata_hash", stable_hash(dict(metadata))),
        "total_compressed_bytes": int(sum(row["compressed_bytes"] for row in ledger)),
        "files": ledger,
        "terminal_completeness_state": terminal_completeness_state
        if terminal_completeness_state is not None
        else metadata.get("terminal_completeness_state", ""),
        "latest_successfully_completed_refit_ordinal": int(
            latest_completed_refit_ordinal
            if latest_completed_refit_ordinal is not None
            else max([int(row.get("refit_ordinal", 0)) for row in refit_lineage] or [0])
        ),
        "provisional": bool(provisional if provisional is not None else metadata.get("provisional", True)),
    }
    if not all_scores.empty:
        expand_compact_oof_v2_to_v1(all_scores, manifest)
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def publish_compact_oof_v2_manifest(run_root: Path, manifest: Mapping[str, Any]) -> None:
    write_json(run_root / "ensemble_oof_scores_manifest_v2.json", dict(manifest))
    digest = stable_hash(dict(manifest))
    os.makedirs(openable_path(run_root), exist_ok=True)
    with open(openable_path(run_root / "ensemble_oof_scores_manifest_v2.sha256"), "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    write_csv(run_root / "ensemble_oof_partition_ledger_v2.csv", list(manifest.get("files", [])))


def validate_compact_oof_v2_manifest(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("score_contract_id") != COMPACT_SCORE_CONTRACT_ID:
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_MANIFEST_CONTRACT_MISMATCH")
    if manifest.get("family") not in set(FAMILY_ENFORCEMENT_SCOPE):
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_UNKNOWN_FAMILY")
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for row in manifest.get("files", []):
        rel = str(row.get("relative_path", ""))
        if not rel or rel in seen:
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_DUPLICATE_PARTITION")
        seen.add(rel)
        path = run_root / rel
        if not os.path.exists(openable_path(path)):
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_MISSING_PARTITION")
        if sha256_file(path) != row.get("sha256"):
            raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_PARTITION_HASH_MISMATCH")
        frames.append(read_compact_oof_v2_partition(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COMPACT_OOF_COLUMNS)
    audit = validate_compact_oof_v2_frame(frame) if not frame.empty else {"valid": True, "row_count": 0}
    if int(manifest.get("row_count", -1)) != int(audit["row_count"]):
        raise EnsembleOOFError("DS24_ENSEMBLE_OOF_COMPACT_V2_MANIFEST_ROW_COUNT_MISMATCH")
    expanded = expand_compact_oof_v2_to_v1(frame, manifest) if not frame.empty else pd.DataFrame(columns=ENSEMBLE_OOF_COLUMNS)
    return {
        "valid": True,
        "row_count": int(audit["row_count"]),
        "expanded_v1_row_count": int(len(expanded)),
        "partition_count": len(seen),
        "total_compressed_bytes": int(sum(int(row.get("compressed_bytes", 0)) for row in manifest.get("files", []))),
        "metadata_hash": manifest.get("metadata_hash", ""),
    }
