"""Read-only reconstruction of the DS24 RF 2019-04-08 13:40 ledger batch."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
)
from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData
from core.research.ml.ds24.prediction_ledger_contract import PredictionProvenance
from infrastructure.data.ds24_prediction_ledger import PredictionLedgerStore
from scripts.local.ds24_clean_v2_family_worker import (
    ROOT,
    _completed_timestamps,
    _prediction_batches_by_timestamp,
    _restore_model_artifact,
    _score_model,
    _scoring_timestamp_batches,
)


RUN_ID = "DS24_CLEAN_V2_TOURNAMENT_R1_20260926"
TIMESTAMP = pd.Timestamp("2019-04-08T13:40:00Z")
RUN_ROOT = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID
PART = RUN_ROOT / (
    "prediction_ledger_v1/family=random_forest/period=2019-04/"
    "part-20190408T133500Z-20190408T190000Z-ffcee8b95df79c60.parquet"
)
CACHE = RUN_ROOT / (
    "shared_panel_package_cache/source=46088c818989f4af/"
    "89ff1b7145e3a93f9a348e5a5815e432.parquet"
)
CHECKPOINT = RUN_ROOT / "family=random_forest/model_checkpoint.json"
OUTPUT = Path(__file__).with_name("forensic_comparison.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    model_path = ROOT / checkpoint["model_path"]
    if _sha256(model_path) != checkpoint["model_hash"]:
        raise RuntimeError("Saved RF model hash changed")
    with model_path.open("rb") as handle:
        artifact = pickle.load(handle)
    model = _restore_model_artifact(
        artifact,
        expected_identity=checkpoint["identity"],
    )

    data = CleanV2CompositeData(ROOT)
    panel = pd.read_parquet(CACHE)
    panel_timestamps = pd.to_datetime(panel["decision_timestamp"], utc=True)
    session_timestamps = tuple(
        sorted(panel_timestamps[panel_timestamps.dt.date == TIMESTAMP.date()].unique())
    )
    completed = _completed_timestamps(
        RUN_ROOT / "family=random_forest/metrics", "random_forest"
    )
    timestamp_batches = _scoring_timestamp_batches(
        "random_forest", session_timestamps, completed
    )
    failed_batch = next(batch for batch in timestamp_batches if TIMESTAMP in batch)
    score_panel = panel[panel_timestamps.isin(failed_batch)].copy()
    predictions = _score_model(
        "random_forest",
        model,
        score_panel,
        panel,
        data.predictors,
        checkpoint["fit_metadata"],
    )

    scientific = checkpoint["identity"]["scientific_identity"]
    refit_timestamp = pd.Timestamp(
        scientific["package_contract"]["refit_T"]
    )
    refit_id = stable_hash(
        {
            "family": "random_forest",
            "refit_T": refit_timestamp.isoformat(),
            "model_hash": checkpoint["model_hash"],
        }
    )
    existing_all = pd.read_parquet(PART)
    existing = existing_all[
        pd.to_datetime(existing_all["decision_timestamp"], utc=True) == TIMESTAMP
    ].copy()
    first = existing.iloc[0]
    publisher = DS24PredictionLedgerPublisher(
        store=PredictionLedgerStore(Path("unused-read-only")),
        provenance=PredictionProvenance(
            source_hash=str(first["source_hash"]),
            feature_authority_hash=str(first["feature_authority_hash"]),
            target_authority_hash=str(first["target_authority_hash"]),
            static_bundle_hash=str(first["static_bundle_hash"]),
            run_id=str(first["run_id"]),
        ),
    )
    candidate_batches = _prediction_batches_by_timestamp(predictions, failed_batch)
    incoming_predictions = candidate_batches[failed_batch.index(TIMESTAMP)]
    incoming = publisher._prepare_batch(
        incoming_predictions,
        family="random_forest",
        refit_id=refit_id,
        refit_timestamp=refit_timestamp,
    )

    key = ["family", "decision_timestamp", "asset_id"]
    merged = existing.merge(
        incoming,
        on=key,
        how="outer",
        suffixes=("_existing", "_incoming"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    absolute_delta = (
        both["raw_prediction_existing"] - both["raw_prediction_incoming"]
    ).abs()
    denominator = np.maximum(
        both["raw_prediction_existing"].abs().to_numpy(),
        both["raw_prediction_incoming"].abs().to_numpy(),
    )
    relative_delta = np.divide(
        absolute_delta.to_numpy(),
        denominator,
        out=np.zeros(len(both), dtype=float),
        where=denominator != 0,
    )
    comparison_columns = [
        "raw_prediction",
        "cross_section_rank",
        "rank_percentile",
        "eligible_universe_size",
        "refit_id",
        "refit_timestamp",
        "evaluation_eligible",
        "feature_authority_hash",
        "target_authority_hash",
        "static_bundle_hash",
        "run_id",
    ]
    disagreement_masks: dict[str, pd.Series] = {}
    for column in comparison_columns:
        left = both[f"{column}_existing"]
        right = both[f"{column}_incoming"]
        if column == "refit_timestamp":
            left = pd.to_datetime(left, utc=True)
            right = pd.to_datetime(right, utc=True)
        disagreement_masks[column] = ~(left.eq(right) | (left.isna() & right.isna()))
    any_disagreement = pd.concat(disagreement_masks, axis=1).any(axis=1)
    sample_columns = ["asset_id"]
    for column in comparison_columns:
        if disagreement_masks[column].any():
            sample_columns.extend([f"{column}_existing", f"{column}_incoming"])
    sample = both.loc[any_disagreement, sample_columns].head(10).copy()
    for column in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[column]):
            sample[column] = sample[column].astype(str)

    report = {
        "classification": (
            "RF_IMMUTABLE_LEDGER_EXACT_IDEMPOTENT_REPLAY"
            if (merged["_merge"] == "both").all() and not any_disagreement.any()
            else "RF_IMMUTABLE_LEDGER_GENUINE_SCIENTIFIC_CONFLICT"
        ),
        "timestamp": TIMESTAMP.isoformat(),
        "failed_scoring_batch_first_timestamp": failed_batch[0].isoformat(),
        "failed_scoring_batch_last_timestamp": failed_batch[-1].isoformat(),
        "failed_scoring_batch_timestamp_count": len(failed_batch),
        "existing_part": PART.relative_to(ROOT).as_posix(),
        "existing_part_sha256": _sha256(PART),
        "cache_path": CACHE.relative_to(ROOT).as_posix(),
        "cache_sha256": _sha256(CACHE),
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "model_sha256": _sha256(model_path),
        "model_vintage_id": refit_id,
        "existing_row_count": int(len(existing)),
        "incoming_row_count": int(len(incoming)),
        "intersecting_key_count": int((merged["_merge"] == "both").sum()),
        "existing_only_key_count": int((merged["_merge"] == "left_only").sum()),
        "incoming_only_key_count": int((merged["_merge"] == "right_only").sum()),
        "exact_payload_match_count": int((~any_disagreement).sum()),
        "conflicting_payload_count": int(any_disagreement.sum()),
        "max_absolute_prediction_delta": float(absolute_delta.max()),
        "max_relative_prediction_delta": float(relative_delta.max()),
        "field_disagreement_counts": {
            column: int(mask.sum())
            for column, mask in disagreement_masks.items()
        },
        "bounded_conflict_sample": sample.to_dict(orient="records"),
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
