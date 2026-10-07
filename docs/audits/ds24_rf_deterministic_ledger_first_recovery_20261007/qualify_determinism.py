"""Qualify deterministic RF scoring against the frozen production failure package."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
)
from core.research.ml.ds24.clean_v2_contracts import clean_source_hash, stable_hash
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData
from core.research.ml.ds24.prediction_ledger_contract import PredictionProvenance
from infrastructure.data.ds24_prediction_ledger import PredictionLedgerStore
from scripts.local.ds24_clean_v2_family_worker import (
    DETERMINISTIC_RANDOM_FOREST_PREDICT_N_JOBS,
    QUALIFIED_RANDOM_FOREST_N_JOBS,
    _completed_timestamps,
    _prediction_batches_by_timestamp,
    _restore_model_artifact,
    _score_model,
    _scoring_timestamp_batches,
)


RUN_ID = "DS24_CLEAN_V2_TOURNAMENT_R1_20260926"
FAILURE_TIMESTAMP = pd.Timestamp("2019-04-08T13:40:00Z")
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
OUTPUT = Path(__file__).with_name("determinism_qualification.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: dict[str, object]) -> object:
    model_path = ROOT / str(checkpoint["model_path"])
    if _sha256(model_path) != checkpoint["model_hash"]:
        raise RuntimeError("Saved RF model hash changed")
    with model_path.open("rb") as handle:
        artifact = pickle.load(handle)
    return _restore_model_artifact(
        artifact,
        expected_identity=checkpoint["identity"],  # type: ignore[arg-type]
    )


def _prediction_identity(frame: pd.DataFrame) -> tuple[bytes, tuple[tuple[str, str], ...]]:
    values = frame["prediction"].to_numpy(dtype=np.float64).tobytes()
    keys = tuple(
        zip(
            frame["decision_timestamp"].astype(str),
            frame["asset_id"].astype(str),
            strict=True,
        )
    )
    return values, keys


def _rank_identity(frame: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    work = frame.copy()
    work["rank"] = work.groupby("decision_timestamp", sort=False)["prediction"].rank(
        method="first", ascending=False
    )
    return tuple(
        (str(row.decision_timestamp), str(row.asset_id), int(row.rank))
        for row in work.itertuples(index=False)
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    model = _load_model(checkpoint)
    estimator = model.steps[-1][1]
    if estimator.n_jobs != QUALIFIED_RANDOM_FOREST_N_JOBS:
        raise RuntimeError("Saved RF fit parallelism is not n_jobs=3")
    panel = pd.read_parquet(CACHE)
    panel_timestamps = pd.to_datetime(panel["decision_timestamp"], utc=True)
    session_timestamps = tuple(
        sorted(
            panel_timestamps[
                panel_timestamps.dt.date == FAILURE_TIMESTAMP.date()
            ].unique()
        )
    )
    completed = _completed_timestamps(
        RUN_ROOT / "family=random_forest/metrics", "random_forest"
    )
    failed_batch = next(
        batch
        for batch in _scoring_timestamp_batches(
            "random_forest", session_timestamps, completed
        )
        if FAILURE_TIMESTAMP in batch
    )
    score_panel = panel[panel_timestamps.isin(failed_batch)].copy()
    predictors = CleanV2CompositeData(ROOT).predictors

    started = time.perf_counter()
    baseline = _score_model(
        "random_forest", model, score_panel, panel, predictors, checkpoint["fit_metadata"]
    )
    deterministic_first_wall = time.perf_counter() - started
    baseline_values, baseline_keys = _prediction_identity(baseline)
    baseline_ranks = _rank_identity(baseline)
    bitwise_disagreements = 0
    key_disagreements = 0
    rank_disagreements = 0
    in_memory_seconds: list[float] = []
    for _ in range(100):
        started = time.perf_counter()
        observed = _score_model(
            "random_forest", model, score_panel, panel, predictors, checkpoint["fit_metadata"]
        )
        in_memory_seconds.append(time.perf_counter() - started)
        values, keys = _prediction_identity(observed)
        bitwise_disagreements += int(values != baseline_values)
        key_disagreements += int(keys != baseline_keys)
        rank_disagreements += int(_rank_identity(observed) != baseline_ranks)
        if estimator.n_jobs != QUALIFIED_RANDOM_FOREST_N_JOBS:
            raise RuntimeError("RF fit n_jobs was not restored after prediction")

    reload_seconds: list[float] = []
    for _ in range(50):
        reloaded = _load_model(checkpoint)
        started = time.perf_counter()
        observed = _score_model(
            "random_forest", reloaded, score_panel, panel, predictors, checkpoint["fit_metadata"]
        )
        reload_seconds.append(time.perf_counter() - started)
        values, keys = _prediction_identity(observed)
        bitwise_disagreements += int(values != baseline_values)
        key_disagreements += int(keys != baseline_keys)
        rank_disagreements += int(_rank_identity(observed) != baseline_ranks)
        if reloaded.steps[-1][1].n_jobs != QUALIFIED_RANDOM_FOREST_N_JOBS:
            raise RuntimeError("Reloaded RF fit n_jobs was not restored")

    scoring_matrix = score_panel.loc[:, list(predictors)]
    transformed = model[:-1].transform(scoring_matrix)
    estimator.set_params(n_jobs=QUALIFIED_RANDOM_FOREST_N_JOBS)
    started = time.perf_counter()
    parallel_values = estimator.predict(transformed)
    parallel_wall = time.perf_counter() - started
    estimator.set_params(n_jobs=QUALIFIED_RANDOM_FOREST_N_JOBS)

    scientific = checkpoint["identity"]["scientific_identity"]
    refit_timestamp = pd.Timestamp(scientific["package_contract"]["refit_T"])
    refit_id = stable_hash(
        {
            "family": "random_forest",
            "refit_T": refit_timestamp.isoformat(),
            "model_hash": checkpoint["model_hash"],
        }
    )
    existing_all = pd.read_parquet(PART)
    existing = existing_all.loc[
        pd.to_datetime(existing_all["decision_timestamp"], utc=True)
        == FAILURE_TIMESTAMP
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
    incoming_batch = _prediction_batches_by_timestamp(
        baseline, failed_batch
    )[failed_batch.index(FAILURE_TIMESTAMP)]
    incoming = publisher._prepare_batch(
        incoming_batch,
        family="random_forest",
        refit_id=refit_id,
        refit_timestamp=refit_timestamp,
    )
    merged = existing.merge(
        incoming,
        on=["family", "decision_timestamp", "asset_id"],
        suffixes=("_existing", "_incoming"),
        how="outer",
        indicator=True,
    )
    both = merged.loc[merged["_merge"] == "both"].copy()
    delta = (both["raw_prediction_existing"] - both["raw_prediction_incoming"]).abs()
    denominator = np.maximum(
        both["raw_prediction_existing"].abs().to_numpy(),
        both["raw_prediction_incoming"].abs().to_numpy(),
    )
    relative = np.divide(
        delta.to_numpy(),
        denominator,
        out=np.zeros(len(both), dtype=float),
        where=denominator != 0,
    )
    exact = both["raw_prediction_existing"].eq(both["raw_prediction_incoming"])
    report = {
        "classification": (
            "RF_DETERMINISTIC_SCORING_OFFLINE_QUALIFIED"
            if bitwise_disagreements == key_disagreements == rank_disagreements == 0
            else "RF_DETERMINISTIC_SCORING_OFFLINE_FAILED"
        ),
        "normalized_source_hash": clean_source_hash(),
        "model_sha256": checkpoint["model_hash"],
        "cache_sha256": _sha256(CACHE),
        "ledger_part_sha256": _sha256(PART),
        "fit_n_jobs": estimator.n_jobs,
        "predict_n_jobs": DETERMINISTIC_RANDOM_FOREST_PREDICT_N_JOBS,
        "population_rows": len(baseline),
        "timestamp_count": len(failed_batch),
        "in_memory_repeat_count": 100,
        "reload_repeat_count": 50,
        "bitwise_prediction_disagreements": bitwise_disagreements,
        "key_order_disagreements": key_disagreements,
        "rank_disagreements": rank_disagreements,
        "deterministic_first_wall_seconds": deterministic_first_wall,
        "deterministic_mean_wall_seconds": float(np.mean(in_memory_seconds)),
        "reload_prediction_mean_wall_seconds": float(np.mean(reload_seconds)),
        "parallel_reference_wall_seconds": parallel_wall,
        "parallel_reference_differs_from_deterministic": bool(
            not np.array_equal(parallel_values, baseline["prediction"].to_numpy())
        ),
        "april_8_existing_rows": len(existing),
        "april_8_incoming_rows": len(incoming),
        "april_8_intersecting_keys": int((merged["_merge"] == "both").sum()),
        "april_8_existing_only_keys": int((merged["_merge"] == "left_only").sum()),
        "april_8_incoming_only_keys": int((merged["_merge"] == "right_only").sum()),
        "april_8_exact_prediction_matches": int(exact.sum()),
        "april_8_max_absolute_delta": float(delta.max()),
        "april_8_max_relative_delta": float(relative.max()),
        "april_8_rank_disagreements": int(
            (both["cross_section_rank_existing"] != both["cross_section_rank_incoming"]).sum()
        ),
        "april_8_eligibility_disagreements": int(
            (both["evaluation_eligible_existing"] != both["evaluation_eligible_incoming"]).sum()
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["classification"] != "RF_DETERMINISTIC_SCORING_OFFLINE_QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
