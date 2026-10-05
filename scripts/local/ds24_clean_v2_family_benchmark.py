from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import pickle
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# Match the admitted Dell worker environment before importing numerical
# libraries. The benchmark must not silently measure a different native-thread
# policy from the production worker it is intended to characterize.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    authority_bundle,
    clean_source_hash,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_data import (  # noqa: E402
    CleanV2CompositeData,
    DataAssemblyProfiler,
)
from core.research.ml.ds24.clean_v2_package_cache import (  # noqa: E402
    PanelPackageKey,
    SharedPanelPackageCache,
)
from core.research.ml.ds24.clean_v2_runtime import QUALIFIER_YEARS  # noqa: E402
from core.research.ml.ds24_metrics_only_evaluator import (  # noqa: E402
    MetricsOnlyEvidenceWriter,
    read_parquet_log,
    resolved_performance_contract_v3_hash,
)
from scripts.local.ds24_clean_v2_family_worker import (  # noqa: E402
    CONTROL_FAMILIES,
    SHORT_LANES,
    _family_contract,
    _fit_model,
    _model_artifact_payload,
    _package_assembly_dates,
    _restore_model_artifact,
    _score_model,
    _scoring_timestamp_batches,
    build_clean_refit_schedule,
)


DELL_FAMILIES = (
    "random_forest",
    "ridge_C5",
    "elastic_net_C5",
    "elastic_net_C6",
    "huber",
    "gradient_boosting_C0",
    "gradient_boosting_C0_W20",
    "gradient_boosting_C0_W40",
    "gradient_boosting_C0_W80",
    "transformer",
    "momentum",
    "equal_weight_no_model",
)
BENCHMARK_VERSION = "DS24_CLEAN_V2_FAMILY_BENCHMARK_V1"


@dataclass(frozen=True)
class StageMeasurement:
    wall_seconds: float
    cpu_seconds: float
    mean_cpu_percent: float
    peak_resident_bytes: int
    peak_private_commit_bytes: int

    def payload(self) -> dict[str, float | int]:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "mean_cpu_percent": self.mean_cpu_percent,
            "peak_resident_bytes": self.peak_resident_bytes,
            "peak_private_commit_bytes": self.peak_private_commit_bytes,
        }


class ProcessHighWaterSampler:
    """Sample one benchmark process without affecting tournament workers."""

    def __init__(self, *, poll_seconds: float = 0.02) -> None:
        self.process = psutil.Process(os.getpid())
        self.poll_seconds = float(poll_seconds)
        self.peak_resident_bytes = 0
        self.peak_private_commit_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _private_commit(memory: Any) -> int:
        return int(
            getattr(memory, "private", getattr(memory, "private_usage", memory.vms))
        )

    def _sample(self) -> None:
        try:
            memory = self.process.memory_info()
        except (psutil.Error, OSError):
            return
        self.peak_resident_bytes = max(self.peak_resident_bytes, int(memory.rss))
        self.peak_private_commit_bytes = max(
            self.peak_private_commit_bytes, self._private_commit(memory)
        )

    def __enter__(self) -> ProcessHighWaterSampler:
        self._sample()

        def sample_until_stopped() -> None:
            while not self._stop.wait(self.poll_seconds):
                self._sample()

        self._thread = threading.Thread(target=sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 4))
        self._sample()


def _cpu_seconds(process: psutil.Process) -> float:
    values = process.cpu_times()
    return float(values.user + values.system)


def _measure(operation: Callable[[], Any]) -> tuple[Any, StageMeasurement]:
    process = psutil.Process(os.getpid())
    cpu_before = _cpu_seconds(process)
    wall_before = time.perf_counter()
    with ProcessHighWaterSampler() as sampler:
        result = operation()
    wall = time.perf_counter() - wall_before
    cpu = max(0.0, _cpu_seconds(process) - cpu_before)
    measurement = StageMeasurement(
        wall_seconds=float(wall),
        cpu_seconds=float(cpu),
        mean_cpu_percent=float(100.0 * cpu / wall) if wall > 0 else 0.0,
        peak_resident_bytes=sampler.peak_resident_bytes,
        peak_private_commit_bytes=sampler.peak_private_commit_bytes,
    )
    return result, measurement


def _population_slices(
    panel: pd.DataFrame,
    package: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = panel[
        panel["session_date"].astype(str).isin(package.training_session_dates)
        & (panel["decision_timestamp"] < package.refit_T)
        & panel["target_is_trainable"].fillna(False).astype(bool)
        & (panel["target_available_timestamp"] <= package.refit_T)
        & np.isfinite(pd.to_numeric(panel["target_value"], errors="coerce"))
    ]
    scoring = panel[
        panel["session_date"].astype(str).isin(package.score_session_dates)
    ]
    return training, scoring


def _family_contract_with_runtime_overrides(
    family: str,
    *,
    random_forest_n_jobs: int | None,
) -> tuple[dict[str, Any], str]:
    if random_forest_n_jobs is not None:
        if family != "random_forest":
            raise ValueError("random_forest_n_jobs is valid only for random_forest")
        if random_forest_n_jobs not in {1, 2, 3}:
            raise ValueError("random_forest_n_jobs must be one of 1, 2, or 3")
    config, lane = _family_contract(family)
    resolved = deepcopy(config)
    if random_forest_n_jobs is not None:
        resolved["parameters"]["n_jobs"] = random_forest_n_jobs
    return resolved, lane


def _model_round_trip(
    model: Any,
    *,
    family: str,
) -> tuple[Any, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="ds24-family-benchmark-") as directory:
        path = Path(directory) / f"{family}.pkl"
        identity = {"family": family, "purpose": "bounded-family-benchmark"}
        save_started = time.perf_counter()
        with path.open("wb") as handle:
            pickle.dump(
                _model_artifact_payload(model=model, identity=identity),
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        save_seconds = time.perf_counter() - save_started
        checkpoint_size = path.stat().st_size
        load_started = time.perf_counter()
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        restored = _restore_model_artifact(
            artifact,
            expected_identity=identity,
        )
        load_seconds = time.perf_counter() - load_started
    return restored, {
        "save_wall_seconds": float(save_seconds),
        "checkpoint_size_bytes": int(checkpoint_size),
        "load_wall_seconds": float(load_seconds),
    }


def _threadpool_state() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return [dict(row) for row in threadpool_info()]


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    schema = stable_hash(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "rows": len(frame),
        }
    )
    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256(schema.encode("ascii"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def benchmark_family(
    *,
    family: str,
    data_root: Path,
    maximum_assets: int,
    maximum_score_timestamps: int,
    package_cache_root: Path | None = None,
    evaluator_publication_mode: str = "batched",
    random_forest_n_jobs: int | None = None,
) -> dict[str, Any]:
    if family not in DELL_FAMILIES:
        raise ValueError(f"Unsupported Dell benchmark family: {family}")
    if evaluator_publication_mode not in {"sequential", "batched"}:
        raise ValueError(
            f"Unsupported evaluator publication mode: {evaluator_publication_mode}"
        )
    config, lane = _family_contract_with_runtime_overrides(
        family, random_forest_n_jobs=random_forest_n_jobs
    )
    data_profiler = DataAssemblyProfiler()
    data, initialization = _measure(
        lambda: CleanV2CompositeData(data_root, profiler=data_profiler)
    )
    spine, spine_measurement = _measure(data.decision_spine)
    qualifier_years = QUALIFIER_YEARS if lane in SHORT_LANES else None
    schedule = build_clean_refit_schedule(
        spine,
        lookback_sessions=int(config["training"]["lookback_sessions"]),
        qualifier_years=qualifier_years,
    )
    if not schedule:
        raise RuntimeError(f"No benchmark schedule for {family}")
    package = schedule[0]
    dates = _package_assembly_dates(family, package)
    cache_evidence: dict[str, Any] = {"enabled": False, "disposition": "DISABLED"}

    def assemble_package() -> pd.DataFrame:
        nonlocal cache_evidence
        if package_cache_root is None or family in CONTROL_FAMILIES:
            return data.assemble_sessions(dates, maximum_assets=maximum_assets)
        result = SharedPanelPackageCache(package_cache_root).get_or_build(
            PanelPackageKey(
                feature_authority_hash=str(data.sidecar_manifest["logical_sha256"]),
                target_authority_hash=str(
                    data.delta_target_manifest["logical_sha256"]
                ),
                static_authority_bundle_sha256=str(
                    authority_bundle()["bundle_sha256"]
                ),
                predictor_order_hash=stable_hash(list(data.predictors)),
                assembly_source_hash=clean_source_hash(),
                session_dates=tuple(dates),
                maximum_assets=maximum_assets,
            ),
            lambda: data.assemble_sessions(dates, maximum_assets=maximum_assets),
        )
        cache_evidence = {
            "enabled": True,
            "disposition": result.disposition,
            "cache_key": result.cache_key,
            "artifact_bytes": result.artifact_bytes,
            "read_wall_seconds": result.read_wall_seconds,
            "write_wall_seconds": result.write_wall_seconds,
        }
        return result.frame

    panel, assembly = _measure(assemble_package)
    if panel.empty:
        raise RuntimeError(f"Empty benchmark panel for {family}")
    populations, slicing = _measure(lambda: _population_slices(panel, package))
    training, scoring = populations
    if training.empty or scoring.empty:
        raise RuntimeError(f"Empty benchmark population for {family}")

    fit_metadata: dict[str, Any]
    checkpoint: dict[str, Any]
    if family in CONTROL_FAMILIES:
        model = None
        fit = StageMeasurement(0.0, 0.0, 0.0, 0, 0)
        fit_metadata = {"kind": "control", "training_rows": 0}
        checkpoint = {
            "save_wall_seconds": 0.0,
            "checkpoint_size_bytes": 0,
            "load_wall_seconds": 0.0,
        }
    else:
        fit_result, fit = _measure(
            lambda: _fit_model(family, config, training, data.predictors)
        )
        model, fit_metadata = fit_result
        model, checkpoint = _model_round_trip(model, family=family)

    available_timestamps = list(
        pd.to_datetime(scoring["decision_timestamp"], utc=True)
        .drop_duplicates()
        .sort_values()
    )
    first_session = available_timestamps[0].date()
    selected_timestamps = [
        timestamp
        for timestamp in available_timestamps
        if timestamp.date() == first_session
    ][:maximum_score_timestamps]
    selected = scoring[
        scoring["decision_timestamp"].isin(selected_timestamps)
    ].copy()
    batches = _scoring_timestamp_batches(family, selected_timestamps, set())
    score_frames: list[pd.DataFrame] = []
    score_stage_timings: dict[str, float] = {}

    def score_operation() -> None:
        for batch in batches:
            batch_rows = selected[selected["decision_timestamp"].isin(batch)].copy()
            timings: dict[str, float] = {}
            predictions = _score_model(
                family,
                model,
                batch_rows,
                panel,
                data.predictors,
                fit_metadata,
                stage_timings=timings,
            )
            score_frames.append(predictions)
            for name, value in timings.items():
                score_stage_timings[name] = score_stage_timings.get(name, 0.0) + value

    _, score = _measure(score_operation)
    predictions = pd.concat(score_frames, ignore_index=True)
    evaluation_stage_totals: dict[str, float] = {}
    historical_rows_read_per_hot_commit: list[int] = []
    evaluator_fingerprints: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ds24-evaluator-benchmark-") as directory:
        evaluator_root = Path(directory)
        writer = MetricsOnlyEvidenceWriter(
            evaluator_root,
            family=family,
            top_n=10_000 if family == "equal_weight_no_model" else 20,
            enable_resolved_performance_v3=True,
            target_loader=lambda request: data.target_loader_from_panel(panel, request),
            terminal_timestamp=max(spine).isoformat(),
            namespace_lease_enabled=False,
            resume_generation=1,
            command_hash="bounded-family-benchmark",
            configuration_hash="bounded-family-benchmark",
            evaluation_contract_hash=resolved_performance_contract_v3_hash(),
        )

        def evaluate_operation() -> None:
            prediction_times = pd.to_datetime(
                predictions["decision_timestamp"], utc=True
            )
            prediction_batches: list[pd.DataFrame] = []
            target_batches: list[pd.DataFrame] = []
            expected_assets_by_timestamp: dict[str, tuple[str, ...]] = {}
            last_metadata: dict[str, Any] = {}
            for timestamp in selected_timestamps:
                score_at_t = selected[selected["decision_timestamp"] == timestamp]
                prediction_batch = predictions[
                    prediction_times == timestamp
                ].copy()
                target_batch = score_at_t[
                    [
                        "asset_id",
                        "decision_timestamp",
                        "target_value",
                        "target_available_timestamp",
                    ]
                ].copy()
                expected_assets = tuple(
                    sorted(score_at_t["asset_id"].astype(str).unique())
                )
                last_metadata = {
                        "model_hash": "bounded-family-benchmark-model",
                        "model_vintage_id": "bounded-family-benchmark-vintage",
                        "preprocessing_hash": "bounded-family-benchmark-preprocessing",
                        "policy_hash": package.policy_hash,
                        "training_cutoff": package.refit_T.isoformat(),
                        "prediction_timestamp": timestamp.isoformat(),
                        "refit_policy": "REFIT_EVERY_5_TRADING_SESSIONS_V1",
                        "attempt_generation": "1",
                        "clean_source_hash": clean_source_hash(),
                        "feature_authority_hash": str(
                            data.sidecar_manifest["logical_sha256"]
                        ),
                        "target_authority_hash": str(
                            data.delta_target_manifest["logical_sha256"]
                        ),
                        "target_contract_hash": str(
                            data.target_contract["resolved_contract_sha256"]
                        ),
                        "static_authority_bundle_sha256": authority_bundle()[
                            "bundle_sha256"
                        ],
                    }
                if evaluator_publication_mode == "batched":
                    prediction_batches.append(prediction_batch)
                    target_batches.append(target_batch)
                    expected_assets_by_timestamp[timestamp.isoformat()] = (
                        expected_assets
                    )
                    continue
                commit = writer.commit_predictions(
                    prediction_batch,
                    targets=target_batch,
                    expected_assets=expected_assets,
                    metadata=last_metadata,
                )
                checkpoint_payload = dict(
                    commit.get("resolved_performance_v3", {}).get("checkpoint")
                    or {}
                )
                historical_rows_read_per_hot_commit.append(
                    int(
                        checkpoint_payload.get(
                            "incremental_hot_path_historical_rows_read", -1
                        )
                    )
                )
                for timing_root in (
                    commit.get("stage_timings_seconds", {}),
                    commit.get("resolved_performance_v3", {}).get(
                        "stage_timings_seconds", {}
                    ),
                ):
                    for name, value in dict(timing_root).items():
                        evaluation_stage_totals[name] = (
                            evaluation_stage_totals.get(name, 0.0) + float(value)
                        )
            if evaluator_publication_mode == "batched":
                commit = writer.commit_prediction_batches(
                    prediction_batches,
                    target_batches=target_batches,
                    expected_assets_by_timestamp=expected_assets_by_timestamp,
                    metadata=last_metadata,
                )
                checkpoint_payload = dict(
                    commit.get("resolved_performance_v3", {}).get("checkpoint")
                    or {}
                )
                historical_rows_read_per_hot_commit.append(
                    int(
                        checkpoint_payload.get(
                            "incremental_hot_path_historical_rows_read", -1
                        )
                    )
                )
                for timing_root in (
                    commit.get("stage_timings_seconds", {}),
                    commit.get("resolved_performance_v3", {}).get(
                        "stage_timings_seconds", {}
                    ),
                ):
                    for name, value in dict(timing_root).items():
                        evaluation_stage_totals[name] = (
                            evaluation_stage_totals.get(name, 0.0) + float(value)
                        )

        _, evaluation = _measure(evaluate_operation)
        for stem in (
            "per_t_metrics",
            "rank_ic_v3",
            "sleeve_maturity_ledger_v3",
            "transaction_cost_ledger_v3",
        ):
            evaluator_fingerprints[stem] = _frame_fingerprint(
                read_parquet_log(evaluator_root, stem)
            )

    timestamp_count = len(selected_timestamps)
    score_and_evaluation_seconds = score.wall_seconds + evaluation.wall_seconds
    seconds_per_timestamp = score_and_evaluation_seconds / timestamp_count
    timestamps_per_hour = 3600.0 / seconds_per_timestamp
    timestamps_per_session: dict[str, int] = {}
    for timestamp in spine:
        session_date = timestamp.date().isoformat()
        timestamps_per_session[session_date] = (
            timestamps_per_session.get(session_date, 0) + 1
        )
    expected_score_timestamps = sum(
        sum(
            timestamps_per_session.get(session_date, 0)
            for session_date in package_spec.score_session_dates
        )
        for package_spec in schedule
    )
    package_seconds = assembly.wall_seconds + slicing.wall_seconds + fit.wall_seconds
    estimated_total_seconds = (
        len(schedule) * package_seconds
        + expected_score_timestamps * seconds_per_timestamp
    )
    peak_resident = max(
        measurement.peak_resident_bytes
        for measurement in (initialization, spine_measurement, assembly, slicing, fit, score, evaluation)
    )
    peak_private = max(
        measurement.peak_private_commit_bytes
        for measurement in (initialization, spine_measurement, assembly, slicing, fit, score, evaluation)
    )
    cpu_seconds = sum(
        measurement.cpu_seconds
        for measurement in (initialization, spine_measurement, assembly, slicing, fit, score, evaluation)
    )
    wall_seconds = sum(
        measurement.wall_seconds
        for measurement in (initialization, spine_measurement, assembly, slicing, fit, score, evaluation)
    )
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "family": family,
        "clean_source_hash": clean_source_hash(),
        "static_authority_bundle_sha256": authority_bundle()["bundle_sha256"],
        "feature_authority_hash": str(data.sidecar_manifest["logical_sha256"]),
        "target_authority_hash": str(data.delta_target_manifest["logical_sha256"]),
        "bounded_scope": {
            "data_root": str(data_root.resolve()),
            "maximum_assets": maximum_assets,
            "maximum_score_timestamps": maximum_score_timestamps,
            "evaluator_publication_mode": evaluator_publication_mode,
            "random_forest_n_jobs": (
                int(config["parameters"]["n_jobs"])
                if family == "random_forest"
                else None
            ),
            "package_ordinal": int(package.ordinal),
            "refit_timestamp": package.refit_T.isoformat(),
            "training_sessions": len(package.training_session_dates),
            "score_sessions": len(package.score_session_dates),
            "panel_rows": len(panel),
            "training_rows": len(training),
            "scoring_rows": len(scoring),
            "scored_rows": len(selected),
            "scored_timestamps": timestamp_count,
        },
        "data_assembly_profile": data_profiler.payload(),
        "shared_package_cache": cache_evidence,
        "stages": {
            "initialization": initialization.payload(),
            "decision_spine": spine_measurement.payload(),
            "package_assembly": assembly.payload(),
            "population_slicing": slicing.payload(),
            "fit": fit.payload(),
            "score": score.payload(),
            "evaluator_and_durable_publication": evaluation.payload(),
        },
        "fit_metadata": fit_metadata,
        "checkpoint": checkpoint,
        "score_stage_seconds": score_stage_timings,
        "evaluation_stage_seconds": evaluation_stage_totals,
        "incremental_evaluator": {
            "state_version": "DS24_INCREMENTAL_EVALUATOR_STATE_V1",
            "historical_rows_read_per_hot_commit": historical_rows_read_per_hot_commit,
            "all_hot_commits_zero_history_reads": all(
                value == 0 for value in historical_rows_read_per_hot_commit[1:]
            ),
        },
        "scientific_fingerprints": {
            "panel": _frame_fingerprint(panel),
            "training_row_identities": _frame_fingerprint(
                training[["asset_id", "decision_timestamp"]].reset_index(drop=True)
            ),
            "scoring_row_identities": _frame_fingerprint(
                scoring[["asset_id", "decision_timestamp"]].reset_index(drop=True)
            ),
            "predictions": _frame_fingerprint(
                predictions.sort_values(
                    ["decision_timestamp", "asset_id"], kind="mergesort"
                ).reset_index(drop=True)
            ),
            "ranks": _frame_fingerprint(
                predictions.assign(
                    rank=predictions.groupby("decision_timestamp")["prediction"].rank(
                        method="first", ascending=False
                    )
                )[
                    ["family", "decision_timestamp", "asset_id", "rank"]
                ]
                .sort_values(["decision_timestamp", "asset_id"], kind="mergesort")
                .reset_index(drop=True)
            ),
            "evaluator_logs": evaluator_fingerprints,
        },
        "bounded_parity_payload": {
            "predictions": predictions[
                ["family", "decision_timestamp", "asset_id", "prediction"]
            ]
            .sort_values(["decision_timestamp", "asset_id"], kind="mergesort")
            .assign(
                decision_timestamp=lambda frame: frame[
                    "decision_timestamp"
                ].map(lambda value: pd.Timestamp(value).isoformat())
            )
            .to_dict(orient="records"),
        },
        "throughput": {
            "seconds_per_refit_package": package_seconds,
            "seconds_per_timestamp": seconds_per_timestamp,
            "timestamps_per_hour": timestamps_per_hour,
            "historical_sessions_per_hour": timestamps_per_hour / 78.0,
            "expected_refit_packages": len(schedule),
            "expected_score_timestamps": expected_score_timestamps,
            "estimated_total_family_wall_seconds": estimated_total_seconds,
        },
        "resources": {
            "peak_resident_bytes": peak_resident,
            "peak_private_commit_bytes": peak_private,
            "measured_cpu_seconds": cpu_seconds,
            "measured_wall_seconds": wall_seconds,
            "mean_cpu_percent": 100.0 * cpu_seconds / wall_seconds,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "threadpools": _threadpool_state(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded read-only real-authority CLEAN V2 family benchmark."
    )
    parser.add_argument("--family", required=True, choices=DELL_FAMILIES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-assets", type=int, default=8)
    parser.add_argument("--maximum-score-timestamps", type=int, default=18)
    parser.add_argument("--package-cache-root", type=Path)
    parser.add_argument("--rf-n-jobs", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--evaluator-publication-mode",
        choices=("sequential", "batched"),
        default="batched",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.maximum_assets <= 0 or args.maximum_score_timestamps <= 0:
        raise ValueError("Benchmark bounds must be positive")
    report = benchmark_family(
        family=args.family,
        data_root=args.data_root,
        maximum_assets=args.maximum_assets,
        maximum_score_timestamps=args.maximum_score_timestamps,
        package_cache_root=args.package_cache_root,
        evaluator_publication_mode=args.evaluator_publication_mode,
        random_forest_n_jobs=args.rf_n_jobs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    if args.quiet:
        print(
            json.dumps(
                {
                    "family": report["family"],
                    "output": str(args.output.resolve()),
                    "fit_wall_seconds": report["stages"]["fit"]["wall_seconds"],
                    "timestamps_per_hour": report["throughput"][
                        "timestamps_per_hour"
                    ],
                    "peak_resident_bytes": report["resources"][
                        "peak_resident_bytes"
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
