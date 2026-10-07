"""Supervisor-governed full-universe qualification of rolling-panel reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.benchmark_lane import (  # noqa: E402
    INCREMENTAL_PANEL_RESULT_SCHEMA,
    GovernedIncrementalPanelRequest,
    benchmark_control_root,
    validate_benchmark_namespace,
)
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
from core.research.ml.ds24.clean_v2_resources import (  # noqa: E402
    AllocationEstimate,
    RESOURCE_POLICY,
    panel_allocation_estimate,
    system_memory_snapshot,
    write_json_object_atomic,
)
from core.research.ml.ds24.clean_v2_runtime import (  # noqa: E402
    QUALIFIER_YEARS,
    RUN_ID,
)
from scripts.local.ds24_clean_v2_family_benchmark import (  # noqa: E402
    SHORT_LANES,
    _estimated_panel_row_upper_bound,
    _family_contract,
    _frame_fingerprint,
    _package_assembly_dates,
    benchmark_family,
    build_clean_refit_schedule,
)
from scripts.local.ds24_governed_benchmark_support import (  # noqa: E402
    current_process_is_in_windows_job,
    prediction_parity,
    production_progress_snapshot,
    reservation_client,
)


def _profile_total(profile: Mapping[str, Any], field: str) -> int:
    return sum(int(value) for value in dict(profile.get(field) or {}).values())


def _panel_io_total(profile: Mapping[str, Any], field: str) -> int:
    values = dict(profile.get(field) or {})
    return sum(
        int(values.get(stage, 0))
        for stage in (
            "feature_reads",
            "sidecar_reads",
            "target_reads",
            "context_reads",
        )
    )


def _panel_key(
    *, data: CleanV2CompositeData, dates: tuple[str, ...]
) -> PanelPackageKey:
    return PanelPackageKey(
        feature_authority_hash=str(data.sidecar_manifest["logical_sha256"]),
        target_authority_hash=str(data.delta_target_manifest["logical_sha256"]),
        static_authority_bundle_sha256=str(authority_bundle()["bundle_sha256"]),
        predictor_order_hash=stable_hash(list(data.predictors)),
        assembly_source_hash=clean_source_hash(),
        session_dates=dates,
        maximum_assets=None,
    )


def _schedule_qualifier_years(lane: str) -> tuple[int, ...] | None:
    """Mirror the production schedule scope for predecessor construction."""

    return QUALIFIER_YEARS if lane in SHORT_LANES else None


def _seed_predecessor(
    *,
    request: GovernedIncrementalPanelRequest,
    family: str,
    cache_root: Path,
    client: Any,
) -> dict[str, Any]:
    profiler = DataAssemblyProfiler()
    data = CleanV2CompositeData(request.data_root, profiler=profiler)
    config, lane = _family_contract(family)
    spine = data.decision_spine()
    schedule = build_clean_refit_schedule(
        spine,
        lookback_sessions=int(config["training"]["lookback_sessions"]),
        qualifier_years=_schedule_qualifier_years(lane),
    )
    if request.package_ordinal >= len(schedule):
        raise RuntimeError(
            f"Package ordinal {request.package_ordinal} is outside {family} schedule"
        )
    predecessor = schedule[request.package_ordinal - 1]
    dates = tuple(_package_assembly_dates(family, predecessor))
    key = _panel_key(data=data, dates=dates)
    estimated_rows = _estimated_panel_row_upper_bound(
        data=data,
        spine=spine,
        session_dates=dates,
        maximum_assets=None,
    )
    cache = SharedPanelPackageCache(cache_root)
    with client.reserve(
        stage=f"benchmark_{family}_incremental_predecessor_panel",
        estimate=panel_allocation_estimate(
            row_count=estimated_rows,
            column_count=len(data.predictors) + 12,
        ),
    ):
        result = cache.get_or_build(
            key,
            lambda: data.assemble_sessions(dates, maximum_assets=None),
        )
    return {
        "package_ordinal": request.package_ordinal - 1,
        "refit_timestamp": predecessor.refit_T.isoformat(),
        "session_dates": list(dates),
        "rows": len(result.frame),
        "panel_fingerprint": _frame_fingerprint(result.frame),
        "cache_key": result.cache_key,
        "cache_disposition": result.disposition,
        "artifact_bytes": result.artifact_bytes,
        "cache_write_wall_seconds": result.write_wall_seconds,
        "data_assembly_profile": profiler.payload(),
    }


def _compact(report: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(report)
    compact.pop("bounded_parity_payload", None)
    return compact


def _cache_inventory(cache_root: Path) -> dict[str, Any]:
    manifests = sorted(cache_root.glob("*.json"))
    parquet = sorted(cache_root.glob("*.parquet"))
    return {
        "manifest_count": len(manifests),
        "parquet_count": len(parquet),
        "total_bytes": sum(path.stat().st_size for path in (*manifests, *parquet)),
        "manifest_names": [path.name for path in manifests],
    }


def _family_qualification(
    *,
    request: GovernedIncrementalPanelRequest,
    family: str,
    client: Any,
) -> dict[str, Any]:
    canonical = benchmark_family(
        family=family,
        data_root=request.data_root,
        maximum_assets=None,
        maximum_score_timestamps=None,
        package_ordinal=request.package_ordinal,
        package_cache_root=None,
        evaluator_publication_mode="batched",
        random_forest_n_jobs=1 if family == "random_forest" else None,
        reservation_client=client,
    )
    cache_root = request.output_root / "cache" / f"family={family}"
    predecessor = _seed_predecessor(
        request=request,
        family=family,
        cache_root=cache_root,
        client=client,
    )
    incremental = benchmark_family(
        family=family,
        data_root=request.data_root,
        maximum_assets=None,
        maximum_score_timestamps=None,
        package_ordinal=request.package_ordinal,
        package_cache_root=cache_root,
        evaluator_publication_mode="batched",
        random_forest_n_jobs=1 if family == "random_forest" else None,
        reservation_client=client,
    )
    parity = prediction_parity(canonical, incremental)
    canonical_profile = canonical["data_assembly_profile"]
    incremental_profile = incremental["data_assembly_profile"]
    canonical_panel_seconds = float(
        canonical["stages"]["package_assembly"]["wall_seconds"]
    )
    incremental_panel_seconds = float(
        incremental["stages"]["package_assembly"]["wall_seconds"]
    )
    predecessor_manifest = cache_root / f"{predecessor['cache_key'][:32]}.json"
    cache_disposition = str(
        incremental["shared_package_cache"]["disposition"]
    )
    exact_fields = (
        "output_keys_equal",
        "rank_fingerprint_equal",
        "evaluator_fingerprints_equal",
        "training_rows_equal",
        "scoring_rows_equal",
        "panel_fingerprint_equal",
        "feature_order_equal",
        "training_model_inputs_equal",
        "scoring_model_inputs_equal",
    )
    exact_parity = (
        all(parity.get(field) is True for field in exact_fields)
        and float(parity["maximum_absolute_prediction_delta"]) == 0.0
        and int(parity["rank_disagreements"]) == 0
    )
    rows_newly_read = int(
        incremental_profile.get("output_rows", {}).get("panel_finalization", 0)
    )
    return {
        "family": family,
        "canonical": _compact(canonical),
        "predecessor_seed": predecessor,
        "incremental": _compact(incremental),
        "parity": parity,
        "exact_parity": exact_parity,
        "incremental_path_used": cache_disposition
        == "MISS_INCREMENTAL_PUBLISHED",
        "performance": {
            "canonical_panel_wall_seconds": canonical_panel_seconds,
            "incremental_panel_wall_seconds": incremental_panel_seconds,
            "seconds_saved_per_refit": (
                canonical_panel_seconds - incremental_panel_seconds
            ),
            "reduction_percent": (
                100.0
                * (1.0 - incremental_panel_seconds / canonical_panel_seconds)
                if canonical_panel_seconds > 0.0
                else 0.0
            ),
            "canonical_source_partition_bytes_opened": _profile_total(
                canonical_profile, "source_partition_bytes_opened"
            ),
            "incremental_source_partition_bytes_opened": _profile_total(
                incremental_profile, "source_partition_bytes_opened"
            ),
            "canonical_panel_source_partition_bytes_opened": _panel_io_total(
                canonical_profile, "source_partition_bytes_opened"
            ),
            "incremental_panel_source_partition_bytes_opened": _panel_io_total(
                incremental_profile, "source_partition_bytes_opened"
            ),
            "canonical_partitions_opened": _profile_total(
                canonical_profile, "partitions_opened"
            ),
            "incremental_partitions_opened": _profile_total(
                incremental_profile, "partitions_opened"
            ),
            "canonical_panel_partitions_opened": _panel_io_total(
                canonical_profile, "partitions_opened"
            ),
            "incremental_panel_partitions_opened": _panel_io_total(
                incremental_profile, "partitions_opened"
            ),
            "rows_reused": max(
                0,
                int(incremental["bounded_scope"]["panel_rows"])
                - rows_newly_read,
            ),
            "rows_newly_read": rows_newly_read,
        },
        "predecessor_retained_after_successor_publication": (
            predecessor_manifest.is_file()
        ),
        "cache_inventory": _cache_inventory(cache_root),
    }


def run_request(request: GovernedIncrementalPanelRequest) -> dict[str, Any]:
    validate_benchmark_namespace(repository_root=ROOT, request=request)
    if clean_source_hash() != request.source_hash:
        raise RuntimeError("Governed benchmark source hash mismatch")
    data = CleanV2CompositeData(request.data_root)
    if str(data.sidecar_manifest["logical_sha256"]) != request.feature_authority_hash:
        raise RuntimeError("Governed benchmark feature authority mismatch")
    if (
        str(data.delta_target_manifest["logical_sha256"])
        != request.target_authority_hash
    ):
        raise RuntimeError("Governed benchmark target authority mismatch")
    if not current_process_is_in_windows_job():
        raise RuntimeError("Governed benchmark is not inside a Windows Job")

    client = reservation_client(request)
    memory_before = system_memory_snapshot().payload()
    production_before = production_progress_snapshot(request)
    families = {
        family: _family_qualification(
            request=request,
            family=family,
            client=client,
        )
        for family in request.families
    }
    return {
        "schema": INCREMENTAL_PANEL_RESULT_SCHEMA,
        "request": request.payload(),
        "job_membership_verified": True,
        "resource_policy_id": RESOURCE_POLICY.policy_id,
        "memory_before": memory_before,
        "memory_after": system_memory_snapshot().payload(),
        "production_progress_before": production_before,
        "production_progress_after": production_progress_snapshot(request),
        "families": families,
        "production_publication_capability": "ABSENT",
        "production_cursor_advance_capability": "ABSENT",
        "production_checkpoint_capability": "ABSENT",
    }


def submit_request(
    *, request_id: str, data_root: Path, package_ordinal: int
) -> Path:
    data = CleanV2CompositeData(data_root)
    control_root = benchmark_control_root(ROOT)
    request = GovernedIncrementalPanelRequest(
        request_id=request_id,
        package_ordinal=package_ordinal,
        data_root=data_root,
        output_root=control_root / "runs" / request_id,
        production_run_root=ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID,
        source_hash=clean_source_hash(),
        feature_authority_hash=str(data.sidecar_manifest["logical_sha256"]),
        target_authority_hash=str(data.delta_target_manifest["logical_sha256"]),
    )
    validate_benchmark_namespace(repository_root=ROOT, request=request)
    request_path = control_root / "pending" / f"{request_id}.json"
    if request_path.exists() or request.output_root.exists():
        raise RuntimeError("Governed benchmark request identity already exists")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_object_atomic(request_path, request.payload())
    return request_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--request", type=Path)
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--request-id")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--package-ordinal", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.submit:
        if not args.request_id or args.data_root is None:
            raise ValueError("--submit requires --request-id and --data-root")
        if args.package_ordinal is None or args.package_ordinal <= 0:
            raise ValueError("--submit requires a positive --package-ordinal")
        request_path = submit_request(
            request_id=args.request_id,
            data_root=args.data_root,
            package_ordinal=args.package_ordinal,
        )
        print(
            json.dumps(
                {
                    "classification": "INCREMENTAL_BENCHMARK_REQUEST_SUBMITTED",
                    "request": str(request_path.resolve()),
                }
            )
        )
        return 0
    if args.request is None:
        raise ValueError("--request is required for benchmark execution")
    request = GovernedIncrementalPanelRequest.from_payload(
        json.loads(args.request.read_text(encoding="utf-8"))
    )
    result = run_request(request)
    output = request.output_root / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    client = reservation_client(request)
    with client.reserve(
        stage="benchmark_isolated_result_publication",
        estimate=AllocationEstimate(16 * 1024**2, 32 * 1024**2),
    ):
        write_json_object_atomic(output, result)
    print(
        json.dumps(
            {"classification": "BENCHMARK_COMPLETE", "output": str(output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
