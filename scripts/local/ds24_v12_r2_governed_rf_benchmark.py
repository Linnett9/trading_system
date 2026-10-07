from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.benchmark_lane import (  # noqa: E402
    BENCHMARK_RESULT_SCHEMA,
    GovernedBenchmarkRequest,
    benchmark_control_root,
    validate_benchmark_namespace,
)
from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    clean_source_hash,
)
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData  # noqa: E402
from core.research.ml.ds24.clean_v2_runtime import RUN_ID  # noqa: E402
from core.research.ml.ds24.clean_v2_resources import (  # noqa: E402
    AllocationEstimate,
    RESOURCE_POLICY,
    system_memory_snapshot,
    write_json_object_atomic,
)
from scripts.local.ds24_clean_v2_family_benchmark import (  # noqa: E402
    benchmark_family,
)
from scripts.local.ds24_governed_benchmark_support import (  # noqa: E402
    current_process_is_in_windows_job as _current_process_is_in_windows_job,
    prediction_parity as _prediction_parity,
    production_progress_snapshot as _production_progress_snapshot,
    reservation_client as _reservation_client,
)


def run_request(request: GovernedBenchmarkRequest) -> dict[str, Any]:
    validate_benchmark_namespace(repository_root=ROOT, request=request)
    if clean_source_hash() != request.source_hash:
        raise RuntimeError("Governed benchmark source hash mismatch")
    data = CleanV2CompositeData(request.data_root)
    if str(data.sidecar_manifest["logical_sha256"]) != request.feature_authority_hash:
        raise RuntimeError("Governed benchmark feature authority mismatch")
    if str(data.delta_target_manifest["logical_sha256"]) != request.target_authority_hash:
        raise RuntimeError("Governed benchmark target authority mismatch")
    if not _current_process_is_in_windows_job():
        raise RuntimeError("Governed benchmark is not inside a Windows Job")

    client = _reservation_client(request)
    memory_before = system_memory_snapshot().payload()
    production_progress_before = _production_progress_snapshot(request)
    variants: dict[str, dict[str, Any]] = {}
    production_progress_after_variant: dict[str, dict[str, Any]] = {}
    for n_jobs in request.requested_rf_n_jobs:
        report = benchmark_family(
            family=request.family,
            data_root=request.data_root,
            maximum_assets=None,
            maximum_score_timestamps=None,
            package_ordinal=request.package_ordinal,
            package_cache_root=None,
            evaluator_publication_mode="batched",
            random_forest_n_jobs=n_jobs,
            reservation_client=client,
        )
        variants[str(n_jobs)] = report
        production_progress_after_variant[str(n_jobs)] = (
            _production_progress_snapshot(request)
        )

    baseline = variants["1"]
    parity = {
        n_jobs: _prediction_parity(baseline, report)
        for n_jobs, report in variants.items()
        if n_jobs != "1"
    }
    compact_variants: dict[str, dict[str, Any]] = {}
    for n_jobs, report in variants.items():
        compact = dict(report)
        compact.pop("bounded_parity_payload", None)
        compact_variants[n_jobs] = compact
    return {
        "schema": BENCHMARK_RESULT_SCHEMA,
        "request": request.payload(),
        "job_membership_verified": True,
        "resource_policy_id": RESOURCE_POLICY.policy_id,
        "memory_before": memory_before,
        "memory_after": system_memory_snapshot().payload(),
        "production_progress_before": production_progress_before,
        "production_progress_after_variant": production_progress_after_variant,
        "variants": compact_variants,
        "parity_against_n_jobs_1": parity,
        "production_publication_capability": "ABSENT",
        "production_cursor_advance_capability": "ABSENT",
        "production_checkpoint_capability": "ABSENT",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one supervisor-admitted full-universe RF qualification."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--request", type=Path)
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--request-id")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--package-ordinal", type=int)
    return parser.parse_args()


def submit_request(
    *, request_id: str, data_root: Path, package_ordinal: int
) -> Path:
    data = CleanV2CompositeData(data_root)
    control_root = benchmark_control_root(ROOT)
    request = GovernedBenchmarkRequest(
        request_id=request_id,
        family="random_forest",
        package_ordinal=package_ordinal,
        data_root=data_root,
        output_root=control_root / "runs" / request_id,
        production_run_root=(
            ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID
        ),
        source_hash=clean_source_hash(),
        feature_authority_hash=str(data.sidecar_manifest["logical_sha256"]),
        target_authority_hash=str(
            data.delta_target_manifest["logical_sha256"]
        ),
    )
    validate_benchmark_namespace(repository_root=ROOT, request=request)
    request_path = control_root / "pending" / f"{request_id}.json"
    if request_path.exists() or request.output_root.exists():
        raise RuntimeError("Governed benchmark request identity already exists")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_object_atomic(request_path, request.payload())
    return request_path


def main() -> int:
    args = _parse_args()
    if args.submit:
        if not args.request_id or args.data_root is None:
            raise ValueError("--submit requires --request-id and --data-root")
        if args.package_ordinal is None or args.package_ordinal < 0:
            raise ValueError("--submit requires a non-negative --package-ordinal")
        request_path = submit_request(
            request_id=args.request_id,
            data_root=args.data_root,
            package_ordinal=args.package_ordinal,
        )
        print(
            json.dumps(
                {
                    "classification": "BENCHMARK_REQUEST_SUBMITTED",
                    "request": str(request_path.resolve()),
                }
            )
        )
        return 0
    if args.request is None:
        raise ValueError("--request is required for benchmark execution")
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    request = GovernedBenchmarkRequest.from_payload(payload)
    result = run_request(request)
    output = request.output_root / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    publication_client = _reservation_client(request)
    with publication_client.reserve(
        stage="benchmark_isolated_result_publication",
        estimate=AllocationEstimate(16 * 1024**2, 32 * 1024**2),
    ):
        write_json_object_atomic(output, result)
    print(json.dumps({"classification": "BENCHMARK_COMPLETE", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
