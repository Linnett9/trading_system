from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PARENT_STATE_CONTRACT = "selector_parent_publication_run_state_v2"
PIPELINE_STATE_CONTRACT = "selector_operational_pipeline_state.v2"


def evaluate_daily_readiness(
    *,
    selector_state: Mapping[str, Any],
    daily_spine_readiness: Mapping[str, Any],
    pipeline_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    code_blockers = []
    if selector_state.get("run_state_version") != PARENT_STATE_CONTRACT:
        code_blockers.append("SELECTOR_STATE_SCHEMA_MISMATCH")
    if (
        daily_spine_readiness.get("status") != "READY"
        or daily_spine_readiness.get("whole_table_to_pylist_used") is not False
    ):
        code_blockers.append("DAILY_SPINE_STREAMING_READINESS_NOT_READY")

    parent_stages = {
        int(row["stage_number"]): row.get("status")
        for row in selector_state.get("stages", [])
        if "stage_number" in row
    }
    production_blockers = [
        f"SELECTOR_PARENT_STAGE_{number}_NOT_COMPLETE"
        for number in range(1, 11)
        if parent_stages.get(number) != "complete"
    ]
    pipeline_stages: dict[str, Any] = {}
    if pipeline_state is None:
        production_blockers.append("DIRECT_PIPELINE_STATE_NOT_PROVIDED")
    elif pipeline_state.get("contract_version") != PIPELINE_STATE_CONTRACT:
        production_blockers.append("DIRECT_PIPELINE_STATE_SCHEMA_MISMATCH")
    else:
        pipeline_stages = dict(pipeline_state.get("stages") or {})
        production_blockers.extend(
            f"DIRECT_PIPELINE_STAGE_NOT_COMPLETE:{stage}"
            for stage in pipeline_state.get("stage_order", [])
            if pipeline_stages.get(stage) != "COMPLETED"
        )

    return {
        "contract_version": "selector_operational_readiness.v2",
        "status": "READY" if not code_blockers else "BLOCKED",
        "code_readiness_status": "READY" if not code_blockers else "BLOCKED",
        "code_readiness_blockers": sorted(code_blockers),
        "production_completion_status": (
            "COMPLETE" if not code_blockers and not production_blockers
            else "INCOMPLETE"
        ),
        "production_completion_blockers": sorted(production_blockers),
        "synthetic_benchmark_evidence_used": False,
        "production_commands_executed_by_check": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-state", required=True, type=Path)
    parser.add_argument("--daily-spine-readiness", required=True, type=Path)
    parser.add_argument("--pipeline-state", type=Path)
    parser.add_argument("--verification-output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_daily_readiness(
        selector_state=json.loads(
            args.selector_state.read_text(encoding="utf-8")
        ),
        daily_spine_readiness=json.loads(
            args.daily_spine_readiness.read_text(encoding="utf-8")
        ),
        pipeline_state=(
            json.loads(args.pipeline_state.read_text(encoding="utf-8"))
            if args.pipeline_state else None
        ),
    )
    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["code_readiness_status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
