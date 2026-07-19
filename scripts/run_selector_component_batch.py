from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.selector_component_batch import run_stage10_component_batch
from core.research.ml.selector_research_campaign import (
    BASELINE_CAMPAIGN_ID,
    RESEARCH_CAMPAIGN_ID,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the authoritative Stage-10 selector component batch."
    )
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--input-inventory", required=True, type=Path)
    parser.add_argument(
        "--campaign-selection",
        required=True,
        choices=("historical", "research"),
    )
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--parent-gate", required=True, type=Path)
    parser.add_argument("--experiment-ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-component-workers", type=int, default=3)
    parser.add_argument("--weighted-capacity", type=int, default=4)
    parser.add_argument("--compute-disabled", action="store_true")
    parser.add_argument("--compute-lease-ledger", type=Path)
    parser.add_argument("--compute-runs-root", type=Path, default=Path("reports/runs"))
    parser.add_argument("--compute-run-id")
    parser.add_argument(
        "--compute-run-registry",
        type=Path,
        default=Path("reports/runs/run_registry.json"),
    )
    args = parser.parse_args()
    campaign_manifest = json.loads(
        args.campaign_manifest.read_text(encoding="utf-8")
    )
    expected_campaign_id = {
        "historical": BASELINE_CAMPAIGN_ID,
        "research": RESEARCH_CAMPAIGN_ID,
    }[args.campaign_selection]
    if campaign_manifest.get("campaign_id") != expected_campaign_id:
        parser.error("campaign selection and manifest identity differ")
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    compute_execution = None
    if not args.compute_disabled:
        if not args.compute_lease_ledger or not args.compute_run_id:
            parser.error(
                "compute adoption requires --compute-lease-ledger and "
                "--compute-run-id (or explicit --compute-disabled)"
            )
        from core.research.compute.machine_profile import detect_runtime_resources
        from core.research.ml.selector_compute_execution import (
            SelectorComputeExecution,
        )

        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        compute_execution = SelectorComputeExecution(
            jobs=list(readiness.get("production_plan") or ()),
            campaign_manifest=campaign_manifest,
            readiness=readiness,
            run_id=args.compute_run_id,
            source_git_commit=source_commit,
            runs_root=args.compute_runs_root,
            lease_ledger_path=args.compute_lease_ledger,
            registry_path=args.compute_run_registry,
            available_memory=lambda: detect_runtime_resources().available_ram_bytes,
        )
    report = run_stage10_component_batch(
        readiness=readiness,
        input_inventory=json.loads(
            args.input_inventory.read_text(encoding="utf-8")
        ),
        parent_gate_path=args.parent_gate,
        ledger_path=args.experiment_ledger,
        output_root=args.output_root,
        max_component_workers=args.max_component_workers,
        weighted_capacity=args.weighted_capacity,
        campaign_manifest=campaign_manifest,
        compute_execution=compute_execution,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
