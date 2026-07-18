from __future__ import annotations

import argparse
import json
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
    report = run_stage10_component_batch(
        readiness=json.loads(args.readiness.read_text(encoding="utf-8")),
        input_inventory=json.loads(
            args.input_inventory.read_text(encoding="utf-8")
        ),
        parent_gate_path=args.parent_gate,
        ledger_path=args.experiment_ledger,
        output_root=args.output_root,
        max_component_workers=args.max_component_workers,
        weighted_capacity=args.weighted_capacity,
        campaign_manifest=campaign_manifest,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
