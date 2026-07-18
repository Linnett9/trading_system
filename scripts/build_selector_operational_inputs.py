from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.selector_operational_inputs import (
    build_operational_inputs,
    build_selector_component_plan,
    validate_inventory,
    validate_selector_component_plan,
)
from core.research.ml.selector_operational_plan import (
    build_selector_operational_plan,
)
from core.research.ml.selector_wave4_input_packages import (
    publish_selector_operational_packages_v2,
)
from core.research.ml.stock_level.selector_dataset import read_selector_dataset_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build authoritative base selector input owners.")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selector-dataset", type=Path)
    parser.add_argument("--parent-gate", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--outcome-maturity-cutoff")
    parser.add_argument("--evaluation-cutoff")
    parser.add_argument("--selector-run-id", default="20260716T091011Z")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--component-plan-only", action="store_true")
    parser.add_argument("--campaign-id")
    parser.add_argument("--source-commit")
    parser.add_argument("--experiment-ledger", type=Path)
    parser.add_argument("--operational-plan-only", action="store_true")
    parser.add_argument(
        "--campaign-selection", choices=("historical", "research")
    )
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--parent-identities", type=Path)
    parser.add_argument("--source-schema-guarantee", action="append", default=[])
    parser.add_argument("--publish-v2-packages", action="store_true")
    parser.add_argument("--operational-plan-v2", type=Path)
    parser.add_argument("--source-guarantees", type=Path)
    parser.add_argument("--rows-by-job", type=Path)
    args = parser.parse_args()
    if args.publish_v2_packages:
        required = (
            args.operational_plan_v2, args.campaign_manifest,
            args.source_guarantees, args.parent_identities,
            args.rows_by_job,
        )
        if not all(required):
            raise ValueError(
                "V2 package publication requires plan, campaign, source "
                "guarantees, parent identities, and explicit frozen rows"
            )
        result = publish_selector_operational_packages_v2(
            plan=json.loads(
                args.operational_plan_v2.read_text(encoding="utf-8")
            ),
            campaign=json.loads(
                args.campaign_manifest.read_text(encoding="utf-8")
            ),
            source_guarantees=json.loads(
                args.source_guarantees.read_text(encoding="utf-8")
            ),
            parent_identities=json.loads(
                args.parent_identities.read_text(encoding="utf-8")
            ),
            rows_by_job=json.loads(
                args.rows_by_job.read_text(encoding="utf-8")
            ),
            output_root=args.output_root,
        )
    elif args.operational_plan_only:
        required = (
            args.campaign_selection, args.campaign_manifest,
            args.protocol_manifest, args.parent_identities,
            args.source_commit,
        )
        if not all(required):
            raise ValueError(
                "Operational plan requires explicit campaign selection, "
                "campaign/protocol manifests, parent identities, and source commit"
            )
        result = build_selector_operational_plan(
            campaign=json.loads(
                args.campaign_manifest.read_text(encoding="utf-8")
            ),
            protocol=json.loads(
                args.protocol_manifest.read_text(encoding="utf-8")
            ),
            campaign_selection=args.campaign_selection,
            parent_identities=json.loads(
                args.parent_identities.read_text(encoding="utf-8")
            ),
            source_git_commit=args.source_commit,
            source_schema_guarantees=args.source_schema_guarantee,
            output_path=args.output_root / "operational_component_plan.v2.json",
        )
    elif args.component_plan_only:
        if not args.selector_dataset or not args.parent_gate:
            raise ValueError(
                "--selector-dataset and --parent-gate are required"
            )
        plan_path = args.output_root / "component_plan.json"
        if args.verify_only:
            result = validate_selector_component_plan(plan_path)
        else:
            result = build_selector_component_plan(
                dataset_manifest=json.loads(args.selector_dataset.read_text(encoding="utf-8")),
                parent_gate=json.loads(args.parent_gate.read_text(encoding="utf-8")),
                output_root=args.output_root,
                campaign_id=args.campaign_id or args.selector_run_id,
                selector_run_id=args.selector_run_id,
                source_commit=args.source_commit or "",
                experiment_ledger_path=args.experiment_ledger,
            )
    elif args.verify_only:
        result = validate_inventory(args.output_root / "inventory.json")
    else:
        if not args.plan or not args.selector_dataset or not args.parent_gate:
            raise ValueError(
                "--plan, --selector-dataset, and --parent-gate are required"
            )
        manifest = json.loads((args.selector_dataset / "manifest.json").read_text(encoding="utf-8"))
        result = build_operational_inputs(
            plan=json.loads(args.plan.read_text(encoding="utf-8")),
            dataset_manifest=manifest,
            parent_gate=json.loads(args.parent_gate.read_text(encoding="utf-8")),
            rows=read_selector_dataset_rows(args.selector_dataset),
            output_root=args.output_root,
            outcome_maturity_cutoff=args.outcome_maturity_cutoff,
            evaluation_cutoff=args.evaluation_cutoff,
            selector_run_id=args.selector_run_id,
            max_workers=args.max_workers,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status", "READY") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
