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
from core.research.ml.stock_level.selector_dataset import read_selector_dataset_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build authoritative base selector input owners.")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selector-dataset", required=True, type=Path)
    parser.add_argument("--parent-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--selector-run-id", default="20260716T091011Z")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--component-plan-only", action="store_true")
    parser.add_argument("--campaign-id")
    parser.add_argument("--source-commit")
    parser.add_argument("--experiment-ledger", type=Path)
    args = parser.parse_args()
    if args.component_plan_only:
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
        if not args.plan:
            raise ValueError("--plan is required unless --component-plan-only is used")
        manifest = json.loads((args.selector_dataset / "manifest.json").read_text(encoding="utf-8"))
        result = build_operational_inputs(
            plan=json.loads(args.plan.read_text(encoding="utf-8")),
            dataset_manifest=manifest,
            parent_gate=json.loads(args.parent_gate.read_text(encoding="utf-8")),
            rows=read_selector_dataset_rows(args.selector_dataset),
            output_root=args.output_root, evaluation_cutoff=args.evaluation_cutoff,
            selector_run_id=args.selector_run_id,
            max_workers=args.max_workers,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status", "READY") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
