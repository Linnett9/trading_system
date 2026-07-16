from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.selector_operational_inputs import build_operational_inputs, validate_inventory
from core.research.ml.stock_level.selector_dataset import read_selector_dataset_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build authoritative base selector input owners.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--selector-dataset", required=True, type=Path)
    parser.add_argument("--parent-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = validate_inventory(args.output_root / "inventory.json")
    else:
        manifest = json.loads((args.selector_dataset / "manifest.json").read_text(encoding="utf-8"))
        result = build_operational_inputs(
            plan=json.loads(args.plan.read_text(encoding="utf-8")),
            dataset_manifest=manifest,
            parent_gate=json.loads(args.parent_gate.read_text(encoding="utf-8")),
            rows=read_selector_dataset_rows(args.selector_dataset),
            output_root=args.output_root, evaluation_cutoff=args.evaluation_cutoff,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status", "READY") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
