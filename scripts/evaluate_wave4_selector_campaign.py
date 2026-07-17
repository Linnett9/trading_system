from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.wave4_gate_campaign import evaluate_wave4_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an existing 15-component Wave 4 campaign.")
    parser.add_argument("--component-plan", required=True, type=Path)
    parser.add_argument("--component-manifest", action="append", type=Path, default=[])
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--momentum-manifest", action="append", type=Path, default=[])
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--experiment-ledger", required=True, type=Path)
    args = parser.parse_args()
    manifests = list(args.component_manifest)
    if args.campaign_root:
        manifests.extend(args.campaign_root.glob("**/manifest.json"))
    result = evaluate_wave4_campaign(
        component_plan_path=args.component_plan,
        component_manifest_paths=manifests,
        momentum_manifest_paths=args.momentum_manifest,
        thresholds=json.loads(args.thresholds.read_text()) if args.thresholds else None,
        output_root=args.output_root,
        experiment_ledger_path=args.experiment_ledger,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["primary_status"] in {"READY_FOR_PORTFOLIO_REPLAY", "REJECTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
