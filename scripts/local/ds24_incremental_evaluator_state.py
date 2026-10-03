from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24_metrics_only_evaluator import (  # noqa: E402
    ResolvedPerformanceV3Writer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deep-verify or rebuild DS24 CLEAN V2 incremental evaluator state. "
            "Run only while the family namespace writer is stopped."
        )
    )
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true")
    action.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    writer = ResolvedPerformanceV3Writer(
        root=args.metrics_root.resolve(),
        family=str(args.family),
    )
    result = writer.verify_or_rebuild_incremental_state(
        rebuild=bool(args.rebuild)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] in {"VALID", "REBUILT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

