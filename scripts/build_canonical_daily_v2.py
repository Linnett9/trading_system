from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.data.canonical_daily_v2_builder import build_full_canonical_daily_v2, retry_failed_partitions


def main() -> int:
    parser = argparse.ArgumentParser(description="Build partitioned canonical daily v2 dataset.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.retry_failed:
        result = retry_failed_partitions(workers=args.workers)
    else:
        result = build_full_canonical_daily_v2(workers=args.workers, force=args.force, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {None, "COMPLETE", "DRY_RUN"} or result.get("failed_partitions", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

