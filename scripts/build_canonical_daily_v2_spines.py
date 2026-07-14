from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.data.canonical_daily_v2_spines import build_selector_spines


def main() -> int:
    parser = argparse.ArgumentParser(description="Build labeled and inference selector spines from canonical daily v2.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/market_data/canonical_daily_v2/full"))
    parser.add_argument("--target-horizon-sessions", type=int, default=10)
    args = parser.parse_args()
    result = build_selector_spines(dataset_root=args.dataset_root, target_horizon_sessions=args.target_horizon_sessions)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

