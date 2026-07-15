from __future__ import annotations

import argparse
from pathlib import Path

from core.research.ml.stock_level.selector_dataset import build_frozen_selector_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--market-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--decision-dates", default="")
    parser.add_argument("--source-sha256")
    parser.add_argument("--config-hash")
    args = parser.parse_args()
    paths = build_frozen_selector_dataset(
        args.source, args.market_root, args.output_root,
        symbols=[x for x in args.symbols.split(",") if x],
        decision_dates=[x for x in args.decision_dates.split(",") if x],
        source_sha256=args.source_sha256,
        config_hash=args.config_hash,
    )
    print(paths.manifest)


if __name__ == "__main__":
    main()
