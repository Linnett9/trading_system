from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.paired_selector_datasets import (
    publish_paired_selector_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an immutable matched price-only/price-plus-news selector dataset pair."
    )
    parser.add_argument("--selector-dataset-root", type=Path, required=True)
    parser.add_argument("--news-feature-store-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, required=True)
    parser.add_argument("--reuse", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = publish_paired_selector_datasets(
        selector_dataset_root=args.selector_dataset_root,
        news_feature_store_root=args.news_feature_store_root,
        output_root=args.output_root,
        lookback_days=args.lookback_days,
        reuse=args.reuse,
    )
    print(json.dumps(result.payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
