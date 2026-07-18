from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_feature_store import (
    publish_pit_news_feature_store,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the canonical partitioned PIT daily news feature store."
    )
    parser.add_argument("--canonical-corpus-manifest", required=True, type=Path)
    parser.add_argument("--score-store-manifest", required=True, type=Path)
    parser.add_argument("--daily-spine-manifest", required=True, type=Path)
    parser.add_argument("--ticker-mapping-manifest", required=True, type=Path)
    parser.add_argument("--scored-articles", required=True, type=Path)
    parser.add_argument("--spine-rows", required=True, type=Path)
    parser.add_argument("--ticker-aliases", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    score_manifest = _json(args.score_store_manifest)
    manifest = publish_pit_news_feature_store(
        canonical_corpus_manifest=_json(args.canonical_corpus_manifest),
        score_store_manifest=score_manifest,
        daily_spine_manifest=_json(args.daily_spine_manifest),
        ticker_mapping_manifest=_json(args.ticker_mapping_manifest),
        scored_articles=_csv(args.scored_articles),
        spine_rows=_csv(args.spine_rows),
        ticker_aliases=_json(args.ticker_aliases),
        output_root=args.output_root,
        finbert_model_identity=score_manifest["finbert_model_identity"],
        source_commit=args.source_commit,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
