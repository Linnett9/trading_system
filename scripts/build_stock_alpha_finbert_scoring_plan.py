from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_finbert_news import (
    FinBertModelIdentity,
)
from core.research.ml.stock_level.stock_alpha_finbert_scoring_plan import (
    publish_finbert_scoring_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish an inference-free production FinBERT scoring plan."
    )
    parser.add_argument("--canonical-corpus-manifest", required=True, type=Path)
    parser.add_argument("--canonical-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--maximum-token-length", type=int, default=256)
    parser.add_argument("--maximum-selected-text-characters", type=int, default=10000)
    parser.add_argument("--scoring-config", required=True, type=Path)
    parser.add_argument("--scope", default="production")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    with args.canonical_corpus.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = publish_finbert_scoring_plan(
        corpus_manifest=json.loads(
            args.canonical_corpus_manifest.read_text(encoding="utf-8")
        ),
        corpus_path=args.canonical_corpus,
        canonical_rows=rows,
        output_path=args.output,
        model_identity=FinBertModelIdentity(
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            inference_device="plan-only",
        ),
        scoring_config=json.loads(args.scoring_config.read_text(encoding="utf-8")),
        chunk_size=args.chunk_size,
        max_token_length=args.maximum_token_length,
        max_characters=args.maximum_selected_text_characters,
        scope=args.scope,
        source_commit=args.source_commit,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
