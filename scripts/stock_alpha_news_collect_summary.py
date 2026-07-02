from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_KEYS: tuple[str, ...] = (
    "total_rows_collected",
    "deduplicated_row_count",
    "provider_row_counts",
    "rows_by_symbol",
    "provider_symbol_counts",
    "provider_symbol_coverage",
    "symbols_with_feed_errors",
    "symbols_skipped_known_error_feeds",
    "symbols_skipped_max_enabled_feeds_per_run",
    "duplicate_headline_count",
    "duplicate_headline_rate",
    "output_written",
    "model_training_invoked",
    "news_transformer_enabled",
)


def build_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SUMMARY_KEYS}


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = ["Stock-alpha news collection summary"]
    lines.extend(f"{key}: {summary.get(key)}" for key in SUMMARY_KEYS)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("Usage: python scripts/stock_alpha_news_collect_summary.py <stock_alpha_news_free_source_collect.json>", file=sys.stderr)
        return 2

    path = Path(args[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(format_summary(build_summary(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
