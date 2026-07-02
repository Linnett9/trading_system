from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_KEYS: tuple[str, ...] = (
    "registry_complete",
    "universe_symbol_count",
    "classification_counts",
    "verified_rss_feed_symbol_count",
    "known_error_feed_symbol_count",
    "no_verified_official_rss_symbol_count",
    "sec_only_candidate_symbol_count",
    "disabled_pending_review_symbol_count",
    "full_universe_known_error_feed_symbols",
    "selected_symbol_count",
    "selected_enabled_feed_symbol_count",
    "selected_row_returning_symbol_count",
    "selected_symbol_row_coverage",
    "total_universe_symbol_count",
    "total_universe_coverage",
    "total_universe_row_coverage",
    "total_universe_enabled_feed_coverage",
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
    registry = dict(payload.get("registry_validation", {}) or {})
    counts = dict(registry.get("classification_counts", {}) or {})
    derived = {
        "registry_complete": registry.get("registry_complete", payload.get("registry_complete")),
        "universe_symbol_count": registry.get("universe_symbol_count"),
        "classification_counts": counts,
        "verified_rss_feed_symbol_count": counts.get("verified_rss_feed"),
        "known_error_feed_symbol_count": counts.get("known_error_feed"),
        "no_verified_official_rss_symbol_count": counts.get("no_verified_official_rss"),
        "sec_only_candidate_symbol_count": counts.get("sec_only_candidate"),
        "disabled_pending_review_symbol_count": counts.get("disabled_pending_review"),
    }
    return {
        key: derived[key] if key in derived else payload.get(key)
        for key in SUMMARY_KEYS
    }


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
