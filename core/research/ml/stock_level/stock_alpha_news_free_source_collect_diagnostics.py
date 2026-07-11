from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.research.framework.data import CsvRowRepository


def _row_diagnostics(rows: list[dict[str, Any]], *, requested_symbol_count: int) -> dict[str, Any]:
    rows_by_provider = Counter(str(row.get("provider", "")) for row in rows)
    rows_by_symbol = Counter(str(row.get("symbol", "")).strip().upper() for row in rows)
    provider_symbols: dict[str, set[str]] = defaultdict(set)
    provider_published: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        provider = str(row.get("provider", ""))
        symbol = str(row.get("symbol", "")).strip().upper()
        published = str(row.get("published_at_utc", "")).strip()
        if symbol:
            provider_symbols[provider].add(symbol)
        if published:
            provider_published[provider].append(published)
    provider_symbol_counts = {
        provider: len(symbols)
        for provider, symbols in sorted(provider_symbols.items())
    }
    denominator = requested_symbol_count or 1
    provider_symbol_coverage = {
        provider: count / denominator
        for provider, count in provider_symbol_counts.items()
    }
    provider_published_ranges = {
        provider: {
            "min_published_at_utc": min(values),
            "max_published_at_utc": max(values),
        }
        for provider, values in sorted(provider_published.items())
        if values
    }
    return {
        "rows_by_provider": dict(sorted(rows_by_provider.items())),
        "rows_by_symbol": dict(sorted(rows_by_symbol.items())),
        "provider_symbol_counts": provider_symbol_counts,
        "provider_symbol_coverage": provider_symbol_coverage,
        "published_at_utc_range_by_provider": provider_published_ranges,
        "text_availability_by_provider": _text_availability_by_provider(rows),
    }

def _filter_rows_to_publication_window(
    rows: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    start_boundary = _publication_start_boundary(start_date)
    end_boundary = _publication_end_boundary(end_date)
    accepted: list[dict[str, Any]] = []
    before_start = 0
    after_end = 0
    for row in rows:
        published = str(row.get("published_at_utc", "")).strip()
        if start_boundary and published and published < start_boundary:
            before_start += 1
            continue
        if end_boundary and published and published > end_boundary:
            after_end += 1
            continue
        accepted.append(row)
    rejected = before_start + after_end
    return accepted, {
        "publication_window_start_utc_inclusive": start_boundary,
        "publication_window_end_utc_inclusive": end_boundary,
        "out_of_window_before_start_count": before_start,
        "out_of_window_after_end_count": after_end,
        "out_of_window_rejected_count": rejected,
    }

def _publication_start_boundary(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        return _canonical_utc_text(text)
    return f"{text[:10]}T00:00:00Z"

def _publication_end_boundary(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        return _canonical_utc_text(text)
    return f"{text[:10]}T23:59:59Z"

def _canonical_utc_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _out_of_window_totals(batch_diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    before = sum(int(item.get("out_of_window_before_start_count", 0) or 0) for item in batch_diagnostics)
    after = sum(int(item.get("out_of_window_after_end_count", 0) or 0) for item in batch_diagnostics)
    return {
        "out_of_window_before_start_count": before,
        "out_of_window_after_end_count": after,
        "out_of_window_rejected_count": before + after,
    }

def _text_availability_by_provider(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        provider = str(row.get("provider", "")).strip()
        if provider:
            grouped[provider].append(row)
    return {
        provider: _text_availability(values)
        for provider, values in sorted(grouped.items())
    }

def _text_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    headline_lengths = [
        len(str(row.get("headline", "")).strip())
        for row in rows
        if str(row.get("headline", "")).strip()
    ]
    body_lengths = [
        len(str(row.get("body_or_summary", "")).strip())
        for row in rows
        if str(row.get("body_or_summary", "")).strip()
    ]
    unique_articles = {
        str(row.get("provider_article_id", "")).strip()
        for row in rows
        if str(row.get("provider_article_id", "")).strip()
    }
    updated = [
        row for row in rows
        if str(row.get("updated_at_utc", "")).strip()
    ]
    symbols_by_article: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        article_id = str(row.get("provider_article_id", "")).strip()
        symbol = str(row.get("symbol", "")).strip().upper()
        if article_id and symbol:
            symbols_by_article[article_id].add(symbol)
    multi_symbol_articles = sum(1 for symbols in symbols_by_article.values() if len(symbols) > 1)
    multi_symbol_expansion = sum(max(0, len(symbols) - 1) for symbols in symbols_by_article.values())
    return {
        "article_symbol_rows": len(rows),
        "unique_article_count": len(unique_articles),
        "headline_availability_rate": len(headline_lengths) / len(rows) if rows else 0.0,
        "summary_availability_rate": len(body_lengths) / len(rows) if rows else 0.0,
        "content_availability_rate": len(body_lengths) / len(rows) if rows else 0.0,
        "updated_timestamp_availability_rate": len(updated) / len(rows) if rows else 0.0,
        "headline_length_distribution": _length_distribution(headline_lengths),
        "summary_length_distribution": _length_distribution(body_lengths),
        "content_length_distribution": _length_distribution(body_lengths),
        "source_distribution": dict(sorted(Counter(str(row.get("source", "")).strip() for row in rows).items())),
        "multi_symbol_article_count": multi_symbol_articles,
        "multi_symbol_expansion_row_count": multi_symbol_expansion,
    }

def _length_distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.5),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }

def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * quantile)
    return values[max(0, min(len(values) - 1, index))]

def _duplicate_diagnostics(
    collected_rows: list[dict[str, Any]],
    deduplicated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    article_symbols: dict[tuple[str, str], set[str]] = defaultdict(set)
    exact_record_keys: Counter[str] = Counter()
    headline_provider_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    cross_provider_headlines: dict[str, set[str]] = defaultdict(set)
    for row in collected_rows:
        provider = str(row.get("provider", "")).strip()
        provider_id = str(row.get("provider_article_id", "")).strip()
        symbol = str(row.get("symbol", "")).strip().upper()
        published = str(row.get("published_at_utc", "")).strip()
        headline = _normalized_headline(row.get("headline", ""))
        if provider and provider_id and symbol:
            article_symbols[(provider, provider_id)].add(symbol)
        if provider and provider_id:
            exact_record_keys["|".join((provider, provider_id, symbol, published))] += 1
        if provider and provider_id and headline:
            headline_provider_ids[(provider, headline)].add(provider_id)
        if provider and headline:
            cross_provider_headlines[headline].add(provider)
    multi_symbol_row_count = sum(max(0, len(symbols) - 1) for symbols in article_symbols.values())
    exact_duplicate_record_count = sum(max(0, count - 1) for count in exact_record_keys.values())
    same_headline_different_provider_id_count = sum(
        1 for provider_ids in headline_provider_ids.values() if len(provider_ids) > 1
    )
    cross_provider_identical_headline_count = sum(
        1 for providers in cross_provider_headlines.values() if len(providers) > 1
    )
    legacy_headlines = [
        _normalized_headline(row.get("headline", ""))
        for row in deduplicated_rows
        if _normalized_headline(row.get("headline", ""))
    ]
    duplicate_headline_count = len(legacy_headlines) - len(set(legacy_headlines))
    return {
        "same_provider_article_id_multi_symbol_row_count": multi_symbol_row_count,
        "exact_duplicate_provider_article_record_count": exact_duplicate_record_count,
        "same_headline_different_provider_article_id_count": same_headline_different_provider_id_count,
        "cross_provider_identical_normalized_headline_count": cross_provider_identical_headline_count,
        "near_duplicate_story_count": 0,
        "near_duplicate_story_detection": "not_run",
        "duplicate_headline_count": same_headline_different_provider_id_count + cross_provider_identical_headline_count,
        "duplicate_headline_rate": (
            (same_headline_different_provider_id_count + cross_provider_identical_headline_count) / len(deduplicated_rows)
            if deduplicated_rows
            else 0.0
        ),
        "legacy_row_based_duplicate_headline_count": duplicate_headline_count,
        "legacy_row_based_duplicate_headline_rate": (
            duplicate_headline_count / len(deduplicated_rows)
            if deduplicated_rows
            else 0.0
        ),
    }

def _normalized_headline(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())

def _existing_output_overlap(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not output_path.exists() or not rows:
        return {
            "checked": output_path.exists(),
            "existing_path": str(output_path),
            "overlap_count": 0,
            "overlap_provider_article_id_count": 0,
            "overlap_normalized_headline_count": 0,
        }
    try:
        existing_rows = CsvRowRepository().read(output_path)
    except (OSError, ValueError):
        return {
            "checked": False,
            "existing_path": str(output_path),
            "overlap_count": 0,
            "overlap_provider_article_id_count": 0,
            "overlap_normalized_headline_count": 0,
        }
    existing_provider_ids = {
        str(row.get("provider_article_id", "")).strip()
        for row in existing_rows
        if str(row.get("provider_article_id", "")).strip()
    }
    existing_headlines = {
        str(row.get("headline", "")).strip().lower()
        for row in existing_rows
        if str(row.get("headline", "")).strip()
    }
    provider_overlap = sum(
        1
        for row in rows
        if str(row.get("provider_article_id", "")).strip() in existing_provider_ids
    )
    headline_overlap = sum(
        1
        for row in rows
        if str(row.get("headline", "")).strip().lower() in existing_headlines
    )
    return {
        "checked": True,
        "existing_path": str(output_path),
        "existing_row_count": len(existing_rows),
        "overlap_count": max(provider_overlap, headline_overlap),
        "overlap_provider_article_id_count": provider_overlap,
        "overlap_normalized_headline_count": headline_overlap,
    }
