from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


def _alpaca_time(value: str, *, end_of_day: bool) -> str:
    text = str(value or "").strip()
    if "T" in text:
        parsed = _utc(text)
        return parsed or text
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return f"{text[:10]}{suffix}"

def _article_symbols(item: Mapping[str, Any], requested_symbols: list[str]) -> list[str]:
    raw_symbols = item.get("symbols") or item.get("tickers") or []
    if isinstance(raw_symbols, str):
        raw_symbols = [raw_symbols]
    available = {
        str(symbol).strip().upper()
        for symbol in raw_symbols or []
        if str(symbol).strip()
    }
    requested = {symbol.upper() for symbol in requested_symbols}
    if not available:
        return sorted(requested)
    return sorted(available & requested)

def _alpaca_publisher(item: Mapping[str, Any]) -> str:
    source = item.get("source")
    if isinstance(source, Mapping):
        value = str(source.get("name") or source.get("source") or "").strip()
        return value
    value = str(source or "").strip()
    if value.lower() == "benzinga":
        return value
    if value:
        return ""
    return "Benzinga"

def _alpaca_author(item: Mapping[str, Any]) -> str:
    author = item.get("author")
    if isinstance(author, Mapping):
        return str(author.get("name") or author.get("author") or "").strip()
    value = str(author or "").strip()
    if value:
        return value
    source = item.get("source")
    raw_source = "" if isinstance(source, Mapping) else str(source or "").strip()
    if raw_source and raw_source.lower() != "benzinga":
        return raw_source
    return ""

def _alpaca_raw_source(item: Mapping[str, Any]) -> str:
    source = item.get("source")
    if isinstance(source, Mapping):
        return json.dumps(source, sort_keys=True)
    return str(source or "").strip()

def _alpaca_batch_diagnostic(
    *,
    rows: list[dict[str, Any]],
    requested_symbols: list[str],
    request_urls: list[str],
    page_count: int,
    pages_completed: int,
    response_rows: int,
    rejected: Mapping[str, int],
    next_page_token: str,
    multi_symbol_articles: int,
    source_distribution: Mapping[str, int],
    secret_key_env: str,
    page_token_start: str,
    stopped_by_page_limit: bool,
    retry_count: int = 0,
    max_retries: int = 0,
    termination_reason: str = "",
    provider_record_ids: list[str] | None = None,
    provider_record_headlines: Mapping[str, str] | None = None,
    page_size: int = 0,
) -> dict[str, Any]:
    published = sorted(str(row.get("published_at_utc", "")) for row in rows if row.get("published_at_utc"))
    headlines = [str(row.get("headline", "")).strip().lower() for row in rows if str(row.get("headline", "")).strip()]
    summaries = [row for row in rows if str(row.get("body_or_summary", "")).strip()]
    updated = [row for row in rows if str(row.get("updated_at_utc", "")).strip()]
    symbols = [str(row.get("symbol", "")).strip().upper() for row in rows if str(row.get("symbol", "")).strip()]
    provider_ids = [str(row.get("provider_article_id", "")).strip() for row in rows if str(row.get("provider_article_id", "")).strip()]
    record_ids = [value for value in (provider_record_ids or []) if value]
    article_symbols: dict[str, set[str]] = {}
    for row in rows:
        provider_id = str(row.get("provider_article_id", "")).strip()
        symbol = str(row.get("symbol", "")).strip().upper()
        if provider_id and symbol:
            article_symbols.setdefault(provider_id, set()).add(symbol)
    multi_symbol_expansion_rows = sum(max(0, len(values) - 1) for values in article_symbols.values())
    headline_ids: dict[str, set[str]] = {}
    for provider_id, headline in (provider_record_headlines or {}).items():
        normalized_headline = str(headline or "").strip().lower()
        if normalized_headline and provider_id:
            headline_ids.setdefault(normalized_headline, set()).add(str(provider_id))
    same_headline_different_ids = sum(1 for values in headline_ids.values() if len(values) > 1)
    exact_duplicate_provider_records = len(record_ids) - len(set(record_ids))
    next_token_present = bool(next_page_token)
    return {
        "alpaca_benzinga_requested_symbols": requested_symbols,
        "alpaca_benzinga_request_urls": request_urls,
        "alpaca_benzinga_page_size": page_size,
        "alpaca_benzinga_termination_reason": termination_reason,
        "alpaca_benzinga_pages_requested": page_count,
        "alpaca_benzinga_pages_completed": pages_completed,
        "alpaca_benzinga_next_page_token_present_at_stop": next_token_present,
        "alpaca_benzinga_stopped_with_more_results_available": next_token_present and termination_reason not in {"end_of_results", "empty_page"},
        "alpaca_benzinga_records_returned": response_rows,
        "alpaca_benzinga_provider_records_returned": response_rows,
        "alpaca_benzinga_records_accepted": len(rows),
        "alpaca_benzinga_records_rejected": sum(int(value) for value in rejected.values()),
        "alpaca_benzinga_rejected_reasons": dict(sorted(rejected.items())),
        "alpaca_benzinga_unique_provider_articles": len(set(record_ids or provider_ids)),
        "alpaca_benzinga_unique_provider_article_ids": len(set(provider_ids)),
        "alpaca_benzinga_article_symbol_rows": len(rows),
        "alpaca_benzinga_multi_symbol_expansion_row_count": multi_symbol_expansion_rows,
        "alpaca_benzinga_same_provider_article_id_multi_symbol_row_count": multi_symbol_expansion_rows,
        "alpaca_benzinga_exact_duplicate_provider_record_count": exact_duplicate_provider_records,
        "alpaca_benzinga_duplicate_provider_article_id_count": exact_duplicate_provider_records,
        "alpaca_benzinga_same_headline_different_provider_article_id_count": same_headline_different_ids,
        "alpaca_benzinga_unique_symbols": len(set(symbols)),
        "alpaca_benzinga_article_counts_by_symbol": dict(sorted(Counter(symbols).items())),
        "alpaca_benzinga_earliest_published_at_utc": published[0] if published else "",
        "alpaca_benzinga_latest_published_at_utc": published[-1] if published else "",
        "alpaca_benzinga_headline_availability_rate": len(headlines) / len(rows) if rows else 0.0,
        "alpaca_benzinga_summary_availability_rate": len(summaries) / len(rows) if rows else 0.0,
        "alpaca_benzinga_content_availability_rate": len(summaries) / len(rows) if rows else 0.0,
        "alpaca_benzinga_updated_timestamp_rate": len(updated) / len(rows) if rows else 0.0,
        "alpaca_benzinga_multi_symbol_article_count": multi_symbol_articles,
        "alpaca_benzinga_multi_symbol_article_rate": multi_symbol_articles / response_rows if response_rows else 0.0,
        "alpaca_benzinga_source_distribution": dict(sorted(source_distribution.items())),
        "alpaca_benzinga_exact_normalized_headline_duplicate_count": same_headline_different_ids,
        "alpaca_benzinga_next_page_token": next_page_token,
        "alpaca_benzinga_page_token_start": page_token_start,
        "alpaca_benzinga_stopped_by_page_limit": stopped_by_page_limit,
        "alpaca_benzinga_secret_key_env": secret_key_env,
        "alpaca_benzinga_retry_count": retry_count,
        "alpaca_benzinga_max_retries": max_retries,
        "pit_semantics": {
            "published_at_utc": "provider article publication timestamp",
            "updated_at_utc": "provider article update timestamp when supplied",
            "ingested_at": "local backfill collection time",
            "availability": "historical provider availability is not proven by this adapter",
        },
    }

def _utc(value: Any) -> str:
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(candidate)
                break
            except ValueError:
                parsed = None
        if parsed is None and len(text) >= 14 and text[:14].isdigit():
            parsed = datetime.strptime(text[:14], "%Y%m%d%H%M%S")
        if parsed is None:
            compact = text.replace("T", "").replace("Z", "").replace("-", "").replace(":", "")
            if len(compact) >= 14 and compact[:14].isdigit():
                parsed = datetime.strptime(compact[:14], "%Y%m%d%H%M%S")
        if parsed is None and text:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError, OverflowError):
                parsed = None
        if parsed is None:
            return ""
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
