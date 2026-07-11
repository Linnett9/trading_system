"""Non-destructive canonical story grouping for stock-alpha news."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Iterable

from core.research.ml.stock_level.news_sources.canonical import (
    CanonicalNewsRecord,
    STORY_GROUPING_VERSION,
)
from core.research.ml.stock_level.news_sources.normalization import (
    normalize_headline,
    normalize_source_name,
    normalize_symbol,
    parse_utc_timestamp,
)


@dataclass(frozen=True)
class DuplicateGroupMetadata:
    """Metadata attached to a record without deleting any row."""

    canonical_duplicate_group_id: str
    duplicate_group_size: int
    earliest_publication_timestamp: str | None
    latest_update_timestamp: str | None
    symbols: tuple[str, ...]
    provider_article_ids: tuple[str, ...]
    provider_urls: tuple[str, ...]
    grouping_method: str
    grouping_version: str = STORY_GROUPING_VERSION


@dataclass(frozen=True)
class GroupedNewsRecord:
    """A canonical news record plus deterministic grouping metadata."""

    record: CanonicalNewsRecord
    duplicate: DuplicateGroupMetadata


def exact_provider_record_key(record: CanonicalNewsRecord) -> str:
    """Identity for exact provider record per symbol."""

    return _stable_id(
        "exact-provider-record",
        record.provider,
        record.provider_article_id or "",
        record.symbol,
    )


def likely_publication_key(record: CanonicalNewsRecord, *, window_minutes: int = 5) -> str:
    """Identity for likely same publication while preserving symbols."""

    published = parse_utc_timestamp(record.published_at_utc)
    bucket = _time_bucket(published, window_minutes=window_minutes)
    source = normalize_source_name(record.source) or normalize_source_name(record.provider) or ""
    headline = normalize_headline(record.headline) or ""
    symbol = normalize_symbol(record.symbol) or ""
    return _stable_id("likely-publication", headline, bucket, symbol, source)


def cross_symbol_story_key(record: CanonicalNewsRecord, *, window_minutes: int = 5) -> str:
    """Identity for one story that may legitimately map to many symbols."""

    published = parse_utc_timestamp(record.published_at_utc)
    bucket = _time_bucket(published, window_minutes=window_minutes)
    source = normalize_source_name(record.source) or normalize_source_name(record.provider) or ""
    headline = normalize_headline(record.headline) or ""
    return _stable_id("cross-symbol-story", headline, bucket, source)


def group_news_records(
    records: Iterable[CanonicalNewsRecord],
    *,
    method: str = "likely_publication",
    window_minutes: int = 5,
) -> list[GroupedNewsRecord]:
    """Attach duplicate metadata without dropping records."""

    materialized = list(records)
    groups: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
    for record in materialized:
        if method == "exact_provider_record":
            key = exact_provider_record_key(record)
        elif method == "cross_symbol_story":
            key = cross_symbol_story_key(record, window_minutes=window_minutes)
        elif method == "likely_publication":
            key = likely_publication_key(record, window_minutes=window_minutes)
        else:
            raise ValueError(f"unsupported grouping method: {method}")
        groups[key].append(record)

    grouped: list[GroupedNewsRecord] = []
    for key, members in groups.items():
        metadata = _metadata_for_group(key, members, method=method)
        grouped.extend(GroupedNewsRecord(record=member, duplicate=metadata) for member in members)
    return grouped


def _metadata_for_group(
    group_id: str,
    members: list[CanonicalNewsRecord],
    *,
    method: str,
) -> DuplicateGroupMetadata:
    published = [_parse_optional(record.published_at_utc) for record in members]
    updated = [_parse_optional(record.updated_at_utc) for record in members]
    published_values = [value for value in published if value is not None]
    updated_values = [value for value in updated if value is not None]
    symbols = sorted({normalize_symbol(record.symbol) or record.symbol for record in members})
    provider_ids = sorted({record.provider_article_id for record in members if record.provider_article_id})
    urls = sorted({record.provider_url for record in members if record.provider_url})
    return DuplicateGroupMetadata(
        canonical_duplicate_group_id=group_id,
        duplicate_group_size=len(members),
        earliest_publication_timestamp=_format(min(published_values)) if published_values else None,
        latest_update_timestamp=_format(max(updated_values)) if updated_values else None,
        symbols=tuple(symbols),
        provider_article_ids=tuple(provider_ids),
        provider_urls=tuple(urls),
        grouping_method=method,
    )


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts)
    return sha256(payload.encode("utf-8")).hexdigest()


def _time_bucket(value: datetime | None, *, window_minutes: int) -> str:
    if value is None:
        return "unknown"
    seconds = window_minutes * 60
    bucket = int(value.timestamp()) // seconds
    return str(bucket)


def _parse_optional(value: str | None) -> datetime | None:
    return parse_utc_timestamp(value) if value else None


def _format(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
