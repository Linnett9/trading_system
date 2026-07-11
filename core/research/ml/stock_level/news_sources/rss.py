from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree

from core.research.ml.stock_level.news_sources.alpaca import _utc
from core.research.ml.stock_level.news_sources.normalization import normalize_url


FixtureRssFetch = Callable[[Mapping[str, Any]], Any]


class FixtureRssProviderAdapter:
    """Fixture-only RSS provider-like adapter for scratch dry-runs.

    This adapter accepts explicit feed metadata and an injected fake fetcher.
    It does not create a network client, read config, or read credentials.
    """

    def __init__(
        self,
        *,
        feeds: Sequence[Mapping[str, Any]],
        fetcher: FixtureRssFetch,
        provider_id: str = "company_press_release_rss",
        provider_family: str = "official_company_rss_fixture",
        collected_at_utc: str = "2026-07-10T00:00:00Z",
    ) -> None:
        self.provider_id = _text(provider_id) or "company_press_release_rss"
        self.provider_family = _text(provider_family) or "official_company_rss_fixture"
        self._feeds = _normalise_fixture_feeds(feeds)
        self._fetcher = fetcher
        self._collected_at_utc = _utc(collected_at_utc)
        self.fetch_calls: list[dict[str, Any]] = []

    def collect(
        self,
        *,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Map fake RSS feed payloads into compatibility rows."""

        requested = {_text(symbol).upper() for symbol in symbols if _text(symbol)}
        rows: list[dict[str, Any]] = []
        for feed in self._feeds:
            if len(rows) >= limit:
                break
            symbol = _text(feed.get("symbol")).upper()
            if requested and symbol not in requested:
                continue
            self.fetch_calls.append({"symbol": symbol, "url": _text(feed.get("url"))})
            for item in _fixture_rss_items(self._fetcher(dict(feed))):
                if len(rows) >= limit:
                    break
                if not _rss_item_in_date_window(item, start_date=start_date, end_date=end_date):
                    continue
                rows.append(self._compatibility_row(feed=feed, item=item))
        return sorted(rows, key=_fixture_row_sort_key)[: max(0, int(limit))]

    def _compatibility_row(
        self,
        *,
        feed: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> dict[str, Any]:
        symbol = _text(feed.get("symbol")).upper()
        url = _text(item.get("link") or item.get("url") or item.get("provider_url"))
        title = _text(item.get("title") or item.get("headline"))
        summary = _text(item.get("summary") or item.get("description") or item.get("body"))
        published = _text(
            item.get("published")
            or item.get("pubDate")
            or item.get("published_at_utc")
            or item.get("updated")
        )
        published_utc = _utc(published) if published else ""
        source = _text(item.get("source") or feed.get("source") or feed.get("name") or _domain(url) or self.provider_id)
        provider_article_id = _text(item.get("id") or item.get("guid") or item.get("provider_article_id"))
        if not provider_article_id:
            provider_article_id = "rss:" + hashlib.sha256(
                f"{self.provider_id}|{symbol}|{url}|{title}|{published_utc}".encode("utf-8")
            ).hexdigest()[:24]
        return {
            "article_id": f"{self.provider_id}:{provider_article_id}:{symbol}",
            "provider": self.provider_id,
            "provider_article_id": provider_article_id,
            "provider_original_article_id": _text(item.get("provider_original_article_id") or provider_article_id),
            "provider_symbols": symbol,
            "symbol": symbol,
            "published_at_utc": published_utc,
            "provider_available_at_utc": _text(item.get("provider_available_at_utc")),
            "collected_at_utc": self._collected_at_utc,
            "source": source,
            "source_type": _text(item.get("source_type") or feed.get("source_type") or "company_rss"),
            "headline": title,
            "summary": summary,
            "body_or_full_text": _text(item.get("body_or_full_text") or item.get("content") or summary),
            "language": _text(item.get("language") or feed.get("language")),
            "event_type": _fixture_event_type(item=item, feed=feed),
            "form_type": _text(item.get("form_type") or feed.get("form_type")),
            "provider_url": url,
            "normalized_provider_url": normalize_url(url),
            "original_source_url": _text(feed.get("url")),
            "relevance_status": _text(item.get("relevance_status") or feed.get("relevance_status")),
        }


def _normalise_rss_feed_registry(feeds: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for symbol, entries in feeds.items():
        symbol_key = str(symbol).strip().upper()
        if not symbol_key:
            continue
        if isinstance(entries, Mapping):
            entries = [entries]
        normalized[symbol_key] = [
            dict(entry)
            for entry in entries or []
            if isinstance(entry, Mapping)
        ]
    return normalized

def _rss_feed_diagnostic(
    *,
    provider: str,
    symbol: str,
    feed_name: str = "",
    feed_url: str = "",
    zero_row_reason: str = "",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "symbol": symbol,
        "feed_name": feed_name,
        "feed_url": _redact_url(feed_url),
        "response_row_count": 0,
        "normalized_row_count": 0,
        "zero_row_reason": zero_row_reason,
        "error_type": "",
        "error_message": "",
        "rate_limited": False,
    }

def _validate_rss_feed_url(feed_url: str) -> None:
    parsed = urlparse(feed_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid RSS feed URL")

def _rss_items(payload: Any) -> list[dict[str, str]]:
    text = _rss_payload_text(payload).strip()
    if not text:
        return []
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError("invalid RSS/XML feed") from exc
    feed_language = _first_descendant_text(root, "language")
    entries = _descendants(root, "item")
    if not entries:
        entries = _descendants(root, "entry")
    return [
        {
            "title": _direct_child_text(entry, "title"),
            "link": _entry_link(entry),
            "published": (
                _direct_child_text(entry, "pubDate")
                or _direct_child_text(entry, "published")
                or _direct_child_text(entry, "updated")
                or _direct_child_text(entry, "date")
            ),
            "summary": (
                _direct_child_text(entry, "description")
                or _direct_child_text(entry, "summary")
                or _direct_child_text(entry, "content")
            ),
            "language": _direct_child_text(entry, "language") or feed_language,
        }
        for entry in entries
    ]

def _rss_payload_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload or "")

def _rss_item_in_date_window(item: Mapping[str, Any], *, start_date: str, end_date: str) -> bool:
    published = _utc(item.get("published", ""))
    if not published:
        return True
    published_date = published[:10]
    return (
        (not start_date or published_date >= start_date)
        and (not end_date or published_date <= end_date)
    )

def _descendants(root: ElementTree.Element, local_name: str) -> list[ElementTree.Element]:
    return [
        element for element in root.iter()
        if _local_name(element.tag) == local_name.lower()
    ]

def _direct_child_text(root: ElementTree.Element, local_name: str) -> str:
    for child in list(root):
        if _local_name(child.tag) == local_name.lower():
            return "".join(child.itertext()).strip()
    return ""

def _first_descendant_text(root: ElementTree.Element, local_name: str) -> str:
    for element in _descendants(root, local_name):
        text = "".join(element.itertext()).strip()
        if text:
            return text
    return ""

def _entry_link(root: ElementTree.Element) -> str:
    for child in list(root):
        if _local_name(child.tag) == "link":
            href = str(child.attrib.get("href", "")).strip()
            if href:
                return href
            return "".join(child.itertext()).strip()
    return ""

def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()

def _domain(url: str) -> str:
    return urlparse(str(url or "")).netloc

def _redact_url(url: str) -> str:
    parsed = urlparse(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return str(url or "")
    return parsed._replace(query="", fragment="").geturl()

def _looks_rate_limited(message: str) -> bool:
    lowered = message.lower()
    return (
        "429" in lowered
        or "too many" in lowered
        or "frequency" in lowered
        or ("rate" in lowered and "limit" in lowered)
    )

def _normalise_fixture_feeds(feeds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, feed in enumerate(feeds, start=1):
        if not isinstance(feed, Mapping):
            raise ValueError(f"feed {index} is not a mapping")
        row = dict(feed)
        symbol = _text(row.get("symbol")).upper()
        if not symbol:
            raise ValueError("fixture RSS feeds require explicit symbol metadata")
        row["symbol"] = symbol
        normalized.append(row)
    return normalized

def _fixture_rss_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        values = payload.get("items", payload.get("entries", [payload]))
        return [_fixture_item(item) for item in _iter_fixture_items(values)]
    if isinstance(payload, list) or isinstance(payload, tuple):
        return [_fixture_item(item) for item in _iter_fixture_items(payload)]
    return [dict(item) for item in _rss_items(payload)]

def _iter_fixture_items(values: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(values, Mapping):
        values = [values]
    for item in values or []:
        if not isinstance(item, Mapping):
            raise ValueError("fixture RSS item is not a mapping")
        yield item

def _fixture_item(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    row.setdefault("title", row.get("headline", ""))
    row.setdefault("link", row.get("url", row.get("provider_url", "")))
    row.setdefault("published", row.get("published_at_utc", row.get("pubDate", "")))
    row.setdefault("summary", row.get("description", row.get("body", "")))
    return row

def _fixture_event_type(*, item: Mapping[str, Any], feed: Mapping[str, Any]) -> str:
    if "event_type" in item:
        return _text(item.get("event_type"))
    if _text(item.get("source_type") or feed.get("source_type")).lower() in {
        "sec",
        "sec_filing",
        "sec_edgar",
        "filing",
    }:
        return ""
    return _text(feed.get("event_type"))

def _fixture_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row.get("published_at_utc")),
        _text(row.get("provider")),
        _text(row.get("provider_article_id")),
        _text(row.get("symbol")),
    )

def _text(value: Any) -> str:
    return str(value or "").strip()
