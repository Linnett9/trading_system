from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


from core.research.ml.stock_level.news_sources.alpaca import (
    _alpaca_author,
    _alpaca_batch_diagnostic,
    _alpaca_publisher,
    _alpaca_raw_source,
    _alpaca_time,
    _article_symbols,
    _utc,
)
from core.research.ml.stock_level.news_sources.gdelt import (
    _alpha_vantage_news_time,
    _gdelt_query_text,
    GDELT_AMBIGUOUS_SYMBOLS,
    GDELT_COMPANY_QUERY_TERMS,
    gdelt_query_terms,
)
from core.research.ml.stock_level.news_sources.rss import (
    _looks_rate_limited,
    _normalise_rss_feed_registry,
    _redact_url,
    _rss_feed_diagnostic,
    _rss_item_in_date_window,
    _rss_items,
    _validate_rss_feed_url,
)

HttpGet = Callable[[str, int], Any]
AlpacaHttpGet = Callable[[str, int, Mapping[str, str]], Any]
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

PROVIDER_METADATA = {
    "alpha_vantage": {"statuses": ["usable_now", "needs_key"], "api_key_required": True},
    "alpaca_benzinga": {"statuses": ["usable_now", "needs_key", "historical_editorial_news"], "api_key_required": True},
    "finnhub": {"statuses": ["usable_now", "needs_key"], "api_key_required": True},
    "gdelt": {"statuses": ["experimental_no_key", "rate_limited_or_retry_later"], "api_key_required": False},
    "sec_edgar": {"statuses": ["official_filings_source", "experimental_no_key"], "api_key_required": False},
    "sec_company_filings": {"statuses": ["official_filings_source", "dry_run_candidate"], "api_key_required": False},
    "massive_stock_news": {"statuses": ["dry_run_candidate", "needs_key"], "api_key_required": True},
    "company_press_release_rss": {"statuses": ["official_company_source_candidate", "dry_run_candidate"], "api_key_required": False},
    "fmp": {"statuses": ["disabled_payment_required", "needs_key"], "api_key_required": True},
    "newsapi": {"statuses": ["disabled_upgrade_required", "needs_key"], "api_key_required": True},
}

SEC_CIK_BY_SYMBOL = {
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "AMZN": "0001018724", "GOOGL": "0001652044", "META": "0001326801",
    "TSLA": "0001318605", "AVGO": "0001730168", "BRK.B": "0001067983",
    "JPM": "0000019617", "V": "0001403161", "MA": "0001141391",
    "XOM": "0000034088", "UNH": "0000731766", "COST": "0000909832",
    "HD": "0000354950", "PG": "0000080424", "JNJ": "0000200406",
    "ABBV": "0001551152", "NFLX": "0001065280", "CRM": "0001108524",
    "AMD": "0000002488", "ORCL": "0001341439", "BAC": "0000070858",
    "KO": "0000021344", "PEP": "0000077476", "WMT": "0000104169",
    "CVX": "0000093410", "MRK": "0000310158", "CSCO": "0000858877",
}


def standard_library_json_get(url: str, timeout: int) -> Any:
    request = Request(url, headers={"User-Agent": "stock-alpha-research/1.0 research-contact@example.invalid"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - bounded configured research endpoints
        return json.loads(response.read().decode("utf-8"))


def standard_library_text_get(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "stock-alpha-research/1.0 research-contact@example.invalid"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - bounded configured research endpoints
        return response.read().decode("utf-8", errors="replace")


def standard_library_alpaca_json_get(
    url: str,
    timeout: int,
    headers: Mapping[str, str],
) -> Any:
    request = Request(
        url,
        headers={
            "User-Agent": "stock-alpha-research/1.0 research-contact@example.invalid",
            **dict(headers),
        },
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - bounded configured research endpoints
        return json.loads(response.read().decode("utf-8"))


class NewsSource:
    name = ""
    api_key_required = True

    def __init__(self, http_get: HttpGet = standard_library_json_get) -> None:
        self._http_get = http_get

    def collect(self, *, symbols: list[str], start_date: str, end_date: str, limit: int, timeout: int, api_key: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        per_symbol_limit = max(1, math.ceil(limit / len(symbols))) if symbols else limit
        for symbol in symbols:
            payload = self._http_get(self._url(symbol, start_date, end_date, per_symbol_limit, api_key), timeout)
            rows.extend(self._rows(payload, symbol)[:per_symbol_limit])
            if len(rows) >= limit:
                break
        return rows[:limit]

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        raise NotImplementedError

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _normalized(self, *, symbol: str, provider_id: Any, url: Any, published: Any, source: Any, headline: Any, body: Any, sentiment: Any = "", relevance: Any = "", novelty: Any = "", event_type: Any = "", language: Any = "en", ingested: Any = "") -> dict[str, Any]:
        published_utc = _utc(published)
        article_id = str(provider_id or "").strip() or hashlib.sha256(f"{self.name}|{symbol}|{url}|{headline}|{published_utc}".encode()).hexdigest()[:24]
        return {
            "article_id": f"{self.name}:{article_id}", "symbol": symbol.upper(),
            "published_at_utc": published_utc, "source": str(source or self.name),
            "headline": str(headline or ""), "body_or_summary": str(body or ""),
            "sentiment_score": _blank_or_value(sentiment), "relevance_score": _blank_or_value(relevance),
            "novelty_score": _blank_or_value(novelty), "event_type": str(event_type or ""),
            "language": str(language or ""), "ingested_at": _utc(ingested) if ingested else _utc(datetime.now(timezone.utc)),
            "provider": self.name, "provider_article_id": str(provider_id or article_id), "provider_url": str(url or ""),
        }


class ProviderRateLimitError(RuntimeError):
    code = 429


class GdeltNewsSource(NewsSource):
    name, api_key_required = "gdelt", False

    def __init__(self, http_get: HttpGet = standard_library_json_get) -> None:
        super().__init__(http_get)
        self.last_batch_diagnostic: dict[str, Any] = {}

    def collect(self, *, symbols: list[str], start_date: str, end_date: str, limit: int, timeout: int, api_key: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        query_terms: dict[str, list[str]] = {}
        skipped_symbols: dict[str, str] = {}
        queryable = []
        for symbol in symbols:
            terms = gdelt_query_terms(symbol)
            if not terms:
                skipped_symbols[symbol.upper()] = "skipped_ambiguous_symbol"
                continue
            query_terms[symbol.upper()] = terms
            queryable.append(symbol)
        self.last_batch_diagnostic = {
            "query_terms": query_terms,
            "skipped_symbols": skipped_symbols,
        }
        per_symbol_limit = max(1, math.ceil(limit / len(queryable))) if queryable else limit
        for symbol in queryable:
            payload = self._http_get(self._url(symbol, start_date, end_date, per_symbol_limit, api_key), timeout)
            rows.extend(self._rows(payload, symbol)[:per_symbol_limit])
            if len(rows) >= limit:
                break
        return rows[:limit]

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        query_text = _gdelt_query_text(gdelt_query_terms(symbol))
        query = urlencode({"query": query_text, "mode": "ArtList", "format": "json", "maxrecords": min(limit, 250), "startdatetime": start.replace("-", "") + "000000", "enddatetime": end.replace("-", "") + "235959"})
        return f"https://api.gdeltproject.org/api/v2/doc/doc?{query}"

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError("unexpected GDELT response shape")
        message = str(payload.get("message") or payload.get("error") or "")
        if message:
            if _looks_rate_limited(message):
                raise ProviderRateLimitError(message)
            raise ValueError(message)
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("seendate"), source=item.get("domain", "gdelt"), headline=item.get("title"), body=item.get("snippet", ""), sentiment="", relevance="", novelty="", event_type="news", language=item.get("language", "")) for item in payload.get("articles", [])]


class AlphaVantageNewsSource(NewsSource):
    name = "alpha_vantage"

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        params: dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": limit,
            "apikey": api_key,
        }
        if start:
            params["time_from"] = _alpha_vantage_news_time(start, end_of_day=False)
        if end:
            params["time_to"] = _alpha_vantage_news_time(end, end_of_day=True)
        return "https://www.alphavantage.co/query?" + urlencode(params)

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if isinstance(payload, Mapping):
            message = str(payload.get("Note") or payload.get("Information") or "")
            if message and any(token in message.lower() for token in ("rate", "frequency", "limit")):
                raise ProviderRateLimitError(message)
            error_message = str(payload.get("Error Message") or "")
            if error_message:
                raise ValueError(error_message)
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("time_published"), source=item.get("source"), headline=item.get("title"), body=item.get("summary"), sentiment=item.get("overall_sentiment_score", ""), event_type="", language="en") for item in payload.get("feed", [])]


class AlpacaBenzingaNewsSource(NewsSource):
    name = "alpaca_benzinga"

    def __init__(
        self,
        http_get: AlpacaHttpGet = standard_library_alpaca_json_get,
        *,
        secret_key_env: str = "ALPACA_SECRET_KEY",
        page_token: str = "",
        page_size: int = 0,
        max_pages_per_batch: int = 0,
        max_retries: int = 1,
        retry_sleep_seconds: float = 0.0,
    ) -> None:
        self._alpaca_http_get = http_get
        self._secret_key_env = secret_key_env
        self._page_token = page_token
        self._page_size = max(0, int(page_size or 0))
        self._max_pages_per_batch = max(0, int(max_pages_per_batch or 0))
        self._max_retries = max(0, int(max_retries or 0))
        self._retry_sleep_seconds = max(0.0, float(retry_sleep_seconds or 0.0))
        self.last_batch_diagnostic: dict[str, Any] = {}
        super().__init__(standard_library_json_get)

    def with_provider_config(self, provider_config: Mapping[str, Any]) -> "AlpacaBenzingaNewsSource":
        return AlpacaBenzingaNewsSource(
            self._alpaca_http_get,
            secret_key_env=str(
                provider_config.get("secret_key_env")
                or provider_config.get("api_secret_env")
                or self._secret_key_env
            ).strip() or self._secret_key_env,
            page_token=str(provider_config.get("page_token", self._page_token) or "").strip(),
            page_size=int(
                provider_config.get("page_size")
                or provider_config.get("request_page_size")
                or self._page_size
                or 0
            ),
            max_pages_per_batch=int(provider_config.get("max_pages_per_batch", self._max_pages_per_batch) or 0),
            max_retries=int(provider_config.get("max_retries", self._max_retries) or 0),
            retry_sleep_seconds=float(provider_config.get("retry_sleep_seconds", self._retry_sleep_seconds) or 0.0),
        )

    def collect(
        self,
        *,
        symbols: list[str],
        start_date: str,
        end_date: str,
        limit: int,
        timeout: int,
        api_key: str = "",
    ) -> list[dict[str, Any]]:
        secret_key = os.environ.get(self._secret_key_env, "")
        if not secret_key:
            raise ValueError(f"missing Alpaca secret key environment variable: {self._secret_key_env}")
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        requested_symbols = [symbol.upper() for symbol in symbols if str(symbol).strip()]
        rows: list[dict[str, Any]] = []
        rejected: Counter[str] = Counter()
        response_rows = 0
        page_count = 0
        pages_completed = 0
        page_token = self._page_token
        next_page_token = ""
        request_urls: list[str] = []
        multi_symbol_articles = 0
        source_distribution: Counter[str] = Counter()
        retry_count = 0
        provider_record_ids: list[str] = []
        provider_record_headlines: dict[str, str] = {}
        termination_reason = ""
        while len(rows) < limit:
            if self._max_pages_per_batch and page_count >= self._max_pages_per_batch:
                termination_reason = "max_pages"
                break
            remaining_rows = limit - len(rows)
            configured_page_size = self._page_size or remaining_rows
            page_limit = max(1, min(50, configured_page_size, remaining_rows))
            url = self._url(
                requested_symbols,
                start_date,
                end_date,
                page_limit,
                page_token,
            )
            request_urls.append(url)
            payload, retries = self._request_payload(url, timeout, headers)
            retry_count += retries
            page_count += 1
            pages_completed += 1
            if not isinstance(payload, Mapping):
                raise ValueError("unexpected Alpaca news response shape")
            articles = payload.get("news", [])
            if articles is None:
                articles = []
            if not isinstance(articles, list):
                raise ValueError("unexpected Alpaca news articles shape")
            if not articles:
                next_page_token = str(payload.get("next_page_token") or "")
                termination_reason = "empty_page"
                break
            for item in articles:
                response_rows += 1
                if not isinstance(item, Mapping):
                    rejected["malformed_record"] += 1
                    continue
                provider_record_id = str(item.get("id") or item.get("news_id") or "").strip()
                if provider_record_id:
                    provider_record_ids.append(provider_record_id)
                    provider_record_headlines[provider_record_id] = str(
                        item.get("headline") or item.get("title") or ""
                    ).strip()
                article_rows, reason = self._rows_for_article(
                    item,
                    requested_symbols=requested_symbols,
                )
                if reason:
                    rejected[reason] += 1
                    continue
                if len(_article_symbols(item, requested_symbols)) > 1:
                    multi_symbol_articles += 1
                for row in article_rows:
                    source_distribution[str(row.get("source", ""))] += 1
                    rows.append(row)
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
            next_page_token = str(payload.get("next_page_token") or "")
            if len(rows) >= limit:
                termination_reason = "max_rows_per_batch"
                break
            if not next_page_token:
                termination_reason = "end_of_results"
                break
            page_token = next_page_token
        if not termination_reason:
            termination_reason = "request_limit" if len(rows) >= limit else "end_of_results"
        self.last_batch_diagnostic = _alpaca_batch_diagnostic(
            rows=rows,
            requested_symbols=requested_symbols,
            request_urls=request_urls,
            page_count=page_count,
            pages_completed=pages_completed,
            response_rows=response_rows,
            rejected=rejected,
            next_page_token=next_page_token,
            multi_symbol_articles=multi_symbol_articles,
            source_distribution=source_distribution,
            secret_key_env=self._secret_key_env,
            page_token_start=self._page_token,
            stopped_by_page_limit=bool(self._max_pages_per_batch and page_count >= self._max_pages_per_batch and next_page_token),
            retry_count=retry_count,
            max_retries=self._max_retries,
            termination_reason=termination_reason,
            provider_record_ids=provider_record_ids,
            provider_record_headlines=provider_record_headlines,
            page_size=self._page_size,
        )
        return rows[:limit]

    def _request_payload(
        self,
        url: str,
        timeout: int,
        headers: Mapping[str, str],
    ) -> tuple[Any, int]:
        attempts = 0
        retries = 0
        while True:
            try:
                return self._alpaca_http_get(url, timeout, headers), retries
            except Exception as exc:
                if getattr(exc, "code", None) == 429 or attempts >= self._max_retries:
                    raise
                attempts += 1
                retries += 1
                if self._retry_sleep_seconds > 0.0:
                    time.sleep(self._retry_sleep_seconds)

    def _url(
        self,
        symbols: list[str],
        start: str,
        end: str,
        limit: int,
        page_token: str,
    ) -> str:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "limit": limit,
            "sort": "asc",
        }
        if start:
            params["start"] = _alpaca_time(start, end_of_day=False)
        if end:
            params["end"] = _alpaca_time(end, end_of_day=True)
        if page_token:
            params["page_token"] = page_token
        return ALPACA_NEWS_URL + "?" + urlencode(params)

    def _rows_for_article(
        self,
        item: Mapping[str, Any],
        *,
        requested_symbols: list[str],
    ) -> tuple[list[dict[str, Any]], str]:
        article_id = str(item.get("id") or item.get("news_id") or "").strip()
        published = _utc(item.get("created_at") or item.get("published_at") or item.get("published_utc") or item.get("date"))
        headline = str(item.get("headline") or item.get("title") or "").strip()
        if not article_id:
            return [], "missing_provider_article_id"
        if not published:
            return [], "missing_publication_timestamp"
        symbols = _article_symbols(item, requested_symbols)
        if not symbols:
            return [], "no_requested_symbol_match"
        updated = _utc(item.get("updated_at") or item.get("updated_utc") or "")
        url = str(item.get("url") or item.get("article_url") or item.get("source_url") or "").strip()
        publisher = _alpaca_publisher(item)
        author = _alpaca_author(item)
        raw_source = _alpaca_raw_source(item)
        summary = str(item.get("summary") or item.get("description") or "").strip()
        body = str(item.get("content") or item.get("body") or item.get("full_text") or "").strip()
        text_value = body or summary
        text_kind = "body_or_full_text" if body else ("summary" if summary else "")
        rows = []
        for symbol in symbols:
            row = self._normalized(
                symbol=symbol,
                provider_id=f"{article_id}:{symbol}",
                url=url,
                published=published,
                source=publisher,
                headline=headline,
                body=text_value,
                sentiment="",
                relevance="",
                novelty="",
                event_type="editorial_news",
                language="en",
            )
            row.update(
                {
                    "provider_article_id": article_id,
                    "provider_original_article_id": article_id,
                    "provider_symbols": ",".join(symbols),
                    "updated_at_utc": updated,
                    "source_type": "editorial_news",
                    "delivery_provider": self.name,
                    "original_source": publisher,
                    "publisher": publisher,
                    "author": author,
                    "raw_source": raw_source,
                    "summary": summary,
                    "body_or_full_text": body,
                    "body_or_summary_kind": text_kind,
                    "collected_at_utc": row["ingested_at"],
                    "historical_availability_note": (
                        "ingested_at is local backfill collection time; "
                        "do not treat it as original article availability time"
                    ),
                }
            )
            row["article_id"] = f"{self.name}:{article_id}:{symbol}"
            rows.append(row)
        return rows, ""


class MassiveStockNewsSource(NewsSource):
    name = "massive_stock_news"

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        params: dict[str, Any] = {
            "ticker": symbol,
            "sort": "published_utc",
            "order": "asc",
            "limit": min(limit, 100),
            "apiKey": api_key,
        }
        if start:
            params["published_utc.gte"] = start
        if end:
            params["published_utc.lte"] = end
        return "https://api.massive.com/v2/reference/news?" + urlencode(params)

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError("unexpected Massive news response shape")
        status = str(payload.get("status") or "").strip().upper()
        if status and status not in {"OK", "DELAYED"}:
            message = str(payload.get("error") or payload.get("message") or status)
            if _looks_rate_limited(message):
                raise ProviderRateLimitError(message)
            raise ValueError(message)
        results = payload.get("results", [])
        if results is None:
            return []
        if not isinstance(results, list):
            raise ValueError("unexpected Massive news results shape")
        return [
            self._row(item, requested_symbol=symbol)
            for item in results
            if isinstance(item, Mapping)
        ]

    def _row(self, item: Mapping[str, Any], *, requested_symbol: str) -> dict[str, Any]:
        tickers = [
            str(ticker).strip().upper()
            for ticker in (item.get("tickers") or [])
            if str(ticker).strip()
        ]
        requested = requested_symbol.upper()
        symbol = requested if requested in tickers or not tickers else tickers[0]
        publisher = item.get("publisher") if isinstance(item.get("publisher"), Mapping) else {}
        provider_id = str(item.get("id") or "").strip()
        stable_id = provider_id or hashlib.sha256(
            f"{self.name}|{symbol}|{item.get('article_url')}|{item.get('title')}|{item.get('published_utc')}".encode()
        ).hexdigest()[:24]
        return self._normalized(
            symbol=symbol,
            provider_id=f"{stable_id}:{symbol}",
            url=item.get("article_url"),
            published=item.get("published_utc"),
            source=publisher.get("name") or self.name,
            headline=item.get("title"),
            body=item.get("description"),
            sentiment="",
            relevance="",
            novelty="",
            event_type="news",
            language=item.get("language", ""),
        )


class CompanyPressReleaseRssSource(NewsSource):
    name, api_key_required = "company_press_release_rss", False

    def __init__(
        self,
        http_get: HttpGet = standard_library_text_get,
        *,
        feeds: Mapping[str, Any] | None = None,
        max_rows_per_feed: int = 20,
        max_enabled_feeds_per_run: int = 0,
        skip_known_error_feeds: bool = False,
    ) -> None:
        super().__init__(http_get)
        self._feeds = _normalise_rss_feed_registry(feeds or {})
        self._max_rows_per_feed = max(1, int(max_rows_per_feed or 20))
        self._max_enabled_feeds_per_run = max(0, int(max_enabled_feeds_per_run or 0))
        self._skip_known_error_feeds = skip_known_error_feeds
        self.last_batch_diagnostic: dict[str, Any] = {}

    def with_provider_config(self, provider_config: Mapping[str, Any]) -> "CompanyPressReleaseRssSource":
        return CompanyPressReleaseRssSource(
            self._http_get,
            feeds=provider_config.get("feeds", {}),
            max_rows_per_feed=int(provider_config.get("max_rows_per_feed", self._max_rows_per_feed)),
            max_enabled_feeds_per_run=int(provider_config.get("max_enabled_feeds_per_run", self._max_enabled_feeds_per_run)),
            skip_known_error_feeds=bool(provider_config.get("skip_known_error_feeds", self._skip_known_error_feeds)),
        )

    def collect(
        self,
        *,
        symbols: list[str],
        start_date: str,
        end_date: str,
        limit: int,
        timeout: int,
        api_key: str = "",
    ) -> list[dict[str, Any]]:
        del api_key
        rows: list[dict[str, Any]] = []
        feed_diagnostics: list[dict[str, Any]] = []
        enabled_feed_attempts = 0
        for symbol_value in symbols:
            symbol = symbol_value.upper()
            symbol_feeds = self._feeds.get(symbol, [])
            if not symbol_feeds:
                feed_diagnostics.append(
                    _rss_feed_diagnostic(
                        provider=self.name,
                        symbol=symbol,
                        zero_row_reason="feed_not_configured",
                    )
                )
                continue
            for feed in symbol_feeds:
                if len(rows) >= limit:
                    break
                diagnostic = _rss_feed_diagnostic(
                    provider=self.name,
                    symbol=symbol,
                    feed_name=str(feed.get("name", "")),
                    feed_url=str(feed.get("url", "")),
                )
                enabled = bool(feed.get("enabled", True))
                feed_url = str(feed.get("url", "")).strip()
                if not enabled:
                    diagnostic["zero_row_reason"] = "feed_disabled"
                    feed_diagnostics.append(diagnostic)
                    continue
                if self._skip_known_error_feeds and bool(feed.get("known_error", False)):
                    diagnostic["zero_row_reason"] = "known_error_feed_skipped"
                    feed_diagnostics.append(diagnostic)
                    continue
                if (
                    self._max_enabled_feeds_per_run > 0
                    and enabled_feed_attempts >= self._max_enabled_feeds_per_run
                ):
                    diagnostic["zero_row_reason"] = "max_enabled_feeds_per_run_reached"
                    feed_diagnostics.append(diagnostic)
                    continue
                if not feed_url:
                    diagnostic["zero_row_reason"] = "feed_url_missing"
                    feed_diagnostics.append(diagnostic)
                    continue
                try:
                    _validate_rss_feed_url(feed_url)
                    enabled_feed_attempts += 1
                    items = _rss_items(self._http_get(feed_url, timeout))
                    diagnostic["response_row_count"] = len(items)
                    selected = [
                        item for item in items
                        if _rss_item_in_date_window(item, start_date=start_date, end_date=end_date)
                    ]
                    per_feed_limit = min(
                        self._max_rows_per_feed,
                        max(0, limit - len(rows)),
                    )
                    normalized = [
                        self._rss_row(symbol=symbol, feed=feed, item=item)
                        for item in selected[:per_feed_limit]
                    ]
                    rows.extend(normalized)
                    diagnostic["normalized_row_count"] = len(normalized)
                    if not items:
                        diagnostic["zero_row_reason"] = "empty_feed"
                    elif not normalized:
                        diagnostic["zero_row_reason"] = "no_items_in_date_range"
                except Exception as exc:  # feed-level isolation is intentional
                    message = str(exc)
                    rate_limited = _looks_rate_limited(message)
                    diagnostic["zero_row_reason"] = "rate_limited" if rate_limited else "provider_error"
                    diagnostic["error_type"] = type(exc).__name__
                    diagnostic["error_message"] = message.replace(feed_url, _redact_url(feed_url))
                    diagnostic["rate_limited"] = rate_limited
                feed_diagnostics.append(diagnostic)
            if len(rows) >= limit:
                break
        self.last_batch_diagnostic = {"feed_diagnostics": feed_diagnostics}
        return rows[:limit]

    def _rss_row(self, *, symbol: str, feed: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("link", "")).strip()
        published = str(item.get("published", "")).strip()
        summary = str(item.get("summary", "")).strip()
        digest = hashlib.sha256(f"{url}|{title}|{published}".encode()).hexdigest()[:24]
        source = str(feed.get("name", "")).strip() or _domain(url) or _domain(str(feed.get("url", ""))) or self.name
        return self._normalized(
            symbol=symbol,
            provider_id=f"rss:{symbol}:{digest}",
            url=url,
            published=published,
            source=source,
            headline=title,
            body=summary,
            sentiment="",
            relevance="",
            novelty="",
            event_type=str(feed.get("event_type", "") or "press_release"),
            language=str(item.get("language", "") or feed.get("language", "")),
        )


class FinnhubNewsSource(NewsSource):
    name = "finnhub"
    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        return "https://finnhub.io/api/v1/company-news?" + urlencode({"symbol": symbol, "from": start, "to": end, "token": api_key})
    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("id"), url=item.get("url"), published=item.get("datetime"), source=item.get("source"), headline=item.get("headline"), body=item.get("summary"), language="en") for item in (payload or [])]


class FmpNewsSource(NewsSource):
    name = "fmp"
    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        return "https://financialmodelingprep.com/stable/news/stock?" + urlencode({"symbols": symbol, "from": start, "to": end, "limit": limit, "apikey": api_key})
    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("publishedDate"), source=item.get("site"), headline=item.get("title"), body=item.get("text"), language="en") for item in (payload or [])]


class NewsApiNewsSource(NewsSource):
    name = "newsapi"
    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        return "https://newsapi.org/v2/everything?" + urlencode({"q": symbol, "from": start, "to": end, "pageSize": min(limit, 100), "apiKey": api_key})
    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("publishedAt"), source=(item.get("source") or {}).get("name"), headline=item.get("title"), body=item.get("description") or item.get("content"), language="en") for item in payload.get("articles", [])]


class SecEdgarNewsSource(NewsSource):
    name, api_key_required = "sec_edgar", False

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        cik = SEC_CIK_BY_SYMBOL.get(symbol.upper())
        if not cik:
            return ""
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def collect(self, *, symbols: list[str], start_date: str, end_date: str, limit: int, timeout: int, api_key: str = "") -> list[dict[str, Any]]:
        supported = [symbol for symbol in symbols if symbol.upper() in SEC_CIK_BY_SYMBOL]
        rows = super().collect(symbols=supported, start_date=start_date, end_date=end_date, limit=limit, timeout=timeout, api_key="")
        return [
            row for row in rows
            if (not start_date or row["published_at_utc"][:10] >= start_date)
            and (not end_date or row["published_at_utc"][:10] <= end_date)
        ][:limit]

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        recent = ((payload.get("filings") or {}).get("recent") or {})
        rows = []
        accessions = list(recent.get("accessionNumber", []))
        for index, accession in enumerate(accessions):
            filing_date = _column_value(recent, "filingDate", index)
            accepted = _column_value(recent, "acceptanceDateTime", index)
            form = _column_value(recent, "form", index)
            document = _column_value(recent, "primaryDocument", index)
            if form not in {"8-K", "10-Q", "10-K", "4", "3", "5"}:
                continue
            accession_path = str(accession).replace("-", "")
            cik = str(int(SEC_CIK_BY_SYMBOL[symbol.upper()]))
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{document}"
            rows.append(self._normalized(
                symbol=symbol, provider_id=accession, url=url,
                published=accepted or filing_date, source="SEC EDGAR",
                headline=_sec_headline(
                    symbol=symbol,
                    form=str(form),
                    filing_date=str(filing_date or ""),
                    accepted=str(accepted or ""),
                    accession=str(accession or ""),
                ),
                body=f"Official SEC filing metadata for Form {form}.",
                sentiment="", relevance="", novelty="",
                event_type=_sec_event_type(str(form)), language="en",
            ))
        return rows


class SecCompanyFilingsSource(NewsSource):
    name, api_key_required = "sec_company_filings", False

    def __init__(
        self,
        http_get: HttpGet = standard_library_json_get,
        *,
        cik_by_symbol: Mapping[str, str] | None = None,
        forms: list[str] | None = None,
        load_official_sec_company_tickers: bool = False,
    ) -> None:
        super().__init__(http_get)
        self._cik_by_symbol = {
            normalize_sec_ticker(str(symbol)): str(cik).strip().zfill(10)
            for symbol, cik in (cik_by_symbol or SEC_CIK_BY_SYMBOL).items()
            if str(symbol).strip() and str(cik).strip()
        }
        self._forms = set(forms or ["8-K", "10-Q", "10-K"])
        self._load_official_sec_company_tickers = load_official_sec_company_tickers
        self.last_batch_diagnostic: dict[str, Any] = {}

    def with_provider_config(self, provider_config: Mapping[str, Any]) -> "SecCompanyFilingsSource":
        forms = [
            str(form).strip().upper()
            for form in provider_config.get("forms", ["8-K", "10-Q", "10-K"]) or []
            if str(form).strip()
        ]
        return SecCompanyFilingsSource(
            self._http_get,
            cik_by_symbol=provider_config.get("cik_by_symbol", self._cik_by_symbol),
            forms=forms,
            load_official_sec_company_tickers=bool(provider_config.get("load_official_sec_company_tickers", self._load_official_sec_company_tickers)),
        )

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        del start, end, limit, api_key
        cik = self._cik_by_symbol.get(normalize_sec_ticker(symbol))
        if not cik:
            return ""
        return sec_submissions_url(cik)

    def collect(
        self,
        *,
        symbols: list[str],
        start_date: str,
        end_date: str,
        limit: int,
        timeout: int,
        api_key: str = "",
    ) -> list[dict[str, Any]]:
        del api_key
        rows: list[dict[str, Any]] = []
        attempted: list[str] = []
        missing_cik: list[str] = []
        cik_by_symbol = dict(self._cik_by_symbol)
        official_mapping_loaded = False
        if self._load_official_sec_company_tickers:
            cik_by_symbol.update(normalize_sec_company_tickers(self._http_get(SEC_COMPANY_TICKERS_URL, timeout)))
            official_mapping_loaded = True
        per_symbol_limit = max(1, math.ceil(limit / len(symbols))) if symbols else limit
        for raw_symbol in symbols:
            if len(rows) >= limit:
                break
            symbol = raw_symbol.upper()
            if normalize_sec_ticker(symbol) not in cik_by_symbol:
                missing_cik.append(symbol)
                continue
            attempted.append(symbol)
            cik = cik_by_symbol[normalize_sec_ticker(symbol)]
            payload = self._http_get(sec_submissions_url(cik), timeout)
            symbol_rows = [
                row for row in self._rows_with_mapping(payload, symbol, cik_by_symbol)
                if (not start_date or row["published_at_utc"][:10] >= start_date)
                and (not end_date or row["published_at_utc"][:10] <= end_date)
            ][:per_symbol_limit]
            rows.extend(symbol_rows)
        self.last_batch_diagnostic = {
            "sec_company_filings_attempted_symbols": attempted,
            "sec_company_filings_missing_cik_symbols": missing_cik,
            "sec_company_filings_resolved_cik_count": len(attempted),
            "sec_company_filings_missing_cik_count": len(missing_cik),
            "sec_company_filings_forms": sorted(self._forms),
            "sec_company_filings_mapping_source_url": SEC_COMPANY_TICKERS_URL if official_mapping_loaded else "built_in_verified_mapping",
        }
        return rows[:limit]

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return self._rows_with_mapping(payload, symbol, self._cik_by_symbol)

    def _rows_with_mapping(self, payload: Any, symbol: str, cik_by_symbol: Mapping[str, str]) -> list[dict[str, Any]]:
        recent = ((payload.get("filings") or {}).get("recent") or {})
        rows: list[dict[str, Any]] = []
        accessions = list(recent.get("accessionNumber", []))
        for index, accession in enumerate(accessions):
            form = str(_column_value(recent, "form", index) or "").upper()
            if form not in self._forms:
                continue
            filing_date = str(_column_value(recent, "filingDate", index) or "")
            accepted = str(_column_value(recent, "acceptanceDateTime", index) or "")
            report_date = str(_column_value(recent, "reportDate", index) or "")
            document = str(_column_value(recent, "primaryDocument", index) or "")
            cik_padded = cik_by_symbol[normalize_sec_ticker(symbol)]
            cik_archive = str(int(cik_padded))
            accession_text = str(accession or "")
            accession_path = accession_text.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_archive}/{accession_path}/"
            primary_url = f"{filing_url}{document}" if document else ""
            published = accepted or filing_date
            row = self._normalized(
                symbol=symbol,
                provider_id=accession_text,
                url=primary_url or filing_url,
                published=published,
                source="SEC EDGAR",
                headline=f"{form} filed by {symbol.upper()}",
                body=f"Official SEC filing metadata for Form {form}.",
                sentiment="",
                relevance="",
                novelty="",
                event_type=_sec_event_type(form),
                language="en",
            )
            row.update(
                {
                    "cik": cik_padded,
                    "source_type": "sec_filing",
                    "form_type": form,
                    "accession_number": accession_text,
                    "filing_date": filing_date,
                    "accepted_datetime": accepted,
                    "report_date": report_date,
                    "filing_url": filing_url,
                    "primary_document_url": primary_url,
                    "collected_at_utc": row["ingested_at"],
                    "published_at_source": "accepted_datetime" if accepted else "filing_date",
                    "timestamp_precision": "datetime" if accepted else "date",
                    "headline_or_title": row["headline"],
                    "source_url": filing_url,
                }
            )
            rows.append(row)
        return rows


def default_news_sources(http_get: HttpGet = standard_library_json_get) -> Mapping[str, NewsSource]:
    rss_http_get = standard_library_text_get if http_get is standard_library_json_get else http_get
    return {
        source.name: source
        for source in (
            GdeltNewsSource(http_get),
            AlphaVantageNewsSource(http_get),
            AlpacaBenzingaNewsSource(),
            MassiveStockNewsSource(http_get),
            CompanyPressReleaseRssSource(rss_http_get),
            FinnhubNewsSource(http_get),
            FmpNewsSource(http_get),
            NewsApiNewsSource(http_get),
            SecEdgarNewsSource(http_get),
            SecCompanyFilingsSource(http_get),
        )
    }


def sec_submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{str(cik).strip().zfill(10)}.json"


def normalize_sec_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def normalize_sec_company_tickers(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected SEC company_tickers response shape")
    result: dict[str, str] = {}
    values = payload.values() if all(str(key).isdigit() for key in payload) else payload.get("data", [])
    for item in values:
        if not isinstance(item, Mapping):
            continue
        ticker = normalize_sec_ticker(str(item.get("ticker", "")))
        cik = str(item.get("cik_str", "")).strip()
        if ticker and cik:
            result[ticker] = cik.zfill(10)
    return result


def _sec_event_type(form: str) -> str:
    if form == "8-K": return "company_event"
    if form in {"10-Q", "10-K"}: return "earnings"
    if form in {"3", "4", "5"}: return "ownership"
    return "filing"


def _sec_headline(
    *,
    symbol: str,
    form: str,
    filing_date: str,
    accepted: str,
    accession: str,
) -> str:
    date_value = (accepted or filing_date).strip()[:10]
    date_label = "accepted" if accepted.strip() else "filed"
    parts = [symbol.upper(), "SEC", form, "filing"]
    if date_value:
        parts.extend((date_label, date_value))
    if accession.strip():
        parts.extend(("accession", accession.strip()))
    return " ".join(parts)


def _column_value(values: Mapping[str, Any], column: str, index: int) -> Any:
    column_values = list(values.get(column, []))
    return column_values[index] if index < len(column_values) else ""


def _blank_or_value(value: Any) -> Any:
    return "" if value is None else value
