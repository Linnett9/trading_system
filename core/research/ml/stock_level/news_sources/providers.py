from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HttpGet = Callable[[str, int], Any]


def standard_library_json_get(url: str, timeout: int) -> Any:
    request = Request(url, headers={"User-Agent": "stock-alpha-research/1.0"})
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


class GdeltNewsSource(NewsSource):
    name, api_key_required = "gdelt", False

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        query = urlencode({"query": symbol, "mode": "ArtList", "format": "json", "maxrecords": min(limit, 250), "startdatetime": start.replace("-", "") + "000000", "enddatetime": end.replace("-", "") + "235959"})
        return f"https://api.gdeltproject.org/api/v2/doc/doc?{query}"

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("seendate"), source=item.get("domain", "gdelt"), headline=item.get("title"), body="", language=item.get("language", "")) for item in payload.get("articles", [])]


class AlphaVantageNewsSource(NewsSource):
    name = "alpha_vantage"
    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        return "https://www.alphavantage.co/query?" + urlencode({"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": limit, "apikey": api_key})
    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("time_published"), source=item.get("source"), headline=item.get("title"), body=item.get("summary"), sentiment=item.get("overall_sentiment_score", ""), event_type="", language="en") for item in payload.get("feed", [])]


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


def default_news_sources(http_get: HttpGet = standard_library_json_get) -> Mapping[str, NewsSource]:
    return {source.name: source for source in (GdeltNewsSource(http_get), AlphaVantageNewsSource(http_get), FinnhubNewsSource(http_get), FmpNewsSource(http_get), NewsApiNewsSource(http_get))}


def _blank_or_value(value: Any) -> Any:
    return "" if value is None else value


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
        if parsed is None:
            return ""
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
