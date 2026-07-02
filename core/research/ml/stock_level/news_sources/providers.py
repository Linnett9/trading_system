from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HttpGet = Callable[[str, int], Any]

PROVIDER_METADATA = {
    "alpha_vantage": {"statuses": ["usable_now", "needs_key"], "api_key_required": True},
    "finnhub": {"statuses": ["usable_now", "needs_key"], "api_key_required": True},
    "gdelt": {"statuses": ["experimental_no_key", "rate_limited_or_retry_later"], "api_key_required": False},
    "sec_edgar": {"statuses": ["official_filings_source", "experimental_no_key"], "api_key_required": False},
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

    def _url(self, symbol: str, start: str, end: str, limit: int, api_key: str) -> str:
        query = urlencode({"query": symbol, "mode": "ArtList", "format": "json", "maxrecords": min(limit, 250), "startdatetime": start.replace("-", "") + "000000", "enddatetime": end.replace("-", "") + "235959"})
        return f"https://api.gdeltproject.org/api/v2/doc/doc?{query}"

    def _rows(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        return [self._normalized(symbol=symbol, provider_id=item.get("url"), url=item.get("url"), published=item.get("seendate"), source=item.get("domain", "gdelt"), headline=item.get("title"), body="", language=item.get("language", "")) for item in payload.get("articles", [])]


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


def default_news_sources(http_get: HttpGet = standard_library_json_get) -> Mapping[str, NewsSource]:
    return {source.name: source for source in (GdeltNewsSource(http_get), AlphaVantageNewsSource(http_get), FinnhubNewsSource(http_get), FmpNewsSource(http_get), NewsApiNewsSource(http_get), SecEdgarNewsSource(http_get))}


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


def _alpha_vantage_news_time(value: str, *, end_of_day: bool) -> str:
    text = str(value or "").strip()
    compact = text.replace("-", "").replace(":", "")
    if "T" in compact and len(compact) >= 13:
        return compact[:13]
    date_part = text[:10].replace("-", "")
    suffix = "T2359" if end_of_day else "T0000"
    return f"{date_part}{suffix}"


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
