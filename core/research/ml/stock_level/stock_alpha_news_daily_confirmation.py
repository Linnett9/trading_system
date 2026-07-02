from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources import default_news_sources


@dataclass(frozen=True)
class StockAlphaNewsDailyConfirmationPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_daily_confirmation(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> StockAlphaNewsDailyConfirmationPaths:
    payload = build_stock_alpha_news_daily_confirmation(config, sources=sources)
    output_dir = _output_dir(config)
    paths = StockAlphaNewsDailyConfirmationPaths(
        json_path=output_dir / "stock_alpha_news_daily_confirmation.json",
        markdown_path=output_dir / "stock_alpha_news_daily_confirmation.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_daily_confirmation(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    settings = dict(ml.get("stock_alpha_news_confirmation", {}) or {})
    enabled = bool(settings.get("enabled", False))
    symbols = _symbols(settings)
    max_symbols = _bounded_int(settings, "max_symbols", default=20, minimum=1, maximum=50)
    max_articles = _bounded_int(settings, "max_articles_per_symbol", default=5, minimum=1, maximum=25)
    max_requests = _bounded_int(settings, "max_provider_requests", default=40, minimum=1, maximum=200)
    lookback_hours = _bounded_int(settings, "lookback_hours", default=72, minimum=1, maximum=24 * 14)
    timeout = _bounded_int(settings, "request_timeout_seconds", default=20, minimum=1, maximum=60)
    as_of = _as_of(settings)
    start = (as_of - timedelta(hours=lookback_hours)).date().isoformat()
    end = as_of.date().isoformat()
    symbols = symbols[:max_symbols]
    adapters = dict(sources or default_news_sources())
    provider_settings = dict(settings.get("providers", {}) or {})

    provider_state = {
        "providers_requested": [],
        "providers_attempted": [],
        "providers_skipped_missing_key": [],
        "providers_failed": {},
        "providers_rate_limited": [],
        "provider_row_counts": {},
    }
    rows_by_symbol = {symbol: [] for symbol in symbols}
    provider_notes_by_symbol: dict[str, dict[str, Any]] = {
        symbol: {} for symbol in symbols
    }
    request_count = 0

    if enabled:
        for provider_name, raw_provider_config in provider_settings.items():
            provider_config = dict(raw_provider_config or {})
            if not bool(provider_config.get("enabled", False)):
                continue
            provider_state["providers_requested"].append(provider_name)
            adapter = adapters.get(provider_name)
            if adapter is None:
                provider_state["providers_failed"][provider_name] = "provider_adapter_unavailable"
                continue
            api_key = _api_key(provider_config)
            if getattr(adapter, "api_key_required", True) and not api_key:
                provider_state["providers_skipped_missing_key"].append(provider_name)
                for symbol in symbols:
                    provider_notes_by_symbol[symbol][provider_name] = {
                        "rate_limit_flag": True,
                        "zero_row_reason": "missing_api_key",
                    }
                continue
            provider_state["providers_attempted"].append(provider_name)
            provider_rows = 0
            for symbol in symbols:
                if request_count >= max_requests:
                    provider_notes_by_symbol[symbol][provider_name] = {
                        "rate_limit_flag": True,
                        "zero_row_reason": "max_provider_requests_reached",
                    }
                    continue
                request_count += 1
                try:
                    rows = adapter.collect(
                        symbols=[symbol],
                        start_date=start,
                        end_date=end,
                        limit=max_articles,
                        timeout=timeout,
                        api_key=api_key,
                    )
                except Exception as exc:  # provider isolation is intentional
                    message = _redacted_message(exc, api_key=api_key)
                    rate_limited = _is_rate_limited(exc)
                    if rate_limited and provider_name not in provider_state["providers_rate_limited"]:
                        provider_state["providers_rate_limited"].append(provider_name)
                    if not rate_limited:
                        provider_state["providers_failed"][provider_name] = (
                            f"{type(exc).__name__}: {message}"
                        )
                    provider_notes_by_symbol[symbol][provider_name] = {
                        "rate_limit_flag": rate_limited,
                        "zero_row_reason": "rate_limited" if rate_limited else "provider_error",
                        "error_type": type(exc).__name__,
                        "error_message": message,
                    }
                    continue
                limited_rows = list(rows[:max_articles])
                provider_rows += len(limited_rows)
                rows_by_symbol[symbol].extend(limited_rows)
                provider_notes_by_symbol[symbol][provider_name] = {
                    "rate_limit_flag": False,
                    "zero_row_reason": "" if limited_rows else "no_recent_news",
                }
            provider_state["provider_row_counts"][provider_name] = provider_rows

    symbol_reports = [
        _symbol_report(
            symbol=symbol,
            rows=rows_by_symbol[symbol],
            provider_notes=provider_notes_by_symbol[symbol],
        )
        for symbol in symbols
    ]
    return {
        "mode": "research",
        "confirmation_only": True,
        "dry_run": bool(settings.get("dry_run", True)),
        "inspection_only": bool(settings.get("inspection_only", True)),
        "enabled": enabled,
        "lookback_hours": lookback_hours,
        "as_of_utc": as_of.isoformat().replace("+00:00", "Z"),
        "start_date": start,
        "end_date": end,
        "symbols_checked": symbols,
        "symbol_count": len(symbols),
        "max_symbols": max_symbols,
        "max_articles_per_symbol": max_articles,
        "max_provider_requests": max_requests,
        "provider_request_count": request_count,
        "symbols_requiring_review": [
            row["symbol"]
            for row in symbol_reports
            if row["confirmation_status"] in {"watch", "negative_news_review", "provider_limited", "provider_error"}
        ],
        "symbol_reports": symbol_reports,
        **provider_state,
        "orders_generated": False,
        "broker_invoked": False,
        "files_ingested": False,
        "features_generated": False,
        "readiness_invoked": False,
        "diagnostics_invoked": False,
        "model_training_invoked": False,
        "news_transformer_enabled": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _symbol_report(
    *,
    symbol: str,
    rows: list[Mapping[str, Any]],
    provider_notes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    news_rows = [row for row in rows if str(row.get("provider", "")) != "sec_edgar"]
    sec_rows = [row for row in rows if str(row.get("provider", "")) == "sec_edgar"]
    latest_news = _latest(news_rows)
    latest_filing = _latest(sec_rows)
    negative_rows = [
        row for row in news_rows
        if _sentiment_value(row.get("sentiment_score")) is not None
        and _sentiment_value(row.get("sentiment_score")) <= -0.15
    ]
    rate_limited = any(bool(note.get("rate_limit_flag")) for note in provider_notes.values())
    provider_error = any(note.get("zero_row_reason") == "provider_error" for note in provider_notes.values())
    article_count = len(news_rows)
    recent_filing = bool(sec_rows)
    if negative_rows:
        status = "negative_news_review"
    elif provider_error and not article_count and not recent_filing:
        status = "provider_error"
    elif rate_limited and not article_count and not recent_filing:
        status = "provider_limited"
    elif article_count or recent_filing:
        status = "watch"
    else:
        status = "no_recent_news"
    return {
        "symbol": symbol,
        "confirmation_status": status,
        "news_review_status": status,
        "article_count": article_count,
        "latest_published_at_utc": str(latest_news.get("published_at_utc", "")) if latest_news else "",
        "top_headlines": [str(row.get("headline", "")) for row in news_rows[:5]],
        "provider_sentiment_summary": _sentiment_summary(news_rows),
        "negative_news_flag": bool(negative_rows),
        "rate_limit_flag": rate_limited,
        "zero_row_reason": _zero_row_reason(provider_notes, article_count, recent_filing),
        "provider_urls": [str(row.get("provider_url", "")) for row in news_rows if str(row.get("provider_url", ""))][:5],
        "sec_recent_filing": recent_filing,
        "recent_filing_count": len(sec_rows),
        "latest_filing_form": _sec_form(latest_filing) if latest_filing else "",
        "latest_filing_published_at_utc": str(latest_filing.get("published_at_utc", "")) if latest_filing else "",
        "latest_filing_url": str(latest_filing.get("provider_url", "")) if latest_filing else "",
        "provider_notes": dict(provider_notes),
        "notes": _notes(status, article_count, recent_filing, rate_limited, provider_error),
    }


def _output_dir(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    value = ml.get("stock_alpha_news_daily_confirmation_report_dir")
    if not value:
        raise ValueError("missing ml.stock_alpha_news_daily_confirmation_report_dir")
    return Path(str(value))


def _symbols(settings: Mapping[str, Any]) -> list[str]:
    return [
        str(symbol).strip().upper()
        for symbol in settings.get("symbols", []) or []
        if str(symbol).strip()
    ]


def _bounded_int(
    settings: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(settings.get(key, default))
    if value < minimum or value > maximum:
        raise ValueError(f"ml.stock_alpha_news_confirmation.{key} must be between {minimum} and {maximum}")
    return value


def _as_of(settings: Mapping[str, Any]) -> datetime:
    value = str(settings.get("as_of_utc", "")).strip()
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _api_key(provider_config: Mapping[str, Any]) -> str:
    env_name = str(provider_config.get("api_key_env", "")).strip()
    return os.environ.get(env_name, "") if env_name else ""


def _redacted_message(exc: Exception, *, api_key: str) -> str:
    message = str(exc)
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _is_rate_limited(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 429:
        return True
    message = str(exc).lower()
    return "too many" in message or "frequency" in message or ("rate" in message and "limit" in message)


def _latest(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: str(row.get("published_at_utc", "")))


def _sentiment_value(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except ValueError:
        return None


def _sentiment_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        value for value in (_sentiment_value(row.get("sentiment_score")) for row in rows)
        if value is not None
    ]
    if not values:
        return {"provider_native_numeric_count": 0, "min": None, "max": None}
    return {
        "provider_native_numeric_count": len(values),
        "min": min(values),
        "max": max(values),
    }


def _zero_row_reason(
    provider_notes: Mapping[str, Mapping[str, Any]],
    article_count: int,
    recent_filing: bool,
) -> str:
    if article_count or recent_filing:
        return ""
    reasons = [
        str(note.get("zero_row_reason", ""))
        for note in provider_notes.values()
        if str(note.get("zero_row_reason", ""))
    ]
    if not reasons:
        return "no_recent_news"
    if "rate_limited" in reasons or "missing_api_key" in reasons or "max_provider_requests_reached" in reasons:
        return "provider_limited"
    if "provider_error" in reasons:
        return "provider_error"
    return "no_recent_news"


def _sec_form(row: Mapping[str, Any] | None) -> str:
    if not row:
        return ""
    parts = str(row.get("headline", "")).split()
    if "SEC" in parts:
        index = parts.index("SEC")
        if index + 1 < len(parts):
            return parts[index + 1]
    return str(row.get("event_type", ""))


def _notes(
    status: str,
    article_count: int,
    recent_filing: bool,
    rate_limited: bool,
    provider_error: bool,
) -> str:
    parts = []
    if article_count:
        parts.append(f"{article_count} provider news article(s)")
    if recent_filing:
        parts.append("recent SEC filing")
    if rate_limited:
        parts.append("provider limited")
    if provider_error:
        parts.append("provider error")
    return "; ".join(parts) if parts else status


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Daily News Confirmation",
        "",
        "Summary:",
        f"- Symbols checked: {payload['symbol_count']}",
        f"- Providers attempted: {payload['providers_attempted']}",
        f"- Providers skipped missing key: {payload['providers_skipped_missing_key']}",
        f"- Provider failures: {payload['providers_failed']}",
        f"- Provider rate limits: {payload['providers_rate_limited']}",
        f"- Symbols requiring review: {payload['symbols_requiring_review']}",
        "- Confirmation only: true",
        "- Trading impact: none",
        "- Orders generated: false",
        "- Broker invoked: false",
        "- Model training invoked: false",
        "- News transformer enabled: false",
        "",
        "| symbol | status | article_count | latest_news_time | recent_sec_filing | notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in payload["symbol_reports"]:
        lines.append(
            f"| {row['symbol']} | {row['confirmation_status']} | "
            f"{row['article_count']} | {row['latest_published_at_utc'] or '-'} | "
            f"{str(row['sec_recent_filing']).lower()} | {row['notes']} |"
        )
    return "\n".join(lines)
