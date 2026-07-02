from __future__ import annotations

import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources import PROVIDER_METADATA, default_news_sources
from core.research.ml.stock_level.stock_alpha_news_contract import REQUIRED_NEWS_CONTRACT_COLUMNS


PROVENANCE_COLUMNS = ("provider", "provider_article_id", "provider_url")


@dataclass(frozen=True)
class StockAlphaNewsFreeSourceCollectPaths:
    json_path: Path
    markdown_path: Path
    output_path: Path


def write_stock_alpha_news_free_source_collect(config: Mapping[str, Any], *, sources: Mapping[str, Any] | None = None) -> StockAlphaNewsFreeSourceCollectPaths:
    ml = dict(config.get("ml", {}) or {})
    report_dir = _required_path(ml, "stock_alpha_news_collect_report_dir")
    output_path = _required_path(ml, "stock_alpha_news_collect_output_path")
    payload, rows = build_stock_alpha_news_free_source_collect(config, sources=sources)
    settings = dict(ml.get("stock_alpha_news_collect", {}) or {})
    dry_run = bool(settings.get("dry_run", True))
    payload["new_row_count"] = len(rows)
    if not dry_run and rows:
        rows_to_write = rows
        if output_path.exists() and bool(settings.get("merge_existing", False)):
            if not bool(settings.get("backup_existing", False)):
                raise ValueError(
                    "ml.stock_alpha_news_collect.merge_existing requires "
                    "backup_existing=true"
                )
            existing_rows = CsvRowRepository().read(output_path)
            # New rows take precedence so normalization improvements replace an
            # older representation of the same stable provider record.
            combined_rows = [*rows, *existing_rows]
            rows_to_write = _deduplicate(combined_rows)
            backup_path = _backup_path(output_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, backup_path)
            payload["existing_input_row_count"] = len(existing_rows)
            payload["merge_deduplicated_row_count"] = len(combined_rows) - len(rows_to_write)
            payload["backup_path"] = str(backup_path)
        elif output_path.exists() and not bool(settings.get("allow_overwrite", False)):
            payload["providers_failed"]["output"] = "output_exists_and_overwrite_disabled"
            payload["next_action"] = "review_collection_report"
            rows_to_write = []
        if rows_to_write:
            ResearchArtifactWriter().write_csv(output_path, rows_to_write, fieldnames=[*REQUIRED_NEWS_CONTRACT_COLUMNS, *PROVENANCE_COLUMNS])
            payload["output_written"] = True
            payload["output_row_count"] = len(rows_to_write)
    paths = StockAlphaNewsFreeSourceCollectPaths(
        report_dir / "stock_alpha_news_free_source_collect.json",
        report_dir / "stock_alpha_news_free_source_collect.md",
        output_path,
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_free_source_collect(config: Mapping[str, Any], *, sources: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ml = dict(config.get("ml", {}) or {})
    output_path = _required_path(ml, "stock_alpha_news_collect_output_path")
    settings = dict(ml.get("stock_alpha_news_collect", {}) or {})
    if not bool(settings.get("enabled", False)):
        return _payload(settings, output_path, [], [], [], {}, {}, 0), []
    dry_run = bool(settings.get("dry_run", True))
    legacy_limit = int(settings.get("max_articles_per_provider", 50))
    if legacy_limit < 1 or legacy_limit > 250:
        raise ValueError("ml.stock_alpha_news_collect.max_articles_per_provider must be between 1 and 250")
    provider_request_limit = int(settings.get("provider_request_limit", legacy_limit))
    if provider_request_limit < 1 or provider_request_limit > 250:
        raise ValueError("ml.stock_alpha_news_collect.provider_request_limit must be between 1 and 250")
    max_rows_per_provider = int(settings.get("max_rows_per_provider", legacy_limit))
    if max_rows_per_provider < 1 or max_rows_per_provider > 10_000:
        raise ValueError("ml.stock_alpha_news_collect.max_rows_per_provider must be between 1 and 10000")
    symbols_per_batch = int(settings.get("symbols_per_batch", 0) or 0)
    timeout = int(settings.get("request_timeout_seconds", 20))
    if timeout < 1 or timeout > 60:
        raise ValueError("ml.stock_alpha_news_collect.request_timeout_seconds must be between 1 and 60")
    rate_limit_sleep_seconds = float(settings.get("rate_limit_sleep_seconds", 0.0))
    if rate_limit_sleep_seconds < 0.0 or rate_limit_sleep_seconds > 60.0:
        raise ValueError("ml.stock_alpha_news_collect.rate_limit_sleep_seconds must be between 0 and 60")
    if bool(settings.get("merge_existing", False)) and not bool(
        settings.get("backup_existing", False)
    ):
        raise ValueError(
            "ml.stock_alpha_news_collect.merge_existing requires backup_existing=true"
        )
    configured_symbols = _configured_symbol_list(settings.get("symbols", []))
    only_symbols = _configured_symbol_list(settings.get("only_symbols", []))
    max_symbols_per_run = int(settings.get("max_symbols_per_run", 0) or 0)
    if max_symbols_per_run < 0 or max_symbols_per_run > 500:
        raise ValueError("ml.stock_alpha_news_collect.max_symbols_per_run must be between 0 and 500")
    symbols = _selected_symbols(
        configured_symbols,
        only_symbols=only_symbols,
        max_symbols_per_run=max_symbols_per_run,
    )
    if symbols_per_batch < 1:
        symbols_per_batch = max(1, len(symbols) or 1)
    if symbols_per_batch > 500:
        raise ValueError("ml.stock_alpha_news_collect.symbols_per_batch must be at most 500")
    provider_settings = dict(settings.get("providers", {}) or {})
    adapters = dict(sources or default_news_sources())
    requested, attempted, skipped, rate_limited, failures, counts, collected = [], [], [], [], {}, {}, []
    batch_counts: dict[str, int] = {}
    batch_diagnostics: list[dict[str, Any]] = []
    zero_row_reasons: dict[str, str] = {}
    for name, provider_config in provider_settings.items():
        provider_config = dict(provider_config or {})
        if not bool(provider_config.get("enabled", False)):
            continue
        requested.append(name)
        adapter = adapters.get(name)
        if adapter is None:
            failures[name] = "provider_adapter_unavailable"
            continue
        adapter = _configured_provider_adapter(adapter, provider_config)
        env_names = _api_key_env_names(provider_config)
        api_key = _first_env_value(env_names)
        if getattr(adapter, "api_key_required", True) and not api_key:
            skipped.append(name)
            continue
        attempted.append(name)
        counts[name] = 0
        provider_rows: list[dict[str, Any]] = []
        batches = _symbol_batches(symbols, symbols_per_batch)
        batch_counts[name] = len(batches)
        provider_stopped = False
        for batch_index, batch_symbols in enumerate(batches):
            remaining = max_rows_per_provider - len(provider_rows)
            if remaining <= 0:
                break
            requested_limit = min(provider_request_limit, remaining)
            diagnostic = _provider_batch_diagnostic(
                provider=name,
                batch_index=batch_index,
                batch_symbols=batch_symbols,
                settings=settings,
                requested_limit=requested_limit,
            )
            try:
                rows = adapter.collect(
                    symbols=batch_symbols,
                    start_date=str(settings.get("start_date", "")),
                    end_date=str(settings.get("end_date", "")),
                    limit=requested_limit,
                    timeout=timeout,
                    api_key=api_key,
                )
                canonical_rows = [
                    _canonical_row(row, name) for row in rows[:remaining]
                ]
                provider_rows.extend(canonical_rows)
                diagnostic["response_row_count"] = len(rows)
                diagnostic["normalized_row_count"] = len(canonical_rows)
                diagnostic.update(_provider_extra_diagnostic(adapter))
                if not rows:
                    diagnostic["zero_row_reason"] = "empty_provider_response_or_no_matching_articles"
            except Exception as exc:  # provider isolation is intentional
                message = _redacted_exception_message(exc, api_key=api_key)
                is_rate_limited = _is_rate_limited(exc)
                diagnostic.update(_provider_extra_diagnostic(adapter))
                diagnostic["response_row_count"] = 0
                diagnostic["normalized_row_count"] = 0
                diagnostic["rate_limited"] = is_rate_limited
                diagnostic["zero_row_reason"] = "rate_limited" if is_rate_limited else "provider_error"
                diagnostic["error_type"] = type(exc).__name__
                diagnostic["error_message"] = message
                if is_rate_limited:
                    if name not in rate_limited:
                        rate_limited.append(name)
                    zero_row_reasons[name] = "rate_limited"
                else:
                    failures[name] = f"{type(exc).__name__}: {message}"
                    zero_row_reasons[name] = "provider_error"
                batch_diagnostics.append(diagnostic)
                provider_stopped = True
                break
            batch_diagnostics.append(diagnostic)
            if (
                rate_limit_sleep_seconds > 0.0
                and batch_index < len(batches) - 1
                and len(provider_rows) < max_rows_per_provider
            ):
                time.sleep(rate_limit_sleep_seconds)
        counts[name] = len(provider_rows)
        if counts[name] == 0 and name not in zero_row_reasons:
            zero_row_reasons[name] = "all_batches_returned_zero_rows"
        collected.extend(provider_rows)
        if provider_stopped:
            continue
    deduplicated = _deduplicate(collected)
    payload = _payload(settings, output_path, requested, attempted, skipped, failures, counts, len(collected))
    payload["deduplicated_row_count"] = len(deduplicated)
    payload["total_rows_collected"] = len(collected)
    payload["configured_symbol_count"] = len(set(configured_symbols))
    payload["requested_symbol_count"] = len(set(symbols))
    payload["only_symbols"] = only_symbols
    payload["max_symbols_per_run"] = max_symbols_per_run
    payload["symbols_per_batch"] = symbols_per_batch
    payload["provider_request_limit"] = provider_request_limit
    payload["max_rows_per_provider"] = max_rows_per_provider
    payload["rate_limit_sleep_seconds"] = rate_limit_sleep_seconds
    payload["provider_batch_counts"] = batch_counts
    payload["provider_batch_diagnostics"] = batch_diagnostics
    payload.update(_rss_registry_diagnostics(settings, symbols, batch_diagnostics))
    payload["provider_zero_row_reasons"] = {
        name: reason for name, reason in sorted(zero_row_reasons.items())
        if counts.get(name, 0) == 0 or reason in {"rate_limited", "provider_error"}
    }
    payload["symbol_count"] = len(
        {str(row.get("symbol", "")).strip().upper() for row in deduplicated}
        - {""}
    )
    payload.update(_row_diagnostics(deduplicated, requested_symbol_count=len(set(symbols))))
    headlines = [
        str(row.get("headline", "")).strip().lower()
        for row in deduplicated
        if str(row.get("headline", "")).strip()
    ]
    payload["duplicate_headline_count"] = len(headlines) - len(set(headlines))
    payload["duplicate_headline_rate"] = (
        payload["duplicate_headline_count"] / len(deduplicated)
        if deduplicated
        else 0.0
    )
    payload["providers_returned_zero_rows"] = sorted(name for name, count in counts.items() if count == 0)
    payload["providers_rate_limited"] = sorted(rate_limited)
    payload["provider_policy"] = {name: PROVIDER_METADATA.get(name, {"statuses": ["unclassified"]}) for name in provider_settings}
    payload["next_action"] = _next_action(dry_run, requested, attempted, skipped, rate_limited, failures, counts, deduplicated)
    return payload, deduplicated


def _canonical_row(row: Mapping[str, Any], provider: str) -> dict[str, Any]:
    normalized = {column: row.get(column, "") for column in REQUIRED_NEWS_CONTRACT_COLUMNS}
    normalized.update({column: row.get(column, "") for column in PROVENANCE_COLUMNS})
    normalized["provider"] = str(normalized.get("provider") or provider)
    normalized["source"] = str(normalized.get("source") or provider)
    return normalized


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for row in rows:
        provider_id = str(row.get("provider_article_id", "")).strip()
        if provider_id:
            key = "|".join(
                str(row.get(field, "")).strip()
                for field in (
                    "provider",
                    "provider_article_id",
                    "symbol",
                    "published_at_utc",
                )
            )
        else:
            key = "|".join(
                str(row.get(field, "")).strip()
                for field in (
                    "article_id",
                    "provider_url",
                    "symbol",
                    "published_at_utc",
                    "headline",
                )
            )
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _configured_symbol_list(values: Any) -> list[str]:
    return [
        str(value).strip().upper()
        for value in values or []
        if str(value).strip()
    ]


def _selected_symbols(
    configured_symbols: list[str],
    *,
    only_symbols: list[str],
    max_symbols_per_run: int,
) -> list[str]:
    symbols = list(configured_symbols)
    if only_symbols:
        allowed = set(only_symbols)
        symbols = [symbol for symbol in symbols if symbol in allowed]
    if max_symbols_per_run > 0:
        symbols = symbols[:max_symbols_per_run]
    return symbols


def _symbol_batches(symbols: list[str], batch_size: int) -> list[list[str]]:
    if not symbols:
        return [[]]
    return [
        symbols[index : index + batch_size]
        for index in range(0, len(symbols), batch_size)
    ]


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
    }


def _payload(settings: Mapping[str, Any], output_path: Path, requested: list[str], attempted: list[str], skipped: list[str], failures: dict[str, str], counts: dict[str, int], total: int) -> dict[str, Any]:
    dry_run = bool(settings.get("dry_run", True))
    return {
        "inspection_only": False, "collection_only": True, "dry_run": dry_run,
        "providers_requested": requested, "providers_attempted": attempted,
        "providers_skipped_missing_key": skipped, "providers_failed": failures,
        "provider_row_counts": counts, "total_rows_collected": total,
        "configured_symbol_count": 0, "requested_symbol_count": 0,
        "only_symbols": [], "max_symbols_per_run": 0,
        "symbols_per_batch": 0,
        "verified_feed_symbol_count": 0, "enabled_feed_symbol_count": 0,
        "known_error_feed_symbol_count": 0,
        "disabled_symbol_count": 0, "symbols_without_verified_feed": [],
        "symbols_with_known_error_feeds": [], "symbols_with_feed_errors": [],
        "symbols_skipped_known_error_feeds": [],
        "symbols_skipped_max_enabled_feeds_per_run": [],
        "max_enabled_feeds_per_run": 0, "skip_known_error_feeds": False,
        "provider_request_limit": 0, "max_rows_per_provider": 0,
        "rate_limit_sleep_seconds": 0.0, "provider_batch_counts": {},
        "provider_batch_diagnostics": [], "provider_zero_row_reasons": {},
        "providers_returned_zero_rows": [],
        "providers_rate_limited": [], "provider_policy": {},
        "deduplicated_row_count": 0, "output_written": False, "output_path": str(output_path),
        "symbol_count": 0, "duplicate_headline_count": 0,
        "duplicate_headline_rate": 0.0,
        "rows_by_provider": {}, "rows_by_symbol": {},
        "provider_symbol_counts": {}, "provider_symbol_coverage": {},
        "published_at_utc_range_by_provider": {},
        "existing_input_row_count": 0, "new_row_count": 0,
        "merge_deduplicated_row_count": 0, "output_row_count": 0,
        "backup_path": "",
        "files_ingested": False, "features_generated": False, "readiness_invoked": False,
        "diagnostics_invoked": False, "model_training_invoked": False,
        "news_transformer_enabled": False, "trading_impact": "none", "production_validated": False,
        "next_action": "run_dry_collection" if dry_run else "review_collection_report",
    }


def _next_action(dry_run: bool, requested: list[str], attempted: list[str], skipped: list[str], rate_limited: list[str], failures: Mapping[str, str], counts: Mapping[str, int], rows: list[dict[str, Any]]) -> str:
    if skipped and not attempted: return "configure_api_keys"
    if "gdelt" in rate_limited: return "retry_gdelt_later_or_reduce_request"
    if any(name in {"fmp", "newsapi"} and ("402" in message or "426" in message) for name, message in failures.items()): return "disable_paid_or_upgrade_required_sources"
    if counts.get("finnhub") == 0: return "adjust_finnhub_symbols_or_date_range"
    if failures: return "review_collection_report"
    if not requested: return "run_dry_collection"
    if dry_run and counts.get("alpha_vantage", 0) > 0: return "write_alpha_vantage_bounded_export"
    if dry_run and rows: return "write_raw_provider_export"
    if dry_run: return "review_collection_report"
    return "run_provider_sample_check" if rows else "review_collection_report"


def _rss_registry_diagnostics(
    settings: Mapping[str, Any],
    symbols: list[str],
    batch_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    provider_config = dict(
        (dict(settings.get("providers", {}) or {}).get("company_press_release_rss") or {})
    )
    feeds_by_symbol = dict(provider_config.get("feeds", {}) or {})
    verified_symbols = set()
    enabled_symbols = set()
    known_error_symbols = set()
    for symbol in symbols:
        entries = feeds_by_symbol.get(symbol, [])
        if isinstance(entries, Mapping):
            entries = [entries]
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            url = str(entry.get("url", "")).strip()
            official = bool(entry.get("official", False))
            verified_source = str(entry.get("verified_source_url", "")).strip()
            if url and (official or verified_source):
                verified_symbols.add(symbol)
            if url and bool(entry.get("enabled", True)):
                enabled_symbols.add(symbol)
            if url and bool(entry.get("known_error", False)):
                known_error_symbols.add(symbol)
    symbols_with_feed_errors = set()
    symbols_skipped_known_error_feeds = set()
    symbols_skipped_max_enabled_feeds_per_run = set()
    for diagnostic in batch_diagnostics:
        for feed in diagnostic.get("feed_diagnostics", []) or []:
            if not isinstance(feed, Mapping):
                continue
            symbol = str(feed.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            zero_row_reason = str(feed.get("zero_row_reason", "")).strip()
            if str(feed.get("error_type", "")).strip() or zero_row_reason in {"provider_error", "rate_limited", "feed_url_missing"}:
                symbols_with_feed_errors.add(symbol)
            if zero_row_reason == "known_error_feed_skipped":
                symbols_skipped_known_error_feeds.add(symbol)
            if zero_row_reason == "max_enabled_feeds_per_run_reached":
                symbols_skipped_max_enabled_feeds_per_run.add(symbol)
    return {
        "verified_feed_symbol_count": len(verified_symbols),
        "enabled_feed_symbol_count": len(enabled_symbols),
        "known_error_feed_symbol_count": len(known_error_symbols),
        "disabled_symbol_count": len(set(symbols) - enabled_symbols),
        "symbols_without_verified_feed": sorted(set(symbols) - verified_symbols),
        "symbols_with_known_error_feeds": sorted(known_error_symbols),
        "symbols_with_feed_errors": sorted(symbols_with_feed_errors),
        "symbols_skipped_known_error_feeds": sorted(symbols_skipped_known_error_feeds),
        "symbols_skipped_max_enabled_feeds_per_run": sorted(symbols_skipped_max_enabled_feeds_per_run),
        "max_enabled_feeds_per_run": max(0, int(provider_config.get("max_enabled_feeds_per_run", 0) or 0)),
        "skip_known_error_feeds": bool(provider_config.get("skip_known_error_feeds", False)),
    }


def _provider_batch_diagnostic(
    *,
    provider: str,
    batch_index: int,
    batch_symbols: list[str],
    settings: Mapping[str, Any],
    requested_limit: int,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "batch_index": batch_index + 1,
        "symbol_count": len(batch_symbols),
        "symbols": list(batch_symbols),
        "start_date": str(settings.get("start_date", "")),
        "end_date": str(settings.get("end_date", "")),
        "requested_limit": requested_limit,
        "response_row_count": 0,
        "normalized_row_count": 0,
        "zero_row_reason": "",
        "rate_limited": False,
    }


def _provider_extra_diagnostic(adapter: Any) -> dict[str, Any]:
    value = getattr(adapter, "last_batch_diagnostic", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _configured_provider_adapter(adapter: Any, provider_config: Mapping[str, Any]) -> Any:
    configure = getattr(adapter, "with_provider_config", None)
    return configure(provider_config) if callable(configure) else adapter


def _is_rate_limited(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 429:
        return True
    message = str(exc).lower()
    return (
        "too many" in message
        or "frequency" in message
        or ("rate" in message and "limit" in message)
    )


def _redacted_exception_message(exc: Exception, *, api_key: str) -> str:
    message = str(exc)
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _api_key_env_names(provider_config: Mapping[str, Any]) -> list[str]:
    names = []
    primary = str(provider_config.get("api_key_env", "")).strip()
    if primary:
        names.append(primary)
    for value in provider_config.get("api_key_env_fallbacks", []) or []:
        name = str(value).strip()
        if name and name not in names:
            names.append(name)
    return names


def _first_env_value(names: list[str]) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _required_path(ml: Mapping[str, Any], key: str) -> Path:
    value = ml.get(key)
    if not value: raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _backup_path(output_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_path.with_name(
        f"{output_path.stem}.backup-{timestamp}{output_path.suffix}"
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stock-Alpha Free News Source Collection",
        "",
        f"- Dry run: {payload['dry_run']}",
        f"- Configured symbols: {payload['configured_symbol_count']}",
        f"- Requested symbols: {payload['requested_symbol_count']}",
        f"- Only symbols: {payload['only_symbols']}",
        f"- Max symbols per run: {payload['max_symbols_per_run']}",
        f"- Verified feed symbols: {payload['verified_feed_symbol_count']}",
        f"- Enabled feed symbols: {payload['enabled_feed_symbol_count']}",
        f"- Known-error feed symbols: {payload['known_error_feed_symbol_count']}",
        f"- Disabled symbols: {payload['disabled_symbol_count']}",
        f"- Symbols without verified feed: {payload['symbols_without_verified_feed']}",
        f"- Symbols with known-error feeds: {payload['symbols_with_known_error_feeds']}",
        f"- Symbols with feed errors: {payload['symbols_with_feed_errors']}",
        f"- Symbols skipped known-error feeds: {payload['symbols_skipped_known_error_feeds']}",
        f"- Symbols skipped max enabled feeds per run: {payload['symbols_skipped_max_enabled_feeds_per_run']}",
        f"- Symbols per batch: {payload['symbols_per_batch']}",
        f"- Provider request limit: {payload['provider_request_limit']}",
        f"- Max rows per provider: {payload['max_rows_per_provider']}",
        f"- Max enabled feeds per run: {payload['max_enabled_feeds_per_run']}",
        f"- Skip known-error feeds: {payload['skip_known_error_feeds']}",
        f"- Rate-limit sleep seconds: {payload['rate_limit_sleep_seconds']}",
        f"- Providers requested: {payload['providers_requested']}",
        f"- Providers attempted: {payload['providers_attempted']}",
        f"- Providers skipped missing key: {payload['providers_skipped_missing_key']}",
        f"- Providers returned zero rows: {payload['providers_returned_zero_rows']}",
        f"- Provider zero-row reasons: {payload['provider_zero_row_reasons']}",
        f"- Providers rate limited: {payload['providers_rate_limited']}",
        f"- Provider policy: {payload['provider_policy']}",
        f"- Providers failed: {payload['providers_failed']}",
        f"- Provider batches: {payload['provider_batch_counts']}",
        f"- Provider batch diagnostic count: {len(payload['provider_batch_diagnostics'])}",
        f"- Rows collected: {payload['total_rows_collected']}",
        f"- Rows by provider: {payload['rows_by_provider']}",
        f"- Rows by symbol: {payload['rows_by_symbol']}",
        f"- Provider symbol coverage: {payload['provider_symbol_coverage']}",
        f"- Published ranges by provider: {payload['published_at_utc_range_by_provider']}",
        f"- Deduplicated rows: {payload['deduplicated_row_count']}",
        f"- Symbols: {payload['symbol_count']}",
        f"- Duplicate headlines: {payload['duplicate_headline_count']}",
        f"- Duplicate headline rate: {payload['duplicate_headline_rate']}",
        f"- Existing input rows: {payload['existing_input_row_count']}",
        f"- New rows: {payload['new_row_count']}",
        f"- Merge duplicates removed: {payload['merge_deduplicated_row_count']}",
        f"- Output rows: {payload['output_row_count']}",
        f"- Backup path: {payload['backup_path'] or 'none'}",
        f"- Output written: {payload['output_written']}",
        f"- Next action: {payload['next_action']}",
        "- Features generated: false",
        "- Model training invoked: false",
        "",
        "Collection scaffold only. No ingest, readiness, diagnostics, training, or trading was invoked.",
    ]
    return "\n".join(lines)
