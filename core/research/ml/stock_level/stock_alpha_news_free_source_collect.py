from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    if not dry_run and rows:
        if output_path.exists() and not bool(settings.get("allow_overwrite", False)):
            payload["providers_failed"]["output"] = "output_exists_and_overwrite_disabled"
            payload["next_action"] = "review_collection_report"
        else:
            ResearchArtifactWriter().write_csv(output_path, rows, fieldnames=[*REQUIRED_NEWS_CONTRACT_COLUMNS, *PROVENANCE_COLUMNS])
            payload["output_written"] = True
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
    limit = int(settings.get("max_articles_per_provider", 50))
    if limit < 1 or limit > 250:
        raise ValueError("ml.stock_alpha_news_collect.max_articles_per_provider must be between 1 and 250")
    timeout = int(settings.get("request_timeout_seconds", 20))
    if timeout < 1 or timeout > 60:
        raise ValueError("ml.stock_alpha_news_collect.request_timeout_seconds must be between 1 and 60")
    symbols = [str(value).strip().upper() for value in settings.get("symbols", []) if str(value).strip()]
    provider_settings = dict(settings.get("providers", {}) or {})
    adapters = dict(sources or default_news_sources())
    requested, attempted, skipped, rate_limited, failures, counts, collected = [], [], [], [], {}, {}, []
    for name, provider_config in provider_settings.items():
        provider_config = dict(provider_config or {})
        if not bool(provider_config.get("enabled", False)):
            continue
        requested.append(name)
        adapter = adapters.get(name)
        if adapter is None:
            failures[name] = "provider_adapter_unavailable"
            continue
        env_name = str(provider_config.get("api_key_env", ""))
        api_key = os.environ.get(env_name, "") if env_name else ""
        if getattr(adapter, "api_key_required", True) and not api_key:
            skipped.append(name)
            continue
        attempted.append(name)
        try:
            rows = adapter.collect(symbols=symbols, start_date=str(settings.get("start_date", "")), end_date=str(settings.get("end_date", "")), limit=limit, timeout=timeout, api_key=api_key)
            normalized = [_canonical_row(row, name) for row in rows[:limit]]
            counts[name] = len(normalized)
            collected.extend(normalized)
        except Exception as exc:  # provider isolation is intentional
            message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
            failures[name] = f"{type(exc).__name__}: {message}"
            if name == "gdelt" and getattr(exc, "code", None) == 429:
                rate_limited.append(name)
    deduplicated = _deduplicate(collected)
    payload = _payload(settings, output_path, requested, attempted, skipped, failures, counts, len(collected))
    payload["deduplicated_row_count"] = len(deduplicated)
    payload["total_rows_collected"] = len(collected)
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
        key = str(row.get("article_id") or row.get("provider_url") or "").strip()
        if not key:
            key = "|".join(str(row.get(field, "")) for field in ("provider", "symbol", "published_at_utc", "headline"))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _payload(settings: Mapping[str, Any], output_path: Path, requested: list[str], attempted: list[str], skipped: list[str], failures: dict[str, str], counts: dict[str, int], total: int) -> dict[str, Any]:
    dry_run = bool(settings.get("dry_run", True))
    return {
        "inspection_only": False, "collection_only": True, "dry_run": dry_run,
        "providers_requested": requested, "providers_attempted": attempted,
        "providers_skipped_missing_key": skipped, "providers_failed": failures,
        "provider_row_counts": counts, "total_rows_collected": total,
        "providers_returned_zero_rows": [],
        "providers_rate_limited": [], "provider_policy": {},
        "deduplicated_row_count": 0, "output_written": False, "output_path": str(output_path),
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


def _required_path(ml: Mapping[str, Any], key: str) -> Path:
    value = ml.get(key)
    if not value: raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(["# Stock-Alpha Free News Source Collection", "", f"- Dry run: {payload['dry_run']}", f"- Providers requested: {payload['providers_requested']}", f"- Providers attempted: {payload['providers_attempted']}", f"- Providers skipped missing key: {payload['providers_skipped_missing_key']}", f"- Providers returned zero rows: {payload['providers_returned_zero_rows']}", f"- Providers rate limited: {payload['providers_rate_limited']}", f"- Provider policy: {payload['provider_policy']}", f"- Providers failed: {payload['providers_failed']}", f"- Rows collected: {payload['total_rows_collected']}", f"- Deduplicated rows: {payload['deduplicated_row_count']}", f"- Output written: {payload['output_written']}", f"- Next action: {payload['next_action']}", "- Features generated: false", "- Model training invoked: false", "", "Collection scaffold only. No ingest, readiness, diagnostics, training, or trading was invoked."])
