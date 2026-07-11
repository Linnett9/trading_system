"""Disabled-by-default RSS scratch dry-run wrapper for fixture fetchers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)
from core.research.ml.stock_level.news_sources.provider_scratch_dry_run import (
    write_provider_scratch_dry_run_report,
)
from core.research.ml.stock_level.news_sources.rss import FixtureRssProviderAdapter


RSS_SCRATCH_DRY_RUN_SCHEMA_VERSION = "stock_alpha_news.rss_scratch_dry_run.v1"
MAX_FEED_CAP = 10
MAX_REQUEST_CAP = 10
MAX_ROW_CAP = 100
MAX_SYMBOL_CAP = 10
LIVE_MAX_FEED_CAP = 3
LIVE_MAX_REQUEST_CAP = 3
LIVE_MAX_ROW_CAP = 25
LIVE_MAX_SYMBOL_CAP = 3
DEFAULT_LIVE_TIMEOUT_SECONDS = 10
LIVE_USER_AGENT = "stock-alpha-news-rss-scratch/1.0"

FixtureFetcher = Callable[[Mapping[str, Any]], Any]
LiveTransport = Callable[[str, int, Mapping[str, str]], Any]


@dataclass(frozen=True)
class RssScratchDryRunPaths:
    """Top-level artifacts written by the RSS scratch dry-run wrapper."""

    report_json_path: Path
    summary_markdown_path: Path
    provider_scratch_dir: Path


def write_rss_scratch_dry_run_report(
    *,
    feeds: Sequence[Mapping[str, Any]],
    symbol_mapping: Mapping[str, str],
    report_dir: str | Path,
    fetcher: FixtureFetcher | None = None,
    start_date: str,
    end_date: str,
    enabled: bool = False,
    network_allowed: bool = False,
    mode: str = "fixture_fetcher",
    max_feeds: int,
    max_requests: int,
    max_rows: int,
    max_symbols: int,
) -> tuple[dict[str, Any], RssScratchDryRunPaths]:
    """Run a guarded RSS fixture dry-run through provider scratch composition."""

    if not enabled:
        raise ValueError("RSS scratch dry-run is disabled by default; pass enabled=True explicitly")
    mode = _validated_mode(mode)
    if not network_allowed and mode != "fixture_fetcher":
        raise ValueError("network_allowed must be True for live_http_fetcher mode")
    if network_allowed and mode != "live_http_fetcher":
        raise ValueError("network_allowed=True requires live_http_fetcher mode")
    if fetcher is None:
        raise ValueError(f"{mode} mode requires an injected fetcher")
    _validate_caps(
        max_feeds=max_feeds,
        max_requests=max_requests,
        max_rows=max_rows,
        max_symbols=max_symbols,
        network_allowed=network_allowed,
    )

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")
    paths = RssScratchDryRunPaths(
        report_json_path=report_root / "rss_scratch_dry_run_report.json",
        summary_markdown_path=report_root / "rss_scratch_dry_run_summary.md",
        provider_scratch_dir=report_root / "provider_scratch",
    )
    _ensure_paths_under_report_dir(report_root, paths)

    prepared_feeds, symbols, warnings = _prepared_feeds(
        feeds,
        symbol_mapping=symbol_mapping,
        max_feeds=max_feeds,
        max_requests=max_requests,
        max_symbols=max_symbols,
    )
    adapter = FixtureRssProviderAdapter(feeds=prepared_feeds, fetcher=fetcher)
    provider_report, provider_paths = write_provider_scratch_dry_run_report(
        adapter,
        paths.provider_scratch_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_symbols=max_symbols,
        max_rows=max_rows,
        max_requests=max_requests,
        enabled=True,
        network_allowed=False,
    )
    report = _report(
        enabled=enabled,
        network_allowed=network_allowed,
        mode=mode,
        real_network_invoked=bool(getattr(fetcher, "_stock_alpha_uses_real_network", False)),
        feed_count=len(feeds),
        feeds_attempted=len(prepared_feeds),
        symbols=symbols,
        max_feeds=max_feeds,
        max_requests=max_requests,
        max_rows=max_rows,
        max_symbols=max_symbols,
        adapter_row_count=int(provider_report.get("adapter_row_count", 0) or 0),
        selected_row_count=int(provider_report.get("selected_row_count", 0) or 0),
        corpus_row_count=int(provider_report.get("corpus_row_count", 0) or 0),
        excluded_row_count=int(provider_report.get("excluded_row_count", 0) or 0),
        sample_skip_reasons=dict(provider_report.get("sample_skip_reasons", {}) or {}),
        provider_scratch_report_path=provider_paths.report_json_path,
        warnings=warnings,
    )
    report["output_files"] = {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
        "provider_scratch_report_json": str(provider_paths.report_json_path),
        "provider_scratch_summary_markdown": str(provider_paths.summary_markdown_path),
    }

    writer = ResearchArtifactWriter()
    writer.write_json(paths.report_json_path, report)
    writer.write_markdown(paths.summary_markdown_path, _markdown(report))
    return report, paths


def _prepared_feeds(
    feeds: Sequence[Mapping[str, Any]],
    *,
    symbol_mapping: Mapping[str, str],
    max_feeds: int,
    max_requests: int,
    max_symbols: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not feeds:
        raise ValueError("explicit feed specs are required")
    if not symbol_mapping:
        raise ValueError("explicit symbol mapping is required")

    warnings: list[str] = []
    prepared: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for feed in feeds:
        if not isinstance(feed, Mapping):
            raise ValueError("feed specs must be mappings")
        feed_row = dict(feed)
        feed_key = _feed_key(feed_row)
        symbol = _text(symbol_mapping.get(feed_key)).upper()
        if not symbol:
            raise ValueError(f"missing explicit symbol mapping for feed: {feed_key}")
        if len(prepared) >= min(max_feeds, max_requests):
            continue
        if symbol not in seen_symbols and len(seen_symbols) >= max_symbols:
            continue
        feed_row["symbol"] = symbol
        prepared.append(feed_row)
        seen_symbols.add(symbol)

    if len(feeds) > max_feeds:
        warnings.append("feeds_capped_to_max_feeds")
    if len(feeds) > max_requests:
        warnings.append("feeds_capped_to_max_requests")
    mapped_symbols = {_text(symbol).upper() for symbol in symbol_mapping.values() if _text(symbol)}
    if len(mapped_symbols) > max_symbols:
        warnings.append("symbols_capped_to_max_symbols")
    if not prepared:
        raise ValueError("no RSS feeds remain after applying caps and symbol mapping")
    return prepared, sorted(seen_symbols), sorted(set(warnings))


def _report(
    *,
    enabled: bool,
    network_allowed: bool,
    mode: str,
    real_network_invoked: bool,
    feed_count: int,
    feeds_attempted: int,
    symbols: Sequence[str],
    max_feeds: int,
    max_requests: int,
    max_rows: int,
    max_symbols: int,
    adapter_row_count: int,
    selected_row_count: int,
    corpus_row_count: int,
    excluded_row_count: int,
    sample_skip_reasons: Mapping[str, Any],
    provider_scratch_report_path: Path,
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": RSS_SCRATCH_DRY_RUN_SCHEMA_VERSION,
        "artifact_type": "rss_scratch_dry_run_report",
        "enabled": enabled,
        "network_allowed": network_allowed,
        "mode": mode,
        "feed_count": feed_count,
        "feeds_attempted": feeds_attempted,
        "symbols": list(symbols),
        "max_feeds": max_feeds,
        "max_requests": max_requests,
        "max_rows": max_rows,
        "max_symbols": max_symbols,
        "adapter_row_count": adapter_row_count,
        "selected_row_count": selected_row_count,
        "corpus_row_count": corpus_row_count,
        "excluded_row_count": excluded_row_count,
        "sample_skip_reasons": dict(sample_skip_reasons),
        "provider_scratch_report_path": str(provider_scratch_report_path),
        "guards": _guards(),
        "safety_flags": _safety_flags(
            mode=mode,
            network_allowed=network_allowed,
            real_network_invoked=real_network_invoked,
        ),
        "blockers": [],
        "warnings": sorted(set(warnings)),
        "output_files": {},
    }


def _validate_caps(
    *,
    max_feeds: int,
    max_requests: int,
    max_rows: int,
    max_symbols: int,
    network_allowed: bool,
) -> None:
    if not 0 < int(max_feeds) <= MAX_FEED_CAP:
        raise ValueError(f"max_feeds must be between 1 and {MAX_FEED_CAP}")
    if not 0 < int(max_requests) <= MAX_REQUEST_CAP:
        raise ValueError(f"max_requests must be between 1 and {MAX_REQUEST_CAP}")
    if not 0 < int(max_rows) <= MAX_ROW_CAP:
        raise ValueError(f"max_rows must be between 1 and {MAX_ROW_CAP}")
    if not 0 < int(max_symbols) <= MAX_SYMBOL_CAP:
        raise ValueError(f"max_symbols must be between 1 and {MAX_SYMBOL_CAP}")
    if network_allowed:
        if int(max_feeds) > LIVE_MAX_FEED_CAP:
            raise ValueError(f"live RSS scratch max_feeds must be <= {LIVE_MAX_FEED_CAP}")
        if int(max_requests) > LIVE_MAX_REQUEST_CAP:
            raise ValueError(f"live RSS scratch max_requests must be <= {LIVE_MAX_REQUEST_CAP}")
        if int(max_rows) > LIVE_MAX_ROW_CAP:
            raise ValueError(f"live RSS scratch max_rows must be <= {LIVE_MAX_ROW_CAP}")
        if int(max_symbols) > LIVE_MAX_SYMBOL_CAP:
            raise ValueError(f"live RSS scratch max_symbols must be <= {LIVE_MAX_SYMBOL_CAP}")


def _guards() -> list[str]:
    return [
        "explicit_enable_flag_required",
        "network_disabled_by_default",
        "live_network_requires_explicit_network_allowed",
        "explicit_feed_specs_required",
        "explicit_symbol_mapping_required",
        "injected_fetcher_required_for_fixture_mode",
        "injected_fetcher_required_for_live_mode",
        "scratch_output_directory_required",
        "protected_active_backfill_path_rejected",
        "max_feed_cap_enforced",
        "max_request_cap_enforced",
        "max_row_cap_enforced",
        "max_symbol_cap_enforced",
        "json_and_markdown_audit_output_required",
        "no_config_or_api_key_read",
        "no_ingestion_wiring",
        "no_feature_generation",
        "no_model_replay_or_trading",
    ]


def _safety_flags(*, mode: str, network_allowed: bool, real_network_invoked: bool) -> dict[str, Any]:
    return {
        "fixture_fetcher_mode_only": mode == "fixture_fetcher",
        "live_http_fetcher_mode": mode == "live_http_fetcher",
        "network_allowed": bool(network_allowed),
        "real_rss_network_invoked": bool(real_network_invoked),
        "network_invoked": bool(real_network_invoked),
        "download_invoked": bool(real_network_invoked),
        "provider_object_instantiated_for_live_collection": False,
        "api_keys_read": False,
        "config_read": False,
        "canonical_ingest_invoked": False,
        "historical_backfill_invoked": False,
        "active_backfill_path_read": False,
        "corpus_assembly_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "trading_impact": "none",
        "protected_active_backfill_path_rejected": True,
    }


def _ensure_paths_under_report_dir(report_root: Path, paths: RssScratchDryRunPaths) -> None:
    root = report_root.resolve(strict=False)
    for path in (paths.report_json_path, paths.summary_markdown_path, paths.provider_scratch_dir):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("RSS scratch dry-run outputs must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _feed_key(feed: Mapping[str, Any]) -> str:
    for key in ("feed_id", "id", "url", "name"):
        value = _text(feed.get(key))
        if value:
            return value
    raise ValueError("feed specs require feed_id, id, url, or name")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# RSS Scratch Dry-Run",
            "",
            f"- Schema version: {report['schema_version']}",
            f"- Artifact type: {report['artifact_type']}",
            f"- Enabled: {report['enabled']}",
            f"- Network allowed: {report['network_allowed']}",
            f"- Mode: {report['mode']}",
            f"- Feed count: {report['feed_count']}",
            f"- Feeds attempted: {report['feeds_attempted']}",
            f"- Symbols: {report['symbols']}",
            f"- Max feeds: {report['max_feeds']}",
            f"- Max requests: {report['max_requests']}",
            f"- Max rows: {report['max_rows']}",
            f"- Max symbols: {report['max_symbols']}",
            f"- Adapter rows: {report['adapter_row_count']}",
            f"- Selected rows: {report['selected_row_count']}",
            f"- Corpus rows: {report['corpus_row_count']}",
            f"- Excluded rows: {report['excluded_row_count']}",
            f"- Sample skip reasons: {report['sample_skip_reasons']}",
            f"- Provider scratch report: {report['provider_scratch_report_path']}",
            f"- Blockers: {report['blockers']}",
            f"- Warnings: {report['warnings']}",
            f"- Fixture fetcher mode only: {safety['fixture_fetcher_mode_only']}",
            f"- Real RSS network invoked: {safety['real_rss_network_invoked']}",
            f"- API keys read: {safety['api_keys_read']}",
            f"- Config read: {safety['config_read']}",
            f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
            f"- Feature generation invoked: {safety['feature_generation_invoked']}",
            f"- Model training invoked: {safety['model_training_invoked']}",
            f"- Model inference invoked: {safety['model_inference_invoked']}",
            f"- Trading impact: {safety['trading_impact']}",
            "",
        ]
    )


def build_live_rss_fetcher(
    *,
    transport: LiveTransport | None = None,
    timeout_seconds: int = DEFAULT_LIVE_TIMEOUT_SECONDS,
    user_agent: str = LIVE_USER_AGENT,
) -> FixtureFetcher:
    """Build an explicit live-capable RSS fetcher with injectable transport."""

    timeout = max(1, min(int(timeout_seconds), DEFAULT_LIVE_TIMEOUT_SECONDS))
    headers = {"User-Agent": _text(user_agent) or LIVE_USER_AGENT}
    http_transport = transport or _urllib_transport

    def fetch(feed: Mapping[str, Any]) -> Any:
        url = _text(feed.get("url"))
        _validate_live_feed_url(url)
        return http_transport(url, timeout, headers)

    setattr(fetch, "_stock_alpha_uses_real_network", transport is None)
    return fetch


def _urllib_transport(url: str, timeout_seconds: int, headers: Mapping[str, str]) -> str:
    request = Request(url, headers=dict(headers))
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - explicit live scratch URL only
        return response.read().decode("utf-8", errors="replace")


def _validate_live_feed_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("live RSS feed URL must be explicit http(s)")


def _validated_mode(mode: str) -> str:
    value = _text(mode) or "fixture_fetcher"
    if value not in {"fixture_fetcher", "live_http_fetcher"}:
        raise ValueError("RSS scratch dry-run mode must be fixture_fetcher or live_http_fetcher")
    return value
