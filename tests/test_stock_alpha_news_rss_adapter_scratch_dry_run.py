from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.research.ml.stock_level.news_sources.canonical import (
    SourceType,
    canonical_from_compatibility_row,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)
from core.research.ml.stock_level.news_sources.provider_scratch_dry_run import (
    MAX_REQUEST_CAP,
    MAX_SYMBOL_CAP,
    write_provider_scratch_dry_run_report,
)
from core.research.ml.stock_level.news_sources.providers import PROVIDER_METADATA
from core.research.ml.stock_level.news_sources.registry import news_source_planning_registry
from core.research.ml.stock_level.news_sources.rss import FixtureRssProviderAdapter


def test_rss_fixture_content_maps_to_compatibility_rows_and_canonical_boundary() -> None:
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetch(feed: Mapping[str, Any]) -> str:
        fetch_calls.append(dict(feed))
        return _rss_xml()

    adapter = FixtureRssProviderAdapter(
        feeds=[
            {
                "symbol": "AAPL",
                "name": "Apple Newsroom",
                "url": "https://example.test/apple/rss.xml",
                "language": "en-US",
                "event_type": "press_release",
            }
        ],
        fetcher=fake_fetch,
    )

    rows = adapter.collect(symbols=["AAPL"], start_date="2026-04-19", end_date="2026-04-21", limit=5)
    canonical = canonical_from_compatibility_row(rows[0], row_number=1)

    assert fetch_calls == [
        {
            "symbol": "AAPL",
            "name": "Apple Newsroom",
            "url": "https://example.test/apple/rss.xml",
            "language": "en-US",
            "event_type": "press_release",
        }
    ]
    assert rows[0]["provider"] == "company_press_release_rss"
    assert rows[0]["source"] == "Apple Newsroom"
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["headline"] == "AAPL announces test expansion"
    assert rows[0]["summary"] == "Official RSS summary only."
    assert rows[0]["published_at_utc"] == "2026-04-20T14:30:00Z"
    assert rows[0]["provider_url"] == "https://example.test/aapl-press-release"
    assert rows[0]["normalized_provider_url"] == "https://example.test/aapl-press-release"
    assert rows[0]["language"] == "en-US"
    assert rows[0]["event_type"] == "press_release"
    assert canonical.provider == "company_press_release_rss"
    assert canonical.symbol == "AAPL"
    assert canonical.source_type == SourceType.COMPANY_RSS
    assert canonical.event_type == "press_release"


def test_rss_fixture_adapter_runs_through_provider_scratch_dry_run(tmp_path: Path) -> None:
    adapter = FixtureRssProviderAdapter(
        feeds=[
            {"symbol": "AAPL", "name": "Apple Newsroom", "url": "https://example.test/apple/rss.xml"},
            {"symbol": "MSFT", "name": "Microsoft Blog", "url": "https://example.test/msft/rss.xml"},
        ],
        fetcher=lambda feed: _rss_xml(symbol=str(feed["symbol"])),
    )

    report, paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "rss-scratch",
        symbols=["MSFT", "AAPL"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        max_symbols=MAX_SYMBOL_CAP,
        max_rows=10,
        max_requests=MAX_REQUEST_CAP,
        enabled=True,
    )
    sample_rows = json.loads((paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(
        encoding="utf-8"
    ))
    corpus_rows = [
        json.loads(line)
        for line in (paths.composition_dir / "corpus" / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["provider_id"] == "company_press_release_rss"
    assert report["provider_family"] == "official_company_rss_fixture"
    assert report["adapter_row_count"] == 2
    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert report["sample_skip_reasons"] == {}
    assert [row["symbol"] for row in sample_rows] == ["AAPL", "MSFT"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "MSFT"]
    assert [call["symbol"] for call in adapter.fetch_calls] == ["AAPL", "MSFT"]
    paths.report_json_path.resolve(strict=False).relative_to((tmp_path / "rss-scratch").resolve(strict=False))
    paths.summary_markdown_path.resolve(strict=False).relative_to((tmp_path / "rss-scratch").resolve(strict=False))


def test_missing_timestamp_or_text_is_excluded_downstream_with_deterministic_reasons(tmp_path: Path) -> None:
    adapter = FixtureRssProviderAdapter(
        feeds=[{"symbol": "AAPL", "name": "Apple Newsroom", "url": "https://example.test/apple/rss.xml"}],
        fetcher=lambda _feed: {
            "items": [
                {
                    "title": "Valid item",
                    "url": "https://example.test/a",
                    "published_at_utc": "2026-04-20T14:30:00Z",
                    "summary": "Valid body",
                },
                {
                    "title": "Missing timestamp",
                    "url": "https://example.test/b",
                    "published_at_utc": "",
                    "summary": "Body is present",
                },
                {
                    "title": "",
                    "url": "https://example.test/c",
                    "published_at_utc": "2026-04-20T15:30:00Z",
                    "summary": "",
                    "body_or_full_text": "",
                },
            ]
        },
    )

    report, paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "rss-scratch",
        symbols=["AAPL"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        max_symbols=1,
        max_rows=10,
        max_requests=1,
        enabled=True,
    )
    sample_audit = json.loads(
        (paths.composition_dir / "sample_selection" / "corpus_sample_selection_audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["adapter_row_count"] == 3
    assert report["selected_row_count"] == 1
    assert report["excluded_row_count"] == 2
    assert report["sample_skip_reasons"] == {
        "missing_publication_timestamp": 1,
        "missing_text": 1,
    }
    assert sample_audit["excluded_rows"] == [
        {
            "row_number": 1,
            "provider": "company_press_release_rss",
            "symbol": "AAPL",
            "provider_article_id": sample_audit["excluded_rows"][0]["provider_article_id"],
            "reasons": ["missing_publication_timestamp"],
        },
        {
            "row_number": 3,
            "provider": "company_press_release_rss",
            "symbol": "AAPL",
            "provider_article_id": sample_audit["excluded_rows"][1]["provider_article_id"],
            "reasons": ["missing_text"],
        },
    ]


def test_explicit_symbol_metadata_is_required() -> None:
    with pytest.raises(ValueError, match="explicit symbol metadata"):
        FixtureRssProviderAdapter(
            feeds=[{"name": "Missing Symbol", "url": "https://example.test/rss.xml"}],
            fetcher=lambda _feed: [],
        )


def test_sec_form_type_is_not_economic_event_and_explicit_event_type_is_preserved(tmp_path: Path) -> None:
    adapter = FixtureRssProviderAdapter(
        feeds=[{"symbol": "TSLA", "name": "SEC-like feed", "url": "https://example.test/tsla/rss.xml"}],
        fetcher=lambda _feed: {
            "items": [
                {
                    "title": "TSLA files 8-K",
                    "url": "https://example.test/tsla-8k",
                    "published_at_utc": "2026-04-20T14:30:00Z",
                    "summary": "SEC filing summary",
                    "source_type": "sec_filing",
                    "form_type": "8-K",
                },
                {
                    "title": "TSLA earnings update",
                    "url": "https://example.test/tsla-earnings",
                    "published_at_utc": "2026-04-20T15:30:00Z",
                    "summary": "Earnings summary",
                    "event_type": "earnings",
                },
            ]
        },
    )

    report, paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "rss-scratch",
        symbols=["TSLA"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        max_symbols=1,
        max_rows=10,
        max_requests=1,
        enabled=True,
    )
    sample_rows = json.loads((paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(
        encoding="utf-8"
    ))

    assert report["selected_row_count"] == 2
    by_url = {row["provider_url"]: row for row in sample_rows}
    assert by_url["https://example.test/tsla-8k"]["source_type"] == "SEC_FILING"
    assert by_url["https://example.test/tsla-8k"]["event_type"] is None
    assert by_url["https://example.test/tsla-earnings"]["event_type"] == "earnings"


def test_no_network_config_api_keys_backfill_or_live_provider_are_needed(tmp_path: Path) -> None:
    adapter = FixtureRssProviderAdapter(
        feeds=[{"symbol": "AAPL", "name": "Apple Newsroom", "url": "https://example.test/apple/rss.xml"}],
        fetcher=lambda _feed: _rss_xml(),
    )

    report, _paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "rss-scratch",
        symbols=["AAPL"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        max_symbols=1,
        max_rows=5,
        max_requests=1,
        enabled=True,
    )

    assert len(adapter.fetch_calls) == 1
    assert report["safety_flags"]["caller_supplied_adapter_used"] is True
    assert report["safety_flags"]["provider_collection_invoked"] is False
    assert report["safety_flags"]["real_provider_object_instantiated"] is False
    assert report["safety_flags"]["network_invoked"] is False
    assert report["safety_flags"]["download_invoked"] is False
    assert report["safety_flags"]["api_keys_read"] is False
    assert report["safety_flags"]["config_read"] is False
    assert report["safety_flags"]["historical_backfill_invoked"] is False
    assert report["safety_flags"]["active_backfill_path_read"] is False
    assert PROVIDER_METADATA["company_press_release_rss"]["api_key_required"] is False
    assert news_source_planning_registry()["company_ir_or_rss"]["collection_enabled"] is False


def test_protected_active_backfill_output_path_is_rejected_before_fetch() -> None:
    adapter = FixtureRssProviderAdapter(
        feeds=[{"symbol": "AAPL", "name": "Apple Newsroom", "url": "https://example.test/apple/rss.xml"}],
        fetcher=lambda _feed: _rss_xml(),
    )

    with pytest.raises(ValueError, match="protected active backfill"):
        write_provider_scratch_dry_run_report(
            adapter,
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "rss-scratch",
            symbols=["AAPL"],
            start_date="2026-04-19",
            end_date="2026-04-21",
            max_symbols=1,
            max_rows=5,
            max_requests=1,
            enabled=True,
        )

    assert adapter.fetch_calls == []


def _rss_xml(*, symbol: str = "AAPL") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>{symbol} Newsroom</title>
        <language>en-US</language>
        <item>
          <title>{symbol} announces test expansion</title>
          <link>https://example.test/{symbol.lower()}-press-release</link>
          <pubDate>Mon, 20 Apr 2026 14:30:00 GMT</pubDate>
          <description>Official RSS summary only.</description>
        </item>
      </channel>
    </rss>
    """
