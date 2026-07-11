from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)
from core.research.ml.stock_level.news_sources.rss_scratch_dry_run import (
    MAX_FEED_CAP,
    MAX_REQUEST_CAP,
    MAX_ROW_CAP,
    MAX_SYMBOL_CAP,
    RSS_SCRATCH_DRY_RUN_SCHEMA_VERSION,
    write_rss_scratch_dry_run_report,
)


def test_rss_scratch_dry_run_is_disabled_by_default_and_does_not_call_fetcher(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    with pytest.raises(ValueError, match="disabled by default"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "rss-scratch",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )

    assert calls == []
    assert not (tmp_path / "rss-scratch").exists()


def test_network_allowed_requires_live_mode_before_fetcher(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    with pytest.raises(ValueError, match="live_http_fetcher mode"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "rss-scratch",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )

    assert calls == []


def test_fixture_fetcher_mode_runs_under_tmp_path_and_writes_nested_bundle(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    def fake_fetch(feed: Mapping[str, Any]) -> str:
        calls.append(dict(feed))
        return _rss_xml(str(feed["symbol"]))

    report_dir = tmp_path / "rss-scratch"
    report, paths = write_rss_scratch_dry_run_report(
        feeds=[
            _feed("aapl-feed", "https://example.test/aapl.xml", name="Apple Newsroom"),
            _feed("msft-feed", "https://example.test/msft.xml", name="Microsoft Blog"),
        ],
        symbol_mapping={"aapl-feed": "AAPL", "msft-feed": "MSFT"},
        report_dir=report_dir,
        fetcher=fake_fetch,
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        max_feeds=MAX_FEED_CAP,
        max_requests=MAX_REQUEST_CAP,
        max_rows=MAX_ROW_CAP,
        max_symbols=MAX_SYMBOL_CAP,
    )
    provider_report = json.loads(
        (paths.provider_scratch_dir / "provider_scratch_dry_run_report.json").read_text(encoding="utf-8")
    )
    sample_rows = json.loads(
        (paths.provider_scratch_dir / "composition" / "sample_selection" / "corpus_sample_rows.json").read_text(
            encoding="utf-8"
        )
    )

    assert json.loads(paths.report_json_path.read_text(encoding="utf-8")) == report
    assert report["schema_version"] == RSS_SCRATCH_DRY_RUN_SCHEMA_VERSION
    assert report["artifact_type"] == "rss_scratch_dry_run_report"
    assert report["enabled"] is True
    assert report["network_allowed"] is False
    assert report["mode"] == "fixture_fetcher"
    assert report["feed_count"] == 2
    assert report["feeds_attempted"] == 2
    assert report["symbols"] == ["AAPL", "MSFT"]
    assert report["adapter_row_count"] == 2
    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert report["sample_skip_reasons"] == {}
    assert provider_report["provider_id"] == "company_press_release_rss"
    assert [row["symbol"] for row in sample_rows] == ["AAPL", "MSFT"]
    assert [call["symbol"] for call in calls] == ["AAPL", "MSFT"]
    assert "Provider scratch report:" in paths.summary_markdown_path.read_text(encoding="utf-8")
    for path in (
        paths.report_json_path,
        paths.summary_markdown_path,
        paths.provider_scratch_dir / "provider_scratch_dry_run_report.json",
        paths.provider_scratch_dir / "provider_scratch_dry_run_summary.md",
        paths.provider_scratch_dir / "composition" / "composition_smoke_report.json",
        paths.provider_scratch_dir / "composition" / "sample_selection" / "corpus_sample_selection_audit.json",
        paths.provider_scratch_dir / "composition" / "corpus" / "corpus_manifest.json",
    ):
        assert path.exists()
        path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))


def test_feed_request_row_and_symbol_caps_are_enforced(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    report, _paths = write_rss_scratch_dry_run_report(
        feeds=[
            _feed("aapl-feed", "https://example.test/aapl.xml"),
            _feed("msft-feed", "https://example.test/msft.xml"),
            _feed("nvda-feed", "https://example.test/nvda.xml"),
        ],
        symbol_mapping={"aapl-feed": "AAPL", "msft-feed": "MSFT", "nvda-feed": "NVDA"},
        report_dir=tmp_path / "rss-scratch",
        fetcher=lambda feed: calls.append(feed) or _rss_xml(str(feed["symbol"]), item_count=3),
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        max_feeds=2,
        max_requests=1,
        max_rows=1,
        max_symbols=1,
    )

    assert report["feed_count"] == 3
    assert report["feeds_attempted"] == 1
    assert report["symbols"] == ["AAPL"]
    assert report["adapter_row_count"] == 1
    assert report["max_feeds"] == 2
    assert report["max_requests"] == 1
    assert report["max_rows"] == 1
    assert report["max_symbols"] == 1
    assert report["warnings"] == [
        "feeds_capped_to_max_feeds",
        "feeds_capped_to_max_requests",
        "symbols_capped_to_max_symbols",
    ]
    assert len(calls) == 1
    assert calls[0]["symbol"] == "AAPL"


def test_missing_feed_specs_or_symbol_mapping_are_rejected_before_fetch(tmp_path: Path) -> None:
    calls: list[Mapping[str, Any]] = []

    with pytest.raises(ValueError, match="explicit feed specs"):
        write_rss_scratch_dry_run_report(
            feeds=[],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "empty-feeds",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            max_feeds=1,
            max_requests=1,
            max_rows=1,
            max_symbols=1,
        )
    with pytest.raises(ValueError, match="explicit symbol mapping"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={},
            report_dir=tmp_path / "missing-mapping",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            max_feeds=1,
            max_requests=1,
            max_rows=1,
            max_symbols=1,
        )
    with pytest.raises(ValueError, match="missing explicit symbol mapping"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"other-feed": "AAPL"},
            report_dir=tmp_path / "wrong-mapping",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            max_feeds=1,
            max_requests=1,
            max_rows=1,
            max_symbols=1,
        )

    assert calls == []


def test_protected_active_backfill_output_path_is_rejected_before_fetch() -> None:
    calls: list[Mapping[str, Any]] = []

    with pytest.raises(ValueError, match="protected active backfill"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "rss-scratch",
            fetcher=lambda feed: calls.append(feed) or _rss_xml("AAPL"),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            max_feeds=1,
            max_requests=1,
            max_rows=1,
            max_symbols=1,
        )

    assert calls == []


def test_bad_rss_rows_are_excluded_downstream_with_deterministic_reasons(tmp_path: Path) -> None:
    report, paths = write_rss_scratch_dry_run_report(
        feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
        symbol_mapping={"aapl-feed": "AAPL"},
        report_dir=tmp_path / "rss-scratch",
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
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        max_feeds=1,
        max_requests=1,
        max_rows=10,
        max_symbols=1,
    )
    sample_audit = json.loads(
        (
            paths.provider_scratch_dir
            / "composition"
            / "sample_selection"
            / "corpus_sample_selection_audit.json"
        ).read_text(encoding="utf-8")
    )

    assert report["adapter_row_count"] == 3
    assert report["selected_row_count"] == 1
    assert report["excluded_row_count"] == 2
    assert report["sample_skip_reasons"] == {
        "missing_publication_timestamp": 1,
        "missing_text": 1,
    }
    assert [row["reasons"] for row in sample_audit["excluded_rows"]] == [
        ["missing_publication_timestamp"],
        ["missing_text"],
    ]


def test_sec_form_type_and_explicit_event_type_semantics_are_preserved(tmp_path: Path) -> None:
    report, paths = write_rss_scratch_dry_run_report(
        feeds=[_feed("tsla-feed", "https://example.test/tsla.xml")],
        symbol_mapping={"tsla-feed": "TSLA"},
        report_dir=tmp_path / "rss-scratch",
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
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        max_feeds=1,
        max_requests=1,
        max_rows=10,
        max_symbols=1,
    )
    sample_rows = json.loads(
        (paths.provider_scratch_dir / "composition" / "sample_selection" / "corpus_sample_rows.json").read_text(
            encoding="utf-8"
        )
    )
    by_url = {row["provider_url"]: row for row in sample_rows}

    assert report["selected_row_count"] == 2
    assert by_url["https://example.test/tsla-8k"]["source_type"] == "SEC_FILING"
    assert by_url["https://example.test/tsla-8k"]["event_type"] is None
    assert by_url["https://example.test/tsla-earnings"]["event_type"] == "earnings"


def test_safety_flags_confirm_no_live_provider_network_config_or_model_paths(tmp_path: Path) -> None:
    report, _paths = write_rss_scratch_dry_run_report(
        feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
        symbol_mapping={"aapl-feed": "AAPL"},
        report_dir=tmp_path / "rss-scratch",
        fetcher=lambda _feed: _rss_xml("AAPL"),
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        max_feeds=1,
        max_requests=1,
        max_rows=5,
        max_symbols=1,
    )

    assert report["safety_flags"] == {
        "fixture_fetcher_mode_only": True,
        "live_http_fetcher_mode": False,
        "network_allowed": False,
        "real_rss_network_invoked": False,
        "network_invoked": False,
        "download_invoked": False,
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
    assert "max_feed_cap_enforced" in report["guards"]
    assert "max_request_cap_enforced" in report["guards"]
    assert "max_row_cap_enforced" in report["guards"]
    assert "max_symbol_cap_enforced" in report["guards"]


def _feed(feed_id: str, url: str, *, name: str = "Official RSS") -> dict[str, str]:
    return {
        "feed_id": feed_id,
        "name": name,
        "url": url,
        "source_type": "company_rss",
        "event_type": "press_release",
        "language": "en-US",
    }


def _rss_xml(symbol: str, *, item_count: int = 1) -> str:
    items = "\n".join(
        f"""
        <item>
          <title>{symbol} announces test expansion {index}</title>
          <link>https://example.test/{symbol.lower()}-{index}</link>
          <pubDate>Mon, 20 Apr 2026 14:3{index}:00 GMT</pubDate>
          <description>Official RSS summary only.</description>
        </item>
        """
        for index in range(item_count)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>{symbol} Newsroom</title>
        <language>en-US</language>
        {items}
      </channel>
    </rss>
    """
