from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)
from core.research.ml.stock_level.news_sources.rss_scratch_dry_run import (
    build_live_rss_fetcher,
    write_rss_scratch_dry_run_report,
)


def test_live_capable_rss_dry_run_refuses_when_disabled(tmp_path: Path) -> None:
    transport_calls: list[tuple[str, int, Mapping[str, str]]] = []
    fetcher = build_live_rss_fetcher(transport=lambda url, timeout, headers: transport_calls.append((url, timeout, headers)))

    with pytest.raises(ValueError, match="disabled by default"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "rss-live",
            fetcher=fetcher,
            start_date="2026-04-19",
            end_date="2026-04-21",
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )

    assert transport_calls == []
    assert not (tmp_path / "rss-live").exists()


def test_live_http_fetcher_is_not_called_when_network_is_not_allowed(tmp_path: Path) -> None:
    transport_calls: list[tuple[str, int, Mapping[str, str]]] = []
    fetcher = build_live_rss_fetcher(transport=lambda url, timeout, headers: transport_calls.append((url, timeout, headers)))

    with pytest.raises(ValueError, match="network_allowed"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "rss-live",
            fetcher=fetcher,
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=False,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )

    assert transport_calls == []


def test_injected_fake_live_fetcher_path_runs_under_tmp_path(tmp_path: Path) -> None:
    transport_calls: list[tuple[str, int, Mapping[str, str]]] = []

    def fake_transport(url: str, timeout: int, headers: Mapping[str, str]) -> str:
        transport_calls.append((url, timeout, dict(headers)))
        return _rss_xml("AAPL")

    report, paths = write_rss_scratch_dry_run_report(
        feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
        symbol_mapping={"aapl-feed": "AAPL"},
        report_dir=tmp_path / "rss-live",
        fetcher=build_live_rss_fetcher(transport=fake_transport, timeout_seconds=2),
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        mode="live_http_fetcher",
        network_allowed=True,
        max_feeds=1,
        max_requests=1,
        max_rows=5,
        max_symbols=1,
    )
    sample_rows = json.loads(
        (paths.provider_scratch_dir / "composition" / "sample_selection" / "corpus_sample_rows.json").read_text(
            encoding="utf-8"
        )
    )

    assert transport_calls == [
        ("https://example.test/aapl.xml", 2, {"User-Agent": "stock-alpha-news-rss-scratch/1.0"})
    ]
    assert report["mode"] == "live_http_fetcher"
    assert report["network_allowed"] is True
    assert report["feed_count"] == 1
    assert report["feeds_attempted"] == 1
    assert report["adapter_row_count"] == 1
    assert report["selected_row_count"] == 1
    assert report["safety_flags"]["live_http_fetcher_mode"] is True
    assert report["safety_flags"]["network_allowed"] is True
    assert report["safety_flags"]["real_rss_network_invoked"] is False
    assert report["safety_flags"]["network_invoked"] is False
    assert sample_rows[0]["symbol"] == "AAPL"
    paths.report_json_path.resolve(strict=False).relative_to((tmp_path / "rss-live").resolve(strict=False))


def test_network_allowed_requires_live_mode_and_small_caps(tmp_path: Path) -> None:
    fetcher = build_live_rss_fetcher(transport=lambda _url, _timeout, _headers: _rss_xml("AAPL"))

    with pytest.raises(ValueError, match="live_http_fetcher mode"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "wrong-mode",
            fetcher=fetcher,
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="fixture_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )
    with pytest.raises(ValueError, match="live RSS scratch max_rows"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "too-many-rows",
            fetcher=fetcher,
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=26,
            max_symbols=1,
        )


def test_live_mode_requires_explicit_feed_specs_symbol_mapping_and_fetcher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an injected fetcher"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "missing-fetcher",
            fetcher=None,
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )
    with pytest.raises(ValueError, match="explicit feed specs"):
        write_rss_scratch_dry_run_report(
            feeds=[],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=tmp_path / "missing-feeds",
            fetcher=build_live_rss_fetcher(transport=lambda _url, _timeout, _headers: _rss_xml("AAPL")),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )
    with pytest.raises(ValueError, match="explicit symbol mapping"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={},
            report_dir=tmp_path / "missing-symbols",
            fetcher=build_live_rss_fetcher(transport=lambda _url, _timeout, _headers: _rss_xml("AAPL")),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )


def test_caps_are_enforced_before_live_fetch(tmp_path: Path) -> None:
    transport_calls: list[str] = []
    fetcher = build_live_rss_fetcher(
        transport=lambda url, _timeout, _headers: transport_calls.append(url) or _rss_xml("AAPL", item_count=3)
    )

    report, _paths = write_rss_scratch_dry_run_report(
        feeds=[
            _feed("aapl-feed", "https://example.test/aapl.xml"),
            _feed("msft-feed", "https://example.test/msft.xml"),
            _feed("nvda-feed", "https://example.test/nvda.xml"),
        ],
        symbol_mapping={"aapl-feed": "AAPL", "msft-feed": "MSFT", "nvda-feed": "NVDA"},
        report_dir=tmp_path / "rss-live",
        fetcher=fetcher,
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        mode="live_http_fetcher",
        network_allowed=True,
        max_feeds=2,
        max_requests=1,
        max_rows=1,
        max_symbols=1,
    )

    assert transport_calls == ["https://example.test/aapl.xml"]
    assert report["feeds_attempted"] == 1
    assert report["symbols"] == ["AAPL"]
    assert report["adapter_row_count"] == 1
    assert report["warnings"] == [
        "feeds_capped_to_max_feeds",
        "feeds_capped_to_max_requests",
        "symbols_capped_to_max_symbols",
    ]


def test_protected_active_backfill_path_is_rejected_before_live_fetch() -> None:
    transport_calls: list[str] = []

    with pytest.raises(ValueError, match="protected active backfill"):
        write_rss_scratch_dry_run_report(
            feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
            symbol_mapping={"aapl-feed": "AAPL"},
            report_dir=Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "rss-live",
            fetcher=build_live_rss_fetcher(
                transport=lambda url, _timeout, _headers: transport_calls.append(url) or _rss_xml("AAPL")
            ),
            start_date="2026-04-19",
            end_date="2026-04-21",
            enabled=True,
            mode="live_http_fetcher",
            network_allowed=True,
            max_feeds=1,
            max_requests=1,
            max_rows=5,
            max_symbols=1,
        )

    assert transport_calls == []


def test_live_guard_safety_flags_need_no_keys_config_backfill_features_or_models(tmp_path: Path) -> None:
    report, _paths = write_rss_scratch_dry_run_report(
        feeds=[_feed("aapl-feed", "https://example.test/aapl.xml")],
        symbol_mapping={"aapl-feed": "AAPL"},
        report_dir=tmp_path / "rss-live",
        fetcher=build_live_rss_fetcher(transport=lambda _url, _timeout, _headers: _rss_xml("AAPL")),
        start_date="2026-04-19",
        end_date="2026-04-21",
        enabled=True,
        mode="live_http_fetcher",
        network_allowed=True,
        max_feeds=1,
        max_requests=1,
        max_rows=5,
        max_symbols=1,
    )

    assert report["safety_flags"] == {
        "fixture_fetcher_mode_only": False,
        "live_http_fetcher_mode": True,
        "network_allowed": True,
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


def test_live_fetcher_helper_uses_fake_transport_and_validates_urls() -> None:
    calls: list[tuple[str, int, Mapping[str, str]]] = []
    fetcher = build_live_rss_fetcher(
        transport=lambda url, timeout, headers: calls.append((url, timeout, dict(headers))) or "payload",
        timeout_seconds=99,
        user_agent="test-agent",
    )

    assert fetcher({"url": "https://example.test/feed.xml"}) == "payload"
    assert calls == [("https://example.test/feed.xml", 10, {"User-Agent": "test-agent"})]
    with pytest.raises(ValueError, match="explicit http"):
        fetcher({"url": "file:///tmp/feed.xml"})


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
