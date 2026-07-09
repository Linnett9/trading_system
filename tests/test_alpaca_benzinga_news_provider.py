from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from config.config_loader import load_config
from core.research.ml.stock_level.news_sources import AlpacaBenzingaNewsSource
from core.research.ml.stock_level.stock_alpha_news_contract import (
    REQUIRED_NEWS_CONTRACT_COLUMNS,
)
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    build_stock_alpha_news_contract_rows,
)
from core.research.ml.stock_level.stock_alpha_news_free_source_collect import (
    build_stock_alpha_news_free_source_collect,
    write_stock_alpha_news_free_source_collect,
)


def test_alpaca_benzinga_maps_response_to_canonical_rows_and_preserves_update_time(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    calls = []

    def fake_get(url, timeout, headers):
        calls.append((url, timeout, headers))
        return {
            "news": [
                {
                    "id": 101,
                    "created_at": "2024-01-02T14:30:00-05:00",
                    "updated_at": "2024-01-02T20:00:00Z",
                    "headline": "Apple and Microsoft rally",
                    "summary": "Benzinga summary",
                    "url": "https://example.test/story/101",
                    "source": "Benzinga",
                    "symbols": ["AAPL", "MSFT", "TSLA"],
                }
            ],
            "next_page_token": "",
        }

    rows = AlpacaBenzingaNewsSource(fake_get).collect(
        symbols=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-03",
        limit=10,
        timeout=5,
        api_key="key",
    )

    assert [row["symbol"] for row in rows] == ["AAPL", "MSFT"]
    assert rows[0]["provider"] == "alpaca_benzinga"
    assert rows[0]["source"] == "Benzinga"
    assert rows[0]["event_type"] == "editorial_news"
    assert rows[0]["published_at_utc"] == "2024-01-02T19:30:00Z"
    assert rows[0]["updated_at_utc"] == "2024-01-02T20:00:00Z"
    assert rows[0]["provider_article_id"] == "101"
    assert rows[0]["provider_symbols"] == "AAPL,MSFT"
    assert rows[0]["body_or_summary"] == "Benzinga summary"
    assert rows[0]["publisher"] == "Benzinga"
    assert rows[0]["author"] == ""
    assert rows[0]["raw_source"] == "Benzinga"
    assert rows[0]["summary"] == "Benzinga summary"
    assert rows[0]["body_or_full_text"] == ""
    assert rows[0]["body_or_summary_kind"] == "summary"
    assert rows[0]["sentiment_score"] == ""
    assert rows[0]["historical_availability_note"].startswith("ingested_at is local backfill")
    assert calls[0][2] == {
        "APCA-API-KEY-ID": "key",
        "APCA-API-SECRET-KEY": "secret",
    }


def test_alpaca_benzinga_preserves_byline_source_without_treating_it_as_publisher(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    def fake_get(url, timeout, headers):
        del url, timeout, headers
        return {
            "news": [
                {
                    "id": 2020,
                    "created_at": "2020-01-02T10:00:00Z",
                    "updated_at": "2020-01-02T10:05:00Z",
                    "headline": "Amazon headline",
                    "content": "Full text when genuinely supplied",
                    "url": "https://example.test/story/2020",
                    "source": "Amit Nag",
                    "symbols": ["AMZN"],
                }
            ],
            "next_page_token": "",
        }

    rows = AlpacaBenzingaNewsSource(fake_get).collect(
        symbols=["AMZN"],
        start_date="2020-01-01",
        end_date="2020-01-31",
        limit=5,
        timeout=5,
        api_key="key",
    )

    assert rows[0]["source"] == "alpaca_benzinga"
    assert rows[0]["publisher"] == ""
    assert rows[0]["author"] == "Amit Nag"
    assert rows[0]["raw_source"] == "Amit Nag"
    assert rows[0]["summary"] == ""
    assert rows[0]["body_or_full_text"] == "Full text when genuinely supplied"
    assert rows[0]["body_or_summary_kind"] == "body_or_full_text"


def test_alpaca_benzinga_paginates_and_stops_on_empty_page(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    page_tokens = []

    def fake_get(url, timeout, headers):
        del timeout, headers
        query = parse_qs(urlparse(url).query)
        page_tokens.append(query.get("page_token", [""])[0])
        if len(page_tokens) == 1:
            return {
                "news": [_article(1, ["AAPL"])],
                "next_page_token": "next-page",
            }
        return {"news": [], "next_page_token": ""}

    source = AlpacaBenzingaNewsSource(fake_get)
    rows = source.collect(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=5,
        timeout=5,
        api_key="key",
    )

    assert len(rows) == 1
    assert page_tokens == ["", "next-page"]
    assert source.last_batch_diagnostic["alpaca_benzinga_pages_requested"] == 2
    assert source.last_batch_diagnostic["alpaca_benzinga_pages_completed"] == 2
    assert source.last_batch_diagnostic["alpaca_benzinga_termination_reason"] == "empty_page"
    assert source.last_batch_diagnostic["alpaca_benzinga_records_returned"] == 1


def test_alpaca_benzinga_follows_next_page_token_when_page_size_allows(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    requested_limits = []

    def fake_get(url, timeout, headers):
        del timeout, headers
        query = parse_qs(urlparse(url).query)
        requested_limits.append(int(query["limit"][0]))
        page_token = query.get("page_token", [""])[0]
        if not page_token:
            return {"news": [_article(1, ["AAPL"])], "next_page_token": "page-2"}
        return {"news": [_article(2, ["AAPL"])], "next_page_token": ""}

    source = AlpacaBenzingaNewsSource(fake_get, page_size=1, max_pages_per_batch=3)
    rows = source.collect(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=3,
        timeout=5,
        api_key="key",
    )

    assert [row["provider_article_id"] for row in rows] == ["1", "2"]
    assert requested_limits == [1, 1]
    assert source.last_batch_diagnostic["alpaca_benzinga_termination_reason"] == "end_of_results"
    assert source.last_batch_diagnostic["alpaca_benzinga_stopped_with_more_results_available"] is False


def test_alpaca_benzinga_reports_row_cap_with_remaining_next_page_token(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    def fake_get(url, timeout, headers):
        del url, timeout, headers
        return {
            "news": [_article(1, ["AAPL"]), _article(2, ["AAPL"])],
            "next_page_token": "more-results",
        }

    source = AlpacaBenzingaNewsSource(fake_get, page_size=2, max_pages_per_batch=3)
    rows = source.collect(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=2,
        timeout=5,
        api_key="key",
    )

    assert len(rows) == 2
    diagnostic = source.last_batch_diagnostic
    assert diagnostic["alpaca_benzinga_termination_reason"] == "max_rows_per_batch"
    assert diagnostic["alpaca_benzinga_next_page_token_present_at_stop"] is True
    assert diagnostic["alpaca_benzinga_stopped_with_more_results_available"] is True


def test_alpaca_benzinga_retries_transient_failures_bounded(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    calls = 0

    def flaky_get(url, timeout, headers):
        nonlocal calls
        del url, timeout, headers
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        return {"news": [_article(1, ["AAPL"])], "next_page_token": ""}

    source = AlpacaBenzingaNewsSource(flaky_get, max_retries=1)
    rows = source.collect(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=5,
        timeout=5,
        api_key="key",
    )

    assert len(rows) == 1
    assert calls == 2
    assert source.last_batch_diagnostic["alpaca_benzinga_retry_count"] == 1


def test_alpaca_benzinga_rejects_malformed_records_conservatively(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    def fake_get(url, timeout, headers):
        del url, timeout, headers
        return {
            "news": [
                {"id": "", "created_at": "2024-01-01T00:00:00Z", "symbols": ["AAPL"]},
                {"id": 2, "created_at": "", "symbols": ["AAPL"]},
                {"id": 3, "created_at": "2024-01-01T00:00:00Z", "symbols": ["MSFT"]},
            ],
            "next_page_token": "",
        }

    source = AlpacaBenzingaNewsSource(fake_get)
    rows = source.collect(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=5,
        timeout=5,
        api_key="key",
    )

    assert rows == []
    assert source.last_batch_diagnostic["alpaca_benzinga_rejected_reasons"] == {
        "missing_provider_article_id": 1,
        "missing_publication_timestamp": 1,
        "no_requested_symbol_match": 1,
    }


def test_alpaca_benzinga_collection_dry_run_does_not_overwrite_and_reports_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    output = tmp_path / "raw.csv"
    output.write_text("article_id,symbol,headline,provider_article_id\nold,AAPL,Existing headline,1\n", encoding="utf-8")

    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect_output_path"] = str(output)
    source = AlpacaBenzingaNewsSource(lambda url, timeout, headers: {
        "news": [_article(1, ["AAPL", "MSFT"], headline="Existing headline")],
        "next_page_token": "",
    })

    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"alpaca_benzinga": source},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert output.read_text(encoding="utf-8").startswith("article_id,symbol,headline")
    assert payload["output_written"] is False
    assert payload["provider_row_counts"] == {"alpaca_benzinga": 2}
    assert payload["provider_batch_diagnostics"][0]["alpaca_benzinga_multi_symbol_article_count"] == 1
    assert payload["provider_batch_diagnostics"][0]["alpaca_benzinga_unique_provider_articles"] == 1
    assert payload["provider_batch_diagnostics"][0]["alpaca_benzinga_article_symbol_rows"] == 2
    assert payload["provider_batch_diagnostics"][0]["alpaca_benzinga_duplicate_provider_article_id_count"] == 0
    assert payload["same_provider_article_id_multi_symbol_row_count"] == 1
    assert payload["exact_duplicate_provider_article_record_count"] == 0
    assert payload["duplicate_headline_count"] == 0
    assert payload["legacy_row_based_duplicate_headline_count"] == 1
    assert payload["overlap_with_existing_output"]["overlap_provider_article_id_count"] == 2
    assert payload["overlap_with_existing_output"]["overlap_normalized_headline_count"] == 2


def test_alpaca_benzinga_write_path_deduplicates_duplicate_pages_and_is_contract_compatible(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    class DuplicatePageSource:
        api_key_required = True

        def with_provider_config(self, provider_config):
            return self

        def collect(self, **kwargs):
            del kwargs
            row = AlpacaBenzingaNewsSource(lambda url, timeout, headers: {
                "news": [_article(1, ["AAPL"])],
                "next_page_token": "",
            }).collect(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                limit=5,
                timeout=5,
                api_key="key",
            )[0]
            return [row, dict(row)]

    config = _collection_config(tmp_path, dry_run=False)
    config["ml"]["stock_alpha_news_collect"]["allow_overwrite"] = True
    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"alpaca_benzinga": DuplicatePageSource()},
    )
    rows = list(csv.DictReader(paths.output_path.open(encoding="utf-8")))
    normalized, audit = build_stock_alpha_news_contract_rows(rows)

    assert len(rows) == 1
    assert set(REQUIRED_NEWS_CONTRACT_COLUMNS) <= set(rows[0])
    assert rows[0]["updated_at_utc"] == "2024-01-01T10:05:00Z"
    assert audit["duplicate_article_id_count"] == 0
    assert len(normalized) == 1


def test_alpaca_benzinga_rate_limit_is_reported_without_secret_leak(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "super-secret")

    def rate_limited(url, timeout, headers):
        del url, timeout, headers
        raise HTTPError("https://data.alpaca.markets/v1beta1/news", 429, "rate limit", {}, None)

    config = _collection_config(tmp_path, dry_run=True)
    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"alpaca_benzinga": AlpacaBenzingaNewsSource(rate_limited)},
    )
    report = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(report)

    assert "super-secret" not in report
    assert payload["providers_rate_limited"] == ["alpaca_benzinga"]
    assert payload["provider_zero_row_reasons"] == {"alpaca_benzinga": "rate_limited"}
    assert payload["provider_batch_diagnostics"][0]["termination_reason"] == "rate_limit"


def test_alpaca_benzinga_entitlement_failure_is_not_zero_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "super-secret")

    def forbidden(url, timeout, headers):
        del url, timeout, headers
        raise HTTPError("https://data.alpaca.markets/v1beta1/news", 403, "forbidden", {}, None)

    config = _collection_config(tmp_path, dry_run=True)
    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"alpaca_benzinga": AlpacaBenzingaNewsSource(forbidden)},
    )
    report = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(report)

    assert "super-secret" not in report
    assert payload["providers_entitlement_failed"] == ["alpaca_benzinga"]
    assert payload["provider_zero_row_reasons"] == {"alpaca_benzinga": "entitlement_error"}
    assert payload["provider_batch_diagnostics"][0]["termination_reason"] == "entitlement_error"
    assert payload["provider_batch_diagnostics"][0]["zero_row_reason"] == "entitlement_error"
    assert payload["providers_returned_zero_rows"] == ["alpaca_benzinga"]


def test_alpaca_benzinga_duplicate_diagnostics_separate_same_id_and_same_headline(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    class DuplicateDiagnosticsSource:
        api_key_required = True

        def with_provider_config(self, provider_config):
            return self

        def collect(self, **kwargs):
            del kwargs
            first = AlpacaBenzingaNewsSource(lambda url, timeout, headers: {
                "news": [_article(1, ["AAPL"])],
                "next_page_token": "",
            }).collect(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                limit=5,
                timeout=5,
                api_key="key",
            )[0]
            same_id_duplicate = dict(first)
            same_headline_other_id = dict(first)
            same_headline_other_id["article_id"] = "alpaca_benzinga:2:AAPL"
            same_headline_other_id["provider_article_id"] = "2"
            return [first, same_id_duplicate, same_headline_other_id]

    payload, rows = build_stock_alpha_news_free_source_collect(
        _collection_config(tmp_path, dry_run=True),
        sources={"alpaca_benzinga": DuplicateDiagnosticsSource()},
    )

    assert len(rows) == 2
    assert payload["exact_duplicate_provider_article_record_count"] == 1
    assert payload["same_headline_different_provider_article_id_count"] == 1
    assert payload["same_provider_article_id_multi_symbol_row_count"] == 0


def test_collection_enforces_inclusive_publication_window_and_reports_rejections(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    page_tokens = []

    def fake_get(url, timeout, headers):
        del timeout, headers
        query = parse_qs(urlparse(url).query)
        page_token = query.get("page_token", [""])[0]
        page_tokens.append(page_token)
        if not page_token:
            return {
                "news": [
                    _article(1, ["AAPL"], created_at="2023-12-31T23:59:59Z"),
                    _article(2, ["AAPL"], created_at="2024-01-01T00:00:00Z"),
                    _article(3, ["AAPL"], created_at="2024-01-31T23:59:59Z"),
                ],
                "next_page_token": "page-2",
            }
        return {
            "news": [
                _article(4, ["AAPL"], created_at="2024-02-01T00:00:00Z"),
                _article(5, ["AAPL"], created_at="2024-01-15T12:00:00Z"),
            ],
            "next_page_token": "",
        }

    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["provider_request_limit"] = 5
    config["ml"]["stock_alpha_news_collect"]["max_rows_per_provider"] = 5
    config["ml"]["stock_alpha_news_collect"]["providers"]["alpaca_benzinga"]["page_size"] = 3
    config["ml"]["stock_alpha_news_collect"]["providers"]["alpaca_benzinga"]["max_pages_per_batch"] = 2
    payload, rows = build_stock_alpha_news_free_source_collect(
        config,
        sources={"alpaca_benzinga": AlpacaBenzingaNewsSource(fake_get)},
    )

    assert page_tokens == ["", "page-2"]
    assert [row["provider_article_id"] for row in rows] == ["2", "3", "5"]
    assert [row["published_at_utc"] for row in rows] == [
        "2024-01-01T00:00:00Z",
        "2024-01-31T23:59:59Z",
        "2024-01-15T12:00:00Z",
    ]
    assert payload["out_of_window_before_start_count"] == 1
    assert payload["out_of_window_after_end_count"] == 1
    assert payload["out_of_window_rejected_count"] == 2
    assert payload["provider_batch_diagnostics"][0]["out_of_window_before_start_count"] == 1
    assert payload["provider_batch_diagnostics"][0]["out_of_window_after_end_count"] == 1
    assert payload["provider_batch_diagnostics"][0]["out_of_window_rejected_count"] == 2
    assert payload["provider_batch_diagnostics"][0]["publication_window_start_utc_inclusive"] == "2024-01-01T00:00:00Z"
    assert payload["provider_batch_diagnostics"][0]["publication_window_end_utc_inclusive"] == "2024-01-31T23:59:59Z"
    assert payload["provider_row_counts"] == {"alpaca_benzinga": 3}
    assert payload["text_availability_by_provider"]["alpaca_benzinga"]["headline_length_distribution"]["count"] == 3


def test_alpaca_provider_coverage_is_separate_from_rss_registry_status(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    source = AlpacaBenzingaNewsSource(lambda url, timeout, headers: {
        "news": [_article(1, ["AAPL"]), _article(2, ["MSFT"])],
        "next_page_token": "",
    })

    paths = write_stock_alpha_news_free_source_collect(
        _collection_config(tmp_path, dry_run=True),
        sources={"alpaca_benzinga": source},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["provider_symbol_coverage"] == {"alpaca_benzinga": 1.0}
    assert payload["rss_registry_status_scope"] == "company_press_release_rss"
    assert payload["rss_registry_loaded"] is False
    assert payload["disabled_symbol_count"] == 2
    assert "RSS disabled symbols" in markdown
    assert "- Disabled symbols:" not in markdown


def test_alpaca_benzinga_probe_config_is_dry_run_and_research_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_alpaca_benzinga_coverage_probe_dry_run.yaml",
        overlay_project_config=True,
    )
    settings = config["ml"]["stock_alpha_news_collect"]

    assert settings["dry_run"] is True
    assert settings["providers"]["alpaca_benzinga"]["api_key_env"] == "ALPACA_API_KEY_ID"
    assert settings["providers"]["alpaca_benzinga"]["secret_key_env"] == "ALPACA_SECRET_KEY"
    assert config["ml"]["stock_alpha_news_enable_transformer"] is False
    rendered_news_settings = json.dumps(settings).lower()
    assert "paper" not in rendered_news_settings
    assert "broker" not in rendered_news_settings


def test_alpaca_benzinga_pagination_probe_config_is_bounded_dry_run():
    config = load_config(
        "config/config.stock_alpha_news_collect_alpaca_benzinga_pagination_probe_dry_run.yaml",
        overlay_project_config=True,
    )
    settings = config["ml"]["stock_alpha_news_collect"]
    provider = settings["providers"]["alpaca_benzinga"]

    assert settings["probe_kind"] == "pagination_probe"
    assert settings["dry_run"] is True
    assert settings["allow_overwrite"] is False
    assert settings["merge_existing"] is False
    assert settings["provider_request_limit"] > provider["page_size"]
    assert settings["max_rows_per_provider"] <= 80
    assert provider["max_pages_per_batch"] == 4
    assert config["ml"]["stock_alpha_news_enable_transformer"] is False


def test_alpaca_benzinga_2016_entitlement_probe_config_is_bounded_dry_run():
    config = load_config(
        "config/config.stock_alpha_news_collect_alpaca_benzinga_2016_entitlement_probe_dry_run.yaml",
        overlay_project_config=True,
    )
    settings = config["ml"]["stock_alpha_news_collect"]
    provider = settings["providers"]["alpaca_benzinga"]

    assert settings["probe_kind"] == "historical_entitlement_coverage_probe"
    assert settings["dry_run"] is True
    assert settings["allow_overwrite"] is False
    assert settings["merge_existing"] is False
    assert settings["backup_existing"] is False
    assert settings["start_date"] == "2016-01-01"
    assert settings["end_date"] == "2016-01-31"
    assert settings["symbols"] == ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM"]
    assert settings["provider_request_limit"] == 60
    assert settings["max_rows_per_provider"] == 120
    assert provider["page_size"] == 20
    assert provider["max_pages_per_batch"] == 3
    assert config["ml"]["stock_alpha_news_enable_transformer"] is False


@pytest.mark.parametrize("year", ["2016", "2020", "2024", "2026"])
def test_alpaca_benzinga_text_availability_probe_configs_are_bounded_dry_runs(year):
    config = load_config(
        f"config/config.stock_alpha_news_collect_alpaca_benzinga_text_availability_{year}_dry_run.yaml",
        overlay_project_config=True,
    )
    settings = config["ml"]["stock_alpha_news_collect"]
    provider = settings["providers"]["alpaca_benzinga"]

    assert settings["probe_kind"] == "text_availability_probe"
    assert settings["dry_run"] is True
    assert settings["allow_overwrite"] is False
    assert settings["merge_existing"] is False
    assert settings["backup_existing"] is False
    assert settings["start_date"] == f"{year}-01-01"
    assert settings["end_date"] == f"{year}-01-31"
    assert settings["symbols"] == ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM"]
    assert settings["provider_request_limit"] == 40
    assert settings["max_rows_per_provider"] == 80
    assert provider["page_size"] == 20
    assert provider["max_pages_per_batch"] == 2
    assert config["ml"]["stock_alpha_news_enable_transformer"] is False


def _article(
    article_id: int,
    symbols: list[str],
    *,
    headline: str = "Headline",
    created_at: str = "2024-01-01T10:00:00Z",
    summary: str = "",
) -> dict:
    return {
        "id": article_id,
        "created_at": created_at,
        "updated_at": "2024-01-01T10:05:00Z",
        "headline": headline,
        "summary": summary,
        "url": f"https://example.test/{article_id}",
        "source": "Benzinga",
        "symbols": symbols,
    }


def _collection_config(tmp_path: Path, *, dry_run: bool) -> dict:
    return {
        "ml": {
            "stock_alpha_news_collect_report_dir": str(tmp_path / "report"),
            "stock_alpha_news_collect_output_path": str(tmp_path / "raw.csv"),
            "stock_alpha_news_collect": {
                "enabled": True,
                "dry_run": dry_run,
                "allow_overwrite": False,
                "max_articles_per_provider": 5,
                "provider_request_limit": 5,
                "max_rows_per_provider": 5,
                "request_timeout_seconds": 2,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "symbols": ["AAPL", "MSFT"],
                "symbols_per_batch": 2,
                "providers": {
                    "alpaca_benzinga": {
                        "enabled": True,
                        "api_key_env": "ALPACA_API_KEY_ID",
                        "secret_key_env": "ALPACA_SECRET_KEY",
                    }
                },
            },
        }
    }
