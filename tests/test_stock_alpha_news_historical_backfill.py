from __future__ import annotations

import csv
import json
from pathlib import Path

from config.config_loader import load_config
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_historical_backfill import (
    SCHEMA_VERSION,
    build_stock_alpha_news_historical_backfill,
    build_stock_alpha_news_historical_corpus_assembly,
    generate_historical_news_partitions,
    write_stock_alpha_news_historical_backfill,
)


class FakeNewsSource:
    api_key_required = False

    def __init__(self, rows=None, *, fail=False, more=False):
        self.rows = list(rows or [_row("1", "AAPL")])
        self.fail = fail
        self.more = more
        self.calls = 0
        self.last_batch_diagnostic = {}

    def with_provider_config(self, provider_config):
        return self

    def collect(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        provider = "alpaca_benzinga"
        self.last_batch_diagnostic = {
            f"{provider}_pages_requested": 1,
            f"{provider}_pages_completed": 1,
            f"{provider}_provider_records_returned": len(self.rows),
            f"{provider}_unique_provider_articles": len({row["provider_article_id"] for row in self.rows}),
            f"{provider}_multi_symbol_expansion_row_count": max(0, len(self.rows) - len({row["provider_article_id"] for row in self.rows})),
            f"{provider}_stopped_with_more_results_available": self.more,
            f"{provider}_termination_reason": "max_rows_per_batch" if self.more else "end_of_results",
        }
        return self.rows


def test_historical_backfill_generates_deterministic_monthly_symbol_batch_partitions(tmp_path):
    config = _config(tmp_path, start="2024-01-15", end="2024-02-02", symbols=["AAPL", "MSFT", "NVDA"], batch_size=2)

    first = generate_historical_news_partitions(config)
    second = generate_historical_news_partitions(config)

    assert first == second
    assert [item["start_date"] for item in first] == ["2024-01-15", "2024-01-15", "2024-02-01", "2024-02-01"]
    assert [item["end_date"] for item in first] == ["2024-01-31", "2024-01-31", "2024-02-02", "2024-02-02"]
    assert first[0]["symbol_batch_id"] == "symbol_batch_001"
    assert first[1]["symbols"] == ["NVDA"]
    assert first[0]["schema_version"] == SCHEMA_VERSION


def test_historical_backfill_resolves_canonical_universe_and_batches(tmp_path):
    stock_rows = tmp_path / "stocks.csv"
    ResearchArtifactWriter().write_csv(
        stock_rows,
        [{"symbol": "MSFT"}, {"symbol": "AAPL"}, {"symbol": "NVDA"}],
        fieldnames=["symbol"],
    )
    config = _config(tmp_path, symbols=[], batch_size=2)
    config["ml"]["stock_alpha_stock_rows_path"] = str(stock_rows)
    settings = config["ml"]["stock_alpha_news_historical_backfill"]
    settings["use_canonical_universe"] = True
    settings["only_symbols"] = ["AAPL", "NVDA"]

    partitions = generate_historical_news_partitions(config)

    assert [partition["symbols"] for partition in partitions] == [["AAPL", "NVDA"]]


def test_historical_backfill_completes_partition_and_writes_manifest_and_artifact(tmp_path):
    config = _config(tmp_path)
    source = FakeNewsSource(rows=[_row("1", "AAPL"), _row("1", "MSFT")])

    paths = write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    record = manifest["partitions"][0]

    assert record["status"] == "complete"
    assert record["attempt_count"] == 1
    assert record["article_symbol_rows"] == 2
    assert record["multi_symbol_expansion_rows"] == 1
    assert Path(record["output_artifact"]).is_file()
    rows = list(csv.DictReader(Path(record["output_artifact"]).open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["publisher"] == "Benzinga"
    assert rows[0]["author"] == "Amit Nag"
    assert rows[0]["raw_source"] == "benzinga"
    assert rows[0]["summary"] == "Short summary"
    assert rows[0]["body_or_full_text"] == ""
    assert rows[0]["body_or_summary_kind"] == "summary"


def test_historical_backfill_resume_skips_complete_partition_without_duplicate_append(tmp_path):
    config = _config(tmp_path)
    source = FakeNewsSource(rows=[_row("1", "AAPL")])

    first = write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    second_payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    artifact = Path(manifest["partitions"][0]["output_artifact"])

    assert source.calls == 1
    assert second_payload["summary"]["skipped_complete"] == 1
    assert len(list(csv.DictReader(artifact.open(encoding="utf-8")))) == 1


def test_historical_backfill_failed_partition_is_recorded_and_retry_can_complete(tmp_path):
    config = _config(tmp_path)

    write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(fail=True)})
    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("2", "AAPL")])})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "complete"
    assert record["attempt_count"] == 2
    assert record["last_error"] == ""


def test_historical_backfill_partial_partition_not_marked_complete_when_more_results_available(tmp_path):
    config = _config(tmp_path)
    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(more=True)})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "partial"
    assert record["stopped_with_more_results_available"] is True
    assert record["output_artifact"] == ""


def test_historical_backfill_existing_artifact_without_complete_manifest_is_rebuilt(tmp_path):
    config = _config(tmp_path)
    partition = generate_historical_news_partitions(config)[0]
    artifact = Path(config["ml"]["stock_alpha_news_historical_backfill"]["work_dir"]) / "partitions" / "alpaca_benzinga" / "2024-01" / f"{partition['partition_id']}.csv"
    ResearchArtifactWriter().write_csv(artifact, [_row("old", "AAPL")], fieldnames=list(_row("old", "AAPL")))

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("new", "AAPL")])})

    assert payload["manifest"]["partitions"][0]["status"] == "complete"
    rows = list(csv.DictReader(artifact.open(encoding="utf-8")))
    assert [row["provider_article_id"] for row in rows] == ["new"]


def test_historical_backfill_filters_publication_window_and_preserves_valid_rows(tmp_path):
    config = _config(tmp_path, start="2024-01-01", end="2024-01-31")
    rows = [
        _row("before", "AAPL", published="2023-12-31T23:59:59Z"),
        _row("start", "AAPL", published="2024-01-01T00:00:00Z"),
        _row("end", "AAPL", published="2024-01-31T23:59:59Z"),
        _row("after", "AAPL", published="2024-02-01T00:00:00Z"),
    ]

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=rows)})
    record = payload["manifest"]["partitions"][0]

    assert record["out_of_window_rejected_count"] == 2
    assert record["article_symbol_rows"] == 2
    artifact_rows = list(csv.DictReader(Path(record["output_artifact"]).open(encoding="utf-8")))
    assert [row["provider_article_id"] for row in artifact_rows] == ["start", "end"]


def test_historical_corpus_assembly_uses_only_complete_partitions_and_deduplicates(tmp_path):
    config = _config(tmp_path)
    write_stock_alpha_news_historical_backfill(
        config,
        sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("1", "AAPL"), _row("1", "AAPL")])},
    )

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["incomplete_partition_count"] == 0
    assert len(rows) == 1
    assert payload["row_count"] == 1
    assert payload["publisher_availability_rate"] == 1.0
    assert payload["author_availability_rate"] == 1.0
    assert payload["text_availability_by_year"]["2024"]["summary_availability_rate"] == 1.0
    assert payload["text_availability_by_year"]["2024"]["body_or_full_text_availability_rate"] == 0.0


def test_historical_corpus_assembly_rejects_incomplete_partitions(tmp_path):
    config = _config(tmp_path)
    build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(more=True)})

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert rows == []
    assert payload["incomplete_partition_count"] == 1
    assert payload["row_count"] == 0


def test_historical_backfill_configs_parse_without_enabling_models_or_trading():
    for path in [
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_tiny_mock_smoke.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_pilot.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_full_template.yaml",
    ]:
        config = load_config(path, overlay_project_config=True)
        ml = config["ml"]
        settings = ml["stock_alpha_news_historical_backfill"]
        assert settings["provider"] == "alpaca_benzinga"
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False


def _config(
    tmp_path: Path,
    *,
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    symbols: list[str] | None = None,
    batch_size: int = 2,
) -> dict:
    return {
        "ml": {
            "stock_alpha_news_historical_backfill": {
                "action": "collect",
                "dry_run": False,
                "work_dir": str(tmp_path / "backfill"),
                "provider": "alpaca_benzinga",
                "start_date": start,
                "end_date": end,
                "partition_frequency": "monthly",
                "symbols": ["AAPL", "MSFT"] if symbols is None else symbols,
                "symbol_batch_size": batch_size,
                "max_partitions_per_run": 10,
                "provider_request_limit": 20,
                "max_rows_per_partition": 100,
                "request_timeout_seconds": 2,
                "rate_limit_sleep_seconds": 0.0,
                "provider_config": {"enabled": True},
            },
            "stock_alpha_news_enable_transformer": False,
        }
    }


def _row(
    provider_id: str,
    symbol: str,
    *,
    published: str = "2024-01-02T10:00:00Z",
    summary: str = "Short summary",
    body: str = "",
) -> dict:
    return {
        "article_id": f"alpaca_benzinga:{provider_id}:{symbol}",
        "symbol": symbol,
        "published_at_utc": published,
        "source": "Benzinga",
        "headline": f"Headline {provider_id}",
        "body_or_summary": body or summary,
        "sentiment_score": "",
        "relevance_score": "",
        "novelty_score": "",
        "event_type": "editorial_news",
        "language": "en",
        "ingested_at": "2024-01-02T10:01:00Z",
        "provider": "alpaca_benzinga",
        "provider_article_id": provider_id,
        "provider_url": f"https://example.test/{provider_id}",
        "provider_symbols": symbol,
        "updated_at_utc": "2024-01-02T10:02:00Z",
        "collected_at_utc": "2024-01-02T10:01:00Z",
        "publisher": "Benzinga",
        "author": "Amit Nag",
        "raw_source": "benzinga",
        "summary": summary,
        "body_or_full_text": body,
        "body_or_summary_kind": "body_or_full_text" if body else "summary",
    }
