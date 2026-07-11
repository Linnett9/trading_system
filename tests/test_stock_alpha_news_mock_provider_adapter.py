from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.news_sources.canonical import (
    SourceType,
    canonical_from_compatibility_row,
)
from core.research.ml.stock_level.news_sources.corpus_composition_smoke import (
    write_corpus_composition_smoke_report,
)


PROTECTED_ACTIVE_BACKFILL_PATH = (
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev"
)


@dataclass(frozen=True)
class MockNewsProviderAdapter:
    """Tiny in-memory provider-like adapter for shape tests only."""

    provider_id: str
    provider_family: str
    rows: tuple[Mapping[str, Any], ...]

    def collect(
        self,
        *,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        del start_date, end_date
        requested = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        selected = [
            dict(row)
            for row in self.rows
            if not requested or str(row.get("symbol", "")).strip().upper() in requested
        ]
        return sorted(selected, key=_row_sort_key)[: max(0, int(limit))]


def test_mock_provider_emits_compatibility_rows_and_converts_to_canonical() -> None:
    adapter = MockNewsProviderAdapter(
        provider_id="mock_news_provider",
        provider_family="mock_provider_family",
        rows=tuple(_mock_rows()),
    )

    rows = adapter.collect(symbols=["NVDA", "AAPL"], start_date="2024-02-01", end_date="2024-02-02", limit=10)
    canonical_rows = [canonical_from_compatibility_row(row, row_number=index) for index, row in enumerate(rows, 1)]

    assert adapter.provider_id == "mock_news_provider"
    assert adapter.provider_family == "mock_provider_family"
    assert [row["symbol"] for row in rows] == ["AAPL", "NVDA"]
    assert [record.symbol for record in canonical_rows] == ["AAPL", "NVDA"]
    assert canonical_rows[0].provider == "mock_news_provider"
    assert canonical_rows[1].event_type == "earnings"


def test_mock_provider_rows_flow_through_composition_smoke_under_tmp_path(tmp_path: Path) -> None:
    adapter = MockNewsProviderAdapter(
        provider_id="mock_news_provider",
        provider_family="mock_provider_family",
        rows=tuple(_mock_rows()),
    )

    report, paths = write_corpus_composition_smoke_report(
        adapter.collect(symbols=[], start_date="2024-02-01", end_date="2024-02-03", limit=10),
        tmp_path / "mock-provider-composition",
        sample_size=10,
    )

    sample_audit = json.loads(
        (paths.sample_selection_dir / "corpus_sample_selection_audit.json").read_text(encoding="utf-8")
    )
    sample_rows = json.loads((paths.sample_selection_dir / "corpus_sample_rows.json").read_text(encoding="utf-8"))
    corpus_rows = [
        json.loads(line)
        for line in (paths.corpus_dir / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert paths.report_json_path.parent == tmp_path / "mock-provider-composition"
    assert report["input_row_count"] == 6
    assert report["selected_row_count"] == 3
    assert report["corpus_row_count"] == 3
    assert report["sample_excluded_row_count"] == 3
    assert report["sample_skip_reasons"] == {
        "missing_provider": 1,
        "missing_publication_timestamp": 1,
        "missing_symbol": 1,
        "missing_text": 1,
    }
    assert sample_audit["excluded_rows"] == [
        {
            "row_number": 1,
            "provider": "mock_news_provider",
            "symbol": "GOOGL",
            "provider_article_id": "mock-missing-publication",
            "reasons": ["missing_publication_timestamp"],
        },
        {
            "row_number": 5,
            "provider": "",
            "symbol": "",
            "provider_article_id": "mock-missing-symbol-provider",
            "reasons": ["missing_symbol", "missing_provider"],
        },
        {
            "row_number": 6,
            "provider": "mock_news_provider",
            "symbol": "AMZN",
            "provider_article_id": "mock-missing-text",
            "reasons": ["missing_text"],
        },
    ]
    assert [row["symbol"] for row in sample_rows] == ["AAPL", "TSLA", "NVDA"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "TSLA", "NVDA"]


def test_mock_provider_preserves_explicit_event_type_and_does_not_infer_sec_form_type(tmp_path: Path) -> None:
    report, paths = write_corpus_composition_smoke_report(
        _mock_rows(),
        tmp_path / "mock-provider-events",
        sample_size=10,
    )
    sample_rows = json.loads((paths.sample_selection_dir / "corpus_sample_rows.json").read_text(encoding="utf-8"))
    corpus_rows = [
        json.loads(line)
        for line in (paths.corpus_dir / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    sample_by_symbol = {row["symbol"]: row for row in sample_rows}
    corpus_by_symbol = {row["symbol"]: row for row in corpus_rows}

    assert report["selected_row_count"] == 3
    assert sample_by_symbol["NVDA"]["event_type"] == "earnings"
    assert corpus_by_symbol["NVDA"]["event_type"] == "earnings"
    assert sample_by_symbol["TSLA"]["source_type"] == "SEC_FILING"
    assert sample_by_symbol["TSLA"]["event_type"] is None
    assert corpus_by_symbol["TSLA"]["source_type"] == "SEC_FILING"
    assert corpus_by_symbol["TSLA"]["event_type"] is None

    canonical_sec = canonical_from_compatibility_row(_mock_rows()[1])
    assert canonical_sec.source_type == SourceType.SEC_FILING
    assert canonical_sec.event_type is None


def test_mock_adapter_does_not_read_api_keys_config_network_or_backfill_paths(tmp_path: Path) -> None:
    adapter = MockNewsProviderAdapter(
        provider_id="mock_news_provider",
        provider_family="mock_provider_family",
        rows=tuple(_mock_rows()),
    )

    report, paths = write_corpus_composition_smoke_report(
        adapter.collect(symbols=["AAPL"], start_date="2024-02-01", end_date="2024-02-02", limit=1),
        tmp_path / "mock-provider-safe",
        sample_size=1,
    )

    assert report["safety_flags"]["provider_collection_invoked"] is False
    assert report["safety_flags"]["network_invoked"] is False
    assert report["safety_flags"]["historical_backfill_invoked"] is False
    assert report["safety_flags"]["feature_generation_invoked"] is False
    assert report["safety_flags"]["model_training_invoked"] is False
    assert report["safety_flags"]["model_inference_invoked"] is False
    assert str(paths.report_json_path).startswith(str(tmp_path))
    assert PROTECTED_ACTIVE_BACKFILL_PATH not in str(paths.report_json_path)


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("published_at_utc") or ""),
        str(row.get("provider") or ""),
        str(row.get("provider_article_id") or ""),
        str(row.get("symbol") or ""),
    )


def _mock_rows() -> list[dict[str, str]]:
    return [
        _row("AAPL", provider_article_id="mock-aapl-1", published_at_utc="2024-02-01T14:30:00Z"),
        _row(
            "TSLA",
            provider_article_id="mock-tsla-8k",
            published_at_utc="2024-02-01T16:05:00Z",
            source="sec",
            source_type="sec_filing",
            form_type="8-K",
        ),
        _row(
            "NVDA",
            provider_article_id="mock-nvda-earnings",
            published_at_utc="2024-02-01T21:05:00Z",
            event_type="earnings",
        ),
        _row("AMZN", provider_article_id="mock-missing-text") | {
            "headline": "",
            "summary": "",
            "body_or_full_text": "",
        },
        _row("GOOGL", provider_article_id="mock-missing-publication") | {
            "published_at_utc": "",
        },
        _row("", provider_article_id="mock-missing-symbol-provider") | {
            "provider": "",
            "provider_symbols": "",
            "symbol": "",
            "source": "",
        },
    ]


def _row(
    symbol: str,
    *,
    provider_article_id: str,
    published_at_utc: str = "2024-02-02T13:00:00Z",
    source: str = "mockwire",
    source_type: str = "newswire",
    event_type: str = "",
    form_type: str = "",
) -> dict[str, str]:
    return {
        "article_id": f"mock_news_provider:{provider_article_id}:{symbol}",
        "provider": "mock_news_provider",
        "provider_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": published_at_utc,
        "provider_available_at_utc": "2024-02-02T13:01:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": source,
        "source_type": source_type,
        "headline": f"{symbol or 'UNKNOWN'} mocked headline",
        "summary": "Mock provider summary",
        "body_or_full_text": "Mock provider body text for canonical and corpus dry-run checks.",
        "language": "en",
        "event_type": event_type,
        "form_type": form_type,
        "relevance_status": "DIRECT" if symbol else "",
    }
