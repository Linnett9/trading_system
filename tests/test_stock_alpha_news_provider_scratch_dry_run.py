from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)
from core.research.ml.stock_level.news_sources.provider_scratch_dry_run import (
    MAX_REQUEST_CAP,
    MAX_ROW_CAP,
    MAX_SYMBOL_CAP,
    PROVIDER_SCRATCH_DRY_RUN_SCHEMA_VERSION,
    write_provider_scratch_dry_run_report,
)


@dataclass
class MockProviderLikeAdapter:
    """Caller-supplied provider-like adapter for Phase 12 tests only."""

    rows: tuple[Mapping[str, Any], ...]
    provider_id: str = "mock_provider_like"
    provider_family: str = "mock_family"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def collect(
        self,
        *,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "symbols": list(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "limit": int(limit),
            }
        )
        requested = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        selected = [
            dict(row)
            for row in self.rows
            if not requested or str(row.get("symbol", "")).strip().upper() in requested
        ]
        return sorted(selected, key=_row_sort_key)


def test_provider_scratch_dry_run_is_disabled_by_default_and_does_not_collect(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    with pytest.raises(ValueError, match="disabled by default"):
        write_provider_scratch_dry_run_report(
            adapter,
            tmp_path / "provider-scratch",
            symbols=["AAPL"],
            start_date="2024-02-01",
            end_date="2024-02-03",
            max_symbols=1,
            max_rows=3,
            max_requests=1,
        )

    assert adapter.calls == []
    assert not (tmp_path / "provider-scratch").exists()


def test_provider_scratch_dry_run_refuses_network_allowed(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    with pytest.raises(ValueError, match="network_allowed must remain False"):
        write_provider_scratch_dry_run_report(
            adapter,
            tmp_path / "provider-scratch",
            symbols=["AAPL"],
            start_date="2024-02-01",
            end_date="2024-02-03",
            max_symbols=1,
            max_rows=3,
            max_requests=1,
            enabled=True,
            network_allowed=True,
        )

    assert adapter.calls == []


def test_provider_like_rows_flow_to_nested_composition_outputs_under_report_dir(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))
    report_dir = tmp_path / "provider-scratch"

    report, paths = write_provider_scratch_dry_run_report(
        adapter,
        report_dir,
        symbols=[],
        start_date="2024-02-01",
        end_date="2024-02-03",
        max_symbols=MAX_SYMBOL_CAP,
        max_rows=10,
        max_requests=MAX_REQUEST_CAP,
        enabled=True,
    )

    persisted_report = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    sample_rows = json.loads((paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(
        encoding="utf-8"
    ))
    corpus_rows = [
        json.loads(line)
        for line in (paths.composition_dir / "corpus" / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert persisted_report == report
    assert report["schema_version"] == PROVIDER_SCRATCH_DRY_RUN_SCHEMA_VERSION
    assert report["artifact_type"] == "provider_scratch_dry_run_report"
    assert report["provider_id"] == "mock_provider_like"
    assert report["enabled"] is True
    assert report["network_allowed"] is False
    assert report["adapter_collect_invoked"] is True
    assert report["adapter_row_count"] == 6
    assert report["selected_row_count"] == 3
    assert report["corpus_row_count"] == 3
    assert report["excluded_row_count"] == 3
    assert report["sample_skip_reasons"] == {
        "missing_provider": 1,
        "missing_publication_timestamp": 1,
        "missing_symbol": 1,
        "missing_text": 1,
    }
    assert [row["symbol"] for row in sample_rows] == ["AAPL", "TSLA", "NVDA"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "TSLA", "NVDA"]
    assert "Composition report:" in paths.summary_markdown_path.read_text(encoding="utf-8")
    for path in (
        paths.report_json_path,
        paths.summary_markdown_path,
        paths.composition_dir / "composition_smoke_report.json",
        paths.composition_dir / "composition_smoke_summary.md",
        paths.composition_dir / "sample_selection" / "corpus_sample_selection_audit.json",
        paths.composition_dir / "readiness" / "corpus_readiness_audit.json",
        paths.composition_dir / "corpus" / "corpus_manifest.json",
    ):
        assert path.exists()
        path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))


def test_caps_are_enforced_before_collect_and_reported(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    report, _paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "provider-scratch",
        symbols=["aapl", "tsla", "nvda"],
        start_date="2024-02-01",
        end_date="2024-02-03",
        max_symbols=2,
        max_rows=1,
        max_requests=1,
        enabled=True,
    )

    assert adapter.calls == [
        {
            "symbols": ["AAPL", "TSLA"],
            "start_date": "2024-02-01",
            "end_date": "2024-02-03",
            "limit": 1,
        }
    ]
    assert report["symbols"] == ["AAPL", "TSLA"]
    assert report["adapter_row_count"] == 1
    assert report["max_symbols"] == 2
    assert report["max_rows"] == 1
    assert report["max_requests"] == 1
    assert "symbols_capped_to_max_symbols" in report["warnings"]
    assert "max_symbol_cap_enforced" in report["guards"]
    assert "max_row_cap_enforced" in report["guards"]
    assert "max_request_cap_enforced" in report["guards"]


def test_cap_values_above_hard_limits_are_rejected(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    with pytest.raises(ValueError, match="max_rows"):
        write_provider_scratch_dry_run_report(
            adapter,
            tmp_path / "provider-scratch",
            symbols=["AAPL"],
            start_date="2024-02-01",
            end_date="2024-02-03",
            max_symbols=MAX_SYMBOL_CAP,
            max_rows=MAX_ROW_CAP + 1,
            max_requests=MAX_REQUEST_CAP,
            enabled=True,
        )

    assert adapter.calls == []


def test_protected_active_backfill_output_path_is_rejected() -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    with pytest.raises(ValueError, match="protected active backfill"):
        write_provider_scratch_dry_run_report(
            adapter,
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "provider-scratch",
            symbols=["AAPL"],
            start_date="2024-02-01",
            end_date="2024-02-03",
            max_symbols=1,
            max_rows=1,
            max_requests=1,
            enabled=True,
        )

    assert adapter.calls == []


def test_event_type_semantics_and_safety_flags_remain_explicit(tmp_path: Path) -> None:
    adapter = MockProviderLikeAdapter(rows=tuple(_rows()))

    report, paths = write_provider_scratch_dry_run_report(
        adapter,
        tmp_path / "provider-scratch",
        symbols=[],
        start_date="2024-02-01",
        end_date="2024-02-03",
        max_symbols=MAX_SYMBOL_CAP,
        max_rows=10,
        max_requests=MAX_REQUEST_CAP,
        enabled=True,
    )
    sample_rows = json.loads((paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(
        encoding="utf-8"
    ))
    sample_by_symbol = {row["symbol"]: row for row in sample_rows}

    assert sample_by_symbol["NVDA"]["event_type"] == "earnings"
    assert sample_by_symbol["TSLA"]["source_type"] == "SEC_FILING"
    assert sample_by_symbol["TSLA"]["event_type"] is None
    assert report["safety_flags"] == {
        "caller_supplied_adapter_used": True,
        "provider_collection_invoked": False,
        "real_provider_object_instantiated": False,
        "network_invoked": False,
        "download_invoked": False,
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


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("published_at_utc") or ""),
        str(row.get("provider") or ""),
        str(row.get("provider_article_id") or ""),
        str(row.get("symbol") or ""),
    )


def _rows() -> list[dict[str, str]]:
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
        "article_id": f"mock_provider_like:{provider_article_id}:{symbol}",
        "provider": "mock_provider_like",
        "provider_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": published_at_utc,
        "provider_available_at_utc": "2024-02-02T13:01:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": source,
        "source_type": source_type,
        "headline": f"{symbol or 'UNKNOWN'} mocked headline",
        "summary": "Mock provider-like summary",
        "body_or_full_text": "Mock provider-like body text for scratch dry-run checks.",
        "language": "en",
        "event_type": event_type,
        "form_type": form_type,
        "relevance_status": "DIRECT" if symbol else "",
    }
