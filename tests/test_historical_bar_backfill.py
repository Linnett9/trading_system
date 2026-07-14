from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

import application.services.historical_bar_backfill_commands as backfill_commands
import infrastructure.data.historical_bar_providers as historical_bar_providers
from application.cli_runtime import FEEDLESS_MODES
from application.services.historical_bar_backfill_commands import _plan, run_historical_bar_backfill_collect
from infrastructure.data.historical_bar_overlap import audit_historical_bar_overlap
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    AlpacaHistoricalBarError,
    BackfillChunkStateStore,
    CollectionManifest,
    HistoricalBarMetrics,
    HistoricalBarPage,
    HistoricalBarRequest,
    ImmutableRawChunkStore,
    SharedRateLimiter,
    _atomic_write_text,
    fetch_chunk_with_retries,
    free_historical_bar_source_inventory,
    resolve_alpaca_credentials,
)
from infrastructure.data.market_sessions import expected_rth_timestamps, is_trading_session, nyse_early_closes, session_type
from infrastructure.data.historical_bar_staging import (
    aggregate_5m_to_1h,
    compare_aggregated_1h,
    consolidate_staging_chunks,
    coverage_gap_audit,
    validate_normalized_bars,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_alpaca_provider_records_provenance_and_pagination_metrics():
    requests = []

    def opener(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        requests.append(query)
        if "page_token" not in query:
            return _Response(
                {
                    "bars": {
                        "AAPL": [
                            {"t": "2026-01-02T14:30:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100}
                        ]
                    },
                    "next_page_token": "two",
                }
            )
        return _Response(
            {
                "bars": {
                    "AAPL": [
                        {"t": "2026-01-02T14:35:00Z", "o": 10.5, "h": 12, "l": 10, "c": 11.5, "v": 200}
                    ]
                },
                "next_page_token": None,
            }
        )

    provider = AlpacaBasicHistoricalBarProvider(
        api_key="key",
        secret_key="secret",
        opener=opener,
        rate_limiter=SharedRateLimiter(requests_per_minute=180, max_in_flight_requests=4, sleeper=lambda _: None),
    )
    rows, chunk = fetch_chunk_with_retries(
        provider,
        HistoricalBarRequest(
            symbols=("AAPL",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            feed="iex",
            raw_chunk_id="chunk-a",
        ),
        max_retries=0,
    )

    assert len(requests) == 2
    assert requests[0]["feed"] == ["iex"]
    assert requests[0]["timeframe"] == ["5Min"]
    assert requests[1]["page_token"] == ["two"]
    assert {key: chunk[key] for key in ("chunk_id", "pages", "rows", "skipped_completed")} == {
        "chunk_id": "chunk-a",
        "pages": 2,
        "rows": 2,
        "skipped_completed": False,
    }
    assert len(chunk["raw_pages"]) == 2
    assert rows[0]["provider"] == "alpaca"
    assert rows[0]["feed"] == "iex"
    assert rows[0]["requested_timeframe"] == "5m"
    assert rows[0]["native_timeframe"] == "5Min"
    assert rows[0]["adjustment_mode"] == "all"
    assert rows[0]["session_policy"] == "all_returned_bars_preserved"
    assert rows[0]["session_type"] == "rth"
    assert rows[0]["raw_chunk_identifier"] == "chunk-a"
    assert rows[0]["normalizer_version"]
    metrics = provider.metrics.as_dict()
    assert metrics["requests_attempted"] == 2
    assert metrics["requests_successful"] == 2
    assert metrics["pages_downloaded"] == 2
    assert metrics["rows_downloaded"] == 2


def test_alpaca_credential_resolution_uses_existing_aliases_without_secret_values(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET", "fake-secret")

    resolved = resolve_alpaca_credentials({})
    report = resolved.public_report()

    assert resolved.credentials_available is True
    assert report == {
        "credential_source": "environment",
        "credentials_available": True,
        "api_key_alias_used": "APCA_API_KEY_ID",
        "secret_key_alias_used": "ALPACA_SECRET",
    }
    assert "fake" not in json.dumps(report)


def test_alpaca_credentials_absent_reports_unavailable(monkeypatch):
    for name in ["ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_SECRET", "APCA_API_SECRET_KEY"]:
        monkeypatch.delenv(name, raising=False)

    report = resolve_alpaca_credentials({}).public_report()

    assert report["credential_source"] == "absent"
    assert report["credentials_available"] is False


def test_shared_rate_limiter_budget_is_global_not_per_worker():
    now = {"value": 0.0}
    sleeps = []

    def sleeper(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    limiter = SharedRateLimiter(
        requests_per_minute=2,
        max_in_flight_requests=4,
        sleeper=sleeper,
        clock=lambda: now["value"],
    )

    limiter.acquire()
    limiter.release()
    limiter.acquire()
    limiter.release()
    limiter.acquire()
    limiter.release()

    assert sleeps == [30.0, 30.0]
    assert limiter.sleep_time_seconds == 60.0


def test_fetch_chunk_retries_429_honors_retry_after_and_persists_state(tmp_path):
    calls = {"count": 0}

    def opener(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "2"},
                None,
            )
        return _Response(
            {
                "bars": {
                    "MSFT": [
                        {"t": "2026-01-02T14:30:00Z", "o": 20, "h": 21, "l": 19, "c": 20.5, "v": 300}
                    ]
                },
                "next_page_token": None,
            }
        )

    sleeps = []
    metrics = HistoricalBarMetrics()
    provider = AlpacaBasicHistoricalBarProvider(
        api_key="key",
        secret_key="secret",
        opener=opener,
        metrics=metrics,
        rate_limiter=SharedRateLimiter(requests_per_minute=180, max_in_flight_requests=4, sleeper=lambda _: None),
    )
    state = BackfillChunkStateStore(tmp_path / "chunks.json")
    rows, chunk = fetch_chunk_with_retries(
        provider,
        HistoricalBarRequest(
            symbols=("MSFT",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            feed="iex",
            raw_chunk_id="chunk-b",
        ),
        state_store=state,
        sleeper=sleeps.append,
    )

    assert len(rows) == 1
    assert chunk["pages"] == 1
    assert sleeps == [2.0]
    assert metrics.requests_attempted == 2
    assert metrics.requests_retried == 1
    assert metrics.http_429_count == 1
    assert state.is_completed("chunk-b")
    skipped, skipped_chunk = fetch_chunk_with_retries(
        provider,
        HistoricalBarRequest(
            symbols=("MSFT",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            feed="iex",
            raw_chunk_id="chunk-b",
        ),
        state_store=state,
    )
    assert skipped == []
    assert skipped_chunk["skipped_completed"] is True


def test_repeated_429_reduces_pressure_and_eventually_fails():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 429, "rate limited", {"Retry-After": "0"}, None)

    limiter = SharedRateLimiter(requests_per_minute=100, max_in_flight_requests=4, sleeper=lambda _: None)
    provider = AlpacaBasicHistoricalBarProvider(
        api_key="key",
        secret_key="secret",
        opener=opener,
        rate_limiter=limiter,
    )

    with pytest.raises(Exception):
        fetch_chunk_with_retries(
            provider,
            HistoricalBarRequest(
                symbols=("AAPL",),
                timeframe="5m",
                start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
                feed="iex",
            ),
            max_retries=2,
            sleeper=lambda _: None,
        )

    assert provider.metrics.http_429_count == 3
    assert limiter.requests_per_minute < 100


def test_permanent_401_and_entitlement_403_are_classified_without_retry():
    for code, classification in [(401, "permanent_authentication_failure"), (403, "entitlement_failure")]:
        calls = {"count": 0}

        def opener(request, timeout, code=code):
            calls["count"] += 1
            raise HTTPError(request.full_url, code, "no", {}, None)

        provider = AlpacaBasicHistoricalBarProvider(api_key="key", secret_key="secret", opener=opener)
        with pytest.raises(Exception) as exc:
            fetch_chunk_with_retries(
                provider,
                HistoricalBarRequest(
                    symbols=("AAPL",),
                    timeframe="5m",
                    start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
                    end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
                    feed="iex",
                ),
                max_retries=3,
            )
        assert exc.value.classification == classification
        assert calls["count"] == 1


def test_empty_valid_window_can_complete():
    provider = AlpacaBasicHistoricalBarProvider(
        api_key="key",
        secret_key="secret",
        opener=lambda request, timeout: _Response({"bars": {}, "next_page_token": None}),
    )
    rows, chunk = fetch_chunk_with_retries(
        provider,
        HistoricalBarRequest(
            symbols=("AAPL",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            feed="iex",
        ),
    )
    assert rows == []
    assert chunk["pages"] == 1


def test_validation_failure_details_are_persisted_in_manifest(tmp_path):
    request = HistoricalBarRequest(
        symbols=("BRK.A",),
        timeframe="5m",
        start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
        feed="sip",
        raw_chunk_id="chunk-brk-a",
        canonical_symbols=("BRK-A",),
        provider_symbol_by_canonical=(("BRK-A", "BRK.A"),),
    )
    manifest = CollectionManifest(tmp_path / "collection_manifest.json")

    class InvalidSymbolProvider:
        name = "alpaca"
        metrics = HistoricalBarMetrics()

        def fetch_page(self, request):
            raise AlpacaHistoricalBarError(
                'Alpaca historical bars request failed (400): {"message":"invalid symbol: BRK.A"}',
                status_code=400,
                response_json={"message": "invalid symbol: BRK.A"},
            )

    result = backfill_commands._collect_one_chunk(
        provider=InvalidSymbolProvider(),
        manifest=manifest,
        raw_store=ImmutableRawChunkStore(tmp_path / "raw"),
        request=request,
        max_retries=0,
        write_raw=True,
        force_refresh=False,
    )

    row = manifest.load()["chunks"]["chunk-brk-a"]
    assert result["status"] == "validation_failure"
    assert row["symbols"] == ["BRK-A"]
    assert row["provider_symbols"] == ["BRK.A"]
    assert row["provider_symbol_map"] == {"BRK-A": "BRK.A"}
    assert row["error_details"]["status_code"] == 400
    assert row["error_details"]["provider_response"] == {"message": "invalid symbol: BRK.A"}
    assert row["error_details"]["invalid_symbol"] == "BRK.A"


def test_symbol_remap_normalizes_provider_rows_back_to_canonical_symbol():
    provider = AlpacaBasicHistoricalBarProvider(
        api_key="key",
        secret_key="secret",
        opener=lambda request, timeout: _Response(
            {
                "bars": {
                    "BRK.B": [
                        {"t": "2026-01-02T14:30:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 100}
                    ]
                },
                "next_page_token": None,
            }
        ),
    )
    rows, _ = fetch_chunk_with_retries(
        provider,
        HistoricalBarRequest(
            symbols=("BRK.B",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            feed="sip",
            raw_chunk_id="chunk-brk-b",
            canonical_symbols=("BRK-B",),
            provider_symbol_by_canonical=(("BRK-B", "BRK.B"),),
        ),
        max_retries=0,
    )

    assert rows[0]["symbol"] == "BRK-B"
    assert rows[0]["provider_symbol"] == "BRK.B"


def test_adjacent_5m_chunk_windows_do_not_overlap_or_skip_boundary_bar():
    requests = _plan(
        ("SPY", "AAPL", "MSFT"),
        datetime(2026, 6, 24, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 6, 30, 21, 0, tzinfo=timezone.utc),
        timeframe="5m",
        feed="iex",
        adjustment="all",
        symbol_batch_size=3,
        date_window_days=5,
    )

    assert len(requests) == 2
    assert requests[0].end == datetime(2026, 6, 29, 13, 25, tzinfo=timezone.utc)
    assert requests[1].start == datetime(2026, 6, 29, 13, 30, tzinfo=timezone.utc)
    assert requests[1].start - requests[0].end == timedelta(minutes=5)


def test_failed_batch_recovery_can_split_into_one_symbol_requests_with_remap():
    requests = _plan(
        ("BOOM", "BP", "BRK-A", "BRK-B"),
        datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
        provider_symbol_map={"BRK-A": "BRK.A", "BRK-B": "BRK.B"},
    )

    assert [request.symbols for request in requests] == [("BOOM",), ("BP",), ("BRK.A",), ("BRK.B",)]
    assert requests[2].canonical_symbols == ("BRK-A",)
    assert requests[2].provider_symbol_by_canonical == (("BRK-A", "BRK.A"),)
    assert "BRK-A" in (requests[2].raw_chunk_id or "")
    assert "BRK.A" not in (requests[2].raw_chunk_id or "")


def test_overlap_audit_compares_without_selecting_or_mixing_canonical_source():
    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    left = [
        {"symbol": "AAPL", "timestamp": timestamp, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "adjustment_mode": "all", "session_policy": "regular"},
        {"symbol": "AAPL", "timestamp": timestamp, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "adjustment_mode": "all", "session_policy": "regular"},
    ]
    right = [
        {"symbol": "AAPL", "timestamp": timestamp, "open": 10, "high": 11, "low": 9, "close": 10.1, "volume": 120, "adjustment_mode": "raw", "session_policy": "extended"},
        {"symbol": "MSFT", "timestamp": timestamp, "open": 20, "high": 21, "low": 19, "close": 20, "volume": 200},
    ]

    report = audit_historical_bar_overlap(left, right, left_provider="alpaca", right_provider="stooq")

    assert report["canonical_source_selected"] is False
    assert report["matched_key_count"] == 1
    assert report["left_duplicate_key_count"] == 1
    assert report["missing_keys_on_left_count"] == 1
    assert report["relative_close_difference_count"] == 1
    assert report["volume_difference_count"] == 1
    assert report["session_alignment_difference_count"] == 1
    assert report["adjustment_difference_count"] == 1


def test_staging_validation_and_conflicting_duplicate_detection():
    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    valid = {
        "symbol": "AAPL",
        "timestamp": timestamp,
        "open": 10,
        "high": 11,
        "low": 9,
        "close": 10,
        "volume": 100,
        "raw_chunk_identifier": "a",
    }
    invalid = {**valid, "symbol": "MSFT", "high": 8}
    assert validate_normalized_bars([valid])["valid"] is True
    assert validate_normalized_bars([invalid])["invalid_ohlc_count"] == 1
    with pytest.raises(ValueError):
        consolidate_staging_chunks([valid, {**valid, "close": 10.5, "raw_chunk_identifier": "b"}])


def test_session_aware_gap_audit_separates_incomplete_from_fully_missing_sessions():
    rows = [
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    report = coverage_gap_audit(
        rows,
        timeframe="5m",
        requested_start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        requested_end=datetime(2026, 1, 5, 14, 45, tzinfo=timezone.utc),
        provider="alpaca",
        feed="iex",
    )[0]
    assert report["intraday_gap_count"] == 1
    assert report["expected_session_count"] == 2
    assert report["observed_session_count"] == 1
    assert report["fully_missing_session_count"] == 1
    assert report["incomplete_session_count"] == 1
    assert report["missing_session_count"] == 1
    assert report["structural_validity_status"] == "valid"
    assert report["completeness_status"] == "incomplete"


def test_session_aware_gap_audit_complete_session_weekend_holiday_and_early_close():
    complete_rows = [
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, minute, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
        for minute in (30, 35, 40)
    ]
    complete = coverage_gap_audit(
        complete_rows,
        timeframe="5m",
        requested_start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        requested_end=datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc),
        provider="alpaca",
        feed="iex",
    )[0]
    assert complete["missing_expected_rth_bars"] == 0
    assert complete["fully_missing_session_count"] == 0
    assert complete["incomplete_session_count"] == 0
    assert complete["completeness_status"] == "complete"

    weekend_holiday = coverage_gap_audit(
        [{"symbol": "AAPL", "timestamp": datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
        timeframe="5m",
        requested_start=datetime(2026, 1, 3, 14, 30, tzinfo=timezone.utc),
        requested_end=datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
        provider="alpaca",
        feed="iex",
        holidays={"2026-01-05"},
    )[0]
    assert weekend_holiday["expected_session_count"] == 0
    assert weekend_holiday["missing_expected_rth_bars"] == 0

    early_close = coverage_gap_audit(
        [{"symbol": "AAPL", "timestamp": datetime(2025, 7, 3, 13, 30, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
        timeframe="5m",
        requested_start=datetime(2025, 7, 3, 13, 30, tzinfo=timezone.utc),
        requested_end=datetime(2025, 7, 3, 21, 0, tzinfo=timezone.utc),
        provider="alpaca",
        feed="iex",
        early_closes={"2025-07-03": "09:35"},
    )[0]
    assert early_close["expected_session_count"] == 1
    assert early_close["missing_expected_rth_bars"] == 0


def test_session_type_uses_eastern_time_and_handles_dst_boundaries():
    assert session_type(datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc)) == "rth"
    assert session_type(datetime(2026, 7, 1, 19, 55, tzinfo=timezone.utc)) == "rth"
    assert session_type(datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)) == "after_hours"

    assert session_type(datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)) == "rth"
    assert session_type(datetime(2026, 1, 2, 20, 55, tzinfo=timezone.utc)) == "rth"
    assert session_type(datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc)) == "after_hours"


def test_expected_normal_full_day_has_78_rth_5m_bars_and_early_close_has_42():
    normal = expected_rth_timestamps(
        datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 0, 0, tzinfo=timezone.utc),
        step=timedelta(minutes=5),
    )
    assert len(normal) == 78
    assert normal[0] == datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    assert normal[-1] == datetime(2026, 1, 2, 20, 55, tzinfo=timezone.utc)

    early = expected_rth_timestamps(
        datetime(2025, 7, 3, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 7, 4, 0, 0, tzinfo=timezone.utc),
        step=timedelta(minutes=5),
    )
    assert len(early) == 42
    assert early[0] == datetime(2025, 7, 3, 13, 30, tzinfo=timezone.utc)
    assert early[-1] == datetime(2025, 7, 3, 16, 55, tzinfo=timezone.utc)


def test_market_calendar_covers_2016_2026_holidays_and_early_closes():
    assert not is_trading_session(datetime(2016, 3, 25).date())  # Good Friday.
    assert not is_trading_session(datetime(2021, 12, 31).date())  # New Year observed.
    assert not is_trading_session(datetime(2025, 1, 9).date())  # Carter mourning closure.
    assert not is_trading_session(datetime(2026, 4, 3).date())  # Good Friday.
    assert not is_trading_session(datetime(2026, 7, 4).date())  # Weekend.
    assert datetime(2025, 11, 28).date() in nyse_early_closes(2025)
    assert datetime(2025, 12, 24).date() in nyse_early_closes(2025)


def test_coverage_audit_uses_rth_research_view_without_forward_fill():
    rows = [
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "session_type": "pre_market"},
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "session_type": "rth"},
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 21, 5, tzinfo=timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "session_type": "after_hours"},
    ]
    report = coverage_gap_audit(
        rows,
        timeframe="5m",
        requested_start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        requested_end=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc),
        provider="alpaca",
        feed="sip",
    )[0]
    assert report["row_count"] == 1
    assert report["research_view"] == "rth_only"
    assert report["no_forward_fill"] is True
    assert report["no_synthetic_bars"] is True
    assert report["no_stale_close_carry"] is True
    assert report["missing_expected_rth_bars"] == 1


def test_5m_to_1h_aggregation_comparison_recommends_local_derivation_when_aligned():
    rows = [
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc), "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
        {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc), "open": 10.5, "high": 12, "low": 10, "close": 11, "volume": 200},
    ]
    aggregated = aggregate_5m_to_1h(rows)
    report = compare_aggregated_1h(aggregated, [{**aggregated[0]}])
    assert report["recommendation"] == "derive_1h_locally_from_validated_5m"


def test_collect_dry_run_writes_no_bars_or_canonical_data(tmp_path, monkeypatch):
    for name in ["ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_SECRET", "APCA_API_SECRET_KEY"]:
        monkeypatch.delenv(name, raising=False)
    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "symbols": ["SPY", "AAPL", "MSFT"],
                    "timeframe": "5m",
                    "start": "2026-01-02T14:30:00+00:00",
                    "end": "2026-01-02T15:00:00+00:00",
                    "output_root": str(tmp_path / "reports"),
                    "raw_root": str(tmp_path / "raw"),
                    "dry_run": True,
                    "symbol_batch_size": 2,
                    "date_window_days": 1,
                }
            }
        }
    )
    report = json.loads((tmp_path / "reports" / "historical_bar_collect_report.json").read_text())
    assert report["dry_run"] is True
    assert report["planned_chunk_count"] == 2
    assert not (tmp_path / "raw").exists()
    assert report["canonical_market_data_modified"] is False
    assert report["manifest_plan_initialization"]["write_performed"] is True


def test_manifest_initialize_plan_bulk_writes_once_for_large_plan(tmp_path):
    writes = []

    def writer(path, text):
        writes.append((path, text))
        path.write_text(text, encoding="utf-8")

    requests = [
        HistoricalBarRequest(
            symbols=(f"SYM{i % 1000:04d}",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            end=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            feed="sip",
            raw_chunk_id=f"chunk-{i:05d}",
        )
        for i in range(40_000)
    ]

    manifest = CollectionManifest(tmp_path / "collection_manifest.json", writer=writer)
    result = manifest.initialize_plan(requests, dry_run=True)

    assert result["final_chunk_count"] == 40_000
    assert result["added_planned_count"] == 40_000
    assert len(writes) == 1
    assert len(json.loads(writes[0][1])["chunks"]) == 40_000

    repeated = manifest.initialize_plan(requests, dry_run=True)

    assert repeated["write_performed"] is False
    assert len(writes) == 1


def test_manifest_initialize_plan_preserves_partial_meaningful_states(tmp_path):
    manifest = CollectionManifest(tmp_path / "collection_manifest.json")
    manifest.update("chunk-complete", "completed", {"rows": 10})
    manifest.update("chunk-progress", "in_progress", {"attempt": 1})
    manifest.update("chunk-failed", "retryable_failure", {"error_message": "rate limited"})
    manifest.update("chunk-planned", "planned", {"dry_run": False})
    requests = [
        HistoricalBarRequest(
            symbols=("SPY",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc),
            feed="sip",
            raw_chunk_id=chunk_id,
        )
        for chunk_id in ["chunk-complete", "chunk-progress", "chunk-failed", "chunk-planned", "chunk-new"]
    ]

    result = manifest.initialize_plan(requests, dry_run=True)
    chunks = manifest.load()["chunks"]

    assert result["final_chunk_count"] == 5
    assert chunks["chunk-complete"]["status"] == "completed"
    assert chunks["chunk-complete"]["rows"] == 10
    assert chunks["chunk-progress"]["status"] == "in_progress"
    assert chunks["chunk-failed"]["status"] == "retryable_failure"
    assert chunks["chunk-planned"]["status"] == "planned"
    assert chunks["chunk-planned"]["dry_run"] is True
    assert chunks["chunk-new"]["status"] == "planned"


def test_manifest_updates_append_journal_without_rewriting_large_snapshot(tmp_path):
    writes = []

    def writer(path, text):
        writes.append((path, len(text)))
        path.write_text(text, encoding="utf-8")

    requests = [
        HistoricalBarRequest(
            symbols=(f"SYM{i % 1000:04d}",),
            timeframe="5m",
            start=datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            end=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            feed="sip",
            raw_chunk_id=f"chunk-{i:05d}",
        )
        for i in range(40_000)
    ]
    manifest = CollectionManifest(tmp_path / "collection_manifest.json", writer=writer)
    manifest.initialize_plan(requests, dry_run=False)

    for request in requests[:1_000]:
        manifest.update(request.raw_chunk_id or "", "completed", {"rows": 1})

    assert len(writes) == 1
    assert (tmp_path / "collection_manifest.json.events.jsonl").exists()
    assert len((tmp_path / "collection_manifest.json.events.jsonl").read_text(encoding="utf-8").splitlines()) == 1_000
    reloaded = CollectionManifest(tmp_path / "collection_manifest.json")
    assert reloaded.is_completed("chunk-00999")
    assert not reloaded.is_completed("chunk-01000")

    reloaded.checkpoint()

    assert (tmp_path / "collection_manifest.json.events.jsonl").read_text(encoding="utf-8") == ""
    assert CollectionManifest(tmp_path / "collection_manifest.json").is_completed("chunk-00999")


def test_manifest_replay_preserves_valid_events_before_truncated_final_line(tmp_path):
    path = tmp_path / "collection_manifest.json"
    path.write_text(json.dumps({"chunks": {"chunk-0": {"chunk_id": "chunk-0", "status": "planned"}}}), encoding="utf-8")
    journal = tmp_path / "collection_manifest.json.events.jsonl"
    journal.write_text(
        json.dumps({"chunk_id": "chunk-0", "status": "completed", "updated_at": "2026-01-01T00:00:00+00:00"})
        + "\n"
        + '{"chunk_id":"chunk-1","status":"completed"',
        encoding="utf-8",
    )

    reloaded = CollectionManifest(path)

    assert reloaded.is_completed("chunk-0")
    assert reloaded.status("chunk-1") is None


def test_manifest_replay_rejects_malformed_nonfinal_journal_line(tmp_path):
    path = tmp_path / "collection_manifest.json"
    path.write_text(json.dumps({"chunks": {}}), encoding="utf-8")
    journal = tmp_path / "collection_manifest.json.events.jsonl"
    journal.write_text(
        '{"chunk_id":"bad","status":"completed"\n'
        + json.dumps({"chunk_id": "chunk-0", "status": "completed", "updated_at": "2026-01-01T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        CollectionManifest(path).load()


def test_manifest_replays_in_progress_and_retryable_states_as_retryable_on_restart(tmp_path):
    manifest = CollectionManifest(tmp_path / "collection_manifest.json")
    manifest.update("chunk-in-progress", "in_progress", {"attempt": 1})
    manifest.update("chunk-retry", "retryable_failure", {"error_message": "transport"})

    reloaded = CollectionManifest(tmp_path / "collection_manifest.json")

    assert reloaded.status("chunk-in-progress") == "in_progress"
    assert reloaded.status("chunk-retry") == "retryable_failure"
    assert not reloaded.is_completed("chunk-in-progress")
    assert not reloaded.is_completed("chunk-retry")


def test_atomic_write_replaces_existing_destination_with_unrelated_temp_present(tmp_path):
    target = tmp_path / "collection_manifest.json"
    target.write_text('{"old": true}', encoding="utf-8")
    stale = tmp_path / "collection_manifest.json.tmp"
    stale.write_text("stale", encoding="utf-8")

    _atomic_write_text(target, '{"new": true}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    assert stale.read_text(encoding="utf-8") == "stale"


def test_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "collection_manifest.json"
    calls = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(historical_bar_providers.os, "replace", flaky_replace)
    monkeypatch.setattr(historical_bar_providers.time, "sleep", lambda _: None)

    _atomic_write_text(target, '{"ok": true}')

    assert calls["count"] == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_write_retry_exhaustion_preserves_previous_destination(tmp_path, monkeypatch):
    target = tmp_path / "collection_manifest.json"
    target.write_text('{"valid": true}', encoding="utf-8")

    def blocked_replace(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(historical_bar_providers.os, "replace", blocked_replace)
    monkeypatch.setattr(historical_bar_providers.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        _atomic_write_text(target, '{"valid": false}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"valid": True}


def test_collect_dry_run_accepts_sip_and_reports_feed_domain_shift(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["SPY"],
                    "timeframe": "5m",
                    "start": "2026-01-02T14:30:00+00:00",
                    "end": "2026-01-02T15:00:00+00:00",
                    "output_root": str(tmp_path / "reports"),
                    "dry_run": True,
                }
            }
        }
    )
    report = json.loads((tmp_path / "reports" / "historical_bar_collect_report.json").read_text())
    assert report["dry_run"] is True
    assert "feed-domain" in report["feed_domain_shift_note"]


def test_collect_resume_reloads_completed_raw_chunk_into_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)
    request = _plan(
        ("SPY",),
        start,
        end,
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
    )[0]
    row = {
        "symbol": "SPY",
        "timestamp": start,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 100.0,
        "provider": "alpaca",
        "feed": "sip",
        "raw_chunk_identifier": request.raw_chunk_id,
        "session_type": "pre_market",
    }
    raw_root = tmp_path / "raw"
    ImmutableRawChunkStore(raw_root).write_completed_chunk(
        request,
        rows=[row],
        raw_pages=[{"bars": {"SPY": []}}],
        metrics={"pages": 1},
    )
    output_root = tmp_path / "reports"
    CollectionManifest(output_root / "collection_manifest.json").update(
        request.raw_chunk_id or "",
        "completed",
        {"rows": 1},
    )

    class NoFetchProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {
                "credential_source": "test",
                "credentials_available": True,
                "api_key_alias_used": None,
                "secret_key_alias_used": None,
                "can_attempt_authenticated_request": True,
            }

        def fetch_page(self, request):
            raise AssertionError("completed resumed chunks should be read from raw storage")

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", NoFetchProvider)

    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["SPY"],
                    "timeframe": "5m",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "output_root": str(output_root),
                    "raw_root": str(raw_root),
                    "dry_run": False,
                    "resume": True,
                    "write_normalized_staging": True,
                }
            }
        }
    )

    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))

    assert report["chunk_results"] == [
        {"chunk_id": request.raw_chunk_id, "status": "skipped_completed", "rows": 1, "raw_reload_deferred": False}
    ]
    assert report["all_session_row_count"] == 1
    assert report["staging_path"]
    assert (output_root / "staging" / "sip" / "5m" / "bars.parquet").exists()


def test_raw_store_shortens_long_symbol_batch_paths_and_preserves_manifest_symbols(tmp_path):
    symbols = tuple(f"SYM{i:02d}" for i in range(50))
    request = HistoricalBarRequest(
        symbols=symbols,
        timeframe="1Day",
        start=datetime(2026, 3, 27, tzinfo=timezone.utc),
        end=datetime(2026, 4, 26, tzinfo=timezone.utc),
        feed="sip",
        raw_chunk_id="long-batch",
    )
    store = ImmutableRawChunkStore(tmp_path / "raw")
    chunk_dir = store.chunk_dir(request)

    assert len(chunk_dir.parent.name) <= 120
    assert chunk_dir.parent.name.startswith("batch-50-SYM00-SYM49-")
    assert "-".join(symbols) not in str(chunk_dir)

    store.write_completed_chunk(
        request,
        rows=[
            {
                "symbol": "SYM00",
                "timestamp": request.start,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            }
        ],
        raw_pages=[{"bars": {}}],
        metrics={"pages": 1},
    )
    manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["symbol_batch"] == list(symbols)
    assert manifest["symbol_batch_path"] == chunk_dir.parent.name
    assert store.read_completed_chunk(request)[0]["symbol"] == "SYM00"


def test_http_success_raw_write_failure_is_retryable_with_failure_phase(tmp_path):
    request = HistoricalBarRequest(
        symbols=("AAPL",),
        timeframe="1Day",
        start=datetime(2026, 3, 27, tzinfo=timezone.utc),
        end=datetime(2026, 4, 26, tzinfo=timezone.utc),
        feed="sip",
        raw_chunk_id="raw-failure",
    )
    manifest = CollectionManifest(tmp_path / "manifest.json")

    class SuccessfulProvider:
        name = "alpaca"

        def __init__(self):
            self.metrics = HistoricalBarMetrics()

        def fetch_page(self, request):
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += 1
            return HistoricalBarPage(
                bars=[
                    {
                        "symbol": "AAPL",
                        "timestamp": request.start,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 100.0,
                    }
                ],
                next_page_token=None,
                raw_payload={"bars": {"AAPL": []}},
                latency_seconds=0.0,
            )

    class FailingRawStore:
        def write_completed_chunk(self, *args, **kwargs):
            raise FileNotFoundError("simulated raw path failure")

    result = backfill_commands._collect_one_chunk(
        provider=SuccessfulProvider(),
        manifest=manifest,
        raw_store=FailingRawStore(),
        request=request,
        max_retries=0,
        write_raw=True,
        force_refresh=False,
    )
    state = manifest.load()["chunks"]["raw-failure"]

    assert result["status"] == "retryable_failure"
    assert result["failure_phase"] == "raw_chunk_publication"
    assert result["rows"] == 1
    assert result["pages"] == 1
    assert result["requests"] == 1
    assert state["failure_phase"] == "raw_chunk_publication"
    assert state["rows_returned"] == 1


def test_collect_resume_does_not_preload_completed_raw_before_bounded_collection(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports"
    raw_root = tmp_path / "raw"
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 4, 14, 30, tzinfo=timezone.utc)
    planned = _plan(
        ("SPY",),
        start,
        end,
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
    )
    completed_row = {
        "symbol": "SPY",
        "timestamp": planned[0].start,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 100.0,
        "provider": "alpaca",
        "feed": "sip",
        "raw_chunk_identifier": planned[0].raw_chunk_id,
        "session_type": "rth",
    }
    ImmutableRawChunkStore(raw_root).write_completed_chunk(
        planned[0],
        rows=[completed_row],
        raw_pages=[{"bars": {"SPY": []}}],
        metrics={"pages": 1},
    )
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    manifest.initialize_plan(planned, dry_run=False)
    manifest.update(planned[0].raw_chunk_id or "", "completed", {"rows": 1})
    reads = []

    def fail_if_read(self, request):
        reads.append(request.raw_chunk_id)
        raise AssertionError("bounded resume should not read completed raw rows before collection")

    class OneChunkProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {
                "credential_source": "test",
                "credentials_available": True,
                "api_key_alias_used": None,
                "secret_key_alias_used": None,
                "can_attempt_authenticated_request": True,
            }

        def fetch_page(self, request):
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += 1
            return HistoricalBarPage(
                bars=[
                    {
                        "symbol": "SPY",
                        "timestamp": request.start,
                        "open": 2.0,
                        "high": 2.0,
                        "low": 2.0,
                        "close": 2.0,
                        "volume": 200.0,
                        "provider": "alpaca",
                        "feed": request.feed,
                        "raw_chunk_identifier": request.raw_chunk_id,
                    }
                ],
                next_page_token=None,
                raw_payload={"bars": {"SPY": []}},
                latency_seconds=0.0,
            )

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", OneChunkProvider)
    monkeypatch.setattr(ImmutableRawChunkStore, "read_completed_chunk", fail_if_read)

    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["SPY"],
                    "timeframe": "5m",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "output_root": str(output_root),
                    "raw_root": str(raw_root),
                    "dry_run": False,
                    "resume": True,
                    "write_normalized_staging": True,
                    "date_window_days": 1,
                    "max_collect_chunks": 1,
                    "progress_report_interval_seconds": 0,
                }
            }
        }
    )

    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))
    assert reads == []
    assert report["staging_deferred_reason"] == "max_collect_chunks_bounded_collection"
    assert report["staging_path"] is None
    assert any(row["status"] == "completed" for row in report["chunk_results"])


def test_bounded_daily_collection_consolidates_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports" / "daily_smoke"
    raw_root = tmp_path / "raw"
    archive_root = tmp_path / "archive_smoke"
    assets, aliases = _daily_registry(tmp_path)
    start = datetime(2026, 6, 22, tzinfo=timezone.utc)
    end = datetime(2026, 6, 24, 23, 59, 59, tzinfo=timezone.utc)
    calls = []

    class DailyProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {"provider": "alpaca", "feed": "sip", "credentials_available": True, "can_attempt_authenticated_request": True}

        def fetch_page(self, request):
            calls.append(request.raw_chunk_id)
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += 12
            rows = []
            for day in (22, 23, 24):
                for symbol in request.symbols:
                    canonical = "BRK-B" if symbol == "BRK.B" else symbol
                    rows.append(
                        {
                            "symbol": canonical,
                            "provider_symbol": symbol if symbol != canonical else None,
                            "timestamp": datetime(2026, 6, day, 4, tzinfo=timezone.utc),
                            "open": 1.0,
                            "high": 2.0,
                            "low": 1.0,
                            "close": 2.0,
                            "volume": 100.0,
                            "trade_count": 10,
                            "vwap": 1.5,
                            "provider": "alpaca",
                            "feed": request.feed,
                            "requested_timeframe": request.timeframe,
                            "native_timeframe": "1Day",
                            "adjustment_mode": request.adjustment,
                            "raw_chunk_identifier": request.raw_chunk_id,
                        }
                    )
            return HistoricalBarPage(bars=rows, next_page_token=None, raw_payload={"bars": {}}, latency_seconds=0.0)

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", DailyProvider)

    config = _daily_collect_config(output_root, raw_root, archive_root, assets, aliases, start, end, allow_bounded=True)
    run_historical_bar_backfill_collect(config)

    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))
    assert calls == ["alpaca-sip-1Day-AAPL-SPY-BRK-B-ABCB-20260622T000000Z-20260624T235959Z"]
    assert report["bounded_consolidation_performed"] is True
    assert report["staging_path"]
    assert report["daily_archive_report"]["written_rows"] == 12
    assert report["daily_session_row_count"] == 12
    assert report["intraday_rth_classification_applicable"] is False
    assert report["rth_row_count"] is None
    assert (archive_root / "symbol=BRK-B" / "year=2026" / "bars.parquet").exists()


def test_bounded_consolidation_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports" / "daily_smoke"
    raw_root = tmp_path / "raw"
    archive_root = tmp_path / "archive_smoke"
    assets, aliases = _daily_registry(tmp_path)
    start = datetime(2026, 6, 22, tzinfo=timezone.utc)
    end = datetime(2026, 6, 24, 23, 59, 59, tzinfo=timezone.utc)

    class EmptyProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {"provider": "alpaca", "feed": "sip", "credentials_available": True, "can_attempt_authenticated_request": True}

        def fetch_page(self, request):
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            return HistoricalBarPage(bars=[], next_page_token=None, raw_payload={"bars": {}}, latency_seconds=0.0)

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", EmptyProvider)

    run_historical_bar_backfill_collect(_daily_collect_config(output_root, raw_root, archive_root, assets, aliases, start, end, allow_bounded=False))

    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))
    assert report["bounded_consolidation_eligible"] is False
    assert report["staging_deferred_reason"] == "max_collect_chunks_bounded_collection"
    assert report["staging_path"] is None
    assert not archive_root.exists()


def test_bounded_daily_rerun_reuses_completed_raw_without_api_request(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports" / "daily_smoke"
    raw_root = tmp_path / "raw"
    archive_root = tmp_path / "archive_smoke"
    assets, aliases = _daily_registry(tmp_path)
    start = datetime(2026, 6, 22, tzinfo=timezone.utc)
    end = datetime(2026, 6, 24, 23, 59, 59, tzinfo=timezone.utc)
    request = _plan(("AAPL", "SPY", "BRK-B", "ABCB"), start, end, timeframe="1Day", feed="sip", adjustment="all", symbol_batch_size=4, date_window_days=7, provider_symbol_map={"BRK-B": "BRK.B"})[0]
    rows = [
        {"symbol": "AAPL", "timestamp": datetime(2026, 6, 22, 4, tzinfo=timezone.utc), "open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "volume": 100.0, "provider": "alpaca", "feed": "sip", "adjustment_mode": "all", "raw_chunk_identifier": request.raw_chunk_id},
        {"symbol": "BRK-B", "provider_symbol": "BRK.B", "timestamp": datetime(2026, 6, 22, 4, tzinfo=timezone.utc), "open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "volume": 100.0, "provider": "alpaca", "feed": "sip", "adjustment_mode": "all", "raw_chunk_identifier": request.raw_chunk_id},
    ]
    ImmutableRawChunkStore(raw_root).write_completed_chunk(request, rows=rows, raw_pages=[{"bars": {}}], metrics={"pages": 1})
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    manifest.initialize_plan([request], dry_run=False)
    manifest.update(request.raw_chunk_id or "", "completed", {"rows": 2})

    class NoFetchProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {"provider": "alpaca", "feed": "sip", "credentials_available": True, "can_attempt_authenticated_request": True}

        def fetch_page(self, request):
            raise AssertionError("completed bounded chunks should be reused")

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", NoFetchProvider)

    run_historical_bar_backfill_collect(_daily_collect_config(output_root, raw_root, archive_root, assets, aliases, start, end, allow_bounded=True))

    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))
    assert report["observed_metrics"]["requests_attempted"] == 0
    assert report["chunk_results"][0]["status"] == "skipped_completed"
    assert report["chunk_results"][0]["rows"] == 2
    assert report["daily_archive_report"]["written_rows"] == 2


def test_collect_resume_retries_stale_in_progress_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports"
    raw_root = tmp_path / "raw"
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)
    planned = _plan(
        ("SPY",),
        start,
        end,
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
    )
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    manifest.initialize_plan(planned, dry_run=False)
    manifest.update(planned[0].raw_chunk_id or "", "in_progress", {"attempt": 1})
    calls = []

    class RetryProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {
                "credential_source": "test",
                "credentials_available": True,
                "api_key_alias_used": None,
                "secret_key_alias_used": None,
                "can_attempt_authenticated_request": True,
            }

        def fetch_page(self, request):
            calls.append(request.raw_chunk_id)
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += 1
            return HistoricalBarPage(
                bars=[
                    {
                        "symbol": "SPY",
                        "timestamp": request.start,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 100.0,
                        "provider": "alpaca",
                        "feed": request.feed,
                        "raw_chunk_identifier": request.raw_chunk_id,
                    }
                ],
                next_page_token=None,
                raw_payload={"bars": {"SPY": []}},
                latency_seconds=0.0,
            )

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", RetryProvider)

    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["SPY"],
                    "timeframe": "5m",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "output_root": str(output_root),
                    "raw_root": str(raw_root),
                    "dry_run": False,
                    "resume": True,
                    "write_normalized_staging": True,
                    "progress_report_interval_seconds": 0,
                }
            }
        }
    )

    assert calls == [planned[0].raw_chunk_id]
    assert CollectionManifest(output_root / "collection_manifest.json").is_completed(planned[0].raw_chunk_id or "")


def test_collect_status_filter_retries_only_validation_failures_and_preserves_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports"
    raw_root = tmp_path / "raw"
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    planned = _plan(
        ("AAPL", "BRK-B", "MSFT"),
        start,
        end,
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
        provider_symbol_map={"BRK-B": "BRK.B"},
    )
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    manifest.initialize_plan(planned, dry_run=False)
    manifest.update(planned[0].raw_chunk_id or "", "completed", {"rows": 1})
    manifest.update(planned[1].raw_chunk_id or "", "validation_failure", {"error_message": "invalid symbol: BRK-B"})
    calls = []

    class RetryValidationFailureProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {
                "credential_source": "test",
                "credentials_available": True,
                "api_key_alias_used": None,
                "secret_key_alias_used": None,
                "can_attempt_authenticated_request": True,
            }

        def fetch_page(self, request):
            calls.append((request.raw_chunk_id, request.symbols))
            self.metrics.requests_attempted += 1
            self.metrics.requests_successful += 1
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += 1
            return HistoricalBarPage(
                bars=[
                    {
                        "symbol": "BRK-B",
                        "provider_symbol": "BRK.B",
                        "timestamp": request.start,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 100.0,
                        "provider": "alpaca",
                        "feed": request.feed,
                        "raw_chunk_identifier": request.raw_chunk_id,
                    }
                ],
                next_page_token=None,
                raw_payload={"bars": {"BRK.B": []}},
                latency_seconds=0.0,
            )

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", RetryValidationFailureProvider)

    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["AAPL", "BRK-B", "MSFT"],
                    "provider_symbol_map": {"BRK-B": "BRK.B"},
                    "timeframe": "5m",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "output_root": str(output_root),
                    "raw_root": str(raw_root),
                    "dry_run": False,
                    "resume": True,
                    "write_normalized_staging": True,
                    "symbol_batch_size": 1,
                    "date_window_days": 1,
                    "collect_statuses": ["validation_failure"],
                    "max_collect_chunks": 1,
                    "progress_report_interval_seconds": 0,
                }
            }
        }
    )

    assert calls == [(planned[1].raw_chunk_id, ("BRK.B",))]
    state = CollectionManifest(output_root / "collection_manifest.json").load()["chunks"]
    assert state[planned[0].raw_chunk_id or ""]["status"] == "completed"
    assert state[planned[1].raw_chunk_id or ""]["status"] == "completed"
    assert state[planned[2].raw_chunk_id or ""]["status"] == "planned"


def test_collect_parallel_workers_do_not_collect_same_chunk_twice_and_retry_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    output_root = tmp_path / "reports"
    raw_root = tmp_path / "raw"
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    planned = _plan(
        ("AAPL", "MSFT", "NVDA", "AMZN"),
        start,
        end,
        timeframe="5m",
        feed="sip",
        adjustment="all",
        symbol_batch_size=1,
        date_window_days=1,
    )
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    manifest.initialize_plan(planned, dry_run=False)
    manifest.update(planned[1].raw_chunk_id or "", "retryable_failure", {"error_message": "previous transport failure"})
    calls = []
    lock = threading.Lock()

    class ConcurrentFakeProvider:
        name = "alpaca"

        def __init__(self, *args, **kwargs):
            self.metrics = HistoricalBarMetrics()

        def check_authentication(self):
            return {
                "credential_source": "test",
                "credentials_available": True,
                "api_key_alias_used": None,
                "secret_key_alias_used": None,
                "can_attempt_authenticated_request": True,
            }

        def fetch_page(self, request):
            with lock:
                calls.append(request.raw_chunk_id)
                self.metrics.requests_attempted += 1
                self.metrics.requests_successful += 1
                self.metrics.pages_downloaded += 1
                self.metrics.rows_downloaded += 1
            time.sleep(0.01)
            return HistoricalBarPage(
                bars=[
                    {
                        "symbol": request.symbols[0],
                        "timestamp": request.start,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 100.0,
                        "provider": "alpaca",
                        "feed": request.feed,
                        "collection_timestamp": "test",
                        "requested_timeframe": request.timeframe,
                        "native_timeframe": "5Min",
                        "adjustment_mode": request.adjustment,
                        "extended_hours": False,
                        "session_policy": "all_returned_bars_preserved",
                        "session_type": "rth",
                        "raw_chunk_identifier": request.raw_chunk_id,
                        "normalizer_version": "test",
                    }
                ],
                next_page_token=None,
                raw_payload={"bars": {request.symbols[0]: []}},
                latency_seconds=0.01,
            )

    monkeypatch.setattr(backfill_commands, "AlpacaBasicHistoricalBarProvider", ConcurrentFakeProvider)

    run_historical_bar_backfill_collect(
        {
            "ml": {
                "historical_bar_backfill": {
                    "provider": "alpaca",
                    "feed": "sip",
                    "symbols": ["AAPL", "MSFT", "NVDA", "AMZN"],
                    "timeframe": "5m",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "output_root": str(output_root),
                    "raw_root": str(raw_root),
                    "dry_run": False,
                    "resume": True,
                    "write_normalized_staging": True,
                    "symbol_batch_size": 1,
                    "date_window_days": 1,
                    "collection_workers": 4,
                    "progress_report_interval_seconds": 0,
                }
            }
        }
    )

    assert sorted(calls) == sorted(request.raw_chunk_id for request in planned)
    assert len(calls) == len(set(calls)) == len(planned)
    reloaded = CollectionManifest(output_root / "collection_manifest.json")
    assert all(reloaded.is_completed(request.raw_chunk_id or "") for request in planned)
    report = json.loads((output_root / "historical_bar_collect_report.json").read_text(encoding="utf-8"))
    assert report["collection_workers"] == 4
    assert report["post_consolidation_validation"]["valid"] is True


def test_free_source_inventory_rejects_paid_sources_and_keeps_stooq_as_overlap_source():
    inventory = free_historical_bar_source_inventory()

    implemented = {row["provider"]: row for row in inventory["implemented_operational_free_sources"]}
    rejected = {row["provider"] for row in inventory["rejected_for_immediate_free_implementation"]}
    assert implemented["alpaca"]["feed"] == "iex"
    assert implemented["stooq"]["timeframes"] == ["1Day"]
    assert "alpaca_sip" in rejected
    assert "databento" in rejected


def test_historical_bar_probe_mode_is_feedless():
    assert "ml-historical-bar-backfill-probe" in FEEDLESS_MODES
    assert "ml-historical-bar-backfill-collect" in FEEDLESS_MODES


def _daily_registry(tmp_path):
    assets = tmp_path / "assets.csv"
    aliases = tmp_path / "aliases.csv"
    asset_fields = ["asset_id", "canonical_symbol", "security_name", "security_type", "share_class", "exchange", "currency", "country", "cik", "sector", "industry", "valid_from", "valid_to", "is_active", "collection_universe_514", "registry_version"]
    alias_fields = ["asset_id", "provider", "provider_symbol", "valid_from", "valid_to", "is_primary", "mapping_reason", "source", "registry_version"]
    with assets.open("w", encoding="utf-8", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=asset_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for symbol in ["AAPL", "SPY", "BRK-B", "ABCB"]:
            writer.writerow({"asset_id": f"asset_{symbol}", "canonical_symbol": symbol, "security_type": "UNKNOWN", "currency": "USD", "country": "US", "valid_from": "1900-01-01", "is_active": "true", "collection_universe_514": "true", "registry_version": "test"})
    with aliases.open("w", encoding="utf-8", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=alias_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for symbol, provider in [("AAPL", "AAPL"), ("SPY", "SPY"), ("BRK-B", "BRK.B"), ("ABCB", "ABCB")]:
            writer.writerow({"asset_id": f"asset_{symbol}", "provider": "alpaca", "provider_symbol": provider, "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "configured_provider_map" if symbol == "BRK-B" else "identity", "source": "test", "registry_version": "test"})
    return assets, aliases


def _daily_collect_config(output_root, raw_root, archive_root, assets, aliases, start, end, *, allow_bounded):
    return {
        "ml": {
            "historical_bar_backfill": {
                "provider": "alpaca",
                "feed": "sip",
                "symbols": ["AAPL", "SPY", "BRK-B", "ABCB"],
                "provider_symbol_map": {"BRK-B": "BRK.B"},
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbol_batch_size": 4,
                "date_window_days": 7,
                "output_root": str(output_root),
                "raw_root": str(raw_root),
                "daily_archive_root": str(archive_root),
                "asset_registry": str(assets),
                "alias_registry": str(aliases),
                "dataset_version": "test_daily_smoke",
                "dry_run": False,
                "resume": True,
                "write_raw": True,
                "write_normalized_staging": True,
                "max_collect_chunks": 1,
                "allow_bounded_consolidation": allow_bounded,
                "progress_report_interval_seconds": 0,
            }
        }
    }
