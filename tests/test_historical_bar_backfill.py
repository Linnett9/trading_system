from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from application.cli_runtime import FEEDLESS_MODES
from application.services.historical_bar_backfill_commands import _plan, run_historical_bar_backfill_collect
from infrastructure.data.historical_bar_overlap import audit_historical_bar_overlap
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    BackfillChunkStateStore,
    HistoricalBarMetrics,
    HistoricalBarRequest,
    SharedRateLimiter,
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

    assert sleeps == [60.0]
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
