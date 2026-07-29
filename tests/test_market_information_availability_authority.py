from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from infrastructure.data import calendar_authority as ca
from core.research.ml.reference.market_information_availability_authority import (
    AFTER_HOURS_SESSION,
    AVAILABLE,
    CONFLICTING_EVIDENCE,
    NOT_YET_AVAILABLE,
    OUTSIDE_SESSION_POLICY,
    REGULAR_SESSION,
    REVISED_AFTER_DECISION,
    UNKNOWN_AVAILABILITY,
    MarketInformationAvailabilityAuthority,
    MarketInformationEvent,
    sec_filing_event_from_row,
)
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    STRICT_COLLECTED_AT,
    StockAlphaNewsPitPolicy,
    article_is_pit_eligible,
)
from core.research.ml.stock_level.stock_fundamentals import (
    build_fundamental_snapshots,
)
from core.research.ml.stock_level.stock_level_alpha_features import (
    ENGINEERED_FEATURE_COLUMNS,
    build_stock_level_alpha_features,
)


def test_pre_market_sec_filing_is_available_for_regular_session_decision() -> None:
    result = _authority().evaluate(
        sec_filing_event_from_row(
            {
                "accepted_datetime": "2026-01-02T13:00:00Z",
                "form_type": "8-K",
                "accession_number": "000-test-pre",
            },
            decision_timestamp="2026-01-02T14:35:00Z",
        )
    )

    assert result["status"] == AVAILABLE
    assert result["trading_session"] == REGULAR_SESSION
    assert result["earliest_permitted_use"] == "2026-01-02T13:00:00Z"


def test_after_close_sec_filing_is_not_available_for_same_day_rth_decision() -> None:
    result = _authority().evaluate(
        sec_filing_event_from_row(
            {
                "accepted_datetime": "2026-01-02T21:30:00Z",
                "form_type": "8-K",
                "accession_number": "000-test-after",
            },
            decision_timestamp="2026-01-02T20:55:00Z",
        )
    )

    assert result["status"] == NOT_YET_AVAILABLE
    assert result["earliest_permitted_use"] == "2026-01-02T21:30:00Z"


def test_holiday_decision_fails_session_policy() -> None:
    result = _authority().evaluate(
        MarketInformationEvent(
            source_kind="news",
            decision_timestamp="2026-01-01T15:00:00Z",
            provider_published_timestamp="2026-01-01T14:00:00Z",
            first_seen_timestamp="2026-01-01T14:01:00Z",
            required_knowledge_fields=("provider_published_timestamp", "first_seen_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "first_seen_timestamp"),
        )
    )

    assert result["status"] == OUTSIDE_SESSION_POLICY
    assert "decision_timestamp_not_trading_day" in result["reason_codes"]


def test_early_close_maps_decision_to_after_hours_session() -> None:
    result = _authority().evaluate(
        sec_filing_event_from_row(
            {"accepted_datetime": "2025-07-03T16:00:00Z", "form_type": "8-K"},
            decision_timestamp="2025-07-03T17:30:00Z",
        )
    )

    assert result["status"] == AVAILABLE
    assert result["trading_session"] == AFTER_HOURS_SESSION
    assert result["session_date"] == "2025-07-03"


def test_dst_transition_uses_exchange_timezone_for_regular_session() -> None:
    result = _authority().evaluate(
        sec_filing_event_from_row(
            {"accepted_datetime": "2026-03-09T13:00:00Z", "form_type": "8-K"},
            decision_timestamp="2026-03-09T13:31:00Z",
        )
    )

    assert result["status"] == AVAILABLE
    assert result["trading_session"] == REGULAR_SESSION
    assert result["timezone"] == "America/New_York"


def test_provider_delay_controls_earliest_permitted_use() -> None:
    result = _authority().evaluate(
        MarketInformationEvent(
            source_kind="news",
            decision_timestamp="2026-01-02T15:00:00Z",
            provider_published_timestamp="2026-01-02T14:00:00Z",
            provider_received_timestamp="2026-01-02T15:05:00Z",
            required_knowledge_fields=("provider_published_timestamp", "provider_received_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "provider_received_timestamp"),
        )
    )

    assert result["status"] == NOT_YET_AVAILABLE
    assert result["earliest_permitted_use"] == "2026-01-02T15:05:00Z"


def test_retrospective_correction_after_decision_fails_closed() -> None:
    result = _authority().evaluate(
        MarketInformationEvent(
            source_kind="fundamental_feature_source",
            decision_timestamp="2026-01-02T15:00:00Z",
            provider_published_timestamp="2026-01-02T14:00:00Z",
            revision_timestamp="2026-01-02T16:00:00Z",
            correction_lineage=[{"kind": "amendment", "source_document_id": "amendment"}],
            required_knowledge_fields=("provider_published_timestamp",),
            availability_basis_fields=("provider_published_timestamp", "revision_timestamp"),
        )
    )

    assert result["status"] == REVISED_AFTER_DECISION
    assert result["usable_for_promotion"] is False


def test_missing_first_seen_time_is_unknown_and_not_promotion_usable() -> None:
    result = _authority().evaluate(
        MarketInformationEvent(
            source_kind="news",
            decision_timestamp="2026-01-02T15:00:00Z",
            provider_published_timestamp="2026-01-02T14:00:00Z",
            required_knowledge_fields=("provider_published_timestamp", "first_seen_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "first_seen_timestamp"),
        )
    )

    assert result["status"] == UNKNOWN_AVAILABILITY
    assert result["usable_for_promotion"] is False


def test_conflicting_timestamps_fail_closed() -> None:
    result = _authority().evaluate(
        MarketInformationEvent(
            source_kind="news",
            decision_timestamp="2026-01-02T15:00:00Z",
            provider_published_timestamp="2026-01-02T14:00:00Z",
            first_seen_timestamp="2026-01-02T13:59:00Z",
            required_knowledge_fields=("provider_published_timestamp", "first_seen_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "first_seen_timestamp"),
        )
    )

    assert result["status"] == CONFLICTING_EVIDENCE
    assert result["evidence_conflicts"] == [
        "first_seen_timestamp_before_provider_published_timestamp"
    ]


def test_authority_result_is_deterministic() -> None:
    event = sec_filing_event_from_row(
        {"accepted_datetime": "2026-01-02T13:00:00Z", "form_type": "8-K"},
        decision_timestamp="2026-01-02T14:35:00Z",
    )

    first = _authority().evaluate(event)
    second = _authority().evaluate(event)

    assert first == second
    assert first["resolution_id"].startswith("mia_")


def test_news_pit_policy_uses_publication_and_first_seen() -> None:
    policy = StockAlphaNewsPitPolicy(
        mode=STRICT_COLLECTED_AT,
        availability_lag_hours=0.0,
        historical_provider_availability_assumed=False,
    )
    decision = _dt("2026-01-02T15:00:00Z")

    assert article_is_pit_eligible(
        {
            "article_id": "news-ok",
            "published_at_utc": _dt("2026-01-02T14:00:00Z"),
            "collected_at_utc": _dt("2026-01-02T14:01:00Z"),
        },
        decision,
        policy,
    )
    assert not article_is_pit_eligible(
        {
            "article_id": "news-missing-first-seen",
            "published_at_utc": _dt("2026-01-02T14:00:00Z"),
        },
        decision,
        policy,
    )


def test_fundamental_snapshots_use_authority_and_exclude_future_filing() -> None:
    mapping = [
        {
            "symbol": "AAPL",
            "reporting_entity_id": "CIK0000320193",
            "security_mapping_identity": "map-aapl",
        }
    ]
    facts = [
        _fact("revenue", 100.0, "2026-01-02T21:30:00Z"),
        _fact("gross_profit", 50.0, "2026-01-02T21:30:00Z"),
    ]
    base_rows = [
        {"symbol": "AAPL", "decision_timestamp": "2026-01-02T20:55:00Z"},
        {"symbol": "AAPL", "decision_timestamp": "2026-01-05T14:35:00Z"},
    ]

    snapshots, audit = build_fundamental_snapshots(
        base_rows,
        mapping,
        facts,
        maximum_data_age_days=None,
        minimum_denominator=1e-9,
    )

    assert snapshots[0]["snapshot_status"] == "no_prior_filing"
    assert snapshots[1]["snapshot_status"] == "available"
    assert snapshots[1]["gross_margin"] == 0.5
    assert audit["availability_authority_status_counts"][AVAILABLE] == 2
    assert audit["availability_authority_status_counts"][NOT_YET_AVAILABLE] == 2


def test_daily_price_features_exclude_future_history_with_authority_evidence() -> None:
    rows = [{"rebalance_date": "2025-03-01", "symbol": "AAA", "sector": "Tech", "industry": "Software", "predicted_momentum_120d": "0.1"}]
    histories = {"AAA": _history(), "SPY": _history()}
    first, first_audit = build_stock_level_alpha_features(rows, histories)
    changed = {symbol: [dict(row) for row in history] for symbol, history in histories.items()}
    for history in changed.values():
        history.append({"date": "2025-12-31", "close": 1000000.0, "high": 1000001.0, "low": 999999.0})

    second, second_audit = build_stock_level_alpha_features(rows, changed)

    assert [{feature: row[feature] for feature in ENGINEERED_FEATURE_COLUMNS} for row in first] == [
        {feature: row[feature] for feature in ENGINEERED_FEATURE_COLUMNS}
        for row in second
    ]
    assert first[0]["daily_price_availability_status"] == AVAILABLE
    assert second[0]["daily_price_availability_status"] == AVAILABLE
    assert first_audit["daily_price_feature_availability_authority"]["future_feature_inclusion_count"] == 0
    assert second_audit["daily_price_feature_availability_authority"]["future_feature_inclusion_count"] == 0


def test_represented_market_halt_fails_session_policy() -> None:
    authority = MarketInformationAvailabilityAuthority(
        market_halts=[
            {
                "start": "2026-01-02T15:00:00Z",
                "end": "2026-01-02T15:15:00Z",
                "reason": "fixture_halt",
            }
        ]
    )

    result = authority.evaluate(
        sec_filing_event_from_row(
            {"accepted_datetime": "2026-01-02T14:00:00Z", "form_type": "8-K"},
            decision_timestamp="2026-01-02T15:05:00Z",
        )
    )

    assert result["status"] == OUTSIDE_SESSION_POLICY
    assert "decision_timestamp_in_represented_market_halt" in result["reason_codes"]


def test_ticket68_maintained_calendar_identity_is_recorded_for_required_cases() -> None:
    pytest.importorskip("exchange_calendars")
    authority = _authority()
    cases = [
        _availability_event(
            accepted="2026-01-02T13:00:00Z",
            decision="2026-01-02T14:35:00Z",
        ),
        _availability_event(
            accepted="2026-01-02T14:35:00Z",
            decision="2026-01-02T14:40:00Z",
        ),
        _availability_event(
            accepted="2026-01-02T21:30:00Z",
            decision="2026-01-02T20:55:00Z",
        ),
        _availability_event(
            accepted="2025-07-03T16:00:00Z",
            decision="2025-07-03T17:30:00Z",
        ),
        _availability_event(
            accepted="2026-01-01T14:00:00Z",
            decision="2026-01-01T15:00:00Z",
        ),
        _availability_event(
            accepted="2026-01-03T14:00:00Z",
            decision="2026-01-03T15:00:00Z",
        ),
        _availability_event(
            accepted="2026-03-09T13:00:00Z",
            decision="2026-03-09T13:31:00Z",
        ),
        _availability_event(
            accepted="2100-01-04T14:00:00Z",
            decision="2100-01-04T15:00:00Z",
        ),
    ]
    halt_result = MarketInformationAvailabilityAuthority(
        market_halts=[
            {
                "start": "2026-01-02T15:00:00Z",
                "end": "2026-01-02T15:15:00Z",
                "reason": "fixture_halt",
            }
        ]
    ).evaluate(
        _availability_event(
            accepted="2026-01-02T14:00:00Z",
            decision="2026-01-02T15:05:00Z",
        )
    )
    results = [authority.evaluate(event) for event in cases] + [halt_result]

    for result in results:
        assert result["exchange"] == "XNYS"
        assert result["calendar_authority_version"] == ca.CALENDAR_AUTHORITY_VERSION
        assert result["calendar_authority_version_identity"]
        assert result["calendar_package"] == ca.SELECTED_MAINTAINED_PACKAGE
        assert result["calendar_package_version"] == ca.SELECTED_MAINTAINED_PACKAGE_VERSION
        assert result["calendar_schedule_hash"]
        assert result["calendar_fallback_used"] is False
        assert result["calendar_source_status"]


def _authority() -> MarketInformationAvailabilityAuthority:
    return MarketInformationAvailabilityAuthority()


def _availability_event(*, accepted: str, decision: str):
    return sec_filing_event_from_row(
        {"accepted_datetime": accepted, "form_type": "8-K"},
        decision_timestamp=decision,
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fact(canonical_fact_id: str, value: float, available: str) -> dict:
    return {
        "provider_id": "official_sec_companyfacts",
        "reporting_entity_id": "CIK0000320193",
        "source_document_id": "filing-q1",
        "filing_accession": "filing-q1",
        "form_type": "10-Q",
        "filing_timestamp": available,
        "acceptance_timestamp": available,
        "available_timestamp": available,
        "period_start": "2025-10-01",
        "period_end": "2025-12-31",
        "canonical_fact_id": canonical_fact_id,
        "normalized_unit": "USD",
        "value": value,
        "fact_period_type": "quarterly_duration",
        "is_amendment": False,
    }


def _history() -> list[dict[str, float | str]]:
    start = date(2024, 1, 1)
    output = []
    for index in range(420):
        value = 100.0 * (1.0 + 0.001 * index + 0.02 * math.sin(index / 9.0))
        output.append(
            {
                "date": (start + timedelta(days=index)).isoformat(),
                "close": value,
                "high": value * 1.01,
                "low": value * 0.99,
            }
        )
    return output
