from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import pytest

from core.research.ml.dataset_build_manifest import (
    build_dataset_build_manifest,
    dataset_manifest_path,
    write_manifest,
)
from core.research.ml.reference.market_information_availability_authority import (
    AVAILABLE,
    OUTSIDE_SESSION_POLICY,
    MarketInformationAvailabilityAuthority,
    sec_filing_event_from_row,
)
from core.research.ml.research_certification import build_research_certification_envelope
from infrastructure.data import calendar_authority as ca
from infrastructure.data.calendar_authority import (
    CLOSED_HOLIDAY_STATUS,
    CLOSED_WEEKEND_STATUS,
    CONFLICTING_CALENDAR_EVIDENCE_STATUS,
    EARLY_CLOSE_STATUS,
    FALLBACK_CALENDAR_USED_STATUS,
    OUTSIDE_AUTHORITY_RANGE_STATUS,
    REGULAR_SESSION_STATUS,
    SPECIAL_CLOSURE_STATUS,
    CompactNyseFallbackBackend,
    ExchangeCalendarAuthority,
    calendar_authority_identity,
    materialize_calendar_schedule,
)
from infrastructure.data.market_sessions import (
    AFTER_HOURS_SESSION,
    PRE_MARKET_SESSION,
    RTH_SESSION,
    exchange_session_context,
    next_trading_session,
    previous_trading_session,
    session_type,
)


def test_regular_early_close_holiday_weekend_and_special_closure() -> None:
    authority = ExchangeCalendarAuthority(backend=CompactNyseFallbackBackend())

    regular = authority.session(date(2026, 1, 2))
    early = authority.session(date(2025, 7, 3))
    holiday = authority.session(date(2026, 1, 1))
    weekend = authority.session(date(2026, 1, 3))
    special = authority.session(date(2025, 1, 9))

    assert regular.base_status == REGULAR_SESSION_STATUS
    assert regular.source_status == FALLBACK_CALENDAR_USED_STATUS
    assert regular.fallback_used is True
    assert early.base_status == EARLY_CLOSE_STATUS
    assert early.close_timestamp == datetime(2025, 7, 3, 17, 0, tzinfo=timezone.utc)
    assert holiday.base_status == CLOSED_HOLIDAY_STATUS
    assert weekend.base_status == CLOSED_WEEKEND_STATUS
    assert special.base_status == SPECIAL_CLOSURE_STATUS


def test_pinned_maintained_authority_covers_required_cases() -> None:
    pytest.importorskip("exchange_calendars")
    authority = ExchangeCalendarAuthority(allow_fallback=False)

    cases = {
        "regular": (date(2026, 1, 2), REGULAR_SESSION_STATUS),
        "holiday": (date(2026, 1, 1), CLOSED_HOLIDAY_STATUS),
        "weekend": (date(2026, 1, 3), CLOSED_WEEKEND_STATUS),
        "early_close": (date(2025, 11, 28), EARLY_CLOSE_STATUS),
        "dst_start": (date(2026, 3, 9), REGULAR_SESSION_STATUS),
        "dst_end": (date(2026, 11, 2), REGULAR_SESSION_STATUS),
        "special_closure": (date(2025, 1, 9), SPECIAL_CLOSURE_STATUS),
    }

    assert authority.backend_status == "MAINTAINED"
    for _name, (day, expected_status) in cases.items():
        record = authority.session(day)
        assert record.package == ca.SELECTED_MAINTAINED_PACKAGE
        assert record.package_version == ca.SELECTED_MAINTAINED_PACKAGE_VERSION
        assert record.fallback_used is False
        assert record.fallback_version is None
        assert record.base_status == expected_status

    early = authority.session(date(2025, 11, 28))
    assert early.close_timestamp == datetime(2025, 11, 28, 18, 0, tzinfo=timezone.utc)

    dst_start = authority.session(date(2026, 3, 9))
    dst_end = authority.session(date(2026, 11, 2))
    assert dst_start.open_timestamp == datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
    assert dst_end.open_timestamp == datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)
    assert authority.previous_session(date(2026, 1, 5)) == date(2026, 1, 2)
    assert authority.next_session(date(2026, 1, 2)) == date(2026, 1, 5)


def test_maintained_identity_and_comparison_are_deterministic() -> None:
    pytest.importorskip("exchange_calendars")
    authority = ExchangeCalendarAuthority(allow_fallback=False)

    first = authority.identity(start="2026-01-02", end="2026-01-05")
    second = authority.identity(start="2026-01-02", end="2026-01-05")
    comparison = ca.maintained_fallback_comparison_rows()

    assert first == second
    assert first["package"] == ca.SELECTED_MAINTAINED_PACKAGE
    assert first["package_version"] == ca.SELECTED_MAINTAINED_PACKAGE_VERSION
    assert first["fallback_used"] is False
    assert first["fallback_state"] == "MAINTAINED_USED"
    assert first["schedule_hash"]
    assert comparison
    assert {row["classification"] for row in comparison} <= {
        "match",
        "maintained authority correction",
        "fallback limitation",
        "unresolved conflict",
    }


def test_dst_start_and_end_use_exchange_timezone() -> None:
    authority = ExchangeCalendarAuthority(backend=CompactNyseFallbackBackend())

    dst_start = authority.session(date(2026, 3, 9))
    dst_end = authority.session(date(2026, 11, 2))

    assert dst_start.open_timestamp == datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
    assert dst_end.open_timestamp == datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)
    assert dst_start.market_timezone == "America/New_York"


def test_pre_market_after_hours_previous_next_and_session_type_compatibility() -> None:
    assert session_type(datetime(2026, 1, 2, 13, 0, tzinfo=timezone.utc)) == PRE_MARKET_SESSION
    assert session_type(datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)) == RTH_SESSION
    assert session_type(datetime(2026, 1, 2, 21, 30, tzinfo=timezone.utc)) == AFTER_HOURS_SESSION

    pre = exchange_session_context(datetime(2026, 1, 2, 13, 0, tzinfo=timezone.utc))
    after = exchange_session_context(datetime(2026, 1, 2, 21, 30, tzinfo=timezone.utc))

    assert pre.trading_session == PRE_MARKET_SESSION
    assert after.trading_session == AFTER_HOURS_SESSION
    assert previous_trading_session(date(2026, 1, 5)) == date(2026, 1, 2)
    assert next_trading_session(date(2026, 1, 2)) == date(2026, 1, 5)


def test_outside_authority_range_fails_ticket68_closed() -> None:
    authority = ExchangeCalendarAuthority(backend=CompactNyseFallbackBackend())
    record = authority.session(date(2100, 1, 4))

    assert record.source_status == OUTSIDE_AUTHORITY_RANGE_STATUS
    assert record.is_trading_day is False

    result = MarketInformationAvailabilityAuthority().evaluate(
        sec_filing_event_from_row(
            {"accepted_datetime": "2100-01-04T14:00:00Z", "form_type": "8-K"},
            decision_timestamp="2100-01-04T15:00:00Z",
        )
    )

    assert result["status"] == OUTSIDE_SESSION_POLICY
    assert "calendar_authority_outside_range" in result["reason_codes"]
    assert result["usable_for_promotion"] is False


def test_installed_maintained_dependency_uses_pinned_exchange_calendar() -> None:
    authority = ExchangeCalendarAuthority()
    if authority.backend_status != "MAINTAINED":
        pytest.skip("pinned exchange_calendars package is not installed")

    regular = authority.session(date(2026, 1, 2))
    holiday = authority.session(date(2026, 1, 1))
    weekend = authority.session(date(2026, 1, 3))
    early = authority.session(date(2025, 7, 3))
    dst_start = authority.session(date(2026, 3, 9))
    dst_end = authority.session(date(2026, 11, 2))
    special = authority.session(date(2025, 1, 9))
    outside = authority.session(date(2100, 1, 4))

    assert authority.backend.package_name == ca.SELECTED_MAINTAINED_PACKAGE
    assert authority.backend.package_version == ca.SELECTED_MAINTAINED_PACKAGE_VERSION
    assert regular.base_status == REGULAR_SESSION_STATUS
    assert regular.fallback_used is False
    assert holiday.base_status == CLOSED_HOLIDAY_STATUS
    assert weekend.base_status == CLOSED_WEEKEND_STATUS
    assert early.base_status == EARLY_CLOSE_STATUS
    assert early.close_timestamp == datetime(2025, 7, 3, 17, 0, tzinfo=timezone.utc)
    assert dst_start.open_timestamp == datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
    assert dst_end.open_timestamp == datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)
    assert special.base_status == SPECIAL_CLOSURE_STATUS
    assert outside.source_status == OUTSIDE_AUTHORITY_RANGE_STATUS
    assert outside.fallback_used is False
    assert authority.previous_session(date(2026, 1, 5)) == date(2026, 1, 2)
    assert authority.next_session(date(2026, 1, 2)) == date(2026, 1, 5)

    pre = exchange_session_context(datetime(2026, 1, 2, 13, 0, tzinfo=timezone.utc))
    after = exchange_session_context(datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc))
    assert pre.trading_session == PRE_MARKET_SESSION
    assert after.trading_session == AFTER_HOURS_SESSION
    assert pre.calendar_fallback_used is False
    assert after.calendar_fallback_used is False
    assert regular.authority_version_identity == authority.session(date(2026, 1, 2)).authority_version_identity
    assert regular.schedule_hash == authority.session(date(2026, 1, 2)).schedule_hash


def test_maintained_dependency_unavailable_uses_explicit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(ca, "load_exchange_calendars_backend", lambda **_: None)

    authority = ExchangeCalendarAuthority()
    record = authority.session(date(2026, 1, 2))

    assert authority.backend_status == "DEPENDENCY_UNAVAILABLE"
    assert record.source_status == FALLBACK_CALENDAR_USED_STATUS
    assert record.fallback_used is True
    assert record.fallback_version == ca.FALLBACK_CALENDAR_VERSION


def test_local_override_and_conflicting_evidence(tmp_path: Path) -> None:
    reviewed = _override_file(tmp_path, reviewed=True)
    conflict = _override_file(tmp_path, reviewed=False, name="unreviewed.json")

    overridden = ExchangeCalendarAuthority(
        backend=CompactNyseFallbackBackend(),
        local_override_paths=(reviewed,),
    ).session(date(2026, 1, 2))
    conflicted = ExchangeCalendarAuthority(
        backend=CompactNyseFallbackBackend(),
        local_override_paths=(conflict,),
    ).session(date(2026, 1, 2))

    assert overridden.source_status == SPECIAL_CLOSURE_STATUS
    assert overridden.closure_reason == "reviewed fixture closure"
    assert conflicted.source_status == CONFLICTING_CALENDAR_EVIDENCE_STATUS
    assert conflicted.conflict is not None
    assert conflicted.conflict.promotion_blocking is True


def test_deterministic_schedule_hash_and_materialisation(tmp_path: Path) -> None:
    authority = ExchangeCalendarAuthority(backend=CompactNyseFallbackBackend())

    first = authority.session(date(2026, 1, 2))
    second = authority.session(date(2026, 1, 2))
    csv_path = tmp_path / "schedule.csv"
    rows = materialize_calendar_schedule(
        "2026-01-01",
        "2026-01-05",
        include_closed=True,
        output_path=csv_path,
    )

    assert first.schedule_hash == second.schedule_hash
    assert rows[1]["session_date"] == "2026-01-02"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written[1]["schedule_hash"] == rows[1]["schedule_hash"]


def test_ticket68a_artifact_writer_materialises_required_files(tmp_path: Path) -> None:
    pytest.importorskip("exchange_calendars")
    pq = pytest.importorskip("pyarrow.parquet")

    artifacts = ca.write_calendar_artifacts(tmp_path)
    required = {
        "calendar_source_audit",
        "calendar_authority_contract",
        "calendar_version",
        "calendar_coverage",
        "calendar_conflicts",
        "maintained_fallback_comparison",
        "materialised_schedule",
        "calendar_validation",
    }

    assert required <= set(artifacts)
    assert artifacts["maintained_fallback_comparison"].exists()
    assert artifacts["materialised_schedule"].exists()
    validation = json.loads(
        artifacts["calendar_validation"].read_text(encoding="utf-8")
    )
    table = pq.read_table(artifacts["materialised_schedule"])
    rows = table.to_pylist()

    assert validation["classification"] == "MAINTAINED_CALENDAR_AUTHORITY_IMPLEMENTED"
    assert rows
    assert rows[0]["package"] == ca.SELECTED_MAINTAINED_PACKAGE
    assert rows[0]["package_version"] == ca.SELECTED_MAINTAINED_PACKAGE_VERSION
    assert rows[0]["authority_version"] == ca.CALENDAR_AUTHORITY_VERSION


def test_ticket68_resolution_records_calendar_lineage() -> None:
    result = MarketInformationAvailabilityAuthority().evaluate(
        sec_filing_event_from_row(
            {"accepted_datetime": "2026-01-02T13:00:00Z", "form_type": "8-K"},
            decision_timestamp="2026-01-02T14:35:00Z",
        )
    )

    assert result["status"] == AVAILABLE
    assert result["calendar_authority_version"] == ca.CALENDAR_AUTHORITY_VERSION
    assert result["calendar_schedule_hash"]
    if result["calendar_fallback_used"]:
        assert result["calendar_source_status"] == FALLBACK_CALENDAR_USED_STATUS
    else:
        assert result["calendar_source_status"] == REGULAR_SESSION_STATUS
        assert result["calendar_package"] == ca.SELECTED_MAINTAINED_PACKAGE
        assert result["calendar_package_version"] == ca.SELECTED_MAINTAINED_PACKAGE_VERSION


def test_manifest_and_certification_record_calendar_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    dataset = tmp_path / "dataset.csv"
    _write_csv(source, [{"feature_id": "parent", "value": "1"}])
    rows = [
        {
            "feature_id": "a",
            "symbol": "AAA",
            "decision_timestamp": "2026-01-02T14:35:00Z",
            "label_available_timestamp": "2026-01-05T14:35:00Z",
        }
    ]
    _write_csv(dataset, rows)
    calendar = calendar_authority_identity(start="2026-01-02", end="2026-01-05")
    manifest = build_dataset_build_manifest(
        dataset_id="calendar-lineage-dataset",
        dataset_type="synthetic",
        schema_version="schema-v1",
        producer_command="test",
        producer_module="tests.test_calendar_authority",
        output_paths=(dataset,),
        source_paths=(source,),
        universe_authority_version="pit_universe_authority_v1",
        identity_authority_version="historical_identity_authority_v1",
        corporate_action_authority_version="corporate_action_authority_v1",
        market_calendar_authority_version=str(calendar["version"]),
        market_calendar_authority=calendar,
        target_contract_version="target-v1",
        feature_code_version="feature-v1",
        label_code_version="label-v1",
        configuration_hash_value="config-v1",
        rows=rows,
        key_fields=("feature_id",),
        source_control={"git_commit": "abc", "dirty_worktree": False},
    )
    write_manifest(dataset_manifest_path(dataset), manifest)

    envelope = build_research_certification_envelope(
        source_control={"git_commit": "abc", "dirty_worktree": False},
        dataset_manifest_paths=(dataset_manifest_path(dataset),),
    )

    assert manifest["market_calendar_authority"]["schedule_hash"] == calendar["schedule_hash"]
    assert (
        envelope["authority_versions"]["market_calendar_authority"]["schedule_hash"]
        == calendar["schedule_hash"]
    )


def _override_file(tmp_path: Path, *, reviewed: bool, name: str = "override.json") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "contract_version": "calendar_local_override.v1",
                "calendar_id": "XNYS",
                "reviewed": reviewed,
                "effective_date": "2026-01-01",
                "overrides": [
                    {
                        "session_date": "2026-01-02",
                        "action": "CLOSE",
                        "reason": "reviewed fixture closure",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
