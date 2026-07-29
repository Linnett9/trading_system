from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from core.research.ml.target_authority import (
    HALT_AFFECTED,
    INELIGIBLE_SOURCE_BAR,
    MATURED_VALID,
    MISSING_SOURCE_BAR,
    NOT_YET_MATURE,
    QUARANTINED_SOURCE_BAR,
    RIGHT_CENSORED,
    SESSION_BOUNDARY_CONFLICT,
    UNKNOWN_SOURCE_GAP,
    TargetContractError,
    build_target_manifest,
    calculate_target,
    require_explicit_target_contract,
    resolve_target_contract,
    target_catalogue,
    target_catalogue_payload,
    validate_target_availability,
)
from infrastructure.data.calendar_authority import default_calendar_authority
from infrastructure.data.market_sessions import trading_sessions


ASSET = "asset-AAA"
SYMBOL = "AAA"


def test_daily_ten_session_semantics_cross_holiday_without_legacy_value_change() -> None:
    sessions = trading_sessions(date(2024, 1, 12), date(2024, 2, 5))
    rows = _daily_bars(sessions, start=50.0)

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-12",
        bar_source=rows,
        target_contract="forward_return_10d",
        source_cutoff=_session_close(sessions[-1]),
    )

    assert result.target_id == "forward_return_10_sessions__decision_1Day"
    assert result.horizon_unit == "eligible_trading_sessions"
    assert result.horizon_value == 10
    assert date(2024, 1, 15) not in sessions
    assert result.target_end_timestamp == _session_close(date(2024, 1, 29))
    assert result.value == pytest.approx((60.0 / 50.0) - 1.0)
    assert result.target_is_trainable is True


def test_daily_holiday_crossing_uses_ordered_sessions_not_calendar_days() -> None:
    sessions = trading_sessions(date(2024, 1, 12), date(2024, 1, 31))
    rows = _daily_bars(sessions, start=100.0)

    result = calculate_target(
        asset_id=SYMBOL,
        decision_timestamp="2024-01-12",
        bar_source=rows,
        target_contract="forward_return_5_sessions__decision_1Day",
        source_cutoff=_session_close(sessions[-1]),
    )

    assert result.target_end_timestamp == _session_close(date(2024, 1, 22))
    assert result.value == pytest.approx((105.0 / 100.0) - 1.0)


def test_daily_early_close_uses_calendar_close_timestamp() -> None:
    authority = default_calendar_authority()
    early = date(2024, 7, 3)
    sessions = trading_sessions(early, date(2024, 7, 10))
    rows = _daily_bars(sessions, start=10.0)

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp=early,
        bar_source=rows,
        target_contract="forward_return_1_session__decision_1Day",
        source_cutoff=_session_close(sessions[-1]),
    )

    assert authority.session(early).early_close is True
    assert result.target_start_timestamp == "2024-07-03T17:00:00Z"
    assert result.target_end_timestamp == _session_close(date(2024, 7, 5))
    assert result.value == pytest.approx((11.0 / 10.0) - 1.0)


def test_hourly_sixty_minute_target() -> None:
    rows = _hourly_bars(date(2024, 1, 2))

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T15:30:00Z",
        bar_source=rows,
        target_contract="forward_return_60m__decision_1h",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == MATURED_VALID
    assert result.target_end_timestamp == "2024-01-02T16:30:00Z"
    assert result.value == pytest.approx((102.0 / 101.0) - 1.0)


def test_hourly_target_near_close_does_not_roll_to_next_session() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T20:30:00Z",
        bar_source=_hourly_bars(date(2024, 1, 2)),
        target_contract="forward_return_60m__decision_1h",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == SESSION_BOUNDARY_CONFLICT
    assert result.target_resolution_reason == "elapsed_market_minutes_crosses_session_close"


def test_five_minute_thirty_minute_target() -> None:
    rows = _five_minute_bars(date(2024, 1, 2))

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_end_timestamp == "2024-01-02T15:05:00Z"
    assert result.value == pytest.approx((106.5 / 100.5) - 1.0)


def test_five_minute_sixty_minute_target() -> None:
    rows = _five_minute_bars(date(2024, 1, 2))

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_60m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_end_timestamp == "2024-01-02T15:35:00Z"
    assert result.value == pytest.approx((112.5 / 100.5) - 1.0)


def test_five_minute_target_near_close_conflicts_instead_of_skipping_bars() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T20:35:00Z",
        bar_source=_five_minute_bars(date(2024, 1, 2)),
        target_contract="forward_return_60m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == SESSION_BOUNDARY_CONFLICT


def test_return_to_close_uses_same_session_close() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T19:35:00Z",
        bar_source=_five_minute_bars(date(2024, 1, 2)),
        target_contract="forward_return_to_close__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_end_timestamp == "2024-01-02T21:00:00Z"
    assert result.target_available_timestamp == "2024-01-02T21:00:00Z"
    assert result.value == pytest.approx((177.5 / 160.5) - 1.0)


def test_next_session_open_uses_first_next_session_bar_open() -> None:
    rows = [
        *_five_minute_bars(date(2024, 1, 2)),
        *_five_minute_bars(date(2024, 1, 3), start=300.0),
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T20:55:00Z",
        bar_source=rows,
        target_contract="forward_return_next_open__decision_5m",
        source_cutoff="2024-01-03T14:35:00Z",
    )

    assert result.target_end_timestamp == "2024-01-03T14:30:00Z"
    assert result.target_available_timestamp == "2024-01-03T14:35:00Z"
    assert result.value == pytest.approx((300.0 / 176.5) - 1.0)


def test_weekend_rollover_for_next_open() -> None:
    rows = [
        *_five_minute_bars(date(2024, 1, 5)),
        *_five_minute_bars(date(2024, 1, 8), start=250.0),
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-05T20:55:00Z",
        bar_source=rows,
        target_contract="forward_return_next_open__decision_5m",
        source_cutoff="2024-01-08T14:35:00Z",
    )

    assert result.target_end_timestamp == "2024-01-08T14:30:00Z"
    assert result.value == pytest.approx((250.0 / 176.5) - 1.0)


def test_holiday_rollover_for_next_open() -> None:
    rows = [
        *_five_minute_bars(date(2024, 1, 12)),
        *_five_minute_bars(date(2024, 1, 16), start=260.0),
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-12T20:55:00Z",
        bar_source=rows,
        target_contract="forward_return_next_open__decision_5m",
        source_cutoff="2024-01-16T14:35:00Z",
    )

    assert result.target_end_timestamp == "2024-01-16T14:30:00Z"
    assert result.value == pytest.approx((260.0 / 176.5) - 1.0)


def test_dst_boundary_for_next_open() -> None:
    rows = [
        *_five_minute_bars(date(2024, 3, 8)),
        *_five_minute_bars(date(2024, 3, 11), start=270.0),
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-03-08T20:55:00Z",
        bar_source=rows,
        target_contract="forward_return_next_open__decision_5m",
        source_cutoff="2024-03-11T13:35:00Z",
    )

    assert result.target_end_timestamp == "2024-03-11T13:30:00Z"
    assert result.target_available_timestamp == "2024-03-11T13:35:00Z"


def test_early_close_to_close_uses_early_close_boundary() -> None:
    rows = _five_minute_bars(date(2024, 7, 3), start=100.0)

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-07-03T16:35:00Z",
        bar_source=rows,
        target_contract="forward_return_to_close__decision_5m",
        source_cutoff="2024-07-03T17:00:00Z",
    )

    assert result.target_end_timestamp == "2024-07-03T17:00:00Z"
    assert result.target_resolution_classification == MATURED_VALID


def test_missing_bar_is_classified_without_silent_skip() -> None:
    rows = [
        row for row in _five_minute_bars(date(2024, 1, 2))
        if row["bar_end_timestamp"] != "2024-01-02T15:05:00Z"
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == MISSING_SOURCE_BAR
    assert result.target_exit_bar_id == ""


def test_five_minute_missing_intermediate_bar_is_not_silently_bridged() -> None:
    rows = [
        row for row in _five_minute_bars(date(2024, 1, 2))
        if row["bar_end_timestamp"] != "2024-01-02T14:50:00Z"
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == MISSING_SOURCE_BAR
    assert result.target_exit_bar_id != ""
    assert result.target_resolution_reason == "required_source_bar_missing:2024-01-02T14:50:00Z"


def test_daily_missing_intermediate_session_is_not_silently_bridged() -> None:
    sessions = trading_sessions(date(2024, 1, 12), date(2024, 1, 31))
    rows = [
        row for row in _daily_bars(sessions, start=100.0)
        if row["session_date"] != "2024-01-17"
    ]

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-12",
        bar_source=rows,
        target_contract="forward_return_5_sessions__decision_1Day",
        source_cutoff=_session_close(sessions[-1]),
    )

    assert result.target_resolution_classification == MISSING_SOURCE_BAR
    assert result.target_exit_bar_id != ""
    assert result.target_resolution_reason == "required_session_source_bar_missing:2024-01-17"


def test_quarantined_bar_is_separate_from_missing() -> None:
    rows = _replace_bar_status(
        _five_minute_bars(date(2024, 1, 2)),
        "2024-01-02T15:05:00Z",
        "QUARANTINED_SOURCE_BAR",
    )

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == QUARANTINED_SOURCE_BAR


def test_ineligible_bar_is_separate_from_missing() -> None:
    rows = _replace_bar_status(
        _five_minute_bars(date(2024, 1, 2)),
        "2024-01-02T15:05:00Z",
        "INELIGIBLE_SOURCE_BAR",
    )

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == INELIGIBLE_SOURCE_BAR


def test_represented_halt_blocks_target() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=_five_minute_bars(date(2024, 1, 2)),
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
        market_halts=(
            {
                "start_timestamp": "2024-01-02T14:45:00Z",
                "end_timestamp": "2024-01-02T14:55:00Z",
                "reason": "fixture_halt",
            },
        ),
    )

    assert result.target_resolution_classification == HALT_AFFECTED
    assert result.target_resolution_reason == "fixture_halt"


def test_right_censoring_does_not_use_future_bars() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=_five_minute_bars(date(2024, 1, 2)),
        target_contract="forward_return_60m__decision_5m",
        source_cutoff="2024-01-02T15:00:00Z",
    )

    assert result.target_resolution_classification == RIGHT_CENSORED
    assert result.target_exit_bar_id == ""
    assert result.no_future_data is True


def test_not_yet_mature_between_daily_end_and_daily_availability() -> None:
    sessions = trading_sessions(date(2024, 1, 2), date(2024, 1, 10))
    rows = _daily_bars(sessions, start=10.0)

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02",
        bar_source=rows,
        target_contract="forward_return_1_session__decision_1Day",
        source_cutoff=_session_close(date(2024, 1, 3)),
    )

    assert result.target_resolution_classification == NOT_YET_MATURE
    assert result.target_available_timestamp == _session_close(date(2024, 1, 4))
    assert result.value is None


def test_unknown_source_gap_state_is_not_compressed_to_missing() -> None:
    rows = _replace_bar_status(
        _five_minute_bars(date(2024, 1, 2)),
        "2024-01-02T15:05:00Z",
        "PROVIDER_GAP",
    )

    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=rows,
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert result.target_resolution_classification == UNKNOWN_SOURCE_GAP


def test_label_availability_and_trainability_cutoff_are_explicit() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T20:55:00Z",
        bar_source=[
            *_five_minute_bars(date(2024, 1, 2)),
            *_five_minute_bars(date(2024, 1, 3), start=300.0),
        ],
        target_contract="forward_return_next_open__decision_5m",
        source_cutoff="2024-01-03T14:35:00Z",
        training_cutoff="2024-01-03T14:30:00Z",
    )
    validation = validate_target_availability([result])

    assert result.target_resolution_classification == MATURED_VALID
    assert result.target_is_mature is True
    assert result.target_is_realised is True
    assert result.target_is_trainable is False
    assert validation["status"] == "PASSED"


def test_legacy_alias_mapping_and_strict_research_mode() -> None:
    resolved = resolve_target_contract("forward_return_10d")

    assert resolved.target_id == "forward_return_10_sessions__decision_1Day"
    with pytest.raises(TargetContractError, match="canonical target_id"):
        require_explicit_target_contract("forward_return_10d")


def test_manifest_determinism_and_no_model_work_flags() -> None:
    result = calculate_target(
        asset_id=ASSET,
        decision_timestamp="2024-01-02T14:35:00Z",
        bar_source=_five_minute_bars(date(2024, 1, 2)),
        target_contract="forward_return_30m__decision_5m",
        source_cutoff="2024-01-02T21:00:00Z",
    )
    selected = resolve_target_contract("forward_return_30m__decision_5m")
    first = build_target_manifest(
        [result],
        selected_target=selected,
        source_cutoff="2024-01-02T21:00:00Z",
        configuration={"fixture": "deterministic"},
        calendar_identity=result.calendar_identity,
    )
    second = build_target_manifest(
        [result],
        selected_target=selected,
        source_cutoff="2024-01-02T21:00:00Z",
        configuration={"fixture": "deterministic"},
        calendar_identity=result.calendar_identity,
    )

    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["row_counts"]["trainable"] == 1
    assert first["model_training_performed"] is False
    assert first["model_or_policy_promoted"] is False


def test_catalogue_separates_decision_timeframe_source_timeframe_and_horizon() -> None:
    payload = target_catalogue_payload()
    by_id = {contract.target_id: contract for contract in target_catalogue()}

    assert payload["contract_version"] == "multi_timeframe_target_contract.v1"
    assert by_id["forward_return_10_sessions__decision_1Day"].horizon_unit == "eligible_trading_sessions"
    assert by_id["forward_return_60m__decision_5m"].horizon_unit == "elapsed_market_minutes"
    assert by_id["forward_return_60m__decision_5m"].decision_timeframe == "5m"
    assert by_id["forward_return_60m__decision_5m"].source_bar_timeframe == "5m"
    assert "forward_return_10d" in by_id["forward_return_10_sessions__decision_1Day"].legacy_aliases


def _daily_bars(sessions: list[date], *, start: float) -> list[dict[str, object]]:
    return [
        {
            "asset_id": ASSET,
            "canonical_symbol": SYMBOL,
            "session_date": session.isoformat(),
            "timeframe": "1Day",
            "open": start + index - 0.25,
            "close": start + index,
            "source_bar_id": f"{SYMBOL}-1d-{session.isoformat()}",
        }
        for index, session in enumerate(sessions)
    ]


def _five_minute_bars(day: date, *, start: float = 100.0) -> list[dict[str, object]]:
    return _intraday_bars(day, minutes=5, timeframe="5m", start=start)


def _hourly_bars(day: date, *, start: float = 100.0) -> list[dict[str, object]]:
    open_ts = datetime.fromisoformat(_session_open(day).replace("Z", "+00:00"))
    close_ts = datetime.fromisoformat(_session_close(day).replace("Z", "+00:00"))
    starts = [open_ts + timedelta(hours=index) for index in range(6)]
    starts.append(close_ts - timedelta(hours=1))
    rows = []
    for index, timestamp in enumerate(starts):
        rows.append(_intraday_row(timestamp, minutes=60, timeframe="1h", open_=start + index, close=start + index + 1))
    return rows


def _intraday_bars(day: date, *, minutes: int, timeframe: str, start: float) -> list[dict[str, object]]:
    open_ts = datetime.fromisoformat(_session_open(day).replace("Z", "+00:00"))
    close_ts = datetime.fromisoformat(_session_close(day).replace("Z", "+00:00"))
    rows = []
    current = open_ts
    index = 0
    while current < close_ts:
        rows.append(
            _intraday_row(
                current,
                minutes=minutes,
                timeframe=timeframe,
                open_=start + index,
                close=start + index + 0.5,
            )
        )
        current += timedelta(minutes=minutes)
        index += 1
    return rows


def _intraday_row(
    timestamp: datetime,
    *,
    minutes: int,
    timeframe: str,
    open_: float,
    close: float,
) -> dict[str, object]:
    end = timestamp + timedelta(minutes=minutes)
    stamp = _format(timestamp)
    return {
        "asset_id": ASSET,
        "canonical_symbol": SYMBOL,
        "timestamp_utc": stamp,
        "bar_end_timestamp": _format(end),
        "session_date": timestamp.astimezone(timezone.utc).date().isoformat(),
        "timeframe": timeframe,
        "open": open_,
        "close": close,
        "bar_status": "OK",
        "source_bar_id": f"{SYMBOL}-{timeframe}-{stamp}",
    }


def _replace_bar_status(
    rows: list[dict[str, object]],
    bar_end_timestamp: str,
    status: str,
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if row["bar_end_timestamp"] == bar_end_timestamp:
            output.append({**row, "bar_status": status})
        else:
            output.append(row)
    return output


def _session_open(day: date) -> str:
    record = default_calendar_authority().session(day)
    assert record.open_timestamp is not None
    return _format(record.open_timestamp)


def _session_close(day: date) -> str:
    record = default_calendar_authority().session(day)
    assert record.close_timestamp is not None
    return _format(record.close_timestamp)


def _format(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
