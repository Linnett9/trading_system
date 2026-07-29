from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from infrastructure.data.calendar_authority import (
    CALENDAR_AUTHORITY_VERSION,
    ExchangeCalendarAuthority,
    default_calendar_authority,
)
from infrastructure.data.market_sessions import (
    EASTERN,
    HALTED_SESSION,
    REGULAR_SESSION,
    exchange_session_context,
)


MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION = "multi_timeframe_target_contract.v1"
TARGET_CATALOGUE_VERSION = "multi_timeframe_target_catalogue.v1"
TARGET_RESOLUTION_POLICY_VERSION = "multi_timeframe_target_resolution_policy.v1"
PRICE_AUTHORITY_VERSION = "source_bar_ohlcv_price_authority.v1"

HORIZON_ELIGIBLE_TRADING_SESSIONS = "eligible_trading_sessions"
HORIZON_ELAPSED_MARKET_MINUTES = "elapsed_market_minutes"
HORIZON_ELIGIBLE_BARS = "eligible_bars"
HORIZON_TO_SESSION_CLOSE = "to_session_close"
HORIZON_TO_NEXT_SESSION_OPEN = "to_next_session_open"

SUPPORTED_HORIZON_UNITS = {
    HORIZON_ELIGIBLE_TRADING_SESSIONS,
    HORIZON_ELAPSED_MARKET_MINUTES,
    HORIZON_ELIGIBLE_BARS,
    HORIZON_TO_SESSION_CLOSE,
    HORIZON_TO_NEXT_SESSION_OPEN,
}

MATURED_VALID = "MATURED_VALID"
NOT_YET_MATURE = "NOT_YET_MATURE"
RIGHT_CENSORED = "RIGHT_CENSORED"
MISSING_SOURCE_BAR = "MISSING_SOURCE_BAR"
QUARANTINED_SOURCE_BAR = "QUARANTINED_SOURCE_BAR"
INELIGIBLE_SOURCE_BAR = "INELIGIBLE_SOURCE_BAR"
SESSION_BOUNDARY_CONFLICT = "SESSION_BOUNDARY_CONFLICT"
HALT_AFFECTED = "HALT_AFFECTED"
UNKNOWN_SOURCE_GAP = "UNKNOWN_SOURCE_GAP"

TARGET_RESOLUTION_STATES = {
    MATURED_VALID,
    NOT_YET_MATURE,
    RIGHT_CENSORED,
    MISSING_SOURCE_BAR,
    QUARANTINED_SOURCE_BAR,
    INELIGIBLE_SOURCE_BAR,
    SESSION_BOUNDARY_CONFLICT,
    HALT_AFFECTED,
    UNKNOWN_SOURCE_GAP,
}

_TIMEFRAME_MINUTES = {
    "5m": 5,
    "1h": 60,
}


class TargetContractError(ValueError):
    """Raised when a target contract cannot be resolved or applied."""


@dataclass(frozen=True)
class TargetContract:
    target_id: str
    legacy_aliases: tuple[str, ...]
    decision_timeframe: str
    source_bar_timeframe: str
    decision_schedule: str
    horizon_value: int | None
    horizon_unit: str
    session_boundary_rule: str
    entry_price_rule: str
    exit_price_rule: str
    target_start_rule: str
    target_end_rule: str
    availability_rule: str
    missing_bar_policy: str
    partial_session_policy: str
    early_close_policy: str
    overnight_policy: str
    calendar_authority_version: str
    price_authority_version: str
    target_code_hash: str
    resolution_policy_version: str
    pre_market_decisions_allowed: bool = False
    after_hours_decisions_allowed: bool = False
    regular_session_only: bool = True
    final_permissible_decision_rule: str = ""
    target_maturity_rule: str = "target_available_timestamp <= source_cutoff"

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_version"] = MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION
        payload["legacy_aliases"] = list(self.legacy_aliases)
        return payload


@dataclass(frozen=True)
class TargetComputationResult:
    target_id: str
    asset_id: str
    value: float | None
    decision_timestamp: str
    target_start_timestamp: str
    label_start_timestamp: str
    target_end_timestamp: str
    target_available_timestamp: str
    training_cutoff: str
    target_source_cutoff: str
    target_is_mature: bool
    target_is_realised: bool
    target_is_trainable: bool
    target_resolution_classification: str
    target_resolution_reason: str
    target_entry_bar_id: str
    target_exit_bar_id: str
    target_source_bar_ids: tuple[str, ...]
    decision_timeframe: str
    source_bar_timeframe: str
    horizon_value: int | None
    horizon_unit: str
    session_boundary_rule: str
    missing_bar_policy: str
    calendar_authority_version: str
    calendar_identity: Mapping[str, Any]
    price_authority_version: str
    target_code_hash: str
    resolution_policy_version: str
    no_future_data: bool

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_source_bar_ids"] = list(self.target_source_bar_ids)
        result["calendar_identity"] = dict(self.calendar_identity)
        return result


@dataclass(frozen=True)
class _NormalisedBar:
    asset_id: str
    symbol: str
    timeframe: str
    session_date: date
    start_timestamp: datetime
    final_timestamp: datetime
    open: float
    close: float
    source_bar_id: str
    status: str
    raw: Mapping[str, Any]


def target_code_hash() -> str:
    path = Path(__file__)
    if not path.exists():
        return canonical_hash({"module": __name__, "contract": MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION})
    return file_sha256(path)


def target_catalogue(
    *,
    calendar_authority_version: str = CALENDAR_AUTHORITY_VERSION,
    price_authority_version: str = PRICE_AUTHORITY_VERSION,
) -> tuple[TargetContract, ...]:
    code_hash = target_code_hash()
    base = {
        "calendar_authority_version": calendar_authority_version,
        "price_authority_version": price_authority_version,
        "target_code_hash": code_hash,
        "resolution_policy_version": TARGET_RESOLUTION_POLICY_VERSION,
    }
    return (
        _contract(
            "forward_return_1_session__decision_1Day",
            decision_timeframe="1Day",
            source_bar_timeframe="1Day",
            decision_schedule="XNYS daily close",
            horizon_value=1,
            horizon_unit=HORIZON_ELIGIBLE_TRADING_SESSIONS,
            session_boundary_rule="ordered eligible exchange sessions",
            entry_price_rule="decision session close",
            exit_price_rule="horizon session close",
            target_start_rule="decision-session close price anchor",
            target_end_rule="first eligible future session close",
            availability_rule="first later daily decision close after target end",
            missing_bar_policy="required daily source bars; classify missing, quarantined, ineligible, halt, or unknown gap",
            partial_session_policy="daily bars on early-close sessions remain eligible",
            early_close_policy="eligible session, use maintained calendar early close timestamp",
            overnight_policy="included across eligible daily sessions",
            **base,
        ),
        _contract(
            "forward_return_5_sessions__decision_1Day",
            decision_timeframe="1Day",
            source_bar_timeframe="1Day",
            decision_schedule="XNYS daily close",
            horizon_value=5,
            horizon_unit=HORIZON_ELIGIBLE_TRADING_SESSIONS,
            session_boundary_rule="ordered eligible exchange sessions",
            entry_price_rule="decision session close",
            exit_price_rule="fifth eligible future session close",
            target_start_rule="decision-session close price anchor",
            target_end_rule="fifth eligible future session close",
            availability_rule="first later daily decision close after target end",
            missing_bar_policy="required daily source bars; classify missing, quarantined, ineligible, halt, or unknown gap",
            partial_session_policy="daily bars on early-close sessions remain eligible",
            early_close_policy="eligible session, use maintained calendar early close timestamp",
            overnight_policy="included across eligible daily sessions",
            legacy_aliases=("forward_return_5d",),
            **base,
        ),
        _contract(
            "forward_return_10_sessions__decision_1Day",
            decision_timeframe="1Day",
            source_bar_timeframe="1Day",
            decision_schedule="XNYS daily close",
            horizon_value=10,
            horizon_unit=HORIZON_ELIGIBLE_TRADING_SESSIONS,
            session_boundary_rule="ordered eligible exchange sessions",
            entry_price_rule="decision session close",
            exit_price_rule="tenth eligible future session close",
            target_start_rule="decision-session close price anchor",
            target_end_rule="tenth eligible future session close",
            availability_rule="first later daily decision close after target end",
            missing_bar_policy="required daily source bars; classify missing, quarantined, ineligible, halt, or unknown gap",
            partial_session_policy="daily bars on early-close sessions remain eligible",
            early_close_policy="eligible session, use maintained calendar early close timestamp",
            overnight_policy="included across eligible daily sessions",
            legacy_aliases=("forward_return_10d",),
            **base,
        ),
        _contract(
            "forward_return_60m__decision_1h",
            decision_timeframe="1h",
            source_bar_timeframe="1h",
            decision_schedule="XNYS regular session hourly bar close",
            horizon_value=60,
            horizon_unit=HORIZON_ELAPSED_MARKET_MINUTES,
            session_boundary_rule="must mature within the same regular session",
            entry_price_rule="last finalised hourly source bar close at decision",
            exit_price_rule="hourly source bar close at elapsed horizon",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="decision timestamp plus 60 elapsed market minutes",
            availability_rule="after target-ending source bar is finalised",
            missing_bar_policy="required hourly source bars; no silent rollover",
            partial_session_policy="early-close sessions shorten the allowed horizon",
            early_close_policy="session close from maintained calendar; near-close targets conflict",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp + 60m <= regular_close",
            **base,
        ),
        _contract(
            "forward_return_240m__decision_1h",
            decision_timeframe="1h",
            source_bar_timeframe="1h",
            decision_schedule="XNYS regular session hourly bar close",
            horizon_value=240,
            horizon_unit=HORIZON_ELAPSED_MARKET_MINUTES,
            session_boundary_rule="must mature within the same regular session",
            entry_price_rule="last finalised hourly source bar close at decision",
            exit_price_rule="hourly source bar close at elapsed horizon",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="decision timestamp plus 240 elapsed market minutes",
            availability_rule="after target-ending source bar is finalised",
            missing_bar_policy="required hourly source bars; no silent rollover",
            partial_session_policy="early-close sessions shorten the allowed horizon",
            early_close_policy="session close from maintained calendar; near-close targets conflict",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp + 240m <= regular_close",
            **base,
        ),
        _contract(
            "forward_return_to_close__decision_1h",
            decision_timeframe="1h",
            source_bar_timeframe="1h",
            decision_schedule="XNYS regular session hourly bar close",
            horizon_value=None,
            horizon_unit=HORIZON_TO_SESSION_CLOSE,
            session_boundary_rule="exit at same regular session close",
            entry_price_rule="last finalised hourly source bar close at decision",
            exit_price_rule="source bar close finalised at session close",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="maintained calendar same-session close",
            availability_rule="after session-close source bar is finalised",
            missing_bar_policy="required hourly source bar ending at session close",
            partial_session_policy="uses early close when present",
            early_close_policy="exit at maintained early close timestamp",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp < regular_close",
            **base,
        ),
        _contract(
            "forward_return_next_session__decision_1h",
            decision_timeframe="1h",
            source_bar_timeframe="1h",
            decision_schedule="XNYS regular session hourly bar close",
            horizon_value=None,
            horizon_unit=HORIZON_TO_NEXT_SESSION_OPEN,
            session_boundary_rule="roll to next eligible session open by contract",
            entry_price_rule="last finalised hourly source bar close at decision",
            exit_price_rule="next eligible session first source bar open",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="maintained calendar next eligible session open",
            availability_rule="after first next-session source bar is finalised",
            missing_bar_policy="required first next-session source bar",
            partial_session_policy="not applicable to exit; entry session may be partial",
            early_close_policy="next eligible session determined by maintained calendar",
            overnight_policy="included by explicit contract",
            final_permissible_decision_rule="decision_timestamp < regular_close",
            **base,
        ),
        _contract(
            "forward_return_30m__decision_5m",
            decision_timeframe="5m",
            source_bar_timeframe="5m",
            decision_schedule="XNYS regular session five-minute bar close",
            horizon_value=30,
            horizon_unit=HORIZON_ELAPSED_MARKET_MINUTES,
            session_boundary_rule="must mature within the same regular session",
            entry_price_rule="last finalised five-minute source bar close at decision",
            exit_price_rule="five-minute source bar close at elapsed horizon",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="decision timestamp plus 30 elapsed market minutes",
            availability_rule="after target-ending source bar is finalised",
            missing_bar_policy="required five-minute source bars; no silent skip",
            partial_session_policy="early-close sessions shorten the allowed horizon",
            early_close_policy="session close from maintained calendar; near-close targets conflict",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp + 30m <= regular_close",
            **base,
        ),
        _contract(
            "forward_return_60m__decision_5m",
            decision_timeframe="5m",
            source_bar_timeframe="5m",
            decision_schedule="XNYS regular session five-minute bar close",
            horizon_value=60,
            horizon_unit=HORIZON_ELAPSED_MARKET_MINUTES,
            session_boundary_rule="must mature within the same regular session",
            entry_price_rule="last finalised five-minute source bar close at decision",
            exit_price_rule="five-minute source bar close at elapsed horizon",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="decision timestamp plus 60 elapsed market minutes",
            availability_rule="after target-ending source bar is finalised",
            missing_bar_policy="required five-minute source bars; no silent skip",
            partial_session_policy="early-close sessions shorten the allowed horizon",
            early_close_policy="session close from maintained calendar; near-close targets conflict",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp + 60m <= regular_close",
            **base,
        ),
        _contract(
            "forward_return_to_close__decision_5m",
            decision_timeframe="5m",
            source_bar_timeframe="5m",
            decision_schedule="XNYS regular session five-minute bar close",
            horizon_value=None,
            horizon_unit=HORIZON_TO_SESSION_CLOSE,
            session_boundary_rule="exit at same regular session close",
            entry_price_rule="last finalised five-minute source bar close at decision",
            exit_price_rule="source bar close finalised at session close",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="maintained calendar same-session close",
            availability_rule="after session-close source bar is finalised",
            missing_bar_policy="required five-minute source bar ending at session close",
            partial_session_policy="uses early close when present",
            early_close_policy="exit at maintained early close timestamp",
            overnight_policy="excluded",
            final_permissible_decision_rule="decision_timestamp < regular_close",
            **base,
        ),
        _contract(
            "forward_return_next_open__decision_5m",
            decision_timeframe="5m",
            source_bar_timeframe="5m",
            decision_schedule="XNYS regular session five-minute bar close",
            horizon_value=None,
            horizon_unit=HORIZON_TO_NEXT_SESSION_OPEN,
            session_boundary_rule="roll to next eligible session open by contract",
            entry_price_rule="last finalised five-minute source bar close at decision",
            exit_price_rule="next eligible session first source bar open",
            target_start_rule="decision timestamp after source bar finalisation",
            target_end_rule="maintained calendar next eligible session open",
            availability_rule="after first next-session source bar is finalised",
            missing_bar_policy="required first next-session source bar",
            partial_session_policy="not applicable to exit; entry session may be partial",
            early_close_policy="next eligible session determined by maintained calendar",
            overnight_policy="included by explicit contract",
            final_permissible_decision_rule="decision_timestamp < regular_close",
            **base,
        ),
    )


def target_catalogue_payload() -> dict[str, Any]:
    contracts = [contract.payload() for contract in target_catalogue()]
    return {
        "catalogue_version": TARGET_CATALOGUE_VERSION,
        "contract_version": MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
        "supported_horizon_units": sorted(SUPPORTED_HORIZON_UNITS),
        "supported_resolution_states": sorted(TARGET_RESOLUTION_STATES),
        "contracts": contracts,
        "content_hash": canonical_hash(contracts),
    }


def resolve_target_contract(
    target_id: str,
    *,
    allow_legacy_aliases: bool = True,
) -> TargetContract:
    requested = str(target_id or "").strip()
    if not requested:
        raise TargetContractError("target_id is required")
    by_id = {contract.target_id: contract for contract in target_catalogue()}
    if requested in by_id:
        return by_id[requested]
    if not allow_legacy_aliases:
        raise TargetContractError(
            f"strict research mode requires a canonical target_id, not {requested!r}"
        )
    for contract in by_id.values():
        if requested in contract.legacy_aliases:
            return contract
    raise TargetContractError(f"unknown target contract: {requested}")


def require_explicit_target_contract(target_id: str) -> TargetContract:
    return resolve_target_contract(target_id, allow_legacy_aliases=False)


def calculate_target(
    *,
    asset_id: str,
    decision_timestamp: datetime | date | str,
    bar_source: Sequence[Mapping[str, Any] | _NormalisedBar],
    target_contract: TargetContract | Mapping[str, Any] | str,
    calendar_authority: ExchangeCalendarAuthority | None = None,
    source_cutoff: datetime | date | str | None = None,
    training_cutoff: datetime | date | str | None = None,
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]] = (),
) -> TargetComputationResult:
    authority = calendar_authority or default_calendar_authority()
    contract = _coerce_contract(target_contract)
    decision = _coerce_decision_timestamp(
        decision_timestamp,
        contract=contract,
        authority=authority,
    )
    calendar_identity = authority.identity(
        start=decision.astimezone(EASTERN).date(),
        end=decision.astimezone(EASTERN).date(),
    )
    all_bars = _normalise_bars(
        bar_source,
        asset_id=asset_id,
        timeframe=contract.source_bar_timeframe,
        authority=authority,
    )
    resolved_source_cutoff = _coerce_timestamp(source_cutoff) if source_cutoff is not None else _max_final_timestamp(all_bars)
    if resolved_source_cutoff is None:
        resolved_source_cutoff = decision
    resolved_training_cutoff = (
        _coerce_timestamp(training_cutoff)
        if training_cutoff is not None
        else resolved_source_cutoff
    )
    visible_bars = [
        bar for bar in all_bars if bar.final_timestamp <= resolved_source_cutoff
    ]
    boundary = _target_boundary(
        contract=contract,
        decision=decision,
        authority=authority,
        market_halts=market_halts,
    )
    if boundary["classification"]:
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=str(boundary["classification"]),
            reason=str(boundary["reason"]),
            target_start=boundary.get("target_start"),
            label_start=boundary.get("label_start"),
            target_end=boundary.get("target_end"),
            target_available=boundary.get("target_available"),
        )
    target_start = _required_timestamp(boundary["target_start"], "target_start")
    label_start = boundary.get("label_start")
    target_end = _required_timestamp(boundary["target_end"], "target_end")
    target_available = _required_timestamp(boundary["target_available"], "target_available")

    if target_end > resolved_source_cutoff:
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=RIGHT_CENSORED,
            reason="target_end_after_source_cutoff",
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
        )
    if target_available > resolved_source_cutoff:
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=NOT_YET_MATURE,
            reason="target_available_after_source_cutoff",
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
        )

    entry = _entry_bar(
        visible_bars,
        contract=contract,
        decision=decision,
        authority=authority,
    )
    if entry is None:
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=MISSING_SOURCE_BAR,
            reason="entry_source_bar_missing",
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
        )
    exit_bar = _exit_bar(
        visible_bars,
        contract=contract,
        boundary=boundary,
    )
    if exit_bar is None:
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=MISSING_SOURCE_BAR,
            reason="exit_source_bar_missing",
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
            entry_bar=entry,
        )
    source_path_bars, source_path_issue = _required_source_path(
        visible_bars,
        contract=contract,
        authority=authority,
        boundary=boundary,
        label_start=label_start,
        target_end=target_end,
        entry_bar=entry,
        exit_bar=exit_bar,
    )
    if source_path_issue is not None:
        classification, reason = source_path_issue
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=classification,
            reason=reason,
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
            entry_bar=entry,
            exit_bar=exit_bar,
            source_bars=source_path_bars,
        )
    bar_issue = _bar_issue(*source_path_bars)
    halt_issue = _halt_issue(decision, target_end, market_halts)
    if halt_issue:
        bar_issue = halt_issue
    if bar_issue is not None:
        classification, reason = bar_issue
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=classification,
            reason=reason,
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
            entry_bar=entry,
            exit_bar=exit_bar,
            source_bars=source_path_bars,
        )
    entry_price = entry.close
    exit_price = exit_bar.open if contract.horizon_unit == HORIZON_TO_NEXT_SESSION_OPEN else exit_bar.close
    if not _finite_positive(entry_price) or not math.isfinite(exit_price):
        return _result(
            contract=contract,
            asset_id=asset_id,
            decision=decision,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            calendar_identity=calendar_identity,
            classification=MISSING_SOURCE_BAR,
            reason="entry_or_exit_price_invalid",
            target_start=target_start,
            label_start=label_start,
            target_end=target_end,
            target_available=target_available,
            entry_bar=entry,
            exit_bar=exit_bar,
        )
    value = (exit_price / entry_price) - 1.0
    return _result(
        contract=contract,
        asset_id=asset_id,
        decision=decision,
        source_cutoff=resolved_source_cutoff,
        training_cutoff=resolved_training_cutoff,
        calendar_identity=calendar_identity,
        classification=MATURED_VALID,
        reason=(
            "value_realised_and_trainable"
            if target_available <= resolved_training_cutoff
            else "value_realised_after_training_cutoff"
        ),
        value=value,
        target_start=target_start,
        label_start=label_start,
        target_end=target_end,
        target_available=target_available,
        entry_bar=entry,
        exit_bar=exit_bar,
        source_bars=source_path_bars,
    )


def normalise_target_bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    asset_id: str,
    timeframe: str,
    calendar_authority: ExchangeCalendarAuthority | None = None,
) -> list[_NormalisedBar]:
    authority = calendar_authority or default_calendar_authority()
    return _normalise_bars(
        rows,
        asset_id=asset_id,
        timeframe=timeframe,
        authority=authority,
    )


def build_target_manifest(
    rows: Sequence[Mapping[str, Any] | TargetComputationResult],
    *,
    selected_target: TargetContract | str,
    source_cutoff: datetime | date | str | None,
    output_paths: Sequence[Path | str] = (),
    source_paths: Sequence[Path | str] = (),
    configuration: Mapping[str, Any] | None = None,
    calendar_identity: Mapping[str, Any] | None = None,
    producer_command: str = "ticket-71-multi-timeframe-target-authority",
    producer_module: str = "core.research.ml.target_authority",
) -> dict[str, Any]:
    contract = _coerce_contract(selected_target)
    payload_rows = [
        row.payload() if isinstance(row, TargetComputationResult) else dict(row)
        for row in rows
    ]
    classifications = Counter(str(row.get("target_resolution_classification") or "") for row in payload_rows)
    reasons = Counter(str(row.get("target_resolution_reason") or "") for row in payload_rows)
    mature = sum(_truthy(row.get("target_is_mature")) for row in payload_rows)
    realised = sum(_truthy(row.get("target_is_realised")) for row in payload_rows)
    trainable = sum(_truthy(row.get("target_is_trainable")) for row in payload_rows)
    manifest: dict[str, Any] = {
        "manifest_schema_version": "multi_timeframe_target_dataset_manifest.v1",
        "target_catalogue_version": TARGET_CATALOGUE_VERSION,
        "target_contract_version": MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
        "selected_target_id": contract.target_id,
        "legacy_aliases": list(contract.legacy_aliases),
        "decision_timeframe": contract.decision_timeframe,
        "source_bar_timeframe": contract.source_bar_timeframe,
        "horizon_value": contract.horizon_value,
        "horizon_unit": contract.horizon_unit,
        "session_boundary_rule": contract.session_boundary_rule,
        "calendar_identity": dict(calendar_identity or {}),
        "calendar_authority_version": contract.calendar_authority_version,
        "price_authority_version": contract.price_authority_version,
        "row_counts": {
            "total": len(payload_rows),
            "matured": mature,
            "realised": realised,
            "trainable": trainable,
            "right_censored": int(classifications.get(RIGHT_CENSORED, 0)),
            "missing_source_bar": int(classifications.get(MISSING_SOURCE_BAR, 0)),
            "quarantined_source_bar": int(classifications.get(QUARANTINED_SOURCE_BAR, 0)),
            "ineligible_source_bar": int(classifications.get(INELIGIBLE_SOURCE_BAR, 0)),
            "session_boundary_conflict": int(classifications.get(SESSION_BOUNDARY_CONFLICT, 0)),
        },
        "resolution_classification_counts": dict(sorted(classifications.items())),
        "resolution_reason_counts": dict(sorted(reasons.items())),
        "content_hash": canonical_hash(payload_rows),
        "code_config_hash": canonical_hash(
            {
                "target_code_hash": contract.target_code_hash,
                "configuration": dict(configuration or {}),
                "contract": contract.payload(),
            }
        ),
        "source_cutoff": _format_timestamp(_coerce_timestamp(source_cutoff)) if source_cutoff is not None else "",
        "output_hashes": [file_identity(Path(path)) for path in output_paths],
        "source_paths": [file_identity(Path(path)) for path in source_paths],
        "git_patch_provenance": source_worktree_provenance(),
        "dataset_manifest_integration": {
            "compatible_manifest_schema": "dataset_build_manifest_v1",
            "target_contract_version_field": contract.target_id,
            "label_code_version_field": contract.target_code_hash,
            "market_calendar_authority_version_field": contract.calendar_authority_version,
        },
        "certification_envelope_integration": {
            "authority_versions.target_contract_version": contract.target_id,
            "authority_versions.market_calendar_authority": dict(calendar_identity or {}),
        },
        "producer_command": producer_command,
        "producer_module": producer_module,
        "model_training_performed": False,
        "model_or_policy_promoted": False,
    }
    manifest["manifest_hash"] = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    return manifest


def validate_target_availability(
    rows: Sequence[Mapping[str, Any] | TargetComputationResult],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    payload_rows = [
        row.payload() if isinstance(row, TargetComputationResult) else dict(row)
        for row in rows
    ]
    for index, row in enumerate(payload_rows):
        decision = _parse_optional_timestamp(row.get("decision_timestamp"))
        end = _parse_optional_timestamp(row.get("target_end_timestamp"))
        available = _parse_optional_timestamp(row.get("target_available_timestamp"))
        training_cutoff = _parse_optional_timestamp(row.get("training_cutoff"))
        row_id = str(row.get("row_id") or row.get("target_row_id") or index)
        if end and available and available < end:
            violations.append({"row_id": row_id, "reason": "target_available_before_target_end"})
        if _truthy(row.get("target_is_trainable")):
            if not (decision and end and available and training_cutoff):
                violations.append({"row_id": row_id, "reason": "trainable_row_missing_required_timestamp"})
                continue
            if end <= decision:
                violations.append({"row_id": row_id, "reason": "target_end_not_after_decision"})
            if available > training_cutoff:
                violations.append({"row_id": row_id, "reason": "target_available_after_training_cutoff"})
            if row.get("target_resolution_classification") != MATURED_VALID:
                violations.append({"row_id": row_id, "reason": "trainable_row_not_matured_valid"})
    return {
        "validation_version": "multi_timeframe_target_availability_validation.v1",
        "status": "PASSED" if not violations else "FAILED",
        "row_count": len(payload_rows),
        "violation_count": len(violations),
        "violations": violations,
        "checked_invariants": [
            "target_available_timestamp >= target_end_timestamp",
            "target_end_timestamp > decision_timestamp for trainable rows",
            "target_available_timestamp <= training_cutoff for trainable rows",
            "target_is_trainable implies MATURED_VALID",
        ],
    }


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def source_worktree_provenance() -> dict[str, Any]:
    return {
        "contract_version": "source_worktree_provenance_v1",
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_branch": _git("branch", "--show-current") or None,
        "dirty_worktree": bool(_git("status", "--short")),
    }


def _contract(target_id: str, *, legacy_aliases: Sequence[str] = (), **kwargs: Any) -> TargetContract:
    if kwargs.get("horizon_unit") not in SUPPORTED_HORIZON_UNITS:
        raise TargetContractError(f"unsupported horizon unit: {kwargs.get('horizon_unit')}")
    return TargetContract(target_id=target_id, legacy_aliases=tuple(legacy_aliases), **kwargs)


def _coerce_contract(value: TargetContract | Mapping[str, Any] | str) -> TargetContract:
    if isinstance(value, TargetContract):
        return value
    if isinstance(value, str):
        return resolve_target_contract(value)
    payload = dict(value)
    legacy = payload.get("legacy_aliases") or ()
    return TargetContract(
        target_id=str(payload["target_id"]),
        legacy_aliases=tuple(str(item) for item in legacy),
        decision_timeframe=str(payload["decision_timeframe"]),
        source_bar_timeframe=str(payload["source_bar_timeframe"]),
        decision_schedule=str(payload["decision_schedule"]),
        horizon_value=(
            None if payload.get("horizon_value") is None else int(payload["horizon_value"])
        ),
        horizon_unit=str(payload["horizon_unit"]),
        session_boundary_rule=str(payload["session_boundary_rule"]),
        entry_price_rule=str(payload["entry_price_rule"]),
        exit_price_rule=str(payload["exit_price_rule"]),
        target_start_rule=str(payload["target_start_rule"]),
        target_end_rule=str(payload["target_end_rule"]),
        availability_rule=str(payload["availability_rule"]),
        missing_bar_policy=str(payload["missing_bar_policy"]),
        partial_session_policy=str(payload["partial_session_policy"]),
        early_close_policy=str(payload["early_close_policy"]),
        overnight_policy=str(payload["overnight_policy"]),
        calendar_authority_version=str(payload["calendar_authority_version"]),
        price_authority_version=str(payload["price_authority_version"]),
        target_code_hash=str(payload["target_code_hash"]),
        resolution_policy_version=str(payload["resolution_policy_version"]),
        pre_market_decisions_allowed=bool(payload.get("pre_market_decisions_allowed", False)),
        after_hours_decisions_allowed=bool(payload.get("after_hours_decisions_allowed", False)),
        regular_session_only=bool(payload.get("regular_session_only", True)),
        final_permissible_decision_rule=str(payload.get("final_permissible_decision_rule") or ""),
        target_maturity_rule=str(payload.get("target_maturity_rule") or "target_available_timestamp <= source_cutoff"),
    )


def _coerce_decision_timestamp(
    value: datetime | date | str,
    *,
    contract: TargetContract,
    authority: ExchangeCalendarAuthority,
) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        if contract.decision_timeframe == "1Day":
            return _session_close_timestamp(authority, value)
        return _session_open_timestamp(authority, value)
    text = str(value).strip() if not isinstance(value, datetime) else ""
    if text and len(text) == 10:
        day = date.fromisoformat(text)
        if contract.decision_timeframe == "1Day":
            return _session_close_timestamp(authority, day)
        return _session_open_timestamp(authority, day)
    return _coerce_timestamp(value)


def _coerce_timestamp(value: datetime | date | str | Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    else:
        text = str(value or "").strip()
        if not text:
            raise TargetContractError("timestamp value is required")
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_optional_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _coerce_timestamp(value)
    except (TypeError, ValueError):
        return None


def _normalise_bars(
    rows: Sequence[Mapping[str, Any] | _NormalisedBar],
    *,
    asset_id: str,
    timeframe: str,
    authority: ExchangeCalendarAuthority,
) -> list[_NormalisedBar]:
    requested_timeframe = _normalise_timeframe(timeframe)
    bars: list[_NormalisedBar] = []
    for row in rows:
        if isinstance(row, _NormalisedBar):
            if _normalised_bar_matches_asset(row, asset_id):
                bars.append(row)
            continue
        if _row_matches_asset(row, asset_id):
            bars.append(_normalise_bar(row, timeframe=requested_timeframe, authority=authority))
    return sorted(
        [bar for bar in bars if bar.timeframe == requested_timeframe],
        key=lambda bar: (bar.asset_id, bar.final_timestamp, bar.source_bar_id),
    )


def _normalise_bar(
    row: Mapping[str, Any],
    *,
    timeframe: str,
    authority: ExchangeCalendarAuthority,
) -> _NormalisedBar:
    raw_timeframe = (
        row.get("source_bar_timeframe")
        or row.get("timeframe")
        or row.get("requested_timeframe")
        or row.get("native_timeframe")
        or timeframe
    )
    resolved_timeframe = _normalise_timeframe(str(raw_timeframe))
    symbol = str(row.get("canonical_symbol") or row.get("symbol") or row.get("asset_id") or "").upper()
    asset_id = str(row.get("asset_id") or symbol)
    session_value = row.get("session_date")
    timestamp_value = (
        row.get("timestamp_utc")
        or row.get("timestamp")
        or row.get("bar_start_timestamp")
        or row.get("bar_timestamp")
        or session_value
    )
    if resolved_timeframe == "1Day":
        session_day = _coerce_date(session_value or timestamp_value)
        start = _session_open_timestamp(authority, session_day)
        final = _session_close_timestamp(authority, session_day)
    else:
        start = _coerce_timestamp(timestamp_value)
        session_day = _coerce_date(session_value) if session_value else start.astimezone(EASTERN).date()
        duration = timedelta(minutes=_timeframe_minutes(resolved_timeframe))
        final = _parse_optional_timestamp(row.get("bar_finalized_timestamp") or row.get("bar_end_timestamp")) or start + duration
    open_price = _float(row.get("open"))
    close_price = _float(row.get("close"))
    status = str(
        row.get("bar_status")
        or row.get("source_bar_status")
        or row.get("status")
        or row.get("session_type")
        or "OK"
    )
    source_bar_id = str(
        row.get("source_bar_id")
        or row.get("source_row_hash")
        or row.get("row_id")
        or f"{asset_id}|{resolved_timeframe}|{_format_timestamp(start)}"
    )
    return _NormalisedBar(
        asset_id=asset_id,
        symbol=symbol,
        timeframe=resolved_timeframe,
        session_date=session_day,
        start_timestamp=start,
        final_timestamp=final,
        open=open_price,
        close=close_price,
        source_bar_id=source_bar_id,
        status=status,
        raw=dict(row),
    )


def _target_boundary(
    *,
    contract: TargetContract,
    decision: datetime,
    authority: ExchangeCalendarAuthority,
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]],
) -> dict[str, Any]:
    if contract.horizon_unit == HORIZON_ELIGIBLE_TRADING_SESSIONS:
        return _daily_session_boundary(contract=contract, decision=decision, authority=authority)
    context = exchange_session_context(decision, market_halts=market_halts)
    if context.trading_session == HALTED_SESSION:
        return {
            "classification": HALT_AFFECTED,
            "reason": "decision_timestamp_overlaps_represented_halt",
            "target_start": decision,
        }
    if contract.regular_session_only and context.trading_session != REGULAR_SESSION:
        return {
            "classification": SESSION_BOUNDARY_CONFLICT,
            "reason": f"decision_timestamp_not_regular_session:{context.trading_session}",
            "target_start": decision,
        }
    if context.regular_close is None or context.regular_open is None:
        return {
            "classification": SESSION_BOUNDARY_CONFLICT,
            "reason": "calendar_has_no_regular_session_for_decision",
            "target_start": decision,
        }
    close = context.regular_close.astimezone(timezone.utc)
    if decision >= close:
        return {
            "classification": SESSION_BOUNDARY_CONFLICT,
            "reason": "decision_timestamp_at_or_after_regular_close",
            "target_start": decision,
            "target_end": close,
            "target_available": close,
        }
    if contract.horizon_unit == HORIZON_ELAPSED_MARKET_MINUTES:
        horizon = int(contract.horizon_value or 0)
        target_end = decision + timedelta(minutes=horizon)
        if target_end > close:
            return {
                "classification": SESSION_BOUNDARY_CONFLICT,
                "reason": "elapsed_market_minutes_crosses_session_close",
                "target_start": decision,
                "label_start": decision + timedelta(minutes=_timeframe_minutes(contract.source_bar_timeframe)),
                "target_end": target_end,
                "target_available": target_end,
            }
        return {
            "classification": "",
            "reason": "",
            "target_start": decision,
            "label_start": decision + timedelta(minutes=_timeframe_minutes(contract.source_bar_timeframe)),
            "target_end": target_end,
            "target_available": target_end,
            "exit_lookup": "final_timestamp",
        }
    if contract.horizon_unit == HORIZON_TO_SESSION_CLOSE:
        return {
            "classification": "",
            "reason": "",
            "target_start": decision,
            "label_start": decision + timedelta(minutes=_timeframe_minutes(contract.source_bar_timeframe)),
            "target_end": close,
            "target_available": close,
            "exit_lookup": "final_timestamp",
        }
    if contract.horizon_unit == HORIZON_TO_NEXT_SESSION_OPEN:
        next_session = authority.next_session(context.session_date)
        if next_session is None:
            return {
                "classification": RIGHT_CENSORED,
                "reason": "calendar_has_no_next_eligible_session",
                "target_start": decision,
            }
        open_ts = _session_open_timestamp(authority, next_session)
        available = open_ts + timedelta(minutes=_timeframe_minutes(contract.source_bar_timeframe))
        return {
            "classification": "",
            "reason": "",
            "target_start": decision,
            "label_start": open_ts,
            "target_end": open_ts,
            "target_available": available,
            "exit_lookup": "start_timestamp",
        }
    raise TargetContractError(f"unsupported horizon unit: {contract.horizon_unit}")


def _daily_session_boundary(
    *,
    contract: TargetContract,
    decision: datetime,
    authority: ExchangeCalendarAuthority,
) -> dict[str, Any]:
    day = decision.astimezone(EASTERN).date()
    record = authority.session(day)
    if not record.is_trading_day:
        return {
            "classification": SESSION_BOUNDARY_CONFLICT,
            "reason": f"decision_date_not_trading_session:{record.base_status}",
            "target_start": decision,
        }
    horizon = int(contract.horizon_value or 0)
    if horizon <= 0:
        raise TargetContractError("daily eligible session horizon must be positive")
    first_future = authority.next_session(day)
    end_session = _nth_next_session(authority, day, horizon)
    if end_session is None:
        return {
            "classification": RIGHT_CENSORED,
            "reason": "calendar_has_no_horizon_session",
            "target_start": decision,
        }
    availability_session = authority.next_session(end_session)
    if availability_session is None:
        return {
            "classification": RIGHT_CENSORED,
            "reason": "calendar_has_no_availability_session",
            "target_start": decision,
            "target_end": _session_close_timestamp(authority, end_session),
        }
    return {
        "classification": "",
        "reason": "",
        "target_start": _session_close_timestamp(authority, day),
        "label_start": _session_close_timestamp(authority, first_future) if first_future else None,
        "target_end": _session_close_timestamp(authority, end_session),
        "target_available": _session_close_timestamp(authority, availability_session),
        "entry_session": day,
        "exit_session": end_session,
        "exit_lookup": "session_date",
    }


def _entry_bar(
    bars: Sequence[_NormalisedBar],
    *,
    contract: TargetContract,
    decision: datetime,
    authority: ExchangeCalendarAuthority,
) -> _NormalisedBar | None:
    if contract.source_bar_timeframe == "1Day":
        day = decision.astimezone(EASTERN).date()
        return _bar_for_session(bars, day)
    eligible = [
        bar for bar in bars
        if bar.final_timestamp <= decision
    ]
    if not eligible:
        return None
    decision_session = decision.astimezone(EASTERN).date()
    same_session = [
        bar for bar in eligible
        if bar.session_date == decision_session
    ]
    candidates = same_session or eligible
    return max(candidates, key=lambda bar: (bar.final_timestamp, bar.source_bar_id))


def _exit_bar(
    bars: Sequence[_NormalisedBar],
    *,
    contract: TargetContract,
    boundary: Mapping[str, Any],
) -> _NormalisedBar | None:
    lookup = str(boundary.get("exit_lookup") or "")
    if lookup == "session_date":
        day = boundary.get("exit_session")
        return _bar_for_session(bars, day) if isinstance(day, date) else None
    if lookup == "start_timestamp":
        target = _required_timestamp(boundary.get("target_end"), "target_end")
        return _bar_for_start(bars, target)
    target = _required_timestamp(boundary.get("target_end"), "target_end")
    return _bar_for_final(bars, target)


def _required_source_path(
    bars: Sequence[_NormalisedBar],
    *,
    contract: TargetContract,
    authority: ExchangeCalendarAuthority,
    boundary: Mapping[str, Any],
    label_start: datetime | None,
    target_end: datetime,
    entry_bar: _NormalisedBar,
    exit_bar: _NormalisedBar,
) -> tuple[tuple[_NormalisedBar, ...], tuple[str, str] | None]:
    required = [entry_bar]
    if contract.horizon_unit == HORIZON_ELIGIBLE_TRADING_SESSIONS:
        entry_session = boundary.get("entry_session")
        exit_session = boundary.get("exit_session")
        if isinstance(entry_session, date) and isinstance(exit_session, date):
            for session_day in authority.sessions(entry_session, exit_session):
                bar = _bar_for_session(bars, session_day)
                if bar is None:
                    return (
                        _dedupe_bars((*required, exit_bar)),
                        (
                            MISSING_SOURCE_BAR,
                            f"required_session_source_bar_missing:{session_day.isoformat()}",
                        ),
                    )
                required.append(bar)
    elif contract.horizon_unit in {HORIZON_ELAPSED_MARKET_MINUTES, HORIZON_TO_SESSION_CLOSE}:
        step = timedelta(minutes=_timeframe_minutes(contract.source_bar_timeframe))
        first_final = label_start or target_end
        for expected_final in _expected_final_timestamps(first_final, target_end, step):
            bar = _bar_for_final(bars, expected_final)
            if bar is None:
                return (
                    _dedupe_bars((*required, exit_bar)),
                    (
                        MISSING_SOURCE_BAR,
                        f"required_source_bar_missing:{_format_timestamp(expected_final)}",
                    ),
                )
            required.append(bar)
    required.append(exit_bar)
    return _dedupe_bars(required), None


def _expected_final_timestamps(
    first_final: datetime,
    target_end: datetime,
    step: timedelta,
) -> tuple[datetime, ...]:
    expected: list[datetime] = []
    current = first_final
    while current <= target_end:
        expected.append(current)
        current += step
    if target_end not in expected:
        expected.append(target_end)
    return tuple(sorted(set(expected)))


def _dedupe_bars(bars: Sequence[_NormalisedBar]) -> tuple[_NormalisedBar, ...]:
    output: list[_NormalisedBar] = []
    seen: set[str] = set()
    for bar in bars:
        if bar.source_bar_id in seen:
            continue
        seen.add(bar.source_bar_id)
        output.append(bar)
    return tuple(output)


def _bar_for_session(bars: Sequence[_NormalisedBar], day: date) -> _NormalisedBar | None:
    matches = [bar for bar in bars if bar.session_date == day]
    if not matches:
        return None
    return max(matches, key=lambda bar: (bar.final_timestamp, bar.source_bar_id))


def _bar_for_final(bars: Sequence[_NormalisedBar], timestamp: datetime) -> _NormalisedBar | None:
    matches = [bar for bar in bars if bar.final_timestamp == timestamp]
    if not matches:
        return None
    return sorted(matches, key=lambda bar: bar.source_bar_id)[0]


def _bar_for_start(bars: Sequence[_NormalisedBar], timestamp: datetime) -> _NormalisedBar | None:
    matches = [bar for bar in bars if bar.start_timestamp == timestamp]
    if not matches:
        return None
    return sorted(matches, key=lambda bar: bar.source_bar_id)[0]


def _bar_issue(*bars: _NormalisedBar) -> tuple[str, str] | None:
    for bar in bars:
        state = _bar_state(bar)
        if state != MATURED_VALID:
            return state, f"{state.lower()}:{bar.source_bar_id}"
    return None


def _bar_state(bar: _NormalisedBar) -> str:
    raw = bar.raw
    status = bar.status.strip().upper()
    quarantine = str(raw.get("canonical_price_quarantine_state") or raw.get("quarantine_state") or "").upper()
    eligibility = str(raw.get("canonical_price_eligibility_state") or raw.get("eligibility_state") or "").upper()
    if _truthy(raw.get("quarantined")) or "QUARANT" in status or (quarantine and quarantine not in {"OK", "CLEAR", "ELIGIBLE"}):
        return QUARANTINED_SOURCE_BAR
    if _truthy(raw.get("ineligible")) or "INELIGIBLE" in status or eligibility in {"INELIGIBLE", "NOT_ELIGIBLE"}:
        return INELIGIBLE_SOURCE_BAR
    if _truthy(raw.get("halted")) or "HALT" in status:
        return HALT_AFFECTED
    if "PROVIDER_GAP" in status or "UNKNOWN_GAP" in status or _truthy(raw.get("provider_gap")):
        return UNKNOWN_SOURCE_GAP
    if status in {"MISSING", "NO_TRADE", "NO_TRADE_BAR", "EMPTY_BAR"}:
        return MISSING_SOURCE_BAR
    return MATURED_VALID


def _halt_issue(
    decision: datetime,
    target_end: datetime,
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]],
) -> tuple[str, str] | None:
    for halt in market_halts:
        if isinstance(halt, Mapping):
            start = _parse_optional_timestamp(halt.get("start") or halt.get("start_timestamp"))
            end = _parse_optional_timestamp(halt.get("end") or halt.get("end_timestamp"))
            reason = str(halt.get("reason") or halt.get("halt_reason") or "represented_market_halt")
        else:
            start = _parse_optional_timestamp(halt[0])
            end = _parse_optional_timestamp(halt[1])
            reason = "represented_market_halt"
        if start and end and start < target_end and end > decision:
            return HALT_AFFECTED, reason
    return None


def _result(
    *,
    contract: TargetContract,
    asset_id: str,
    decision: datetime,
    source_cutoff: datetime,
    training_cutoff: datetime,
    calendar_identity: Mapping[str, Any],
    classification: str,
    reason: str,
    value: float | None = None,
    target_start: datetime | None = None,
    label_start: datetime | None = None,
    target_end: datetime | None = None,
    target_available: datetime | None = None,
    entry_bar: _NormalisedBar | None = None,
    exit_bar: _NormalisedBar | None = None,
    source_bars: Sequence[_NormalisedBar] | None = None,
) -> TargetComputationResult:
    if classification not in TARGET_RESOLUTION_STATES:
        raise TargetContractError(f"unsupported target resolution classification: {classification}")
    mature = bool(target_available and target_available <= source_cutoff)
    realised = classification == MATURED_VALID
    trainable = realised and bool(target_available and target_available <= training_cutoff)
    resolved_source_bars = tuple(
        bar.source_bar_id for bar in (entry_bar, exit_bar)
        if bar is not None
    ) if source_bars is None else tuple(bar.source_bar_id for bar in _dedupe_bars(source_bars))
    no_future = all(
        bar.final_timestamp <= source_cutoff
        for bar in (
            _dedupe_bars(source_bars)
            if source_bars is not None
            else tuple(bar for bar in (entry_bar, exit_bar) if bar is not None)
        )
    )
    return TargetComputationResult(
        target_id=contract.target_id,
        asset_id=str(asset_id),
        value=value if realised else None,
        decision_timestamp=_format_timestamp(decision),
        target_start_timestamp=_format_timestamp(target_start),
        label_start_timestamp=_format_timestamp(label_start),
        target_end_timestamp=_format_timestamp(target_end),
        target_available_timestamp=_format_timestamp(target_available),
        training_cutoff=_format_timestamp(training_cutoff),
        target_source_cutoff=_format_timestamp(source_cutoff),
        target_is_mature=mature,
        target_is_realised=realised,
        target_is_trainable=trainable,
        target_resolution_classification=classification,
        target_resolution_reason=reason,
        target_entry_bar_id=entry_bar.source_bar_id if entry_bar else "",
        target_exit_bar_id=exit_bar.source_bar_id if exit_bar else "",
        target_source_bar_ids=resolved_source_bars,
        decision_timeframe=contract.decision_timeframe,
        source_bar_timeframe=contract.source_bar_timeframe,
        horizon_value=contract.horizon_value,
        horizon_unit=contract.horizon_unit,
        session_boundary_rule=contract.session_boundary_rule,
        missing_bar_policy=contract.missing_bar_policy,
        calendar_authority_version=contract.calendar_authority_version,
        calendar_identity=calendar_identity,
        price_authority_version=contract.price_authority_version,
        target_code_hash=contract.target_code_hash,
        resolution_policy_version=contract.resolution_policy_version,
        no_future_data=no_future,
    )


def _nth_next_session(
    authority: ExchangeCalendarAuthority,
    day: date,
    count: int,
) -> date | None:
    current = day
    for _ in range(count):
        next_day = authority.next_session(current)
        if next_day is None:
            return None
        current = next_day
    return current


def _session_open_timestamp(authority: ExchangeCalendarAuthority, day: date) -> datetime:
    record = authority.session(day)
    if record.open_timestamp is None:
        raise TargetContractError(f"calendar has no open timestamp for {day}")
    return record.open_timestamp.astimezone(timezone.utc)


def _session_close_timestamp(authority: ExchangeCalendarAuthority, day: date) -> datetime:
    record = authority.session(day)
    if record.close_timestamp is None:
        raise TargetContractError(f"calendar has no close timestamp for {day}")
    return record.close_timestamp.astimezone(timezone.utc)


def _normalise_timeframe(value: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "1d": "1Day",
        "1day": "1Day",
        "daily": "1Day",
        "day": "1Day",
        "5min": "5m",
        "5minute": "5m",
        "5minutes": "5m",
        "5_minute": "5m",
        "5_min": "5m",
        "5m": "5m",
        "1hour": "1h",
        "hourly": "1h",
        "60m": "1h",
        "1h": "1h",
    }
    if text in aliases:
        return aliases[text]
    if value in {"1Day", "5m", "1h"}:
        return value
    raise TargetContractError(f"unsupported source bar timeframe: {value}")


def _timeframe_minutes(value: str) -> int:
    timeframe = _normalise_timeframe(value)
    if timeframe not in _TIMEFRAME_MINUTES:
        raise TargetContractError(f"timeframe {timeframe} has no intraday minute duration")
    return _TIMEFRAME_MINUTES[timeframe]


def _row_matches_asset(row: Mapping[str, Any], asset_id: str) -> bool:
    requested = str(asset_id).strip().upper()
    candidates = {
        str(row.get("asset_id") or "").strip().upper(),
        str(row.get("canonical_symbol") or "").strip().upper(),
        str(row.get("symbol") or "").strip().upper(),
        str(row.get("provider_symbol") or "").strip().upper(),
    }
    return requested in candidates


def _normalised_bar_matches_asset(bar: _NormalisedBar, asset_id: str) -> bool:
    requested = str(asset_id).strip().upper()
    return requested in {
        str(bar.asset_id or "").strip().upper(),
        str(bar.symbol or "").strip().upper(),
    }


def _max_final_timestamp(bars: Sequence[_NormalisedBar]) -> datetime | None:
    if not bars:
        return None
    return max(bar.final_timestamp for bar in bars)


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        raise TargetContractError("date value is required")
    return datetime.fromisoformat(text[:10]).date()


def _required_timestamp(value: Any, name: str) -> datetime:
    parsed = _parse_optional_timestamp(value)
    if parsed is None:
        raise TargetContractError(f"{name} timestamp is required")
    return parsed


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _float(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, date)):
        return _format_timestamp(value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc))
    return value


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


__all__ = [
    "HORIZON_ELAPSED_MARKET_MINUTES",
    "HORIZON_ELIGIBLE_BARS",
    "HORIZON_ELIGIBLE_TRADING_SESSIONS",
    "HORIZON_TO_NEXT_SESSION_OPEN",
    "HORIZON_TO_SESSION_CLOSE",
    "INELIGIBLE_SOURCE_BAR",
    "MATURED_VALID",
    "MISSING_SOURCE_BAR",
    "MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION",
    "NOT_YET_MATURE",
    "PRICE_AUTHORITY_VERSION",
    "QUARANTINED_SOURCE_BAR",
    "RIGHT_CENSORED",
    "SESSION_BOUNDARY_CONFLICT",
    "SUPPORTED_HORIZON_UNITS",
    "TARGET_CATALOGUE_VERSION",
    "TARGET_RESOLUTION_POLICY_VERSION",
    "TARGET_RESOLUTION_STATES",
    "TargetComputationResult",
    "TargetContract",
    "TargetContractError",
    "UNKNOWN_SOURCE_GAP",
    "build_target_manifest",
    "calculate_target",
    "canonical_hash",
    "require_explicit_target_contract",
    "resolve_target_contract",
    "target_catalogue",
    "target_catalogue_payload",
    "target_code_hash",
    "validate_target_availability",
]
