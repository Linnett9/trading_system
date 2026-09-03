from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence


AUTHORITY_VERSION = "sequence_window_authority_v1"
_STABLE_SORTED = sorted
ENTITY_CONTEXT_KEYS = (
    "permanent_asset_id",
    "verified_entity_key",
    "entity_id",
    "asset_id",
    "canonical_asset_id",
    "symbol",
)
CHRONOLOGICAL_TIMESTAMP_KEYS = (
    "chronological_timestamp",
    "timestamp",
    "feature_timestamp",
    "decision_timestamp",
    "rebalance_date",
    "feature_date",
)
DECISION_TIMESTAMP_KEYS = (
    "decision_timestamp",
    "prediction_timestamp",
    "feature_timestamp",
    "rebalance_date",
    "feature_date",
)
FEATURE_CUTOFF_KEYS = (
    "feature_cutoff",
    "feature_timestamp",
    "rebalance_date",
    "feature_date",
)
SPLIT_KEYS = ("split", "split_identity", "split_role")


class SequenceAuthorityContextError(ValueError):
    """Raised when strict mode cannot prove authoritative sequence context."""


@dataclass(frozen=True)
class SequenceWindowConfig:
    window_length: int
    minimum_history: int | None = None
    maximum_allowed_gap: timedelta | float | int | None = None
    missing_bar_policy: str = "allow"
    duplicate_timestamp_policy: str = "reject"
    allow_ticker_change_with_stable_entity: bool = False
    strict_context_required: bool = False
    require_split_identity: bool = False
    authority_version: str = AUTHORITY_VERSION


@dataclass(frozen=True)
class SequenceWindow:
    indices: tuple[int, ...]
    sequence_id: str
    entity_id: str
    variant_policy_id: str
    horizon_id: str
    source_row_ids: tuple[str, ...]
    start_timestamp: str
    end_timestamp: str
    prediction_timestamp: str
    feature_cutoff: str
    split: str
    gap_diagnostics: dict[str, Any]
    authority_version: str
    deterministic_lineage_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceWindowBuildResult:
    windows: tuple[SequenceWindow, ...]
    rejected_windows: tuple[dict[str, Any], ...]


def sequence_context_rows_from_metadata(
    metadata: Sequence[Mapping[str, Any]] | None,
    sample_count: int,
    *,
    feature_dates: Sequence[Any] | None = None,
    feature_ids: Sequence[Any] | None = None,
    label_start_dates: Sequence[Any] | None = None,
    label_end_dates: Sequence[Any] | None = None,
    split_identity: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        source = dict(metadata[index]) if metadata and index < len(metadata) else {}
        feature_date = (
            feature_dates[index]
            if feature_dates is not None and index < len(feature_dates)
            else source.get("feature_date")
        )
        row_id = (
            feature_ids[index]
            if feature_ids is not None and index < len(feature_ids)
            else source.get("row_id")
            or source.get("feature_id")
            or index
        )
        context_fields = {
            "entity": _has_any(source, ENTITY_CONTEXT_KEYS),
            "chronological_timestamp": (
                _has_any(source, CHRONOLOGICAL_TIMESTAMP_KEYS)
                or feature_date is not None
            ),
            "decision_timestamp": (
                _has_any(source, DECISION_TIMESTAMP_KEYS)
                or feature_date is not None
            ),
            "feature_cutoff": (
                _has_any(source, FEATURE_CUTOFF_KEYS)
                or feature_date is not None
            ),
            "split": _has_any(source, SPLIT_KEYS) or split_identity is not None,
        }
        row = {
            **source,
            "row_id": str(row_id),
            "source_index": index,
            "chronological_timestamp": _first_present(
                source,
                CHRONOLOGICAL_TIMESTAMP_KEYS,
                feature_date if feature_date is not None else index,
            ),
            "decision_timestamp": _first_present(
                source,
                DECISION_TIMESTAMP_KEYS,
                feature_date if feature_date is not None else index,
            ),
            "feature_cutoff": _first_present(
                source,
                FEATURE_CUTOFF_KEYS,
                feature_date if feature_date is not None else index,
            ),
            "target_availability_timestamp": _first_present(
                source,
                (
                    "target_availability_timestamp",
                    "label_available_timestamp",
                    "label_end_timestamp",
                    "label_end_date",
                ),
                label_end_dates[index]
                if label_end_dates is not None and index < len(label_end_dates)
                else None,
            ),
            "target_start_timestamp": _first_present(
                source,
                ("label_start_timestamp", "label_start_date", "target_start_timestamp"),
                label_start_dates[index]
                if label_start_dates is not None and index < len(label_start_dates)
                else None,
            ),
            "split": _first_present(
                source,
                SPLIT_KEYS,
                split_identity or "default",
            ),
            "_sequence_authority_context_fields": context_fields,
        }
        rows.append(row)
    return rows


def build_authoritative_sequence_windows(
    rows: Sequence[Mapping[str, Any]],
    config: SequenceWindowConfig,
) -> SequenceWindowBuildResult:
    if config.window_length < 2:
        raise ValueError("window_length must be at least two")
    minimum_history = config.minimum_history or config.window_length
    if minimum_history < config.window_length:
        raise ValueError("minimum_history must be >= window_length")
    if config.missing_bar_policy not in {"allow", "exclude", "reject"}:
        raise ValueError("missing_bar_policy must be one of: allow, exclude, reject")
    if config.duplicate_timestamp_policy not in {"reject", "allow"}:
        raise ValueError("duplicate_timestamp_policy must be one of: reject, allow")
    if config.strict_context_required:
        _validate_strict_context_rows(
            rows,
            require_split_identity=config.require_split_identity,
        )

    normalized = [_normalise_row(row, index) for index, row in enumerate(rows)]
    grouped: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    base_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in normalized:
        base_groups.setdefault(
            (
                row["entity_id"],
                row["variant_policy_id"],
                row["horizon_id"],
                row["split"],
            ),
            [],
        ).append(row)

    for base_key, group_rows in base_groups.items():
        ordered = _STABLE_SORTED(
            group_rows,
            key=lambda row: (row["timestamp_sort_key"], row["row_id"], row["source_index"]),
        )
        segment = 0
        previous: dict[str, Any] | None = None
        for row in ordered:
            if _starts_new_segment(
                previous,
                row,
                allow_ticker_change_with_stable_entity=(
                    config.allow_ticker_change_with_stable_entity
                ),
            ):
                segment += 1
            grouped.setdefault((*base_key, segment), []).append(row)
            previous = row

    windows: list[SequenceWindow] = []
    for group_key, group_rows in _STABLE_SORTED(grouped.items(), key=lambda item: item[0]):
        timestamps = [row["timestamp_identity"] for row in group_rows]
        if (
            config.duplicate_timestamp_policy == "reject"
            and len(timestamps) != len(set(timestamps))
        ):
            rejected.append(
                _rejection(
                    "DUPLICATE_TIMESTAMP",
                    group_rows,
                    group_key=group_key,
                )
            )
            continue
        if len(group_rows) < minimum_history:
            rejected.append(
                _rejection(
                    "REQUIRED_HISTORY_INCOMPLETE",
                    group_rows,
                    group_key=group_key,
                )
            )
            continue
        for end_offset in range(config.window_length - 1, len(group_rows)):
            candidate = group_rows[
                end_offset - config.window_length + 1 : end_offset + 1
            ]
            reason = _candidate_rejection_reason(candidate, config)
            if reason:
                rejected.append(_rejection(reason, candidate, group_key=group_key))
                continue
            windows.append(_window_from_candidate(candidate, config))
    windows.sort(key=lambda window: (window.end_timestamp, window.entity_id, window.sequence_id))
    return SequenceWindowBuildResult(tuple(windows), tuple(rejected))


def build_sequence_indices_from_context(
    context_rows: Sequence[Mapping[str, Any]] | None,
    sample_count: int,
    sequence_length: int,
    *,
    maximum_allowed_gap: timedelta | float | int | None = None,
    missing_bar_policy: str = "allow",
    duplicate_timestamp_policy: str = "reject",
    allow_ticker_change_with_stable_entity: bool = False,
    strict_context_required: bool = False,
    require_split_identity: bool = False,
) -> list[list[int]]:
    has_complete_context = (
        context_rows is not None and len(context_rows) == sample_count
    )
    if not has_complete_context and strict_context_required:
        found = 0 if context_rows is None else len(context_rows)
        raise SequenceAuthorityContextError(
            "sequence authority strict_context_required=True requires "
            f"context rows for every sample; expected={sample_count}; found={found}"
        )
    rows = (
        list(context_rows)
        if has_complete_context
        else _legacy_global_context_rows(sample_count)
    )
    result = build_authoritative_sequence_windows(
        rows,
        SequenceWindowConfig(
            window_length=sequence_length,
            maximum_allowed_gap=maximum_allowed_gap,
            missing_bar_policy=missing_bar_policy,
            duplicate_timestamp_policy=duplicate_timestamp_policy,
            allow_ticker_change_with_stable_entity=allow_ticker_change_with_stable_entity,
            strict_context_required=strict_context_required,
            require_split_identity=require_split_identity,
        ),
    )
    return [list(window.indices) for window in result.windows]


def sequence_metadata_from_window(window: SequenceWindow) -> dict[str, Any]:
    return dict(window.metadata)


def _legacy_global_context_rows(sample_count: int) -> list[dict[str, Any]]:
    return [
        {
            "row_id": str(index),
            "source_index": index,
            "sequence_group_id": "global",
            "chronological_timestamp": index,
            "decision_timestamp": index,
            "feature_cutoff": index,
        }
        for index in range(sample_count)
    ]


def _validate_strict_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_split_identity: bool,
) -> None:
    missing: list[str] = []
    split_is_applicable = require_split_identity or any(
        _strict_field_present(row, "split", SPLIT_KEYS) for row in rows
    )
    for index, row in enumerate(rows):
        required_fields = {
            "entity": _strict_field_present(row, "entity", ENTITY_CONTEXT_KEYS),
            "chronological_timestamp": _strict_field_present(
                row,
                "chronological_timestamp",
                CHRONOLOGICAL_TIMESTAMP_KEYS,
            ),
            "decision_timestamp": _strict_field_present(
                row,
                "decision_timestamp",
                DECISION_TIMESTAMP_KEYS,
            ),
            "feature_cutoff": _strict_field_present(
                row,
                "feature_cutoff",
                FEATURE_CUTOFF_KEYS,
            ),
        }
        if split_is_applicable:
            required_fields["split"] = _strict_field_present(
                row,
                "split",
                SPLIT_KEYS,
            )
        missing.extend(
            f"row {index} missing {field}"
            for field, present in required_fields.items()
            if not present
        )
    if missing:
        preview = "; ".join(missing[:6])
        suffix = f"; additional_missing={len(missing) - 6}" if len(missing) > 6 else ""
        raise SequenceAuthorityContextError(
            "sequence authority strict_context_required=True requires "
            "entity identity, timestamps, and feature cutoff for every row; "
            f"{preview}{suffix}"
        )


def _strict_field_present(
    row: Mapping[str, Any],
    field: str,
    aliases: Sequence[str],
) -> bool:
    context_fields = row.get("_sequence_authority_context_fields")
    if isinstance(context_fields, Mapping) and field in context_fields:
        return bool(context_fields[field])
    return _has_any(row, aliases)


def _normalise_row(row: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    timestamp = _first_present(
        row,
        (
            "chronological_timestamp",
            "timestamp",
            "feature_timestamp",
            "decision_timestamp",
            "rebalance_date",
            "feature_date",
        ),
        source_index,
    )
    decision_timestamp = _first_present(
        row,
        ("decision_timestamp", "prediction_timestamp", "rebalance_date", "feature_date"),
        timestamp,
    )
    feature_cutoff = _first_present(
        row,
        ("feature_cutoff", "feature_timestamp", "rebalance_date", "feature_date"),
        timestamp,
    )
    entity_id = _entity_id(row)
    variant_policy_id = _variant_policy_id(row)
    horizon_id = str(
        _first_present(
            row,
            (
                "forecast_horizon",
                "horizon_id",
                "target_horizon",
                "target_observation_count",
            ),
            "default",
        )
    )
    split = str(
        _first_present(row, ("split", "split_identity", "split_role"), "default")
    )
    return {
        "source": dict(row),
        "source_index": int(row.get("source_index", source_index)),
        "row_id": str(row.get("row_id") or row.get("feature_id") or source_index),
        "entity_id": entity_id,
        "symbol": str(row.get("symbol") or row.get("ticker") or ""),
        "variant_policy_id": variant_policy_id,
        "horizon_id": horizon_id,
        "split": split,
        "corporate_identity_id": str(
            _first_present(
                row,
                ("corporate_identity_id", "corporate_identity", "legal_entity_id"),
                entity_id,
            )
        ),
        "timestamp_raw": timestamp,
        "timestamp_identity": _timestamp_identity(timestamp),
        "timestamp_sort_key": _timestamp_sort_key(timestamp),
        "decision_timestamp": decision_timestamp,
        "feature_cutoff": feature_cutoff,
        "target_start_timestamp": _first_present(
            row,
            ("label_start_timestamp", "label_start_date", "target_start_timestamp"),
            None,
        ),
        "target_availability_timestamp": _first_present(
            row,
            (
                "target_availability_timestamp",
                "label_available_timestamp",
                "label_end_timestamp",
                "label_end_date",
            ),
            None,
        ),
        "missing_bar": _truthy(
            _first_present(
                row,
                ("missing_bar", "is_missing_bar", "missing", "bar_missing"),
                False,
            )
        )
        or str(row.get("bar_status", "")).lower() in {"missing", "gap"},
        "corporate_break": _truthy(
            _first_present(
                row,
                (
                    "corporate_identity_break",
                    "corporate_identity_discontinuity",
                    "identity_continuity_break",
                ),
                False,
            )
        ),
    }


def _first_present(
    row: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _has_any(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(row.get(key) not in (None, "") for key in keys)


def _entity_id(row: Mapping[str, Any]) -> str:
    return str(
        _first_present(
            row,
            (
                "permanent_asset_id",
                "verified_entity_key",
                "entity_id",
                "asset_id",
                "canonical_asset_id",
                "symbol",
                "sequence_group_id",
            ),
            "global",
        )
    )


def _variant_policy_id(row: Mapping[str, Any]) -> str:
    policy = _first_present(
        row,
        ("policy_id", "portfolio_policy_id", "strategy_id", "model_policy_id"),
        "default_policy",
    )
    variant = _first_present(
        row,
        (
            "variant_id",
            "strategy_variant_id",
            "research_variant_id",
            "variant_universe",
        ),
        "default_variant",
    )
    return f"{policy}|{variant}"


def _starts_new_segment(
    previous: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    *,
    allow_ticker_change_with_stable_entity: bool,
) -> bool:
    if previous is None:
        return False
    if row["corporate_break"]:
        return True
    if row["corporate_identity_id"] != previous["corporate_identity_id"]:
        return True
    previous_symbol = str(previous.get("symbol") or "")
    current_symbol = str(row.get("symbol") or "")
    if (
        previous_symbol
        and current_symbol
        and previous_symbol != current_symbol
        and not allow_ticker_change_with_stable_entity
    ):
        return True
    return False


def _candidate_rejection_reason(
    rows: Sequence[Mapping[str, Any]],
    config: SequenceWindowConfig,
) -> str | None:
    if len({row["entity_id"] for row in rows}) != 1:
        return "MULTIPLE_ENTITIES"
    if len({row["variant_policy_id"] for row in rows}) != 1:
        return "MULTIPLE_VARIANTS_OR_POLICIES"
    if len({row["horizon_id"] for row in rows}) != 1:
        return "MULTIPLE_HORIZONS"
    if len({row["split"] for row in rows}) != 1:
        return "MULTIPLE_SPLITS"
    if any(row["missing_bar"] for row in rows) and config.missing_bar_policy in {
        "exclude",
        "reject",
    }:
        return "MISSING_BAR"
    if any(row["corporate_break"] for row in rows[1:]):
        return "CORPORATE_IDENTITY_DISCONTINUITY"
    timestamps = [row["timestamp_sort_key"] for row in rows]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        return "TIMESTAMPS_NOT_STRICTLY_ORDERED"
    if _gap_exceeded(rows, config.maximum_allowed_gap):
        return "MAXIMUM_ALLOWED_GAP_EXCEEDED"
    end = rows[-1]
    prediction_sort_key = _timestamp_sort_key(end["decision_timestamp"])
    if any(
        _timestamp_sort_key(row["feature_cutoff"]) > prediction_sort_key
        for row in rows
    ):
        return "FEATURE_CUTOFF_AFTER_PREDICTION_TIME"
    for row in rows:
        target_boundary = row["target_start_timestamp"] or row[
            "target_availability_timestamp"
        ]
        if target_boundary is not None and _timestamp_sort_key(
            row["feature_cutoff"]
        ) >= _timestamp_sort_key(target_boundary):
            return "TARGET_INFORMATION_IN_FEATURE_WINDOW"
    return None


def _gap_exceeded(
    rows: Sequence[Mapping[str, Any]],
    maximum_allowed_gap: timedelta | float | int | None,
) -> bool:
    if maximum_allowed_gap is None:
        return False
    for left, right in zip(rows, rows[1:]):
        gap = _gap_between(left["timestamp_raw"], right["timestamp_raw"])
        if gap is None:
            return True
        if isinstance(gap, timedelta):
            limit = (
                maximum_allowed_gap
                if isinstance(maximum_allowed_gap, timedelta)
                else timedelta(days=float(maximum_allowed_gap))
            )
            if gap > limit:
                return True
        else:
            limit_number = (
                maximum_allowed_gap.total_seconds()
                if isinstance(maximum_allowed_gap, timedelta)
                else float(maximum_allowed_gap)
            )
            if gap > limit_number:
                return True
    return False


def _window_from_candidate(
    rows: Sequence[Mapping[str, Any]],
    config: SequenceWindowConfig,
) -> SequenceWindow:
    source_row_ids = tuple(str(row["row_id"]) for row in rows)
    indices = tuple(int(row["source_index"]) for row in rows)
    gap_diagnostics = _gap_diagnostics(rows, config.maximum_allowed_gap)
    metadata_identity = {
        "authority_version": config.authority_version,
        "entity_id": rows[-1]["entity_id"],
        "variant_policy_id": rows[-1]["variant_policy_id"],
        "horizon_id": rows[-1]["horizon_id"],
        "split": rows[-1]["split"],
        "source_row_ids": list(source_row_ids),
        "start_timestamp": rows[0]["timestamp_identity"],
        "end_timestamp": rows[-1]["timestamp_identity"],
        "prediction_timestamp": _timestamp_identity(rows[-1]["decision_timestamp"]),
        "feature_cutoff": _timestamp_identity(rows[-1]["feature_cutoff"]),
        "window_length": config.window_length,
        "minimum_history": config.minimum_history or config.window_length,
        "missing_bar_policy": config.missing_bar_policy,
        "duplicate_timestamp_policy": config.duplicate_timestamp_policy,
        "maximum_allowed_gap": str(config.maximum_allowed_gap),
    }
    lineage = canonical_hash(metadata_identity)
    sequence_id = "seq_" + canonical_hash(
        {
            "authority_version": config.authority_version,
            "entity_id": rows[-1]["entity_id"],
            "variant_policy_id": rows[-1]["variant_policy_id"],
            "horizon_id": rows[-1]["horizon_id"],
            "source_row_ids": list(source_row_ids),
        }
    )[:24]
    metadata = {
        **metadata_identity,
        "sequence_id": sequence_id,
        "source_indices": list(indices),
        "gap_diagnostics": gap_diagnostics,
        "deterministic_lineage_hash": lineage,
    }
    return SequenceWindow(
        indices=indices,
        sequence_id=sequence_id,
        entity_id=str(rows[-1]["entity_id"]),
        variant_policy_id=str(rows[-1]["variant_policy_id"]),
        horizon_id=str(rows[-1]["horizon_id"]),
        source_row_ids=source_row_ids,
        start_timestamp=str(rows[0]["timestamp_identity"]),
        end_timestamp=str(rows[-1]["timestamp_identity"]),
        prediction_timestamp=str(_timestamp_identity(rows[-1]["decision_timestamp"])),
        feature_cutoff=str(_timestamp_identity(rows[-1]["feature_cutoff"])),
        split=str(rows[-1]["split"]),
        gap_diagnostics=gap_diagnostics,
        authority_version=config.authority_version,
        deterministic_lineage_hash=lineage,
        metadata=metadata,
    )


def _gap_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    maximum_allowed_gap: timedelta | float | int | None,
) -> dict[str, Any]:
    gaps = [_gap_between(left["timestamp_raw"], right["timestamp_raw"]) for left, right in zip(rows, rows[1:])]
    rendered = [
        gap.total_seconds() if isinstance(gap, timedelta) else gap
        for gap in gaps
    ]
    exceeded = _gap_exceeded(rows, maximum_allowed_gap)
    return {
        "gap_count": len(gaps),
        "observed_gaps": rendered,
        "maximum_allowed_gap": str(maximum_allowed_gap),
        "maximum_allowed_gap_exceeded": exceeded,
    }


def _rejection(
    reason: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: Sequence[Any],
) -> dict[str, Any]:
    return {
        "reason": reason,
        "group_key": [str(value) for value in group_key],
        "source_row_ids": [str(row["row_id"]) for row in rows],
        "source_indices": [int(row["source_index"]) for row in rows],
    }


def _timestamp_sort_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    parsed = _parse_datetime(value)
    if parsed is not None:
        return (1, parsed.timestamp())
    return (2, str(value))


def _timestamp_identity(value: Any) -> str:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.isoformat()
    return str(value)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _gap_between(left: Any, right: Any) -> timedelta | float | None:
    left_datetime = _parse_datetime(left)
    right_datetime = _parse_datetime(right)
    if left_datetime is not None and right_datetime is not None:
        return right_datetime - left_datetime
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(right) - float(left)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()
