from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import NewsTransformerConfig


TEXT_MISSING = "headline/article text missing"
AVAILABILITY_MISSING = "availability timestamps missing"
PUBLICATION_MISSING = "publication timestamps missing"
DUPLICATES_MISSING = "duplicate/syndication grouping missing"
EVENT_LABELS_MISSING = "event labels missing"
SPLIT_MISSING = "chronological split missing"
LABELS_MISSING = "labels missing"
VALIDATION_INCOMPLETE = "validation spine incomplete"
WALK_FORWARD_MISSING = "walk-forward not implemented"
PLACEBO_MISSING = "placebo not implemented"
MATCHED_CONTROLS_MISSING = "matched controls not implemented"
AUDITS_MISSING = "corporate-action/survivorship/missing-news audits incomplete"


def validate_news_sequence_schema(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    if not rows:
        return [TEXT_MISSING, AVAILABILITY_MISSING, PUBLICATION_MISSING, DUPLICATES_MISSING, EVENT_LABELS_MISSING, SPLIT_MISSING, LABELS_MISSING]
    if not any(_text_available(row) for row in rows):
        failures.append(TEXT_MISSING)
    if not all(row.get("availability_timestamp") for row in rows):
        failures.append(AVAILABILITY_MISSING)
    if not all(row.get("publication_timestamp") for row in rows):
        failures.append(PUBLICATION_MISSING)
    if not all(row.get("duplicate_group_id") for row in rows):
        failures.append(DUPLICATES_MISSING)
    if not all(row.get("event_category") for row in rows):
        failures.append(EVENT_LABELS_MISSING)
    if not validate_no_random_split(rows):
        failures.append(SPLIT_MISSING)
    if not validate_label_readiness(rows):
        failures.append(LABELS_MISSING)
    return failures


def validate_point_in_time_text_fields(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("availability_timestamp") for row in rows)


def validate_no_random_split(rows: Sequence[Mapping[str, Any]]) -> bool:
    split_names = {str(row.get("split_name", "")).lower() for row in rows}
    if not split_names or "" in split_names:
        return False
    return not any("random" in split_name for split_name in split_names)


def validate_duplicate_grouping_readiness(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("duplicate_group_id") for row in rows)


def validate_label_readiness(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        row.get("label_name") and row.get("label_horizon_days") is not None and row.get("label_value") is not None
        for row in rows
    )


def build_news_transformer_readiness_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    validation_spine_complete: bool = False,
    walk_forward_complete: bool = False,
    placebo_complete: bool = False,
    matched_controls_complete: bool = False,
    audits_complete: bool = False,
    config: NewsTransformerConfig | None = None,
) -> dict[str, Any]:
    config = config or NewsTransformerConfig()
    failures = validate_news_sequence_schema(rows or [])
    if not validation_spine_complete:
        failures.append(VALIDATION_INCOMPLETE)
    if not walk_forward_complete:
        failures.append(WALK_FORWARD_MISSING)
    if not placebo_complete:
        failures.append(PLACEBO_MISSING)
    if not matched_controls_complete:
        failures.append(MATCHED_CONTROLS_MISSING)
    if not audits_complete:
        failures.append(AUDITS_MISSING)
    return {
        "schema_name": "stock_alpha_news_transformer_readiness",
        "schema_version": 1,
        "status": "NOT_READY",
        "transformer_readiness": "NOT_READY",
        "bert_readiness": "NOT_READY",
        "finbert_readiness": "NOT_READY",
        "enabled": False,
        "training_enabled": False,
        "inference_enabled": False,
        "used_in_strategy": False,
        "used_in_replay": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "failures": sorted(set(failures)),
        "warnings": [
            "Disabled research-only scaffold; no training or inference is performed.",
            "Availability timestamp, duplicate grouping, chronological split, and future labels are required before modelling.",
        ],
        "config": {
            "lookback_window_days": config.lookback_window_days,
            "label_horizon_days": config.label_horizon_days,
            "enabled": config.enabled,
            "training_enabled": config.training_enabled,
            "inference_enabled": config.inference_enabled,
            "used_in_strategy": config.used_in_strategy,
            "used_in_replay": config.used_in_replay,
        },
    }


def _text_available(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("headline_text") or row.get("headline") or "").strip()) or bool(
        str(row.get("article_text") or row.get("text") or "").strip()
    )
