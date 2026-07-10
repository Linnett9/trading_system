from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Mapping

from core.research.ml.stock_level.news_risk_overlay import (
    DECISION_TIMESTAMP_COLUMNS,
    NewsRiskOverlayConfig,
    evaluate_candidate,
    shadow_decision_row,
)


def apply_probabilities(rows: list[dict[str, Any]], probabilities: Mapping[int, float], column: str) -> None:
    for index, value in probabilities.items():
        rows[index][column] = value


def assign_candidate_ids(rows: list[dict[str, Any]], price_score_column: str) -> None:
    for index, row in enumerate(rows):
        decision_timestamp = str(row.get("decision_timestamp", row.get("rebalance_date", "")))
        symbol = str(row.get("symbol", "")).upper()
        row.setdefault("decision_timestamp", decision_timestamp)
        row.setdefault("model_version", str(row.get("source_model_type") or row.get("source_model_version") or "news-risk-overlay-research-v1"))
        payload = "|".join(
            [
                decision_timestamp,
                symbol,
                str(row.get(price_score_column, "")),
                str(row.get("price_plus_news_risk_probability", "")),
                str(row.get("model_version", "")),
                str(index),
            ]
        )
        row["candidate_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_news_decisions(
    rows: list[dict[str, Any]],
    config: NewsRiskOverlayConfig,
    price_score_column: str,
) -> list[dict[str, Any]]:
    decision_rows = []
    for row in rows:
        probability = _number(row.get("price_plus_news_risk_probability"))
        decision = evaluate_candidate(
            symbol=str(row.get("symbol", "")),
            decision_timestamp=_timestamp(row),
            base_position_size=1.0,
            price_model_score=_number(row.get(price_score_column)) or 0.0,
            recent_features=row,
            risk_probability=probability,
            config=config,
        )
        row["news_action"] = decision.action
        row["news_position_multiplier"] = decision.recommended_position_multiplier
        decision_rows.append(
            shadow_decision_row(
                timestamp=_timestamp(row),
                symbol=str(row.get("symbol", "")),
                price_score=_number(row.get(price_score_column)) or 0.0,
                price_only_position_size=1.0,
                decision=decision,
                order_submitted=False,
                relevant_news_features={key: row[key] for key in row if key.startswith("news_")},
            )
        )
    return decision_rows


def _timestamp(row: Mapping[str, Any]) -> datetime:
    for column in ("decision_timestamp", *DECISION_TIMESTAMP_COLUMNS):
        value = row.get(column)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    raise ValueError("row missing decision timestamp")


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
