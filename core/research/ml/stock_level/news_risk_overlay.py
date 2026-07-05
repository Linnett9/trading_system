from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping


TIMESTAMP_COLUMNS = (
    "ingested_at",
    "available_at_utc",
    "available_at_timestamp",
    "available_at",
    "collected_at_utc",
    "first_seen_at",
    "published_at_utc",
    "published_at",
    "event_timestamp",
    "rebalance_date",
    "feature_date",
    "date",
)
DECISION_TIMESTAMP_COLUMNS = (
    "decision_timestamp",
    "rebalance_timestamp",
    "feature_timestamp",
    "timestamp",
    "rebalance_date",
    "feature_date",
    "date",
)
NON_FEATURE_COLUMNS = {
    "symbol",
    "event_id",
    "event_key",
    "headline",
    "title",
    "summary",
    "summary_or_text",
    "url",
    "url_or_accession",
    *TIMESTAMP_COLUMNS,
}


@dataclass(frozen=True)
class NewsRiskOverlayConfig:
    decision_timestamp_column: str | None = None
    news_timestamp_preference: tuple[str, ...] = TIMESTAMP_COLUMNS
    adverse_return_threshold: float = -0.05
    block_threshold: float = 0.70
    reduce_threshold: float = 0.50
    reduce_multiplier: float = 0.50
    model_version: str = "news-risk-overlay-mvp"


@dataclass(frozen=True)
class NewsRiskDecision:
    news_risk_probability: float | None
    news_coverage_status: str
    action: str
    recommended_position_multiplier: float
    model_version: str
    feature_timestamp: str | None
    diagnostics: dict[str, Any]


def join_news_to_stock_alpha_observations(
    stock_rows: Iterable[Mapping[str, Any]],
    news_rows: Iterable[Mapping[str, Any]],
    config: NewsRiskOverlayConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config or NewsRiskOverlayConfig()
    normalized_news = _prepare_news_rows(news_rows, settings)
    news_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_news:
        news_by_symbol.setdefault(row["symbol"], []).append(row)
    for rows in news_by_symbol.values():
        rows.sort(key=lambda item: item["effective_timestamp"])

    enriched: list[dict[str, Any]] = []
    future_rejected = 0
    no_coverage_rows = 0
    covered_symbols = set()
    stock_symbols = set()
    max_news_by_decision: list[dict[str, str | None]] = []
    decision_dates = set()
    news_dates = set()
    for raw_stock in stock_rows:
        stock = dict(raw_stock)
        symbol = _symbol(stock.get("symbol"))
        stock_symbols.add(symbol)
        decision_timestamp = _decision_timestamp(stock, settings)
        decision_dates.add(decision_timestamp.date().isoformat())
        candidate_news = news_by_symbol.get(symbol, [])
        future_rejected += sum(
            1 for row in candidate_news if row["effective_timestamp"] > decision_timestamp
        )
        eligible = [
            row for row in candidate_news if row["effective_timestamp"] <= decision_timestamp
        ]
        joined = _aggregate_news_features(eligible[-1:]) if eligible else {}
        if eligible:
            covered_symbols.add(symbol)
            latest = eligible[-1]["effective_timestamp"]
            news_dates.add(latest.date().isoformat())
            coverage = "COVERED"
        else:
            latest = None
            no_coverage_rows += 1
            coverage = "NO_COVERAGE"
        if latest and latest > decision_timestamp:
            raise ValueError("News leakage detected: joined news after decision timestamp")
        row = {
            **stock,
            **joined,
            "decision_timestamp": decision_timestamp.isoformat(),
            "news_feature_timestamp": latest.isoformat() if latest else "",
            "news_coverage_status": coverage,
            "news_missing_coverage": coverage == "NO_COVERAGE",
        }
        enriched.append(row)
        max_news_by_decision.append(
            {
                "symbol": symbol,
                "decision_timestamp": decision_timestamp.isoformat(),
                "max_news_timestamp": latest.isoformat() if latest else None,
            }
        )
    audit = {
        "future_news_rows_rejected": future_rejected,
        "rows_with_no_news_coverage": no_coverage_rows,
        "stock_row_count": len(enriched),
        "covered_row_count": len(enriched) - no_coverage_rows,
        "symbol_coverage": {
            "stock_symbol_count": len(stock_symbols),
            "covered_symbol_count": len(covered_symbols),
            "missing_symbol_count": len(stock_symbols - covered_symbols),
            "missing_symbols": sorted(stock_symbols - covered_symbols),
        },
        "date_coverage": {
            "decision_date_count": len(decision_dates),
            "covered_news_date_count": len(news_dates),
        },
        "max_news_timestamp_by_decision": max_news_by_decision,
        "leakage_violation_count": 0,
    }
    return enriched, audit


def build_news_risk_labels(
    rows: Iterable[Mapping[str, Any]],
    config: NewsRiskOverlayConfig | None = None,
) -> list[dict[str, Any]]:
    settings = config or NewsRiskOverlayConfig()
    output = []
    for row in rows:
        payload = dict(row)
        adverse = _first_number(
            payload,
            (
                "max_adverse_excursion",
                "forward_max_adverse_excursion",
                "actual_max_adverse_excursion",
            ),
        )
        forward_return = _first_number(
            payload,
            (
                "actual_forward_return_20d",
                "actual_forward_return_10d",
                "actual_forward_return_5d",
                "forward_return",
            ),
        )
        stop_hit = _boolish(payload.get("stop_hit_before_target"))
        risky = stop_hit or (
            adverse is not None and adverse <= settings.adverse_return_threshold
        ) or (
            forward_return is not None and forward_return <= settings.adverse_return_threshold
        )
        payload["news_risk_label"] = int(risky)
        output.append(payload)
    return output


def chronological_splits(
    rows: list[Mapping[str, Any]],
    *,
    folds: int = 3,
    embargo_days: int = 0,
) -> list[tuple[list[int], list[int]]]:
    dated = sorted(
        enumerate(rows),
        key=lambda item: _parse_timestamp(
            _first_present(item[1], DECISION_TIMESTAMP_COLUMNS)
        )
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    if folds < 1:
        raise ValueError("folds must be positive")
    if len(dated) < folds + 1:
        return []
    fold_size = max(1, len(dated) // (folds + 1))
    splits = []
    for fold in range(1, folds + 1):
        test_start = fold * fold_size
        test_end = len(dated) if fold == folds else min(len(dated), test_start + fold_size)
        test = dated[test_start:test_end]
        if not test:
            continue
        first_test_time = _row_time(test[0][1])
        train_cutoff = first_test_time - timedelta(days=embargo_days)
        train = [item for item in dated[:test_start] if _row_time(item[1]) < train_cutoff]
        splits.append(([index for index, _ in train], [index for index, _ in test]))
    return splits


def evaluate_candidate(
    *,
    symbol: str,
    decision_timestamp: datetime,
    base_position_size: float,
    price_model_score: float,
    recent_features: Mapping[str, Any],
    risk_probability: float | None,
    config: NewsRiskOverlayConfig | None = None,
) -> NewsRiskDecision:
    del symbol, decision_timestamp, base_position_size, price_model_score
    settings = config or NewsRiskOverlayConfig()
    coverage = str(recent_features.get("news_coverage_status") or "NO_COVERAGE")
    feature_timestamp = str(recent_features.get("news_feature_timestamp") or "") or None
    if coverage == "NO_COVERAGE" or risk_probability is None:
        return NewsRiskDecision(
            news_risk_probability=None,
            news_coverage_status="NO_COVERAGE",
            action="NO_COVERAGE",
            recommended_position_multiplier=1.0,
            model_version=settings.model_version,
            feature_timestamp=feature_timestamp,
            diagnostics={"reason": "missing_news_coverage"},
        )
    probability = float(max(0.0, min(1.0, risk_probability)))
    if probability >= settings.block_threshold:
        action = "BLOCK"
        multiplier = 0.0
    elif probability >= settings.reduce_threshold:
        action = "REDUCE"
        multiplier = settings.reduce_multiplier
    else:
        action = "ALLOW"
        multiplier = 1.0
    return NewsRiskDecision(
        news_risk_probability=probability,
        news_coverage_status=coverage,
        action=action,
        recommended_position_multiplier=multiplier,
        model_version=settings.model_version,
        feature_timestamp=feature_timestamp,
        diagnostics={
            "block_threshold": settings.block_threshold,
            "reduce_threshold": settings.reduce_threshold,
        },
    )


def shadow_decision_row(
    *,
    timestamp: datetime,
    symbol: str,
    price_score: float,
    price_only_position_size: float,
    decision: NewsRiskDecision,
    order_submitted: bool = False,
    relevant_news_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adjusted = price_only_position_size * decision.recommended_position_multiplier
    return {
        "timestamp": timestamp.isoformat(),
        "symbol": _symbol(symbol),
        "price_score": float(price_score),
        "price_only_position_size": float(price_only_position_size),
        "news_risk_probability": decision.news_risk_probability,
        "coverage_status": decision.news_coverage_status,
        "news_action": decision.action,
        "news_adjusted_position_size": adjusted,
        "relevant_news_features": dict(relevant_news_features or {}),
        "model_version": decision.model_version,
        "order_submitted": bool(order_submitted),
    }


def append_shadow_decision_log(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(materialized[0].keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(materialized)


def validate_news_paper_safety(
    *,
    paper_orders: bool,
    alpaca_endpoint: str,
    allow_env: str | None,
    readiness_ok: bool,
    leakage_ok: bool,
    model_loaded: bool,
    inputs_fresh: bool,
) -> None:
    endpoint = alpaca_endpoint.lower()
    if "paper" not in endpoint or "live" in endpoint:
        raise ValueError("News paper mode requires a paper Alpaca endpoint")
    blockers = []
    if not paper_orders:
        blockers.append("--paper-orders")
    if allow_env != "1":
        blockers.append("ALLOW_NEWS_PAPER_ORDERS=1")
    if not readiness_ok:
        blockers.append("successful news readiness audit")
    if not leakage_ok:
        blockers.append("successful timestamp-leakage audit")
    if not model_loaded:
        blockers.append("loaded and versioned model")
    if not inputs_fresh:
        blockers.append("non-stale price and news inputs")
    if blockers:
        raise ValueError("News paper mode blocked: " + ", ".join(blockers))


def evaluate_price_vs_news_overlay(
    rows: Iterable[Mapping[str, Any]],
    *,
    return_column: str = "actual_forward_return_10d",
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    control_returns = [_position_return(row, return_column, 1.0) for row in materialized]
    experiment_returns = [
        _position_return(
            row,
            return_column,
            float(row.get("news_position_multiplier", row.get("recommended_position_multiplier", 1.0)) or 0.0),
        )
        for row in materialized
    ]
    actions = [str(row.get("news_action") or "ALLOW") for row in materialized]
    return {
        "control": _portfolio_stats(control_returns),
        "experiment": _portfolio_stats(experiment_returns),
        "trades_allowed": actions.count("ALLOW"),
        "trades_reduced": actions.count("REDUCE"),
        "trades_blocked": actions.count("BLOCK"),
        "net_pnl_added_by_news_gate": sum(experiment_returns) - sum(control_returns),
    }


def _prepare_news_rows(
    rows: Iterable[Mapping[str, Any]],
    config: NewsRiskOverlayConfig,
) -> list[dict[str, Any]]:
    deduped = {}
    for row in rows:
        payload = dict(row)
        symbol = _symbol(payload.get("symbol"))
        timestamp = _effective_news_timestamp(payload, config)
        if not symbol or timestamp is None:
            continue
        event_id = str(payload.get("event_id") or payload.get("event_key") or "")
        key = (symbol, timestamp, event_id or repr(sorted(payload.items())))
        deduped[key] = {
            **payload,
            "symbol": symbol,
            "effective_timestamp": timestamp,
        }
    return list(deduped.values())


def _aggregate_news_features(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in NON_FEATURE_COLUMNS or key == "effective_timestamp":
                continue
            parsed = _number(value)
            if parsed is not None:
                values.setdefault(f"news_{key}", []).append(parsed)
    return {key: mean(items) for key, items in values.items()}


def _decision_timestamp(
    row: Mapping[str, Any],
    config: NewsRiskOverlayConfig,
) -> datetime:
    if config.decision_timestamp_column:
        parsed = _parse_timestamp(row.get(config.decision_timestamp_column))
        if parsed:
            return parsed
    parsed = _parse_timestamp(_first_present(row, DECISION_TIMESTAMP_COLUMNS))
    if parsed is None:
        raise ValueError("stock row is missing a decision timestamp")
    return parsed


def _effective_news_timestamp(
    row: Mapping[str, Any],
    config: NewsRiskOverlayConfig,
) -> datetime | None:
    for column in config.news_timestamp_preference:
        parsed = _parse_timestamp(row.get(column))
        if parsed:
            return parsed
    return None


def _row_time(row: Mapping[str, Any]) -> datetime:
    parsed = _parse_timestamp(_first_present(row, DECISION_TIMESTAMP_COLUMNS))
    if parsed is None:
        raise ValueError("row is missing a timestamp")
    return parsed


def _parse_timestamp(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if row.get(name) not in {None, ""}:
            return row.get(name)
    return None


def _first_number(row: Mapping[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _position_return(row: Mapping[str, Any], column: str, multiplier: float) -> float:
    value = _number(row.get(column)) or 0.0
    size = _number(row.get("price_only_position_size")) or 1.0
    return value * size * multiplier


def _portfolio_stats(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "total_return": 0.0,
            "hit_rate": 0.0,
            "average_gain": 0.0,
            "average_loss": 0.0,
        }
    gains = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    return {
        "total_return": sum(returns),
        "hit_rate": len(gains) / len(returns),
        "average_gain": mean(gains) if gains else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
    }


def decision_to_dict(decision: NewsRiskDecision) -> dict[str, Any]:
    return asdict(decision)
