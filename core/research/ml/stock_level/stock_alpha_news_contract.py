from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


REQUIRED_NEWS_FIELDS = (
    "article_id",
    "symbol",
    "published_at_utc",
    "ingested_at",
    "source",
    "headline",
    "body_or_summary",
    "sentiment_score",
    "relevance_score",
    "novelty_score",
    "event_type",
    "language",
)
REQUIRED_NEWS_AGGREGATE_FEATURES = (
    "news_count_1d",
    "news_count_3d",
    "news_count_7d",
    "avg_sentiment_1d",
    "avg_sentiment_3d",
    "sentiment_change_3d",
    "negative_news_count_7d",
    "earnings_news_count_14d",
    "analyst_news_count_14d",
    "guidance_news_count_30d",
    "litigation_news_count_30d",
    "mna_news_count_30d",
    "news_volume_zscore",
)
GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}


@dataclass(frozen=True)
class NewsContractValidation:
    available: bool
    reason: str
    required_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    row_count: int
    symbol_coverage: float
    date_coverage: float
    future_article_count: int
    contract_valid: bool = False
    transformer_enabled: bool = False
    aggregate_features_valid: bool = False
    missing_aggregate_features: tuple[str, ...] = ()
    invalid_article_id_count: int = 0
    invalid_symbol_count: int = 0
    invalid_published_at_count: int = 0
    invalid_ingested_at_count: int = 0
    synthetic_zero_news_count: int = 0
    symbols_covered: tuple[str, ...] = ()
    rebalance_dates_covered: tuple[str, ...] = ()
    missing_symbol_coverage: tuple[str, ...] = ()
    missing_date_coverage: tuple[str, ...] = ()
    article_counts_by_window: Mapping[str, int] | None = None
    coverage_rows: tuple[Mapping[str, Any], ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "required_fields": list(self.required_fields),
            "missing_fields": list(self.missing_fields),
            "row_count": self.row_count,
            "symbol_coverage": self.symbol_coverage,
            "date_coverage": self.date_coverage,
            "future_article_count": self.future_article_count,
            "contract_valid": self.contract_valid,
            "transformer_enabled": self.transformer_enabled,
            "aggregate_features_valid": self.aggregate_features_valid,
            "required_aggregate_features": list(REQUIRED_NEWS_AGGREGATE_FEATURES),
            "missing_aggregate_features": list(self.missing_aggregate_features),
            "invalid_article_id_count": self.invalid_article_id_count,
            "invalid_symbol_count": self.invalid_symbol_count,
            "invalid_published_at_count": self.invalid_published_at_count,
            "invalid_ingested_at_count": self.invalid_ingested_at_count,
            "synthetic_zero_news_count": self.synthetic_zero_news_count,
            "symbols_covered": list(self.symbols_covered),
            "rebalance_dates_covered": list(self.rebalance_dates_covered),
            "missing_symbol_coverage": list(self.missing_symbol_coverage),
            "missing_date_coverage": list(self.missing_date_coverage),
            "article_counts_by_window": dict(self.article_counts_by_window or {}),
            **GUARDRAILS,
        }


@dataclass(frozen=True)
class NewsContractValidationPaths:
    json_path: Path
    markdown_path: Path
    coverage_csv_path: Path


@dataclass(frozen=True)
class NewsFeatureAggregationPaths:
    features_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path


def write_stock_alpha_news_features(
    config: Mapping[str, Any],
    stock_rows: list[Mapping[str, Any]],
) -> NewsFeatureAggregationPaths:
    ml = dict(config.get("ml", {}) or {})
    _validate_research_guardrails(ml)
    _validate_stock_rows(stock_rows)
    validation = validate_news_contract(config, stock_rows)
    if not validation.contract_valid:
        raise ValueError(f"cannot aggregate stock-alpha news features: {validation.reason}")
    contract_path = Path(str(_ml_value(ml, "stock_alpha_news_contract_path", "stock_news_contract_path")))
    news_rows = [_normalize_news_row(row) for row in CsvRowRepository().read(contract_path)]
    feature_rows, audit = build_stock_alpha_news_features(news_rows, stock_rows)
    output = stock_alpha_output_dir(config) / "news_features"
    configured_features_path = ml.get("stock_alpha_news_features_path")
    features_path = Path(str(configured_features_path)) if configured_features_path else output / "stock_alpha_news_features.csv"
    paths = NewsFeatureAggregationPaths(
        features_csv_path=features_path,
        audit_json_path=output / "stock_alpha_news_features_audit.json",
        audit_markdown_path=output / "stock_alpha_news_features_audit.md",
    )
    audit.update(
        {
            "source_news_path": str(contract_path),
            "stock_rows_path": str(ml.get("stock_alpha_news_stock_rows_path", ml.get("stock_level_prediction_artifacts_path", ""))),
            "output_features_path": str(paths.features_csv_path),
            "news_contract_path": str(contract_path),
            "news_features_path": str(paths.features_csv_path),
            "coverage_summary": validation.payload(),
            "transformer_available": bool(
                ml.get("stock_alpha_news_enable_transformer", False)
                and validation.available
            ),
            "news_analysis_transformer_available": False,
            "news_analysis_transformer_status": "unavailable_until_features_pass_coverage_gates",
            **GUARDRAILS,
        }
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(
        paths.features_csv_path,
        feature_rows,
        fieldnames=[
            "rebalance_date",
            "symbol",
            *REQUIRED_NEWS_AGGREGATE_FEATURES,
            "news_has_coverage_30d",
            "news_article_count_30d",
        ],
    )
    writer.write_json(paths.audit_json_path, audit)
    writer.write_markdown(paths.audit_markdown_path, _feature_markdown(audit))
    return paths


def write_stock_alpha_news_features_from_config(
    config: Mapping[str, Any],
) -> NewsFeatureAggregationPaths:
    ml = dict(config.get("ml", {}) or {})
    rows_path = _ml_value(
        ml,
        "stock_alpha_news_stock_rows_path",
        "stock_level_prediction_artifacts_path",
    )
    if not rows_path:
        raise ValueError(
            "ml.stock_alpha_news_stock_rows_path or ml.stock_level_prediction_artifacts_path "
            "must point to stock rows with rebalance_date and symbol"
        )
    path = Path(str(rows_path))
    if not path.exists():
        raise ValueError(f"stock-alpha news feature stock rows file not found: {path}")
    stock_rows = CsvRowRepository().read(path)
    if not stock_rows:
        raise ValueError(f"stock-alpha news feature stock rows file is empty: {path}")
    _validate_stock_rows(stock_rows)
    return write_stock_alpha_news_features(config, stock_rows)


def build_stock_alpha_news_features(
    news_rows: list[Mapping[str, Any]],
    stock_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_news = [_normalize_news_row(row) for row in news_rows]
    invalid_timestamps = sum(
        1
        for row in normalized_news
        if row.get("published_at_utc") is None or row.get("ingested_at") is None
    )
    if invalid_timestamps:
        raise ValueError("news feature aggregation requires valid published_at_utc and ingested_at timestamps")
    rows: list[dict[str, Any]] = []
    prior_counts_by_symbol: dict[str, list[float]] = {}
    future_article_count = 0
    for stock_row in sorted(stock_rows, key=lambda row: (str(row.get("symbol", "")).upper(), str(row.get("rebalance_date", "")))):
        symbol = str(stock_row.get("symbol", "")).strip().upper()
        rebalance = _parse_date(stock_row.get("rebalance_date"))
        if not symbol or rebalance is None:
            continue
        windows = {
            "1d": _window_articles(normalized_news, symbol, rebalance, 1),
            "3d": _window_articles(normalized_news, symbol, rebalance, 3),
            "7d": _window_articles(normalized_news, symbol, rebalance, 7),
            "14d": _window_articles(normalized_news, symbol, rebalance, 14),
            "30d": _window_articles(normalized_news, symbol, rebalance, 30),
        }
        future_article_count += sum(
            1
            for row in normalized_news
            if row.get("symbol") == symbol
            and (row["published_at_utc"] > rebalance or row["ingested_at"] > rebalance)
        )
        count_7d = float(len(windows["7d"]))
        previous_counts = prior_counts_by_symbol.setdefault(symbol, [])
        zscore = _zscore_against_history(count_7d, previous_counts)
        previous_counts.append(count_7d)
        row = {
            "rebalance_date": rebalance.date().isoformat(),
            "symbol": symbol,
            "news_count_1d": len(windows["1d"]),
            "news_count_3d": len(windows["3d"]),
            "news_count_7d": len(windows["7d"]),
            "avg_sentiment_1d": _average_sentiment(windows["1d"]),
            "avg_sentiment_3d": _average_sentiment(windows["3d"]),
            "sentiment_change_3d": _sentiment_change_3d(windows["3d"], rebalance),
            "negative_news_count_7d": _event_count(windows["7d"], negative=True),
            "earnings_news_count_14d": _event_count(windows["14d"], event_markers=("earnings",)),
            "analyst_news_count_14d": _event_count(windows["14d"], event_markers=("analyst",)),
            "guidance_news_count_30d": _event_count(windows["30d"], event_markers=("guidance",)),
            "litigation_news_count_30d": _event_count(windows["30d"], event_markers=("litigation", "lawsuit", "legal")),
            "mna_news_count_30d": _event_count(windows["30d"], event_markers=("mna", "m&a", "merger", "acquisition")),
            "news_volume_zscore": zscore,
            "news_has_coverage_30d": bool(windows["30d"]),
            "news_article_count_30d": len(windows["30d"]),
        }
        rows.append(row)
    audit = {
        "row_count": len(rows),
        "stock_row_count": len(stock_rows),
        "feature_row_count": len(rows),
        "news_article_count": len(normalized_news),
        "article_count": len(normalized_news),
        "symbol_count": len({row["symbol"] for row in rows}),
        "rebalance_date_count": len({row["rebalance_date"] for row in rows}),
        "invalid_timestamp_count": invalid_timestamps,
        "future_article_candidate_count": future_article_count,
        "pit_violations_count": future_article_count,
        "missing_or_invalid_timestamp_count": invalid_timestamps,
        "feature_columns": list(REQUIRED_NEWS_AGGREGATE_FEATURES),
        "coverage_row_count": sum(1 for row in rows if row["news_has_coverage_30d"]),
        "missing_coverage_row_count": sum(1 for row in rows if not row["news_has_coverage_30d"]),
        "synthetic_news_features_created": False,
        "point_in_time_filters": {
            "published_at_utc_lte_rebalance_date": True,
            "ingested_at_lte_rebalance_date": True,
            "future_statistics_used": False,
        },
    }
    return rows, audit


def _validate_research_guardrails(ml: Mapping[str, Any]) -> None:
    failures = [
        key
        for key, expected in GUARDRAILS.items()
        if ml.get(key) != expected
    ]
    if failures:
        raise ValueError(
            "stock-alpha news feature generation requires research-only guardrails: "
            + ", ".join(failures)
        )


def _validate_stock_rows(stock_rows: list[Mapping[str, Any]]) -> None:
    if not stock_rows:
        raise ValueError("stock-alpha news feature stock rows are empty")
    required = ("rebalance_date", "symbol")
    missing = [field for field in required if field not in stock_rows[0]]
    if missing:
        raise ValueError(
            "stock-alpha news feature stock rows missing required fields: "
            + ", ".join(missing)
        )


def write_stock_alpha_news_contract_validation(
    config: Mapping[str, Any],
    stock_rows: list[Mapping[str, Any]],
) -> NewsContractValidationPaths:
    validation = validate_news_contract(config, stock_rows)
    output = stock_alpha_output_dir(config) / "news_contract"
    paths = NewsContractValidationPaths(
        json_path=output / "news_contract_validation.json",
        markdown_path=output / "news_contract_validation.md",
        coverage_csv_path=output / "news_contract_coverage.csv",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, validation.payload())
    writer.write_markdown(paths.markdown_path, _markdown(validation))
    writer.write_csv(
        paths.coverage_csv_path,
        validation.coverage_rows,
        fieldnames=[
            "rebalance_date",
            "symbol",
            "article_count_1d",
            "article_count_3d",
            "article_count_7d",
            "article_count_14d",
            "article_count_30d",
        ],
    )
    return paths


def validate_news_contract(
    config: Mapping[str, Any],
    stock_rows: list[Mapping[str, Any]],
) -> NewsContractValidation:
    ml = dict(config.get("ml", {}) or {})
    raw_path = _ml_value(ml, "stock_alpha_news_contract_path", "stock_news_contract_path")
    transformer_enabled = bool(ml.get("stock_alpha_news_enable_transformer", False))
    min_symbol = float(_ml_value(ml, "stock_alpha_news_min_symbol_coverage", "stock_news_min_symbol_coverage", default=0.80))
    min_date = float(_ml_value(ml, "stock_alpha_news_min_date_coverage", "stock_news_min_date_coverage", default=0.80))
    if not raw_path:
        return _unavailable("missing ml.stock_alpha_news_contract_path", transformer_enabled=transformer_enabled)
    path = Path(str(raw_path))
    if not path.exists():
        return _unavailable(f"news contract file not found: {path}", transformer_enabled=transformer_enabled)
    rows = CsvRowRepository().read(path)
    if not rows:
        return _unavailable("news contract file is empty", transformer_enabled=transformer_enabled)
    missing = tuple(field for field in REQUIRED_NEWS_FIELDS if field not in rows[0])
    if missing:
        return NewsContractValidation(False, "missing required news contract fields", REQUIRED_NEWS_FIELDS, missing, len(rows), 0.0, 0.0, 0, transformer_enabled=transformer_enabled)

    normalized_rows = [_normalize_news_row(row) for row in rows]
    invalid_article_ids = sum(1 for row in normalized_rows if not row["article_id"])
    invalid_symbols = sum(1 for row in normalized_rows if not row["symbol"])
    invalid_published = sum(1 for row in normalized_rows if row["published_at_utc"] is None)
    invalid_ingested = sum(1 for row in normalized_rows if row["ingested_at"] is None)
    synthetic = sum(1 for row in normalized_rows if _looks_like_synthetic_zero_news(row))
    future = _future_article_count(normalized_rows, stock_rows)
    if invalid_article_ids:
        return _invalid("news contract contains empty article_id", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, invalid_article_ids=invalid_article_ids)
    if invalid_symbols:
        return _invalid("news contract contains empty symbol", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, invalid_symbols=invalid_symbols)
    if invalid_published:
        return _invalid("news contract contains invalid published_at_utc", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, invalid_published=invalid_published)
    if invalid_ingested:
        return _invalid("news contract contains invalid ingested_at", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, invalid_ingested=invalid_ingested)
    if synthetic:
        return _invalid("news contract contains synthetic zero-news fake coverage", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, synthetic=synthetic)
    if future:
        return _invalid("news contract contains future articles", normalized_rows, stock_rows, min_symbol, min_date, transformer_enabled, future=future)

    coverage = _coverage_summary(normalized_rows, stock_rows, min_symbol, min_date)
    if coverage["symbol_coverage"] < min_symbol:
        return _from_coverage("news contract symbol coverage below minimum", normalized_rows, coverage, transformer_enabled)
    if coverage["date_coverage"] < min_date:
        return _from_coverage("news contract date coverage below minimum", normalized_rows, coverage, transformer_enabled)

    aggregate_valid, missing_aggregate = _validate_aggregate_features(ml)
    contract_valid = True
    if transformer_enabled and not aggregate_valid:
        return _from_coverage(
            "missing required news aggregate features",
            normalized_rows,
            coverage,
            transformer_enabled,
            contract_valid=contract_valid,
            aggregate_features_valid=False,
            missing_aggregate_features=missing_aggregate,
        )
    if not transformer_enabled:
        return _from_coverage(
            "news_analysis_transformer unavailable: missing valid point-in-time news contract or transformer disabled",
            normalized_rows,
            coverage,
            transformer_enabled,
            contract_valid=contract_valid,
            aggregate_features_valid=aggregate_valid,
            missing_aggregate_features=missing_aggregate,
        )
    return _from_coverage(
        "news contract satisfied",
        normalized_rows,
        coverage,
        transformer_enabled,
        available=True,
        contract_valid=contract_valid,
        aggregate_features_valid=aggregate_valid,
    )


def _ml_value(ml: Mapping[str, Any], primary: str, legacy: str, *, default: Any = None) -> Any:
    return ml.get(primary, ml.get(legacy, default))


def _validate_aggregate_features(ml: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    path_value = ml.get("stock_alpha_news_features_path")
    if not path_value:
        return False, REQUIRED_NEWS_AGGREGATE_FEATURES
    path = Path(str(path_value))
    if not path.exists():
        return False, REQUIRED_NEWS_AGGREGATE_FEATURES
    rows = CsvRowRepository().read(path)
    if not rows:
        return False, REQUIRED_NEWS_AGGREGATE_FEATURES
    missing = tuple(column for column in REQUIRED_NEWS_AGGREGATE_FEATURES if column not in rows[0])
    return not missing, missing


def _unavailable(reason: str, *, transformer_enabled: bool = False) -> NewsContractValidation:
    return NewsContractValidation(False, reason, REQUIRED_NEWS_FIELDS, REQUIRED_NEWS_FIELDS, 0, 0.0, 0.0, 0, transformer_enabled=transformer_enabled)


def _invalid(
    reason: str,
    rows: list[dict[str, Any]],
    stock_rows: list[Mapping[str, Any]],
    min_symbol: float,
    min_date: float,
    transformer_enabled: bool,
    *,
    invalid_article_ids: int = 0,
    invalid_symbols: int = 0,
    invalid_published: int = 0,
    invalid_ingested: int = 0,
    synthetic: int = 0,
    future: int = 0,
) -> NewsContractValidation:
    coverage = _coverage_summary(rows, stock_rows, min_symbol, min_date)
    return _from_coverage(
        reason,
        rows,
        coverage,
        transformer_enabled,
        invalid_article_id_count=invalid_article_ids,
        invalid_symbol_count=invalid_symbols,
        invalid_published_at_count=invalid_published,
        invalid_ingested_at_count=invalid_ingested,
        synthetic_zero_news_count=synthetic,
        future_article_count=future,
    )


def _from_coverage(
    reason: str,
    rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    transformer_enabled: bool,
    *,
    available: bool = False,
    contract_valid: bool = False,
    aggregate_features_valid: bool = False,
    missing_aggregate_features: tuple[str, ...] = (),
    invalid_article_id_count: int = 0,
    invalid_symbol_count: int = 0,
    invalid_published_at_count: int = 0,
    invalid_ingested_at_count: int = 0,
    synthetic_zero_news_count: int = 0,
    future_article_count: int = 0,
) -> NewsContractValidation:
    return NewsContractValidation(
        available,
        reason,
        REQUIRED_NEWS_FIELDS,
        (),
        len(rows),
        coverage["symbol_coverage"],
        coverage["date_coverage"],
        future_article_count,
        contract_valid=contract_valid,
        transformer_enabled=transformer_enabled,
        aggregate_features_valid=aggregate_features_valid,
        missing_aggregate_features=missing_aggregate_features,
        invalid_article_id_count=invalid_article_id_count,
        invalid_symbol_count=invalid_symbol_count,
        invalid_published_at_count=invalid_published_at_count,
        invalid_ingested_at_count=invalid_ingested_at_count,
        synthetic_zero_news_count=synthetic_zero_news_count,
        symbols_covered=tuple(coverage["symbols_covered"]),
        rebalance_dates_covered=tuple(coverage["rebalance_dates_covered"]),
        missing_symbol_coverage=tuple(coverage["missing_symbol_coverage"]),
        missing_date_coverage=tuple(coverage["missing_date_coverage"]),
        article_counts_by_window=coverage["article_counts_by_window"],
        coverage_rows=tuple(coverage["coverage_rows"]),
    )


def _coverage_summary(
    news_rows: list[dict[str, Any]],
    stock_rows: list[Mapping[str, Any]],
    min_symbol: float,
    min_date: float,
) -> dict[str, Any]:
    del min_symbol, min_date
    stock_symbols = sorted({str(row.get("symbol", "")).upper() for row in stock_rows if str(row.get("symbol", ""))})
    stock_dates = sorted({str(row.get("rebalance_date", ""))[:10] for row in stock_rows if str(row.get("rebalance_date", ""))})
    news_symbols = sorted({row["symbol"] for row in news_rows if row.get("symbol")})
    coverage_rows = []
    covered_symbols: set[str] = set()
    covered_dates: set[str] = set()
    counts_by_window = {"1d": 0, "3d": 0, "7d": 0, "14d": 0, "30d": 0}
    for stock_row in stock_rows:
        symbol = str(stock_row.get("symbol", "")).upper()
        rebalance = _parse_date(stock_row.get("rebalance_date"))
        if not symbol or rebalance is None:
            continue
        counts = {window: _article_count(news_rows, symbol, rebalance, days) for window, days in {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30}.items()}
        for window, count in counts.items():
            counts_by_window[window] += count
        if counts["30d"] > 0:
            covered_symbols.add(symbol)
            covered_dates.add(rebalance.date().isoformat())
        coverage_rows.append(
            {
                "rebalance_date": rebalance.date().isoformat(),
                "symbol": symbol,
                "article_count_1d": counts["1d"],
                "article_count_3d": counts["3d"],
                "article_count_7d": counts["7d"],
                "article_count_14d": counts["14d"],
                "article_count_30d": counts["30d"],
            }
        )
    return {
        "symbol_coverage": len(set(stock_symbols) & covered_symbols) / len(stock_symbols) if stock_symbols else 0.0,
        "date_coverage": len(set(stock_dates) & covered_dates) / len(stock_dates) if stock_dates else 0.0,
        "symbols_covered": sorted(covered_symbols),
        "rebalance_dates_covered": sorted(covered_dates),
        "missing_symbol_coverage": sorted(set(stock_symbols) - covered_symbols),
        "missing_date_coverage": sorted(set(stock_dates) - covered_dates),
        "article_counts_by_window": counts_by_window,
        "coverage_rows": coverage_rows,
        "news_symbols": news_symbols,
    }


def _article_count(news_rows: list[dict[str, Any]], symbol: str, rebalance: datetime, days: int) -> int:
    start = rebalance - timedelta(days=days)
    return sum(
        1
        for row in news_rows
        if row.get("symbol") == symbol
        and row.get("published_at_utc") is not None
        and row.get("ingested_at") is not None
        and start <= row["published_at_utc"] <= rebalance
        and row["ingested_at"] <= rebalance
    )


def _window_articles(
    news_rows: list[dict[str, Any]],
    symbol: str,
    rebalance: datetime,
    days: int,
) -> list[dict[str, Any]]:
    start = rebalance - timedelta(days=days)
    return [
        row
        for row in news_rows
        if row.get("symbol") == symbol
        and row.get("published_at_utc") is not None
        and row.get("ingested_at") is not None
        and start <= row["published_at_utc"] <= rebalance
        and row["ingested_at"] <= rebalance
    ]


def _average_sentiment(rows: list[Mapping[str, Any]]) -> float | str:
    values = [_float_or_none(row.get("sentiment_score")) for row in rows]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else ""


def _sentiment_change_3d(rows: list[dict[str, Any]], rebalance: datetime) -> float | str:
    if not rows:
        return ""
    midpoint = rebalance - timedelta(days=1.5)
    recent = [row for row in rows if row["published_at_utc"] > midpoint]
    earlier = [row for row in rows if row["published_at_utc"] <= midpoint]
    recent_average = _average_sentiment(recent)
    earlier_average = _average_sentiment(earlier)
    if recent_average == "" or earlier_average == "":
        return ""
    return float(recent_average) - float(earlier_average)


def _event_count(
    rows: list[Mapping[str, Any]],
    *,
    event_markers: tuple[str, ...] = (),
    negative: bool = False,
) -> int:
    count = 0
    for row in rows:
        event_type = str(row.get("event_type", "")).lower()
        sentiment = _float_or_none(row.get("sentiment_score"))
        if event_markers and any(marker in event_type for marker in event_markers):
            count += 1
        elif negative and sentiment is not None and sentiment < 0:
            count += 1
    return count


def _zscore_against_history(value: float, history: list[float]) -> float | str:
    if len(history) < 2:
        return ""
    mean_value = sum(history) / len(history)
    variance = sum((item - mean_value) ** 2 for item in history) / len(history)
    std = variance ** 0.5
    if std == 0:
        return ""
    return (value - mean_value) / std


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _future_article_count(news_rows: list[dict[str, Any]], stock_rows: list[Mapping[str, Any]]) -> int:
    max_rebalance = max((_parse_date(row.get("rebalance_date")) for row in stock_rows), default=None)
    if max_rebalance is None:
        return 0
    return sum(
        1
        for row in news_rows
        if row.get("published_at_utc") is None
        or row.get("ingested_at") is None
        or row["published_at_utc"] > max_rebalance
        or row["ingested_at"] > max_rebalance
    )


def _normalize_news_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "article_id": str(row.get("article_id", "")).strip(),
        "symbol": str(row.get("symbol", "")).strip().upper(),
        "published_at_utc": _parse_datetime(row.get("published_at_utc")),
        "ingested_at": _parse_datetime(row.get("ingested_at")),
    }


def _looks_like_synthetic_zero_news(row: Mapping[str, Any]) -> bool:
    text = " ".join(str(row.get(column, "")).lower() for column in ("article_id", "source", "headline", "body_or_summary", "event_type"))
    if any(marker in text for marker in ("synthetic", "zero_news", "no_news", "fake_neutral", "placeholder")):
        return True
    numeric = [row.get(column) for column in ("sentiment_score", "relevance_score", "novelty_score")]
    return all(str(value).strip() in {"0", "0.0", ""} for value in numeric) and not str(row.get("headline", "")).strip()


def _parse_date(value: Any) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _markdown(validation: NewsContractValidation) -> str:
    payload = validation.payload()
    return "\n".join(
        [
            "# Stock Alpha News Contract Validation",
            "",
            "Research only. Trading impact: none. Production validated: false.",
            "",
            f"- Available to news_analysis_transformer: {payload['available']}",
            f"- Reason: {payload['reason']}",
            f"- Row count: {payload['row_count']}",
            f"- Symbol coverage: {payload['symbol_coverage']}",
            f"- Date coverage: {payload['date_coverage']}",
            f"- Future article count: {payload['future_article_count']}",
            f"- Synthetic zero-news count: {payload['synthetic_zero_news_count']}",
            f"- Missing aggregate features: {payload['missing_aggregate_features']}",
            "- Synthetic news features created: false",
            "- Promotion thresholds changed: false",
            "",
        ]
    )


def _feature_markdown(audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stock Alpha News Feature Aggregation",
            "",
            "Research only. Trading impact: none. Production validated: false.",
            "",
            f"- News contract: {audit['news_contract_path']}",
            f"- News features: {audit['news_features_path']}",
            f"- Feature rows: {audit['row_count']}",
            f"- News articles: {audit['news_article_count']}",
            f"- Rows with 30d news coverage: {audit['coverage_row_count']}",
            f"- Rows missing 30d news coverage: {audit['missing_coverage_row_count']}",
            f"- Future article candidates excluded by PIT filters: {audit['future_article_candidate_count']}",
            "- published_at_utc <= rebalance_date enforced: true",
            "- ingested_at <= rebalance_date enforced: true",
            "- Synthetic news features created: false",
            f"- news_analysis_transformer available: {audit['news_analysis_transformer_available']}",
            f"- news_analysis_transformer status: {audit['news_analysis_transformer_status']}",
            "- Promotion thresholds changed: false",
            "",
        ]
    )
