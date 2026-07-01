from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter


LABEL_COLUMNS = ("actual_forward_return_5d", "actual_forward_return_10d", "forward_return_5d", "forward_return_10d")


@dataclass(frozen=True)
class StockAlphaNewsSourceDiagnosticsPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_source_diagnostics(config: Mapping[str, Any]) -> StockAlphaNewsSourceDiagnosticsPaths:
    payload = build_stock_alpha_news_source_diagnostics(config)
    output = _path(config, "stock_alpha_news_source_diagnostics_report_dir")
    paths = StockAlphaNewsSourceDiagnosticsPaths(
        output / "stock_alpha_news_source_diagnostics.json",
        output / "stock_alpha_news_source_diagnostics.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_source_diagnostics(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    contract_path = _path(config, "stock_alpha_news_contract_path")
    stock_path = _path(config, "stock_alpha_stock_rows_path")
    source_column = str(ml.get("stock_alpha_news_source_column", "source"))
    provider_column = str(ml.get("stock_alpha_news_provider_column", "provider"))
    contract_exists, stock_exists = contract_path.is_file(), stock_path.is_file()
    news = CsvRowRepository().read(contract_path) if contract_exists else []
    stocks = CsvRowRepository().read(stock_path) if stock_exists else []
    columns = set(news[0]) if news else set()
    identifier_column = provider_column if provider_column in columns else source_column
    blocking = []
    if not contract_exists:
        blocking.append("news_contract_file_not_found")
    elif not news:
        blocking.append("news_contract_file_empty")
    if not stock_exists:
        blocking.append("stock_rows_file_not_found")
    elif not stocks:
        blocking.append("stock_rows_file_empty")
    if news and identifier_column not in columns:
        blocking.append("source_identifier_column_missing")

    normalized, timestamp_quality = _normalize_news(news, identifier_column)
    if timestamp_quality["invalid_timestamp_count"] or timestamp_quality["ingested_before_published_count"]:
        blocking.append("timestamp_leakage_detected")
    alignment = _align(normalized, stocks)
    if alignment["included_future_article_count"]:
        blocking.append("timestamp_leakage_detected")
    quality = _quality(normalized)
    profiles = _profiles(normalized)
    coverage = _source_coverage(normalized, stocks)
    agreement = _agreement(alignment["windows"])
    bias = _bias(normalized)
    correlations = _label_correlations(alignment["windows"], stocks)
    warnings = []
    if quality["sources_with_duplicate_headlines"]:
        warnings.append("source_duplication_detected")
    if bias["dominated_sources"] or bias["dominant_source"]:
        warnings.append("source_bias_detected")
    next_action = _next_action(blocking, warnings)
    sources = {row["source_id"] for row in normalized if row["source_id"]}
    return {
        "next_action": next_action,
        "blocking_issues": list(dict.fromkeys(blocking)),
        "warning_issues": warnings,
        "input_status": {
            "news_contract_path": str(contract_path),
            "stock_rows_path": str(stock_path),
            "news_contract_exists": contract_exists,
            "stock_rows_exists": stock_exists,
            "news_row_count": len(news),
            "stock_row_count": len(stocks),
            "news_symbol_count": _unique(news, "symbol"),
            "stock_symbol_count": _unique(stocks, "symbol"),
            "rebalance_date_count": _unique(stocks, "rebalance_date"),
            "source_provider_count": len(sources),
            "source_identifier_column": identifier_column,
            "provider_column_present": provider_column in columns,
        },
        "source_coverage": coverage,
        "source_quality": {**quality, **timestamp_quality},
        "source_sentiment_event_profile": profiles,
        "cross_source_agreement": agreement,
        "bias_diagnostics": bias,
        "pit_safety": {
            "future_article_candidate_count": alignment["future_article_candidate_count"],
            "future_article_excluded_count": alignment["future_article_excluded_count"],
            "included_future_article_count": alignment["included_future_article_count"],
            "published_at_utc_lte_rebalance_date": True,
            "ingested_at_lte_rebalance_date": True,
        },
        "exploratory_label_relationship": correlations,
        "inspection_only": True,
        "features_generated": False,
        "files_ingested": False,
        "readiness_invoked": False,
        "model_training_invoked": False,
        "diagnostics_invoked": False,
        "news_transformer_enabled": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _normalize_news(rows: list[Mapping[str, Any]], source_column: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    result, invalid, reversed_count = [], 0, 0
    for row in rows:
        published, ingested = _date(row.get("published_at_utc")), _date(row.get("ingested_at"))
        if published is None or ingested is None:
            invalid += 1
        elif ingested < published:
            reversed_count += 1
        result.append({
            **dict(row), "source_id": str(row.get(source_column, "")).strip(),
            "symbol": str(row.get("symbol", "")).strip().upper(),
            "published": published, "ingested": ingested,
            "sentiment": _float(row.get("sentiment_score")),
            "relevance": _float(row.get("relevance_score")),
            "novelty": _float(row.get("novelty_score")),
        })
    return result, {"invalid_timestamp_count": invalid, "ingested_before_published_count": reversed_count}


def _align(news: list[Mapping[str, Any]], stocks: list[Mapping[str, Any]]) -> dict[str, Any]:
    windows, future = [], 0
    for stock in stocks:
        symbol, rebalance = str(stock.get("symbol", "")).strip().upper(), _date(stock.get("rebalance_date"))
        if not symbol or rebalance is None:
            continue
        eligible = []
        for article in news:
            if article.get("symbol") != symbol or article.get("published") is None or article.get("ingested") is None:
                continue
            if article["published"] <= rebalance and article["ingested"] <= rebalance:
                if article["published"] >= rebalance - timedelta(days=30):
                    eligible.append(article)
            else:
                future += 1
        windows.append({"symbol": symbol, "rebalance_date": rebalance.date().isoformat(), "articles": eligible})
    return {"windows": windows, "future_article_candidate_count": future, "future_article_excluded_count": future, "included_future_article_count": 0}


def _source_coverage(news: list[Mapping[str, Any]], stocks: list[Mapping[str, Any]]) -> dict[str, Any]:
    sources = sorted({str(row.get("source_id", "")) for row in news if row.get("source_id")})
    total_stock = len(stocks)
    result = {}
    for source in sources:
        articles = [row for row in news if row.get("source_id") == source]
        covered = 0
        for stock in stocks:
            symbol, date = str(stock.get("symbol", "")).upper(), _date(stock.get("rebalance_date"))
            if date and any(row.get("symbol") == symbol and row.get("published") and row.get("ingested") and row["published"] <= date and row["ingested"] <= date for row in articles):
                covered += 1
        result[source] = {
            "article_count": len(articles),
            "symbols_covered": sorted({row["symbol"] for row in articles if row.get("symbol")}),
            "publication_dates_covered": sorted({row["published"].date().isoformat() for row in articles if row.get("published")}),
            "stock_row_coverage_rate": covered / total_stock if total_stock else 0.0,
        }
    counts = [value["article_count"] for value in result.values()]
    return {"by_source": result, "sparse_sources": [key for key, value in result.items() if value["stock_row_coverage_rate"] < 0.2], "dominant_sources": [key for key, value in result.items() if news and value["article_count"] / len(news) > 0.75], "article_count_range": [min(counts), max(counts)] if counts else []}


def _quality(news: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for source in sorted({row.get("source_id") for row in news if row.get("source_id")}):
        rows = [row for row in news if row.get("source_id") == source]
        ids = [str(row.get("article_id", "")).strip() for row in rows]
        headlines = [str(row.get("headline", "")).strip().lower() for row in rows]
        delays = [(row["ingested"] - row["published"]).total_seconds() / 60 for row in rows if row.get("published") and row.get("ingested")]
        result[str(source)] = {
            "duplicate_article_id_count": len(ids) - len(set(ids)),
            "duplicate_headline_count": len(headlines) - len(set(headlines)),
            "missing_headline_rate": sum(not value for value in headlines) / len(rows) if rows else 0.0,
            "missing_body_rate": sum(not str(row.get("body_or_summary", "")).strip() for row in rows) / len(rows) if rows else 0.0,
            "publication_to_ingestion_delay_minutes": _summary(delays),
            "language_counts": _counts(str(row.get("language", "")).lower() for row in rows),
        }
    return {"by_source": result, "sources_with_duplicate_headlines": [source for source, value in result.items() if value["duplicate_headline_count"]]}


def _profiles(news: list[Mapping[str, Any]]) -> dict[str, Any]:
    result, extremes = {}, []
    for source in sorted({row.get("source_id") for row in news if row.get("source_id")}):
        rows = [row for row in news if row.get("source_id") == source]
        sentiments = [row["sentiment"] for row in rows if row.get("sentiment") is not None]
        result[str(source)] = {
            "average_sentiment": mean(sentiments) if sentiments else None,
            "sentiment_missing_rate": (len(rows) - len(sentiments)) / len(rows) if rows else 0.0,
            "relevance_summary": _summary([row["relevance"] for row in rows if row.get("relevance") is not None]),
            "novelty_summary": _summary([row["novelty"] for row in rows if row.get("novelty") is not None]),
            "event_type_counts": _counts(str(row.get("event_type", "")) for row in rows),
        }
        extremes.extend(sorted(({"source": source, "symbol": row.get("symbol"), "sentiment": row["sentiment"], "article_id": row.get("article_id")} for row in rows if row.get("sentiment") is not None), key=lambda item: abs(item["sentiment"]), reverse=True)[:2])
    return {"by_source": result, "extreme_sentiment_rows": extremes}


def _agreement(windows: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for window in windows:
        by_source: dict[str, list[float]] = {}
        for article in window["articles"]:
            if article.get("source_id") and article.get("sentiment") is not None:
                by_source.setdefault(article["source_id"], []).append(article["sentiment"])
        values = [mean(items) for items in by_source.values()]
        if len(values) >= 2:
            consensus = mean(values)
            disagreement = max(values) - min(values)
            rows.append({"symbol": window["symbol"], "rebalance_date": window["rebalance_date"], "source_count": len(values), "consensus_sentiment": consensus, "sentiment_disagreement_score": disagreement, "sentiment_agreement_score": max(0.0, 1.0 - min(1.0, disagreement / 2.0))})
    return {"multi_source_window_count": len(rows), "windows": rows, "high_disagreement_symbols_dates": [row for row in rows if row["sentiment_disagreement_score"] >= 0.75]}


def _bias(news: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_source = {}
    total = len(news)
    source_counts = _counts(str(row.get("source_id", "")) for row in news)
    for source in source_counts:
        rows = [row for row in news if row.get("source_id") == source]
        dimensions = {"symbol": _counts(row.get("symbol") for row in rows), "date": _counts(row["published"].date().isoformat() for row in rows if row.get("published")), "event_type": _counts(str(row.get("event_type", "")) for row in rows)}
        by_source[source] = {name: {"counts": counts, "max_concentration": max(counts.values()) / len(rows) if counts else 0.0} for name, counts in dimensions.items()}
    return {"by_source": by_source, "dominated_sources": [source for source, dims in by_source.items() if any(value["max_concentration"] > 0.8 for value in dims.values())], "dominant_source": max(source_counts, key=source_counts.get) if source_counts and max(source_counts.values()) / total > 0.75 else None}


def _label_correlations(windows: list[Mapping[str, Any]], stocks: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [column for column in LABEL_COLUMNS if stocks and column in stocks[0]]
    if not labels:
        return {"status": "skipped_labels_absent", "exploratory_only": True, "in_sample_only": True, "correlations": {}}
    label = labels[0]
    stock_map = {(str(row.get("rebalance_date", ""))[:10], str(row.get("symbol", "")).upper()): _float(row.get(label)) for row in stocks}
    pairs: dict[str, list[tuple[float, float]]] = {}
    for window in windows:
        target = stock_map.get((window["rebalance_date"], window["symbol"]))
        by_source: dict[str, list[float]] = {}
        for article in window["articles"]:
            if article.get("sentiment") is not None:
                by_source.setdefault(article.get("source_id", ""), []).append(article["sentiment"])
        if target is not None:
            for source, values in by_source.items():
                pairs.setdefault(source, []).append((mean(values), target))
    return {"status": "computed", "label_column": label, "exploratory_only": True, "in_sample_only": True, "no_alpha_claim": True, "correlations": {source: _correlation(values) for source, values in pairs.items()}}


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs); xm, ym = mean(xs), mean(ys)
    denominator = math.sqrt(sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys))
    return sum((x-xm)*(y-ym) for x, y in pairs) / denominator if denominator else None


def _next_action(blocking: list[str], warnings: list[str]) -> str:
    if "news_contract_file_not_found" in blocking: return "provide_news_contract"
    if "stock_rows_file_not_found" in blocking: return "provide_stock_rows"
    if "timestamp_leakage_detected" in blocking: return "fix_timestamp_leakage"
    if "source_duplication_detected" in warnings: return "investigate_source_duplication"
    if "source_bias_detected" in warnings: return "investigate_source_bias"
    return "ready_for_source_specific_feature_design"


def _path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value: raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _date(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    except (TypeError, ValueError): return None


def _float(value: Any) -> float | None:
    try: return float(value) if str(value).strip() else None
    except (TypeError, ValueError): return None


def _summary(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "min": min(values) if values else None, "max": max(values) if values else None, "mean": mean(values) if values else None}


def _counts(values: Any) -> dict[str, int]:
    result = {}
    for value in values:
        if value: result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def _unique(rows: list[Mapping[str, Any]], key: str) -> int:
    return len({str(row.get(key, "")).strip().upper() for row in rows if str(row.get(key, "")).strip()})


def _markdown(payload: Mapping[str, Any]) -> str:
    status, pit = payload["input_status"], payload["pit_safety"]
    return "\n".join(["# Stock-Alpha News Source Diagnostics", "", f"- Next action: {payload['next_action']}", f"- News rows: {status['news_row_count']}", f"- Sources/providers: {status['source_provider_count']}", f"- Future articles correctly excluded: {pit['future_article_excluded_count']}", f"- Included future articles: {pit['included_future_article_count']}", "- Inspection only: true", "- Features generated: false", "- Model training invoked: false", "- Model diagnostics invoked: false", "", "Exploratory, in-sample source attribution only. No alpha claim is made."])
