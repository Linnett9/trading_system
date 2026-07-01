from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config.config_loader import load_config
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter


@dataclass(frozen=True)
class StockAlphaNewsCollectionPlanPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_collection_plan(config: Mapping[str, Any]) -> StockAlphaNewsCollectionPlanPaths:
    payload = build_stock_alpha_news_collection_plan(config)
    output = _path(config, "stock_alpha_news_collection_plan_report_dir")
    paths = StockAlphaNewsCollectionPlanPaths(
        output / "stock_alpha_news_collection_plan.json",
        output / "stock_alpha_news_collection_plan.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_collection_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    stock_path = _path(config, "stock_alpha_stock_rows_path")
    audit_config_path = _path(config, "stock_alpha_news_provider_audit_config_path")
    collect = dict(ml.get("stock_alpha_news_collect", {}) or {})
    configured_symbols = _symbols(collect.get("symbols", []))
    stock_rows = CsvRowRepository().read(stock_path) if stock_path.is_file() else []
    stock_symbols = _symbols(row.get("symbol", "") for row in stock_rows)
    audit_config = load_config(str(audit_config_path), overlay_project_config=True)
    audit_ml = dict(audit_config.get("ml", {}) or {})
    raw_path = Path(str(audit_ml.get("stock_alpha_news_raw_path", "")))
    raw_rows = CsvRowRepository().read(raw_path) if raw_path.is_file() else []
    raw_symbols = _symbols(row.get("symbol", "") for row in raw_rows)
    min_articles = int(audit_ml.get("stock_alpha_news_provider_audit_min_article_count", 1))
    min_symbols = int(audit_ml.get("stock_alpha_news_provider_audit_min_symbol_count", 1))
    article_gap = max(0, min_articles - len(raw_rows))
    symbol_gap = max(0, min_symbols - len(raw_symbols))
    recommended_symbols = (stock_symbols or configured_symbols)[: max(min_symbols, len(configured_symbols))]
    configured_cap = int(collect.get("max_articles_per_provider", 50))
    recommended_cap = min(250, max(configured_cap, min_articles, min_symbols * 4))
    providers = dict(collect.get("providers", {}) or {})
    next_action = _next_action(
        configured_symbols=len(configured_symbols), recommended_symbols=len(recommended_symbols),
        current_articles=len(raw_rows), current_symbols=len(raw_symbols),
        min_articles=min_articles, min_symbols=min_symbols,
        configured_cap=configured_cap, recommended_cap=recommended_cap,
        finnhub_enabled=bool(dict(providers.get("finnhub", {}) or {}).get("enabled", False)),
    )
    return {
        "next_action": next_action,
        "input_status": {
            "stock_rows_path": str(stock_path), "stock_rows_exists": stock_path.is_file(),
            "stock_row_count": len(stock_rows), "provider_audit_config_path": str(audit_config_path),
            "provider_audit_config_exists": audit_config_path.is_file(),
            "raw_export_path": str(raw_path), "raw_export_exists": raw_path.is_file(),
        },
        "current_configured_symbols": configured_symbols,
        "stock_row_symbols_available": stock_symbols,
        "recommended_symbol_list": recommended_symbols,
        "provider_minimum_article_threshold": min_articles,
        "provider_minimum_symbol_threshold": min_symbols,
        "current_raw_export_row_count": len(raw_rows),
        "current_raw_export_symbol_count": len(raw_symbols),
        "current_raw_export_symbols": raw_symbols,
        "article_threshold_gap": article_gap,
        "symbol_threshold_gap": symbol_gap,
        "configured_max_articles_per_provider": configured_cap,
        "recommended_max_articles_per_provider": recommended_cap,
        "recommended_providers": {
            "alpha_vantage": {"enabled": True, "reason": "currently_usable"},
            "finnhub": {"enabled": False, "optional": True, "reason": "wider_window_if_alpha_vantage_insufficient"},
            "gdelt": {"enabled": False, "optional": True, "reason": "prior_http_429"},
            "fmp": {"enabled": False, "reason": "prior_http_402"},
            "newsapi": {"enabled": False, "reason": "prior_http_426"},
        },
        "inspection_only": True, "collection_invoked": False, "raw_export_written": False,
        "features_generated": False, "model_training_invoked": False,
        "news_transformer_enabled": False, "trading_impact": "none", "production_validated": False,
    }


def _next_action(*, configured_symbols: int, recommended_symbols: int, current_articles: int, current_symbols: int, min_articles: int, min_symbols: int, configured_cap: int, recommended_cap: int, finnhub_enabled: bool) -> str:
    if current_articles >= min_articles and current_symbols >= min_symbols:
        return "run_provider_audit"
    if configured_symbols < min_symbols or recommended_symbols < min_symbols:
        return "expand_symbol_list"
    if configured_cap < recommended_cap:
        return "increase_alpha_vantage_article_cap"
    if finnhub_enabled and current_symbols < min_symbols:
        return "try_finnhub_wider_window"
    return "run_broader_alpha_vantage_dry_run"


def _symbols(values: Any) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _markdown(payload: Mapping[str, Any]) -> str:
    status = payload["input_status"]
    return "\n".join([
        "# Stock-Alpha News Collection Plan", "",
        f"- Next action: {payload['next_action']}",
        f"- Stock rows exist: {status['stock_rows_exists']}",
        f"- Current raw rows: {payload['current_raw_export_row_count']}",
        f"- Current raw symbols: {payload['current_raw_export_symbol_count']}",
        f"- Article threshold gap: {payload['article_threshold_gap']}",
        f"- Symbol threshold gap: {payload['symbol_threshold_gap']}",
        f"- Recommended symbol count: {len(payload['recommended_symbol_list'])}",
        f"- Recommended provider cap: {payload['recommended_max_articles_per_provider']}",
        "- Collection invoked: false", "- Raw export written: false",
        "- Model training invoked: false", "",
        "Read-only plan. Provider audit thresholds are unchanged.",
    ])
