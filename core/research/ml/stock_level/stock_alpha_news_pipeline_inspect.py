from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import GUARDRAILS, REQUIRED_NEWS_CONTRACT_COLUMNS
from core.research.ml.stock_level.stock_alpha_news_pipeline_preflight import STAGE_ORDER


COUNT_THRESHOLDS = {
    "stock_alpha_news_provider_audit_min_symbol_count",
    "stock_alpha_news_provider_audit_min_article_count",
    "stock_alpha_news_coverage_min_article_count",
    "stock_alpha_news_coverage_min_covered_stock_rows",
    "stock_alpha_news_coverage_max_pit_violation_count",
}
RATE_THRESHOLDS = {
    "stock_alpha_news_provider_audit_max_missing_body_rate",
    "stock_alpha_news_provider_audit_max_duplicate_headline_rate",
    "stock_alpha_news_provider_audit_max_invalid_timestamp_rate",
    "stock_alpha_news_provider_audit_max_ingested_before_published_rate",
    "stock_alpha_news_coverage_min_symbol_coverage",
    "stock_alpha_news_coverage_min_date_coverage",
    "stock_alpha_news_min_symbol_coverage",
    "stock_alpha_news_min_date_coverage",
}
THRESHOLDS = COUNT_THRESHOLDS | RATE_THRESHOLDS


@dataclass(frozen=True)
class StockAlphaNewsPipelineInspectPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_pipeline_inspect(config: Mapping[str, Any]) -> StockAlphaNewsPipelineInspectPaths:
    payload = build_stock_alpha_news_pipeline_inspect(config)
    output = _required_path(config, "stock_alpha_news_pipeline_inspect_output_dir")
    paths = StockAlphaNewsPipelineInspectPaths(
        output / "stock_alpha_news_pipeline_inspect.json",
        output / "stock_alpha_news_pipeline_inspect.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_pipeline_inspect(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    thresholds = _validated_thresholds(ml)
    provider_map = ml.get("stock_alpha_news_provider_column_map", {}) or {}
    if not isinstance(provider_map, Mapping):
        raise ValueError("ml.stock_alpha_news_provider_column_map must be a mapping")

    raw = _required_path(config, "stock_alpha_news_raw_path")
    contract = _required_path(config, "stock_alpha_news_contract_path")
    stock_rows = _required_path(config, "stock_alpha_news_stock_rows_path")
    features = _required_path(config, "stock_alpha_news_features_path")
    coverage_dir = _required_path(config, "stock_alpha_news_coverage_audit_dir")
    pipeline_dir = _required_path(config, "stock_alpha_news_pipeline_preflight_output_dir")
    inspect_dir = _required_path(config, "stock_alpha_news_pipeline_inspect_output_dir")
    run_output = _required_path(config, "stock_alpha_report_root") / str(ml.get("stock_alpha_run_size", "dev"))
    feature_audit = run_output / "news_features" / "stock_alpha_news_features_audit.json"
    expected_nearby_audit = features.parent / "news_features" / feature_audit.name
    missing_map_keys = sorted(set(REQUIRED_NEWS_CONTRACT_COLUMNS) - set(provider_map)) if provider_map else []
    enable_flag = bool(ml.get("stock_alpha_news_enable_transformer", False))

    resolved_paths = {
        "raw_provider_file": str(raw),
        "canonical_contract_output": str(contract),
        "stock_rows_input": str(stock_rows),
        "coverage_audit_json": str(coverage_dir / "stock_alpha_news_coverage_audit.json"),
        "feature_output": str(features),
        "feature_audit_json": str(feature_audit),
        "readiness_json": str(run_output / "stock_alpha_news_readiness_preflight.json"),
        "pipeline_preflight_json": str(pipeline_dir / "stock_alpha_news_pipeline_preflight.json"),
    }
    return {
        "next_action": _next_action(raw.is_file(), stock_rows.is_file(), bool(provider_map), missing_map_keys, enable_flag),
        "resolved_paths": resolved_paths,
        "existence_checks": {
            "raw_provider_file_exists": raw.is_file(),
            "stock_rows_file_exists": stock_rows.is_file(),
            "output_directories": {
                "contract_parent": _directory_status(contract.parent),
                "coverage_audit_dir": _directory_status(coverage_dir),
                "feature_parent": _directory_status(features.parent),
                "feature_audit_dir": _directory_status(feature_audit.parent),
                "readiness_dir": _directory_status(run_output),
                "pipeline_preflight_dir": _directory_status(pipeline_dir),
                "inspection_dir": _directory_status(inspect_dir),
            },
            "nearby_feature_audit_path_coherent": expected_nearby_audit == feature_audit,
            "expected_nearby_feature_audit_path": str(expected_nearby_audit),
        },
        "config_summary": {
            "provider_column_map": dict(provider_map),
            "provider_column_map_missing_contract_keys": missing_map_keys,
            "thresholds": thresholds,
            "stock_alpha_news_enable_transformer": enable_flag,
            "research_guardrails": {key: ml.get(key) for key in GUARDRAILS},
        },
        "stage_order": list(STAGE_ORDER),
        "stage_prerequisites": {
            "provider_audit": ["raw_provider_file", "provider_column_map", "provider_thresholds"],
            "contract_ingest": ["provider_audit_safe", "raw_provider_file"],
            "coverage_audit": ["canonical_contract", "stock_rows_input"],
            "feature_generation": ["coverage_audit_safe", "research_guardrails"],
            "readiness_preflight": ["feature_output", "feature_audit", "stock_rows_input"],
        },
        "inspection_only": True,
        "files_ingested": False,
        "features_generated": False,
        "model_training_invoked": False,
        "diagnostics_invoked": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _validated_thresholds(ml: Mapping[str, Any]) -> dict[str, float]:
    unknown = sorted(key for key in ml if key.startswith("stock_alpha_news_") and ("_min_" in key or "_max_" in key) and key not in THRESHOLDS)
    if unknown:
        raise ValueError("unknown stock-alpha news threshold(s): " + ", ".join(unknown))
    values = {}
    for key in sorted(THRESHOLDS):
        if key not in ml:
            raise ValueError(f"missing ml.{key}")
        try:
            value = float(ml[key])
        except (TypeError, ValueError):
            raise ValueError(f"ml.{key} must be numeric") from None
        if value < 0 or (key in RATE_THRESHOLDS and value > 1):
            raise ValueError(f"ml.{key} is outside its valid range")
        values[key] = value
    return values


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _directory_status(path: Path) -> dict[str, Any]:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    return {"path": str(path), "exists": path.is_dir(), "can_be_created": path.is_dir() or os.access(ancestor, os.W_OK)}


def _next_action(raw_exists: bool, stock_rows_exists: bool, map_used: bool, missing_map_keys: list[str], enabled: bool) -> str:
    if not raw_exists:
        return "provide_raw_news_file"
    if map_used and missing_map_keys:
        return "fix_provider_column_map"
    if not stock_rows_exists:
        return "provide_stock_rows_file"
    if enabled:
        return "do_not_train_transformer"
    return "run_pipeline_preflight"


def _markdown(payload: Mapping[str, Any]) -> str:
    checks = payload["existence_checks"]
    config = payload["config_summary"]
    return "\n".join([
        "# Stock-Alpha News Pipeline Inspection",
        "",
        f"- Next action: {payload['next_action']}",
        f"- Raw provider exists: {checks['raw_provider_file_exists']}",
        f"- Stock rows exist: {checks['stock_rows_file_exists']}",
        f"- Transformer enabled: {config['stock_alpha_news_enable_transformer']}",
        "- Inspection only: true",
        "- Files ingested: false",
        "- Features generated: false",
        "- Model training invoked: false",
        "- Diagnostics invoked: false",
        "",
        "## Resolved Paths",
        *[f"- {name}: {value}" for name, value in payload["resolved_paths"].items()],
        "",
        "## Stage Order",
        *[f"{index}. {name}" for index, name in enumerate(payload["stage_order"], 1)],
        "",
        "Research-only inspection. No pipeline stage was executed.",
    ])
