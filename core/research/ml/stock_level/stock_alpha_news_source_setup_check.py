from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.reporting import ResearchArtifactWriter


PROVIDER_KEYS = {
    "gdelt": None,
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "fmp": "FMP_API_KEY",
    "newsapi": "NEWSAPI_API_KEY",
    "sec_edgar": None,
}


@dataclass(frozen=True)
class StockAlphaNewsSourceSetupCheckPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_source_setup_check(config: Mapping[str, Any]) -> StockAlphaNewsSourceSetupCheckPaths:
    payload = build_stock_alpha_news_source_setup_check(config)
    output = _path(config, "stock_alpha_news_source_setup_check_report_dir")
    paths = StockAlphaNewsSourceSetupCheckPaths(
        output / "stock_alpha_news_source_setup_check.json",
        output / "stock_alpha_news_source_setup_check.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_source_setup_check(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    collect = dict(ml.get("stock_alpha_news_collect", {}) or {})
    providers = dict(collect.get("providers", {}) or {})
    configured, enabled, statuses, missing = [], [], {}, []
    for name, provider_config in providers.items():
        configured.append(name)
        provider_config = dict(provider_config or {})
        is_enabled = bool(provider_config.get("enabled", False))
        if is_enabled:
            enabled.append(name)
        default_env = PROVIDER_KEYS.get(name)
        env_name = str(provider_config.get("api_key_env") or default_env or "")
        requires_key = bool(default_env)
        env_present = bool(os.environ.get(env_name)) if env_name else False
        if is_enabled and requires_key and not env_present:
            missing.append(name)
        statuses[name] = {
            "enabled": is_enabled,
            "api_key_required": requires_key,
            "api_key_env": env_name,
            "environment_variable_present": env_present,
        }
    literal_paths = _key_literal_paths(config)
    dry_run = bool(collect.get("dry_run", True))
    overwrite_protected = not bool(collect.get("allow_overwrite", False))
    blocking = []
    if missing:
        blocking.append("enabled_provider_api_key_missing")
    if literal_paths:
        blocking.append("key_like_literal_present_in_config")
    next_action = _next_action(enabled, missing, literal_paths, dry_run)
    secret_names = sorted({value for value in PROVIDER_KEYS.values() if value})
    return {
        "next_action": next_action,
        "blocking_issues": blocking,
        "providers_configured": configured,
        "providers_enabled": enabled,
        "provider_setup": statuses,
        "enabled_providers_missing_key": missing,
        "key_like_config_value_paths": literal_paths,
        "dry_run_enabled": dry_run,
        "output_overwrite_protected": overwrite_protected,
        "recommended_local_env_commands": [f"export {name}='<set-locally>'" for name in secret_names],
        "recommended_github_secret_names": secret_names,
        "inspection_only": True,
        "collection_invoked": False,
        "raw_export_written": False,
        "files_ingested": False,
        "features_generated": False,
        "readiness_invoked": False,
        "diagnostics_invoked": False,
        "model_training_invoked": False,
        "news_transformer_enabled": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _key_literal_paths(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if lowered != "api_key_env" and any(marker in lowered for marker in ("api_key", "apikey", "token", "secret")) and str(item or "").strip():
                found.append(path)
            else:
                found.extend(_key_literal_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_key_literal_paths(item, f"{prefix}[{index}]"))
    return found


def _next_action(enabled: list[str], missing: list[str], literals: list[str], dry_run: bool) -> str:
    if literals:
        return "remove_key_values_from_config"
    if missing:
        return f"set_{missing[0]}_api_key"
    if "gdelt" not in enabled and not enabled:
        return "enable_gdelt_dry_run"
    if dry_run:
        return "run_free_source_dry_collection"
    return "write_bounded_raw_provider_export"


def _path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Stock-Alpha News Source Setup Check", "",
        f"- Next action: {payload['next_action']}",
        f"- Providers configured: {payload['providers_configured']}",
        f"- Providers enabled: {payload['providers_enabled']}",
        f"- Enabled providers missing key: {payload['enabled_providers_missing_key']}",
        f"- Dry run enabled: {payload['dry_run_enabled']}",
        f"- Output overwrite protected: {payload['output_overwrite_protected']}",
        f"- GitHub Secret names: {payload['recommended_github_secret_names']}",
        "- Collection invoked: false", "- Raw export written: false",
        "- Model training invoked: false", "",
        "Environment presence only; no secret values are read into this report.",
    ])
