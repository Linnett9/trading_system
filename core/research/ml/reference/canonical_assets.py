from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


REGISTRY_SCHEMA_VERSION = "canonical_asset_registry.v1"
ALIAS_SCHEMA_VERSION = "provider_symbol_aliases.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "dataset_manifest.v1"
DAILY_SPINE_SCHEMA_VERSION = "canonical_daily_spine.v1"
FEATURE_FAMILY_SCHEMA_VERSION = "pit_feature_family.v1"
VALID_FROM = "1900-01-01"
VALID_TO = ""
PROVIDERS = frozenset({"canonical", "stooq", "alpaca", "sec", "news"})
MANIFEST_STATUSES = frozenset({"READY", "READY_WITH_CONDITIONS", "BLOCKED", "FAILED"})
CANONICAL_ASSET_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "security_name",
    "security_type",
    "share_class",
    "exchange",
    "currency",
    "country",
    "cik",
    "sector",
    "industry",
    "valid_from",
    "valid_to",
    "is_active",
    "collection_universe_514",
    "registry_version",
)
PROVIDER_ALIAS_FIELDS = (
    "asset_id",
    "provider",
    "provider_symbol",
    "valid_from",
    "valid_to",
    "is_primary",
    "mapping_reason",
    "source",
    "registry_version",
)
DAILY_SPINE_FIELDS = (
    "row_id",
    "asset_id",
    "canonical_symbol",
    "session_date",
    "decision_timestamp",
    "feature_cutoff_timestamp",
    "universe_version",
    "eligible_at_decision",
    "eligibility_reason",
    "daily_price_dataset_version",
    "symbol_registry_version",
    "calendar_version",
    "target_horizon_sessions",
    "target_start_timestamp",
    "target_end_timestamp",
    "target_available_timestamp",
    "benchmark_asset_id",
    "target_definition_version",
)
FEATURE_FAMILY_FIELDS = (
    "row_id",
    "asset_id",
    "decision_timestamp",
    "feature_available_timestamp",
    "feature_family",
    "feature_version",
    "source_dataset_version",
)
DATASET_MANIFEST_FIELDS = (
    "dataset_id",
    "dataset_type",
    "schema_version",
    "created_at",
    "code_commit",
    "config_hash",
    "source_paths",
    "source_dataset_ids",
    "source_checksums",
    "symbol_registry_version",
    "calendar_version",
    "universe_version",
    "row_grain",
    "primary_keys",
    "row_count",
    "symbol_count",
    "date_min",
    "date_max",
    "row_identity_checksum",
    "feature_versions",
    "target_version",
    "adjustment_policy",
    "provider",
    "timeframe",
    "status",
    "warnings",
)


@dataclass(frozen=True)
class CanonicalAsset:
    asset_id: str
    canonical_symbol: str
    security_name: str | None
    security_type: str
    share_class: str | None
    exchange: str | None
    currency: str
    country: str
    cik: str | None
    sector: str | None
    industry: str | None
    valid_from: str
    valid_to: str
    is_active: bool
    collection_universe_514: bool
    registry_version: str


@dataclass(frozen=True)
class ProviderAlias:
    asset_id: str
    provider: str
    provider_symbol: str
    valid_from: str
    valid_to: str
    is_primary: bool
    mapping_reason: str
    source: str
    registry_version: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    dataset_type: str
    schema_version: str
    created_at: str
    code_commit: str | None
    config_hash: str | None
    source_paths: tuple[str, ...]
    source_dataset_ids: tuple[str, ...]
    source_checksums: Mapping[str, str]
    symbol_registry_version: str | None
    calendar_version: str | None
    universe_version: str | None
    row_grain: str
    primary_keys: tuple[str, ...]
    row_count: int | None
    symbol_count: int | None
    date_min: str | None
    date_max: str | None
    row_identity_checksum: str | None
    feature_versions: Mapping[str, str]
    target_version: str | None
    adjustment_policy: str | None
    provider: str | None
    timeframe: str | None
    status: str
    warnings: tuple[str, ...]


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def alpaca_provider_symbol(canonical_symbol: str, configured_map: Mapping[str, str] | None = None) -> str:
    symbol = normalize_symbol(canonical_symbol)
    configured = {normalize_symbol(k): normalize_symbol(v) for k, v in (configured_map or {}).items()}
    if symbol in configured:
        return configured[symbol]
    if symbol in {"BRK-A", "BRK-B"}:
        return symbol.replace("-", ".")
    return symbol


def canonical_asset_id(canonical_symbol: str) -> str:
    symbol = normalize_symbol(canonical_symbol)
    digest = hashlib.sha256(f"canonical_asset:{symbol}".encode("utf-8")).hexdigest()[:16]
    return f"asset_{digest}"


def daily_spine_row_id(
    *,
    asset_id: str,
    decision_timestamp: str,
    target_horizon_sessions: int,
    universe_version: str,
    daily_price_dataset_version: str,
    target_definition_version: str,
) -> str:
    payload = {
        "asset_id": asset_id,
        "decision_timestamp": decision_timestamp,
        "target_horizon_sessions": int(target_horizon_sessions),
        "universe_version": universe_version,
        "daily_price_dataset_version": daily_price_dataset_version,
        "target_definition_version": target_definition_version,
    }
    return "row_" + _sha256_json(payload)[:24]


def validate_feature_family_row(row: Mapping[str, Any]) -> None:
    missing = [field for field in FEATURE_FAMILY_FIELDS if field not in row]
    if missing:
        raise ValueError("missing feature-family fields: " + ", ".join(missing))
    decision = _parse_datetime(row["decision_timestamp"])
    available = _parse_datetime(row["feature_available_timestamp"])
    if available > decision:
        raise ValueError("feature_available_timestamp must be <= decision_timestamp")


def build_dataset_manifest(
    *,
    dataset_type: str,
    row_grain: str,
    primary_keys: Sequence[str],
    source_paths: Sequence[str | Path] = (),
    source_dataset_ids: Sequence[str] = (),
    symbol_registry_version: str | None = None,
    calendar_version: str | None = None,
    universe_version: str | None = None,
    row_count: int | None = None,
    symbol_count: int | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    row_identity_checksum: str | None = None,
    feature_versions: Mapping[str, str] | None = None,
    target_version: str | None = None,
    adjustment_policy: str | None = None,
    provider: str | None = None,
    timeframe: str | None = None,
    config: Mapping[str, Any] | None = None,
    required_source_paths: Sequence[str | Path] = (),
    created_at: str | None = None,
) -> DatasetManifest:
    source_path_text = tuple(str(path) for path in source_paths)
    missing = [str(path) for path in required_source_paths if not Path(path).exists()]
    warnings = tuple(f"missing_required_source:{path}" for path in missing)
    status = "BLOCKED" if missing else "READY"
    identity = {
        "dataset_type": dataset_type,
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "source_paths": source_path_text,
        "source_dataset_ids": tuple(source_dataset_ids),
        "symbol_registry_version": symbol_registry_version,
        "calendar_version": calendar_version,
        "universe_version": universe_version,
        "row_grain": row_grain,
        "primary_keys": tuple(primary_keys),
        "feature_versions": dict(feature_versions or {}),
        "target_version": target_version,
        "adjustment_policy": adjustment_policy,
        "provider": provider,
        "timeframe": timeframe,
    }
    return DatasetManifest(
        dataset_id=f"{dataset_type}-{_sha256_json(identity)[:16]}",
        dataset_type=dataset_type,
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        created_at=created_at or _utc_now(),
        code_commit=_git_commit(),
        config_hash=_sha256_json(config or {}) if config is not None else None,
        source_paths=source_path_text,
        source_dataset_ids=tuple(source_dataset_ids),
        source_checksums={path: file_sha256(Path(path)) for path in source_path_text if Path(path).is_file()},
        symbol_registry_version=symbol_registry_version,
        calendar_version=calendar_version,
        universe_version=universe_version,
        row_grain=row_grain,
        primary_keys=tuple(primary_keys),
        row_count=row_count,
        symbol_count=symbol_count,
        date_min=date_min,
        date_max=date_max,
        row_identity_checksum=row_identity_checksum,
        feature_versions=dict(feature_versions or {}),
        target_version=target_version,
        adjustment_policy=adjustment_policy,
        provider=provider,
        timeframe=timeframe,
        status=status,
        warnings=warnings,
    )


def build_registry_from_universe(
    universe_path: Path = Path("config/universes/alpaca_514_symbols.txt"),
    *,
    provider_symbol_map: Mapping[str, str] | None = None,
) -> tuple[list[CanonicalAsset], list[ProviderAlias], str]:
    symbols = read_symbol_lines(universe_path)
    preliminary_assets = [
        _asset(symbol, registry_version="pending", collection_universe_514=True)
        for symbol in sorted(symbols)
    ]
    preliminary_aliases = _aliases(preliminary_assets, provider_symbol_map or {}, registry_version="pending")
    registry_version = registry_content_hash(preliminary_assets, preliminary_aliases, include_registry_version=False)
    assets = [_asset(asset.canonical_symbol, registry_version=registry_version, collection_universe_514=True) for asset in preliminary_assets]
    aliases = _aliases(assets, provider_symbol_map or {}, registry_version=registry_version)
    validate_registry(assets, aliases)
    return assets, aliases, registry_version


def validate_registry(assets: Sequence[CanonicalAsset], aliases: Sequence[ProviderAlias]) -> None:
    duplicate_assets = duplicate_active_canonical_symbols(assets)
    if duplicate_assets:
        raise ValueError("duplicate active canonical symbols: " + ", ".join(duplicate_assets))
    duplicate_aliases = duplicate_provider_aliases(aliases)
    if duplicate_aliases:
        formatted = [f"{row['provider']}:{row['provider_symbol']}" for row in duplicate_aliases]
        raise ValueError("duplicate overlapping provider aliases: " + ", ".join(formatted))
    unsupported = sorted({alias.provider for alias in aliases} - PROVIDERS)
    if unsupported:
        raise ValueError("unsupported providers: " + ", ".join(unsupported))


def duplicate_active_canonical_symbols(assets: Sequence[CanonicalAsset]) -> list[str]:
    duplicates = []
    for left_index, left in enumerate(assets):
        for right in assets[left_index + 1 :]:
            if left.canonical_symbol == right.canonical_symbol and _intervals_overlap(left.valid_from, left.valid_to, right.valid_from, right.valid_to):
                duplicates.append(left.canonical_symbol)
    return sorted(set(duplicates))


def duplicate_provider_aliases(aliases: Sequence[ProviderAlias]) -> list[dict[str, str]]:
    duplicates = []
    for left_index, left in enumerate(aliases):
        for right in aliases[left_index + 1 :]:
            if (
                left.provider == right.provider
                and left.provider_symbol == right.provider_symbol
                and left.asset_id != right.asset_id
                and _intervals_overlap(left.valid_from, left.valid_to, right.valid_from, right.valid_to)
            ):
                duplicates.append({"provider": left.provider, "provider_symbol": left.provider_symbol})
    return sorted(duplicates, key=lambda row: (row["provider"], row["provider_symbol"]))


def ambiguous_aliases(aliases: Sequence[ProviderAlias]) -> list[dict[str, str]]:
    return duplicate_provider_aliases(aliases)


def registry_content_hash(
    assets: Sequence[CanonicalAsset],
    aliases: Sequence[ProviderAlias],
    *,
    include_registry_version: bool = True,
) -> str:
    asset_payload = [_registry_dict(asset, include_registry_version=include_registry_version) for asset in sort_assets(assets)]
    alias_payload = [_registry_dict(alias, include_registry_version=include_registry_version) for alias in sort_aliases(aliases)]
    return _sha256_json(
        {
            "asset_schema": CANONICAL_ASSET_FIELDS,
            "alias_schema": PROVIDER_ALIAS_FIELDS,
            "assets": asset_payload,
            "aliases": alias_payload,
        }
    )


def write_registry_outputs(
    assets: Sequence[CanonicalAsset],
    aliases: Sequence[ProviderAlias],
    *,
    asset_output: Path,
    alias_output: Path,
    parquet_output: Path | None = None,
) -> None:
    _write_csv_if_safe(asset_output, (_csv_row(asset, CANONICAL_ASSET_FIELDS) for asset in sort_assets(assets)), CANONICAL_ASSET_FIELDS)
    _write_csv_if_safe(alias_output, (_csv_row(alias, PROVIDER_ALIAS_FIELDS) for alias in sort_aliases(aliases)), PROVIDER_ALIAS_FIELDS)
    if parquet_output is not None:
        _write_parquet(assets, parquet_output)


def read_assets_csv(path: Path) -> list[CanonicalAsset]:
    rows = _read_csv(path)
    return [
        CanonicalAsset(
            asset_id=row["asset_id"],
            canonical_symbol=row["canonical_symbol"],
            security_name=_none(row["security_name"]),
            security_type=row["security_type"],
            share_class=_none(row["share_class"]),
            exchange=_none(row["exchange"]),
            currency=row["currency"],
            country=row["country"],
            cik=_none(row["cik"]),
            sector=_none(row["sector"]),
            industry=_none(row["industry"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            is_active=_bool(row["is_active"]),
            collection_universe_514=_bool(row["collection_universe_514"]),
            registry_version=row["registry_version"],
        )
        for row in rows
    ]


def read_aliases_csv(path: Path) -> list[ProviderAlias]:
    rows = _read_csv(path)
    return [
        ProviderAlias(
            asset_id=row["asset_id"],
            provider=row["provider"],
            provider_symbol=row["provider_symbol"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            is_primary=_bool(row["is_primary"]),
            mapping_reason=row["mapping_reason"],
            source=row["source"],
            registry_version=row["registry_version"],
        )
        for row in rows
    ]


def audit_registry(
    assets: Sequence[CanonicalAsset],
    aliases: Sequence[ProviderAlias],
    *,
    universe_path: Path = Path("config/universes/alpaca_514_symbols.txt"),
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    collection_symbols = read_symbol_lines(universe_path) if universe_path.exists() else []
    canonical_symbols = {asset.canonical_symbol for asset in assets}
    stooq_symbols = discover_daily_symbols(repo_root)
    alpaca_config_symbols = discover_alpaca_config_symbols(repo_root)
    news_symbols = discover_news_registry_symbols(repo_root)
    sec_symbols = discover_sec_config_symbols(repo_root)
    alias_lookup = {(alias.provider, alias.provider_symbol): alias for alias in aliases}
    missing_alpaca = [
        symbol for symbol in collection_symbols
        if ("alpaca", alpaca_provider_symbol(symbol)) not in alias_lookup
    ]
    missing_stooq = sorted(set(collection_symbols) - stooq_symbols)
    missing_ciks = sorted(asset.canonical_symbol for asset in assets if not asset.cik)
    ambiguous = ambiguous_aliases(aliases)
    duplicate_canonical = duplicate_active_canonical_symbols(assets)
    duplicate_alias = duplicate_provider_aliases(aliases)
    brk = {
        "BRK-A": {
            "canonical_present": "BRK-A" in canonical_symbols,
            "alpaca_alias": alias_lookup.get(("alpaca", "BRK.A")).asset_id if ("alpaca", "BRK.A") in alias_lookup else None,
            "status": "ok" if ("alpaca", "BRK.A") in alias_lookup else "missing",
        },
        "BRK-B": {
            "canonical_present": "BRK-B" in canonical_symbols,
            "alpaca_alias": alias_lookup.get(("alpaca", "BRK.B")).asset_id if ("alpaca", "BRK.B") in alias_lookup else None,
            "status": "ok" if ("alpaca", "BRK.B") in alias_lookup else "missing",
        },
    }
    return {
        "schema_version": "canonical_asset_registry_audit.v1",
        "canonical_asset_count": len(assets),
        "collection_universe_symbol_count": len(collection_symbols),
        "resolved_collection_symbol_count": len(set(collection_symbols) & canonical_symbols),
        "unresolved_collection_symbols": sorted(set(collection_symbols) - canonical_symbols),
        "ambiguous_aliases": ambiguous,
        "duplicate_active_canonical_symbols": duplicate_canonical,
        "duplicate_provider_aliases": duplicate_alias,
        "missing_alpaca_aliases": missing_alpaca,
        "missing_stooq_matches": missing_stooq,
        "missing_sec_ciks": missing_ciks,
        "symbols_found_in_daily_files_not_collection_universe": sorted(stooq_symbols - set(collection_symbols)),
        "symbols_found_in_alpaca_configs_not_canonical_registry": sorted(alpaca_config_symbols - canonical_symbols),
        "news_symbols_not_canonical_registry": sorted(news_symbols - canonical_symbols),
        "sec_symbols_not_canonical_registry": sorted(sec_symbols - canonical_symbols),
        "daily_data_available_symbol_count": len(stooq_symbols & canonical_symbols),
        "intraday_config_symbol_count": len(alpaca_config_symbols & canonical_symbols),
        "sec_config_symbol_count": len(sec_symbols & canonical_symbols),
        "news_registry_symbol_count": len(news_symbols & canonical_symbols),
        "selector_eligibility_inferred": False,
        "brk_mapping_status": brk,
        "registry_content_hash": registry_content_hash(assets, aliases),
        "registry_version": assets[0].registry_version if assets else None,
    }


def write_audit_reports(audit: Mapping[str, Any], *, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "registry_audit.json", audit)
    (report_dir / "registry_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    _write_csv(
        report_dir / "unresolved_symbols.csv",
        [{"symbol": symbol, "status": "unresolved_collection_symbol"} for symbol in audit.get("unresolved_collection_symbols", [])],
        ("symbol", "status"),
    )
    _write_csv(
        report_dir / "ambiguous_symbols.csv",
        audit.get("ambiguous_aliases", []),
        ("provider", "provider_symbol"),
    )
    manifest = build_dataset_manifest(
        dataset_type="canonical_asset_registry_audit",
        row_grain="one canonical asset or provider alias audit snapshot",
        primary_keys=("registry_content_hash",),
        source_paths=(),
        symbol_registry_version=str(audit.get("registry_version") or ""),
        row_count=int(audit.get("canonical_asset_count", 0) or 0),
        symbol_count=int(audit.get("canonical_asset_count", 0) or 0),
        row_identity_checksum=str(audit.get("registry_content_hash") or ""),
        feature_versions={},
    )
    _write_json(report_dir / "manifest.json", asdict(manifest))


def build_and_audit(
    *,
    dry_run: bool = False,
    audit_only: bool = False,
    registry_output: Path = Path("data/reference/assets/canonical_asset_registry.csv"),
    alias_output: Path = Path("data/reference/assets/provider_symbol_aliases.csv"),
    parquet_output: Path = Path("data/reference/assets/canonical_asset_registry.parquet"),
    report_dir: Path = Path("reports/data_lineage/canonical_asset_registry"),
    universe_path: Path = Path("config/universes/alpaca_514_symbols.txt"),
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    provider_map = discover_configured_alpaca_provider_symbol_map(repo_root)
    if audit_only:
        assets = read_assets_csv(registry_output)
        aliases = read_aliases_csv(alias_output)
    else:
        assets, aliases, _registry_version = build_registry_from_universe(universe_path, provider_symbol_map=provider_map)
        if not dry_run:
            write_registry_outputs(assets, aliases, asset_output=registry_output, alias_output=alias_output, parquet_output=parquet_output)
    audit = audit_registry(assets, aliases, universe_path=universe_path, repo_root=repo_root)
    write_audit_reports(audit, report_dir=report_dir)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and audit the canonical asset registry.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--registry-output", type=Path, default=Path("data/reference/assets/canonical_asset_registry.csv"))
    parser.add_argument("--alias-output", type=Path, default=Path("data/reference/assets/provider_symbol_aliases.csv"))
    parser.add_argument("--parquet-output", type=Path, default=Path("data/reference/assets/canonical_asset_registry.parquet"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/data_lineage/canonical_asset_registry"))
    parser.add_argument("--universe-path", type=Path, default=Path("config/universes/alpaca_514_symbols.txt"))
    args = parser.parse_args(argv)
    if args.dry_run and args.audit_only:
        raise SystemExit("--dry-run and --audit-only cannot be combined")
    audit = build_and_audit(
        dry_run=args.dry_run,
        audit_only=args.audit_only,
        registry_output=args.registry_output,
        alias_output=args.alias_output,
        parquet_output=args.parquet_output,
        report_dir=args.report_dir,
        universe_path=args.universe_path,
    )
    print(json.dumps({
        "canonical_asset_count": audit["canonical_asset_count"],
        "resolved_collection_symbol_count": audit["resolved_collection_symbol_count"],
        "unresolved_collection_symbols": audit["unresolved_collection_symbols"],
        "ambiguous_aliases": audit["ambiguous_aliases"],
        "registry_content_hash": audit["registry_content_hash"],
        "report_dir": str(args.report_dir),
        "dry_run": args.dry_run,
        "audit_only": args.audit_only,
    }, indent=2))
    return 0


def read_symbol_lines(path: Path) -> list[str]:
    seen: set[str] = set()
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = normalize_symbol(line.split("#", 1)[0])
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def discover_configured_alpaca_provider_symbol_map(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((repo_root / "config").glob("config.historical_bar_backfill_alpaca*.yaml")):
        payload = _read_yaml(path)
        settings = (((payload.get("ml") or {}).get("historical_bar_backfill") or {}) if isinstance(payload, Mapping) else {})
        raw = settings.get("provider_symbol_map", {}) if isinstance(settings, Mapping) else {}
        if isinstance(raw, Mapping):
            for canonical, provider in raw.items():
                result[normalize_symbol(canonical)] = normalize_symbol(provider)
    return result


def discover_daily_symbols(repo_root: Path) -> set[str]:
    roots = [
        repo_root / "data/reference/adjusted_prices",
        repo_root / "data/processed",
        repo_root / "data/processed/stooq_parquet",
        repo_root / "data/raw/stooq_bulk/data/daily",
    ]
    symbols: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name == "processed":
            for path in root.glob("*/1Day/bars.parquet"):
                symbols.add(normalize_symbol(path.parents[1].name))
        else:
            for path in root.glob("*.csv"):
                if path.name.lower() != "manifest.json":
                    symbols.add(normalize_symbol(path.stem))
            for path in root.glob("*.parquet"):
                symbols.add(normalize_symbol(path.stem))
    return {symbol for symbol in symbols if symbol}


def discover_alpaca_config_symbols(repo_root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((repo_root / "config").glob("config.historical_bar_backfill_alpaca*.yaml")):
        payload = _read_yaml(path)
        settings = (((payload.get("ml") or {}).get("historical_bar_backfill") or {}) if isinstance(payload, Mapping) else {})
        if not isinstance(settings, Mapping):
            continue
        if settings.get("universe_file"):
            universe_path = repo_root / str(settings["universe_file"])
            if universe_path.exists():
                symbols.update(read_symbol_lines(universe_path))
        symbols.update(normalize_symbol(symbol) for symbol in settings.get("symbols", []) or [])
        raw = settings.get("provider_symbol_map", {})
        if isinstance(raw, Mapping):
            symbols.update(normalize_symbol(symbol) for symbol in raw)
    return {symbol for symbol in symbols if symbol}


def discover_news_registry_symbols(repo_root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((repo_root / "config").glob("news_source_registry*.yaml")):
        payload = _read_yaml(path)
        if not isinstance(payload, Mapping):
            continue
        grouped = payload.get("_classifications", {})
        if isinstance(grouped, Mapping):
            for values in grouped.values():
                symbols.update(_news_registry_symbol(value) for value in values or [])
        for container_name in ("funds", "exceptions"):
            container = payload.get(container_name, {})
            if isinstance(container, Mapping):
                symbols.update(_news_registry_symbol(symbol) for symbol in container)
        metadata_keys = {
            "version",
            "provider_policy",
            "mapping_requirements",
            "_classifications",
            "_classification_overrides",
            "funds",
            "exceptions",
        }
        for key, value in payload.items():
            symbol = _news_registry_symbol(key)
            if symbol and not symbol.startswith("_") and str(key).lower() not in metadata_keys and isinstance(value, Mapping):
                symbols.add(symbol)
    return {symbol for symbol in symbols if symbol}


def discover_sec_config_symbols(repo_root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((repo_root / "config").glob("config.stock_alpha_news_collect_sec_company_filings*.yaml")):
        payload = _read_yaml(path)
        settings = (((payload.get("ml") or {}).get("stock_alpha_news") or {}) if isinstance(payload, Mapping) else {})
        if isinstance(settings, Mapping):
            symbols.update(normalize_symbol(symbol) for symbol in settings.get("symbols", []) or [])
            symbols.update(normalize_symbol(symbol) for symbol in settings.get("only_symbols", []) or [])
    return {symbol for symbol in symbols if symbol}


def sort_assets(assets: Sequence[CanonicalAsset]) -> list[CanonicalAsset]:
    return sorted(assets, key=lambda asset: (asset.canonical_symbol, asset.valid_from, asset.asset_id))


def sort_aliases(aliases: Sequence[ProviderAlias]) -> list[ProviderAlias]:
    return sorted(aliases, key=lambda alias: (alias.provider, alias.provider_symbol, alias.valid_from, alias.asset_id))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(symbol: str, *, registry_version: str, collection_universe_514: bool) -> CanonicalAsset:
    canonical = normalize_symbol(symbol)
    share_class = canonical.split("-", 1)[1] if "-" in canonical else None
    return CanonicalAsset(
        asset_id=canonical_asset_id(canonical),
        canonical_symbol=canonical,
        security_name=None,
        security_type="UNKNOWN",
        share_class=share_class,
        exchange=None,
        currency="USD",
        country="US",
        cik=None,
        sector=None,
        industry=None,
        valid_from=VALID_FROM,
        valid_to=VALID_TO,
        is_active=True,
        collection_universe_514=collection_universe_514,
        registry_version=registry_version,
    )


def _aliases(assets: Sequence[CanonicalAsset], provider_symbol_map: Mapping[str, str], *, registry_version: str) -> list[ProviderAlias]:
    rows: list[ProviderAlias] = []
    for asset in sort_assets(assets):
        canonical = asset.canonical_symbol
        rows.append(_alias(asset, "canonical", canonical, "canonical_symbol", "seeded_collection_universe", registry_version))
        rows.append(_alias(asset, "stooq", canonical, "stooq_daily_filename", "seeded_collection_universe", registry_version))
        rows.append(_alias(asset, "news", canonical, "news_symbol_registry", "seeded_collection_universe", registry_version))
        rows.append(_alias(asset, "sec", canonical, "sec_ticker_normalization", "seeded_collection_universe", registry_version))
        alpaca = alpaca_provider_symbol(canonical, provider_symbol_map)
        rows.append(_alias(asset, "alpaca", alpaca, "alpaca_backfill_provider_symbol_map", "configured_provider_map" if alpaca != canonical else "identity", registry_version))
    return rows


def _alias(asset: CanonicalAsset, provider: str, provider_symbol: str, source: str, reason: str, registry_version: str) -> ProviderAlias:
    return ProviderAlias(
        asset_id=asset.asset_id,
        provider=provider,
        provider_symbol=provider_symbol,
        valid_from=asset.valid_from,
        valid_to=asset.valid_to,
        is_primary=True,
        mapping_reason=reason,
        source=source,
        registry_version=registry_version,
    )


def _intervals_overlap(left_from: str, left_to: str, right_from: str, right_to: str) -> bool:
    left_end = left_to or "9999-12-31"
    right_end = right_to or "9999-12-31"
    return left_from <= right_end and right_from <= left_end


def _registry_dict(row: Any, *, include_registry_version: bool) -> dict[str, Any]:
    payload = asdict(row)
    if not include_registry_version:
        payload["registry_version"] = ""
    return payload


def _csv_row(row: Any, fields: Sequence[str]) -> dict[str, str]:
    payload = asdict(row)
    result = {}
    for field in fields:
        value = payload[field]
        if value is None:
            result[field] = ""
        elif isinstance(value, bool):
            result[field] = "true" if value else "false"
        else:
            result[field] = str(value)
    return result


def _write_csv_if_safe(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    rendered = _render_csv(rows, fieldnames)
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to overwrite existing registry source with different content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8", newline="")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_csv(rows, fieldnames), encoding="utf-8", newline="")


def _render_csv(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return handle.getvalue()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_parquet(assets: Sequence[CanonicalAsset], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [
        ("asset_id", pa.string()),
        ("canonical_symbol", pa.string()),
        ("security_name", pa.string()),
        ("security_type", pa.string()),
        ("share_class", pa.string()),
        ("exchange", pa.string()),
        ("currency", pa.string()),
        ("country", pa.string()),
        ("cik", pa.string()),
        ("sector", pa.string()),
        ("industry", pa.string()),
        ("valid_from", pa.string()),
        ("valid_to", pa.string()),
        ("is_active", pa.bool_()),
        ("collection_universe_514", pa.bool_()),
        ("registry_version", pa.string()),
    ]
    schema = pa.schema(fields)
    table = pa.Table.from_pylist([asdict(asset) for asset in sort_assets(assets)], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _audit_markdown(audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Canonical Asset Registry Audit",
            "",
            "Research only. No market, SEC, news, selector, exposure or execution data was moved.",
            "",
            f"- Canonical assets: {audit['canonical_asset_count']}",
            f"- 514 collection symbols resolved: {audit['resolved_collection_symbol_count']}",
            f"- Unresolved collection symbols: {len(audit['unresolved_collection_symbols'])}",
            f"- Ambiguous aliases: {len(audit['ambiguous_aliases'])}",
            f"- Duplicate active canonical symbols: {len(audit['duplicate_active_canonical_symbols'])}",
            f"- Duplicate provider aliases: {len(audit['duplicate_provider_aliases'])}",
            f"- Missing Alpaca aliases: {len(audit['missing_alpaca_aliases'])}",
            f"- Missing Stooq matches: {len(audit['missing_stooq_matches'])}",
            f"- Missing SEC CIKs: {len(audit['missing_sec_ciks'])}",
            f"- BRK-A Alpaca mapping: {audit['brk_mapping_status']['BRK-A']['status']}",
            f"- BRK-B Alpaca mapping: {audit['brk_mapping_status']['BRK-B']['status']}",
            f"- Registry content hash: {audit['registry_content_hash']}",
            "",
        ]
    )


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _news_registry_symbol(value: Any) -> str:
    symbol = normalize_symbol(value)
    if symbol == "BRK.B":
        return "BRK-B"
    if symbol == "BRK.A":
        return "BRK-A"
    return symbol


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit or None


def _none(value: str) -> str | None:
    return value if value else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
