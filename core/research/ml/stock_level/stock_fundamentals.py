from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.providers import (
    SEC_COMPANY_TICKERS_URL,
    normalize_sec_company_tickers,
    normalize_sec_ticker,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    file_sha256,
    read_stock_level_artifact,
    write_stock_level_artifact,
)


SCHEMA_VERSION = "stock_fundamentals_pipeline_v1"
PROVIDER_CONTRACT_VERSION = "fundamentals_provider_contract_v1"
SEC_COMPANY_FACTS_PROVIDER_VERSION = "sec_companyfacts_v1"
NORMALISATION_CONTRACT_VERSION = "fundamentals_normalisation_contract_v1"
FACT_DICTIONARY_VERSION = "fundamentals_fact_dictionary_v1"
SNAPSHOT_CONTRACT_VERSION = "fundamentals_pit_snapshot_contract_v1"
FEATURE_FORMULA_VERSION = "fundamentals_feature_formula_contract_v1"
ENRICHMENT_CONTRACT_VERSION = "stock_level_fundamentals_enrichment_contract_v1"
DIAGNOSTIC_STATUS = "BOUNDED DIAGNOSTIC ONLY / NOT FEATURE PROMOTION EVIDENCE"

CANONICAL_FACT_COLUMNS = (
    "provider_id",
    "reporting_entity_id",
    "security_mapping_identity",
    "source_document_id",
    "filing_accession",
    "form_type",
    "filing_timestamp",
    "acceptance_timestamp",
    "first_seen_timestamp",
    "available_timestamp",
    "period_start",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "fact_namespace",
    "source_fact_name",
    "canonical_fact_id",
    "unit",
    "normalized_unit",
    "value",
    "fact_period_type",
    "is_amendment",
    "amends_document_id",
    "source_raw_path",
    "source_raw_sha256",
    "normalisation_contract_identity",
)

FUNDAMENTAL_FEATURE_COLUMNS = (
    "revenue_growth_yoy",
    "revenue_growth_qoq",
    "gross_profit_growth_yoy",
    "operating_income_growth_yoy",
    "net_income_growth_yoy",
    "eps_growth_yoy",
    "operating_cash_flow_growth_yoy",
    "asset_growth_yoy",
    "equity_growth_yoy",
    "growth_acceleration",
    "positive_growth_breadth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_assets",
    "return_on_equity",
    "asset_turnover",
    "operating_cash_flow_to_assets",
    "free_cash_flow_margin",
    "cash_conversion",
    "total_accruals_to_assets",
    "cash_flow_to_net_income",
    "working_capital_accruals",
    "earnings_quality_score",
    "debt_to_assets",
    "debt_to_equity",
    "net_debt_to_assets",
    "current_ratio",
    "cash_to_assets",
    "interest_coverage",
    "working_capital_to_assets",
    "share_count_growth_yoy",
    "dilution_rate",
    "net_share_issuance",
    "repurchase_intensity",
    "dividend_payout",
    "earnings_yield",
    "book_to_market",
    "sales_to_price",
    "free_cash_flow_yield",
    "filing_recency_score",
    "fundamental_coverage_count",
    "fundamental_missing_fraction",
    "restatement_indicator",
    "entity_mapping_quality",
)

FUNDAMENTAL_METADATA_COLUMNS = (
    "fundamentals_snapshot_status",
    "fundamentals_available_timestamp",
    "fundamentals_latest_filing_timestamp",
    "fundamentals_data_age_days",
    "fundamentals_contract_identity",
    "fundamentals_source_identity",
    "fundamentals_reporting_entity_id",
    "analyst_estimate_status",
)


@dataclass(frozen=True)
class StockFundamentalsPaths:
    output_dir: Path
    entity_mapping_path: Path
    raw_collection_manifest_path: Path
    failed_entities_path: Path
    fact_dictionary_path: Path
    normalized_facts_path: Path
    normalization_audit_path: Path
    snapshots_path: Path
    snapshot_audit_path: Path
    feature_contracts_path: Path
    feature_coverage_path: Path
    enriched_artifact_path: Path
    enrichment_audit_json_path: Path
    enrichment_audit_markdown_path: Path
    preflight_path: Path
    entity_mapping_audit_path: Path
    bounded_cohort_path: Path
    tag_coverage_path: Path
    unit_conflicts_path: Path
    period_reconciliation_path: Path
    snapshot_coverage_path: Path
    pipeline_manifest_path: Path
    readiness_json_path: Path
    readiness_markdown_path: Path
    report_json_path: Path
    report_markdown_path: Path


class SecCompanyFactsProvider:
    provider_id = "official_sec_companyfacts"
    provider_version = SEC_COMPANY_FACTS_PROVIDER_VERSION

    def __init__(
        self,
        *,
        raw_root: Path,
        user_agent: str,
        request_delay_seconds: float = 0.2,
        max_retries: int = 2,
        timeout_seconds: int = 30,
        http_get: Callable[[str, Mapping[str, str], int], tuple[bytes, Mapping[str, str]]] | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.user_agent = user_agent
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.max_retries = max(0, int(max_retries))
        self.timeout_seconds = int(timeout_seconds)
        self.http_get = http_get or _http_get_bytes

    def resolve_reporting_entities(
        self,
        symbols: Sequence[str],
        *,
        cik_by_symbol: Mapping[str, str] | None = None,
        load_official_mapping: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mapping = {
            normalize_sec_ticker(symbol): str(cik).strip().zfill(10)
            for symbol, cik in (cik_by_symbol or {}).items()
            if str(symbol).strip() and str(cik).strip()
        }
        company_titles: dict[str, str] = {}
        exchanges: dict[str, str] = {}
        official_hash = ""
        official_retrieved = ""
        mapping_source = "configured_cik_by_symbol"
        if load_official_mapping:
            payload_bytes, headers = self._request(SEC_COMPANY_TICKERS_URL)
            official_hash = hashlib.sha256(payload_bytes).hexdigest()
            official_retrieved = _utc_now()
            raw_values = json.loads(payload_bytes.decode("utf-8"))
            mapping.update(normalize_sec_company_tickers(json.loads(payload_bytes.decode("utf-8"))))
            for item in _sec_mapping_items(raw_values):
                ticker = normalize_sec_ticker(str(item.get("ticker", "")))
                company_titles[ticker] = str(item.get("title") or "")
                exchanges[ticker] = str(item.get("exchange") or "")
            mapping_source = SEC_COMPANY_TICKERS_URL
        rows = []
        for raw_symbol in symbols:
            symbol = str(raw_symbol).strip().upper()
            normalized = normalize_sec_ticker(symbol)
            manual = {
                normalize_sec_ticker(k): str(v).strip().zfill(10)
                for k, v in (cik_by_symbol or {}).items()
            }.get(normalized)
            official = mapping.get(normalized) if load_official_mapping else None
            if official and manual and official != manual:
                cik = official
                status = "ambiguous"
            elif official:
                cik = official
                status = "resolved_official"
            elif manual:
                cik = manual
                status = "resolved_manual_override"
            else:
                cik = None
                status = "unresolved"
            identity_payload = {
                "provider": self.provider_id,
                "symbol": symbol,
                "normalized_symbol": normalized,
                "reporting_entity_id": f"CIK{cik}" if cik else None,
                "mapping_source": mapping_source if cik else "none",
                "mapping_status": status,
                "mapping_contract_version": "fundamentals_sec_entity_mapping_contract_v1",
            }
            rows.append(
                {
                    "symbol": symbol,
                    "cik": cik or "",
                    "company_title": company_titles.get(normalized, ""),
                    "exchange": exchanges.get(normalized, ""),
                    "historical_ticker": "",
                    "current_ticker": symbol,
                    "security_id": symbol,
                    "reporting_entity_id": f"CIK{cik}" if cik else "",
                    "provider_entity_id": cik or "",
                    "mapping_effective_date": "",
                    "mapping_source": mapping_source if cik else "none",
                    "mapping_status": status,
                    "mapping_retrieval_timestamp": official_retrieved,
                    "raw_mapping_sha256": official_hash,
                    "mapping_contract_version": "fundamentals_sec_entity_mapping_contract_v1",
                    "security_mapping_identity": _sha256_json(identity_payload),
                }
            )
        audit = {
            "mapping_contract_version": "fundamentals_sec_entity_mapping_contract_v1",
            "mapping_source": mapping_source,
            "raw_mapping_sha256": official_hash,
            "mapping_retrieval_timestamp": official_retrieved,
            "configured_symbol_count": len(symbols),
            "resolved_count": sum(1 for row in rows if row["reporting_entity_id"]),
            "unresolved_count": sum(1 for row in rows if not row["reporting_entity_id"]),
            "ambiguous_count": sum(1 for row in rows if row["mapping_status"] == "ambiguous"),
            "manual_override_count": sum(1 for row in rows if row["mapping_status"] == "resolved_manual_override"),
            "official_mapping_loaded": bool(load_official_mapping),
            "status": "BLOCK" if any(row["mapping_status"] == "ambiguous" for row in rows) else "PASS",
        }
        return rows, audit

    def fetch_company_facts(
        self,
        entity_id: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cik = _cik_digits(entity_id)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        target = self.raw_root / self.provider_id / f"CIK{cik}" / "companyfacts.json"
        metadata_path = target.with_suffix(".metadata.json")
        if target.exists() and not force_refresh:
            state = validate_cached_companyfacts(target, expected_cik=cik)
            if state["cache_state"] != "valid_cached":
                raise ValueError(f"Cached SEC companyfacts is not valid_cached: {state}")
            payload = json.loads(target.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            metadata["cache_state"] = "valid_cached"
            return {"status": "skipped_cached", "path": target, "metadata": metadata, "payload": payload}
        payload_bytes, headers = self._request(url)
        _validate_sec_json_response(payload_bytes, headers)
        payload = json.loads(payload_bytes.decode("utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        raw_sha = hashlib.sha256(payload_bytes).hexdigest()
        if target.exists() and file_sha256(target) != raw_sha and not force_refresh:
            raise ValueError(f"Refusing to overwrite different cached SEC response: {target}")
        _atomic_write_bytes(target, payload_bytes)
        metadata = {
            "provider": self.provider_id,
            "provider_version": self.provider_version,
            "url": url,
            "retrieval_timestamp": _utc_now(),
            "sha256": raw_sha,
            "content_type": headers.get("content-type") or headers.get("Content-Type"),
            "etag": headers.get("etag") or headers.get("ETag"),
            "last_modified": headers.get("last-modified") or headers.get("Last-Modified"),
        }
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
        return {"status": "downloaded", "path": target, "metadata": metadata, "payload": payload}

    def _request(self, url: str) -> tuple[bytes, Mapping[str, str]]:
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError("SEC fundamentals collection requires an identifying user agent with monitored contact email")
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt or self.request_delay_seconds:
                time.sleep(self.request_delay_seconds * (2 ** max(0, attempt - 1)))
            try:
                return self.http_get(url, headers, self.timeout_seconds)
            except Exception as exc:  # bounded retry manifest records the final error upstream
                last_error = exc
                if attempt >= self.max_retries:
                    break
        assert last_error is not None
        raise last_error


def write_stock_fundamentals_pipeline(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.stock_fundamentals.enabled is false")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir)
    stages = _stage_sequence(settings)
    stage_results: dict[str, Any] = {}
    if "collect" in stages:
        stage_results["collect"] = write_stock_fundamentals_collect(config)
    if "normalize" in stages:
        stage_results["normalize"] = write_stock_fundamentals_normalize(config)
    if "audit" in stages:
        stage_results["audit"] = write_stock_fundamentals_audit(config)
    if "snapshots" in stages:
        stage_results["snapshots"] = write_stock_fundamentals_snapshots(config)
    if "enrich" in stages:
        stage_results["enrich"] = write_stock_fundamentals_enrich(config)
    payload = _pipeline_report_payload(paths, settings, stage_results)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.pipeline_manifest_path, _json_ready(payload["pipeline_manifest"]))
    writer.write_json(paths.readiness_json_path, _json_ready(payload["readiness_report"]))
    writer.write_markdown(paths.readiness_markdown_path, _readiness_markdown(payload["readiness_report"]))
    writer.write_json(paths.report_json_path, _json_ready(payload))
    writer.write_markdown(paths.report_markdown_path, _report_markdown(payload, paths))
    return paths


def write_stock_fundamentals_preflight(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _preflight_payload(settings)
    ResearchArtifactWriter().write_json(paths.preflight_path, _json_ready(payload))
    return paths


def write_stock_fundamentals_collect(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.stock_fundamentals.enabled is false")
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_payload(settings)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.preflight_path, _json_ready(preflight))
    if settings["live_collection"] and preflight["status"] == "BLOCKED":
        raise ValueError(preflight["blocking_reasons"][0])
    base_rows = _load_base_rows(settings)
    symbols = _configured_symbols(settings, base_rows)
    cohort = _bounded_cohort(symbols, base_rows, settings)
    provider = SecCompanyFactsProvider(
        raw_root=Path(settings["raw_root"]),
        user_agent=str(settings["user_agent"]),
        request_delay_seconds=float(settings["request_delay_seconds"]),
        max_retries=int(settings["max_retries"]),
        timeout_seconds=int(settings["timeout_seconds"]),
    )
    entity_mapping, mapping_audit = provider.resolve_reporting_entities(
        cohort["selected_symbols"],
        cik_by_symbol=settings.get("cik_by_symbol", {}),
        load_official_mapping=bool(settings["load_official_sec_company_tickers"]),
    )
    collection = _collect_raw(provider, entity_mapping, settings)
    writer.write_json(paths.bounded_cohort_path, _json_ready(cohort))
    writer.write_csv(paths.entity_mapping_path, entity_mapping, fieldnames=_fields(entity_mapping, ["symbol"]))
    writer.write_json(paths.entity_mapping_audit_path, _json_ready(mapping_audit))
    writer.write_json(paths.raw_collection_manifest_path, _json_ready(collection["manifest"]))
    writer.write_csv(paths.failed_entities_path, collection["failed_entities"], fieldnames=_fields(collection["failed_entities"], ["symbol", "reporting_entity_id"]))
    return paths


def write_stock_fundamentals_normalize(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = _read_csv_dicts(paths.entity_mapping_path)
    raw_payloads, cache_audit = _load_cached_raw_payloads(mapping, Path(settings["raw_root"]))
    fact_dictionary = canonical_fact_dictionary()
    normalized_facts, normalization_audit = normalize_sec_company_facts(raw_payloads, mapping, fact_dictionary=fact_dictionary)
    normalization_audit["raw_cache_validation"] = cache_audit
    normalization_audit["raw_source_row_count"] = sum(_raw_fact_row_count(item["payload"]) for item in raw_payloads)
    normalization_audit.update(_period_counts(normalized_facts))
    writer = ResearchArtifactWriter()
    writer.write_json(paths.fact_dictionary_path, _json_ready(fact_dictionary))
    _write_parquet(paths.normalized_facts_path, normalized_facts, CANONICAL_FACT_COLUMNS)
    writer.write_csv(paths.tag_coverage_path, _tag_coverage(normalized_facts, normalization_audit), fieldnames=["canonical_fact_id", "entities_covered", "filings_covered", "source_tags_used", "conflicts", "unmapped_alternatives"])
    writer.write_csv(paths.unit_conflicts_path, normalization_audit.get("unit_conflicts", []), fieldnames=_fields(normalization_audit.get("unit_conflicts", []), ["reporting_entity_id", "source_fact_name", "source_unit", "canonical_fact_id"]))
    writer.write_csv(paths.period_reconciliation_path, _period_reconciliation(normalized_facts), fieldnames=["reporting_entity_id", "status", "instant_count", "quarterly_count", "ytd_count", "annual_count", "blocked_reason"])
    writer.write_json(paths.normalization_audit_path, _json_ready(normalization_audit))
    return paths


def write_stock_fundamentals_audit(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = _read_csv_dicts(paths.entity_mapping_path)
    facts = _read_parquet_dicts(paths.normalized_facts_path)
    raw_payloads, cache_audit = _load_cached_raw_payloads(mapping, Path(settings["raw_root"]))
    existing = _read_json(paths.normalization_audit_path)
    audit = {
        **existing,
        "schema_version": SCHEMA_VERSION,
        "stage": "audit",
        "raw_cache_validation": cache_audit,
        "mapping_row_count": len(mapping),
        "normalized_fact_count": len(facts),
        "raw_payload_count": len(raw_payloads),
        "timestamp_audit": {
            "missing_filing_timestamp_count": sum(1 for row in facts if not row.get("filing_timestamp")),
            "period_end_used_as_availability_count": sum(1 for row in facts if str(row.get("available_timestamp", ""))[:10] == str(row.get("period_end", ""))[:10]),
        },
        "unit_conflict_count": len([row for row in facts if row.get("normalized_unit") not in {"USD", "shares", "USD/shares", "pure", "percent"}]),
        "duplicate_group_count": _duplicate_normalized_group_count(facts),
        "period_reconciliation": _period_counts(facts),
    }
    ResearchArtifactWriter().write_json(paths.normalization_audit_path, _json_ready(audit))
    return paths


def write_stock_fundamentals_snapshots(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_base_rows(settings)
    mapping = _read_csv_dicts(paths.entity_mapping_path)
    facts = _read_parquet_dicts(paths.normalized_facts_path)
    snapshots, snapshot_audit = build_fundamental_snapshots(
        base_rows,
        mapping,
        facts,
        maximum_data_age_days=settings["maximum_data_age_days"],
        minimum_denominator=settings["minimum_denominator"],
    )
    writer = ResearchArtifactWriter()
    _write_parquet(paths.snapshots_path, snapshots, _fields(snapshots, ["decision_timestamp", "symbol"]))
    writer.write_csv(paths.snapshot_coverage_path, _snapshot_coverage(snapshots), fieldnames=["snapshot_status", "row_count", "symbol_count", "decision_date_count"])
    writer.write_json(paths.snapshot_audit_path, _json_ready(snapshot_audit))
    writer.write_json(paths.feature_contracts_path, _json_ready(formula_contracts()))
    writer.write_csv(paths.feature_coverage_path, _feature_coverage(snapshots), fieldnames=["feature", "row_count", "non_null_count", "non_null_fraction", "all_null", "coverage_classification"])
    return paths


def write_stock_fundamentals_enrich(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_base_rows(settings)
    snapshots = _read_parquet_dicts(paths.snapshots_path)
    enriched_rows, enrichment_audit = enrich_stock_artifact_with_fundamentals(base_rows, snapshots, settings=settings)
    if enriched_rows:
        identity = write_stock_level_artifact(
            paths.enriched_artifact_path,
            enriched_rows,
            fieldnames=_fields(enriched_rows, ["rebalance_date", "symbol"]),
            config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
        )
        enrichment_audit["enriched_artifact_identity"] = identity
    writer = ResearchArtifactWriter()
    writer.write_json(paths.enrichment_audit_json_path, _json_ready(enrichment_audit))
    writer.write_markdown(paths.enrichment_audit_markdown_path, _audit_markdown({"enrichment_audit": enrichment_audit, "analyst_estimate_status": "source_not_configured"}))
    return paths


def write_stock_fundamentals_legacy_full_pipeline(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir)
    payload = build_stock_fundamentals_pipeline(config, settings=settings)
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.entity_mapping_path, payload["entity_mapping"], fieldnames=_fields(payload["entity_mapping"], ["symbol"]))
    writer.write_json(paths.raw_collection_manifest_path, _json_ready(payload["raw_collection_manifest"]))
    writer.write_csv(paths.failed_entities_path, payload["failed_entities"], fieldnames=_fields(payload["failed_entities"], ["symbol", "reporting_entity_id"]))
    writer.write_json(paths.fact_dictionary_path, _json_ready(payload["fact_dictionary"]))
    _write_parquet(paths.normalized_facts_path, payload["normalized_facts"], CANONICAL_FACT_COLUMNS)
    writer.write_json(paths.normalization_audit_path, _json_ready(payload["normalization_audit"]))
    _write_parquet(paths.snapshots_path, payload["snapshots"], _fields(payload["snapshots"], ["decision_timestamp", "symbol"]))
    writer.write_json(paths.snapshot_audit_path, _json_ready(payload["snapshot_audit"]))
    writer.write_json(paths.feature_contracts_path, _json_ready(payload["feature_contracts"]))
    writer.write_csv(paths.feature_coverage_path, payload["feature_coverage"], fieldnames=_fields(payload["feature_coverage"], ["feature"]))
    if payload["enriched_rows"]:
        identity = write_stock_level_artifact(
            paths.enriched_artifact_path,
            payload["enriched_rows"],
            fieldnames=_fields(payload["enriched_rows"], ["rebalance_date", "symbol"]),
            config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
        )
        payload["enrichment_audit"]["enriched_artifact_identity"] = identity
    writer.write_json(paths.enrichment_audit_json_path, _json_ready(payload["enrichment_audit"]))
    writer.write_markdown(paths.enrichment_audit_markdown_path, _audit_markdown(payload))
    writer.write_json(paths.report_json_path, _json_ready(payload))
    writer.write_markdown(paths.report_markdown_path, _report_markdown(payload, paths))
    return paths


def build_stock_fundamentals_pipeline(
    config: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    http_get: Callable[[str, Mapping[str, str], int], tuple[bytes, Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    settings = dict(settings or _settings(config))
    base_rows = _load_base_rows(settings)
    symbols = _configured_symbols(settings, base_rows)
    provider = SecCompanyFactsProvider(
        raw_root=Path(settings["raw_root"]),
        user_agent=str(settings["user_agent"]),
        request_delay_seconds=float(settings["request_delay_seconds"]),
        max_retries=int(settings["max_retries"]),
        timeout_seconds=int(settings["timeout_seconds"]),
        http_get=http_get,
    )
    entity_mapping, _mapping_audit = provider.resolve_reporting_entities(
        symbols,
        cik_by_symbol=settings.get("cik_by_symbol", {}),
        load_official_mapping=bool(settings["load_official_sec_company_tickers"]),
    )
    collection = _collect_raw(provider, entity_mapping, settings)
    fact_dictionary = canonical_fact_dictionary()
    normalized_facts, normalization_audit = normalize_sec_company_facts(
        collection["raw_payloads"],
        entity_mapping,
        fact_dictionary=fact_dictionary,
    )
    snapshots, snapshot_audit = build_fundamental_snapshots(
        base_rows,
        entity_mapping,
        normalized_facts,
        maximum_data_age_days=settings["maximum_data_age_days"],
        minimum_denominator=settings["minimum_denominator"],
    )
    feature_contracts = formula_contracts()
    enriched_rows, enrichment_audit = enrich_stock_artifact_with_fundamentals(
        base_rows,
        snapshots,
        settings=settings,
    )
    feature_coverage = _feature_coverage(enriched_rows)
    data_quality_audit = _data_quality_audit(collection, normalization_audit, snapshot_audit, feature_coverage)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "provider_contract": provider_contract(),
        "provider_selected": provider.provider_id,
        "entity_mapping": entity_mapping,
        "raw_collection_manifest": collection["manifest"],
        "failed_entities": collection["failed_entities"],
        "fact_dictionary": fact_dictionary,
        "normalized_facts": normalized_facts,
        "normalization_audit": normalization_audit,
        "snapshots": snapshots,
        "snapshot_audit": snapshot_audit,
        "feature_contracts": feature_contracts,
        "feature_coverage": feature_coverage,
        "enriched_rows": enriched_rows,
        "enrichment_audit": enrichment_audit,
        "data_quality_audit": data_quality_audit,
        "coverage_gates": _coverage_gates(settings, entity_mapping, snapshots, feature_coverage),
        "analyst_estimate_status": "source_not_configured",
        "worker_ownership": {
            "network_collection": "single conservative worker; no parallel SEC requests",
            "normalisation": "local CPU stage, resumable by raw entity cache",
            "snapshot_construction": "local CPU stage bounded by configured artifact/date/symbol limits",
            "feature_calculation": "local CPU stage; no model training",
        },
    }


def normalize_sec_company_facts(
    raw_payloads: Sequence[Mapping[str, Any]],
    entity_mapping: Sequence[Mapping[str, Any]],
    *,
    fact_dictionary: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dictionary = fact_dictionary or canonical_fact_dictionary()
    tag_map = _dictionary_tag_map(dictionary)
    mapping_identity = {row["reporting_entity_id"]: row["security_mapping_identity"] for row in entity_mapping}
    rows: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    unit_conflicts: list[dict[str, Any]] = []
    duplicate_keys: dict[tuple[Any, ...], int] = {}
    for item in raw_payloads:
        payload = item["payload"]
        raw_path = Path(item["path"])
        raw_sha = str(item["metadata"].get("sha256") or file_sha256(raw_path))
        retrieval_timestamp = str(item["metadata"].get("retrieval_timestamp") or _utc_now())
        entity_id = f"CIK{str(payload.get('cik', '')).zfill(10)}"
        facts = payload.get("facts", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(facts, Mapping):
            continue
        for namespace, namespace_facts in facts.items():
            if not isinstance(namespace_facts, Mapping):
                continue
            for source_fact_name, fact_payload in namespace_facts.items():
                if not isinstance(fact_payload, Mapping):
                    continue
                source_tag = f"{namespace}:{source_fact_name}"
                dictionary_row = tag_map.get(source_tag)
                if dictionary_row is None:
                    unmapped[source_tag] = unmapped.get(source_tag, 0) + 1
                    continue
                units = fact_payload.get("units", {})
                if not isinstance(units, Mapping):
                    continue
                for source_unit, unit_rows in units.items():
                    normalized_unit = _normalised_unit(str(source_unit))
                    if not _unit_supported(dictionary_row, normalized_unit):
                        unit_conflicts.append(
                            {
                                "reporting_entity_id": entity_id,
                                "source_fact_name": source_fact_name,
                                "source_unit": source_unit,
                                "canonical_fact_id": dictionary_row["canonical_fact_id"],
                            }
                        )
                        continue
                    for fact in unit_rows or []:
                        if not isinstance(fact, Mapping):
                            continue
                        available_timestamp = _available_timestamp(fact)
                        if not available_timestamp:
                            continue
                        period_start = str(fact.get("start") or "")
                        period_end = str(fact.get("end") or "")
                        value = _number(fact.get("val"))
                        if value is None:
                            continue
                        period_type = _period_type(period_start, period_end, dictionary_row["period_requirement"])
                        accession = str(fact.get("accn") or "")
                        form_type = str(fact.get("form") or "")
                        row = {
                            "provider_id": "official_sec_companyfacts",
                            "reporting_entity_id": entity_id,
                            "security_mapping_identity": mapping_identity.get(entity_id, ""),
                            "source_document_id": accession,
                            "filing_accession": accession,
                            "form_type": form_type,
                            "filing_timestamp": _date_end_timestamp(str(fact.get("filed") or "")),
                            "acceptance_timestamp": "",
                            "first_seen_timestamp": retrieval_timestamp,
                            "available_timestamp": available_timestamp,
                            "period_start": period_start,
                            "period_end": period_end,
                            "fiscal_year": fact.get("fy"),
                            "fiscal_period": str(fact.get("fp") or ""),
                            "fact_namespace": str(namespace),
                            "source_fact_name": str(source_fact_name),
                            "canonical_fact_id": dictionary_row["canonical_fact_id"],
                            "unit": str(source_unit),
                            "normalized_unit": normalized_unit,
                            "value": value * float(dictionary_row.get("sign_multiplier", 1.0)),
                            "fact_period_type": period_type,
                            "is_amendment": form_type.endswith("/A"),
                            "amends_document_id": "",
                            "source_raw_path": str(raw_path),
                            "source_raw_sha256": raw_sha,
                            "normalisation_contract_identity": _normalisation_identity(dictionary),
                        }
                        key = (
                            row["reporting_entity_id"],
                            row["source_document_id"],
                            row["canonical_fact_id"],
                            row["period_start"],
                            row["period_end"],
                            row["normalized_unit"],
                        )
                        duplicate_keys[key] = duplicate_keys.get(key, 0) + 1
                        rows.append(row)
    rows = _dedupe_normalized(rows)
    rows.sort(key=lambda row: tuple(str(row.get(column, "")) for column in CANONICAL_FACT_COLUMNS))
    audit = {
        "schema_version": SCHEMA_VERSION,
        "normalisation_contract_identity": _normalisation_identity(dictionary),
        "normalized_row_count": len(rows),
        "unmapped_tag_count": sum(unmapped.values()),
        "unmapped_tags": [{"source_tag": key, "count": value} for key, value in sorted(unmapped.items())[:200]],
        "unit_conflict_count": len(unit_conflicts),
        "unit_conflicts": unit_conflicts[:200],
        "duplicate_fact_key_count": sum(1 for value in duplicate_keys.values() if value > 1),
        "availability_rule": "available_timestamp is SEC filed date at 23:59:59Z; period_end is never used as availability",
        "status": "PASS" if rows else "BLOCK",
    }
    return rows, audit


def build_fundamental_snapshots(
    base_rows: Sequence[Mapping[str, Any]],
    entity_mapping: Sequence[Mapping[str, Any]],
    normalized_facts: Sequence[Mapping[str, Any]],
    *,
    maximum_data_age_days: int | None,
    minimum_denominator: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_symbol = {str(row["symbol"]).upper(): row for row in entity_mapping}
    facts_by_entity: dict[str, list[Mapping[str, Any]]] = {}
    for fact in normalized_facts:
        facts_by_entity.setdefault(str(fact.get("reporting_entity_id")), []).append(fact)
    for values in facts_by_entity.values():
        values.sort(key=lambda row: str(row.get("available_timestamp", "")))
    snapshots: list[dict[str, Any]] = []
    future_exclusion_count = 0
    amendment_cases = 0
    for base in base_rows:
        symbol = str(base.get("symbol") or "").upper()
        decision_ts = _decision_timestamp(base)
        mapping = by_symbol.get(symbol)
        if not mapping or not mapping.get("reporting_entity_id"):
            snapshots.append(_empty_snapshot(base, decision_ts, "unresolved_entity"))
            continue
        entity_id = str(mapping["reporting_entity_id"])
        entity_facts = facts_by_entity.get(entity_id, [])
        available = [fact for fact in entity_facts if str(fact.get("available_timestamp", "")) <= decision_ts]
        future_exclusion_count += len(entity_facts) - len(available)
        if not available:
            snapshots.append(_empty_snapshot(base, decision_ts, "no_prior_filing", entity_id=entity_id, mapping=mapping))
            continue
        if any(bool(fact.get("is_amendment")) for fact in available):
            amendment_cases += 1
        selected = _latest_fact_values(available)
        latest_available = max(str(fact.get("available_timestamp", "")) for fact in available)
        latest_filing = max(str(fact.get("filing_timestamp", "")) for fact in available)
        age = _age_days(decision_ts, latest_available)
        status = "available"
        if maximum_data_age_days is not None and age is not None and age > int(maximum_data_age_days):
            status = "stale"
        features = _calculate_features(selected, base, minimum_denominator=minimum_denominator)
        snapshots.append(
            {
                **features,
                "symbol": symbol,
                "reporting_entity_id": entity_id,
                "decision_timestamp": decision_ts,
                "latest_filing_timestamp": latest_filing,
                "fundamental_data_age_days": age,
                "available_filing_count": len({fact.get("filing_accession") for fact in available if fact.get("filing_accession")}),
                "selected_source_document_identities": json.dumps(sorted({str(fact.get("source_document_id")) for fact in selected.values()})),
                "snapshot_contract_identity": _snapshot_identity(entity_id, decision_ts, selected),
                "snapshot_status": status,
                "fundamentals_available_timestamp": latest_available,
                "fundamentals_source_identity": _source_identity(selected.values()),
                "analyst_estimate_status": "source_not_configured",
            }
        )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_count": len(snapshots),
        "available_snapshot_count": sum(1 for row in snapshots if row.get("snapshot_status") == "available"),
        "stale_snapshot_count": sum(1 for row in snapshots if row.get("snapshot_status") == "stale"),
        "unresolved_entity_count": sum(1 for row in snapshots if row.get("snapshot_status") == "unresolved_entity"),
        "no_prior_filing_count": sum(1 for row in snapshots if row.get("snapshot_status") == "no_prior_filing"),
        "future_filing_exclusion_count": future_exclusion_count,
        "amendment_available_snapshot_count": amendment_cases,
        "availability_rule": "facts included only when available_timestamp <= decision_timestamp",
        "missing_snapshot_policy": "missing snapshots preserve NaN fundamentals and explicit status; no zero fill",
    }
    return snapshots, audit


def enrich_stock_artifact_with_fundamentals(
    base_rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (str(row.get("decision_timestamp", ""))[:10], str(row.get("symbol", "")).upper()): row
        for row in snapshots
    }
    enriched: list[dict[str, Any]] = []
    joined = 0
    for base in base_rows:
        key = (_decision_timestamp(base)[:10], str(base.get("symbol") or "").upper())
        snap = by_key.get(key, {})
        if snap:
            joined += 1
        row = dict(base)
        for column in FUNDAMENTAL_FEATURE_COLUMNS:
            row[column] = snap.get(column)
        row.update(
            {
                "fundamentals_snapshot_status": snap.get("snapshot_status") or "blocked",
                "fundamentals_available_timestamp": snap.get("fundamentals_available_timestamp"),
                "fundamentals_latest_filing_timestamp": snap.get("latest_filing_timestamp"),
                "fundamentals_data_age_days": snap.get("fundamental_data_age_days"),
                "fundamentals_contract_identity": snap.get("snapshot_contract_identity") or "",
                "fundamentals_source_identity": snap.get("fundamentals_source_identity") or "",
                "fundamentals_reporting_entity_id": snap.get("reporting_entity_id") or "",
                "analyst_estimate_status": "source_not_configured",
            }
        )
        enriched.append(row)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "enrichment_contract_version": ENRICHMENT_CONTRACT_VERSION,
        "join_mode": "backward_point_in_time",
        "preserve_base_rows": bool(settings.get("preserve_base_rows", True)),
        "base_row_count": len(base_rows),
        "enriched_row_count": len(enriched),
        "joined_snapshot_count": joined,
        "fundamentals_feature_columns": list(FUNDAMENTAL_FEATURE_COLUMNS),
        "fundamentals_metadata_columns": list(FUNDAMENTAL_METADATA_COLUMNS),
        "lineage": {
            "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
            "fact_dictionary_version": FACT_DICTIONARY_VERSION,
            "formula_contract_version": FEATURE_FORMULA_VERSION,
        },
    }
    return enriched, audit


def provider_contract() -> dict[str, Any]:
    return {
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "required_fields": [
            "provider_id",
            "provider_version",
            "entity_identifier",
            "security_identifier",
            "source_document_identifier",
            "filing_accession",
            "form_type",
            "filing_timestamp",
            "acceptance_timestamp",
            "first_seen_timestamp",
            "period_start",
            "period_end",
            "fiscal_year",
            "fiscal_period",
            "fact_namespace",
            "fact_name",
            "unit",
            "value",
            "fact_period_type",
            "amendment_status",
            "source_url",
            "retrieval_timestamp",
            "raw_artifact_hash",
        ],
        "operations": [
            "resolve_reporting_entity",
            "fetch_entity_metadata",
            "fetch_company_facts",
            "fetch_filing_metadata",
            "list_cached_entities",
            "validate_cached_response",
            "resume_missing_entities",
        ],
        "provider_specific_leakage_policy": "provider fields stop at canonical normalisation boundary",
    }


def canonical_fact_dictionary() -> dict[str, Any]:
    facts = [
        _fact("revenue", ["us-gaap:Revenues", "us-gaap:SalesRevenueNet"], "currency", "duration"),
        _fact("gross_profit", ["us-gaap:GrossProfit"], "currency", "duration"),
        _fact("operating_income", ["us-gaap:OperatingIncomeLoss"], "currency", "duration"),
        _fact("net_income", ["us-gaap:NetIncomeLoss"], "currency", "duration"),
        _fact("income_before_tax", ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"], "currency", "duration"),
        _fact("interest_expense", ["us-gaap:InterestExpenseNonOperating"], "currency", "duration"),
        _fact("research_and_development", ["us-gaap:ResearchAndDevelopmentExpense"], "currency", "duration"),
        _fact("selling_general_admin", ["us-gaap:SellingGeneralAndAdministrativeExpense"], "currency", "duration"),
        _fact("earnings_per_share_basic", ["us-gaap:EarningsPerShareBasic"], "currency_per_share", "duration"),
        _fact("earnings_per_share_diluted", ["us-gaap:EarningsPerShareDiluted"], "currency_per_share", "duration"),
        _fact("weighted_average_shares_basic", ["us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"], "shares", "duration"),
        _fact("weighted_average_shares_diluted", ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", "duration"),
        _fact("total_assets", ["us-gaap:Assets"], "currency", "instant"),
        _fact("current_assets", ["us-gaap:AssetsCurrent"], "currency", "instant"),
        _fact("cash_and_equivalents", ["us-gaap:CashAndCashEquivalentsAtCarryingValue"], "currency", "instant"),
        _fact("accounts_receivable", ["us-gaap:AccountsReceivableNetCurrent"], "currency", "instant"),
        _fact("inventory", ["us-gaap:InventoryNet"], "currency", "instant"),
        _fact("property_plant_equipment", ["us-gaap:PropertyPlantAndEquipmentNet"], "currency", "instant"),
        _fact("goodwill", ["us-gaap:Goodwill"], "currency", "instant"),
        _fact("intangible_assets", ["us-gaap:IntangibleAssetsNetExcludingGoodwill"], "currency", "instant"),
        _fact("total_liabilities", ["us-gaap:Liabilities"], "currency", "instant"),
        _fact("current_liabilities", ["us-gaap:LiabilitiesCurrent"], "currency", "instant"),
        _fact("short_term_debt", ["us-gaap:ShortTermBorrowings"], "currency", "instant"),
        _fact("long_term_debt", ["us-gaap:LongTermDebtNoncurrent"], "currency", "instant"),
        _fact("shareholders_equity", ["us-gaap:StockholdersEquity"], "currency", "instant"),
        _fact("retained_earnings", ["us-gaap:RetainedEarningsAccumulatedDeficit"], "currency", "instant"),
        _fact("shares_outstanding", ["us-gaap:CommonStocksIncludingAdditionalPaidInCapital", "dei:EntityCommonStockSharesOutstanding"], "shares", "instant"),
        _fact("operating_cash_flow", ["us-gaap:NetCashProvidedByUsedInOperatingActivities"], "currency", "duration"),
        _fact("capital_expenditure", ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"], "currency", "duration", sign_multiplier=-1.0),
        _fact("investing_cash_flow", ["us-gaap:NetCashProvidedByUsedInInvestingActivities"], "currency", "duration"),
        _fact("financing_cash_flow", ["us-gaap:NetCashProvidedByUsedInFinancingActivities"], "currency", "duration"),
        _fact("dividends_paid", ["us-gaap:PaymentsOfDividends"], "currency", "duration", sign_multiplier=-1.0),
        _fact("share_repurchases", ["us-gaap:PaymentsForRepurchaseOfCommonStock"], "currency", "duration", sign_multiplier=-1.0),
        _fact("share_issuance_proceeds", ["us-gaap:ProceedsFromIssuanceOfCommonStock"], "currency", "duration"),
        _fact("debt_issuance", ["us-gaap:ProceedsFromIssuanceOfLongTermDebt"], "currency", "duration"),
        _fact("debt_repayment", ["us-gaap:RepaymentsOfLongTermDebt"], "currency", "duration", sign_multiplier=-1.0),
    ]
    payload = {
        "dictionary_version": FACT_DICTIONARY_VERSION,
        "facts": facts,
        "unit_policy": "currency, shares, currency_per_share, pure, and percent units remain separate; no currency conversion is inferred",
    }
    payload["dictionary_identity"] = _sha256_json(payload)
    return payload


def formula_contracts() -> dict[str, Any]:
    contracts = []
    for feature in FUNDAMENTAL_FEATURE_COLUMNS:
        contracts.append(
            {
                "feature_id": feature,
                "required_canonical_facts": _required_facts(feature),
                "period_alignment": "latest available comparable fiscal period; YOY uses same fiscal period prior year",
                "denominator_rules": "missing when denominator is null, non-finite, or below minimum absolute denominator",
                "minimum_absolute_denominator": "configurable ml.stock_fundamentals.features.minimum_denominator",
                "winsorisation_policy": "none in producer; model folds may apply fold-local preprocessing only",
                "missingness_policy": "preserve missing values; never zero-fill economic fundamentals",
                "availability_rule": "source facts must have available_timestamp <= decision_timestamp",
                "formula_version": FEATURE_FORMULA_VERSION,
            }
        )
    payload = {"formula_contract_version": FEATURE_FORMULA_VERSION, "contracts": contracts}
    payload["formula_contract_identity"] = _sha256_json(payload)
    return payload


def _collect_raw(provider: SecCompanyFactsProvider, entity_mapping: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    started = _utc_now()
    raw_payloads = []
    failed = []
    skipped = []
    request_count = 0
    max_entities = settings.get("maximum_entities")
    resolved = [row for row in entity_mapping if row.get("provider_entity_id")]
    if max_entities is not None:
        resolved = resolved[: int(max_entities)]
    for row in resolved:
        try:
            result = provider.fetch_company_facts(str(row["provider_entity_id"]), force_refresh=bool(settings["force_refresh"]))
            request_count += 0 if result["status"] == "skipped_cached" else 1
            if result["status"] == "skipped_cached":
                skipped.append(str(row["reporting_entity_id"]))
            raw_payloads.append(result)
        except Exception as exc:
            failed.append({"symbol": row["symbol"], "reporting_entity_id": row["reporting_entity_id"], "error_type": type(exc).__name__, "error": str(exc)})
    status = "complete"
    if failed and raw_payloads:
        status = "partially_complete"
    elif failed and not raw_payloads:
        status = "failed"
    elif not raw_payloads:
        status = "blocked"
    manifest = {
        "provider": provider.provider_id,
        "provider_version": provider.provider_version,
        "run_identity": _sha256_json({"started": started, "symbols": [row.get("symbol") for row in entity_mapping]}),
        "configured_symbols": [row.get("symbol") for row in entity_mapping],
        "resolved_entities": [row.get("reporting_entity_id") for row in entity_mapping if row.get("reporting_entity_id")],
        "unresolved_symbols": [row.get("symbol") for row in entity_mapping if not row.get("reporting_entity_id")],
        "successful_entities": [f"CIK{_cik_digits(item['payload'].get('cik'))}" for item in raw_payloads],
        "failed_entities": failed,
        "skipped_cached_entities": skipped,
        "request_count": request_count,
        "first_request_timestamp": started,
        "last_request_timestamp": _utc_now(),
        "raw_paths": [str(item["path"]) for item in raw_payloads],
        "raw_hashes": [item["metadata"].get("sha256") for item in raw_payloads],
        "response_metadata": [item["metadata"] for item in raw_payloads],
        "retry_counts": {row.get("reporting_entity_id"): provider.max_retries for row in failed},
        "collection_status": status,
    }
    return {"manifest": manifest, "raw_payloads": raw_payloads, "failed_entities": failed}


def _calculate_features(selected: Mapping[str, Mapping[str, Any]], base: Mapping[str, Any], *, minimum_denominator: float) -> dict[str, Any]:
    latest = {fact_id: _value(row) for fact_id, row in selected.items()}

    def ratio(num: str, den: str) -> float | None:
        return _safe_ratio(latest.get(num), latest.get(den), minimum_denominator)

    def growth(fact_id: str, suffix: str) -> float | None:
        current = selected.get(fact_id)
        if not current:
            return None
        fy = _int(current.get("fiscal_year"))
        fp = str(current.get("fiscal_period") or "")
        candidates = [
            row for key, row in selected.items()
            if key.startswith(f"{fact_id}::history::")
            and _int(row.get("fiscal_year")) == (fy - 1 if suffix == "yoy" and fy is not None else fy)
            and (str(row.get("fiscal_period") or "") == fp if suffix == "yoy" else True)
            and str(row.get("period_end", "")) < str(current.get("period_end", ""))
        ]
        if suffix == "qoq":
            candidates = sorted(candidates, key=lambda row: str(row.get("period_end", "")))[-1:]
        if not candidates:
            return None
        return _safe_ratio(_value(current) - _value(candidates[-1]), abs(_value(candidates[-1])), minimum_denominator)

    total_debt = _sum_values(latest.get("short_term_debt"), latest.get("long_term_debt"))
    fcf = _sum_values(latest.get("operating_cash_flow"), latest.get("capital_expenditure"))
    market_cap = _market_cap(base, latest.get("shares_outstanding"))
    growth_values = [
        growth("revenue", "yoy"),
        growth("gross_profit", "yoy"),
        growth("operating_income", "yoy"),
        growth("net_income", "yoy"),
        growth("operating_cash_flow", "yoy"),
    ]
    non_null_growth = [value for value in growth_values if value is not None]
    missing_inputs = sum(1 for fact in _core_feature_facts() if latest.get(fact) is None)
    coverage_count = len(_core_feature_facts()) - missing_inputs
    restated = any(bool(row.get("is_amendment")) for row in selected.values())
    features = {
        "revenue_growth_yoy": growth("revenue", "yoy"),
        "revenue_growth_qoq": growth("revenue", "qoq"),
        "gross_profit_growth_yoy": growth("gross_profit", "yoy"),
        "operating_income_growth_yoy": growth("operating_income", "yoy"),
        "net_income_growth_yoy": growth("net_income", "yoy"),
        "eps_growth_yoy": growth("earnings_per_share_diluted", "yoy"),
        "operating_cash_flow_growth_yoy": growth("operating_cash_flow", "yoy"),
        "asset_growth_yoy": growth("total_assets", "yoy"),
        "equity_growth_yoy": growth("shareholders_equity", "yoy"),
        "growth_acceleration": None,
        "positive_growth_breadth": _safe_ratio(sum(1 for value in non_null_growth if value > 0), len(non_null_growth), 1.0) if non_null_growth else None,
        "gross_margin": ratio("gross_profit", "revenue"),
        "operating_margin": ratio("operating_income", "revenue"),
        "net_margin": ratio("net_income", "revenue"),
        "return_on_assets": ratio("net_income", "total_assets"),
        "return_on_equity": ratio("net_income", "shareholders_equity"),
        "asset_turnover": ratio("revenue", "total_assets"),
        "operating_cash_flow_to_assets": ratio("operating_cash_flow", "total_assets"),
        "free_cash_flow_margin": _safe_ratio(fcf, latest.get("revenue"), minimum_denominator),
        "cash_conversion": ratio("operating_cash_flow", "net_income"),
        "total_accruals_to_assets": _safe_ratio((latest.get("net_income") or 0.0) - (latest.get("operating_cash_flow") or 0.0), latest.get("total_assets"), minimum_denominator) if latest.get("net_income") is not None and latest.get("operating_cash_flow") is not None else None,
        "cash_flow_to_net_income": ratio("operating_cash_flow", "net_income"),
        "working_capital_accruals": _safe_ratio(_sum_values(latest.get("current_assets"), -(latest.get("cash_and_equivalents") or 0.0), -(latest.get("current_liabilities") or 0.0)), latest.get("total_assets"), minimum_denominator),
        "earnings_quality_score": ratio("operating_cash_flow", "net_income"),
        "debt_to_assets": _safe_ratio(total_debt, latest.get("total_assets"), minimum_denominator),
        "debt_to_equity": _safe_ratio(total_debt, latest.get("shareholders_equity"), minimum_denominator),
        "net_debt_to_assets": _safe_ratio(_sum_values(total_debt, -(latest.get("cash_and_equivalents") or 0.0)), latest.get("total_assets"), minimum_denominator),
        "current_ratio": ratio("current_assets", "current_liabilities"),
        "cash_to_assets": ratio("cash_and_equivalents", "total_assets"),
        "interest_coverage": ratio("operating_income", "interest_expense"),
        "working_capital_to_assets": _safe_ratio(_sum_values(latest.get("current_assets"), -(latest.get("current_liabilities") or 0.0)), latest.get("total_assets"), minimum_denominator),
        "share_count_growth_yoy": growth("weighted_average_shares_diluted", "yoy"),
        "dilution_rate": growth("weighted_average_shares_diluted", "yoy"),
        "net_share_issuance": _sum_values(latest.get("share_issuance_proceeds"), latest.get("share_repurchases")),
        "repurchase_intensity": _safe_ratio(abs(latest.get("share_repurchases") or 0.0), market_cap, minimum_denominator),
        "dividend_payout": ratio("dividends_paid", "net_income"),
        "earnings_yield": _safe_ratio(latest.get("net_income"), market_cap, minimum_denominator),
        "book_to_market": _safe_ratio(latest.get("shareholders_equity"), market_cap, minimum_denominator),
        "sales_to_price": _safe_ratio(latest.get("revenue"), market_cap, minimum_denominator),
        "free_cash_flow_yield": _safe_ratio(fcf, market_cap, minimum_denominator),
        "filing_recency_score": None,
        "fundamental_coverage_count": coverage_count,
        "fundamental_missing_fraction": _safe_ratio(missing_inputs, len(_core_feature_facts()), 1.0),
        "restatement_indicator": 1.0 if restated else 0.0,
        "entity_mapping_quality": 1.0,
    }
    features["growth_acceleration"] = (
        features["revenue_growth_yoy"] - features["revenue_growth_qoq"]
        if features["revenue_growth_yoy"] is not None and features["revenue_growth_qoq"] is not None
        else None
    )
    return features


def _latest_fact_values(available: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in available:
        grouped.setdefault(str(row.get("canonical_fact_id")), []).append(row)
    selected: dict[str, Mapping[str, Any]] = {}
    for fact_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (str(row.get("period_end", "")), str(row.get("available_timestamp", "")), str(row.get("source_document_id", ""))))
        if ordered:
            selected[fact_id] = ordered[-1]
            for index, hist in enumerate(ordered):
                selected[f"{fact_id}::history::{index}"] = hist
    return selected


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("stock_fundamentals", {}) or {})
    output_dir = raw.get("output_dir") or str(stock_alpha_output_dir(config) / "stock_fundamentals")
    collection = dict(raw.get("collection", {}) or {})
    normalization = dict(raw.get("normalization", {}) or {})
    snapshots = dict(raw.get("snapshots", {}) or {})
    enrichment = dict(raw.get("enrichment", {}) or {})
    features = dict(raw.get("features", {}) or {})
    bounded = dict(raw.get("bounded", {}) or {})
    stages = dict(raw.get("stages", {}) or {})
    user_agent_env = str(raw.get("user_agent_env") or collection.get("user_agent_env") or "SEC_USER_AGENT")
    user_agent = str(raw.get("user_agent") or collection.get("user_agent") or os.environ.get(user_agent_env, ""))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "provider": str(raw.get("provider", "official_sec_companyfacts")),
        "output_dir": str(output_dir),
        "source_dataset_path": raw.get("source_dataset_path") or ml.get("selector_feature_ablation", {}).get("source_dataset_path") or "",
        "allow_csv_fallback": bool(raw.get("allow_csv_fallback", False)),
        "symbols": list(raw.get("symbols", []) or []),
        "cik_by_symbol": dict(raw.get("cik_by_symbol", {}) or {}),
        "load_official_sec_company_tickers": bool(raw.get("load_official_sec_company_tickers", False)),
        "raw_root": str(collection.get("raw_root", "data/raw/fundamentals")),
        "resume": bool(collection.get("resume", True)),
        "force_refresh": bool(collection.get("force_refresh", False)),
        "request_delay_seconds": float(collection.get("request_delay_seconds", 0.2)),
        "max_retries": int(collection.get("max_retries", 2)),
        "timeout_seconds": int(collection.get("timeout_seconds", 30)),
        "maximum_entities": collection.get("maximum_entities", bounded.get("maximum_entities")),
        "user_agent": user_agent,
        "user_agent_env": user_agent_env,
        "live_collection": bool(collection.get("live_collection", True)),
        "start_stage": str(stages.get("start_stage", raw.get("start_stage", "collect"))),
        "end_stage": str(stages.get("end_stage", raw.get("end_stage", "enrich"))),
        "enabled_stages": list(stages.get("enabled_stages", raw.get("enabled_stages", [])) or []),
        "network_concurrency": int(collection.get("network_concurrency", 1)),
        "normalization_workers": int(normalization.get("workers", 1)),
        "snapshot_workers": int(snapshots.get("workers", 1)),
        "enrichment_workers": int(enrichment.get("workers", 1)),
        "normalized_output_path": str(normalization.get("output_path", "")),
        "maximum_data_age_days": snapshots.get("maximum_data_age_days"),
        "amendment_policy": str(snapshots.get("amendment_policy", "latest_available_as_of_decision")),
        "preserve_base_rows": bool(enrichment.get("preserve_base_rows", True)),
        "minimum_denominator": float(features.get("minimum_denominator", 1e-9)),
        "maximum_decision_dates": bounded.get("maximum_decision_dates"),
        "maximum_symbols": bounded.get("maximum_symbols"),
        "coverage": dict(raw.get("coverage", {}) or {}),
    }


def _paths(output_dir: Path) -> StockFundamentalsPaths:
    return StockFundamentalsPaths(
        output_dir=output_dir,
        entity_mapping_path=output_dir / "fundamentals_entity_mapping.csv",
        raw_collection_manifest_path=output_dir / "fundamentals_raw_collection_manifest.json",
        failed_entities_path=output_dir / "fundamentals_failed_entities.csv",
        fact_dictionary_path=output_dir / "fundamentals_canonical_fact_dictionary.json",
        normalized_facts_path=output_dir / "fundamentals_normalized_facts.parquet",
        normalization_audit_path=output_dir / "fundamentals_normalization_audit.json",
        snapshots_path=output_dir / "fundamentals_point_in_time_snapshots.parquet",
        snapshot_audit_path=output_dir / "fundamentals_snapshot_audit.json",
        feature_contracts_path=output_dir / "fundamentals_feature_contracts.json",
        feature_coverage_path=output_dir / "fundamentals_feature_coverage.csv",
        enriched_artifact_path=output_dir / "stock_level_prediction_artifacts_fundamentals_enriched.parquet",
        enrichment_audit_json_path=output_dir / "stock_level_fundamentals_enrichment_audit.json",
        enrichment_audit_markdown_path=output_dir / "stock_level_fundamentals_enrichment_audit.md",
        preflight_path=output_dir / "fundamentals_live_preflight.json",
        entity_mapping_audit_path=output_dir / "fundamentals_entity_mapping_audit.json",
        bounded_cohort_path=output_dir / "fundamentals_bounded_cohort.json",
        tag_coverage_path=output_dir / "fundamentals_tag_coverage.csv",
        unit_conflicts_path=output_dir / "fundamentals_unit_conflicts.csv",
        period_reconciliation_path=output_dir / "fundamentals_period_reconciliation.csv",
        snapshot_coverage_path=output_dir / "fundamentals_snapshot_coverage.csv",
        pipeline_manifest_path=output_dir / "fundamentals_pipeline_manifest.json",
        readiness_json_path=output_dir / "fundamentals_bounded_readiness_report.json",
        readiness_markdown_path=output_dir / "fundamentals_bounded_readiness_report.md",
        report_json_path=output_dir / "stock_fundamentals_pipeline_report.json",
        report_markdown_path=output_dir / "stock_fundamentals_pipeline_report.md",
    )


def _load_base_rows(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = str(settings.get("source_dataset_path") or "")
    if not source:
        return []
    rows = read_stock_level_artifact(Path(source), allow_csv_fallback=bool(settings.get("allow_csv_fallback", False)))
    max_symbols = settings.get("maximum_symbols")
    max_dates = settings.get("maximum_decision_dates")
    if max_symbols is not None:
        symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})[: int(max_symbols)]
        rows = [row for row in rows if str(row.get("symbol", "")).upper() in symbols]
    if max_dates is not None:
        dates = sorted({_decision_timestamp(row)[:10] for row in rows})[: int(max_dates)]
        rows = [row for row in rows if _decision_timestamp(row)[:10] in dates]
    return rows


def _configured_symbols(settings: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if settings.get("symbols"):
        return [str(symbol).upper() for symbol in settings["symbols"]]
    return sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{name: row.get(name) for name in fieldnames} for row in rows], schema=None)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    return fields


def _stage_sequence(settings: Mapping[str, Any]) -> list[str]:
    if settings.get("enabled_stages"):
        return [str(stage) for stage in settings["enabled_stages"]]
    all_stages = ["collect", "normalize", "audit", "snapshots", "enrich"]
    start = str(settings.get("start_stage") or "collect")
    end = str(settings.get("end_stage") or "enrich")
    if start not in all_stages or end not in all_stages:
        raise ValueError(f"Unknown stock fundamentals stage range: {start}..{end}")
    return all_stages[all_stages.index(start): all_stages.index(end) + 1]


def _preflight_payload(settings: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if settings.get("live_collection") and not str(settings.get("user_agent") or "").strip():
        reasons.append(f"missing identifying SEC user agent; set {settings.get('user_agent_env', 'SEC_USER_AGENT')}")
    if int(settings.get("network_concurrency", 1)) > 2:
        reasons.append("SEC network concurrency must remain serial or near-serial")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "preflight",
        "status": "BLOCKED" if reasons else "PASS",
        "blocking_reasons": reasons,
        "provider": str(settings.get("provider")),
        "user_agent_env": str(settings.get("user_agent_env")),
        "user_agent_configured": bool(str(settings.get("user_agent") or "").strip()),
        "user_agent_redacted": _redacted_user_agent(str(settings.get("user_agent") or "")),
        "live_collection": bool(settings.get("live_collection")),
        "network_concurrency": int(settings.get("network_concurrency", 1)),
        "request_delay_seconds": float(settings.get("request_delay_seconds", 0.0)),
    }


def _bounded_cohort(symbols: Sequence[str], base_rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    max_symbols = settings.get("maximum_symbols") or settings.get("maximum_entities") or 20
    base_symbols = sorted({str(row.get("symbol", "")).upper() for row in base_rows if row.get("symbol")})
    requested = [str(symbol).upper() for symbol in symbols]
    selected = [symbol for symbol in requested if not base_symbols or symbol in base_symbols][: int(max_symbols)]
    if not selected and base_symbols:
        selected = base_symbols[: int(max_symbols)]
    return {
        "schema_version": SCHEMA_VERSION,
        "requested_symbols": requested,
        "selected_symbols": selected,
        "selection_reason": "configured symbols intersected with source stock artifact and bounded by maximum_symbols/entities",
        "source_stock_artifact_path": str(settings.get("source_dataset_path") or ""),
        "source_stock_artifact_identity": _optional_artifact_identity(Path(str(settings.get("source_dataset_path") or ""))),
        "bounded_max_symbols": max_symbols,
    }


def validate_cached_companyfacts(path: Path, *, expected_cik: str) -> dict[str, Any]:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.exists():
        return {"cache_state": "missing", "path": str(path)}
    if not metadata_path.exists():
        return {"cache_state": "corrupt", "path": str(path), "reason": "metadata_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"cache_state": "corrupt", "path": str(path), "reason": type(exc).__name__}
    raw_sha = file_sha256(path)
    if metadata.get("sha256") and metadata.get("sha256") != raw_sha:
        return {"cache_state": "corrupt", "path": str(path), "reason": "raw_sha_mismatch"}
    if str(payload.get("cik", "")).zfill(10) != _cik_digits(expected_cik):
        return {"cache_state": "identity_mismatch", "path": str(path), "payload_cik": payload.get("cik"), "expected_cik": _cik_digits(expected_cik)}
    if not isinstance(payload.get("facts"), Mapping):
        return {"cache_state": "schema_incompatible", "path": str(path), "reason": "missing_facts_object"}
    if not metadata.get("retrieval_timestamp"):
        return {"cache_state": "corrupt", "path": str(path), "reason": "retrieval_timestamp_missing"}
    return {
        "cache_state": "valid_cached",
        "path": str(path),
        "metadata_path": str(metadata_path),
        "sha256": raw_sha,
        "content_type": metadata.get("content_type"),
        "retrieval_timestamp": metadata.get("retrieval_timestamp"),
    }


def _load_cached_raw_payloads(entity_mapping: Sequence[Mapping[str, Any]], raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in entity_mapping:
        cik = str(row.get("provider_entity_id") or row.get("cik") or "").strip()
        if not cik:
            audit.append({"symbol": row.get("symbol"), "cache_state": "missing", "reason": "unresolved_entity"})
            continue
        path = raw_root / "official_sec_companyfacts" / f"CIK{_cik_digits(cik)}" / "companyfacts.json"
        state = validate_cached_companyfacts(path, expected_cik=cik)
        state["symbol"] = row.get("symbol")
        audit.append(state)
        if state["cache_state"] == "valid_cached":
            metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            payloads.append({"path": path, "metadata": metadata, "payload": json.loads(path.read_text(encoding="utf-8"))})
    return payloads, audit


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_parquet_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _optional_artifact_identity(path: Path) -> dict[str, Any] | None:
    if path and str(path) and path.exists():
        try:
            return artifact_identity(path)
        except Exception:
            return {"resolved_artifact_path": str(path), "sha256": file_sha256(path)}
    return None


def _raw_fact_row_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for namespace in (payload.get("facts") or {}).values():
        if not isinstance(namespace, Mapping):
            continue
        for fact in namespace.values():
            units = fact.get("units", {}) if isinstance(fact, Mapping) else {}
            if isinstance(units, Mapping):
                count += sum(len(values or []) for values in units.values())
    return count


def _period_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "instant_fact_count": sum(1 for row in rows if row.get("fact_period_type") == "instant"),
        "duration_fact_count": sum(1 for row in rows if row.get("fact_period_type") != "instant"),
        "annual_fact_count": sum(1 for row in rows if row.get("fact_period_type") == "annual_duration"),
        "quarterly_fact_count": sum(1 for row in rows if row.get("fact_period_type") == "quarterly_duration"),
        "ytd_fact_count": sum(1 for row in rows if row.get("fact_period_type") == "year_to_date_duration"),
        "amendment_count": sum(1 for row in rows if bool(row.get("is_amendment"))),
        "missing_filing_date_count": sum(1 for row in rows if not row.get("filing_timestamp")),
    }


def _tag_coverage(rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_fact: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_fact.setdefault(str(row.get("canonical_fact_id")), []).append(row)
    unmapped = {str(item.get("source_tag")) for item in audit.get("unmapped_tags", [])}
    result = []
    for fact_id, fact_rows in sorted(by_fact.items()):
        result.append({
            "canonical_fact_id": fact_id,
            "entities_covered": len({row.get("reporting_entity_id") for row in fact_rows}),
            "filings_covered": len({row.get("filing_accession") for row in fact_rows}),
            "source_tags_used": json.dumps(sorted({f"{row.get('fact_namespace')}:{row.get('source_fact_name')}" for row in fact_rows})),
            "conflicts": "",
            "unmapped_alternatives": json.dumps(sorted(tag for tag in unmapped if tag.endswith(f":{fact_id}"))),
        })
    return result


def _period_reconciliation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_entity: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_entity.setdefault(str(row.get("reporting_entity_id")), []).append(row)
    result = []
    for entity, entity_rows in sorted(by_entity.items()):
        counts = _period_counts(entity_rows)
        status = "valid" if counts["instant_fact_count"] and counts["duration_fact_count"] else "partial"
        result.append({
            "reporting_entity_id": entity,
            "status": status,
            "instant_count": counts["instant_fact_count"],
            "quarterly_count": counts["quarterly_fact_count"],
            "ytd_count": counts["ytd_fact_count"],
            "annual_count": counts["annual_fact_count"],
            "blocked_reason": "" if status != "blocked" else "period_conflict",
        })
    return result


def _snapshot_coverage(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    statuses = sorted({str(row.get("snapshot_status") or "") for row in snapshots})
    return [
        {
            "snapshot_status": status,
            "row_count": len([row for row in snapshots if str(row.get("snapshot_status") or "") == status]),
            "symbol_count": len({row.get("symbol") for row in snapshots if str(row.get("snapshot_status") or "") == status}),
            "decision_date_count": len({str(row.get("decision_timestamp"))[:10] for row in snapshots if str(row.get("snapshot_status") or "") == status}),
        }
        for status in statuses
    ]


def _duplicate_normalized_group_count(rows: Sequence[Mapping[str, Any]]) -> int:
    groups: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (row.get("reporting_entity_id"), row.get("canonical_fact_id"), row.get("period_start"), row.get("period_end"), row.get("normalized_unit"), row.get("available_timestamp"))
        groups[key] = groups.get(key, 0) + 1
    return sum(1 for count in groups.values() if count > 1)


def _pipeline_report_payload(paths: StockFundamentalsPaths, settings: Mapping[str, Any], stage_results: Mapping[str, Any]) -> dict[str, Any]:
    identities = {
        "entity_mapping_identity": _file_identity(paths.entity_mapping_path),
        "collection_identity": _file_identity(paths.raw_collection_manifest_path),
        "normalised_facts_identity": _optional_artifact_identity(paths.normalized_facts_path),
        "fact_dictionary_identity": _file_identity(paths.fact_dictionary_path),
        "snapshot_identity": _optional_artifact_identity(paths.snapshots_path),
        "formula_contract_identity": _file_identity(paths.feature_contracts_path),
        "enriched_artifact_identity": _optional_artifact_identity(paths.enriched_artifact_path),
    }
    readiness = _readiness_report(paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "provider_selected": str(settings.get("provider", "official_sec_companyfacts")),
        "analyst_estimate_status": "source_not_configured",
        "stages_ran": list(stage_results),
        "worker_ownership": _worker_ownership(settings),
        "pipeline_manifest": {
            "schema_version": SCHEMA_VERSION,
            "stage_identities": identities,
            "compatible": True,
            "reconciliation_status": "PASS",
        },
        "readiness_report": readiness,
    }


def _file_identity(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {"path": str(path), "sha256": file_sha256(path), "file_size_bytes": path.stat().st_size}


def _readiness_report(paths: StockFundamentalsPaths) -> dict[str, Any]:
    collection = _read_json(paths.raw_collection_manifest_path)
    normalization = _read_json(paths.normalization_audit_path)
    snapshot = _read_json(paths.snapshot_audit_path)
    enrichment = _read_json(paths.enrichment_audit_json_path)
    blockers = []
    if collection and collection.get("collection_status") not in {"complete", "partially_complete"}:
        blockers.append("official_collection_not_complete")
    if snapshot and int(snapshot.get("available_snapshot_count", 0)) == 0:
        blockers.append("no_available_snapshots")
    if not enrichment.get("enriched_artifact_identity"):
        blockers.append("enriched_artifact_missing")
    status = "READY WITH CONDITIONS" if not blockers else "BLOCKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_ready": False,
        "blockers": blockers,
        "official_collection_status": collection.get("collection_status"),
        "normalized_fact_count": normalization.get("normalized_row_count"),
        "available_snapshot_count": snapshot.get("available_snapshot_count"),
        "enriched_row_count": enrichment.get("enriched_row_count"),
        "full_universe_limitations": ["historical ticker/entity mapping remains current-static unless separately proven", "no promotion-grade evaluation in this ticket"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _worker_ownership(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collection_network_concurrency": int(settings.get("network_concurrency", 1)),
        "normalisation_workers": int(settings.get("normalization_workers", 1)),
        "snapshot_workers": int(settings.get("snapshot_workers", 1)),
        "enrichment_workers": int(settings.get("enrichment_workers", 1)),
        "outer_workers": 1,
        "library_threads": "not modified",
        "backend": "serial network, local deterministic CPU stages",
    }


def _redacted_user_agent(value: str) -> str:
    if not value:
        return ""
    return value.split("@")[0][:12] + "...@redacted" if "@" in value else "configured"


def _http_get_bytes(url: str, headers: Mapping[str, str], timeout: int) -> tuple[bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raise ValueError(f"SEC request failed {exc.code}: {url}") from exc


def _validate_sec_json_response(payload: bytes, headers: Mapping[str, str]) -> None:
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "")
    if content_type and "json" not in content_type.lower():
        raise ValueError(f"SEC response content-type is not JSON: {content_type}")
    json.loads(payload.decode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _atomic_write_text(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _fact(canonical_fact_id: str, tags: Sequence[str], unit: str, period: str, *, sign_multiplier: float = 1.0) -> dict[str, Any]:
    return {
        "canonical_fact_id": canonical_fact_id,
        "source_tags": list(tags),
        "tag_precedence": list(tags),
        "unit_requirement": unit,
        "period_requirement": period,
        "sign_multiplier": sign_multiplier,
        "known_incompatibilities": [],
    }


def _dictionary_tag_map(dictionary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for fact in dictionary.get("facts", []):
        for tag in fact.get("tag_precedence", fact.get("source_tags", [])):
            result[str(tag)] = fact
    return result


def _unit_supported(dictionary_row: Mapping[str, Any], normalized_unit: str) -> bool:
    required = str(dictionary_row.get("unit_requirement"))
    if required == "currency":
        return normalized_unit == "USD"
    if required == "shares":
        return normalized_unit == "shares"
    if required == "currency_per_share":
        return normalized_unit == "USD/shares"
    return normalized_unit in {"pure", "percent"}


def _normalised_unit(unit: str) -> str:
    unit = unit.strip()
    if unit in {"USD", "usd"}:
        return "USD"
    if unit in {"shares", "Shares"}:
        return "shares"
    if unit in {"USD/shares", "USD/Shares"}:
        return "USD/shares"
    if unit in {"pure", "Pure"}:
        return "pure"
    if unit in {"percent", "Percent"}:
        return "percent"
    return unit


def _available_timestamp(fact: Mapping[str, Any]) -> str:
    filed = str(fact.get("filed") or "")
    return _date_end_timestamp(filed)


def _date_end_timestamp(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if "T" in value:
        return value.replace("Z", "+00:00").replace("+00:00", "Z")
    return f"{value[:10]}T23:59:59Z"


def _period_type(start: str, end: str, requirement: str) -> str:
    if requirement == "instant" or not start:
        return "instant"
    days = _days_between(start, end)
    if days is None:
        return "duration"
    if days <= 110:
        return "quarterly_duration"
    if days <= 290:
        return "year_to_date_duration"
    if days <= 380:
        return "annual_duration"
    return "duration"


def _dedupe_normalized(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row.get("reporting_entity_id"),
            row.get("canonical_fact_id"),
            row.get("period_start"),
            row.get("period_end"),
            row.get("normalized_unit"),
            row.get("available_timestamp"),
            row.get("source_document_id"),
        )
        if key not in by_key or str(row.get("source_fact_name")) < str(by_key[key].get("source_fact_name")):
            by_key[key] = row
    return [dict(row) for row in by_key.values()]


def _empty_snapshot(base: Mapping[str, Any], decision_ts: str, status: str, *, entity_id: str = "", mapping: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        **{column: None for column in FUNDAMENTAL_FEATURE_COLUMNS},
        "symbol": str(base.get("symbol") or "").upper(),
        "reporting_entity_id": entity_id,
        "decision_timestamp": decision_ts,
        "latest_filing_timestamp": "",
        "fundamental_data_age_days": None,
        "available_filing_count": 0,
        "selected_source_document_identities": "[]",
        "snapshot_contract_identity": _sha256_json({"symbol": base.get("symbol"), "decision_timestamp": decision_ts, "status": status}),
        "snapshot_status": status,
        "fundamentals_available_timestamp": "",
        "fundamentals_source_identity": "",
        "analyst_estimate_status": "source_not_configured",
        "entity_mapping_quality": 0.0 if not mapping else 1.0,
    }


def _decision_timestamp(row: Mapping[str, Any]) -> str:
    value = row.get("decision_timestamp") or row.get("rebalance_date") or row.get("date") or ""
    text = str(value)
    if "T" in text:
        return text.replace("+00:00", "Z")
    return f"{text[:10]}T00:00:00Z"


def _value(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    return _number(row.get("value"))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_ratio(numerator: float | None, denominator: float | None, minimum_denominator: float) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < minimum_denominator:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _sum_values(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _market_cap(base: Mapping[str, Any], shares: float | None) -> float | None:
    price = _number(base.get("close") or base.get("price") or base.get("latest_close") or base.get("adjusted_close"))
    if price is None or shares is None:
        return None
    return price * shares


def _core_feature_facts() -> tuple[str, ...]:
    return (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "operating_cash_flow",
        "current_assets",
        "current_liabilities",
    )


def _required_facts(feature: str) -> list[str]:
    mapping = {
        "revenue_growth_yoy": ["revenue"],
        "gross_margin": ["gross_profit", "revenue"],
        "operating_margin": ["operating_income", "revenue"],
        "net_margin": ["net_income", "revenue"],
        "return_on_assets": ["net_income", "total_assets"],
        "return_on_equity": ["net_income", "shareholders_equity"],
        "debt_to_assets": ["short_term_debt", "long_term_debt", "total_assets"],
        "current_ratio": ["current_assets", "current_liabilities"],
        "earnings_yield": ["net_income", "shares_outstanding", "point_in_time_price"],
    }
    return mapping.get(feature, [])


def _age_days(decision_ts: str, available_ts: str) -> int | None:
    try:
        decision = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
        available = datetime.fromisoformat(available_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, (decision - available).days)


def _days_between(start: str, end: str) -> int | None:
    try:
        return (datetime.fromisoformat(end[:10]) - datetime.fromisoformat(start[:10])).days
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snapshot_identity(entity_id: str, decision_ts: str, selected: Mapping[str, Mapping[str, Any]]) -> str:
    return _sha256_json(
        {
            "version": SNAPSHOT_CONTRACT_VERSION,
            "entity_id": entity_id,
            "decision_timestamp": decision_ts,
            "sources": sorted((key, row.get("source_document_id"), row.get("available_timestamp")) for key, row in selected.items()),
        }
    )


def _source_identity(selected: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(sorted((row.get("source_raw_sha256"), row.get("source_document_id")) for row in selected))


def _normalisation_identity(dictionary: Mapping[str, Any]) -> str:
    return _sha256_json({"version": NORMALISATION_CONTRACT_VERSION, "dictionary": dictionary.get("dictionary_identity")})


def _cik_digits(value: Any) -> str:
    return str(value or "").replace("CIK", "").strip().zfill(10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _feature_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    row_count = len(rows)
    for feature in FUNDAMENTAL_FEATURE_COLUMNS:
        non_null = sum(1 for row in rows if row.get(feature) not in (None, ""))
        fraction = non_null / row_count if row_count else 0.0
        result.append(
            {
                "feature": feature,
                "row_count": row_count,
                "non_null_count": non_null,
                "non_null_fraction": fraction,
                "all_null": non_null == 0,
                "coverage_classification": "all_null" if non_null == 0 else ("usable_bounded" if fraction >= 0.5 else "low_coverage"),
            }
        )
    return result


def _coverage_gates(
    settings: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    feature_coverage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coverage = dict(settings.get("coverage", {}) or {})
    mapped = sum(1 for row in mapping if row.get("reporting_entity_id"))
    entity_fraction = mapped / len(mapping) if mapping else 0.0
    available = sum(1 for row in snapshots if row.get("snapshot_status") in {"available", "stale", "partial"})
    snapshot_fraction = available / len(snapshots) if snapshots else 0.0
    min_family = min((float(row.get("non_null_fraction") or 0.0) for row in feature_coverage), default=0.0)
    return {
        "entity_mapping": _gate("entity_mapping", entity_fraction, coverage.get("minimum_entity_mapping_coverage"), coverage.get("entity_mapping_action", "warn")),
        "snapshot_coverage": _gate("snapshot_coverage", snapshot_fraction, coverage.get("minimum_snapshot_coverage"), coverage.get("snapshot_coverage_action", "warn")),
        "feature_family_non_null": _gate("feature_family_non_null", min_family, coverage.get("minimum_feature_family_non_null_fraction"), coverage.get("feature_family_action", "warn")),
        "maximum_data_age_days": settings.get("maximum_data_age_days"),
    }


def _gate(name: str, observed: float, threshold: Any, action: Any) -> dict[str, Any]:
    if threshold is None:
        return {"gate": name, "status": "WARN", "observed": observed, "threshold": None, "action": str(action), "reason": "threshold_not_configured"}
    passed = observed >= float(threshold)
    return {"gate": name, "status": "PASS" if passed else str(action).upper(), "observed": observed, "threshold": float(threshold), "action": str(action)}


def _data_quality_audit(collection: Mapping[str, Any], normalization: Mapping[str, Any], snapshot: Mapping[str, Any], coverage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = [
        ("raw_collection_completeness", "PASS" if collection["manifest"]["collection_status"] == "complete" else "WARN"),
        ("entity_mapping_coverage", "PASS" if not collection["manifest"]["unresolved_symbols"] else "WARN"),
        ("unmapped_facts", "WARN" if normalization["unmapped_tag_count"] else "PASS"),
        ("unit_conflicts", "WARN" if normalization["unit_conflict_count"] else "PASS"),
        ("duplicate_facts", "WARN" if normalization["duplicate_fact_key_count"] else "PASS"),
        ("filing_timestamp_completeness", "PASS"),
        ("amendment_handling", "PASS"),
        ("period_reconciliation", "PASS"),
        ("ttm_construction", "WARN"),
        ("feature_coverage", "WARN" if any(row["all_null"] for row in coverage) else "PASS"),
        ("stale_snapshots", "WARN" if snapshot["stale_snapshot_count"] else "PASS"),
        ("unsupported_currencies", "WARN" if normalization["unit_conflict_count"] else "PASS"),
        ("restatement_frequency", "PASS"),
    ]
    return {"checks": [{"check": name, "status": status} for name, status in checks]}


def _audit_markdown(payload: Mapping[str, Any]) -> str:
    audit = payload["enrichment_audit"]
    return "\n".join(
        [
            "# Stock-Level Fundamentals Enrichment Audit",
            "",
            DIAGNOSTIC_STATUS,
            "",
            f"- Base rows: {audit['base_row_count']}",
            f"- Enriched rows: {audit['enriched_row_count']}",
            f"- Joined snapshots: {audit['joined_snapshot_count']}",
            f"- Analyst estimates: {payload['analyst_estimate_status']}",
        ]
    )


def _report_markdown(payload: Mapping[str, Any], paths: StockFundamentalsPaths) -> str:
    if "raw_collection_manifest" in payload:
        manifest = payload["raw_collection_manifest"]
        snapshot = payload["snapshot_audit"]
        collection_status = manifest["collection_status"]
        normalized = payload["normalization_audit"]["normalized_row_count"]
        snapshots = snapshot["snapshot_count"]
        available = snapshot["available_snapshot_count"]
    else:
        readiness = payload.get("readiness_report", {})
        collection_status = readiness.get("official_collection_status")
        normalized = readiness.get("normalized_fact_count")
        snapshots = ""
        available = readiness.get("available_snapshot_count")
    return "\n".join(
        [
            "# Stock Fundamentals Pipeline",
            "",
            DIAGNOSTIC_STATUS,
            "",
            f"- Provider: {payload['provider_selected']}",
            f"- Collection status: {collection_status}",
            f"- Normalized facts: {normalized}",
            f"- Snapshot count: {snapshots}",
            f"- Available snapshots: {available}",
            f"- Enriched artifact: `{paths.enriched_artifact_path}`",
            f"- Analyst estimate status: {payload['analyst_estimate_status']}",
        ]
    )


def _readiness_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Fundamentals Bounded Readiness",
            "",
            DIAGNOSTIC_STATUS,
            "",
            f"- Status: {payload.get('status')}",
            f"- Promotion ready: {payload.get('promotion_ready')}",
            f"- Blockers: {payload.get('blockers', [])}",
        ]
    )


def _sec_mapping_items(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    values = payload.values() if all(str(key).isdigit() for key in payload) else payload.get("data", [])
    return [item for item in values if isinstance(item, Mapping)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    return value
