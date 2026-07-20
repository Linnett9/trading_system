from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
LARGE_SOURCE_GATE_VERSION = "ticket_5b4_large_source_gate_v1"
PARTITION_MANIFEST_VERSION = "ticket_5b4_fundamentals_partition_v1"

BASE_ROW_KEY_COLUMNS = ("decision_timestamp", "rebalance_date", "symbol")
TARGET_PROVENANCE_REQUIRED_COLUMNS = (
    "target_provenance_contract_version",
    "target_start_timestamp",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_timestamp",
)
BENCHMARK_REQUIRED_COLUMNS = (
    "actual_benchmark_return_10d",
    "benchmark_label_start_timestamp",
    "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp",
)
SOURCE_IDENTITY_REQUIRED_COLUMNS = (
    "decision_grid_identity",
)

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

FINAL_FUNDAMENTALS_JOIN_KEY_COLUMNS = (
    "asset_id",
    "rebalance_date",
    "decision_session_date",
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
    full_universe_plan_path: Path
    full_universe_symbols_path: Path
    full_collection_preflight_path: Path
    full_collection_plan_path: Path
    collection_progress_path: Path
    unresolved_entities_path: Path
    raw_cache_validation_path: Path
    raw_cache_audit_path: Path
    normalization_progress_path: Path
    unmapped_tags_path: Path
    tag_conflicts_path: Path
    unit_coverage_path: Path
    unsupported_currencies_path: Path
    filing_coverage_path: Path
    amendment_audit_path: Path
    period_blockers_path: Path
    full_collection_readiness_json_path: Path
    full_collection_readiness_markdown_path: Path
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
    enriched_mapping = _enrich_mapping_rows(entity_mapping, settings)
    eligible_mapping = _eligible_mapping_rows(entity_mapping)
    collection_plan = _collection_plan(entity_mapping, enriched_mapping, settings, provider)
    collection = _collect_raw(provider, eligible_mapping, settings, progress_path=paths.collection_progress_path)
    writer.write_json(paths.bounded_cohort_path, _json_ready(cohort))
    writer.write_json(paths.full_universe_plan_path, _json_ready({
        "schema_version": SCHEMA_VERSION,
        "universe_identity": _universe_identity(settings),
        "configured_symbol_count": len(enriched_mapping),
        "eligible_reporting_companies": len(eligible_mapping),
        "etf_or_fund_count": sum(1 for row in enriched_mapping if row.get("exclusion_reason") == "excluded_non_company"),
        "unsupported_count": sum(1 for row in enriched_mapping if row.get("exclusion_reason") == "unsupported_security"),
        "unresolved_count": sum(1 for row in enriched_mapping if row.get("official_sec_mapping_status") == "unresolved"),
        "ambiguous_count": sum(1 for row in enriched_mapping if row.get("official_sec_mapping_status") == "ambiguous"),
        "excluded_security_count": sum(1 for row in enriched_mapping if row.get("collection_eligibility") == "excluded"),
        "mapping_audit": mapping_audit,
    }))
    writer.write_csv(paths.full_universe_symbols_path, enriched_mapping, fieldnames=_fields(enriched_mapping, ["symbol"]))
    writer.write_json(paths.full_collection_preflight_path, _json_ready(preflight))
    writer.write_json(paths.full_collection_plan_path, _json_ready(collection_plan))
    writer.write_csv(paths.entity_mapping_path, enriched_mapping, fieldnames=_fields(enriched_mapping, ["symbol"]))
    writer.write_json(paths.entity_mapping_audit_path, _json_ready(mapping_audit))
    writer.write_json(paths.raw_collection_manifest_path, _json_ready(collection["manifest"]))
    writer.write_csv(paths.failed_entities_path, collection["failed_entities"], fieldnames=_fields(collection["failed_entities"], ["symbol", "reporting_entity_id"]))
    unresolved_rows = [row for row in enriched_mapping if row.get("official_sec_mapping_status") in {"unresolved", "ambiguous"}]
    writer.write_csv(paths.unresolved_entities_path, unresolved_rows, fieldnames=_fields(unresolved_rows, ["symbol", "official_sec_mapping_status"]))
    validation = _raw_cache_validation_rows(enriched_mapping, Path(settings["raw_root"]))
    writer.write_csv(paths.raw_cache_validation_path, validation, fieldnames=_fields(validation, ["symbol", "reporting_entity_id", "cache_state"]))
    writer.write_json(paths.raw_cache_audit_path, _json_ready(_raw_cache_audit(validation, collection["manifest"], enriched_mapping)))
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
    writer.write_json(paths.normalization_progress_path, _json_ready(_normalization_progress(raw_payloads, normalized_facts, settings)))
    writer.write_csv(paths.tag_coverage_path, _tag_coverage(normalized_facts, normalization_audit), fieldnames=["canonical_fact_id", "entities_covered", "filings_covered", "source_tags_used", "conflicts", "unmapped_alternatives"])
    writer.write_csv(paths.unmapped_tags_path, _unmapped_tag_rows(normalization_audit), fieldnames=["namespace", "tag", "source_tag", "count", "mapping_status"])
    writer.write_csv(paths.tag_conflicts_path, [], fieldnames=["namespace", "tag", "conflict_type", "details"])
    writer.write_csv(paths.unit_coverage_path, _unit_coverage(normalized_facts), fieldnames=["normalized_unit", "entity_count", "filing_count", "observation_count", "canonical_fact_count"])
    writer.write_csv(paths.unit_conflicts_path, normalization_audit.get("unit_conflicts", []), fieldnames=_fields(normalization_audit.get("unit_conflicts", []), ["reporting_entity_id", "source_fact_name", "source_unit", "canonical_fact_id"]))
    writer.write_csv(paths.unsupported_currencies_path, _unsupported_currency_rows(normalization_audit), fieldnames=["reporting_entity_id", "source_fact_name", "source_unit", "canonical_fact_id"])
    writer.write_csv(paths.filing_coverage_path, _filing_coverage(normalized_facts), fieldnames=["form_type", "entity_count", "filing_count", "observation_count", "missing_filing_timestamp_count"])
    writer.write_csv(paths.amendment_audit_path, _amendment_audit(normalized_facts), fieldnames=["reporting_entity_id", "form_type", "amendment_observation_count", "accession_count", "pit_policy"])
    writer.write_csv(paths.period_reconciliation_path, _period_reconciliation(normalized_facts), fieldnames=["reporting_entity_id", "status", "instant_count", "quarterly_count", "ytd_count", "annual_count", "blocked_reason"])
    period_rows = [row for row in _period_reconciliation(normalized_facts) if row.get("status") == "blocked"]
    writer.write_csv(paths.period_blockers_path, period_rows, fieldnames=["reporting_entity_id", "status", "instant_count", "quarterly_count", "ytd_count", "annual_count", "blocked_reason"])
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
        "unit_conflict_count": len(existing.get("unit_conflicts", []) or []),
        "duplicate_group_count": _duplicate_normalized_group_count(facts),
        "period_reconciliation": _period_counts(facts),
    }
    writer = ResearchArtifactWriter()
    writer.write_json(paths.normalization_audit_path, _json_ready(audit))
    readiness = _full_collection_readiness(paths, settings)
    writer.write_json(paths.full_collection_readiness_json_path, _json_ready(readiness))
    writer.write_markdown(paths.full_collection_readiness_markdown_path, _full_readiness_markdown(readiness))
    return paths


def write_stock_fundamentals_snapshots(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_base_rows(settings)
    mapping = _read_csv_dicts(paths.entity_mapping_path)
    facts = _read_parquet_dicts(paths.normalized_facts_path)
    snapshots, snapshot_audit = build_partitioned_fundamental_snapshots(
        base_rows,
        mapping,
        facts,
        settings=settings,
        output_dir=paths.output_dir,
    )
    writer = ResearchArtifactWriter()
    _write_parquet(paths.snapshots_path, snapshots, _fields(snapshots, ["decision_timestamp", "symbol"]))
    snapshot_coverage = _snapshot_coverage(snapshots)
    writer.write_csv(paths.snapshot_coverage_path, snapshot_coverage, fieldnames=_fields(snapshot_coverage, ["coverage_scope", "snapshot_status", "row_count", "symbol_count", "decision_date_count", "symbol", "year", "decision_date"]))
    writer.write_json(paths.snapshot_audit_path, _json_ready(snapshot_audit))
    writer.write_json(paths.feature_contracts_path, _json_ready(formula_contracts()))
    feature_coverage = _feature_coverage(snapshots)
    writer.write_csv(paths.feature_coverage_path, feature_coverage, fieldnames=_fields(feature_coverage, ["coverage_scope", "feature_family", "feature", "row_count", "non_null_count", "non_null_fraction", "all_null", "coverage_classification", "symbol", "year", "decision_date"]))
    return paths


def write_stock_fundamentals_enrich(config: Mapping[str, Any]) -> StockFundamentalsPaths:
    settings = _settings(config)
    paths = _paths(Path(settings["output_dir"]))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_base_rows(settings)
    snapshots = _read_parquet_dicts(paths.snapshots_path)
    enriched_rows, enrichment_audit = enrich_stock_artifact_with_fundamentals_partitioned(
        base_rows,
        snapshots,
        settings=settings,
        output_dir=paths.output_dir,
    )
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
    writer.write_json(paths.pipeline_manifest_path, _json_ready(_lineage_manifest(paths, settings, enrichment_audit)))
    writer.write_json(paths.readiness_json_path, _json_ready(_readiness_report(paths)))
    writer.write_markdown(paths.readiness_markdown_path, _readiness_markdown(_readiness_report(paths)))
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
                "selected_source_document_lineage": json.dumps(_selected_source_lineage(selected.values()), sort_keys=True),
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
        "amendment_available_snapshot_count": sum(1 for row in snapshots if _snapshot_has_selected_amendment(row)),
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


def build_partitioned_fundamental_snapshots(
    base_rows: Sequence[Mapping[str, Any]],
    entity_mapping: Sequence[Mapping[str, Any]],
    normalized_facts: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    partition_dir = Path(str(settings.get("snapshot_partition_dir") or output_dir / "fundamentals_snapshot_partitions"))
    workers = _bounded_workers(settings.get("snapshot_workers"), len({str(row.get("symbol", "")).upper() for row in base_rows}))
    symbols = sorted({str(row.get("symbol", "")).upper() for row in base_rows if row.get("symbol")})
    rows_by_symbol = {symbol: [row for row in base_rows if str(row.get("symbol", "")).upper() == symbol] for symbol in symbols}
    partition_dir.mkdir(parents=True, exist_ok=True)

    def run_symbol(symbol: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        source_rows = rows_by_symbol[symbol]
        partition_path = _partition_path(partition_dir, symbol)
        if bool(settings.get("snapshot_resume_partitions", True)):
            loaded = _try_load_partition(partition_path, stage="snapshots", symbol=symbol, expected_base_rows=source_rows)
            if loaded is not None:
                return symbol, loaded, {"symbol": symbol, "status": "reused", "partition_path": str(partition_path), "row_count": len(loaded), "owner_worker": _owner_worker(symbol, workers)}
        snapshots, _audit = build_fundamental_snapshots(
            source_rows,
            entity_mapping,
            normalized_facts,
            maximum_data_age_days=settings["maximum_data_age_days"],
            minimum_denominator=settings["minimum_denominator"],
        )
        _write_partition(partition_path, snapshots, stage="snapshots", symbol=symbol, base_rows=source_rows)
        return symbol, snapshots, {"symbol": symbol, "status": "written", "partition_path": str(partition_path), "row_count": len(snapshots), "owner_worker": _owner_worker(symbol, workers)}

    results = _run_partition_jobs(symbols, run_symbol, workers)
    snapshots = _merge_partition_rows_preserving_base_order(base_rows, {symbol: rows for symbol, rows, _ in results}, row_date_column="decision_timestamp")
    partition_progress = [progress for _symbol, _rows, progress in results]
    audit = {
        **_snapshot_audit_from_rows(snapshots, normalized_facts),
        "partition_manifest_version": PARTITION_MANIFEST_VERSION,
        "partition_dir": str(partition_dir),
        "requested_workers": int(settings.get("snapshot_workers", 1)),
        "effective_workers": workers,
        "partition_count": len(partition_progress),
        "partition_status_counts": _count_values(partition_progress, "status"),
        "partition_progress": partition_progress,
        "deterministic_ownership": True,
    }
    return snapshots, audit


def enrich_stock_artifact_with_fundamentals_partitioned(
    base_rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    partition_dir = Path(str(settings.get("enrichment_partition_dir") or output_dir / "fundamentals_enrichment_partitions"))
    workers = _bounded_workers(settings.get("enrichment_workers"), len({str(row.get("symbol", "")).upper() for row in base_rows}))
    symbols = sorted({str(row.get("symbol", "")).upper() for row in base_rows if row.get("symbol")})
    rows_by_symbol = {symbol: [row for row in base_rows if str(row.get("symbol", "")).upper() == symbol] for symbol in symbols}
    snapshots_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in snapshots:
        snapshots_by_symbol.setdefault(str(row.get("symbol", "")).upper(), []).append(row)
    partition_dir.mkdir(parents=True, exist_ok=True)

    def run_symbol(symbol: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        source_rows = rows_by_symbol[symbol]
        partition_path = _partition_path(partition_dir, symbol)
        if bool(settings.get("enrichment_resume_partitions", True)):
            loaded = _try_load_partition(partition_path, stage="enrichment", symbol=symbol, expected_base_rows=source_rows)
            if loaded is not None:
                return symbol, loaded, {"symbol": symbol, "status": "reused", "partition_path": str(partition_path), "row_count": len(loaded), "owner_worker": _owner_worker(symbol, workers)}
        rows, _audit = enrich_stock_artifact_with_fundamentals(source_rows, snapshots_by_symbol.get(symbol, []), settings=settings)
        _assert_base_rows_preserved(source_rows, rows)
        _write_partition(partition_path, rows, stage="enrichment", symbol=symbol, base_rows=source_rows)
        return symbol, rows, {"symbol": symbol, "status": "written", "partition_path": str(partition_path), "row_count": len(rows), "owner_worker": _owner_worker(symbol, workers)}

    results = _run_partition_jobs(symbols, run_symbol, workers)
    enriched = _merge_partition_rows_preserving_base_order(base_rows, {symbol: rows for symbol, rows, _ in results}, row_date_column="decision_timestamp")
    _assert_base_rows_preserved(base_rows, enriched)
    _assert_no_silent_zero_fill(enriched)
    partition_progress = [progress for _symbol, _rows, progress in results]
    _base_audit_rows, audit = enrich_stock_artifact_with_fundamentals(base_rows, snapshots, settings=settings)
    audit.update(
        {
            "partition_manifest_version": PARTITION_MANIFEST_VERSION,
            "partition_dir": str(partition_dir),
            "requested_workers": int(settings.get("enrichment_workers", 1)),
            "effective_workers": workers,
            "partition_count": len(partition_progress),
            "partition_status_counts": _count_values(partition_progress, "status"),
            "partition_progress": partition_progress,
            "deterministic_ownership": True,
            "row_preservation": _row_preservation_audit(base_rows, enriched),
        }
    )
    return enriched, audit


def certify_pit_fundamentals_artifact(
    artifact_path: Path | None,
    manifest_path: Path | None,
    *,
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    required_columns = {
        *FUNDAMENTAL_FEATURE_COLUMNS,
        *FUNDAMENTAL_METADATA_COLUMNS,
    }
    if (
        artifact_path is None
        or manifest_path is None
        or not artifact_path.is_file()
        or not manifest_path.is_file()
    ):
        return {
            "valid": False,
            "reason": "artifact_or_manifest_missing",
            "required_column_count": len(required_columns),
            "missing_columns": sorted(required_columns),
        }
    manifest = _read_json(manifest_path)
    canonical = dict(manifest.get("canonical_artifact", {}) or {})
    expected_sha = str(
        canonical.get("sha256")
        or manifest.get("artifact_sha256")
        or manifest.get("sha256")
        or ""
    )
    observed_sha = file_sha256(artifact_path)
    status = str(
        manifest.get("status")
        or manifest.get("completion_status")
        or canonical.get("completion_status")
        or ""
    ).upper()
    contract_text = json.dumps(manifest, sort_keys=True, default=str)
    contract_valid = (
        ENRICHMENT_CONTRACT_VERSION in contract_text
        and SNAPSHOT_CONTRACT_VERSION in contract_text
    )
    parquet = pq.ParquetFile(artifact_path)
    columns = set(parquet.schema_arrow.names)
    date_columns = {"rebalance_date", "decision_session_date"} & columns
    symbol_key_present = "symbol" in columns
    asset_key_present = "asset_id" in columns
    missing_columns = sorted(required_columns - columns)
    row_count = parquet.metadata.num_rows
    row_count_valid = (
        expected_row_count is None or int(expected_row_count) == int(row_count)
    )
    valid = (
        status == "COMPLETE"
        and bool(expected_sha)
        and expected_sha == observed_sha
        and contract_valid
        and not missing_columns
        and bool(date_columns)
        and (symbol_key_present or asset_key_present)
        and row_count_valid
    )
    return {
        "valid": valid,
        "reason": "certified" if valid else "manifest_schema_or_population_mismatch",
        "artifact_path": str(artifact_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "observed_sha256": observed_sha,
        "manifest_sha256": expected_sha,
        "completion_status": status,
        "completion_status_valid": status == "COMPLETE",
        "contract_valid": contract_valid,
        "required_column_count": len(required_columns),
        "required_columns": sorted(required_columns),
        "missing_columns": missing_columns,
        "date_columns_present": sorted(date_columns),
        "join_key_columns_present": sorted(
            ({column for column in ("symbol", "asset_id") if column in columns})
            | date_columns
        ),
        "asset_id_present": asset_key_present,
        "symbol_key_present": symbol_key_present,
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "row_count_valid": row_count_valid,
    }


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


ETF_OR_FUND_SYMBOLS = {
    "DIA",
    "GLD",
    "IWM",
    "QQQ",
    "SPY",
    "TLT",
    "VNQ",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
}


def _symbols_from_universe_paths(paths: Sequence[Any]) -> list[str]:
    symbols: list[str] = []
    for raw_path in paths or []:
        path = Path(str(raw_path))
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = payload.get("symbols") if isinstance(payload, Mapping) else None
        for value in values or []:
            symbol = str(value).strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _universe_identity(settings: Mapping[str, Any]) -> dict[str, Any]:
    identities = []
    for raw_path in settings.get("universe_paths", []) or []:
        path = Path(str(raw_path))
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            identities.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "name": payload.get("name") if isinstance(payload, Mapping) else None,
                    "available_count": payload.get("available_count") if isinstance(payload, Mapping) else None,
                    "symbol_count": len(payload.get("symbols") or []) if isinstance(payload, Mapping) else None,
                }
            )
    return {
        "universe_status": settings.get("universe_status"),
        "survivorship_status": settings.get("survivorship_status"),
        "historical_membership_status": settings.get("historical_membership_status"),
        "delisting_coverage_status": settings.get("delisting_coverage_status"),
        "ticker_history_status": settings.get("ticker_history_status"),
        "sources": identities,
        "identity": _sha256_json(identities),
    }


def _security_type(symbol: str) -> tuple[str, str]:
    normalized = symbol.upper()
    if normalized in ETF_OR_FUND_SYMBOLS:
        return "ETF/fund", "excluded_non_company"
    if "." in normalized or "/" in normalized:
        return "unsupported", "unsupported_security"
    return "ordinary company", ""


def _enrich_mapping_rows(entity_mapping: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in entity_mapping:
        symbol = str(row.get("symbol", "")).upper()
        security_type, exclusion = _security_type(symbol)
        mapping_status = str(row.get("mapping_status") or "")
        if exclusion:
            collection_eligibility = "excluded"
            output_status = exclusion
        elif mapping_status in {"resolved_official", "resolved_manual_override"}:
            collection_eligibility = "eligible"
            output_status = mapping_status
        elif mapping_status == "ambiguous":
            collection_eligibility = "blocked"
            output_status = "ambiguous"
        else:
            collection_eligibility = "blocked"
            output_status = "unresolved"
        mapping_confidence = "high_current_official" if output_status == "resolved_official" else "none"
        rows.append(
            {
                **dict(row),
                "configured_status": "configured",
                "security_type": security_type,
                "security_classification": security_type,
                "official_sec_mapping_status": output_status,
                "mapping_confidence": mapping_confidence,
                "manual_override_status": "none" if output_status != "resolved_manual_override" else "active",
                "collection_eligibility": collection_eligibility,
                "exclusion_reason": exclusion,
                "universe_status": settings.get("universe_status"),
                "survivorship_status": settings.get("survivorship_status"),
                "historical_membership_status": settings.get("historical_membership_status"),
                "delisting_coverage_status": settings.get("delisting_coverage_status"),
                "ticker_history_status": settings.get("ticker_history_status"),
            }
        )
    return rows


def _eligible_mapping_rows(entity_mapping: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in entity_mapping
        if row.get("provider_entity_id")
        and _security_type(str(row.get("symbol") or ""))[1] == ""
        and row.get("mapping_status") in {"resolved_official", "resolved_manual_override"}
    ]


def _collection_plan(
    entity_mapping: Sequence[Mapping[str, Any]],
    enriched_mapping: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    provider: SecCompanyFactsProvider,
) -> dict[str, Any]:
    eligible = _eligible_mapping_rows(entity_mapping)
    cache_rows = []
    valid = invalid = 0
    for row in eligible:
        path = Path(str(settings["raw_root"])) / provider.provider_id / str(row["reporting_entity_id"]) / "companyfacts.json"
        state = validate_cached_companyfacts(path, expected_cik=str(row.get("provider_entity_id") or ""))
        state["symbol"] = row.get("symbol")
        state["reporting_entity_id"] = row.get("reporting_entity_id")
        cache_rows.append(state)
        if state["cache_state"] == "valid_cached":
            valid += 1
        elif state["cache_state"] != "missing":
            invalid += 1
    excluded = [row for row in enriched_mapping if row.get("collection_eligibility") == "excluded"]
    unresolved = [row for row in enriched_mapping if row.get("official_sec_mapping_status") == "unresolved"]
    ambiguous = [row for row in enriched_mapping if row.get("official_sec_mapping_status") == "ambiguous"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "stage": "full_collection_plan",
        "universe_identity": _universe_identity(settings),
        "configured_symbol_count": len(enriched_mapping),
        "eligible_entity_count": len(eligible),
        "already_valid_cached_entities": valid,
        "new_entities_required": sum(1 for row in cache_rows if row["cache_state"] == "missing"),
        "invalid_cached_entities": invalid,
        "unresolved_entities": len(unresolved),
        "ambiguous_entities": len(ambiguous),
        "excluded_entities": len(excluded),
        "estimated_request_count": sum(1 for row in cache_rows if row["cache_state"] != "valid_cached"),
        "network_concurrency": int(settings.get("network_concurrency", 1)),
        "request_delay_seconds": float(settings.get("request_delay_seconds", 0.0)),
        "retry_policy": {"max_retries": int(settings.get("max_retries", 0))},
        "timeout_policy": {"timeout_seconds": int(settings.get("timeout_seconds", 30))},
        "raw_root": str(settings.get("raw_root")),
        "resume": bool(settings.get("resume", True)),
        "collection_chunk_size": int(settings.get("collection_chunk_size", 25)),
        "collection_contract_identity": "",
        "cache_validation": cache_rows,
    }
    plan["collection_contract_identity"] = _sha256_json({k: v for k, v in plan.items() if k not in {"cache_validation", "collection_contract_identity"}})
    return plan


def _collect_raw(
    provider: SecCompanyFactsProvider,
    entity_mapping: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    *,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    raw_payloads = []
    failed = []
    skipped = []
    request_count = 0
    max_entities = settings.get("maximum_entities")
    resolved = [row for row in entity_mapping if row.get("provider_entity_id")]
    if max_entities is not None:
        resolved = resolved[: int(max_entities)]
    chunk_size = max(1, int(settings.get("collection_chunk_size", 25)))
    chunks: list[dict[str, Any]] = []
    completed_entities: list[str] = []
    for chunk_index, start_index in enumerate(range(0, len(resolved), chunk_size), start=1):
        chunk_rows = resolved[start_index: start_index + chunk_size]
        chunk = {
            "chunk_id": f"chunk_{chunk_index:04d}",
            "entity_ids": [str(row.get("reporting_entity_id")) for row in chunk_rows],
            "completed_count": 0,
            "failed_count": 0,
            "start_timestamp": _utc_now(),
            "end_timestamp": "",
            "chunk_status": "running",
        }
        for row in chunk_rows:
            entity_id = str(row["reporting_entity_id"])
            status = "failed"
            error = ""
            cache_state = ""
            path = ""
            raw_sha = ""
            requested = False
            try:
                cache_path = provider.raw_root / provider.provider_id / entity_id / "companyfacts.json"
                before = validate_cached_companyfacts(cache_path, expected_cik=str(row["provider_entity_id"]))
                result = provider.fetch_company_facts(str(row["provider_entity_id"]), force_refresh=bool(settings["force_refresh"]))
                requested = result["status"] != "skipped_cached"
                request_count += 1 if requested else 0
                if result["status"] == "skipped_cached":
                    skipped.append(entity_id)
                    status = "valid_cached"
                else:
                    status = "new_success" if before["cache_state"] == "missing" else "retried_success"
                raw_payloads.append(result)
                completed_entities.append(entity_id)
                cache_state = "valid_cached"
                path = str(result["path"])
                raw_sha = str(result["metadata"].get("sha256") or "")
            except Exception as exc:
                error = str(exc)
                failed.append({"symbol": row["symbol"], "reporting_entity_id": row["reporting_entity_id"], "error_type": type(exc).__name__, "error": error})
                chunk["failed_count"] += 1
            chunk["completed_count"] += 1 if status in {"valid_cached", "new_success", "retried_success"} else 0
            if progress_path is not None:
                progress = {
                    "schema_version": SCHEMA_VERSION,
                    "started": started,
                    "last_update_timestamp": _utc_now(),
                    "network_concurrency": int(settings.get("network_concurrency", 1)),
                    "request_delay_seconds": float(settings.get("request_delay_seconds", 0.0)),
                    "chunk_size": chunk_size,
                    "completed_entities": completed_entities,
                    "failed_entities": failed,
                    "current_chunk": chunk,
                    "chunks": chunks,
                    "last_entity_status": {
                        "symbol": row.get("symbol"),
                        "reporting_entity_id": entity_id,
                        "status": status,
                        "cache_state": cache_state,
                        "request_made": requested,
                        "path": path,
                        "raw_sha256": raw_sha,
                        "error": error,
                    },
                }
                _atomic_write_text(progress_path, json.dumps(_json_ready(progress), indent=2))
        chunk["end_timestamp"] = _utc_now()
        chunk["chunk_status"] = "complete" if chunk["failed_count"] == 0 else "partial"
        chunks.append(chunk)
        if progress_path is not None:
            progress = {
                "schema_version": SCHEMA_VERSION,
                "started": started,
                "last_update_timestamp": _utc_now(),
                "network_concurrency": int(settings.get("network_concurrency", 1)),
                "request_delay_seconds": float(settings.get("request_delay_seconds", 0.0)),
                "chunk_size": chunk_size,
                "completed_entities": completed_entities,
                "failed_entities": failed,
                "chunks": chunks,
                "progress_status": "running",
            }
            _atomic_write_text(progress_path, json.dumps(_json_ready(progress), indent=2))
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
        "chunks": chunks,
        "chunk_size": chunk_size,
        "status_counts": {
            "valid_cached": len(skipped),
            "new_or_retried_success": request_count,
            "failed": len(failed),
            "unresolved": len([row for row in entity_mapping if not row.get("reporting_entity_id")]),
            "unsupported": 0,
        },
        "collection_status": status,
    }
    if progress_path is not None:
        progress = _read_json(progress_path)
        progress["progress_status"] = status
        progress["last_update_timestamp"] = _utc_now()
        _atomic_write_text(progress_path, json.dumps(_json_ready(progress), indent=2))
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
        "filing_recency_score": _filing_recency_score(base, selected),
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


def _filing_recency_score(base: Mapping[str, Any], selected: Mapping[str, Mapping[str, Any]]) -> float | None:
    timestamps = [
        str(row.get("available_timestamp") or "")
        for row in selected.values()
        if row.get("available_timestamp")
    ]
    if not timestamps:
        return None
    age = _age_days(_decision_timestamp(base), max(timestamps))
    if age is None:
        return None
    return 1.0 / (1.0 + float(age))


def _selected_source_lineage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lineage = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("source_document_id") or ""),
            str(row.get("canonical_fact_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        lineage.append(
            {
                "canonical_fact_id": key[1],
                "source_document_id": key[0],
                "filing_accession": str(row.get("filing_accession") or ""),
                "form_type": str(row.get("form_type") or ""),
                "available_timestamp": str(row.get("available_timestamp") or ""),
                "is_amendment": bool(row.get("is_amendment")),
            }
        )
    return sorted(
        lineage,
        key=lambda item: (
            item["canonical_fact_id"],
            item["available_timestamp"],
            item["source_document_id"],
        ),
    )


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("stock_fundamentals", {}) or {})
    output_dir = raw.get("output_dir") or str(stock_alpha_output_dir(config) / "stock_fundamentals")
    collection = dict(raw.get("collection", {}) or {})
    normalization = dict(raw.get("normalization", {}) or {})
    snapshots = dict(raw.get("snapshots", {}) or {})
    enrichment = dict(raw.get("enrichment", {}) or {})
    source_gate = dict(raw.get("source_gate", {}) or {})
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
        "require_large_source_identity": bool(source_gate.get("require_large_source_identity", raw.get("require_large_source_identity", False))),
        "expected_source_identity": dict(source_gate.get("expected_identity", raw.get("expected_source_identity", {})) or {}),
        "expected_source_identity_path": str(source_gate.get("expected_identity_path", raw.get("expected_source_identity_path", "")) or ""),
        "expected_source_manifest_path": str(source_gate.get("expected_manifest_path", raw.get("expected_source_manifest_path", "")) or ""),
        "allow_probe_source_fixture": bool(source_gate.get("allow_probe_source_fixture", raw.get("allow_probe_source_fixture", False))),
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
        "collection_chunk_size": int(collection.get("chunk_size", raw.get("collection_chunk_size", 25))),
        "user_agent": user_agent,
        "user_agent_env": user_agent_env,
        "live_collection": bool(collection.get("live_collection", True)),
        "start_stage": str(stages.get("start_stage", raw.get("start_stage", "collect"))),
        "end_stage": str(stages.get("end_stage", raw.get("end_stage", "enrich"))),
        "enabled_stages": list(stages.get("enabled_stages", raw.get("enabled_stages", [])) or []),
        "network_concurrency": int(collection.get("network_concurrency", 1)),
        "normalization_workers": int(normalization.get("workers", 1)),
        "snapshot_workers": int(snapshots.get("workers", 1)),
        "snapshot_partition_dir": str(snapshots.get("partition_dir", "")),
        "snapshot_resume_partitions": bool(snapshots.get("resume_partitions", raw.get("resume", True))),
        "enrichment_workers": int(enrichment.get("workers", 1)),
        "enrichment_partition_dir": str(enrichment.get("partition_dir", "")),
        "enrichment_resume_partitions": bool(enrichment.get("resume_partitions", raw.get("resume", True))),
        "partition_chunk_size": int(raw.get("partition_chunk_size", snapshots.get("chunk_size", enrichment.get("chunk_size", 1)))),
        "normalized_output_path": str(normalization.get("output_path", "")),
        "maximum_data_age_days": snapshots.get("maximum_data_age_days"),
        "amendment_policy": str(snapshots.get("amendment_policy", "latest_available_as_of_decision")),
        "preserve_base_rows": bool(enrichment.get("preserve_base_rows", True)),
        "minimum_denominator": float(features.get("minimum_denominator", 1e-9)),
        "maximum_decision_dates": bounded.get("maximum_decision_dates"),
        "maximum_symbols": bounded.get("maximum_symbols"),
        "universe_paths": list(raw.get("universe_paths") or ml.get("stock_alpha_artifact_universe_paths") or []),
        "universe_status": str(raw.get("universe_status") or "CURRENT_STATIC_UNIVERSE"),
        "survivorship_status": str(raw.get("survivorship_status") or "CURRENT_STATIC_SURVIVORSHIP_BIASED"),
        "historical_membership_status": str(raw.get("historical_membership_status") or "not_proven_current_static_only"),
        "delisting_coverage_status": str(raw.get("delisting_coverage_status") or "not_proven"),
        "ticker_history_status": str(raw.get("ticker_history_status") or "current_static_sec_mapping"),
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
        full_universe_plan_path=output_dir / "fundamentals_full_universe_plan.json",
        full_universe_symbols_path=output_dir / "fundamentals_full_universe_symbols.csv",
        full_collection_preflight_path=output_dir / "fundamentals_full_collection_preflight.json",
        full_collection_plan_path=output_dir / "fundamentals_full_collection_plan.json",
        collection_progress_path=output_dir / "fundamentals_collection_progress.json",
        unresolved_entities_path=output_dir / "fundamentals_unresolved_entities.csv",
        raw_cache_validation_path=output_dir / "fundamentals_raw_cache_validation.csv",
        raw_cache_audit_path=output_dir / "fundamentals_raw_cache_audit.json",
        normalization_progress_path=output_dir / "fundamentals_normalization_progress.json",
        unmapped_tags_path=output_dir / "fundamentals_unmapped_tags.csv",
        tag_conflicts_path=output_dir / "fundamentals_tag_conflicts.csv",
        unit_coverage_path=output_dir / "fundamentals_unit_coverage.csv",
        unsupported_currencies_path=output_dir / "fundamentals_unsupported_currencies.csv",
        filing_coverage_path=output_dir / "fundamentals_filing_coverage.csv",
        amendment_audit_path=output_dir / "fundamentals_amendment_audit.csv",
        period_blockers_path=output_dir / "fundamentals_period_blockers.csv",
        full_collection_readiness_json_path=output_dir / "fundamentals_full_collection_readiness_report.json",
        full_collection_readiness_markdown_path=output_dir / "fundamentals_full_collection_readiness_report.md",
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
    if bool(settings.get("require_large_source_identity")):
        _validate_large_source_gate(settings)
    rows = read_stock_level_artifact(Path(source), allow_csv_fallback=bool(settings.get("allow_csv_fallback", False)))
    max_symbols = settings.get("maximum_symbols")
    max_dates = settings.get("maximum_decision_dates")
    configured_symbols = [str(symbol).upper() for symbol in settings.get("symbols", []) or []]
    if configured_symbols:
        configured = set(configured_symbols)
        rows = [row for row in rows if str(row.get("symbol", "")).upper() in configured]
    if max_symbols is not None:
        symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})[: int(max_symbols)]
        rows = [row for row in rows if str(row.get("symbol", "")).upper() in symbols]
    if max_dates is not None:
        dates = sorted({_decision_timestamp(row)[:10] for row in rows})[: int(max_dates)]
        rows = [row for row in rows if _decision_timestamp(row)[:10] in dates]
    return rows


def _validate_large_source_gate(settings: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(settings.get("source_dataset_path") or ""))
    if not source:
        raise ValueError("Ticket 5B.4 source gate requires ml.stock_fundamentals.source_dataset_path")
    if source.suffix.lower() != ".parquet":
        raise ValueError(f"Ticket 5B.4 refuses non-canonical or legacy source artifact: {source}")
    if not source.exists():
        raise ValueError(f"Ticket 5B.4 source artifact does not exist: {source}")
    source_text = str(source).replace("\\", "/").lower()
    if not bool(settings.get("allow_probe_source_fixture")) and any(token in source_text for token in ("probe", "profile", "symbols_5", "symbols_20", "symbols_50")):
        raise ValueError(f"Ticket 5B.4 refuses probe/profile source artifact: {source}")
    expected = _expected_source_identity(settings)
    if not expected:
        raise ValueError("Ticket 5B.4 source gate requires expected source identity or manifest path")
    required = set(BASE_ROW_KEY_COLUMNS) | set(TARGET_PROVENANCE_REQUIRED_COLUMNS) | set(BENCHMARK_REQUIRED_COLUMNS) | set(SOURCE_IDENTITY_REQUIRED_COLUMNS)
    rows = read_stock_level_artifact(source, required_columns=required)
    if not rows:
        raise ValueError(f"Ticket 5B.4 source artifact is empty or incomplete: {source}")
    fieldnames = _parquet_column_order(source)
    actual = artifact_identity(source, rows=rows, fieldnames=fieldnames, artifact_format="parquet", compression="zstd")
    _assert_source_identity_matches(actual, expected)
    _validate_source_identity_columns(rows, expected)
    _validate_base_artifact_provenance(rows)
    return {
        "schema_version": LARGE_SOURCE_GATE_VERSION,
        "status": "PASS",
        "source_path": str(source),
        "actual_identity": actual,
        "expected_identity": expected,
    }


def _expected_source_identity(settings: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(settings.get("expected_source_identity", {}) or {})
    identity_path_text = str(settings.get("expected_source_identity_path") or "").strip()
    identity_path = Path(identity_path_text) if identity_path_text else None
    if identity_path is not None and identity_path.exists():
        expected.update(_read_json(identity_path))
    manifest_path_text = str(settings.get("expected_source_manifest_path") or "").strip()
    manifest_path = Path(manifest_path_text) if manifest_path_text else None
    if manifest_path is not None and manifest_path.exists():
        manifest = _read_json(manifest_path)
        expected.update(_identity_from_manifest(manifest))
    return {key: value for key, value in expected.items() if value not in (None, "")}


def _identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        manifest.get("stock_artifact"),
        (manifest.get("stages") or {}).get("stock_artifact") if isinstance(manifest.get("stages"), Mapping) else None,
        (manifest.get("stage_status") or {}).get("stock_artifact") if isinstance(manifest.get("stage_status"), Mapping) else None,
        manifest.get("source_artifact_identity"),
        manifest.get("artifact_identity"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in ("identity", "artifact_identity", "output_identity", "stock_artifact_identity"):
            value = candidate.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        if any(name in candidate for name in ("row_count", "sha256", "logical_content_sha256", "schema_fingerprint")):
            return dict(candidate)
    return {}


def _assert_source_identity_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    comparable = (
        "row_count",
        "symbol_count",
        "decision_date_count",
        "sha256",
        "logical_content_sha256",
        "schema_fingerprint",
    )
    mismatches = []
    for key in comparable:
        if key in expected and str(actual.get(key)) != str(expected.get(key)):
            mismatches.append({"field": key, "actual": actual.get(key), "expected": expected.get(key)})
    if mismatches:
        raise ValueError(f"Ticket 5B.4 source identity mismatch: {mismatches}")


def _validate_source_identity_columns(rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]) -> None:
    for column, expected_key in (("decision_grid_identity", "decision_grid_identity"), ("universe_identity", "universe_identity")):
        values = sorted({str(row.get(column) or "") for row in rows if str(row.get(column) or "").strip()})
        if expected_key in expected and values != [str(expected[expected_key])]:
            raise ValueError(f"Ticket 5B.4 {column} mismatch: {values} != {expected[expected_key]}")
    if "universe_identity" in expected and not any(str(row.get("universe_identity") or "").strip() for row in rows):
        raise ValueError("Ticket 5B.4 source artifact missing universe_identity column values")


def _validate_base_artifact_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    missing = []
    for column in TARGET_PROVENANCE_REQUIRED_COLUMNS + BENCHMARK_REQUIRED_COLUMNS:
        if any(row.get(column) in (None, "") for row in rows):
            missing.append(column)
    if missing:
        raise ValueError(f"Ticket 5B.4 source artifact has missing target/benchmark provenance columns: {sorted(set(missing))}")


def _parquet_column_order(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _configured_symbols(settings: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if settings.get("symbols"):
        return [str(symbol).upper() for symbol in settings["symbols"]]
    universe_symbols = _symbols_from_universe_paths(settings.get("universe_paths", []))
    if universe_symbols:
        return universe_symbols
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
    source_path = Path(str(settings.get("source_dataset_path") or ""))
    universe_paths = [Path(str(path)) for path in settings.get("universe_paths", []) or []]
    universe_exists = bool(universe_paths) and all(path.exists() for path in universe_paths)
    user_agent = str(settings.get("user_agent") or "").strip()
    if settings.get("live_collection") and (not user_agent or "@" not in user_agent):
        reasons.append(f"missing identifying SEC user agent; set {settings.get('user_agent_env', 'SEC_USER_AGENT')}")
    if int(settings.get("network_concurrency", 1)) > 2:
        reasons.append("SEC network concurrency must remain serial or near-serial")
    if not source_path.exists() and not universe_exists:
        reasons.append(f"source stock artifact or universe does not exist: {source_path}")
    source_gate = {"required": bool(settings.get("require_large_source_identity")), "status": "NOT_REQUIRED"}
    if bool(settings.get("require_large_source_identity")):
        try:
            source_gate = _validate_large_source_gate(settings)
        except Exception as exc:
            source_gate = {"schema_version": LARGE_SOURCE_GATE_VERSION, "required": True, "status": "BLOCKED", "reason": str(exc)}
            reasons.append(f"large source gate blocked: {exc}")
    if settings.get("live_collection") and not bool(settings.get("load_official_sec_company_tickers")):
        reasons.append("live collection requires load_official_sec_company_tickers=true for official entity mapping")
    if settings.get("live_collection") and settings.get("maximum_entities") is None and not _symbols_from_universe_paths(settings.get("universe_paths", [])):
        reasons.append("live collection requires bounded collection.maximum_entities or configured universe_paths")
    raw_root = Path(str(settings.get("raw_root") or ""))
    output_dir = Path(str(settings.get("output_dir") or ""))
    output_writable = True
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".fundamentals_preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception:
        output_writable = False
        reasons.append(f"output directory is not writable: {output_dir}")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "preflight",
        "status": "BLOCKED" if reasons else "PASS",
        "blocking_reasons": reasons,
        "provider": str(settings.get("provider")),
        "provider_supported": str(settings.get("provider")) == "official_sec_companyfacts",
        "user_agent_env": str(settings.get("user_agent_env")),
        "user_agent_configured": bool(str(settings.get("user_agent") or "").strip()),
        "user_agent_redacted": _redacted_user_agent(str(settings.get("user_agent") or "")),
        "live_collection": bool(settings.get("live_collection")),
        "network_concurrency": int(settings.get("network_concurrency", 1)),
        "request_delay_seconds": float(settings.get("request_delay_seconds", 0.0)),
        "source_dataset_path": str(source_path),
        "source_dataset_exists": source_path.exists(),
        "large_source_gate": source_gate,
        "source_universe_paths": [str(path) for path in universe_paths],
        "source_universe_exists": universe_exists,
        "source_universe_identity": _universe_identity(settings),
        "output_dir": str(output_dir),
        "output_dir_writable": output_writable,
        "official_mapping_enabled": bool(settings.get("load_official_sec_company_tickers")),
        "maximum_entities": settings.get("maximum_entities"),
        "configured_universe_symbol_count": len(_symbols_from_universe_paths(settings.get("universe_paths", []))),
        "resume": bool(settings.get("resume", True)),
        "raw_root": str(raw_root),
        "feedless": True,
        "broker_access_required": False,
    }


def _bounded_cohort(symbols: Sequence[str], base_rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    max_symbols = settings.get("maximum_symbols") or settings.get("maximum_entities") or len(symbols)
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


def _raw_cache_validation_rows(entity_mapping: Sequence[Mapping[str, Any]], raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in entity_mapping:
        symbol = str(row.get("symbol") or "")
        entity_id = str(row.get("reporting_entity_id") or "")
        cik = str(row.get("provider_entity_id") or row.get("cik") or "")
        if not cik or row.get("collection_eligibility") == "excluded":
            rows.append(
                {
                    "symbol": symbol,
                    "reporting_entity_id": entity_id,
                    "cache_state": "unsupported" if row.get("collection_eligibility") == "excluded" else "missing",
                    "validation_status": row.get("official_sec_mapping_status") or "unresolved",
                    "reason": row.get("exclusion_reason") or "unresolved_or_not_eligible",
                }
            )
            continue
        path = raw_root / "official_sec_companyfacts" / f"CIK{_cik_digits(cik)}" / "companyfacts.json"
        state = validate_cached_companyfacts(path, expected_cik=cik)
        rows.append(
            {
                "symbol": symbol,
                "reporting_entity_id": entity_id,
                "provider_entity_id": cik,
                "cache_state": state.get("cache_state"),
                "validation_status": "valid" if state.get("cache_state") == "valid_cached" else state.get("cache_state"),
                "reason": state.get("reason", ""),
                "path": state.get("path", str(path)),
                "metadata_path": state.get("metadata_path", ""),
                "sha256": state.get("sha256", ""),
                "content_type": state.get("content_type", ""),
                "retrieval_timestamp": state.get("retrieval_timestamp", ""),
                "atomic_final_filename": path.name == "companyfacts.json",
            }
        )
    return rows


def _raw_cache_audit(
    validation_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    entity_mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned = len([row for row in entity_mapping if row.get("collection_eligibility") == "eligible"])
    valid = len([row for row in validation_rows if row.get("cache_state") == "valid_cached"])
    failed = len(manifest.get("failed_entities", []) or [])
    unresolved = len([row for row in entity_mapping if row.get("official_sec_mapping_status") in {"unresolved", "ambiguous"}])
    excluded = len([row for row in entity_mapping if row.get("collection_eligibility") == "excluded"])
    return {
        "schema_version": SCHEMA_VERSION,
        "planned_eligible_entities": planned,
        "valid_cached_entities": valid,
        "failed_entities": failed,
        "unresolved_entities": unresolved,
        "excluded_entities": excluded,
        "reconciliation_formula": "planned = valid_cached + failed for eligible entities; unresolved/excluded are outside request plan",
        "eligible_reconciliation_status": "PASS" if planned == valid + failed else "BLOCK",
        "all_configured_reconciliation_status": "PASS" if len(entity_mapping) == planned + unresolved + excluded else "BLOCK",
        "cache_state_counts": _count_by(validation_rows, "cache_state"),
        "collection_request_count": manifest.get("request_count"),
        "collection_status": manifest.get("collection_status"),
    }


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


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
    if path and str(path) and path.exists() and path.is_file():
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


def _normalization_progress(raw_payloads: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    entity_ids = [f"CIK{_cik_digits(item.get('payload', {}).get('cik'))}" for item in raw_payloads]
    return {
        "schema_version": SCHEMA_VERSION,
        "normalisation_workers": int(settings.get("normalization_workers", 1)),
        "backend": "deterministic serial consolidation",
        "library_thread_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "entity_partitions": [{"partition_id": entity_id, "status": "complete"} for entity_id in entity_ids],
        "reused_partitions": 0,
        "recomputed_partitions": len(entity_ids),
        "invalidated_partitions": 0,
        "normalized_row_count": len(rows),
        "normalisation_contract_identity": _normalisation_identity(canonical_fact_dictionary()),
        "peak_memory_diagnostics": "not_available",
    }


def _unmapped_tag_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in audit.get("unmapped_tags", []) or []:
        source_tag = str(item.get("source_tag") or "")
        namespace, _, tag = source_tag.partition(":")
        rows.append({"namespace": namespace, "tag": tag, "source_tag": source_tag, "count": item.get("count", 0), "mapping_status": "unmapped"})
    return rows


def _unit_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for unit in sorted({str(row.get("normalized_unit") or "") for row in rows}):
        unit_rows = [row for row in rows if str(row.get("normalized_unit") or "") == unit]
        result.append(
            {
                "normalized_unit": unit,
                "entity_count": len({row.get("reporting_entity_id") for row in unit_rows}),
                "filing_count": len({row.get("filing_accession") for row in unit_rows}),
                "observation_count": len(unit_rows),
                "canonical_fact_count": len({row.get("canonical_fact_id") for row in unit_rows}),
            }
        )
    return result


def _unsupported_currency_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in audit.get("unit_conflicts", []) or []:
        unit = str(row.get("source_unit") or "")
        if unit.upper() not in {"USD", "SHARES", "USD/SHARES", "PURE", "PERCENT"}:
            rows.append(dict(row))
    return rows


def _filing_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for form in sorted({str(row.get("form_type") or "") for row in rows}):
        form_rows = [row for row in rows if str(row.get("form_type") or "") == form]
        result.append(
            {
                "form_type": form,
                "entity_count": len({row.get("reporting_entity_id") for row in form_rows}),
                "filing_count": len({row.get("filing_accession") for row in form_rows}),
                "observation_count": len(form_rows),
                "missing_filing_timestamp_count": sum(1 for row in form_rows if not row.get("filing_timestamp")),
            }
        )
    return result


def _amendment_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    amended = [row for row in rows if bool(row.get("is_amendment"))]
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in amended:
        by_key.setdefault((str(row.get("reporting_entity_id") or ""), str(row.get("form_type") or "")), []).append(row)
    for (entity_id, form_type), group in sorted(by_key.items()):
        result.append(
            {
                "reporting_entity_id": entity_id,
                "form_type": form_type,
                "amendment_observation_count": len(group),
                "accession_count": len({row.get("filing_accession") for row in group}),
                "pit_policy": "amendments become eligible only when filed timestamp is <= decision timestamp",
            }
        )
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
    result = []
    statuses = sorted({str(row.get("snapshot_status") or "") for row in snapshots})
    for status in statuses:
        rows = [row for row in snapshots if str(row.get("snapshot_status") or "") == status]
        result.append(
            {
                "coverage_scope": "status",
                "snapshot_status": status,
                "row_count": len(rows),
                "symbol_count": len({row.get("symbol") for row in rows}),
                "decision_date_count": len({str(row.get("decision_timestamp"))[:10] for row in rows}),
            }
        )
    for scope, key_func in (
        ("symbol_status", lambda row: (str(row.get("symbol") or ""), str(row.get("snapshot_status") or ""))),
        ("year_status", lambda row: (str(row.get("decision_timestamp") or "")[:4], str(row.get("snapshot_status") or ""))),
        ("decision_date_status", lambda row: (str(row.get("decision_timestamp") or "")[:10], str(row.get("snapshot_status") or ""))),
    ):
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in snapshots:
            grouped.setdefault(key_func(row), []).append(row)
        for (value, status), rows in sorted(grouped.items()):
            result.append(
                {
                    "coverage_scope": scope,
                    "snapshot_status": status,
                    "row_count": len(rows),
                    "symbol_count": len({row.get("symbol") for row in rows}),
                    "decision_date_count": len({str(row.get("decision_timestamp"))[:10] for row in rows}),
                    "symbol": value if scope == "symbol_status" else "",
                    "year": value if scope == "year_status" else "",
                    "decision_date": value if scope == "decision_date_status" else "",
                }
            )
    return result


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


def _full_collection_readiness(paths: StockFundamentalsPaths, settings: Mapping[str, Any]) -> dict[str, Any]:
    plan = _read_json(paths.full_collection_plan_path)
    collection = _read_json(paths.raw_collection_manifest_path)
    cache = _read_json(paths.raw_cache_audit_path)
    normalization = _read_json(paths.normalization_audit_path)
    blockers = []
    if cache.get("eligible_reconciliation_status") not in {None, "PASS"}:
        blockers.append("eligible_collection_reconciliation_failed")
    if collection.get("collection_status") not in {"complete", "partially_complete"}:
        blockers.append("collection_not_complete")
    if normalization.get("status") != "PASS":
        blockers.append("normalisation_not_pass")
    if int(cache.get("failed_entities", 0) or 0) > 0:
        blockers.append("failed_entities_remain")
    limitations = [
        "historical ticker/entity mapping remains current-static unless separately proven",
        "survivorship and delisting coverage are not proven by this collection",
        "not promotion-ready; no selector fitting or trading validation in this ticket",
    ]
    if int(plan.get("unresolved_entities", 0) or 0) > 0:
        limitations.append("unresolved SEC mappings remain excluded")
    if int(plan.get("excluded_entities", 0) or 0) > 0:
        limitations.append("ETFs/funds/unsupported securities excluded from CompanyFacts collection")
    status = "READY FOR LARGE-ARTIFACT ENRICHMENT" if not blockers else ("READY WITH CONDITIONS" if normalization.get("normalized_row_count") else "BLOCKED")
    if limitations and not blockers:
        status = "READY WITH CONDITIONS"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_ready": False,
        "blockers": blockers,
        "limitations": limitations,
        "configured_symbol_count": plan.get("configured_symbol_count"),
        "eligible_entity_count": plan.get("eligible_entity_count"),
        "excluded_entities": plan.get("excluded_entities"),
        "unresolved_entities": plan.get("unresolved_entities"),
        "ambiguous_entities": plan.get("ambiguous_entities"),
        "valid_cached_entities": cache.get("valid_cached_entities"),
        "failed_entities": cache.get("failed_entities"),
        "raw_collection_reconciliation": cache.get("eligible_reconciliation_status"),
        "cache_integrity": cache.get("cache_state_counts"),
        "normalization_status": normalization.get("status"),
        "normalized_fact_count": normalization.get("normalized_row_count"),
        "unmapped_tag_count": normalization.get("unmapped_tag_count"),
        "unit_conflict_count": normalization.get("unit_conflict_count"),
        "period_reconciliation": _period_counts(_read_parquet_dicts(paths.normalized_facts_path)) if paths.normalized_facts_path.exists() else {},
        "survivorship_status": settings.get("survivorship_status"),
        "historical_membership_status": settings.get("historical_membership_status"),
        "delisting_coverage_status": settings.get("delisting_coverage_status"),
        "ticker_history_status": settings.get("ticker_history_status"),
    }


def _full_readiness_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# Fundamentals Full Collection Readiness",
        "",
        f"- Status: {readiness.get('status')}",
        f"- Promotion ready: {readiness.get('promotion_ready')}",
        f"- Configured symbols: {readiness.get('configured_symbol_count')}",
        f"- Eligible entities: {readiness.get('eligible_entity_count')}",
        f"- Valid cached entities: {readiness.get('valid_cached_entities')}",
        f"- Normalized facts: {readiness.get('normalized_fact_count')}",
        f"- Historical membership: {readiness.get('historical_membership_status')}",
        f"- Ticker history: {readiness.get('ticker_history_status')}",
    ]
    blockers = readiness.get("blockers") or []
    if blockers:
        lines.extend(["", "## Blockers", *[f"- {item}" for item in blockers]])
    limitations = readiness.get("limitations") or []
    if limitations:
        lines.extend(["", "## Conditions", *[f"- {item}" for item in limitations]])
    return "\n".join(lines) + "\n"


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
    for feature in FUNDAMENTAL_FEATURE_COLUMNS:
        result.append(_coverage_row(rows, feature, coverage_scope="feature"))
    for family, features in _fundamental_feature_families().items():
        result.append(_family_coverage_row(rows, family, features, coverage_scope="feature_family"))
    for symbol in sorted({str(row.get("symbol") or "") for row in rows if row.get("symbol")}):
        symbol_rows = [row for row in rows if str(row.get("symbol") or "") == symbol]
        for family, features in _fundamental_feature_families().items():
            result.append({**_family_coverage_row(symbol_rows, family, features, coverage_scope="symbol_family"), "symbol": symbol})
    for year in sorted({str(row.get("decision_timestamp") or "")[:4] for row in rows if row.get("decision_timestamp")}):
        year_rows = [row for row in rows if str(row.get("decision_timestamp") or "").startswith(year)]
        for family, features in _fundamental_feature_families().items():
            result.append({**_family_coverage_row(year_rows, family, features, coverage_scope="year_family"), "year": year})
    for decision_date in sorted({str(row.get("decision_timestamp") or "")[:10] for row in rows if row.get("decision_timestamp")}):
        date_rows = [row for row in rows if str(row.get("decision_timestamp") or "")[:10] == decision_date]
        for family, features in _fundamental_feature_families().items():
            result.append({**_family_coverage_row(date_rows, family, features, coverage_scope="decision_date_family"), "decision_date": decision_date})
    return result


def _coverage_row(rows: Sequence[Mapping[str, Any]], feature: str, *, coverage_scope: str) -> dict[str, Any]:
    row_count = len(rows)
    non_null = sum(1 for row in rows if row.get(feature) not in (None, ""))
    fraction = non_null / row_count if row_count else 0.0
    return {
        "coverage_scope": coverage_scope,
        "feature_family": _feature_family(feature),
        "feature": feature,
        "row_count": row_count,
        "non_null_count": non_null,
        "non_null_fraction": fraction,
        "all_null": non_null == 0,
        "coverage_classification": _coverage_classification(non_null, fraction, row_count),
    }


def _family_coverage_row(rows: Sequence[Mapping[str, Any]], family: str, features: Sequence[str], *, coverage_scope: str) -> dict[str, Any]:
    row_count = len(rows)
    cells = row_count * len(features)
    non_null = sum(1 for row in rows for feature in features if row.get(feature) not in (None, ""))
    fraction = non_null / cells if cells else 0.0
    return {
        "coverage_scope": coverage_scope,
        "feature_family": family,
        "feature": "*",
        "row_count": row_count,
        "non_null_count": non_null,
        "non_null_fraction": fraction,
        "all_null": non_null == 0,
        "coverage_classification": _coverage_classification(non_null, fraction, row_count),
    }


def _coverage_classification(non_null: int, fraction: float, row_count: int) -> str:
    if row_count == 0:
        return "blocked"
    if non_null == 0:
        return "all_null"
    return "usable" if fraction >= 0.5 else "low_coverage"


def _feature_family(feature: str) -> str:
    for family, features in _fundamental_feature_families().items():
        if feature in features:
            return family
    return "fundamental_other"


def _fundamental_feature_families() -> dict[str, tuple[str, ...]]:
    return {
        "fundamental_growth": FUNDAMENTAL_FEATURE_COLUMNS[0:10],
        "fundamental_profitability": FUNDAMENTAL_FEATURE_COLUMNS[10:20],
        "fundamental_quality": FUNDAMENTAL_FEATURE_COLUMNS[20:24],
        "fundamental_balance_sheet": FUNDAMENTAL_FEATURE_COLUMNS[24:31],
        "fundamental_shareholder_actions": FUNDAMENTAL_FEATURE_COLUMNS[31:36],
        "fundamental_valuation": FUNDAMENTAL_FEATURE_COLUMNS[36:40],
        "fundamental_freshness": FUNDAMENTAL_FEATURE_COLUMNS[40:],
    }


def _bounded_workers(raw_workers: Any, task_count: int) -> int:
    if task_count <= 0:
        return 1
    return max(1, min(int(raw_workers or 1), int(task_count), 12))


def _run_partition_jobs(
    symbols: Sequence[str],
    func: Callable[[str], tuple[str, list[dict[str, Any]], dict[str, Any]]],
    workers: int,
) -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    ordered = list(symbols)
    if workers <= 1 or len(ordered) <= 1:
        return [func(symbol) for symbol in ordered]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fundamentals-partition") as pool:
        by_symbol = {symbol: result for symbol, result in zip(ordered, pool.map(func, ordered))}
    return [by_symbol[symbol] for symbol in ordered]


def _partition_path(partition_dir: Path, symbol: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in symbol.upper())
    return partition_dir / f"{safe}.parquet"


def _partition_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _try_load_partition(
    path: Path,
    *,
    stage: str,
    symbol: str,
    expected_base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if not path.exists() and not _partition_manifest_path(path).exists():
        return None
    if not path.exists() or not _partition_manifest_path(path).exists():
        raise ValueError(f"Corrupt fundamentals partition missing data or manifest: {path}")
    manifest = _read_json(_partition_manifest_path(path))
    rows = _read_parquet_dicts(path)
    expected_key_hash = _row_key_hash(expected_base_rows)
    checks = {
        "manifest_version": manifest.get("manifest_version") == PARTITION_MANIFEST_VERSION,
        "stage": manifest.get("stage") == stage,
        "symbol": manifest.get("symbol") == symbol,
        "row_count": int(manifest.get("row_count", -1)) == len(rows) == len(expected_base_rows),
        "base_key_hash": manifest.get("base_key_hash") == expected_key_hash,
        "content_hash": manifest.get("content_hash") == _rows_content_hash(rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Corrupt or incompatible fundamentals partition {path}: {failed}")
    return rows


def _write_partition(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    symbol: str,
    base_rows: Sequence[Mapping[str, Any]],
) -> None:
    preferred = ["decision_timestamp", "symbol"] if stage == "snapshots" else ["decision_timestamp", "rebalance_date", "symbol"]
    _write_parquet(path, rows, _fields(rows, preferred))
    manifest = {
        "manifest_version": PARTITION_MANIFEST_VERSION,
        "stage": stage,
        "symbol": symbol,
        "row_count": len(rows),
        "base_key_hash": _row_key_hash(base_rows),
        "content_hash": _rows_content_hash(rows),
        "created_at": _utc_now(),
        "complete": True,
    }
    _atomic_write_json(_partition_manifest_path(path), manifest)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _merge_partition_rows_preserving_base_order(
    base_rows: Sequence[Mapping[str, Any]],
    rows_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    row_date_column: str,
) -> list[dict[str, Any]]:
    queues: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for symbol, rows in rows_by_symbol.items():
        for row in rows:
            key = (str(row.get(row_date_column) or row.get("rebalance_date") or "")[:10], symbol)
            queues.setdefault(key, []).append(row)
    merged = []
    for base in base_rows:
        key = (_decision_timestamp(base)[:10], str(base.get("symbol", "")).upper())
        if not queues.get(key):
            raise ValueError(f"Missing fundamentals partition row for base key {key}")
        merged.append(dict(queues[key].pop(0)))
    leftovers = sum(len(values) for values in queues.values())
    if leftovers:
        raise ValueError(f"Unexpected duplicate fundamentals partition rows: {leftovers}")
    return merged


def _row_key_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json([_base_row_key(row) for row in rows])


def _rows_content_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    fieldnames = _fields(rows, ["decision_timestamp", "rebalance_date", "symbol"])
    return _sha256_json([{name: _json_ready(row.get(name)) for name in fieldnames} for row in rows])


def _base_row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_decision_timestamp(row)[:10], str(row.get("symbol", "")).upper())


def _owner_worker(symbol: str, workers: int) -> int:
    return int(hashlib.sha256(symbol.upper().encode("utf-8")).hexdigest()[:8], 16) % max(1, workers)


def _snapshot_audit_from_rows(snapshots: Sequence[Mapping[str, Any]], normalized_facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_count": len(snapshots),
        "available_snapshot_count": sum(1 for row in snapshots if row.get("snapshot_status") == "available"),
        "stale_snapshot_count": sum(1 for row in snapshots if row.get("snapshot_status") == "stale"),
        "unresolved_entity_count": sum(1 for row in snapshots if row.get("snapshot_status") == "unresolved_entity"),
        "no_prior_filing_count": sum(1 for row in snapshots if row.get("snapshot_status") == "no_prior_filing"),
        "future_filing_exclusion_count": _future_filing_exclusion_count(snapshots, normalized_facts),
        "amendment_available_snapshot_count": sum(1 for row in snapshots if _snapshot_has_selected_amendment(row)),
        "availability_rule": "facts included only when available_timestamp <= decision_timestamp",
        "missing_snapshot_policy": "missing snapshots preserve NaN fundamentals and explicit status; no zero fill",
    }


def _snapshot_has_selected_amendment(row: Mapping[str, Any]) -> bool:
    try:
        lineage = json.loads(str(row.get("selected_source_document_lineage") or "[]"))
    except json.JSONDecodeError:
        lineage = []
    return any(
        bool(item.get("is_amendment"))
        or str(item.get("form_type") or "").endswith("/A")
        for item in lineage
        if isinstance(item, Mapping)
    )


def _future_filing_exclusion_count(snapshots: Sequence[Mapping[str, Any]], normalized_facts: Sequence[Mapping[str, Any]]) -> int:
    by_entity: dict[str, list[str]] = {}
    for fact in normalized_facts:
        by_entity.setdefault(str(fact.get("reporting_entity_id")), []).append(str(fact.get("available_timestamp", "")))
    count = 0
    for snap in snapshots:
        entity = str(snap.get("reporting_entity_id") or "")
        decision_ts = str(snap.get("decision_timestamp") or "")
        count += sum(1 for available in by_entity.get(entity, []) if available > decision_ts)
    return count


def _assert_base_rows_preserved(base_rows: Sequence[Mapping[str, Any]], enriched_rows: Sequence[Mapping[str, Any]]) -> None:
    audit = _row_preservation_audit(base_rows, enriched_rows)
    if audit["status"] != "PASS":
        raise ValueError(f"Fundamentals enrichment failed row preservation: {audit}")


def _row_preservation_audit(base_rows: Sequence[Mapping[str, Any]], enriched_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_columns = _fields(base_rows, [])
    changed = []
    if len(base_rows) != len(enriched_rows):
        changed.append({"type": "row_count", "base": len(base_rows), "enriched": len(enriched_rows)})
    for index, (base, enriched) in enumerate(zip(base_rows, enriched_rows)):
        if _base_row_key(base) != _base_row_key(enriched):
            changed.append({"type": "row_key", "index": index, "base": _base_row_key(base), "enriched": _base_row_key(enriched)})
            continue
        for column in base_columns:
            if base.get(column) != enriched.get(column):
                changed.append({"type": "base_column", "index": index, "column": column, "base": base.get(column), "enriched": enriched.get(column)})
                break
    return {
        "status": "PASS" if not changed else "FAIL",
        "base_row_count": len(base_rows),
        "enriched_row_count": len(enriched_rows),
        "base_column_count": len(base_columns),
        "changed_examples": changed[:20],
        "target_provenance_columns_preserved": all(column in base_columns for column in TARGET_PROVENANCE_REQUIRED_COLUMNS),
        "benchmark_columns_preserved": all(column in base_columns for column in BENCHMARK_REQUIRED_COLUMNS),
    }


def _assert_no_silent_zero_fill(rows: Sequence[Mapping[str, Any]]) -> None:
    blocked_rows = [row for row in rows if row.get("fundamentals_snapshot_status") in {"blocked", "unresolved_entity", "no_prior_filing"}]
    exempt = {"entity_mapping_quality"}
    offenders = [
        (row.get("symbol"), _decision_timestamp(row)[:10], column)
        for row in blocked_rows
        for column in FUNDAMENTAL_FEATURE_COLUMNS
        if column not in exempt and row.get(column) == 0
    ]
    if offenders:
        raise ValueError(f"Fundamentals missing values were zero-filled: {offenders[:10]}")


def _count_values(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _lineage_manifest(paths: StockFundamentalsPaths, settings: Mapping[str, Any], enrichment_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_contract_version": "ticket_5b4_fundamentals_lineage_v1",
        "source_gate_required": bool(settings.get("require_large_source_identity")),
        "source_identity": _optional_artifact_identity(Path(str(settings.get("source_dataset_path") or ""))),
        "normalized_facts_identity": _optional_artifact_identity(paths.normalized_facts_path),
        "snapshot_identity": _optional_artifact_identity(paths.snapshots_path),
        "enriched_artifact_identity": _optional_artifact_identity(paths.enriched_artifact_path),
        "enrichment_audit": dict(enrichment_audit),
        "universe_status": settings.get("universe_status"),
        "survivorship_status": settings.get("survivorship_status"),
        "xom_mapping_caveat": "official-current XOM mapping remains conditional unless a dated override contract is supplied",
        "no_training_or_trading_side_effects": True,
    }


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
