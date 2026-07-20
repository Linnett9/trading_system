"""Fail-closed historical asset-to-reporting-entity authority.

This module deliberately does not infer history from SEC's current ticker file.
It validates independently supplied, versioned interval evidence and provides a
mutation-free preflight for production publication.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTRACT_VERSION = "historical_reporting_entity_mapping_v1"
TEMPORAL_RULE_IDENTITY = "effective_start_lte_decision_lt_effective_end_and_knowledge_lte_decision_v1"
PRECEDENCE = (
    "versioned_manual_override",
    "canonical_asset_registry_historical_identity",
    "sec_issuer_temporal_evidence",
    "provider_symbol_history",
    "independently_proven_current_static_mapping",
)
NON_COMPANY_POLICY = {
    "TLT": "fixed-income ETF",
    "XLK": "sector ETF",
    "XLF": "sector ETF",
    "XLE": "sector ETF",
    "XLV": "sector ETF",
    "XLI": "sector ETF",
    "XLP": "sector ETF",
    "XLY": "sector ETF",
    "XLU": "sector ETF",
    "XLB": "sector ETF",
    "VNQ": "real-estate ETF",
}
STATUSES = {
    "resolved_company", "resolved_non_company_asset",
    "unresolved_entity", "ambiguous_entity", "inactive_asset",
    "not_applicable", "insufficient_historical_evidence",
}
REQUIRED_FIELDS = (
    "asset_id", "canonical_symbol", "security_type", "reporting_entity_id", "cik",
    "mapping_status", "mapping_quality", "effective_start_date",
    "effective_end_date", "knowledge_available_timestamp",
    "source_identity", "source_record_identity", "evidence_type",
    "contract_version",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(sorted(row.items())) for row in sorted(rows, key=interval_key)],
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def interval_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("asset_id", "")), str(row.get("effective_start_date", "")),
        str(row.get("effective_end_date", "")), str(row.get("reporting_entity_id", "")),
        str(row.get("source_record_identity", "")),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def authority_inventory(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    owners = [
        ("canonical asset IDs; canonical symbols; registry validity", "canonical asset registry",
         "asset_registry", "canonical_asset_registry_v1", "1900-01-01/open; current universe",
         "valid_from is synthetic for this population; CIK and security metadata empty", True),
        ("provider symbols; ticker aliases", "provider alias registry", "aliases",
         "provider_symbol_alias_registry_v1", "1900-01-01/open; current aliases",
         "identity/remap aliases, not certified corporate-action history", False),
        ("CIK mappings; issuer names", "stock fundamentals SEC resolver", "current_mapping",
         "fundamentals_sec_entity_mapping_contract_v1", "retrieval-time current mapping",
         "no asset_id or effective dates; explicitly current-static", False),
        ("SEC company facts; accession histories", "official SEC CompanyFacts cache", "companyfacts_root",
         "stock_fundamentals_schema_v1", "filing-dependent",
         "facts identify issuers but do not prove security/ticker intervals", False),
        ("SEC submissions metadata", "stock-alpha SEC filings collector", "submissions_root",
         "stock_alpha_news_event_v1", "cached collection-dependent",
         "accessions are evidence inputs, not a certified bridge", False),
        ("non-company classification", "stock-alpha ETF/fund registry", "fund_policy",
         "news_source_registry_etf_funds_v1", "current classification",
         "classification is usable; fund entity collection remains blocked", True),
        ("ticker changes; mergers; acquisitions; spin-offs; symbol reuse; delistings",
         "unresolved", "historical_evidence", "", "", "no certified repository owner found", False),
        ("manual historical overrides", "historical reporting entity contract", "overrides",
         CONTRACT_VERSION, "interval-specific", "optional; none accepted without evidence and review", True),
    ]
    result = []
    for responsibility, owner, key, version, coverage, limitation, usable in owners:
        path = paths.get(key)
        result.append({
            "responsibility": responsibility, "existing_owner": owner,
            "source_path": str(path) if path else "",
            "source_manifest": _sidecar(path), "contract_version": version,
            "temporal_coverage": coverage, "known_limitations": limitation,
            "usable_for_production": usable and bool(path and path.exists()),
        })
    return result


def validate_intervals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append({"row": index, "code": "SCHEMA_FIELDS_MISSING", "fields": missing})
        asset_id = str(row.get("asset_id", "")).strip()
        status = str(row.get("mapping_status", "")).strip()
        start, end = str(row.get("effective_start_date", "")).strip(), str(row.get("effective_end_date", "")).strip()
        knowledge = str(row.get("knowledge_available_timestamp", "")).strip()
        if not asset_id:
            errors.append({"row": index, "code": "SELECTOR_ASSET_ID_UNRESOLVED"})
        if status not in STATUSES:
            errors.append({"row": index, "code": "MAPPING_STATUS_INVALID"})
        if status == "resolved_company" and (not start or not knowledge):
            errors.append({"row": index, "code": "HISTORICAL_EFFECTIVE_INTERVAL_UNRESOLVED"})
        if status == "resolved_company" and not str(row.get("cik", "")).strip():
            errors.append({"row": index, "code": "SEC_ENTITY_EVIDENCE_UNRESOLVED"})
        if status in {"resolved_non_company_asset", "not_applicable"} and str(row.get("cik", "")).strip():
            errors.append({"row": index, "code": "NON_COMPANY_FAKE_CIK"})
        if start and end and start >= end:
            errors.append({"row": index, "code": "INTERVAL_ORDER_INVALID"})
        key = interval_key(row)
        if key in seen:
            duplicate_count += 1
        seen.add(key)
        if asset_id:
            by_asset[asset_id].append(row)
    overlaps = 0
    for asset_rows in by_asset.values():
        ordered = sorted(asset_rows, key=interval_key)
        for left, right in zip(ordered, ordered[1:]):
            left_end = str(left.get("effective_end_date", "")).strip()
            right_start = str(right.get("effective_start_date", "")).strip()
            if not left_end or not right_start or right_start < left_end:
                overlaps += 1
                errors.append({"asset_id": left.get("asset_id"), "code": "MAPPING_OVERLAP_DETECTED"})
    return {
        "status": "PASS" if not errors else "FAILED", "errors": errors,
        "missing_asset_id_count": sum(e["code"] == "SELECTOR_ASSET_ID_UNRESOLVED" for e in errors),
        "overlapping_interval_count": overlaps, "duplicate_interval_count": duplicate_count,
        "ambiguous_active_mapping_count": sum(str(r.get("mapping_status")) == "ambiguous_entity" for r in rows),
    }


def resolve(rows: Sequence[Mapping[str, Any]], asset_id: str, decision_timestamp: str) -> Mapping[str, Any] | None:
    decision_date = decision_timestamp[:10]
    active = [
        row for row in rows
        if str(row.get("asset_id")) == asset_id
        and str(row.get("effective_start_date", "")) <= decision_date
        and (not str(row.get("effective_end_date", "")) or decision_date < str(row.get("effective_end_date")))
        and str(row.get("knowledge_available_timestamp", "")) <= decision_timestamp
    ]
    if len(active) > 1:
        raise ValueError("MAPPING_AMBIGUITY_DETECTED")
    return active[0] if active else None


def preflight(paths: Mapping[str, Path], *, path_budget: int = 240) -> dict[str, Any]:
    before = _tree_identity(paths.get("output_root"))
    blockers: list[str] = []
    required = ("asset_registry", "aliases", "selector", "selector_manifest", "current_mapping", "fund_policy")
    for key in required:
        if not paths.get(key) or not paths[key].is_file():
            blockers.append("SOURCE_MANIFEST_INCOMPLETE")
    evidence = paths.get("historical_evidence")
    if not evidence or not evidence.is_file():
        blockers += ["SYMBOL_HISTORY_AUTHORITY_UNRESOLVED", "HISTORICAL_EFFECTIVE_INTERVAL_UNRESOLVED"]
    if paths.get("output_root") and len(str(paths["output_root"].resolve())) > path_budget:
        blockers.append("PATH_LENGTH_BUDGET_EXCEEDED")
    registry_rows = read_csv(paths["asset_registry"]) if paths.get("asset_registry") and paths["asset_registry"].is_file() else []
    selector_symbols = _selector_symbols(paths.get("selector"))
    registry_by_symbol = {r.get("canonical_symbol", "").upper(): r for r in registry_rows}
    missing_symbols = sorted(symbol for symbol in selector_symbols if not registry_by_symbol.get(symbol, {}).get("asset_id"))
    if missing_symbols:
        blockers.append("SELECTOR_ASSET_ID_UNRESOLVED")
    evidence_rows = read_csv(evidence) if evidence and evidence.is_file() else []
    validation = validate_intervals(evidence_rows) if evidence_rows else {}
    if validation and validation["status"] != "PASS":
        blockers.extend(sorted({e["code"] for e in validation["errors"]}))
    historical_company_rows = [r for r in evidence_rows if r.get("mapping_status") == "resolved_company"]
    if not historical_company_rows:
        blockers.append("SEC_ENTITY_EVIDENCE_UNRESOLVED")
    policy_missing = sorted(set(NON_COMPANY_POLICY) - selector_symbols)
    if policy_missing:
        blockers.append("NON_COMPANY_ASSET_POLICY_UNRESOLVED")
    blockers = sorted(set(blockers))
    after = _tree_identity(paths.get("output_root"))
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "READY" if not blockers else "BLOCKED",
        "completion_status": "COMPLETE_WITH_EXPLICIT_NON_COMPANY_ASSETS" if not blockers else "BLOCKED_UNRESOLVED_COMPANY_HISTORY",
        "publication_complete": False, "blockers": blockers,
        "mapping_precedence": list(PRECEDENCE), "mapping_precedence_identity": _hash_json(PRECEDENCE),
        "temporal_rule_identity": TEMPORAL_RULE_IDENTITY,
        "non_company_policy": NON_COMPANY_POLICY,
        "non_company_policy_identity": _hash_json(NON_COMPANY_POLICY),
        "selector_symbol_count": len(selector_symbols), "registry_asset_count": len(registry_rows),
        "selector_symbols_missing_asset_id": missing_symbols,
        "historical_evidence_row_count": len(evidence_rows),
        "validation": validation, "authority_inventory": authority_inventory(paths),
        "source_hashes": {key: sha256_file(path) for key, path in paths.items() if path.is_file()},
        "mutation_free_proof": {"output_root_before": before, "output_root_after": after, "unchanged": before == after},
    }


def selector_audit(selector_path: Path, intervals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import pyarrow.parquet as pq
    table = pq.read_table(selector_path, columns=["symbol", "rebalance_date", "decision_timestamp"])
    rows = table.to_pylist()
    by_symbol = {str(r.get("canonical_symbol", "")).upper(): str(r.get("asset_id", "")) for r in intervals}
    counts = Counter()
    earliest = latest = ""
    future = 0
    for row in rows:
        symbol = str(row["symbol"]).upper()
        timestamp = str(row.get("decision_timestamp") or row["rebalance_date"]).replace(" ", "T")
        asset_id = by_symbol.get(symbol, "")
        match = resolve(intervals, asset_id, timestamp) if asset_id else None
        status = str(match.get("mapping_status")) if match else "unresolved_entity"
        counts[status] += 1
        if not match:
            date = timestamp[:10]
            earliest = min(filter(None, (earliest, date)), default=date)
            latest = max(latest, date)
        elif str(match.get("knowledge_available_timestamp", "")) > timestamp:
            future += 1
    validation = validate_intervals(intervals)
    return {
        "total_selector_rows": len(rows),
        "resolved_company_rows": counts["resolved_company"],
        "resolved_non_company_rows": sum(counts[s] for s in ("resolved_non_company_asset", "not_applicable")),
        "unresolved_rows": counts["unresolved_entity"] + counts["insufficient_historical_evidence"],
        "ambiguous_rows": counts["ambiguous_entity"],
        "future_knowledge_violation_count": future,
        "overlapping_interval_count": validation["overlapping_interval_count"],
        "missing_asset_id_count": validation["missing_asset_id_count"],
        "symbol_count_by_status": dict(counts),
        "earliest_unresolved_date": earliest, "latest_unresolved_date": latest,
    }


def validate_override(row: Mapping[str, Any]) -> None:
    required = ("asset_id", "canonical_symbol", "effective_start_date",
                "knowledge_available_timestamp", "reason", "evidence_reference", "review_status")
    if any(not str(row.get(field, "")).strip() for field in required):
        raise ValueError("MANUAL_OVERRIDE_INCOMPLETE")
    if row.get("review_status") != "approved":
        raise ValueError("MANUAL_OVERRIDE_NOT_APPROVED")


def _selector_symbols(path: Path | None) -> set[str]:
    if not path or not path.is_file():
        return set()
    import pyarrow.parquet as pq
    return {str(value).upper() for value in pq.read_table(path, columns=["symbol"]).column("symbol").unique().to_pylist()}


def _sidecar(path: Path | None) -> str:
    if not path:
        return ""
    for candidate in (path.with_suffix(".manifest.json"), path.with_suffix(".json")):
        if candidate.is_file():
            return str(candidate)
    return ""


def _tree_identity(path: Path | None) -> str:
    if not path or not path.exists():
        return "ABSENT"
    return _hash_json(sorted((str(p.relative_to(path)), p.stat().st_size) for p in path.rglob("*") if p.is_file()))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
