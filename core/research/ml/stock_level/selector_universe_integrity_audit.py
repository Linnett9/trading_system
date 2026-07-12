from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    read_stock_level_artifact,
)


SCHEMA_VERSION = "selector_universe_integrity_audit_v1"
UNIVERSE_CONTRACT_VERSION = "selector_universe_membership_contract_v1"
SECURITY_IDENTITY_CONTRACT_VERSION = "selector_security_identity_contract_v1"
DIAGNOSTIC_STATUS = "BOUNDED DIAGNOSTIC ONLY / NOT PROMOTION EVIDENCE"


@dataclass(frozen=True)
class SelectorUniverseIntegrityAuditPaths:
    output_dir: Path
    universe_contract_path: Path
    historical_membership_audit_path: Path
    security_identity_mapping_path: Path
    ticker_change_audit_path: Path
    delisting_coverage_audit_path: Path
    corporate_action_audit_path: Path
    historical_liquidity_audit_path: Path
    classification_mapping_audit_path: Path
    breadth_universe_coverage_path: Path
    report_json_path: Path
    report_markdown_path: Path


def write_selector_universe_integrity_audit(config: Mapping[str, Any]) -> SelectorUniverseIntegrityAuditPaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.selector_universe_integrity_audit.enabled is false")
    source_path = Path(settings["source_artifact_path"])
    rows = read_stock_level_artifact(
        source_path,
        required_columns={"rebalance_date", "symbol"},
        allow_csv_fallback=bool(settings["allow_csv_fallback"]),
    )
    payload = build_selector_universe_integrity_audit(rows, settings=settings, source_path=source_path)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SelectorUniverseIntegrityAuditPaths(
        output_dir=output_dir,
        universe_contract_path=output_dir / "selector_universe_contract.json",
        historical_membership_audit_path=output_dir / "selector_historical_membership_audit.csv",
        security_identity_mapping_path=output_dir / "selector_security_identity_mapping.csv",
        ticker_change_audit_path=output_dir / "selector_ticker_change_audit.csv",
        delisting_coverage_audit_path=output_dir / "selector_delisting_coverage_audit.csv",
        corporate_action_audit_path=output_dir / "selector_corporate_action_audit.json",
        historical_liquidity_audit_path=output_dir / "selector_historical_liquidity_audit.csv",
        classification_mapping_audit_path=output_dir / "selector_classification_mapping_audit.csv",
        breadth_universe_coverage_path=output_dir / "selector_breadth_universe_coverage.csv",
        report_json_path=output_dir / "selector_universe_integrity_report.json",
        report_markdown_path=output_dir / "selector_universe_integrity_report.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.universe_contract_path, payload["universe_contract"])
    writer.write_csv(paths.historical_membership_audit_path, payload["membership_audit"], fieldnames=_fields(payload["membership_audit"], ["symbol"]))
    writer.write_csv(paths.security_identity_mapping_path, payload["security_identity_mapping"], fieldnames=_fields(payload["security_identity_mapping"], ["security_id", "historical_ticker"]))
    writer.write_csv(paths.ticker_change_audit_path, payload["ticker_change_audit"], fieldnames=_fields(payload["ticker_change_audit"], ["symbol"]))
    writer.write_csv(paths.delisting_coverage_audit_path, payload["delisting_coverage_audit"], fieldnames=_fields(payload["delisting_coverage_audit"], ["symbol"]))
    writer.write_json(paths.corporate_action_audit_path, payload["corporate_action_audit"])
    writer.write_csv(paths.historical_liquidity_audit_path, payload["historical_liquidity_audit"], fieldnames=_fields(payload["historical_liquidity_audit"], ["symbol"]))
    writer.write_csv(paths.classification_mapping_audit_path, payload["classification_mapping_audit"], fieldnames=_fields(payload["classification_mapping_audit"], ["symbol"]))
    writer.write_csv(paths.breadth_universe_coverage_path, payload["breadth_universe_coverage"], fieldnames=_fields(payload["breadth_universe_coverage"], ["rebalance_date"]))
    writer.write_json(paths.report_json_path, payload)
    writer.write_markdown(paths.report_markdown_path, _markdown(payload))
    return paths


def build_selector_universe_integrity_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    bounded = _bounded_rows(rows, settings)
    if not bounded:
        raise ValueError("No rows available for selector universe integrity audit")
    source_symbols = sorted({str(row.get("symbol", "")).upper() for row in bounded if row.get("symbol")})
    dates = sorted({str(row.get("rebalance_date", "")) for row in bounded if row.get("rebalance_date")})
    universe = _load_universe(Path(settings["universe_path"]) if settings.get("universe_path") else None)
    configured_symbols = sorted(set(universe["symbols"]) or set(source_symbols))
    price_manifest = _load_price_manifest(Path(settings["price_manifest_path"]) if settings.get("price_manifest_path") else None)
    classification = _load_classification_mapping(Path(settings["classification_mapping_path"]) if settings.get("classification_mapping_path") else None)
    membership = _membership_audit(configured_symbols, source_symbols, dates, universe, price_manifest)
    security = _security_identity_mapping(configured_symbols)
    ticker = _ticker_change_audit(configured_symbols)
    delisting = _delisting_audit(configured_symbols, source_symbols, price_manifest)
    liquidity = _liquidity_audit(bounded, configured_symbols)
    class_audit = _classification_audit(configured_symbols, classification)
    breadth = _breadth_universe_coverage(bounded, configured_symbols, source_symbols, dates)
    universe_classification = _classify_universe(universe, membership, ticker, delisting)
    blockers = _promotion_blockers(universe_classification, ticker, delisting, class_audit, settings)
    contract = _universe_contract(
        configured_symbols=configured_symbols,
        source_symbols=source_symbols,
        dates=dates,
        universe=universe,
        membership=membership,
        universe_classification=universe_classification,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "source_artifact_identity": _source_identity(source_path, bounded),
        "universe_contract": contract,
        "universe_classification": universe_classification,
        "membership_audit": membership,
        "security_identity_mapping": security,
        "ticker_change_audit": ticker,
        "delisting_coverage_audit": delisting,
        "corporate_action_audit": _corporate_action_audit(price_manifest),
        "historical_liquidity_audit": liquidity,
        "classification_mapping_audit": class_audit,
        "breadth_universe_coverage": breadth,
        "portfolio_disappearance_audit": _portfolio_disappearance_audit(),
        "promotion_blockers": blockers,
        "large_artifact_handling": {
            "do_not_restart_running_regeneration": True,
            "after_completion_steps": [
                "audit completed base artifact with ml-selector-universe-integrity-audit",
                "re-enrich completed base artifact with stock-level alpha features",
                "attach universe integrity identities to downstream reports",
                "run selector feature-ablation plan-only",
                "block promotion-grade ablation while historical membership remains unresolved",
            ],
        },
        "training_performed": False,
        "final_fit_performed": False,
        "trading_impact": "none",
        "paper_state_modified": False,
        "configuration_hash": _hash(settings),
        "code_commit": MLCoreArtifactWriter.git_commit(),
    }


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("selector_universe_integrity_audit", {}) or {})
    output_dir = raw.get("output_dir") or "reports/ml/development/selector_universe_integrity_audit"
    source = raw.get("source_artifact_path") or raw.get("source_base_artifact_path") or ml.get("stock_level_base_prediction_artifacts_path") or ml.get("stock_level_prediction_artifacts_path")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "source_artifact_path": str(source or ""),
        "universe_path": str(raw.get("universe_path") or ml.get("universe_path") or ""),
        "price_manifest_path": str(raw.get("price_manifest_path") or "data/reference/adjusted_prices/manifest.json"),
        "classification_mapping_path": str(raw.get("classification_mapping_path") or ml.get("sector_reference_path") or "data/reference/sector_by_symbol.json"),
        "output_dir": str(output_dir),
        "allow_csv_fallback": bool(raw.get("allow_csv_fallback", False)),
        "maximum_decision_dates": raw.get("maximum_decision_dates"),
        "maximum_symbols": raw.get("maximum_symbols"),
        "require_historical_universe_for_promotion": bool(raw.get("require_historical_universe_for_promotion", True)),
        "minimum_membership_coverage": float(raw.get("minimum_membership_coverage", 0.95)),
        "minimum_observation_coverage": float(raw.get("minimum_observation_coverage", 0.90)),
        "unknown_membership_action": str(raw.get("unknown_membership_action", "block")).lower(),
    }


def _load_universe(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"path": str(path) if path else None, "symbols": [], "source": "source_artifact_symbols", "classification": "UNKNOWN"}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    symbols = [str(symbol).upper() for symbol in payload.get("symbols", []) if str(symbol).strip()]
    return {"path": str(path), "symbols": sorted(set(symbols)), "source": str(payload.get("source", "unknown")), "payload": payload, "classification": "CURRENT STATIC UNIVERSE"}


def _load_price_manifest(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"path": str(path) if path else None, "symbols": {}, "available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = {str(row.get("symbol", "")).upper(): row for row in payload.get("symbols", []) if row.get("symbol")}
    return {"path": str(path), "payload": payload, "symbols": symbols, "available": True}


def _load_classification_mapping(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"path": str(path) if path else None, "mapping": {}, "status": "UNUSABLE"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(symbol).upper(): str(value) for symbol, value in payload.items() if str(value).strip()}
    return {"path": str(path), "mapping": mapping, "status": "STATIC WITH MATERIAL LIMITATIONS", "identity": _hash(mapping)}


def _membership_audit(configured: Sequence[str], source_symbols: Sequence[str], dates: Sequence[str], universe: Mapping[str, Any], price_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_set = set(source_symbols)
    prices = dict(price_manifest.get("symbols", {}) or {})
    first_date = min(dates) if dates else None
    last_date = max(dates) if dates else None
    rows = []
    for symbol in configured:
        price = prices.get(symbol, {})
        in_artifact = symbol in source_set
        rows.append({
            "symbol": symbol,
            "canonical_security_id": f"static:{symbol}",
            "membership_start_date": first_date,
            "membership_end_date": last_date,
            "eligibility_reason": "configured_static_universe",
            "membership_status": "member_observed" if in_artifact else "member_but_not_in_artifact",
            "source_identity": universe.get("path") or universe.get("source"),
            "source_publication_timestamp": "",
            "mapping_confidence": "low_static_not_historical",
            "price_available": bool(price),
            "price_first_date": price.get("first_date"),
            "price_last_date": price.get("last_date"),
            "unknown_membership": True,
            "not_member_on_date_supported": False,
            "insufficient_history_status_available": bool(price),
        })
    for symbol in sorted(source_set - set(configured)):
        rows.append({
            "symbol": symbol,
            "canonical_security_id": f"artifact:{symbol}",
            "membership_start_date": first_date,
            "membership_end_date": last_date,
            "eligibility_reason": "observed_in_source_artifact_not_configured",
            "membership_status": "unknown_membership_observed",
            "source_identity": "source_artifact",
            "source_publication_timestamp": "",
            "mapping_confidence": "unknown",
            "price_available": bool(prices.get(symbol)),
            "price_first_date": (prices.get(symbol) or {}).get("first_date"),
            "price_last_date": (prices.get(symbol) or {}).get("last_date"),
            "unknown_membership": True,
            "not_member_on_date_supported": False,
            "insufficient_history_status_available": bool(prices.get(symbol)),
        })
    return rows


def _security_identity_mapping(symbols: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "security_id": f"static:{symbol}",
        "historical_ticker": symbol,
        "ticker_valid_from": "",
        "ticker_valid_to": "",
        "successor": "",
        "predecessor": "",
        "mapping_source": "symbol_only_no_security_master",
        "mapping_status": "UNRESOLVED_STATIC_TICKER",
    } for symbol in symbols]


def _ticker_change_audit(symbols: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "symbol": symbol,
        "ticker_change_coverage": "unavailable",
        "reused_symbol_coverage": "unavailable",
        "security_identity_risk": "promotion_blocker",
        "status": "NO_SECURITY_MASTER",
    } for symbol in symbols]


def _delisting_audit(configured: Sequence[str], source_symbols: Sequence[str], price_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_symbols = dict(price_manifest.get("symbols", {}) or {})
    rows = []
    for symbol in sorted(set(configured) | set(source_symbols)):
        price = manifest_symbols.get(symbol, {})
        rows.append({
            "symbol": symbol,
            "delisted_expected": "unknown",
            "delisted_present": False,
            "last_available_trading_date": price.get("last_date"),
            "terminal_price_treatment": "unknown_no_delisting_return_table",
            "delisting_return_treatment": "unavailable_not_zero_filled",
            "cash_position_after_disappearance": "unresolved",
            "disappearance_status": "unknown_without_security_master",
            "promotion_status": "block",
        })
    return rows


def _corporate_action_audit(price_manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(price_manifest.get("payload", {}) or {})
    adjusted = {bool(row.get("adjusted_ohlc")) for row in payload.get("symbols", []) if isinstance(row, dict)}
    return {
        "price_source": payload.get("source"),
        "adjusted_ohlc_values": sorted(adjusted),
        "adjustment_owner": payload.get("source", "unknown"),
        "adjustment_timestamp": payload.get("download_date"),
        "historical_revision_behaviour": "provider_adjusted_history_may_revise",
        "volume_adjustment_behaviour": "not audited beyond manifest",
        "compatible_feature_target_semantics": len(adjusted) <= 1,
        "corporate_action_contract_identity": _hash(payload),
    }


def _liquidity_audit(rows: Sequence[Mapping[str, Any]], configured: Sequence[str]) -> list[dict[str, Any]]:
    by_symbol = {symbol: [] for symbol in configured}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if symbol in by_symbol:
            by_symbol[symbol].append(row)
    output = []
    for symbol, symbol_rows in by_symbol.items():
        values = [_number(row.get("average_dollar_volume_21d")) for row in symbol_rows]
        finite = [value for value in values if value is not None]
        output.append({
            "symbol": symbol,
            "historical_liquidity_source": "artifact_average_dollar_volume_21d" if finite else "unavailable",
            "liquidity_eligible": "diagnostic_only",
            "liquidity_metric_min": min(finite) if finite else None,
            "liquidity_metric_median": sorted(finite)[len(finite)//2] if finite else None,
            "future_liquidity_projected_backward": False if finite else "unknown",
            "eligibility_status": "observed_metric_no_threshold_contract" if finite else "missing_metric",
        })
    return output


def _classification_audit(symbols: Sequence[str], classification: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = dict(classification.get("mapping", {}) or {})
    return [{
        "symbol": symbol,
        "classification": mapping.get(symbol, ""),
        "mapping_source": classification.get("path"),
        "mapping_status": classification.get("status"),
        "historically_versioned": False,
        "effective_date": "",
        "mapping_identity": classification.get("identity", ""),
        "industry_relative_eligible": False,
    } for symbol in symbols]


def _breadth_universe_coverage(rows: Sequence[Mapping[str, Any]], configured: Sequence[str], source_symbols: Sequence[str], dates: Sequence[str]) -> list[dict[str, Any]]:
    configured_set = set(configured)
    source_set = set(source_symbols)
    output = []
    for date in dates:
        observed = {str(row.get("symbol", "")).upper() for row in rows if str(row.get("rebalance_date")) == date}
        output.append({
            "rebalance_date": date,
            "historically_eligible_symbol_count": "",
            "configured_static_symbol_count": len(configured_set),
            "symbols_with_usable_observations": len(observed),
            "symbols_missing_price_history": len(configured_set - observed),
            "symbols_excluded_for_liquidity": "",
            "symbols_with_unknown_membership": len(configured_set),
            "membership_coverage_fraction": 0.0,
            "observation_coverage_fraction": len(observed & configured_set) / len(configured_set) if configured_set else 0.0,
            "breadth_universe_identity": _hash({"configured": sorted(configured_set), "source": sorted(source_set)}),
            "promotion_status": "block_unknown_historical_membership",
        })
    return output


def _classify_universe(universe: Mapping[str, Any], membership: Sequence[Mapping[str, Any]], ticker: Sequence[Mapping[str, Any]], delisting: Sequence[Mapping[str, Any]]) -> str:
    if universe.get("classification") == "CURRENT STATIC UNIVERSE":
        return "CURRENT STATIC UNIVERSE"
    if any(row.get("unknown_membership") for row in membership):
        return "UNKNOWN"
    return "PARTIALLY HISTORICAL"


def _promotion_blockers(universe_classification: str, ticker: Sequence[Mapping[str, Any]], delisting: Sequence[Mapping[str, Any]], classification_rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> list[str]:
    blockers = []
    if settings.get("require_historical_universe_for_promotion", True) and universe_classification in {"CURRENT STATIC UNIVERSE", "UNKNOWN"}:
        blockers.append("historical_universe_membership_unresolved")
    if any(row.get("status") == "NO_SECURITY_MASTER" for row in ticker):
        blockers.append("security_identity_and_ticker_change_mapping_unavailable")
    if any(row.get("promotion_status") == "block" for row in delisting):
        blockers.append("delisting_return_coverage_unavailable")
    if any(row.get("mapping_status") != "HISTORICALLY POINT-IN-TIME" for row in classification_rows):
        blockers.append("historical_sector_industry_classification_unavailable")
    return blockers


def _universe_contract(*, configured_symbols: Sequence[str], source_symbols: Sequence[str], dates: Sequence[str], universe: Mapping[str, Any], membership: Sequence[Mapping[str, Any]], universe_classification: str) -> dict[str, Any]:
    contract = {
        "contract_version": UNIVERSE_CONTRACT_VERSION,
        "universe_classification": universe_classification,
        "universe_source_identity": universe.get("path") or universe.get("source"),
        "membership_row_count": len(membership),
        "configured_symbol_count": len(configured_symbols),
        "source_artifact_symbol_count": len(source_symbols),
        "date_coverage": {"first": min(dates) if dates else None, "last": max(dates) if dates else None, "count": len(dates)},
        "unknown_membership_count": sum(1 for row in membership if row.get("unknown_membership")),
        "supports_not_member_state": False,
        "supports_delisting_state": False,
        "supports_ticker_change_state": False,
        "membership_states": ["member_observed", "member_but_not_in_artifact", "unknown_membership_observed"],
    }
    contract["universe_contract_identity"] = _hash(contract)
    return contract


def _portfolio_disappearance_audit() -> dict[str, Any]:
    return {
        "replay_uses_prediction_rows_with_realized_forward_returns": True,
        "missing_next_period_price_handling": "not independently audited in this ticket",
        "delisting_during_hold_handling": "unresolved_without_delisting_return_table",
        "promotion_status": "block_when_terminal_outcomes_unknown",
    }


def _source_identity(path: Path | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if path and path.exists():
        return artifact_identity(path)
    return {"resolved_artifact_path": str(path) if path else None, "row_count": len(rows)}


def _bounded_rows(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    bounded = [dict(row) for row in rows]
    max_dates = settings.get("maximum_decision_dates")
    if max_dates:
        dates = sorted({str(row.get("rebalance_date")) for row in bounded if row.get("rebalance_date")})[-int(max_dates):]
        bounded = [row for row in bounded if str(row.get("rebalance_date")) in set(dates)]
    max_symbols = settings.get("maximum_symbols")
    if max_symbols:
        symbols = sorted({str(row.get("symbol")).upper() for row in bounded if row.get("symbol")})[: int(max_symbols)]
        bounded = [row for row in bounded if str(row.get("symbol")).upper() in set(symbols)]
    return sorted(bounded, key=lambda row: (str(row.get("rebalance_date")), str(row.get("symbol")).upper()))


def _fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Universe Integrity Audit",
        "",
        DIAGNOSTIC_STATUS,
        "",
        f"- Universe classification: {payload['universe_classification']}",
        f"- Membership rows: {payload['universe_contract']['membership_row_count']}",
        f"- Unknown membership count: {payload['universe_contract']['unknown_membership_count']}",
        f"- Promotion blockers: {', '.join(payload['promotion_blockers']) or 'none'}",
        f"- Training performed: {payload['training_performed']}",
        f"- Trading impact: {payload['trading_impact']}",
        "",
        "## Interpretation",
        "",
        "Historical membership, delisting returns, ticker-change mappings, and historical classifications are not proven by the available local data. Breadth and peer-relative features remain diagnostic until those contracts are supplied.",
        "",
    ]
    return "\n".join(lines)

