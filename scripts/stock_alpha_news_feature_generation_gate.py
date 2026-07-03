from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


THIN_EXCEPTION_REASON = (
    "full 120-month SEC recovery completed; symbol remains below 10-row floor; "
    "retained as audited thin-symbol exception"
)


def build_stock_alpha_news_feature_generation_gate(
    *,
    coverage_audit: Mapping[str, Any],
    contract_preflight: Mapping[str, Any],
    audited_thin_symbol_exceptions: Sequence[str] = (),
) -> dict[str, Any]:
    exceptions = sorted({symbol.strip().upper() for symbol in audited_thin_symbol_exceptions if symbol.strip()})
    thin_before = _symbols(coverage_audit.get("symbols_with_1_to_9_valid_official_rows") or coverage_audit.get("symbols_with_less_than_10_rows"))
    thin_after = sorted(symbol for symbol in thin_before if symbol not in set(exceptions))
    invalid_rows_by_provider = dict(contract_preflight.get("invalid_rows_by_provider", {}) or {})
    invalid_row_count = sum(int(count or 0) for count in invalid_rows_by_provider.values())
    unresolved_timeouts = _symbols(
        contract_preflight.get("unresolved_provider_timeout_symbols")
        or coverage_audit.get("unresolved_provider_timeout_symbols")
    )
    provider_timeouts = _symbols(
        contract_preflight.get("provider_timeout_symbols")
        or coverage_audit.get("provider_timeout_symbols")
    )
    historical_timeouts = provider_timeouts
    recovered_timeouts = sorted(set(provider_timeouts) - set(unresolved_timeouts))
    blocking_reasons: list[str] = []

    total_official_rows = int(coverage_audit.get("official_row_count") or coverage_audit.get("valid_official_row_count") or 0)
    if total_official_rows <= 0:
        rows_by_symbol = dict(coverage_audit.get("valid_official_rows_by_symbol", {}) or {})
        total_official_rows = sum(int(count or 0) for count in rows_by_symbol.values())
    duplicate_event_key_count = int(contract_preflight.get("duplicate_event_key_count") or coverage_audit.get("duplicate_event_key_count") or 0)
    future_timestamp_count = int(contract_preflight.get("future_timestamp_count") or coverage_audit.get("future_timestamp_count") or 0)
    event_row_mismatch_count = int(coverage_audit.get("event_row_mismatch_count") or 0)
    outputs_outside_reports = int(coverage_audit.get("outputs_outside_reports") or 0)
    touches_data_news = bool(coverage_audit.get("touches_data_news", False))

    if total_official_rows <= 0:
        blocking_reasons.append("total official rows are not above the feature-generation floor")
    if duplicate_event_key_count:
        blocking_reasons.append("duplicate provider/symbol/url/timestamp event keys detected")
    if invalid_row_count:
        blocking_reasons.append("one or more provider rows failed common schema validation")
    if future_timestamp_count:
        blocking_reasons.append("one or more rows have future published_at_utc timestamps")
    if unresolved_timeouts:
        blocking_reasons.append("unresolved provider timeout symbols remain")
    if thin_after:
        blocking_reasons.append("one or more thin symbols lack audited exceptions")
    if event_row_mismatch_count:
        blocking_reasons.append("SEC provider rows and event rows are mismatched")
    if outputs_outside_reports:
        blocking_reasons.append("one or more outputs are outside reports/")
    if touches_data_news:
        blocking_reasons.append("audit indicates data/news writes")

    warnings = _warnings(coverage_audit, contract_preflight, provider_timeouts, recovered_timeouts)
    approved = not blocking_reasons
    return {
        "feature_generation_gate_status": "approved" if approved else "blocked",
        "approved": approved,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "audited_exceptions": {
            "thin_symbols": {
                symbol: THIN_EXCEPTION_REASON
                for symbol in exceptions
                if symbol in set(thin_before)
            }
        },
        "audited_thin_symbol_exceptions": exceptions,
        "thin_symbol_exception_reasons": {
            symbol: THIN_EXCEPTION_REASON
            for symbol in exceptions
            if symbol in set(thin_before)
        },
        "thin_symbols_before_exceptions": thin_before,
        "thin_symbols_after_exceptions": thin_after,
        "historical_provider_timeout_symbols": historical_timeouts,
        "recovered_provider_timeout_symbols": recovered_timeouts,
        "unresolved_provider_timeout_symbols": unresolved_timeouts,
        "total_official_rows": total_official_rows,
        "duplicate_event_key_count": duplicate_event_key_count,
        "invalid_row_count": invalid_row_count,
        "future_timestamp_count": future_timestamp_count,
        "event_row_mismatch_count": event_row_mismatch_count,
        "outputs_outside_reports": outputs_outside_reports,
        "touches_data_news": touches_data_news,
        "next_allowed_step": (
            "build_news_transformer_feature_dataset_report_only"
            if approved
            else "resolve_feature_generation_gate_blockers"
        ),
    }


def _warnings(
    coverage_audit: Mapping[str, Any],
    contract_preflight: Mapping[str, Any],
    provider_timeouts: Sequence[str],
    recovered_timeouts: Sequence[str],
) -> list[str]:
    warnings: list[str] = []
    preflight_reasons = list(contract_preflight.get("unsafe_reasons", []) or [])
    if "contract ingest preflight is report-only and has not approved feature generation" in preflight_reasons:
        warnings.append("contract ingest preflight remains report-only; feature-generation gate performed explicit checks")
    coverage_reasons = list(coverage_audit.get("unsafe_reasons", []) or [])
    if any("fewer than 10 rows" in str(reason) for reason in coverage_reasons):
        warnings.append("coverage audit reported thin symbols; audited exceptions were applied explicitly")
    if provider_timeouts and recovered_timeouts:
        warnings.append("historical provider timeout symbols are visible but recovered")
    return warnings


def _symbols(value: Any) -> list[str]:
    return sorted({str(symbol).strip().upper() for symbol in value or [] if str(symbol).strip()})


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report-only stock-alpha news feature-generation gate.")
    parser.add_argument("--coverage-audit", required=True)
    parser.add_argument("--contract-preflight", required=True)
    parser.add_argument("--audited-thin-symbol-exception", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = build_stock_alpha_news_feature_generation_gate(
        coverage_audit=_read_json(args.coverage_audit),
        contract_preflight=_read_json(args.contract_preflight),
        audited_thin_symbol_exceptions=args.audited_thin_symbol_exception,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
