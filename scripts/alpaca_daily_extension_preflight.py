from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infrastructure.data.alpaca_daily_preflight import run_preflight, run_reconcile_only


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded Alpaca daily extension discovery/preflight.")
    parser.add_argument("--report-root", type=Path, default=Path("reports/data_lineage/alpaca_daily_extension_preflight"))
    parser.add_argument("--stooq-root", type=Path, default=Path("data/processed/stooq_parquet"))
    parser.add_argument("--asset-registry", type=Path, default=Path("data/reference/assets/canonical_asset_registry.csv"))
    parser.add_argument("--alias-registry", type=Path, default=Path("data/reference/assets/provider_symbol_aliases.csv"))
    parser.add_argument("--daily-archive-root", type=Path, default=Path("data/processed/alpaca/symbol_bars/sip/1d"))
    parser.add_argument("--smoke-output-root", type=Path, default=Path("reports/market_data/historical_bar_backfill/daily_sip_smoke_abcb"))
    parser.add_argument("--smoke-archive-root", type=Path, default=Path("data/processed/alpaca/symbol_bars/sip/1d_smoke"))
    parser.add_argument("--smoke-symbols", nargs="+", default=["AAPL", "SPY", "BRK-B", "ABCB"])
    parser.add_argument("--alpaca-archive-root", type=Path, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=180)
    parser.add_argument("--symbol-batch-size", type=int, default=50)
    parser.add_argument("--date-window-days", type=int, default=31)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.reconcile_only:
        if args.alpaca_archive_root is None:
            parser.error("--reconcile-only requires --alpaca-archive-root")
        if not args.start or not args.end:
            parser.error("--reconcile-only requires --start and --end")
        symbols = _symbols_arg(args.symbols or args.smoke_symbols)
        payload = run_reconcile_only(
            alpaca_archive_root=args.alpaca_archive_root,
            stooq_root=args.stooq_root,
            report_root=args.report_root,
            symbols=symbols,
            start=args.start,
            end=args.end,
            smoke_output_root=args.smoke_output_root,
            dry_run=args.dry_run,
        )
        summary = payload["reconciliation"]
        print(json.dumps({
            "reconcile_only": True,
            "alpaca_archive_row_count": summary["alpaca_archive_row_count"],
            "stooq_comparison_row_count": summary["stooq_comparison_row_count"],
            "matched_rows": summary["matched_rows"],
            "alpaca_only_rows": summary["alpaca_only_rows"],
            "stooq_only_rows": summary["stooq_only_rows"],
            "provider_compatibility_decision": summary["provider_compatibility_decision"],
            "report_root": str(args.report_root),
            "dry_run": args.dry_run,
        }, indent=2))
        return 0

    payload = run_preflight(
        report_root=args.report_root,
        stooq_root=args.stooq_root,
        asset_registry=args.asset_registry,
        alias_registry=args.alias_registry,
        daily_archive_root=args.daily_archive_root,
        smoke_output_root=args.smoke_output_root,
        smoke_archive_root=args.smoke_archive_root,
        smoke_symbols=args.smoke_symbols,
        requests_per_minute=args.requests_per_minute,
        full_universe_symbol_batch_size=args.symbol_batch_size,
        date_window_days=args.date_window_days,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "direct_daily_supported": payload["direct_daily_supported"],
        "alpaca_timeframe": payload["alpaca_timeframe"],
        "source_freshness_rows": payload["source_freshness_rows"],
        "universe_rows": payload["universe_rows"],
        "valid_alpaca_mappings": payload["valid_alpaca_mappings"],
        "recommended_overlap_start": payload["plan"]["recommended_overlap_start"],
        "latest_spy_session": payload["plan"]["latest_spy_session"],
        "estimated_request_count": payload["plan"]["estimated_request_count"],
        "report_root": str(args.report_root),
        "dry_run": args.dry_run,
    }, indent=2))
    return 0


def _symbols_arg(values: Sequence[str]) -> list[str]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip().upper() for part in str(value).split(",") if part.strip())
    return symbols


if __name__ == "__main__":
    raise SystemExit(main())
