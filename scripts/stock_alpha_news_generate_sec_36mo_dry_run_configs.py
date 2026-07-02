from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ETF_FUND_SYMBOLS = {
    "GLD", "QQQ", "SPY", "TLT", "VNQ", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY",
}
AUDITED_EXCEPTION_SYMBOLS = {"B"}
ISOLATED_SYMBOLS = {"AAPL", "CIEN"}
DEFAULT_SOURCE_GLOB = "config.stock_alpha_news_collect_sec_company_filings_batch_*_12mo_dry_run.yaml"
DEFAULT_OUTPUT_PREFIX = "config.stock_alpha_news_collect_sec_company_filings_36mo_part"
DEFAULT_REPORT_ROOT = "reports/ml/benchmark/regime_transformer_meta_ensemble_v1"


def build_sec_36mo_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
) -> list[tuple[Path, dict[str, Any]]]:
    if max_symbols_per_config < 1 or max_symbols_per_config > 5:
        raise ValueError("max_symbols_per_config must be between 1 and 5")

    root = Path(config_dir)
    symbols = _eligible_symbols(root.glob(DEFAULT_SOURCE_GLOB))
    groups = _cap_safe_symbol_groups(symbols, max_symbols_per_config=max_symbols_per_config)
    return [
        (
            root / f"{DEFAULT_OUTPUT_PREFIX}_{index:03d}_dry_run.yaml",
            _config_payload(index=index, symbols=group, max_symbols_per_config=max_symbols_per_config),
        )
        for index, group in enumerate(groups, start=1)
    ]


def write_sec_36mo_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
) -> list[Path]:
    generated = build_sec_36mo_dry_run_configs(
        config_dir=config_dir,
        max_symbols_per_config=max_symbols_per_config,
    )
    written: list[Path] = []
    for path, payload in generated:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written


def _eligible_symbols(paths: Iterable[Path]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for path in sorted(paths):
        payload = _read_yaml(path)
        collect = dict(payload.get("ml", {}).get("stock_alpha_news_collect", {}) or {})
        for symbol in collect.get("only_symbols", []) or []:
            normalized = str(symbol).strip().upper()
            if (
                normalized
                and normalized not in seen
                and normalized not in ETF_FUND_SYMBOLS
                and normalized not in AUDITED_EXCEPTION_SYMBOLS
            ):
                seen.add(normalized)
                symbols.append(normalized)
    if not symbols:
        raise ValueError("no eligible SEC company symbols found")
    return symbols


def _cap_safe_symbol_groups(
    symbols: Sequence[str],
    *,
    max_symbols_per_config: int,
) -> list[list[str]]:
    groups: list[list[str]] = []
    buffer: list[str] = []
    for symbol in symbols:
        if symbol in ISOLATED_SYMBOLS:
            if buffer:
                groups.append(buffer)
                buffer = []
            groups.append([symbol])
            continue
        buffer.append(symbol)
        if len(buffer) >= max_symbols_per_config:
            groups.append(buffer)
            buffer = []
    if buffer:
        groups.append(buffer)
    return groups


def _config_payload(
    *,
    index: int,
    symbols: Sequence[str],
    max_symbols_per_config: int,
) -> dict[str, Any]:
    run_name = f"stock_alpha_news_collect_sec_company_filings_36mo_part_{index:03d}_dry_run"
    report_dir = f"{DEFAULT_REPORT_ROOT}/{run_name}/dev"
    return {
        "ml": {
            "stock_alpha_news_collect_report_dir": report_dir,
            "stock_alpha_news_collect_output_path": f"{report_dir}/sec_company_filings_36mo_dry_run_rows.csv",
            "stock_alpha_news_collect": {
                "enabled": True,
                "dry_run": True,
                "output_written": False,
                "allow_overwrite": False,
                "merge_existing": False,
                "backup_existing": False,
                "max_articles_per_provider": 250,
                "max_rows_per_provider": 1000,
                "provider_request_limit": 250,
                "symbols_per_batch": len(symbols),
                "max_symbols_per_run": len(symbols),
                "source_window": "36mo",
                "source_universe": "sec_company_filings_batch_01_to_16_12mo_dry_run",
                "source_batch_policy": "cap_safe_sub_batches",
                "max_symbols_per_generated_config": max_symbols_per_config,
                "cap_starvation_policy": "small_sub_batches_with_attempted_symbol_diagnostics",
                "symbols": list(symbols),
                "only_symbols": list(symbols),
                "rate_limit_sleep_seconds": 0,
                "request_timeout_seconds": 20,
                "start_date": "2023-07-01",
                "end_date": "2026-07-02",
                "providers": {
                    "sec_company_filings": {
                        "enabled": True,
                        "forms": ["8-K", "10-Q", "10-K"],
                        "load_official_sec_company_tickers": True,
                    },
                    "company_press_release_rss": {"enabled": False},
                    "alpha_vantage": {"enabled": False},
                    "sec_edgar": {"enabled": False},
                    "massive_stock_news": {"enabled": False},
                    "finnhub": {"enabled": False},
                    "gdelt": {"enabled": False},
                    "fmp": {"enabled": False},
                    "newsapi": {"enabled": False},
                },
            },
            "stock_alpha_news_enable_transformer": False,
            "research_only": True,
            "trading_impact": "none",
            "production_validated": False,
            "promotion_thresholds_changed": False,
        }
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate cap-safe 36-month SEC company-filings dry-run configs."
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--max-symbols-per-config", type=int, default=4)
    args = parser.parse_args(argv)

    written = write_sec_36mo_dry_run_configs(
        config_dir=args.config_dir,
        max_symbols_per_config=args.max_symbols_per_config,
    )
    print(f"wrote_config_count={len(written)}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
