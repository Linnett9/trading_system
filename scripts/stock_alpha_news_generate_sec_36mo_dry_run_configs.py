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
DEFAULT_REPORT_ROOT = "reports/ml/benchmark/regime_transformer_meta_ensemble_v1"
SUPPORTED_WINDOW_MONTHS = {36, 60, 120}
WINDOW_START_DATES = {
    36: "2023-07-01",
    60: "2021-07-01",
    120: "2016-07-01",
}
WINDOW_END_DATE = "2026-07-02"
WINDOW_RECOVERY_SYMBOLS = {
    120: ("AMD", "GOOGL", "NFLX"),
}


def build_sec_36mo_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
) -> list[tuple[Path, dict[str, Any]]]:
    return build_sec_window_dry_run_configs(
        config_dir=config_dir,
        max_symbols_per_config=max_symbols_per_config,
        window_months=36,
    )


def build_sec_window_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
    window_months: int = 36,
) -> list[tuple[Path, dict[str, Any]]]:
    if max_symbols_per_config < 1 or max_symbols_per_config > 5:
        raise ValueError("max_symbols_per_config must be between 1 and 5")
    if window_months not in SUPPORTED_WINDOW_MONTHS:
        raise ValueError("window_months must be one of 36, 60, or 120")

    root = Path(config_dir)
    symbols = _eligible_symbols(
        root.glob(DEFAULT_SOURCE_GLOB),
        additional_symbols=WINDOW_RECOVERY_SYMBOLS.get(window_months, ()),
    )
    groups = _cap_safe_symbol_groups(symbols, max_symbols_per_config=max_symbols_per_config)
    output_prefix = f"config.stock_alpha_news_collect_sec_company_filings_{window_months}mo_part"
    return [
        (
            root / f"{output_prefix}_{index:03d}_dry_run.yaml",
            _config_payload(
                index=index,
                symbols=group,
                max_symbols_per_config=max_symbols_per_config,
                window_months=window_months,
            ),
        )
        for index, group in enumerate(groups, start=1)
    ]


def write_sec_36mo_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
) -> list[Path]:
    return write_sec_window_dry_run_configs(
        config_dir=config_dir,
        max_symbols_per_config=max_symbols_per_config,
        window_months=36,
    )


def write_sec_window_dry_run_configs(
    *,
    config_dir: str | Path = "config",
    max_symbols_per_config: int = 4,
    window_months: int = 36,
) -> list[Path]:
    generated = build_sec_36mo_dry_run_configs(
        config_dir=config_dir,
        max_symbols_per_config=max_symbols_per_config,
    ) if window_months == 36 else build_sec_window_dry_run_configs(
        config_dir=config_dir,
        max_symbols_per_config=max_symbols_per_config,
        window_months=window_months,
    )
    written: list[Path] = []
    for path, payload in generated:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written


def _eligible_symbols(paths: Iterable[Path], *, additional_symbols: Iterable[str] = ()) -> list[str]:
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
    for symbol in additional_symbols:
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
    window_months: int,
) -> dict[str, Any]:
    window_label = f"{window_months}mo"
    run_name = f"stock_alpha_news_collect_sec_company_filings_{window_label}_part_{index:03d}_dry_run"
    report_dir = f"{DEFAULT_REPORT_ROOT}/{run_name}/dev"
    return {
        "ml": {
            "stock_alpha_news_collect_report_dir": report_dir,
            "stock_alpha_news_collect_output_path": f"{report_dir}/sec_company_filings_{window_label}_dry_run_rows.csv",
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
                "source_window": window_label,
                "source_universe": "sec_company_filings_batch_01_to_16_12mo_dry_run",
                "source_batch_policy": "cap_safe_sub_batches",
                "max_symbols_per_generated_config": max_symbols_per_config,
                "cap_starvation_policy": "small_sub_batches_with_attempted_symbol_diagnostics",
                "symbols": list(symbols),
                "only_symbols": list(symbols),
                "rate_limit_sleep_seconds": 0,
                "request_timeout_seconds": 20,
                "start_date": WINDOW_START_DATES[window_months],
                "end_date": WINDOW_END_DATE,
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
        description="Generate cap-safe SEC company-filings dry-run configs for a fixed history window."
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--max-symbols-per-config", type=int, default=4)
    parser.add_argument("--window-months", type=int, choices=sorted(SUPPORTED_WINDOW_MONTHS), default=36)
    args = parser.parse_args(argv)

    written = write_sec_window_dry_run_configs(
        config_dir=args.config_dir,
        max_symbols_per_config=args.max_symbols_per_config,
        window_months=args.window_months,
    )
    print(f"wrote_config_count={len(written)}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
