from __future__ import annotations

from pathlib import Path

from core.research.ml.stock_feature_bank import (
    ACTIVE_TIMEFRAMES,
    build_market_data_inventory,
    build_stock_feature_bank,
    validate_stock_feature_bank,
    write_feature_validation_report,
)


def run_market_data_inventory(config: dict) -> None:
    ml_config = config.get("ml", {})
    market_config = ml_config.get("market_data", {})
    data_root = market_config.get("processed_root", "data/processed")
    report_dir = ml_config.get("stock_feature_report_dir", "reports/data")
    summary = build_market_data_inventory(data_root=data_root, report_dir=report_dir)
    print("\nMARKET DATA INVENTORY")
    print("mode=research | trading_impact=none")
    for timeframe, item in summary["timeframes"].items():
        print(
            f"{timeframe}: symbols={item['symbol_count']} rows={item['total_rows']} "
            f"range={item['earliest_timestamp']} -> {item['latest_timestamp']} "
            f"duplicates={item['duplicate_timestamp_total']} stale={item['stale_symbol_count']}"
        )
    print(f"Reports: {report_dir}/market_data_inventory.*")


def run_build_stock_features(config: dict, *, timeframes: list[str] | None = None, symbols: list[str] | None = None) -> None:
    ml_config = config.get("ml", {})
    market_config = ml_config.get("market_data", {})
    data_root = market_config.get("processed_root", "data/processed")
    output_dir = ml_config.get("stock_feature_output_dir", "cache/ml/features")
    report_dir = ml_config.get("stock_feature_report_dir", "reports/data")
    requested = timeframes or list(ml_config.get("stock_feature_timeframes", ACTIVE_TIMEFRAMES))
    results = []
    print("\nSTOCK FEATURE BANK BUILD")
    print("mode=research | trading_impact=none")
    for timeframe in requested:
        result = build_stock_feature_bank(
            timeframe,
            data_root=data_root,
            output_dir=output_dir,
            symbols=symbols,
        )
        results.append(result)
        print(
            f"{result.timeframe}: {result.row_count} rows, {result.column_count} columns, "
            f"{result.symbol_count} symbols -> {result.path}"
        )
    write_feature_validation_report([result.path for result in results], report_dir=report_dir)
    print(f"Validation report: {report_dir}/stock_feature_bank_validation.*")


def run_validate_stock_features(config: dict, *, timeframes: list[str] | None = None) -> None:
    ml_config = config.get("ml", {})
    output_dir = Path(ml_config.get("stock_feature_output_dir", "cache/ml/features"))
    report_dir = ml_config.get("stock_feature_report_dir", "reports/data")
    requested = timeframes or list(ml_config.get("stock_feature_timeframes", ACTIVE_TIMEFRAMES))
    paths = [output_dir / f"stock_features_{timeframe}.parquet" for timeframe in requested]
    payload = write_feature_validation_report(paths, report_dir=report_dir)
    print("\nSTOCK FEATURE BANK VALIDATION")
    print("mode=research | trading_impact=none")
    for item in payload["datasets"]:
        result = validate_stock_feature_bank(item["path"])
        print(
            f"{result.timeframe}: rows={result.row_count} columns={result.column_count} "
            f"symbols={result.symbol_count} duplicate_keys={result.duplicate_key_count} "
            f"infinite_values={result.infinite_value_count} -> {result.path}"
        )
    print(f"Validation report: {report_dir}/stock_feature_bank_validation.*")
