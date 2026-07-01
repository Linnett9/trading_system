from core.research.ml.data_inventory import build_data_inventory
from core.research.ml.universe_builder import build_universe_files


def run_ml_data_inventory(config):
    ml_config = config.get("ml", {})
    inventories = build_data_inventory(
        parquet_dir=ml_config.get("parquet_dir", ml_config.get(
            "stooq_parquet_dir", "data/processed/stooq_parquet"
        )),
        output_dir=ml_config.get("inventory_output_dir", "reports/ml"),
        min_history_years=int(ml_config.get("min_history_years", 9)),
        max_latest_gap_days=int(ml_config.get("max_latest_gap_days", 14)),
        min_average_dollar_volume_252d=float(
            ml_config.get("min_average_dollar_volume_252d", 50_000_000)
        ),
    )
    included_count = sum(1 for item in inventories if item.included)
    missing_count = len(inventories) - included_count
    output_dir = ml_config.get("inventory_output_dir", "reports/ml")
    print("\nML DATA INVENTORY")
    print("mode=research | trading_impact=none")
    print(f"Scanned symbols: {len(inventories)}")
    print(f"Included symbols: {included_count}")
    print(f"Excluded symbols: {missing_count}")
    print(f"Inventory: {output_dir}/data_inventory.json")
    print(f"Coverage CSV: {output_dir}/symbol_coverage.csv")

def run_ml_build_universes(config):
    ml_config = config.get("ml", {})
    inventory_output_dir = ml_config.get("inventory_output_dir", "reports/ml")
    results = build_universe_files(
        inventory_path=f"{inventory_output_dir}/data_inventory.json",
        output_dir=ml_config.get("universe_output_dir", "data/reference/universes"),
        parquet_dir=ml_config.get("parquet_dir", ml_config.get(
            "stooq_parquet_dir", "data/processed/stooq_parquet"
        )),
        inventory_output_dir=inventory_output_dir,
        min_history_years=int(ml_config.get("min_history_years", 9)),
        max_latest_gap_days=int(ml_config.get("max_latest_gap_days", 14)),
        min_average_dollar_volume_252d=float(
            ml_config.get("min_average_dollar_volume_252d", 50_000_000)
        ),
    )
    print("\nML UNIVERSE BUILD")
    print("mode=research | trading_impact=none")
    for result in results:
        print(
            f"{result.name}: {result.symbol_count} symbols "
            f"(available={result.available_count}) -> {result.path}"
        )
