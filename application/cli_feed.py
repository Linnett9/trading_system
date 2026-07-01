


def build_feed(config):
    provider = config["backtest"].get("provider", "alpaca").lower()
    if provider == "stooq":
        from infrastructure.data.stooq_data_feed import StooqDataFeed

        return StooqDataFeed()
    if provider == "stooq_csv":
        from infrastructure.data.stooq_csv_data_feed import StooqCsvDataFeed

        return StooqCsvDataFeed(
            data_dir=config["backtest"].get("data_dir", "data/raw/stooq")
        )
    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed

        return StooqParquetDataFeed(
            data_dir=config["backtest"].get(
                "data_dir", "data/processed/stooq_parquet"
            )
        )
    if provider != "alpaca":
        raise ValueError(f"Unsupported historical data provider: {provider}")

    alpaca_config = config.get("alpaca", {})
    if not alpaca_config.get("api_key") or not alpaca_config.get("secret_key"):
        raise RuntimeError(
            "Alpaca data provider requires ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "environment variables."
        )

    from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed

    return AlpacaDataFeed(
        api_key=alpaca_config["api_key"],
        secret_key=alpaca_config["secret_key"],
        data_feed=config["backtest"].get("data_feed", "iex"),
        adjustment=config["backtest"].get("data_adjustment", "all"),
        historical_bar_limit=int(
            config["backtest"].get("historical_bar_limit", 10_000)
        ),
    )
