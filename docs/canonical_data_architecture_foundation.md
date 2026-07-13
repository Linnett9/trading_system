# Canonical Data Architecture Foundation

This first bounded foundation ticket introduces shared reference contracts only. It does not move, rewrite or merge raw market data, model artifacts, SEC data, news data, exposure datasets or execution datasets.

The `config/universes/alpaca_514_symbols.txt` file is treated as the broad collection universe. It is not the daily selector universe, and collection membership does not imply selector eligibility. Selector eligibility remains point in time and must be represented later on the daily selector spine with explicit decision timestamps and eligibility reasons.

Stooq daily data remains the current daily research source for selector work. Alpaca SIP 5-minute bars remain the intended intraday archive for execution and intraday research. These sources are not merged into a single table by this ticket.

Future datasets should join through stable `asset_id`, deterministic daily-spine `row_id`, explicit event/availability/decision/label timestamps and versioned manifests. Provider symbols are aliases, not model-dataset primary keys.

The canonical asset registry is a reference layer. It does not alter current model outputs, target definitions, price features, training flows, live trading or paper trading.

