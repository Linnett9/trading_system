# Ticket 71 Multi-Timeframe Target Authority

- Classification: `MULTI_TIMEFRAME_TARGET_AUTHORITY_IMPLEMENTED`
- Contract version: `multi_timeframe_target_contract.v1`
- Catalogue version: `multi_timeframe_target_catalogue.v1`
- Output directory: `docs\audits\ticket_71`
- Existing target surfaces inventoried: `8`
- Ambiguities recorded: `8`
- Contracts defined: `11`
- Availability validation: `PASSED`
- Manifest trainable example rows: `8`

Daily semantics preserve the legacy `forward_return_10d` value as ten eligible future daily trading sessions while exposing the clearer canonical ID `forward_return_10_sessions__decision_1Day`.

Hourly and five-minute elapsed-minute targets require maturity inside the same regular session. To-close and next-open contracts declare their session-boundary behavior explicitly.

Missing-bar, quarantine, ineligible, represented halt, unknown gap, right-censored, and not-yet-mature states are separate target-resolution outcomes.

No model training, strategy comparison, portfolio replay, paper trading, production deployment, or policy promotion was performed.
