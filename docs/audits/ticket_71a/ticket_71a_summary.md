# Ticket 71A Bounded Five-Minute Target Pilot

- Classification: `FIVE_MINUTE_TARGET_PILOT_VALIDATED`
- Output directory: `docs\audits\ticket_71a`
- Selected symbols: `13`
- Source rows loaded: `25195`
- Target rows: `1872`
- Session validation: `PASSED`
- PIT validation: `PASSED`
- Missing five-minute bars observed: `0`

Coverage by contract:
- `forward_return_30m__decision_5m` rows=468 trainable=312 session_boundary_failures=156 missing=0
- `forward_return_60m__decision_5m` rows=468 trainable=221 session_boundary_failures=247 missing=0
- `forward_return_next_open__decision_5m` rows=468 trainable=390 session_boundary_failures=78 missing=0
- `forward_return_to_close__decision_5m` rows=468 trainable=390 session_boundary_failures=78 missing=0

The pilot uses Ticket 71 target contracts only. No fitting, Sharpe calculation, stock ranking, position creation, replay, or production/paper trading changes were performed.
