# Ticket 37 PIT Universe Fixture

This directory is the tracked acceptance reference for the Ticket 37 synthetic
point-in-time universe authority. It intentionally does not contain real market
data and does not certify the production universe.

Fixture cases:

- active asset listed before the research period;
- IPO during the research period;
- delisting during the research period;
- ticker change with unchanged permanent asset ID;
- merger predecessor and successor separation;
- spin-off parent and child eligibility dates;
- bankruptcy terminal event;
- unknown listing date;
- unknown delisting status;
- conflicting membership sources;
- authority correction recorded after the original event;
- knowledge cutoff before that correction;
- static current-registry adapter classified as uncertified;
- provider alias change independent from permanent identity.

The Ticket 36 audit artifacts remain under ignored `reports/audits` paths:

- `reports/audits/historical_universe_authority_audit_20260728/historical_universe_authority_audit.md`
- `reports/audits/historical_universe_authority_audit_20260728/historical_universe_authority_audit_summary.json`
