# Ticket 38 Historical Identity Fixture

Synthetic bounded fixture only. This is not production symbol history and does
not certify survivorship-bias safety.

The fixture covers:

- permanent IDs across ticker and company-name changes;
- provider aliases, suffixes, conflicting aliases, and ambiguous aliases;
- ticker reuse by distinct securities;
- merger, acquisition, spin-off, bankruptcy, liquidation, delisting, relisting,
  exchange-transfer, and split lineage;
- a retrospective correction recorded after its effective date;
- a Ticket 37 PIT universe adapter case for `perm_ticker_004`.
