# Ticket 62A-R PIT Authority Population Reference

## Authority

- Authority version: `ticket62_research_grade_partial_authority_v1`
- Classification: `RESEARCH_GRADE_PARTIAL_AUTHORITY`
- Output root: `data/reference/pit_authority/version=ticket62_research_grade_partial_authority_v1`
- Universe ID: `ticket62_reconstructed_us_liquid_v1`
- Permitted use: research reconstruction only; not promotion grade
- Model training or promotion: none

## Artifact Identity

- Artifact inventory hash: `8b3503d38717a421b74001c80d1a4cae70e893cee38783004ed20245f377721f`
- Coverage summary hash: `7b4364c2e810675f843008b25d5fc113a3fed8e9a02721c65abe28dc1550ed0c`
- Validation artifact hash: `99c0a9928d5a5b43e5371e6de4a53e735d7652f78c74b0b18cd484d727b9c547`
- Ticket summary hash: `a449c6b78d1253774c0bb321335bd57710996a6e8ac9d025b265c2875fa4c9ce`
- Validation self-hash policy: `pit_authority_validation.json` is excluded from its own non-recursive artifact hash map.

## Coverage Snapshot

- Symbols requested: 514
- Symbols populated: 514
- Symbols unresolved: 514
- Security identities created: 514
- Externally corroborated identities: 0
- Internal reconstructed identities: 514
- Static-symbol fallbacks: 514
- CIK coverage: 0
- Exchange coverage: 0
- Symbol history rows: 3,084
- Verified historical symbol rows: 0
- Corporate event rows: 514
- Official corporate event rows: 0
- Universe membership rows: 26,604
- Reconstructed eligibility intervals: 26,604
- Unknown provider knowledge-time rows: 57,320
- Identity unresolved/conflict rows: 1,028
- Universe unresolved/conflict rows: 135

## Validation

- Required artifacts: 20
- Missing required artifacts: 0
- JSON parse validation: passed
- Parquet open and row-count reconciliation: passed
- CSV schema validation: passed
- Non-self artifact hash validation: passed
- Temporary or incomplete output files: none
- Bounded determinism check: passed on two independent 12-symbol roots under `C:\tmp`
- Tests: `python -m py_compile core/research/ml/reference/ticket_62_pit_authority_population.py`; `python -m pytest tests/test_ticket_62_pit_authority_population.py tests/test_historical_identity_authority.py tests/test_pit_universe_authority.py tests/test_market_information_availability_authority.py -rs`
- Test result: 49 passed, 1 skipped because optional `exchange_calendars` is not installed

## Known Limitations

- SEC submissions bulk evidence is present but not extracted into a complete ticker-to-CIK mapping.
- Complete official historical universe membership is not available in this workspace.
- Complete official IPO, delisting, exchange-transfer, symbol-change and corporate-action histories are not available.
- Current registry symbols and aliases are current-scope evidence only and are not certified historical identity.
- Canonical daily-v2 bars are used for observed market presence and rules-based PIT eligibility reconstruction, not official membership.
- Provider publication and first-seen knowledge times remain incomplete, blocking promotion-grade classification.

## Provenance

- Git head recorded by validation: `0c09699a54a9f5d49b85e2daf4d1740b481afac3`
- Git branch recorded by validation: `feature/selector-compute-adoption-20260718`
- Implementation reference: `core/research/ml/reference/ticket_62_pit_authority_population.py`
- Focused test reference: `tests/test_ticket_62_pit_authority_population.py`
- Workspace note: `data/` is ignored by `.gitignore`; this document is the tracked reference for the ignored authority root.
