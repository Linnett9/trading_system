# Ticket 68A-P Summary

## Classification

MAINTAINED_CALENDAR_AUTHORITY_IMPLEMENTED

## Repository State

Branch: `feature/selector-compute-adoption-20260718`.

Commit: `0c09699a54a9f5d49b85e2daf4d1740b481afac3`.

The worktree is dirty with substantial pre-existing unrelated changes. Ticket 68A-P edits were kept to calendar authority validation, focused tests, one calendar-sensitive legacy assertion, and Ticket 68A audit artifacts.

## Dependency State

Python: `3.11.9`.

Pinned requirement: `exchange-calendars==4.13.2` in `requirements.txt`.

Validation environment: `.venv-ticket-68a` with system site packages.

Install command: `.\\.venv-ticket-68a\\Scripts\\python.exe -m pip install exchange-calendars==4.13.2`.

Resolved package: `exchange_calendars 4.13.2`.

Import result: `import exchange_calendars` succeeded; `exchange_calendars.get_calendar("XNYS")` succeeded.

Dependency conflicts: `pip check` reports an existing unrelated conflict: `torch 2.13.0+cpu` requires `setuptools>=77.0.3`, while the venv has `setuptools 65.5.0`.

## Maintained Authority Activation

The current validated Ticket 68A-P runtime used maintained mode, not fallback.

Runtime backend status: `MAINTAINED`.

Runtime package/version: `exchange_calendars 4.13.2`.

Runtime fallback state: `fallback_used=false`.

Calendar version identity: `market_calendar_authority.v1:exchange_calendars:exchange_calendars:4.13.2`.

Validation schedule hash: `DB2DFE26E3CA6C96B23DAFD01BDEEFA33B356362AEE4991F1826E8DD222E4E1D`.

## Contract And Adapter

All direct third-party calendar access remains centralized in `infrastructure.data.calendar_authority`.

`infrastructure.data.market_sessions` remains a compatibility wrapper for legacy callers, including `session_type()`, RTH timestamp generation, previous/next session helpers, and calendar status lookup.

The deterministic compact fallback remains present and explicit.

## Maintained-Mode Validation

Command:

`.\\.venv-ticket-68a\\Scripts\\python.exe -m pytest tests/test_calendar_authority.py tests/test_market_information_availability_authority.py tests/test_dataset_build_manifest.py tests/test_frozen_selector_lineage_guard.py tests/test_selector_dataset_lineage.py tests/test_research_certification.py tests/test_historical_bar_backfill.py tests/test_alpaca_5m_symbol_year_finalizer.py tests/test_stock_level_prediction_artifacts.py -q`

Result: `191 passed, 1 warning`.

The warning is the existing registry deprecation warning for deprecated read-only `target_contract`.

Maintained-path assertions cover regular sessions, holidays, weekends, early closes, DST start/end, pre-market, after-hours, previous session, next session, special closures, outside range handling, deterministic version identity, and deterministic schedule hashes. Maintained-path cases assert `fallback_used=false`.

## Fallback Validation

Forced fallback command:

`.\\.venv-ticket-68a\\Scripts\\python.exe -m pytest tests/test_calendar_authority.py -k "fallback or session_type or regular_early_close_holiday_weekend" -q`

Result: `3 passed, 10 deselected`.

Dependency-unavailable fallback command:

`python -m pytest tests/test_calendar_authority.py tests/test_market_information_availability_authority.py -q`

Result: `23 passed, 5 skipped, 1 warning`.

Fallback records `fallback_used=true`, fallback version `compact_nyse_like_rth_2016_2026_v1`, and fails closed outside verified fallback coverage.

## Maintained/Fallback Comparison

Comparison artifact: `maintained_fallback_comparison.csv`.

Compared fields: session dates, open timestamps, close timestamps, early-close flags, holiday/special closure state.

Compared dates: `2018-12-05`, `2021-12-31`, `2025-01-09`, `2025-07-03`, `2025-11-27`, `2025-11-28`, `2025-12-24`, `2025-12-25`, `2026-01-01`, `2026-01-02`, `2026-01-03`, `2026-03-09`, `2026-11-02`, `2026-12-24`.

Recorded mismatch: `2021-12-31` is a maintained authority correction. `exchange_calendars` marks XNYS open; the compact fallback marks it closed as an observed New Year holiday.

All other compared dates match on the required comparison fields.

## Ticket 68 Integration

Ticket 68 availability results now record calendar authority version, calendar identity, package, package version, schedule hash, exchange, fallback state, source/base status, and closure reason.

Maintained-mode Ticket 68 cases cover pre-market filing, regular-session publication, after-close publication, early close, holiday, weekend, DST, outside range, and represented halt.

## Manifest Integration

Generic dataset manifests store structured `market_calendar_authority` identity alongside the legacy `market_calendar_authority_version` string.

Frozen selector manifest compatibility remains preserved through the selector dataset lineage tests.

Legacy manifests without structured calendar identity remain readable and cannot claim the maintained authority unless the structured identity is present.

## Certification Integration

Ticket 65 research-certification envelopes lift structured calendar authority identity into `authority_versions.market_calendar_authority`.

Focused tests prove calendar schedule hash/version lineage is preserved and changes when the structured calendar authority identity changes.

## Materialised Schedule

Artifact: `materialised_schedule.parquet`.

Rows: 14 bounded validation dates only.

Fields include calendar ID, authority version, authority version identity, session date, UTC open, UTC close, early-close flag, closure metadata, package, package version, and schedule hash.

## Artifacts

- `calendar_source_audit.json`
- `calendar_authority_contract.json`
- `calendar_version.json`
- `calendar_coverage.json`
- `calendar_conflicts.csv`
- `maintained_fallback_comparison.csv`
- `materialised_schedule.parquet`
- `calendar_validation.json`
- `ticket_68a_summary.md`

## Backwards Compatibility

`session_type()` still returns the legacy strings `pre_market`, `rth`, and `after_hours`.

`exchange_session_context()` preserves prior fields and adds calendar-lineage fields.

The compact fallback remains deterministic and available for dependency-unavailable runs.

## Remaining Risks

The repo has many unrelated dirty changes, so this validation should be reviewed or committed with care.

The venv uses system site packages; the maintained calendar package itself is isolated in `.venv-ticket-68a`, but shared installed packages are still visible.

The `pip check` setuptools conflict is unrelated to `exchange-calendars`, but it remains an environment hygiene item.

Future `exchange_calendars` upgrades can alter historical or future schedules; exact version and schedule hashes must continue to be retained for replay.

## Recommended Next Action

Promote the exact pinned dependency into the project runtime used for research and production validation, then run the same focused maintained-mode suite before any promotion-grade calendar-dependent release.
