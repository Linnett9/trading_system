# Ticket 68B News Contract Triage

Classification: `ALL_NEWS_CONTRACT_FAILURES_RESOLVED`

## Repository State

- Branch: `feature/selector-compute-adoption-20260718`
- Commit: `0c09699a54a9f5d49b85e2daf4d1740b481afac3`
- Starting dirty tree: dirty before Ticket 68B work; branch was ahead of origin by 3 commits with many unrelated modified, deleted, staged, and untracked files. Unrelated user changes were left intact.
- Baseline output: `docs/audits/ticket_68b_baseline_pytest.txt`
- Historical check output: `docs/audits/ticket_68b_historical_pytest.txt`

## Baseline Failures

`python -m pytest tests/test_stock_alpha_news_contract.py -vv` produced `7 failed, 129 passed`.

The same seven node ids failed at clean historical commit `959b5dbd7345eea0fd542dde60aa5bc95b5647b3`, so they were not Ticket 68 regressions.

## Failure Classification

| Test | Classification | Root cause |
| --- | --- | --- |
| `test_news_feature_diagnostics_tiny_report_is_read_only` | `PRE_EXISTING_FIXTURE_DRIFT` | Tiny diagnostics config depended on an ignored `reports/` feature artifact that was absent in a clean checkout. |
| `test_news_source_setup_check_gdelt_only_needs_no_key_and_is_read_only` | `PROVIDER_SETUP_DEFECT` | News setup checker scanned unrelated project overlay credentials and misreported source setup. |
| `test_news_source_setup_check_key_presence_without_value_disclosure` | `PROVIDER_SETUP_DEFECT` | Same over-broad literal scan masked the actual missing keyed news provider setup. |
| `test_stock_alpha_news_coverage_audit_can_prefer_12mo_sec_artifacts` | `SEC_ARTIFACT_SELECTION_DEFECT` | SEC selector searched for slash-specific markers and included both short and 12-month artifacts on Windows. |
| `test_stock_alpha_news_coverage_audit_can_merge_36mo_pilots_without_replacing_12mo` | `SEC_ARTIFACT_SELECTION_DEFECT` | 36-month pilot rows leaked into the computed 12-month baseline on Windows. |
| `test_contract_ingest_preflight_can_prefer_12mo_sec_artifacts` | `SEC_ARTIFACT_SELECTION_DEFECT` | Ingest preflight had the same slash-specific prefer_12mo selector defect. |
| `test_contract_ingest_preflight_can_merge_36mo_pilots_without_replacing_12mo` | `SEC_ARTIFACT_SELECTION_DEFECT` | Ingest preflight had the same baseline/overlay selector defect. |

## Repairs

- Added tracked tiny diagnostics fixtures under `tests/fixtures/stock_alpha_news/feature_diagnostics_tiny/`.
- Pointed `config/config.stock_alpha_news_feature_diagnostics_tiny_fixture.yaml` at that tracked fixture instead of ignored `reports/`.
- Scoped source setup literal-secret detection to `stock_alpha_news_collect.providers`.
- Normalized SEC artifact path matching in both `scripts/stock_alpha_news_coverage_audit.py` and `scripts/stock_alpha_news_contract_ingest_preflight.py`.
- Added SEC artifact candidate diagnostics with selected/rejected reason codes.
- Strengthened tests to assert provider literal-scan scope and SEC rejection reason codes.

## SEC Artifact Selection Findings

- Artifact identity is selected deterministically by normalized path markers, batch id, and existing SEC event key de-duplication.
- `prefer_12mo` now selects the batch `_12mo_dry_run` artifact over shorter batch dry runs.
- `merge_36mo_pilots` preserves 12-month baseline artifacts and adds non-`data/news` 36-month pilot overlays only.
- `data/news` duplicates are rejected with `rejected_data_news_duplicate_artifact`.
- The fixture rows include symbol, CIK, form type, filing date, published timestamp, collection timestamp, filing URL, and accession number. They do not model amendment status or acceptance timestamps; Ticket 68 authority/fundamental tests cover SEC acceptance timestamp precedence and date-only fallback.

## Validation

- `tests/test_stock_alpha_news_contract.py`: `136 passed`
- `tests/test_market_information_availability_authority.py`: `14 passed`
- News PIT subset: `4 passed, 132 deselected`
- `tests/test_stock_fundamentals.py` plus date-only news timestamp test: `23 passed`

## Remaining Failures

None in the required Ticket 68B validation.

## Remaining Risks

- The broader repository remains dirty with many unrelated local changes.
- The SEC artifact-selection helper remains duplicated across two scripts; behavior is now aligned, but a future shared helper would reduce drift.
- Fixture SEC event rows do not include amendment-status coverage; existing SEC/fundamental tests cover acceptance and availability timing.

## Recommended Next Action

Stage and review the Ticket 68B files separately from the larger dirty worktree, then consider extracting the duplicated SEC artifact selection helper into a shared module in a follow-up cleanup.
