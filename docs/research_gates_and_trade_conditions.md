# Research Gates And Trade Conditions

This document separates research correctness, research quality, candidate
triage, and paper/live trading gates. It is based on current stock-alpha report
guardrails, experiment validation logic, portfolio replay/sweep checks, and
repository architecture. Current stock-alpha outputs are research-only and do
not authorize trades.

## Required Guardrails

Current stock-alpha research reports should preserve:

```text
research_only: true
trading_impact: none
production_validated: false
promotion_thresholds_changed: false
```

If any report violates those fields, stop and inspect the producing module or
config before using the output.

## 1. Research Correctness Gates

These gates answer: can we trust the generated artifact bundle mechanically?

| Gate | Pass condition | Fail examples |
|---|---|---|
| Canonical output root | Stage outputs are under the configured stock-alpha run-size dir. | Files written to `reports/ml/benchmark/ml/` unexpectedly. |
| No legacy mixing | Legacy paths are not used when `stock_alpha_allow_legacy_output_paths: false`. | Experiment report detects legacy paths. |
| Required artifacts exist | Stock artifact, feature audit, benchmark, targets, portfolio, sweep, and summary exist as required by report level. | Missing JSON/CSV/MD outputs. |
| Fresh output bundle | Mixed artifact mtimes are within configured max age. | Stale mixed outputs across stages. |
| Target columns present | Configured targets exist and have non-null rows. | `column_missing`, `column_present_all_null`. |
| Skips explained | Skipped targets have explicit `skip_reason_code`. | Silent target omissions. |
| OOS predictions exist | Benchmark writes nonzero OOS rows/dates. | `oos_date_count` missing or zero. |
| Baseline signals exist | `momentum_120d` and/or configured baseline signals are present. | Policy sweep `baseline_signal_available: false`. |
| Experiment validation clean | `validation.errors` is empty. | Output-root, guardrail, OOS, parse, or winner errors. |
| Winners are feasible | Winners are not infeasible/error/unavailable and have nonzero dates. | Winner has `infeasible_reason` or zero date count. |

Correctness gate failure means reject the run or fix the pipeline before
discussing model quality.

## 2. Research Quality Gates

These gates answer: does the result look economically interesting enough for
deeper research?

| Gate | Desired evidence | Notes |
|---|---|---|
| ML beats `momentum_120d` on ranking | Best ML model exceeds OOS momentum on mean Spearman IC and top-minus-bottom spread. | Implemented as `ml_beats_momentum_120d` in benchmark. |
| ML beats momentum after costs | Portfolio replay/sweep `beats_momentum_120d` true after configured costs. | Ranking wins do not guarantee portfolio wins. |
| Positive net return | Best candidate net return after cost drag is positive. | Gross return alone is insufficient. |
| Positive Sharpe | Sharpe is positive and not driven by tiny sample size. | Inspect OOS date count. |
| Acceptable max drawdown | Drawdown is within manual risk tolerance. | No universal threshold is documented in code for stock-alpha promotion. |
| Reasonable turnover | Average turnover and 95th percentile turnover are not excessive. | High turnover raises execution sensitivity. |
| Reasonable concentration | Position weights and concentration percentiles are controlled. | Avoid one-symbol dependence. |
| Not one-period dependent | Performance is not dominated by one year/month/symbol. | Inspect performance by year and holdings. |
| Robust targets | Target comparison does not show success only on one fragile target. | Skipped targets weaken evidence. |
| Robust enriched comparison | Enriched features improve or at least do not obviously degrade against baseline. | If enriched loses, features may be noise. |

Quality gate failure does not always mean the pipeline is broken. It often
means the candidate is not worth promotion.

## 3. Candidate Triage Gates

Use these states for research review:

| State | Meaning | Action |
|---|---|---|
| Red | Correctness failure or clearly poor quality. | Reject, fix data/path/config, or rerun. |
| Yellow | Mechanically valid and promising, but incomplete or fragile. | Run deeper diagnostics, benchmark/full, attribution, or manual review. |
| Green | Mechanically valid, robust across ranking/portfolio/targets, and worth deeper benchmark/full validation. | Continue research review; still not trading approval. |

Examples:

- Red: output-root validation error, missing OOS predictions, infeasible winner,
  `promotion_thresholds_changed: true`.
- Yellow: ML beats momentum on ranking but loses after costs; skipped targets
  are explained but important.
- Green: validation clean, sufficient OOS dates, ML beats momentum on ranking
  and after costs, drawdown/turnover/concentration acceptable, robust targets.

## 4. Paper/Live Trading Gates

Stock-alpha research does not trade. Paper/live trading requires separate
explicit approval and must remain in execution-adjacent modules.

Required separation:

- No broker/paper/live/order imports in research-only stock-alpha modules.
- No order placement from metrics, reports, replay, or policy sweep.
- No inference that `production_validated: false` can drive execution.
- No silent promotion-threshold changes.
- No use of generated reports as automatic order instructions.

Paper/live review must happen through the separate paper/live workflows and
approval mechanisms. Research reports can inform a human decision, but they are
not approval artifacts.

## When Do We NOT Trade?

Do not trade when any of these are true:

- Result is dev-only.
- Benchmark is incomplete.
- Full/benchmark has missing required artifacts.
- Baseline comparison is missing.
- ML loses to momentum after costs.
- Drawdown is high or not manually acceptable.
- Turnover is high enough that cost assumptions look fragile.
- Concentration depends on one stock, sector, year, or month.
- Output-root validation errors exist.
- Legacy/stale outputs were mixed into the run.
- Targets were skipped without a clear reason.
- Target columns are missing or mostly null.
- OOS date count is zero or too small for confidence.
- A winner is infeasible, unavailable, errored, or has zero dates.
- `promotion_thresholds_changed: true`.
- `production_validated: false`.
- The result has not been reviewed manually.

The last two are intentionally strict: current stock-alpha research reports
state `production_validated: false`, so they must not be used to drive
execution.

