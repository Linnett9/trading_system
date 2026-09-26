# DS24 PAPER provenance forensic audit

Audit ID: `DS24_PAPER_PROVENANCE_FORENSIC_20260926`

Repository HEAD audited: `afe9307cb1d5107f6ae343935a4f5e75232773c6`

Audit date: 2026-09-26

Scope: Huber, LightGBM RankXENDCG, and Elastic Net R40

## Executive conclusion

| Model | Terminal classification | Was progression toward PAPER technically justified? | Currently commissioned by inspected PAPER configuration? | Recommendation |
|---|---|---|---|---|
| Huber R37 replay | `PAPER_PROVENANCE_BLOCKED` | No | No | Keep inactive; rebuild causal features and perform a newly sealed evaluation. |
| LightGBM RankXENDCG / R47A | `PAPER_PROVENANCE_BLOCKED` | No, despite strong artifact provenance | No | Keep inactive; preserve R47A as lineage evidence, not as deployable evidence. |
| Elastic Net R40 | `PAPER_PROVENANCE_BLOCKED` | No | No | Keep inactive; complete the refresh only after repairing feature causality and resealing evaluation. |

The exact stored artifact chains are mostly intact. That does **not** make the results scientifically valid. All three models consume the same 101-predictor authority, and three predictors use the current session's final close at earlier decision timestamps: `overnight_gap`, `previous_session_return`, and `two_session_return`. A synthetic perturbation and a persisted AAPL row independently prove the leak. This is sufficient to block every model from PAPER.

No inspected active PAPER configuration names any of the three models. `configs/paper/alpaca_7day_trial.yaml` commissions `dual_momentum`; the DS24 order adapter is bound to an older P0_C0 `GradientBoostingRegressor` authority. Therefore the requested Huber safeguard passes narrowly: PAPER uses neither the quarantined `metrics_only_v3` namespace nor the R37 replay. The active `dual_momentum` strategy and the older P0_C0 incumbent were not scientifically re-audited here.

The complete machine-readable recomputation is in `evidence.json` (SHA-256 `fd96713e8a62b816f7220b50605586c6488093ccfb475baa0d2d2869f2351724`). It was produced without fitting a model, generating predictions, rebuilding data, placing orders, or modifying a canonical artifact.

## Decisive point-in-time failure

The shared predictor manifest contains 101 predictors. Its recorded manifest hash `d21722c41f0e54edc02519184a10596a2847091f0c04e80e4a0503251ffb60ca` and compact feature-order hash `22db0fcfe1219a30e0f6926dd8f0af11b4cd8ead2a6462a0df278549fadeb5a0` both recompute exactly. The problem is the implementation, not an untraceable schema.

`core/research/ml/ds24/master_5m_validation_stats.py` computes a per-session final close with `groupby(...).transform("last")` and shifts that row-expanded series. Consequently, the following early-session features depend on a later bar in the same session:

- `previous_session_return`: current session's final close divided by current session's opening price, then row-shifted;
- `overnight_gap`: current session open divided by the row-shifted current session final close;
- `two_session_return`: current decision close divided by a row-shifted session-final close.

Two independent checks establish causality failure:

1. In a synthetic session, changing only the 23:55 close changes all three values at the 14:35 decision row, while causal control `ret_5m` is unchanged.
2. In persisted AAPL 2024 data, the 2024-01-02 14:35 UTC feature row uses the 23:55 UTC close `183.23`. The persisted values match the future-close calculations: `overnight_gap=0.03356437385082245`, `previous_session_return=-0.03247439116239548`, and `two_session_return=0.013098292052745819`.

The prior `r6_pit_validation.json` reports no violations, but it validates recorded source timestamps rather than perturbing the feature implementation. It therefore did not detect this formula-level look-ahead.

## Answer matrix

`PASS` means the stated invariant was independently supported. `FAIL` means contrary evidence was found. `INCONCLUSIVE` is used where the requested invariant is not represented by that model's artifact.

| # | Audit question | Huber R37 | RankXENDCG R47A | Elastic Net R40 |
|---:|---|---|---|---|
| 1 | Exact artifact/model/policy identified | **PASS** — fresh `metrics_only_v3_r37_huber_replay`, Huber pipeline, policy hash `506101…e6ada` | **PASS** — R47A authority hash `09610d…e6f61`, stable artifact hash `c69017…c8b85` | **PASS** — `metrics_only_v3_r40_elastic_net`, ElasticNet adapter/pipeline, same comparable policy |
| 2 | Target contract and cadence | **PASS** — `forward_return_60m__decision_5m`; score each eligible 5-minute T, frozen for five sessions per refit | **PASS** — same target ID; one opening ranking per session | **PASS** — same target ID; score each eligible 5-minute T, frozen for five sessions per refit |
| 3 | Predictions genuinely out-of-sample | **FAIL** — fit/score rows are disjoint, but score-time predictors are not causal | **FAIL** — prequential fits are disjoint, but score-time predictors are not causal | **FAIL** — fit/score rows are disjoint, but score-time predictors are not causal |
| 4 | No future information enters predictor/preprocess/fit/rank/score | **FAIL** — three shared predictors use a later same-session close; no separate preprocessing leak found | **FAIL** — same feature-order hash includes all three; rank-label ordering itself is correct | **FAIL** — same three predictors leak; imputer/scalers are fitted on train rows only |
| 5 | Recorded training cutoff strictly precedes decision time | **FAIL** — 0 refits have a persisted cutoff strictly before first score; 2,613 comparable records equal first score | **PASS** — all 2,262 OOF records store a maximum training timestamp strictly before score time | **FAIL** — all 312 persisted cutoffs equal first score; none is strictly earlier |
| 6 | Target maturity/availability enforced PIT | **PASS** — worker filters `target_is_trainable` and `target_available_timestamp <= refit_T` | **PASS** — frozen engine applies the same availability condition before fitting | **PASS** — same worker guard |
| 7 | Holdout untouched during selection/evaluation evidence | **FAIL** — 24,216 scored rows are on/after the 2025-04-02 locked boundary | **PASS** — zero holdout rows; scores end 2024-12-31 | **FAIL** — all 24,216 scored rows are on/after the locked boundary |
| 8 | No result produced from an in-sample fit | **PASS** — source filters training decisions `< refit_T`; preprocessing is inside the training pipeline | **PASS** — source sorts the training rows, constructs within-time rank labels/groups, and scores a later timestamp | **PASS** — source filters training decisions `< refit_T`; train-only imputer/scalers/target transform |
| 9 | No duplicate-generation/stale-artifact contamination of selected authority | **PASS** — R37 explicitly reuses no quarantined metrics; 2,616 referenced model hashes and all log parts verify; duplicate keys are zero | **PASS** — 2,262 unique ordinals, no gaps/duplicates, every OOF part verifies. Limitation: source manifest says uncommitted Mac source over advertised HEAD, and the OOF manifest retains stale `PARTIAL_RESUMABLE`/`provisional` fields | **PASS** — 312 referenced model hashes and every evaluated log part verify; duplicate keys are zero |
| 10 | Current PAPER config points to intended model/policy/version/hash | **FAIL** — no binding | **FAIL** — no binding | **FAIL** — no binding |
| 11 | No PAPER pointer to stale/quarantined/provisional audited artifact | **PASS** — PAPER points to neither Huber namespace; cost-aware research authority points to R37, not quarantined `metrics_only_v3` | **PASS** — no PAPER pointer to R47A or its stale provisional OOF status | **PASS** — no PAPER pointer to provisional R40 |
| 12 | Overlapping 60-minute outcomes are not annualized as independent 5-minute returns | **PASS** — returns aggregate matured sleeve contributions by session and annualize daily returns | **PASS** — one decision per session; no intraday overlap multiplication | **PASS** — same daily aggregation method |
| 13 | Twelve-sleeve accounting independently verifies as capital constrained | **FAIL** — sleeve fraction is exactly `1/12` and simultaneous capital max is `1.0`, but the published daily log omits the 2026-06-30 ledger row | **INCONCLUSIVE** — R47A is a once-daily ranking artifact and has no twelve-sleeve ledger | **FAIL** — fractions/capital limit are correct, but the published daily log omits the 2026-06-30 ledger row |
| 14 | Transaction cost/slippage assumption identified | **PASS** — verified 0 bps and zero cost rows; current unrelated PAPER YAML uses 10 + 5 bps | **PASS** — verified 0 bps and gross equals net | **PASS** — verified 0 bps and zero cost rows; current unrelated PAPER YAML uses 10 + 5 bps |
| 15 | Headline stored metrics reproduce without retraining | **PASS with limitation** — IC and published-daily Sharpe/return reproduce exactly; complete sleeve ledger changes the return metrics slightly | **PASS** — retained OOF IC recomputes to `0.25064654` (the compact saved summary is `0.25064645`) | **PASS with limitation** — IC/NW and published-daily Sharpe/return reproduce exactly; complete sleeve ledger changes the return metrics slightly |

Questions 3 and 4 are intentionally stricter than question 8. The score rows were not fed back into the fit, but a chronologically separate score is still not a causal out-of-sample prediction when its predictor values use future prices.

## Current PAPER binding

Two repository paths were inspected:

```text
configs/paper/alpaca_7day_trial.yaml
  -> candidate source: dual_momentum
  -> Alpaca paper=true, submit_orders=true
  -> transaction_cost_bps=10, slippage_bps=5
  -> no Huber / XENDCG / Elastic Net identifier

core/paper/ds24_order_adapter.py
  -> config hash e9ed65f33adcad0d768077d6350737666ba5441a633afbdf6501952d2485f219
  -> training-policy hash 46426e0b4ff9e8948b089feb2d5192be0002a7e4f9a09479fa0b262d56344d21
  -> ds24_p8_r9.../01_r8_candidate_lock.json
  -> P0_C0, sklearn GradientBoostingRegressor
  -> canonical 101-feature model dataset
  -> forward_return_60m__decision_5m
```

This establishes absence of a current binding for the three audited candidates. It does not certify `dual_momentum` or the P0_C0 incumbent.

## Model lineages

### Huber

```text
canonical 5m features + 60m targets
  -> extended model-data authority (2016-2026; 101 predictors)
  -> comparable policy 506101…e6ada (20 train sessions, five score sessions)
  -> Huber train-only imputer + scaler + HuberRegressor
  +-> metrics_only_v3                         [QUARANTINED by R36]
  |     duplicate generations + pending-horizon fatal
  +-> metrics_only_v3_r36_replay              [prepared/intermediate]
  +-> metrics_only_v3_r37_huber_replay        [selected fresh namespace]
        -> 2,616 refits / 203,271 resolved rows
        -> status PROVISIONAL
        -> later cost-aware programme authority (not PAPER)
```

R37's lineage record states `selected_namespace_reuses_quarantined_metrics=false`. Every R37 manifest part and all 2,616 referenced model hashes match; duplicate primary keys and reported chronology violations are zero. PAPER uses neither branch. This satisfies the Huber-specific quarantine check, but R37 remains blocked by shared feature leakage, holdout consumption, zero-cost evaluation, equal cutoff/score metadata, and the missing terminal daily aggregation row.

### LightGBM RankXENDCG

```text
Mac frozen producer source 5c5e8f…6b17d
  -> sorted training rows: decision_timestamp, asset_id
  -> within-timestamp quintile rank labels + matching group sizes
  -> LightGBM objective=rank_xendcg
  -> causal train-row/maturity filters
  -> 2,262 model checkpoints + 2,262 OOF partitions (2016-01-05..2024-12-31)
  -> Dell import without recomputation
  -> R47A authority 09610d…e6f61 / stable artifact c69017…c8b85
  -> cost-aware score admission; executable exit evidence blocked
  -X no current PAPER binding
```

The rank construction has no discovered ordering or group-size error. All 2,262 OOF partitions match their recorded hashes and row counts: 1,150,792 asset-score rows, 514 assets, 2,262 distinct timestamps and ordinals, no missing or duplicate ordinal, and zero locked-holdout rows. Three retained model/OOF parity fixtures also match. The limitation is scientific: the feature order includes the same three future-dependent predictors. Operationally, the cost-aware programme also records an unpriceable held exit.

### Elastic Net R40

```text
canonical 5m features + 60m targets
  -> extended model-data authority (2016-2026; 101 predictors)
  -> comparable policy 506101…e6ada
  -> median imputer + feature StandardScaler
  -> target StandardScaler + ElasticNet(alpha=.001, l1_ratio=.25, random_state=23)
  -> metrics_only_v3_r40_elastic_net
     -> 312 refits / 24,216 decisions / 20,484 valid IC rows
     -> 2025-04-02..2026-06-30, status PROVISIONAL, 0 bps
  -> cost-aware scores admitted, economics quarantined/account depleted
  -X no current PAPER binding
```

R40's approximately 20,000 valid IC timestamps versus RankXENDCG's 2,262 are primarily explained by native 5-minute scoring versus one opening ranking per session. They are nevertheless apples-to-oranges as performance evidence: the date windows differ (R40 is 2025-2026; RankXENDCG is 2016-2024), R40 is entirely inside the locked-holdout era, the scoring densities differ, and the evaluation artifact shapes differ. Direct metric ranking between them is invalid.

## Reproduction and accounting

No model was retrained. The audit read every referenced Huber/Elastic Net rank-IC, sleeve, daily-return, refit, and cost partition; verified each part hash and row count; verified all referenced checkpoint hashes; and recomputed summaries. It also read and verified every RankXENDCG OOF partition.

| Metric | Huber R37 | Elastic Net R40 | RankXENDCG R47A |
|---|---:|---:|---:|
| Resolved/evaluation timestamps | 203,271 | 24,216 | 2,262 |
| Valid rank-IC timestamps | 171,863 | 20,484 | 2,259 in per-T artifact |
| Mean Spearman rank IC | 0.5023150229 | 0.4632185166 | 0.2506465446 recomputed |
| Positive-IC fraction | 0.9985919017 | 0.9953134153 | 0.9716688800 |
| Published annualized return | 0.9964717113 | 1.4983844449 | 1.7748838980, zero-cost daily arithmetic return |
| Published daily Sharpe | 9.1783974959 | 8.1603997202 | 13.2337501755 |
| Published mean turnover | 0.4861027889 | 0.4412041625 | Not represented equivalently |
| Transaction cost | 0 bps | 0 bps | 0 bps |

The Huber and Elastic Net mean IC, positive fraction, Newey-West interval, published daily Sharpe, annualized return, drawdown, turnover, and cumulative return match the saved summaries to numerical tolerance. The unusually high values are therefore reproducible properties of the stored evaluation files, not transcription errors. They must not be interpreted as credible edge because the underlying features leak future prices.

### Newey-West verification

The evaluator's lag-12 implementation was independently replicated with Bartlett weights, autocovariance denominator `n`, and standard error `sqrt(long_run_variance / n)`. A deterministic regression vector verifies the implementation. Elastic Net's 20,484 observations reproduce:

- mean `0.4632185165883415`;
- standard error `0.002592517960534033`;
- 95% interval `[0.4581371813856948, 0.4682998517909882]`;
- lag `12`.

Lag 12 is the repository's explicit convention for a 60-minute horizon on a 5-minute decision grid. A convention of 11 lags may also be defensible for eleven serial offsets, but there is no coding discrepancy between the evaluator, saved summary, and independent calculation. This inferential choice is not the source of the extreme IC.

### Sleeve verification and terminal-row defect

Both intraday models record `sleeve_count=12`, `sleeve_capital_fraction=1/12`, and maximum simultaneous capital fraction `1.0`. Reaggregation of every common maturity date matches the saved daily rows to floating-point tolerance. However, each published daily log omits a 2026-06-30 row present in the complete sleeve ledger:

| Model | Omitted matured decisions | Omitted net daily return | Saved daily rows | Complete-ledger daily rows | Saved vs complete-ledger Sharpe |
|---|---:|---:|---:|---:|---:|
| Huber | 8 | 0.007515409754857164 | 1,874 | 1,875 | 9.1783974959 vs 9.1845926438 |
| Elastic Net | 6 | 0.00469852729502412 | 234 | 235 | 8.1603997202 vs 8.1703886852 |

Thus the capital allocation formula is correct, and the evaluator avoids treating all overlapping 60-minute forward returns as independent full-capital returns, but the published daily aggregation is incomplete. This independently fails question 13.

## Exact primary evidence and hashes

Paths are repository-relative unless marked external.

| Evidence | Path | SHA-256 |
|---|---|---|
| Machine-readable audit | `docs/audits/ds24_paper_provenance_forensic_20260926/evidence.json` | `fd96713e8a62b816f7220b50605586c6488093ccfb475baa0d2d2869f2351724` |
| Predictor manifest | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json` | `b225048d0e2c593b66c4eedb696f5e47af16057456c16afd0acacb825fc2796a` |
| Feature implementation | `core/research/ml/ds24/master_5m_validation_stats.py` | `a29c1dfbf6e06425b35cd141a0cd164324493d92b9eaca4e68f8228066685e10` |
| Extended model-data authority | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/92_r7_r1_extended_model_data_authority.json` | `6597b5f379cc3f5213f06eadc9df9cdc174f139f355fcb0429562f63a96b77f8` |
| Extended data partition manifest | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/91_r7_r1_extended_model_data_partition_manifest.csv` | `661460bb297f45278a08b05afbc5ef888016445de74a16e091a8a9230488c7bc` |
| Target builder | `core/research/ml/five_minute_target_dataset.py` | `85c131ea8484da683d9d9274e54f1f3794704b9d4fe45811d6d90bc370a98757` |
| Comparable policy | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R7_R14_01_policy_authority.json` | `8a5cd1a5ff8e041b93171ff8ad3933860d11b6679812f6b2215b4715ec0f09d7` |
| Huber R36 containment | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R7_R36_04_huber_fatal_and_namespace_audit.json` | `a57c1fd4e3d1c6c7750ad2696448e2764325b2fd6de4effbab46ed9da049dd4e` |
| Huber R37 lineage | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R7_R37_03_namespace_lineage.json` | `849953e0c062643cad2534f938237d9b5941891a9a200e14b6691bf7a4ea6caf` |
| Huber R37 summary | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_summary_v3.json` | `6e2c3d8374453dede892b72a5d1b0d714db008b5961ab3a4f38ef79572f84604` |
| Huber R37 rank manifest | same namespace, `rank_ic_v3_manifest.json` | `0ff6f3a84cc4feeea0ae02b950fd56f90d53fc02d8e1dd237db42fb8992fd036` |
| Huber R37 sleeve manifest | same namespace, `sleeve_maturity_ledger_v3_manifest.json` | `38e245b68e73e0ea6a4d44ccb6c59f587028c347108c64aed04e58f83e000f6f` |
| R47A import authority file | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R47A_xendcg_import_authority.json` | `e2fdb969741605232f2f80f905219efdf070de191f6a22860a02aa0ef26d9db6` |
| R47A artifact validation | same stage root, `R47A_xendcg_artifact_validation.json` | `5313bc2e42c1cce47ca6ec2a00e827060fcc2feb879fd37809bb6d101e3e808c` |
| RankXENDCG OOF manifest | `mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/ensemble_oof_scores_manifest_v2.json` | `65a49db5d67c0e6b165a00c432e5f75c68a47f6fccfbbeb90d6bdfa1a82514f6` |
| RankXENDCG summary | same family root, `metrics_only_v3/resolved_performance_summary_v3.json` | `95d86023e1c50fe8b2599668d7d8f3313eff8ce178e16e2c425601296c8c1a7c` |
| Frozen Mac producer source (external retained authority) | `C:\Users\Brandon\Desktop\ds24-prospective-paper\authority\xendcg\ds24_xendcg_producer_authority_r1\core\research\ml\ds24\mac_aux_queue_r44f2.py` | `5c5e8ff1486d870f8c2d3b1ceb2c406f5544e38a34e5a768a816abc06076b17d` |
| Frozen Mac prequential engine (external) | same authority root, `core\research\ml\ds24\canonical_prequential_engine.py` | `1d4917f05345aad355984be943fd4c94be69a302f16162ac4fe904640e6a7fa9` |
| Elastic R40 summary | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_summary_v3.json` | `b31417b9b6165143cf6c5084f10d7b5a4e62fcf2922af47b95603cf060b7128d` |
| Elastic R40 rank manifest | same namespace, `rank_ic_v3_manifest.json` | `8596a2437138c616b50cfa48f8a028d379b6e88e6c50122db0e86e8a39b14f3d` |
| Elastic R40 sleeve manifest | same namespace, `sleeve_maturity_ledger_v3_manifest.json` | `ca4b7c79cd37f38ee2d48a607a2b9ff752226bad3d053291e3d97b15646eeff0` |
| Active PAPER YAML | `configs/paper/alpaca_7day_trial.yaml` | `f75bd5acf29a1e5f23a9c9fabdd99752340e148320c01831138f6822947bac29` |
| DS24 order adapter | `core/paper/ds24_order_adapter.py` | `f98476b9351a8301418a41d80803995fd28719d492672b889d71f7eb6521bfe2` |
| Bound P0_C0 authority | `docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r9_20260822T000000Z/01_r8_candidate_lock.json` | `71ee035829cfc27ee951dfab37b716251ba4c991ae4fa3a4563161acb30586e5` |
| Cost-aware programme state | `docs/dream_system/components/DS-24_independent_five_minute_selector/cost_aware_policy_experiment/cost_aware_policy_terminal_r4_20260917T141014Z/current_programme_state.json` | `1200bd2ee1439917c181ae6a9894afb67cbcadc34f51a4824a540ff81be326a0` |

`evidence.json` contains every remaining manifest, source, fixture, model, part-count, and digest result without expanding this report into thousands of file hashes.

## Residual limitations

- Metrics-only policy intentionally discarded unbounded full prediction histories for Huber and Elastic Net. The audit verified all retained metric, sleeve, cost, refit, daily, and checkpoint artifacts, but it cannot reconstruct every asset-level score from data without prohibited retraining.
- The exact RankXENDCG producer source is retained outside the repository and hashes correctly. Its source manifest classifies it as uncommitted worktree source over advertised Git HEAD `b2ffdf0966c0424c82632ea21f83cefb29905cc1`; preservation is adequate for audit, weaker than a clean committed producer revision.
- R47A validates the completed import, while the imported OOF manifest still says `provisional=true` and `terminal_completeness_state=PARTIAL_RESUMABLE`. Hash/count evidence supports completion, but the stale metadata should be corrected only in a new superseding authority, never by editing the imported artifact.
- No current PAPER binding was found for the three models. Absence is established for the active PAPER YAML and DS24 adapter inspected at the audited HEAD; external deployment state is outside repository evidence.

## Required remediation before any PAPER reconsideration

1. Correct the three session-close feature formulas and add row-level availability assertions plus future-bar perturbation tests to the canonical feature build.
2. Rebuild the affected feature/model datasets under new versioned authorities. Do not overwrite existing artifacts.
3. Refit and score under a newly preregistered development/confirmation split. The already-observed 2025-2026 holdout must not be relabelled as untouched holdout.
4. Persist unambiguous `max_training_decision_timestamp` and `max_training_target_available_timestamp` fields distinct from `refit_T`.
5. Repair terminal daily aggregation and verify exact sleeve-to-daily parity before publication.
6. Evaluate realistic nonzero transaction cost, slippage, capacity, and executable-price evidence.
7. Only after certification, bind PAPER through a single explicit candidate authority containing model hashes, feature/data/target hashes, policy hash, costs, and environment gates.
