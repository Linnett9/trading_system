# Model and Gate Explainer

This document explains the stock-alpha model families, model names, and
research gates used to interpret a run. It does not claim any model has won a
benchmark, and it does not change promotion gates.

Primary modules and reports to inspect:

| Topic | Inspect |
| --- | --- |
| Model names and feature columns | `core/research/ml/stock_level_benchmark_types.py` |
| Tabular and sequence factories | `core/research/ml/stock_level_benchmark_models.py` |
| Ranking benchmark | `core/research/ml/stock_level/stock_level_model_ranking_benchmark.py` |
| Ranking metrics | `core/research/framework/ranking.py` |
| Target comparison | `core/research/ml/stock_level/stock_level_target_comparison.py` |
| Portfolio replay | `core/research/ml/stock_level/stock_level_portfolio_replay.py` |
| Policy sweep | `core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py` |
| Experiment validation | `core/research/ml/stock_level/stock_alpha_experiment_report.py` |
| Parallelism audit | `core/research/ml/stock_level/stock_alpha_parallelism_audit.py` |

## Non-Negotiable Interpretation Rules

- Green means worth deeper validation.
- Green does not mean trade.
- Portfolio replay and portfolio policy sweep are simulations.
- `production_validated: false` means no paper/live authorization.
- `promotion_thresholds_changed: false` must remain false.
- A model win is only meaningful if the same run also passes output-root,
  freshness, guardrail, OOS, baseline, and feasibility checks.

## Model Family Map

| Family | Receives | Predicts | Why it exists | Strengths | Weaknesses | Runtime cost | Overfitting risk | How to interpret a win/loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Simple baselines | Existing baseline columns such as `predicted_momentum_120d` and `predicted_risk_adjusted_momentum`. | No fitted model; the signal itself ranks stocks. | Provides a strong, transparent floor. | Cheap, interpretable, hard to overfit. | Limited feature interaction and regime awareness. | Very low | Low | ML should beat these on aligned OOS dates before it is interesting. |
| Linear tabular models | Base/enriched tabular feature columns. | Forward return target, default `actual_forward_return_10d`. | Tests whether a stable linear combination of features improves ranking. | Interpretable, relatively robust, fast. | Misses nonlinear interactions. | Low | Low to medium | A win suggests broad feature signal; a loss means simple baselines may already capture most useful information. |
| Tree models | Tabular feature columns with imputation. | Forward return target. | Tests nonlinear thresholds and interactions. | Handles nonlinear feature effects. | Can fit noise if samples are thin. | Medium | Medium to high | A win needs confirmation across targets and replay because trees can exploit unstable splits. |
| Sequence/deep models | Time-ordered rows or sequence windows built by the stock-level sequence adapter. | Forward return target. | Tests whether temporal context helps beyond static rows. | Can model sequence shape and regime transitions. | Higher dependency/runtime cost and more parameters. | Medium to high | High | A win is interesting only with stable OOS counts and no unavailable-model shortcuts. |
| Market/context models | Stock features plus context columns such as market breadth and SPY volatility/drawdown. | Forward return target. | Tests whether market environment changes stock ranking behavior. | Can adapt rankings to regimes. | Can overfit macro context or noisy regime proxies. | Medium to high | High | A win should be checked against target comparison and replay under costs. |
| News models | Stock features plus point-in-time news or sentiment columns if present. | Forward return target. | Tests whether symbol-level text/sentiment context adds signal. | Could capture event information unavailable in prices. | Unavailable when no valid news columns exist; synthetic news is forbidden. | High | High | If unavailable, that is expected unless input contains point-in-time news/sentiment coverage. |

## Stock-Alpha Models

The ranking benchmark writes `stock_level_model_ranking_benchmark.{csv,json,md}`
and `stock_level_model_oos_predictions.csv`. The model leaderboard ranks by
descending mean Spearman IC, then descending top-minus-bottom spread.

| Model | Receives | Predicts | Why it exists | Expected strengths | Expected weaknesses | Runtime cost | Overfitting risk | How to interpret win/loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `momentum_120d` | `predicted_momentum_120d`. | Not fitted; ranks by 120-day momentum. | Main simple momentum baseline. | Transparent and hard to fool. | Slow to adapt after trend breaks. | Very low | Low | ML wins should beat this on both mean Spearman IC and spread; losses mean baseline remains competitive. |
| `risk_adjusted_momentum` | `predicted_risk_adjusted_momentum`. | Not fitted; ranks by risk-adjusted momentum. | Baseline for momentum per unit of recent risk. | Useful when raw winners are too volatile. | Can overfavor low-volatility names with weak absolute returns. | Very low | Low | A win over raw momentum but not this baseline suggests risk normalization matters. |
| `ridge` | Median-imputed, scaled tabular features. | Forward return target. | Tests regularized linear relationships. | Stable and fast with correlated features. | Underfits nonlinear effects. | Low | Low | A ridge win is a useful sign of broad linear signal; a loss suggests features are weak or nonlinear. |
| `elastic_net` | Median-imputed, scaled tabular features. | Forward return target. | Tests sparse-ish linear combinations with L1/L2 regularization. | Can downweight unhelpful features. | Sensitive to alpha/l1 settings and scaling. | Low | Low to medium | A win suggests only a subset of features matters; a loss can mean signal is not sparse or not linear. |
| `random_forest` | Median-imputed tabular features. | Forward return target. | Tests bagged nonlinear interactions. | Captures thresholds and feature interactions. | Can average away weak ranking signal or overfit small samples. | Medium | Medium | A win should be checked for stable OOS date count and replay robustness. |
| `gradient_boosting` | Median-imputed tabular features. | Forward return target. | Tests boosted shallow nonlinear rules. | Strong on tabular nonlinear patterns. | Can fit noisy target quirks despite conservative depth. | Medium | Medium to high | A win is promising but needs target and policy confirmation; a loss can mean weak nonlinear structure. |
| `dlinear` | Sequence-regressor input from stock-level features. | Forward return target. | Tests a simple deep linear sequence baseline. | Lower complexity than transformer variants. | May underfit complex context. | Medium | Medium | A win suggests sequence shape matters without heavy attention models. |
| `patchtst` | Sequence windows of stock-level features. | Forward return target. | Tests patch-based time-series representation. | Can capture local temporal patterns. | Needs enough sequence coverage and can be costly. | High | High | A win should be checked for sufficient sequence count and repeatability. |
| `transformer` | Sequence windows of stock-level features. | Forward return target. | Generic attention-based temporal model. | Flexible temporal interactions. | Parameter-heavy for small stock-alpha samples. | High | High | Treat wins cautiously unless robust across targets and replay. |
| `itransformer` | Sequence windows adapted to inverted transformer architecture. | Forward return target. | Tests a different attention orientation for multivariate time series. | May handle feature-wise relationships differently. | Same data hunger and instability risks as transformers. | High | High | A win is a research lead, not promotion evidence. |
| `momentum_transformer` | Sequence windows focused on momentum-like feature behavior. | Forward return target. | Tests whether momentum history benefits from attention. | May exploit temporal momentum shape. | Can rediscover baseline momentum with more complexity. | High | High | Compare against `momentum_120d` and `risk_adjusted_momentum` first. |
| `multitask_transformer` | Sequence features and configured auxiliary target behavior where supported. | Forward return target for stock-alpha ranking. | Tests whether related objectives improve representation. | Can share signal across related outcomes. | Auxiliary tasks can distract from ranking target. | High | High | A win should show ranking and portfolio evidence, not only training fit. |
| `market_context_encoder` | Stock features plus context columns from `CONTEXT_COLUMNS`. | Forward return target. | Tests market-regime-aware stock ranking. | Can adapt to breadth/volatility/drawdown context. | Regime proxies can be noisy and overfit. | High | High | A win should be inspected with context coverage and target comparison. |
| `news_analysis_transformer` | Stock features plus point-in-time news/sentiment columns if available. | Forward return target. | Tests news/sentiment signal. | Could capture event-driven information. | Unavailable when no valid news/sentiment columns exist; synthetic inputs are forbidden. | High | High | If listed in `unavailable_models`, do not treat it as a failed model; treat it as missing valid input coverage. |
| `temporal_fusion_transformer` | Stock features plus market context columns. | Forward return target. | Tests a richer temporal/context architecture. | Can combine temporal and context information. | Most complex and easiest to overfit in thin samples. | High | High | A win is only a candidate for deeper validation after replay, sweep, and guardrails pass. |

## Model Report Fields to Inspect

| Field | Where | Why it matters |
| --- | --- | --- |
| `leaderboard` | Benchmark JSON | Shows IC, spread, hit rate, spread Sharpe, and rank. |
| `best_ml_model` | Benchmark JSON | Identifies the best ML model by benchmark ranking rule. |
| `best_ml_vs_momentum_120d` | Benchmark, replay, sweep JSON | Shows whether ML beats aligned momentum baseline. |
| `ml_beats_momentum_120d` | Benchmark JSON | Requires ML to exceed momentum on mean Spearman IC and top-minus-bottom spread. |
| `completed_models` | Benchmark JSON | Shows which requested models actually ran. |
| `unavailable_models` | Benchmark JSON | Explains skipped or errored models, including missing news/sentiment coverage. |
| `oos_date_count`, `oos_row_count`, `oos_symbol_count` | Benchmark JSON | Confirms the size of the out-of-sample evaluation. |
| `walk_forward` | Benchmark JSON | Confirms chronological expanding-window split, embargo, and OOS-only setting. |
| `parallelism` | Benchmark JSON and parallelism audit | Confirms effective workers and nested thread caps. |
| `feature_columns` | Benchmark JSON | Confirms whether baseline or engineered feature set was used. |

## Stock-Alpha Metrics

Metrics are evidence for research triage. They are not trading instructions,
and they should be read together rather than cherry-picked.

### Ranking Metrics

These appear primarily in `stock_level_model_ranking_benchmark.{csv,json,md}`
and target comparison outputs.

| Metric | What it measures | Better direction | Where to inspect | Common trap |
| --- | --- | --- | --- | --- |
| `mean_pearson_ic` | Average linear correlation between signal values and future target returns across OOS dates. | Higher positive is better. | Benchmark `leaderboard`. | Can look good because of outliers even when rank ordering is weak. |
| `mean_spearman_ic` | Average rank correlation between signal ranks and future target ranks across OOS dates. | Higher positive is better. | Benchmark `leaderboard`; primary ranking sort. | Positive IC can still fail to create usable top/bottom portfolio spread. |
| `top_decile_return` | Average future target return of the highest-ranked bucket. | Higher is better. | Benchmark `leaderboard`. | Can be positive during broad market rallies even if stock selection is weak. |
| `bottom_decile_return` | Average future target return of the lowest-ranked bucket. | Lower is better for long-short separation. | Benchmark `leaderboard`. | For long-only use, this matters less than top bucket quality. |
| `top_minus_bottom_spread` | Difference between top-decile and bottom-decile future returns. | Higher positive is better. | Benchmark `leaderboard`; ML-vs-momentum comparison. | Strong spread can be hard to monetize if driven by unshortable or illiquid names. |
| `top_decile_hit_rate` | Fraction of OOS dates where the top bucket is positive or beats the relevant comparison used by the evaluator. | Higher is better. | Benchmark and target comparison summaries. | High hit rate with tiny returns can still be economically weak. |
| `risk_adjusted_spread` | Spread adjusted by a risk or variability denominator in the ranking evaluator. | Higher is better. | Benchmark `leaderboard`. | Depends on enough OOS dates to be stable. |
| `spread_sharpe` | Sharpe-like score for the top-minus-bottom spread series. | Higher is better. | Benchmark `leaderboard`. | Can be unstable with short OOS history. |
| `rank` | Leaderboard rank after sorting by mean Spearman IC and then spread. | Lower rank number is better. | Benchmark CSV/JSON. | Rank is relative to completed models only; check `unavailable_models`. |
| `ml_beats_momentum_120d` | Boolean requiring best ML to beat aligned `momentum_120d` on mean Spearman IC and top-minus-bottom spread. | `true` is better. | Benchmark JSON. | A single `true` is still not a trading gate. |

### Portfolio Simulation Metrics

These appear in `stock_level_portfolio_replay_summary.*` and
`stock_level_portfolio_policy_sweep.*`. They are simulation metrics over OOS
predictions.

| Metric | What it measures | Better direction | Where to inspect | Common trap |
| --- | --- | --- | --- | --- |
| `gross_return` | Sum of simulated returns before transaction cost and slippage drag. | Higher is better. | Replay/sweep summary. | Can disappear after costs. |
| `transaction_cost_drag` | Return lost to turnover, configured cost bps, slippage bps, and in sweep short borrow drag where applicable. | Lower is better. | Replay/sweep summary. | Low drag can come from too little trading rather than good signal. |
| `net_return` | Simulated return after cost/slippage assumptions. | Higher is better. | Replay/sweep winners and summary. | Still simulated; not paper/live performance. |
| `total_return` | Equity-curve total return over the replayed OOS periods. | Higher is better. | Replay/sweep summary. | Can hide severe drawdowns. |
| `annualized_return` | Total return scaled by observed rebalance-date spacing. | Higher is better. | Replay summary. | Fragile when date count is small or gaps are irregular. |
| `volatility` | Standard deviation of simulated period net returns. | Context-dependent. | Replay summary. | Lower volatility can also mean low opportunity. |
| `sharpe` | Mean period return divided by period volatility and annualized. | Higher is better. | Replay/sweep winners. | Unstable with few periods or near-zero volatility. |
| `max_drawdown` | Worst simulated equity drawdown. Values are usually negative. | Closer to zero is better. | Replay/sweep winners. | Strong return with deep drawdown may be unacceptable. |
| `calmar_ratio` | Annualized return divided by absolute max drawdown. | Higher is better. | Replay/sweep winners. | Undefined or misleading when drawdown is near zero. |
| `hit_rate` | Fraction of simulated periods with positive net return. | Higher is better. | Replay summary. | Many small wins can still lose to rare large losses. |
| `average_turnover` | Average absolute portfolio weight change per rebalance. | Lower is usually better. | Replay/sweep summary. | Too-low turnover can mean stale positions rather than robust alpha. |
| `average_number_of_positions` | Average holdings count per rebalance. | Context-dependent. | Replay summary. | Very low count increases concentration risk. |
| `max_position_weight` | Largest absolute simulated position weight. | Lower is usually safer. | Replay summary and holdings. | Caps can make a policy infeasible or leave cash unused. |
| `date_count` | Number of OOS rebalance dates simulated. | Higher is more reliable. | Replay/sweep summary. | Small date count makes all performance metrics fragile. |
| `symbol_count` | Number of symbols used in simulated holdings. | Higher usually improves breadth. | Replay summary. | High symbol count does not guarantee diversification if weights concentrate. |

### Validation And Run Metrics

These appear in experiment reports, parallelism audits, and overnight summaries.

| Metric or field | What it measures | Better direction | Where to inspect | Common trap |
| --- | --- | --- | --- | --- |
| `validation_passed` | Whether experiment report validation found no errors. | `true` is required before interpretation. | `stock_alpha_experiment_report.json`. | Warnings may still matter even when validation passes. |
| `checked_artifact_count` | Number of artifacts loaded and validated. | Should match expected report level. | Experiment report `validation`. | A partial dev report can check fewer artifacts by design. |
| `missing_artifact_count` | Required or candidate artifacts not found. | Zero is best for complete runs. | Experiment report `validation`. | Missing optional files may be warnings depending on report level. |
| `output_root_validation_passed` | Whether outputs live under the configured canonical root. | `true` is required when legacy paths are disabled. | Experiment report `validation`. | Legacy paths can hide output propagation bugs. |
| `effective_row_count`, `effective_date_count`, `effective_symbol_count` | Profile-filtered coverage after dev/benchmark/full caps. | More coverage is usually more stable. | Stage JSON payloads. | Counts can differ legitimately across artifacts and OOS predictions. |
| `requested_workers`, `effective_workers` | Requested and usable workers for a stage. | For benchmark outer stages, expected effective workers are normally 4 where enough work exists. | Parallelism audit and stage JSON. | Nested workers should stay capped to avoid oversubscription. |
| `elapsed_seconds` | Stage runtime. | Context-dependent. | Stage JSON and overnight summary. | Fast can mean skipped work; slow can mean over-parallelism or data issues. |
| `promotion_thresholds_changed` | Whether promotion thresholds changed. | Must be `false`. | Every research report JSON. | Any `true` is a stop sign, not a warning to ignore. |
| `production_validated` | Whether output claims production approval. | Must be `false` for current stock-alpha research reports. | Every research report JSON. | `false` means no paper/live authorization. |

## Gate Types

### Research Correctness Gates

Correctness gates ask whether the run is internally coherent before judging
performance.

| Gate | Protects against | Pass example | Fail example | Inspect | What to do when it fails |
| --- | --- | --- | --- | --- | --- |
| Canonical output root | Mixed or legacy paths. | Every artifact path is under the active `stock_alpha/{run_size}/` directory. | Artifact comes from `reports/ml/benchmark/ml/` while legacy paths are disabled. | `validation.output_root_validation_passed`, `unexpected_output_paths`, `legacy_output_paths_detected` | Fix path propagation and rerun the affected stage. |
| Required files | Partial runs being interpreted as complete. | Required JSON/CSV/Markdown outputs exist. | Missing benchmark, replay, sweep, or overnight summary. | `missing_artifact_count`, `validation.errors` | Re-run only the missing focused stage when appropriate. |
| Freshness | Stale mixed outputs. | File mtimes are within the configured max age spread. | Old replay paired with new benchmark. | `stale_mixed_outputs` errors | Regenerate stale stages under the same output root. |
| Guardrails | Accidental promotion or gate loosening. | `research_only: true`, `trading_impact: none`, `production_validated: false`, `promotion_thresholds_changed: false`. | Any field missing or changed. | Every report JSON and experiment report validation | Stop and inspect producing module/config before interpreting results. |
| Run size | Dev/benchmark/full mixing. | Report `run_size` matches expected run size. | Benchmark artifact claims `dev` during benchmark summary. | `run_size` validation errors | Clear the wrong stage input path and rerun canonical output. |
| OOS dates | In-sample or empty evaluation. | `oos_date_count` is positive. | Zero OOS dates or missing walk-forward metadata. | Benchmark/replay/sweep JSON | Adjust split settings or data coverage; do not interpret metrics. |
| Winner feasibility | Infeasible policies being selected. | Winners have dates and no infeasible reason. | `winner_eligibility` or `winner_date_count` error. | Experiment report validation | Inspect policy constraints and run coverage. |

### Research Quality Gates

Quality gates ask whether coherent outputs show enough evidence to justify
more research.

| Gate | Protects against | Pass example | Fail example | Inspect | What to do when it fails |
| --- | --- | --- | --- | --- | --- |
| Beats momentum where relevant | Complex models that do not beat simple baselines. | `best_ml_vs_momentum_120d.beats_momentum_120d: true` on aligned OOS dates. | ML wins one metric but loses spread or replay net return. | Benchmark, replay, sweep JSON | Treat as yellow/red; investigate feature value before adding complexity. |
| Positive ranking signal | Noisy model leaderboard. | Positive mean Spearman IC and positive top-minus-bottom spread. | High hit rate but negative spread. | `leaderboard` | Check target coverage and alternative targets. |
| Target robustness | Single-label overfitting. | Multiple targets complete and show consistent best models/positive metrics. | Most targets skipped or only one target works. | Target comparison JSON | Improve label coverage or narrow claims. |
| Replay after costs | Ranking signal fails portfolio translation. | Positive net return after cost/slippage assumptions. | Gross return positive but transaction cost drag dominates. | Portfolio replay summary | Revisit turnover, policy, and cost assumptions. |
| Policy feasibility | Best policy cannot be implemented even in simulation. | Winners are completed and have enough positions. | `minimum_positions_not_met`, turnover cap exceeded, or missing baseline. | Policy sweep JSON | Change research assumptions or data coverage; do not promote. |
| Risk and concentration | Attractive return hides path risk. | Acceptable max drawdown, turnover, position concentration, and hit rate. | Extreme drawdown or concentrated holdings. | Replay/sweep summaries and holdings | Require deeper validation or constraints. |

### Candidate Triage Gates

Triage turns correctness and quality evidence into a research status.

| Status | Meaning | Example | Next action |
| --- | --- | --- | --- |
| Red | Do not trust the run or candidate. | Output-root error, changed guardrail, no OOS dates, infeasible winner. | Fix pipeline/data issue or reject candidate. |
| Yellow | Some evidence exists but is incomplete or fragile. | One target works, replay weak after costs, or model unavailable due to missing coverage. | Add focused validation before drawing conclusions. |
| Green | Worth deeper validation. | Correct outputs, intact guardrails, positive aligned metrics, feasible replay/sweep evidence. | Run deeper benchmark/full validation and manual review. |

Green is a research status only. It is not a trade signal, not a paper-trading
approval, and not production validation.

### Paper/Live Trading Gates

Paper/live gates are outside the stock-alpha research reports. Current
stock-alpha reports explicitly state `production_validated: false`, so they do
not authorize paper/live trading.

| Gate | Protects against | Pass example | Fail example | Inspect | What to do when it fails |
| --- | --- | --- | --- | --- | --- |
| Explicit production validation | Research output being treated as deployment approval. | A separate production validation process exists and explicitly approves. | Stock-alpha research report alone is used as approval. | Promotion checklist docs and paper/live services | Stop; research evidence is insufficient. |
| Promotion threshold integrity | Quietly weakened standards. | `promotion_thresholds_changed: false`. | `promotion_thresholds_changed: true`. | Report JSON guardrails | Stop and inspect config/module changes. |
| Broker/order isolation | Research code placing orders. | Research writes reports only. | Research imports broker or paper order modules to place/cancel/query orders. | Source imports, architecture tests | Remove coupling before any further validation. |
| Manual review | Blind automation from leaderboard to orders. | Human review of lineage, metrics, guardrails, and risk. | Automatic trade from `best_ml_model`. | Review checklist and reports | Require explicit review and separate execution workflow. |

## How to Read a Model Win

1. Confirm `validation_passed` is true in `stock_alpha_experiment_report.json`.
2. Confirm all report guardrails are intact.
3. Confirm the benchmark has positive OOS date and symbol counts.
4. Confirm the best ML model beats `momentum_120d` on aligned OOS ranking
   metrics, not just on one convenient metric.
5. Confirm target comparison does not show that the result only works on one
   fragile target.
6. Confirm replay and sweep results remain feasible after costs and constraints.
7. Treat the result as a candidate for deeper validation, not as a trade.

## How to Read a Model Loss

1. Check whether the model actually completed or appears in `unavailable_models`.
2. Check feature coverage and target availability before judging the algorithm.
3. Compare against both `momentum_120d` and `risk_adjusted_momentum`.
4. Look for OOS sample size problems before concluding the family is useless.
5. Prefer a simpler baseline if complex models fail to add OOS value.

## Common Failure Patterns

| Pattern | Likely cause | Inspect | Response |
| --- | --- | --- | --- |
| Strong leaderboard, weak replay | Turnover, costs, or concentration erase ranking edge. | Replay/sweep `transaction_cost_drag`, `average_turnover`, holdings. | Treat as yellow/red and revisit policy assumptions. |
| Good one-target result, weak alternatives | Target-specific overfit. | Target comparison `targets`, `skipped_targets`. | Require broader target robustness. |
| Sequence models unavailable or erroring | Missing dependencies, insufficient sequences, or missing news/context columns. | `unavailable_models`, `model_timings`. | Do not count unavailable models as evidence. |
| Policy sweep winner infeasible | Constraints too tight or data too sparse. | `infeasible_reason`, `baseline_coverage`. | Fix constraints or reject policy. |
| Effective workers wrong | Profile/cap mismatch. | `stock_alpha_parallelism_audit.json`. | Fix parallelism settings before running long benchmark. |
| Guardrail field changed | Report or config drift. | Report JSON and experiment validation errors. | Stop interpretation immediately. |
