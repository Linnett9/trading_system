# Financial Sentiment Pipeline Design

Research-only design note. Do not implement runtime code in this step.

## Purpose

The financial sentiment pipeline is a proposed research-only data source for adding timestamp-disciplined news, filing, and commentary context to ML research. Its first use should be optional features for research reports, shadow overlays, and meta-ensemble experiments. It must not directly affect broker, paper trading, live trading, execution, or portfolio sizing.

Primary goals:

- Ingest financial text events with reliable event timestamps.
- Score optional sentiment features, potentially with FinBERT.
- Join sentiment to market/rebalance rows without lookahead.
- Track source reliability and coverage gaps.
- Make leakage risks visible in audits.

## Timestamp Discipline

Every text record must carry multiple timestamps when available:

- `published_at`: timestamp shown by the source as publication time.
- `first_seen_at`: first time the local pipeline observed the item.
- `retrieved_at`: time the pipeline downloaded or refreshed the item.
- `event_time`: canonical time used for joins.
- `source_updated_at`: later correction/update timestamp, if available.

Canonical rule:

- Use `event_time = max(published_at, first_seen_at)` when both are available.
- If `first_seen_at` is missing, mark the record as lower confidence.
- Never use `retrieved_at` as evidence that the event was knowable earlier.

Timezone rules:

- Store all timestamps in UTC.
- Preserve source timezone if available.
- Convert market/rebalance dates using the trading calendar.
- For daily bars, define whether sentiment after market close applies to the same trading date or the next trading date.

Join cutoff rule:

- A sentiment event can join to a feature row only if `event_time <= feature_timestamp`.
- For rebalance features generated before the open, only events known before that decision timestamp are eligible.
- For end-of-day research features, only events known before the close cutoff are eligible.

## Source Reliability

Each source should be scored and audited.

Recommended source fields:

- `source_name`
- `source_type`
- `publisher`
- `url`
- `canonical_id`
- `headline`
- `body_hash`
- `published_at`
- `first_seen_at`
- `retrieved_at`
- `event_time`
- `symbol_tags`
- `sector_tags`
- `reliability_tier`
- `dedupe_group_id`

Reliability tiers:

- Tier 1: direct company filings, exchange notices, official company press releases.
- Tier 2: established financial news providers with reliable timestamps.
- Tier 3: secondary news aggregators.
- Tier 4: social/media/commentary sources with uncertain timestamps.

Rules:

- Prefer primary sources over aggregators.
- Track whether an item is original reporting or a syndicated copy.
- Deduplicate syndicated stories and repeated headlines.
- Audit records with missing or suspicious timestamps.
- Avoid sources that revise timestamps without preserving update history.

## Leakage Prevention

Hard leakage rules:

- Do not join text published after the model feature timestamp.
- Do not use article revisions or summaries that include post-event information unless their update time is also before the feature timestamp.
- Do not use labels derived from future price moves as sentiment features.
- Do not use retrospective analyst summaries that describe outcomes after the forecast horizon.
- Do not use backfilled sentiment databases unless the vendor provides first-available timestamps.

Suspicious cases to flag:

- `published_at` is later than `retrieved_at`.
- `published_at` appears rounded or date-only.
- Multiple unrelated sources share identical body text with different timestamps.
- Source timestamp differs materially from first local observation.
- Article body changes after first retrieval.

Audit outputs:

- Count of records excluded for future timestamps.
- Count of records missing `first_seen_at`.
- Count of records with date-only timestamps.
- Count of records deduplicated.
- Count of records by reliability tier.
- Sentiment coverage by symbol/universe/date.

## Event-Time Joins

Recommended join keys:

- `symbol`
- `event_time`
- `feature_timestamp`
- optional `sector`
- optional `universe`

Join windows:

- Intraday unavailable baseline: aggregate events into rolling windows ending at feature timestamp.
- Daily research baseline:
  - `sentiment_1d`
  - `sentiment_3d`
  - `sentiment_5d`
  - `sentiment_10d`
  - `sentiment_21d`

Aggregation examples:

- Count of positive/neutral/negative events.
- Reliability-weighted mean sentiment.
- Recency-weighted sentiment.
- Maximum negative sentiment shock.
- Sentiment dispersion.
- News volume surprise relative to trailing baseline.
- Company-specific sentiment versus market/sector sentiment.

Event-time rule:

```text
include event if:
  event_time <= feature_timestamp
  and event_time > feature_timestamp - lookback_window
```

Never join by calendar date alone unless the decision timestamp is explicitly defined.

## Optional FinBERT Features

FinBERT can be used as an optional research sentiment scorer.

Candidate raw outputs:

- `finbert_positive_probability`
- `finbert_neutral_probability`
- `finbert_negative_probability`
- `finbert_sentiment_score`

Suggested score:

```text
finbert_sentiment_score = positive_probability - negative_probability
```

Aggregated features:

- `finbert_sentiment_mean_1d`
- `finbert_sentiment_mean_5d`
- `finbert_sentiment_mean_21d`
- `finbert_negative_shock_5d`
- `finbert_news_count_5d`
- `finbert_reliability_weighted_sentiment_10d`
- `finbert_sector_relative_sentiment_21d`

FinBERT constraints:

- Record model version and tokenizer version.
- Store input text hash, not necessarily full text, if licensing requires.
- Avoid scoring article text that was not available at `event_time`.
- Treat model output as research-only and uncalibrated until validated.

## Storage And Manifests

Recommended raw event manifest:

- `event_id`
- `source_name`
- `source_type`
- `url`
- `published_at`
- `first_seen_at`
- `retrieved_at`
- `event_time`
- `symbols`
- `text_hash`
- `dedupe_group_id`
- `reliability_tier`
- `ingestion_run_id`

Recommended scored sentiment table:

- `event_id`
- `event_time`
- `symbol`
- `source_name`
- `reliability_tier`
- `finbert_positive_probability`
- `finbert_neutral_probability`
- `finbert_negative_probability`
- `finbert_sentiment_score`
- `model_name`
- `model_version`
- `scored_at`

Recommended feature table:

- `feature_id`
- `feature_timestamp`
- `symbol` or `universe`
- rolling sentiment aggregates
- coverage counts
- exclusion counts
- `sentiment_feature_version`

## Integration With ML Research

Initial integration should be optional and disabled by default.

Possible config:

```yaml
ml:
  sentiment_features_enabled: false
  sentiment_feature_version: sentiment_v1
  sentiment_min_reliability_tier: 2
  sentiment_lookback_windows: [1, 5, 10, 21]
  sentiment_require_first_seen_timestamp: true
```

Use cases:

- Additional context features for `should_reduce_exposure`.
- Optional meta-ensemble v3 features.
- Diagnostics around negative news shocks and regime changes.
- Research-only HTML/report sections.

Do not make sentiment features mandatory for existing models.

## Validation Rules

Required checks:

- No event with `event_time > feature_timestamp` appears in model features.
- All joined events have non-null `event_time`.
- All timestamps are timezone-aware UTC.
- Deduplication happens before aggregation.
- Aggregations are recomputed per training fold if they involve normalization.
- Coverage gaps are reported by symbol and date.

Recommended tests:

- Event after feature timestamp is excluded.
- Event exactly at feature timestamp is included only if decision timestamp permits it.
- Date-only source timestamp is flagged.
- Duplicate article bodies collapse to one dedupe group.
- Revised article body after feature timestamp is excluded from that feature row.
- FinBERT score aggregation uses only eligible events.
- Reliability-tier filtering works.
- Feature generation is deterministic from the same manifest.

## Research-Only Boundary

Allowed:

- Offline ingestion experiments.
- Sentiment feature audits.
- Research model features.
- Shadow overlays.
- Meta-ensemble v3 optional features.

Not allowed:

- Broker changes.
- Paper trading behavior changes.
- Live trading behavior changes.
- Execution behavior changes.
- Direct order or portfolio sizing decisions from sentiment signals.

Sentiment features should be treated as noisy research context until they pass walk-forward validation, calibration checks, and promotion gates.
