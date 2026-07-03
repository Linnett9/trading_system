# Stock-Alpha Third-Party News Provider Evaluation

Research-only planning note. No third-party news collection is enabled by this document.

## Polygon/Massive Candidate

- Provider candidate: Polygon/Massive.
- Endpoint family: stock news.
- Expected fields: ticker, title, article URL, publisher, published timestamp, description, and sentiment or insights if available.

## Risks

- Licensing and plan limits may restrict storage, redistribution, or historical backfills.
- Duplicate stories may appear across publishers, syndication channels, or updated article URLs.
- Ticker association noise can attach a story to symbols that are only mentioned in passing.
- Publisher mix can introduce source bias.
- Lookahead risk is material if published timestamps are missing, delayed, revised, or normalized incorrectly.

## Separation Rule

Third-party news must remain separate from official SEC/RSS rows. No third-party news can contribute to `safe_for_feature_generation` until it has a separate provenance, duplicate, timestamp, relevance, and licensing audit.
