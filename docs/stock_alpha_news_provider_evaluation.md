# Stock-Alpha News Provider Evaluation

Research-only planning document. Official issuer sources and third-party/media/API sources must remain separate providers and separate provenance paths.

## Recommended Priority

1. SEC EDGAR `sec_company_filings`
2. Official company RSS cleanup and feed-error remediation
3. GDELT dry-run evaluation
4. Polygon/Massive dry-run evaluation
5. Finnhub dry-run evaluation
6. Alpha Vantage dry-run evaluation

## SEC EDGAR

- Classification: official issuer/filing data.
- Access: free, no API key, official SEC endpoints.
- Rate limits: requires SEC-compliant user agent and bounded requests.
- Licensing/terms: official public filings; still respect SEC fair-access policies.
- Coverage benefits: broad historical issuer-event coverage, especially 8-K, 10-Q, and 10-K.
- History depth: usually many years through company submissions and archive URLs.
- Fields: symbol, CIK, form, accession, filing date, accepted datetime, report date, filing URL, primary document URL.
- Dedup strategy: provider plus accession number plus symbol.
- Symbol matching risks: CIK mappings must be verified; do not fabricate mappings.
- Timestamp reliability: accepted datetime is best; filing date is date-only and must be labeled as such.
- Separate provider: yes, `sec_company_filings`.
- Before transformer training: appropriate as official metadata after coverage/audit gates pass.

## GDELT

- Classification: third-party media, not official issuer data.
- Access: free, no API key for common endpoints.
- Rate limits: can be throttled; request volume must be bounded.
- Licensing/terms: must review current GDELT usage terms before scaling.
- Coverage benefits: broad media coverage and international breadth.
- History depth: substantial historical archive, but query semantics can be noisy.
- Fields: URL, domain/source, title, snippet, seen date, language.
- Dedup strategy: URL plus symbol plus published/seen timestamp.
- Symbol matching risks: ticker ambiguity and company-name disambiguation are material.
- Timestamp reliability: seen date is media-observation time, not issuer event time.
- Separate provider: yes.
- Before transformer training: only after strict relevance and duplicate audits.

## Polygon/Massive

- Classification: third-party/API market news.
- Access: API key; paid plan or plan limits may apply.
- Rate limits: plan-dependent.
- Licensing/terms: must verify redistribution and derived-data rights before persistence at scale.
- Coverage benefits: structured market-news metadata and ticker tagging.
- History depth: plan-dependent.
- Fields: article ID, URL, title, description, publisher, tickers, published timestamp.
- Dedup strategy: provider article ID plus ticker plus URL.
- Symbol matching risks: provider ticker tags may include multiple companies; attribution needs review.
- Timestamp reliability: generally strong if provider timestamp is UTC.
- Separate provider: yes.
- Before transformer training: useful after licensing and coverage checks.

## Finnhub

- Classification: third-party/API company news.
- Access: API key required; free tier exists but rate-limited.
- Rate limits: tier-dependent and easy to exhaust during universe collection.
- Licensing/terms: must review before storing and training at scale.
- Coverage benefits: convenient per-symbol company-news endpoint.
- History depth: endpoint/window limits may constrain backfills.
- Fields: ID, URL, source, headline, summary, datetime.
- Dedup strategy: provider ID plus symbol, fallback URL.
- Symbol matching risks: per-symbol endpoint helps, but source quality varies.
- Timestamp reliability: Unix timestamps are usually precise.
- Separate provider: yes.
- Before transformer training: only after API-key, terms, and coverage gates.

## Alpha Vantage

- Classification: third-party/API news and sentiment.
- Access: API key required; free tier heavily rate-limited.
- Rate limits: strict request limits on free tier.
- Licensing/terms: must review before persistent training use.
- Coverage benefits: combines news metadata with vendor sentiment.
- History depth: endpoint dependent.
- Fields: URL, title, summary, source, time published, ticker sentiment, overall sentiment.
- Dedup strategy: URL plus symbol plus published timestamp.
- Symbol matching risks: vendor relevance scoring must not be treated as official truth.
- Timestamp reliability: time-published field is useful but should be normalized and audited.
- Separate provider: yes.
- Before transformer training: metadata may be useful, but vendor sentiment fields must not be mixed into official RSS sentiment or treated as model labels.
