# Ticket 1D-A deferred integration map

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

No integration described here is part of Ticket 1D-A.

| Integration point | Required input artifact | Population gate | Search-count source | Block policy source | Final-holdout restriction | Later output | Why deferred |
|---|---|---|---|---|---|---|---|
| `core/research/ml/selector_evaluation.py` | Completed strict-OOS evaluation partitions with ordered date-level outcomes | Exact date/policy/model population checksum | Authoritative experiment-ledger campaign summary | Pre-registered evaluation configuration | Holdout access recorded and excluded from model selection | Per-comparison safeguard envelope | Live evaluator is excluded while publication is active |
| Five-date operational aggregate | Five verified component/evaluation owners | Same five resolved dates and matched row population | Full material trial count for the operational campaign | Explicit operational block policy; likely descriptive-only due tiny sample | Never treated as final inferential evidence | `operational_statistical_safeguards.json` with insufficient-data states where appropriate | Components do not yet exist |
| 46-date multi-regime aggregate | Atomic 46-date evaluation aggregate | Exact 46-date panel checksum across every candidate | Campaign-level ledger materialization including failed/rejected trials | Pre-specified overlap-aware block length | Protected audit dates excluded from selection | Block bootstrap, DSR, PBO, SPA, MCS and seed-dispersion section | Authoritative panel has not run |
| Experiment-ledger search summary | General ledger events plus future campaign/hypothesis materialization | Trial identity and campaign membership validation | Ledger itself, including failed/rejected/invalidated material trials | Not applicable | Final-holdout accesses counted separately | Effective-search-count summary by campaign/hypothesis | Ticket 0.2 extension is not part of this ticket |
| Selector promotion report | Verified evaluation aggregate and safeguard envelope | Promotion candidates share exact dates and benchmark | Signed campaign search-count artifact | Evaluation contract block policy identity | Repeated holdout use blocks promotion | Statistical-evidence gate and blocking reasons | Promotion must wait for real strict-OOS history |
| Portfolio-policy comparison | Matched policy-period returns after identical cost/capacity assumptions | Exact period, benchmark and policy population | Policy-search campaign count | Portfolio horizon/overlap block policy | Final policy audit cannot select among policies repeatedly | Dependency-aware policy comparison and MCS | No authoritative policy comparison exists yet |

Before any integration, the consuming owner must preserve the common result envelope,
population checksum, parameter checksum, block-policy identity, and effective-search
count provenance. It must not reinterpret a blocked result as numeric evidence.

