# DS24 Vast Reverse Nine-Family Queue R1

Queue ID: `DS24_VAST_REVERSE_NINE_FAMILY_R1`

Terminal classification: `DS24_VAST_REVERSE_NINE_FAMILY_QUEUE_READY_NOT_LAUNCHED_EXTERNAL_COORDINATION_TRANSPORT_PENDING`

This is a queue-only authority package. It does not rent Vast hardware, connect
to cloud storage, install dependencies, launch an executor, fit a model, score a
model, open a holdout, write full predictions, or place paper/live orders.

Vast runs the nine-family lane bottom-to-top so a future Mac queue can continue
top-to-bottom. The canonical family IDs are derived from
`core/research/ml/ds24/remote_family_queue.py` and the planned Vast order is:

1. `temporal_fusion_transformer`
2. `market_context_encoder`
3. `momentum_transformer`
4. `itransformer`
5. `transformer`
6. `patchtst`
7. `dlinear`
8. `lightgbm_lambdarank`
9. `lightgbm_rank_xendcg`

Dell and Mac ownership must be supplied later as files matching
`external_family_status.schema.json`. A mere queue membership row is not a live
claim. A compatible live claim blocks Vast admission; a compatible verified
completion is skipped as `SKIPPED_EXTERNAL_VERIFIED`; stale, missing,
malformed, contradictory, dead-PID or incompatible evidence fails closed.

Local Vast claims are deterministic JSON artifacts under `queue_state/claims/`.
They are atomic local writes and remain `RESERVED_LOCAL_ONLY_NOT_LAUNCHED`.
Expired claims require explicit recovery and are not silently replaced.

Dry-run next family from neutral fresh snapshots:
`temporal_fusion_transformer`.

Cross-host atomicity is not implemented here. The file/schema interface is ready
for a later transport ticket.

Future Vast launcher inputs:

1. A queue root containing `queue_state/queue_state.json`.
2. Fresh Dell and Mac snapshots that validate against
   `external_family_status.schema.json`.
3. A non-expired claim artifact matching `vast_family_claim.schema.json`.
4. A later-ticket executor adapter for the claimed canonical family ID.
5. Vast host identity and environment evidence supplied outside this package.
