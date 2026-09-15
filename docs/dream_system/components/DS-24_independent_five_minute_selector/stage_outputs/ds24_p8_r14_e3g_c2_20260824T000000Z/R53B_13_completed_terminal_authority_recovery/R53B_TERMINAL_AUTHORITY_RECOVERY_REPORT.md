# DS24 R53B Terminal Tournament Authority Recovery

Classification: `DS24_R53B_COMPLETED_TERMINAL_AUTHORITY_RECOVERY_PARTIAL_EXTERNAL_TRANSFER_REQUIRED`

- Completed lifecycle families: 13
- Open lifecycle families: 6
- R53/R53A outputs were not overwritten.
- Read-only audit: no model execution, resume, scoring, queue mutation, broker call, or holdout access.

## Completed Families

- ridge_policy_v1_control: CURRENT_DS24_TERMINAL_LEGACY | QUARANTINED_AFTER_COMPLETION | 2024-12-31T14:45:00+00:00 to 2026-06-30T19:00:00+00:00
- pca_ridge_policy_v1_control: CURRENT_DS24_TERMINAL_LEGACY | QUARANTINED_AFTER_COMPLETION | 2020-05-11T18:50:00+00:00 to 2026-06-30T19:00:00+00:00
- spline_additive_ridge: CURRENT_DS24_TERMINAL_LEGACY | QUARANTINED_AFTER_COMPLETION | 2025-04-02T14:20:00+00:00 to 2026-06-29T15:35:00+00:00
- elastic_net: CURRENT_DS24_TERMINAL_V3 | ACCEPTED_FINAL | 2025-04-02T13:35:00+00:00 to 2026-06-30T19:00:00+00:00
- rff_ridge: CURRENT_DS24_TERMINAL_V3 | ACCEPTED_FINAL | 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00
- huber: CURRENT_DS24_TERMINAL_V3 | ACCEPTED_FINAL | 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00
- mlp: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY | TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window
- random_forest: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY | TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window
- extra_trees: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY | TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window
- gradient_boosting: TERMINAL_AUTHORITY_NOT_LOCAL | TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window
- lightgbm_rank_xendcg: CURRENT_DS24_IMPORTED_MAC_TERMINAL | ACCEPTED_IMPORTED_FINAL | 2016-01-05T14:35:00+00:00 to 2024-12-31T20:00:00+00:00
- lightgbm_lambdarank: LAMBDARANK_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER | MAC_TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window
- DLinear: DLINEAR_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER | MAC_TERMINAL_AUTHORITY_NOT_LOCAL | no local terminal window to no local terminal window

## Open Families

- PatchTST: NOT_COMPLETED
- Transformer: NOT_COMPLETED
- iTransformer: NOT_COMPLETED
- Momentum Transformer: NOT_COMPLETED
- Market Context Encoder: NOT_COMPLETED
- Temporal Fusion Transformer: NOT_COMPLETED
