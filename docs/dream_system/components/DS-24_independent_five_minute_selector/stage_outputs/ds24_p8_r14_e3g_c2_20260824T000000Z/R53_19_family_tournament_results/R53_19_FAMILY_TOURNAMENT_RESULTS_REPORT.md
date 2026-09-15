# DS24 R53 19-Family Tournament Results Report

## Executive summary

- 19 requested families: 19 accounted for.
- Accepted final results: 3.
- Accepted imported final results: 1.
- Legacy-limited complete results: 3.
- Running provisional results: 7.
- Ready/not started: 3.
- Blocked: 1.
- Missing result authority: 1.

A high Rank IC does not prove live profitability. A high simulated Sharpe does not prove live tradability. Zero-cost economics are not genuine net-of-cost economics. Different evaluation periods are not directly comparable. Imported Mac results remain valid historical tournament evidence when accepted, but must be labelled as such. Provisional running results are not final. Legacy/quarantined evidence is not promotion authority.

## Current tournament state

- ridge_policy_v1_control: COMPLETE_LEGACY_RESULT_LIMITED (LEGACY_LIMITED)
- pca_ridge_policy_v1_control: COMPLETE_LEGACY_RESULT_LIMITED (LEGACY_LIMITED)
- spline_additive_ridge: COMPLETE_LEGACY_RESULT_LIMITED (LEGACY_LIMITED)
- elastic_net: COMPLETE_FINAL_RESULT_AVAILABLE (ACCEPTED_FINAL)
- rff_ridge: COMPLETE_FINAL_RESULT_AVAILABLE (ACCEPTED_FINAL)
- huber: COMPLETE_FINAL_RESULT_AVAILABLE (ACCEPTED_FINAL)
- mlp: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- random_forest: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- extra_trees: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- gradient_boosting: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- lightgbm_rank_xendcg: COMPLETE_IMPORTED_RESULT_AVAILABLE (ACCEPTED_IMPORTED_FINAL)
- lightgbm_lambdarank: EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE (MISSING)
- DLinear: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- PatchTST: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- Transformer: RUNNING_PROVISIONAL_RESULT (PROVISIONAL_RUNNING)
- iTransformer: READY_NOT_STARTED (NOT_STARTED)
- Momentum Transformer: READY_NOT_STARTED (NOT_STARTED)
- Market Context Encoder: READY_NOT_STARTED (NOT_STARTED)
- Temporal Fusion Transformer: CONFIGURATION_AUTHORITY_REQUIRED (BLOCKED)

## Completed models

### elastic_net
- Source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_summary_v3.json
- Rank IC: 0.4632185165883415; Sharpe: 8.160399720183758; annual return: 1.4983844449054737; max drawdown: -0.0009051097625262239.
- Evaluation window: 2025-04-02T13:35:00+00:00 to 2026-06-30T19:00:00+00:00.
### rff_ridge
- Source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/resolved_performance_summary_v3.json
- Rank IC: 0.03431086029850828; Sharpe: 1.720283881180856; annual return: 0.052542661888421924; max drawdown: -0.03629523046963268.
- Evaluation window: 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00.
### huber
- Source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_summary_v3.json
- Rank IC: 0.5023150228928017; Sharpe: 9.17839749588992; annual return: 0.9964717113372477; max drawdown: -0.010197421011264862.
- Evaluation window: 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00.
### lightgbm_rank_xendcg
- Source: mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json
- Rank IC: 0.2506464517395151; Sharpe: None; annual return: None; max drawdown: None.
- Evaluation window: None to None.

## Running models

- mlp: cursor 2024-05-15T20:00:00+00:00, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_summary_v3.json
- random_forest: cursor 2016-12-27T21:00:00+00:00, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_summary_v3.json
- extra_trees: cursor 2018-08-15T20:00:00+00:00, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_summary_v3.json
- gradient_boosting: cursor 2016-05-11T20:00:00+00:00, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/progress.json
- DLinear: cursor None, source None
- PatchTST: cursor None, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/resolved_performance_summary_v3.json
- Transformer: cursor None, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/resolved_performance_summary_v3.json

## Rank-IC leaderboard

- 1. huber: 0.5023150228928017 (DIRECTLY_COMPARABLE_V3_COMMON_WINDOW)
- 2. elastic_net: 0.4632185165883415 (V3_DIFFERENT_EVALUATION_WINDOWS)
- 3. lightgbm_rank_xendcg: 0.2506464517395151 (IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW)
- 4. rff_ridge: 0.03431086029850828 (DIRECTLY_COMPARABLE_V3_COMMON_WINDOW)

## Economic leaderboard

- 1. huber: Sharpe 9.17839749588992, annual return 0.9964717113372477, cost ZERO_COST_ECONOMICS
- 2. elastic_net: Sharpe 8.160399720183758, annual return 1.4983844449054737, cost ZERO_COST_ECONOMICS
- 3. rff_ridge: Sharpe 1.720283881180856, annual return 0.052542661888421924, cost ZERO_COST_ECONOMICS

## Mac vs Dell results

- lightgbm_rank_xendcg: owner COMPLETE_IMPORTED, state COMPLETE_IMPORTED_RESULT_AVAILABLE, source mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json
- lightgbm_lambdarank: owner MAC_OWNED, state EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE, source docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R51_lambdarank_import_result.json
- DLinear: owner MAC_OWNED, state RUNNING_PROVISIONAL_RESULT, source None

## Evaluation-window differences

- ridge_policy_v1_control: 2024-12-31T14:45:00+00:00 to 2026-06-30T19:00:00+00:00 (LEGACY_NOT_DIRECTLY_COMPARABLE)
- pca_ridge_policy_v1_control: 2020-05-11T18:50:00+00:00 to 2026-06-30T19:00:00+00:00 (LEGACY_NOT_DIRECTLY_COMPARABLE)
- spline_additive_ridge: 2025-04-02T14:20:00+00:00 to 2026-06-29T15:35:00+00:00 (LEGACY_NOT_DIRECTLY_COMPARABLE)
- elastic_net: 2025-04-02T13:35:00+00:00 to 2026-06-30T19:00:00+00:00 (V3_DIFFERENT_EVALUATION_WINDOWS)
- rff_ridge: 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00 (DIRECTLY_COMPARABLE_V3_COMMON_WINDOW)
- huber: 2016-02-02T14:35:00+00:00 to 2026-06-30T19:00:00+00:00 (DIRECTLY_COMPARABLE_V3_COMMON_WINDOW)
- mlp: 2016-02-02T14:35:00+00:00 to 2024-05-16T19:00:00+00:00 (PROVISIONAL_NOT_FINAL)
- random_forest: 2016-02-02T14:35:00+00:00 to 2016-12-27T20:00:00+00:00 (PROVISIONAL_NOT_FINAL)
- extra_trees: 2016-02-02T14:35:00+00:00 to 2018-08-15T19:00:00+00:00 (PROVISIONAL_NOT_FINAL)
- gradient_boosting: None to None (PROVISIONAL_NOT_FINAL)
- lightgbm_rank_xendcg: None to None (IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW)
- lightgbm_lambdarank: None to None (NO_RESULT)
- PatchTST: 2016-02-02T14:35:00+00:00 to 2019-07-23T18:00:00+00:00 (PROVISIONAL_NOT_FINAL)
- Transformer: 2016-02-02T14:35:00+00:00 to 2018-12-28T19:00:00+00:00 (PROVISIONAL_NOT_FINAL)

## Cost-model limitations

- ridge_policy_v1_control: ZERO_COST_ECONOMICS; turnover 0.2656175488793515
- pca_ridge_policy_v1_control: ZERO_COST_ECONOMICS; turnover 0.3313794401907871
- spline_additive_ridge: ZERO_COST_ECONOMICS; turnover 0.4034839924670435
- elastic_net: ZERO_COST_ECONOMICS; turnover 0.44120416253716555
- rff_ridge: ZERO_COST_ECONOMICS; turnover 0.9521087612104044
- huber: ZERO_COST_ECONOMICS; turnover 0.4861027888877411
- mlp: ZERO_COST_ECONOMICS; turnover 0.5390842877142347
- random_forest: ZERO_COST_ECONOMICS; turnover 0.5243866838825578
- extra_trees: ZERO_COST_ECONOMICS; turnover 0.4332546269316562
- gradient_boosting: COST_INFORMATION_ABSENT; turnover None
- lightgbm_rank_xendcg: COST_INFORMATION_ABSENT; turnover None
- lightgbm_lambdarank: COST_INFORMATION_ABSENT; turnover None
- PatchTST: ZERO_COST_ECONOMICS; turnover 0.5650986607220608
- Transformer: ZERO_COST_ECONOMICS; turnover 0.6111466489439036

## Results not directly comparable

- ridge_policy_v1_control: LEGACY_NOT_DIRECTLY_COMPARABLE
- pca_ridge_policy_v1_control: LEGACY_NOT_DIRECTLY_COMPARABLE
- spline_additive_ridge: LEGACY_NOT_DIRECTLY_COMPARABLE
- elastic_net: V3_DIFFERENT_EVALUATION_WINDOWS
- mlp: PROVISIONAL_NOT_FINAL
- random_forest: PROVISIONAL_NOT_FINAL
- extra_trees: PROVISIONAL_NOT_FINAL
- gradient_boosting: PROVISIONAL_NOT_FINAL
- lightgbm_rank_xendcg: IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW
- lightgbm_lambdarank: NO_RESULT
- DLinear: PROVISIONAL_NOT_FINAL
- PatchTST: PROVISIONAL_NOT_FINAL
- Transformer: PROVISIONAL_NOT_FINAL
- iTransformer: NO_RESULT
- Momentum Transformer: NO_RESULT
- Market Context Encoder: NO_RESULT
- Temporal Fusion Transformer: NO_RESULT

## Missing authorities

{
  "completed_family_with_missing_final_result": [],
  "configuration_blocker": [
    "Temporal Fusion Transformer"
  ],
  "generated_at_utc": "2026-09-15T19:10:10.542843+00:00",
  "legacy_result_with_no_accepted_v3_authority": [
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge"
  ],
  "mac_only_result_not_locally_present": [
    "lightgbm_lambdarank"
  ],
  "running_result_only": [
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "DLinear",
    "PatchTST",
    "Transformer"
  ],
  "unavailable_common_window_evidence": [
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "elastic_net",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer"
  ],
  "unavailable_cost_information": [
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank"
  ],
  "unavailable_evaluation_dates": [
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer"
  ]
}

## Key scientific cautions

- ridge_policy_v1_control: EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;LEGACY_RESULT
- pca_ridge_policy_v1_control: EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;LEGACY_RESULT
- spline_additive_ridge: ZERO_COST_WITH_NONZERO_TURNOVER;LEGACY_RESULT
- elastic_net: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;DIFFERENT_WINDOW_FROM_RANKING_FAMILIES
- rff_ridge: ZERO_COST_WITH_NONZERO_TURNOVER
- huber: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER
- mlp: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;PROVISIONAL_RESULT
- random_forest: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;PROVISIONAL_RESULT
- extra_trees: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;PROVISIONAL_RESULT
- gradient_boosting: PROVISIONAL_RESULT
- lightgbm_rank_xendcg: IMPORTED_RESULT
- lightgbm_lambdarank: IMPORTED_RESULT;MAC_AUTHORITY_NOT_LOCAL
- DLinear: PROVISIONAL_RESULT
- PatchTST: EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;PROVISIONAL_RESULT
- Transformer: EXTREME_RANK_IC_REQUIRES_SCRUTINY;EXTREME_SHARPE_REQUIRES_SCRUTINY;ZERO_COST_WITH_NONZERO_TURNOVER;PROVISIONAL_RESULT
- Temporal Fusion Transformer: CONFIGURATION_BLOCKED

## Exact source manifest

{
  "families": {
    "DLinear": {
      "all_sources_used": [],
      "alternatives_rejected": [],
      "selected": null
    },
    "Market Context Encoder": {
      "all_sources_used": [],
      "alternatives_rejected": [],
      "selected": null
    },
    "Momentum Transformer": {
      "all_sources_used": [],
      "alternatives_rejected": [],
      "selected": null
    },
    "PatchTST": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "f9f4563ebbdc23d3ef91b0d1b4133aea2bc55058f95e7902998ef8ccccedfab0",
          "mtime_utc": "2026-09-15T19:01:44.927228+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "b2aa8ee29af7747095d12de7b693abd1ae751241392957cef94f42b9f1b8f307",
          "mtime_utc": "2026-09-10T15:18:26.475149+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R52_patchtst_real_path_smoke/PatchTST/metrics_only_v3_r52_smoke/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "830b6ff27711157ef679e89b93751d0a6c0dcaf67c5ff38da8f2f2ffab569329",
          "mtime_utc": "2026-09-15T19:02:49.850039+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "9e02c214f73f705b92a0456d1adc1cd0b4b55c8f64fbac62f97fa46e00dbaa60",
          "mtime_utc": "2026-09-15T19:02:21.231995+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "4b3792c1eecdb8dc7c383458d532f55b0f7a8d685b51943ef5b7245418c14828",
          "mtime_utc": "2026-09-15T19:02:01.903616+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "93e6dbee9db159af2ad05e19c3c32e97690f77aa33a76f4fb12fb23e6cd063bf",
          "mtime_utc": "2026-09-15T19:01:34.519143+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "0b319c79612bd1c18978ed6b9c5f626f9dcc740345edf335d309fedb3398021b",
          "mtime_utc": "2026-09-15T19:01:02.486876+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "8aa88f579d0c8736474d90da5dda686cdc5eb81888c4506f820aba3a1446bdfa",
          "mtime_utc": "2026-09-15T19:01:02.358551+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "a63b921a6b7baa1ac4055349365c54089f1c033c230129d25c7708092df84e3f",
          "mtime_utc": "2026-09-10T15:18:26.616911+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R52_patchtst_real_path_smoke/PatchTST/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "45945427b222b221a7ba576a47d57f10e57d202d3b0dce165cf2da1fce480ed3",
          "mtime_utc": "2026-09-10T15:18:26.584277+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R52_patchtst_real_path_smoke/PatchTST/metrics_only_v3_r52_smoke/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "002f29b2586c6e00d4807fb69ff87071f2f3c37bfce99aeb1c556ed33602c045",
          "mtime_utc": "2026-09-10T15:18:26.496637+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R52_patchtst_real_path_smoke/PatchTST/metrics_only_v3_r52_smoke/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R52_patchtst_real_path_smoke/PatchTST/metrics_only_v3_r52_smoke/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_rejected": "stronger source tier or newer accepted authority selected"
        }
      ],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "f9f4563ebbdc23d3ef91b0d1b4133aea2bc55058f95e7902998ef8ccccedfab0",
        "mtime_utc": "2026-09-15T19:01:44.927228+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/PatchTST/metrics_only_v3_r40_patchtst/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "Temporal Fusion Transformer": {
      "all_sources_used": [],
      "alternatives_rejected": [],
      "selected": null
    },
    "Transformer": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "bed93a0152859bfa110fb4ec5500ca5b947fff26855efcbce7d440de4bbe6292",
          "mtime_utc": "2026-09-15T19:07:50.458037+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "a800f5e1a6f82a2776b124435fad02f7360b1d05c5c6c165dc2e6d0a718e0b93",
          "mtime_utc": "2026-09-15T19:08:40.155785+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "b6f05b046c6dfe6072eb44136d13b51dc4dc3f62ed03064e44f78563af34cafa",
          "mtime_utc": "2026-09-15T19:08:18.362864+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "f1f302afdfe6577b0f2f5f95870709b5f890e7306e9b69ad64801c00dd7b0f12",
          "mtime_utc": "2026-09-15T19:08:01.851284+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "996aa1bdcfbfadea7d78988133edfa4779dba4771c195290d1e2eedaffa71c05",
          "mtime_utc": "2026-09-15T19:07:39.756973+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "f563bd4977f7e996d0f545d9d594bd64a0db05324ac6d37faacb3ad7811588ed",
          "mtime_utc": "2026-09-15T19:07:22.481619+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "0ab21394faf6cde544f896ec78d68d4f3ce6f43c70d8a90946f84e9543c64634",
          "mtime_utc": "2026-09-15T19:07:22.356882+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "bed93a0152859bfa110fb4ec5500ca5b947fff26855efcbce7d440de4bbe6292",
        "mtime_utc": "2026-09-15T19:07:50.458037+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/Transformer/metrics_only_v3_r40_transformer/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "elastic_net": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "b31417b9b6165143cf6c5084f10d7b5a4e62fcf2922af47b95603cf060b7128d",
          "mtime_utc": "2026-09-12T04:10:41.651606+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "fcccc1f9ec75c18936531c7da61fed49d4017a90d2856f7271400dbbcfc39be1",
          "mtime_utc": "2026-09-12T04:10:52.951306+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "e87c9eb0ec2b5585a1a55a40882d76d023f30c08f58421c6a7db5d27a03eb014",
          "mtime_utc": "2026-09-12T04:10:52.787098+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "d32d9965592cdb6bebeaca36ab854868ff6ac77a2dfdf78f4bb8a0e0cdcd3689",
          "mtime_utc": "2026-09-12T04:10:46.568526+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "74ad7dbf7d9771ef2731f2fd86505ce68231050c73f1456f97f771273bbf8393",
          "mtime_utc": "2026-09-12T04:10:38.735062+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "c50d1fdc502e59c2b9040f853556dee28842241fafdad1a1da28e35464b5eb64",
          "mtime_utc": "2026-09-12T04:10:32.941272+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "8596a2437138c616b50cfa48f8a028d379b6e88e6c50122db0e86e8a39b14f3d",
          "mtime_utc": "2026-09-12T04:10:32.863937+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "f4f13f46d6273d4cb88302b164a3603db517ef263b7c05bdb8b791c9e90c62cb",
          "mtime_utc": "2026-08-27T02:27:25.972502+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "b31417b9b6165143cf6c5084f10d7b5a4e62fcf2922af47b95603cf060b7128d",
        "mtime_utc": "2026-09-12T04:10:41.651606+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "extra_trees": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "0fddc1354f6822ab3110b94552ee88adb5b61905890580994669b735da4a7ca6",
          "mtime_utc": "2026-09-10T00:25:16.204553+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "2aa7a7ad99c020cee0e69fa4d4b3c32ce1f6f76899920a475c9889f3261287ca",
          "mtime_utc": "2026-09-10T00:29:10.939693+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "25d1cd5569afc0ce098676b9d009a800881c239b86deb79537a482a7941c3ccc",
          "mtime_utc": "2026-09-10T00:29:01.303592+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "1d0e9de6df6a6123ba5230294f2c4fa95723cc3ad6db1fa23da81091d29c5aa4",
          "mtime_utc": "2026-09-10T00:29:01.225987+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "700929bd054918b627f17e83eb152f3f6942b063cf998328df70980976c4affc",
          "mtime_utc": "2026-09-10T00:25:35.449098+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "3ca763602d89f04f33360314c1882e2dd94be43d03f1e7e26139fe1efb55848d",
          "mtime_utc": "2026-09-10T00:25:35.176619+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "85ba94ecc7d8d4842b1175003e2a8340a59f007d167f751bf884e617b4ca2b51",
          "mtime_utc": "2026-09-10T00:25:25.429424+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "0fddc1354f6822ab3110b94552ee88adb5b61905890580994669b735da4a7ca6",
        "mtime_utc": "2026-09-10T00:25:16.204553+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "gradient_boosting": {
      "all_sources_used": [
        {
          "artifact_type": "progress.json",
          "file_sha256": "13bc0f8a606daefb50922089ba9668100d19451367f0da09bf1506b543ca2bfe",
          "mtime_utc": "2026-09-10T00:24:13.883806+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "064cf282f1f97ccd7638c35026acac4438faaf672e956536be6d7564323d16bb",
          "mtime_utc": "2026-09-10T00:24:13.843956+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/metrics_only_v3_r40_gradient_boosting/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "25defaa1bccc52bd488d0936f5d9da73a918d171ea6be94490b19e9b9a91088b",
          "mtime_utc": "2026-09-10T00:24:08.846358+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/metrics_only_v3_r40_gradient_boosting/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "checkpoint.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/metrics_only_v3_r40_gradient_boosting/checkpoint.json",
          "source_tier": "B",
          "why_rejected": "no summary artifact available"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/metrics_only_v3_r40_gradient_boosting/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_rejected": "no summary artifact available"
        }
      ],
      "selected": {
        "artifact_type": "progress.json",
        "file_sha256": "13bc0f8a606daefb50922089ba9668100d19451367f0da09bf1506b543ca2bfe",
        "mtime_utc": "2026-09-10T00:24:13.883806+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/gradient_boosting/progress.json",
        "source_tier": "B",
        "why_selected": "selected highest-authority available source"
      }
    },
    "huber": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "6e2c3d8374453dede892b72a5d1b0d714db008b5961ab3a4f38ef79572f84604",
          "mtime_utc": "2026-09-06T19:39:07.103761+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "35b822aecba07e0396618890909640bcd10698c88599b692450457cfe83d629c",
          "mtime_utc": "2026-08-28T14:15:55.672760+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "r31_performance_summary.json",
          "file_sha256": "c43761112022afffafa9508399fd3dd696cdbe0293d11105db42317effc324fe",
          "mtime_utc": "2026-09-07T10:59:49.114615+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/r31_performance_summary.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "e6cac75549d41ba580161b2cca775be407aab57d426f77c02e027618cf3b65b1",
          "mtime_utc": "2026-09-06T19:40:40.936969+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "ffa00aec0308899f1552903c4322c0f4b6b1211f182726b3785c6b3f0138c311",
          "mtime_utc": "2026-09-06T19:40:39.488320+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "4058bd7d651f65c07a883eab4133b559794d0276d5ab2098c9506934e1561e55",
          "mtime_utc": "2026-09-06T19:39:49.188571+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "873d322d109cc78747c433dc3ce5ea2ce6c3eb508effedd1de1ef4f2f3f37255",
          "mtime_utc": "2026-09-06T19:38:41.735116+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "85e2076e9ceb6feca39a99d91e8ea4fa6b613dbf22a279265fdb4c2f38e2b6c7",
          "mtime_utc": "2026-09-06T19:37:53.085626+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "0ff6f3a84cc4feeea0ae02b950fd56f90d53fc02d8e1dd237db42fb8992fd036",
          "mtime_utc": "2026-09-06T19:37:52.613747+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "c6f3bb858b9178a141ee002fb2b33b377c86dd81b895a1ad433d1aa5ae186d52",
          "mtime_utc": "2026-08-28T14:15:56.851960+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "20a29374e5c6f987a2e4e7af69cec0b468223e481834ab812e5fd32d5f65aec6",
          "mtime_utc": "2026-08-28T14:15:55.778888+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "02d37edb037b7f88ee3f741379aa9f4256c7358ebab68d431434ef094723af51",
          "mtime_utc": "2026-08-28T14:15:55.482086+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "6683d26bfbb688843d0a406a2e7b591c5608d841bfec64db0f2287c3b93cafdd",
          "mtime_utc": "2026-08-28T14:15:55.429452+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "ee56ee1cd587d028d738402bbc4f3230d4fcff1562d6c1bf42a0856602850ab8",
          "mtime_utc": "2026-08-28T10:19:06.285148+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_rejected": "accepted live progress metrics_root has precedence"
        },
        {
          "artifact_type": "r31_performance_summary.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/r31_performance_summary.json",
          "source_tier": "E",
          "why_rejected": "accepted live progress metrics_root has precedence"
        }
      ],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "6e2c3d8374453dede892b72a5d1b0d714db008b5961ab3a4f38ef79572f84604",
        "mtime_utc": "2026-09-06T19:39:07.103761+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "iTransformer": {
      "all_sources_used": [],
      "alternatives_rejected": [],
      "selected": null
    },
    "lightgbm_lambdarank": {
      "all_sources_used": [
        {
          "artifact_type": "R51_lambdarank_import_result.json",
          "file_sha256": "282fefe253ffe804965d5fef23e8a056bb9384a67af42123e42068661b011057",
          "mtime_utc": "2026-09-10T14:26:49.920597+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R51_lambdarank_import_result.json",
          "source_tier": "C",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "R51_lambdarank_import_result.json",
        "file_sha256": "282fefe253ffe804965d5fef23e8a056bb9384a67af42123e42068661b011057",
        "mtime_utc": "2026-09-10T14:26:49.920597+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R51_lambdarank_import_result.json",
        "source_tier": "C",
        "why_selected": "selected highest-authority available source"
      }
    },
    "lightgbm_rank_xendcg": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "95d86023e1c50fe8b2599668d7d8f3313eff8ce178e16e2c425601296c8c1a7c",
          "mtime_utc": "2026-09-05T14:19:32.305130+00:00",
          "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json",
          "source_tier": "C",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "family_execution_summary.json",
          "file_sha256": "bb68578b0a75d952def54ef3d6de800a1ff4fe99c0db7da0a38e691749006a06",
          "mtime_utc": "2026-09-05T14:19:39+00:00",
          "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/family_execution_summary.json",
          "source_tier": "C",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "R47A_xendcg_import_result.json",
          "file_sha256": "8b81abd0fdc7165be141cf7c1c438829e2e3f9b42454d7fd805c9266a798d981",
          "mtime_utc": "2026-09-09T15:10:05.707448+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R47A_xendcg_import_result.json",
          "source_tier": "C",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "R47A_xendcg_source_discovery.json",
          "file_sha256": "a3d20a5251b1075b854d98a7dcab9bfc2fdfec14be9b6b3f048d5342b949c673",
          "mtime_utc": "2026-09-09T15:10:05.336473+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R47A_xendcg_source_discovery.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "ensemble_oof_scores_manifest_v2.json",
          "file_sha256": "65a49db5d67c0e6b165a00c432e5f75c68a47f6fccfbbeb90d6bdfa1a82514f6",
          "mtime_utc": "2026-09-05T14:19:04+00:00",
          "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/ensemble_oof_scores_manifest_v2.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "e380f3354cb751566be5c5d97c0b2108a32e239da6a0eca7353306bcc5ca291f",
          "mtime_utc": "2026-09-05T14:18:50+00:00",
          "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "family_execution_summary.json",
          "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/family_execution_summary.json",
          "source_tier": "C",
          "why_rejected": "stronger source tier or newer accepted authority selected"
        }
      ],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "95d86023e1c50fe8b2599668d7d8f3313eff8ce178e16e2c425601296c8c1a7c",
        "mtime_utc": "2026-09-05T14:19:32.305130+00:00",
        "path": "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json",
        "source_tier": "C",
        "why_selected": "selected highest-authority available source"
      }
    },
    "mlp": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "dde5d615d4d4d22fa45b8c306af0010a34cbfa93ec86231b10b6a404fca47b97",
          "mtime_utc": "2026-09-10T00:28:35.564761+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "0be60c33a7b803072776e441522a46b87162bab59a7dac5b8f3aff72e14804fe",
          "mtime_utc": "2026-08-28T19:25:08.538744+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "d2e5180f4a21fd9e515fd23275338bd40fbdeaf927457dc60d552152bd4c40b6",
          "mtime_utc": "2026-09-10T00:29:05.264332+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "a636c368fc04db73e57e95a842351838b6e7d477fd69ce4a5f03789b0838c1b7",
          "mtime_utc": "2026-09-10T00:27:43.106624+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "2cc4d1b6cfdb495752fe011d500a8de66707d289cf23ce72e11d4ffb0325c2aa",
          "mtime_utc": "2026-09-10T00:27:42.964825+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "f8b39c6f627cfbeea8ec325fa4f5e0836a8f8b4c4b1a81eca60929b03c789f15",
          "mtime_utc": "2026-09-10T00:24:56.775299+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "0dc261ea3330e877f00c5dedbfb60948478253f7c249e098e050720053ed1b90",
          "mtime_utc": "2026-09-10T00:24:55.723496+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "0c90984040ad16ebaa28d8bf5b4609be44f20268d910bd9127d1c03ff471f47e",
          "mtime_utc": "2026-09-10T00:18:54.147117+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "ae9d1344e65813a3a2e6022b175ba0546af1ffe45157e7ce87d7eea589bc2bba",
          "mtime_utc": "2026-08-28T19:25:09.715877+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "b4b014a2e30ac9cd9f38488a78e6759147689bf7382e7adb201c2cbcac0f6dc9",
          "mtime_utc": "2026-08-28T19:25:08.729149+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "58b3248be2865a626dbfd3ddb2ff5dc5a1f4dcd2abad682affc000c9a27ad9bf",
          "mtime_utc": "2026-08-28T19:25:08.253207+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "a228ca99c5d77ac8dd020ecefe3a58092fbda8c48818117032d835174d17674f",
          "mtime_utc": "2026-08-28T19:25:08.198496+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "73a2d9464b88c4673a02ad2221e32e20a3b3acfb11670581c1d286f34260ddca",
          "mtime_utc": "2026-08-28T11:29:39.267205+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_retry/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_rejected": "accepted live progress metrics_root has precedence"
        }
      ],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "dde5d615d4d4d22fa45b8c306af0010a34cbfa93ec86231b10b6a404fca47b97",
        "mtime_utc": "2026-09-10T00:28:35.564761+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "pca_ridge_policy_v1_control": {
      "all_sources_used": [
        {
          "artifact_type": "r31_performance_summary.json",
          "file_sha256": "4a87fa2d2d4b55311f80474cebd12c0f5eb61600a13f867051b5e87e5b9fdab6",
          "mtime_utc": "2026-08-28T12:05:11.258893+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/pca_ridge_policy_v1_control/r31_performance_summary.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "2ce16f2b69a8b4b16670866d3b5b237cb5e0cf4514818be3dee1280ee57a575b",
          "mtime_utc": "2026-08-27T18:06:55.653919+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/pca_ridge_policy_v1_control/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "403bd055256b897cc4d8f5b9a681fcb447e1d37e996f6d2a4d3a1a6d2bb7b830",
          "mtime_utc": "2026-08-27T18:06:54.817748+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/pca_ridge_policy_v1_control/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "r31_performance_summary.json",
        "file_sha256": "4a87fa2d2d4b55311f80474cebd12c0f5eb61600a13f867051b5e87e5b9fdab6",
        "mtime_utc": "2026-08-28T12:05:11.258893+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/pca_ridge_policy_v1_control/r31_performance_summary.json",
        "source_tier": "E",
        "why_selected": "selected highest-authority available source"
      }
    },
    "random_forest": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "c9aaf84930286e69855245a0067b64822048340db49a51b42002812b7b655758",
          "mtime_utc": "2026-09-15T18:48:16.029408+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "c9a38b5515a218e619a544f08fd27d8582ecb6304f093de8827ac43de96a9b6d",
          "mtime_utc": "2026-09-15T18:48:25.954858+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "75991ccfd520f8fe2feb64884bddb3ad442e946e11b05242ceb9309280e3abb0",
          "mtime_utc": "2026-09-15T18:48:25.814794+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "c98dd542ce595fe3531a6e7148f13c69ba3bc6321fd6923b2cbe47f4ca56282e",
          "mtime_utc": "2026-09-15T18:48:19.789244+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "25c6a220039bd24572e390de74154e7ab89e59ab274407b847e13ea4c8013201",
          "mtime_utc": "2026-09-15T18:48:08.831416+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "1c1e299b4164c1f9b28b0825cc251252fd6597bfa51edcb58e8866ba4a129d65",
          "mtime_utc": "2026-09-15T18:48:08.755047+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "c9aaf84930286e69855245a0067b64822048340db49a51b42002812b7b655758",
        "mtime_utc": "2026-09-15T18:48:16.029408+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "rff_ridge": {
      "all_sources_used": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "87c7004430c8ebee24072f1b0dff3419c33c89eea154daf2db5ce7f33e31045b",
          "mtime_utc": "2026-09-05T23:28:28.566417+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "bb6c9452ab72642ae34d47426c02e381241363f43045168459533f371edfa9ba",
          "mtime_utc": "2026-08-28T16:15:38.775002+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "file_sha256": "e649c28fb2f8b51c040c652850353d18bd4f22b7d58c761108c921c1e37bbc3f",
          "mtime_utc": "2026-08-28T13:13:44.078609+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "r31_performance_summary.json",
          "file_sha256": "a0e562a1b98c8938835b9a00677d9986ad838346f531ad466abc1e8525cd2a9d",
          "mtime_utc": "2026-09-07T10:26:39.411114+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/r31_performance_summary.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "3a35a7da26fcf463bdebcc05b6bf41ef325ab48e438a344393f886c42feb6935",
          "mtime_utc": "2026-09-05T23:29:57.145120+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "a4900656bd82926f447c432905a2d6ea68d0e9220189450122ab9746a0e92393",
          "mtime_utc": "2026-09-05T23:29:55.812377+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "0b8d309f362ebf7b3ac84a0b22f9627fa21b8eb66c02e7ce0c8cb373949b1db2",
          "mtime_utc": "2026-09-05T23:29:08.060573+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "daily_portfolio_returns_v3_manifest.json",
          "file_sha256": "f9ada847bafefade99cda6cd6d53b5c913e5eb3587cf34fcce0ffb11b2833dfa",
          "mtime_utc": "2026-09-05T23:27:59.277304+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/daily_portfolio_returns_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "469fec98b778ee4f4e63084ab3b9d5675af655d04029d2a0793c9047358e5004",
          "mtime_utc": "2026-09-05T23:27:09.200006+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "78efccbbad84a94abbb0a5bb00a6d114b9468c2a9edd705ddd7e2c488de6dc17",
          "mtime_utc": "2026-09-05T23:27:09.031107+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "15fa357a0fba8a3673ec68c3d9302ec694d7d62ce71b3ca79d551a43f3674d6e",
          "mtime_utc": "2026-08-28T16:15:39.628332+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "2076cb27d85177aff47acc19efba689eac209a7c6b49b9766b322211a52a5cf7",
          "mtime_utc": "2026-08-28T16:15:38.811194+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "36a0ae329d99d341ec67ef02102d8ef7fd5def75de40abc6613fa3f8752579c7",
          "mtime_utc": "2026-08-28T16:15:38.516872+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "22c50646677b9ef8219ada26412ad2065935c8189a79b1cffd6aaf8d9bc3503b",
          "mtime_utc": "2026-08-28T16:15:38.452260+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "9093dae155a436681e02d9be586b6729bf446ce91426113a602a89735a1c7f68",
          "mtime_utc": "2026-08-28T13:13:45.137559+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "resolved_performance_checkpoint_v3.json",
          "file_sha256": "b546678b791cfbc15809719a26847a713b7480381dc04fc9091518cb04c305b7",
          "mtime_utc": "2026-08-28T13:13:44.131305+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/resolved_performance_checkpoint_v3.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "transaction_costs_v3_manifest.json",
          "file_sha256": "0046b8fa8abdd500d8ca301489e2b99784dabcd038e30d7c8f4423828175f4de",
          "mtime_utc": "2026-08-28T13:13:43.839766+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/transaction_costs_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "rank_ic_v3_manifest.json",
          "file_sha256": "27098e261b0ac397dd84359d141ed8712d052dae5e0ef762e8ae1be64167400b",
          "mtime_utc": "2026-08-28T13:13:43.786487+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/rank_ic_v3_manifest.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "b6f39ceead46d3b133c3bf47f44c5e6d4a9d4027dc1109a2b1796543a2eaf696",
          "mtime_utc": "2026-08-28T09:57:59.733870+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "6766b82fd84f3dc26b08aa871047312638c589fe160fc3c9231c0b8728d66b9b",
          "mtime_utc": "2026-08-27T20:20:43.689871+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/SUPERSEDED_EVALUATION_CONTRACT_V1/progress.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "709a45cf7f4cac103a7798cddd45b186a16d6924835590bf87449a0647bf244b",
          "mtime_utc": "2026-08-27T20:20:43.598558+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/SUPERSEDED_EVALUATION_CONTRACT_V1/metrics_only/checkpoint.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_replay/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_rejected": "accepted live progress metrics_root has precedence"
        },
        {
          "artifact_type": "resolved_performance_summary_v3.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3/resolved_performance_summary_v3.json",
          "source_tier": "A",
          "why_rejected": "accepted live progress metrics_root has precedence"
        },
        {
          "artifact_type": "r31_performance_summary.json",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/r31_performance_summary.json",
          "source_tier": "E",
          "why_rejected": "accepted live progress metrics_root has precedence"
        }
      ],
      "selected": {
        "artifact_type": "resolved_performance_summary_v3.json",
        "file_sha256": "87c7004430c8ebee24072f1b0dff3419c33c89eea154daf2db5ce7f33e31045b",
        "mtime_utc": "2026-09-05T23:28:28.566417+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/resolved_performance_summary_v3.json",
        "source_tier": "A",
        "why_selected": "selected highest-authority available source"
      }
    },
    "ridge_policy_v1_control": {
      "all_sources_used": [
        {
          "artifact_type": "r31_performance_summary.json",
          "file_sha256": "be79de7331fdd53291393ff8025f4d86fb3ffb80009f58af43f5876a1da8015b",
          "mtime_utc": "2026-08-28T12:05:11.247416+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/ridge_policy_v1_control/r31_performance_summary.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "5a65adc148d1441a7b9f9420f5845ec072385fe93d48d21880878360f5dd932c",
          "mtime_utc": "2026-08-27T17:24:38.186306+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/ridge_policy_v1_control/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "3b58589aa19ebbe559630c13e70c9a99ad1903aacd494d3bae984c0379862e4e",
          "mtime_utc": "2026-08-27T17:24:38.025738+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/ridge_policy_v1_control/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "r31_performance_summary.json",
        "file_sha256": "be79de7331fdd53291393ff8025f4d86fb3ffb80009f58af43f5876a1da8015b",
        "mtime_utc": "2026-08-28T12:05:11.247416+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/ridge_policy_v1_control/r31_performance_summary.json",
        "source_tier": "E",
        "why_selected": "selected highest-authority available source"
      }
    },
    "spline_additive_ridge": {
      "all_sources_used": [
        {
          "artifact_type": "r31_performance_summary.json",
          "file_sha256": "04aa89388495eb00ee8c6c1b3abd55fc90bf3a3b37b269fbf4a3bd2184907e7b",
          "mtime_utc": "2026-08-28T12:05:11.271388+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/spline_additive_ridge/r31_performance_summary.json",
          "source_tier": "E",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "progress.json",
          "file_sha256": "402e38ba1b29f740c1991f7d793fb32fb0b4fb310d8b0b9dcb2460045d2e6a81",
          "mtime_utc": "2026-08-27T17:14:44.480969+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/spline_additive_ridge/progress.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        },
        {
          "artifact_type": "checkpoint.json",
          "file_sha256": "5624ed12906f990b0d6003116c3a919ac8ca1734055b2ab875390ba9e4027996",
          "mtime_utc": "2026-08-27T17:14:44.341972+00:00",
          "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/spline_additive_ridge/metrics_only/checkpoint.json",
          "source_tier": "B",
          "why_selected": "candidate source inspected"
        }
      ],
      "alternatives_rejected": [],
      "selected": {
        "artifact_type": "r31_performance_summary.json",
        "file_sha256": "04aa89388495eb00ee8c6c1b3abd55fc90bf3a3b37b269fbf4a3bd2184907e7b",
        "mtime_utc": "2026-08-28T12:05:11.271388+00:00",
        "path": "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/spline_additive_ridge/r31_performance_summary.json",
        "source_tier": "E",
        "why_selected": "selected highest-authority available source"
      }
    }
  },
  "generated_at_utc": "2026-09-15T19:10:09.820633+00:00",
  "ownership_sources": [
    "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R44_cross_host_family_ownership.json",
    "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R47A_cross_host_ownership_state.json",
    "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R49_cross_host_ownership_state.json",
    "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R51_cross_host_ownership_state.json"
  ],
  "repository_authoritative_population": [
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "elastic_net",
    "rff_ridge",
    "huber",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer"
  ],
  "requested_19_family_population": [
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "elastic_net",
    "rff_ridge",
    "huber",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer"
  ]
}

## Runtime safety

- Pre worker PIDs: []
- Post worker PIDs: []
- R53 workers stopped: 0; workers launched: 0; supervisor restarted: 0; model fits: 0; predictions generated: 0; paper orders: 0; live orders: 0; outer holdout accessed: false.
