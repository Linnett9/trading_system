# DS24 Vast B2 Bootstrap, Autostart, Durable Sync and Dell Repatriation R1

Terminal classification: `DS24_VAST_B2_BOOTSTRAP_REVERSE_AUTOSTART_AND_DELL_REPATRIATION_READY_NOT_EXECUTED`

This package prepares the deployment system only. It does not rent Vast, contact Backblaze, read credentials, launch live DS24 model work, inspect holdout outcomes, or place paper/live orders.

The bootstrap consumes the accepted R49 reverse queue authority and requires fresh Dell/Mac ownership evidence before any future launch.

## Future Commands
### finalize_b2_dataset_after_upload
`python scripts/local/ds24_vast_b2_bootstrap_r1.py finalize-dataset --repo-root <DELL_REPO_ROOT> --source-manifest <SOURCE_MANIFEST> --remote-inventory <B2_INVENTORY_JSON> --authority-root C:/Users/Brandon/trading_system/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r50_vast_b2_bootstrap_reverse_autostart_durable_sync_repatriation_r1 --execute-real-b2 --confirm-token FINALIZE_DS24_B2_DATASET_AFTER_UPLOAD_COMPLETE`

### generate_dell_status_evidence
`python scripts/local/ds24_vast_reverse_queue_r1.py status --repo-root <DELL_REPO_ROOT> --queue-root <DELL_QUEUE_STATE_ROOT> > <DELL_STATUS_SNAPSHOT_JSON>`

### generate_mac_status_evidence
`python -m core.research.ml.ds24.mac_aux_queue_r44f2 status --queue-id DS24_MAC_AUX_NINE_FAMILY_R1 > <MAC_STATUS_SNAPSHOT_JSON>`

### create_ownership_plan
`python scripts/local/ds24_vast_b2_bootstrap_r1.py create-ownership-plan --repo-root <DELL_REPO_ROOT> --dell-snapshot <DELL_STATUS_SNAPSHOT_JSON> --mac-snapshot <MAC_STATUS_SNAPSHOT_JSON> --authority-root C:/Users/Brandon/trading_system/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r50_vast_b2_bootstrap_reverse_autostart_durable_sync_repatriation_r1`

### publish_acknowledgement
`python scripts/local/ds24_vast_b2_bootstrap_r1.py publish-ack --machine <dell|mac> --plan <OWNERSHIP_PLAN_JSON> --output <ACK_JSON>`

### vast_preflight
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --preflight-only --vast-instance-id <VAST_INSTANCE_ID>`

### vast_download_only
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --download-only --vast-instance-id <VAST_INSTANCE_ID>`

### vast_verify_only
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --verify-only --vast-instance-id <VAST_INSTANCE_ID>`

### vast_start_after_verify
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --start-after-verify --vast-instance-id <VAST_INSTANCE_ID> --execute-live --confirm-token AUTHORIZE_DS24_VAST_LIVE_START_AFTER_VERIFY`

### vast_status
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --status --vast-instance-id <VAST_INSTANCE_ID>`

### vast_resume
`python scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --resume --vast-instance-id <VAST_INSTANCE_ID>`

### vast_output_publisher_status
`python scripts/local/ds24_vast_b2_bootstrap_r1.py publisher --config <PUBLISHER_CONFIG_JSON> --status --run-id <RUN_ID>`

### dell_compact_artifact_retrieval
`python scripts/local/ds24_vast_b2_bootstrap_r1.py repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --once --run-id <RUN_ID> --tier compact`

### dell_full_artifact_retrieval
`python scripts/local/ds24_vast_b2_bootstrap_r1.py repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --once --run-id <RUN_ID> --tier all`

### dell_verification
`python scripts/local/ds24_vast_b2_bootstrap_r1.py repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --verify-only --run-id <RUN_ID>`

### safe_terminal_closeout
`python scripts/local/ds24_vast_b2_bootstrap_r1.py prepare-package --repo-root <DELL_REPO_ROOT> --authority-root C:/Users/Brandon/trading_system/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r50_vast_b2_bootstrap_reverse_autostart_durable_sync_repatriation_r1`

## Gates

The dataset finalizer must run after the current upload completes and must publish `DATASET_COMPLETE.json` last. The Vast bootstrap treats that marker as necessary but still verifies the downloaded files locally.

Ownership planning is generation-scoped and expires. Dell acknowledgement is required; Mac acknowledgement or a documented unavailability reason is required. The plan freezes a static non-overlapping Vast partition for that generation.

The publisher uses copy semantics and writes committed markers last. The Dell repatriation client imports into staging, validates hashes, quarantines conflicts, and writes receipts without publishing into live worker namespaces.

Scientific acceptance of any imported Vast artifact remains a separate decision.
