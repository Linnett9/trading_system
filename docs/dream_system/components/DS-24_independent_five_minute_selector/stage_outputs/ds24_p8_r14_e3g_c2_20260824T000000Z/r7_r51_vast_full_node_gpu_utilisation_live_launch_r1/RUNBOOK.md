# DS24 Vast R51 Jupyter-Proxy Live Launch

Classification: `DS24_VAST_FULL_NODE_GPU_UTILISATION_LIVE_LAUNCH_IMPLEMENTED_SYNTHETICALLY_VALIDATED_READY_NOT_EXECUTED`.

Use one Vast browser-terminal session. Do not rely on direct SSH or proxy SSH.

The launcher clones `https://github.com/Linnett9/trading_system.git` at `${DS24_BOOTSTRAP_COMMIT}`, verifies the R49/R50 authority commit `7ce811617`, configures Backblaze through rclone environment variables without writing secrets to the repo, downloads `TradingSystemDataset44/ds24/full_data_r1`, verifies `18505` files and `47323707293` bytes excluding `DATASET_COMPLETE.json`, runs GPU admission, starts the durable publisher in tmux, then starts the accepted reverse queue in tmux.

Required terminal variables:

```bash
export B2_APPLICATION_KEY_ID='<Backblaze key id>'
export B2_APPLICATION_KEY='<Backblaze application key>'
export DS24_BOOTSTRAP_COMMIT='<FINAL_R51_COMMIT>'
export DS24_VAST_LIVE_CONFIRM_TOKEN='AUTHORIZE_DS24_VAST_JUPYTER_PROXY_LIVE_LAUNCH_R1'
export DS24_DELL_STATUS_SNAPSHOT_PATH='<fresh Dell snapshot JSON path, or omit only with DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP=1>'
export DS24_MAC_STATUS_SNAPSHOT_PATH='<fresh Mac snapshot JSON path, or omit only with DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP=1>'
# Optional single-paste alternative to paths:
# export DS24_DELL_STATUS_SNAPSHOT_JSON_B64='<base64 -w0 dell_status_snapshot.json>'
# export DS24_MAC_STATUS_SNAPSHOT_JSON_B64='<base64 -w0 mac_status_snapshot.json>'
```

Then run:

```bash
curl -fsSL "https://raw.githubusercontent.com/Linnett9/trading_system/${DS24_BOOTSTRAP_COMMIT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r51_vast_full_node_gpu_utilisation_live_launch_r1/vast_jupyter_proxy_bootstrap.sh" | bash
```

Monitoring commands are materialised in `monitoring_commands.json` and `vast_show_status.sh`.
