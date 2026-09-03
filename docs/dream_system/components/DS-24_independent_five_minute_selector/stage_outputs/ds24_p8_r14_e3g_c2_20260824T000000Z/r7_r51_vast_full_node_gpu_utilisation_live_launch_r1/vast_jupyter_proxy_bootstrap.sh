#!/usr/bin/env bash
set -Eeuo pipefail
set +x
umask 077

: "${DS24_VAST_LIVE_CONFIRM_TOKEN:?Set DS24_VAST_LIVE_CONFIRM_TOKEN}"
if [ "$DS24_VAST_LIVE_CONFIRM_TOKEN" != "AUTHORIZE_DS24_VAST_JUPYTER_PROXY_LIVE_LAUNCH_R1" ]; then
  echo "Refusing launch: DS24_VAST_LIVE_CONFIRM_TOKEN mismatch" >&2
  exit 64
fi
: "${B2_APPLICATION_KEY_ID:?Set B2_APPLICATION_KEY_ID in the Jupyter terminal}"
: "${B2_APPLICATION_KEY:?Set B2_APPLICATION_KEY in the Jupyter terminal}"

export DS24_REPO_URL="${DS24_REPO_URL:-https://github.com/Linnett9/trading_system.git}"
export DS24_BRANCH="${DS24_BRANCH:-ds24-mac-tournament-sync-20260901}"
: "${DS24_BOOTSTRAP_COMMIT:?Set DS24_BOOTSTRAP_COMMIT to the R51 commit from the Dell closeout}"
export DS24_R49_R50_COMMIT="7ce81161711b8519ada39995f8018d959f3d468e"
export DS24_WORKSPACE="${DS24_WORKSPACE:-/workspace/ds24}"
export DS24_SOURCE_ROOT="${DS24_SOURCE_ROOT:-$DS24_WORKSPACE/source}"
export DS24_DATASET_ROOT="${DS24_DATASET_ROOT:-$DS24_WORKSPACE/data/full_data_r1}"
export DS24_RUN_ID="${DS24_RUN_ID:-vast_r51_$(date -u +%Y%m%dT%H%M%SZ)}"
export DS24_RUN_ROOT="${DS24_RUN_ROOT:-$DS24_WORKSPACE/output/remote_vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=$DS24_RUN_ID}"
export DS24_CONTROL_ROOT="${DS24_CONTROL_ROOT:-$DS24_WORKSPACE/control}"
export DS24_EXPECTED_GPU_REGEX="${DS24_EXPECTED_GPU_REGEX:-RTX}"
export DS24_MAX_RUNTIME_HOURS="${DS24_MAX_RUNTIME_HOURS:-20}"
export DS24_MAX_ESTIMATED_COST_USD="${DS24_MAX_ESTIMATED_COST_USD:-8.40}"
export DS24_HOURLY_PRICE_USD="${DS24_HOURLY_PRICE_USD:-0}"
export DS24_VAST_FORCE_CUDA=1
export DS24_VAST_SEQUENCE_DEVICE=cuda
export DS24_VAST_DATALOADER_WORKERS="${DS24_VAST_DATALOADER_WORKERS:-4}"
export DS24_VAST_PREFETCH_FACTOR="${DS24_VAST_PREFETCH_FACTOR:-2}"
export DS24_VAST_PIN_MEMORY=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_APPLICATION_KEY_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APPLICATION_KEY"
export RCLONE_CONFIG_B2_HARD_DELETE=false

mkdir -p "$DS24_WORKSPACE" "$DS24_CONTROL_ROOT" "$DS24_RUN_ROOT"/{logs,queue_state,publisher,telemetry,checkpoints,manifests,config}
date -u +%FT%TZ > "$DS24_RUN_ROOT/INSTANCE_START_TIMESTAMP"

if ! command -v git >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1 || ! command -v rclone >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git tmux rclone python3-pip
fi

if [ ! -d "$DS24_SOURCE_ROOT/.git" ]; then
  git clone --branch "$DS24_BRANCH" "$DS24_REPO_URL" "$DS24_SOURCE_ROOT"
fi
cd "$DS24_SOURCE_ROOT"
git fetch origin "$DS24_BRANCH" --tags
git checkout --detach "$DS24_BOOTSTRAP_COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$DS24_BOOTSTRAP_COMMIT")"
git cat-file -e "$DS24_R49_R50_COMMIT^{commit}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cat > "$DS24_CONTROL_ROOT/VAST_BOOTSTRAP_CONFIG_JSON" <<'JSON'
{
  "bootstrap_commit": "<FINAL_R51_COMMIT>",
  "branch": "ds24-mac-tournament-sync-20260901",
  "bucket": "TradingSystemDataset44",
  "config_hash": "3f35e260b59a9f1f2571c58f4aaf8d19a74ea2908995fc74fb808b403abac7b1",
  "credential_environment": {
    "B2_APPLICATION_KEY": "<set in Vast terminal; never stored>",
    "B2_APPLICATION_KEY_ID": "<set in Vast terminal; never stored>"
  },
  "dataset_marker_key": "ds24/full_data_r1/DATASET_COMPLETE.json",
  "dataset_root": "/workspace/ds24/data/full_data_r1",
  "dell_status_snapshot_base64_env": "DS24_DELL_STATUS_SNAPSHOT_JSON_B64",
  "dell_status_snapshot_path_env": "DS24_DELL_STATUS_SNAPSHOT_PATH",
  "expected_bytes": 47323707293,
  "expected_gpu_regex": "RTX",
  "expected_object_count": 18505,
  "forbidden": [
    "full_prediction",
    "full-prediction",
    "prediction_partitions",
    "holdout",
    "paper_order",
    "live_order",
    ".env",
    "credential",
    "secret",
    "rclone.conf"
  ],
  "gpu_admission_required": true,
  "live_confirm_token": "AUTHORIZE_DS24_VAST_JUPYTER_PROXY_LIVE_LAUNCH_R1",
  "mac_status_snapshot_base64_env": "DS24_MAC_STATUS_SNAPSHOT_JSON_B64",
  "mac_status_snapshot_path_env": "DS24_MAC_STATUS_SNAPSHOT_PATH",
  "ownership_plan_path": "/workspace/ds24/control/ownership_plan.json",
  "prefix": "ds24/full_data_r1",
  "publisher_config_path": "/workspace/ds24/control/PUBLISHER_CONFIG_JSON",
  "queue_id": "DS24_VAST_REVERSE_NINE_FAMILY_R1",
  "queue_order": [
    "temporal_fusion_transformer",
    "market_context_encoder",
    "momentum_transformer",
    "itransformer",
    "transformer",
    "patchtst",
    "dlinear",
    "lightgbm_lambdarank",
    "lightgbm_rank_xendcg"
  ],
  "r49_r50_commit": "7ce81161711b8519ada39995f8018d959f3d468e",
  "repo_url": "https://github.com/Linnett9/trading_system.git",
  "run_root": "/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=<RUN_ID>",
  "schema_version": "ds24_vast_full_node_live_launch_config.v1",
  "source_root": "/workspace/ds24/source",
  "workspace_root": "/workspace/ds24"
}
JSON
cat > "$DS24_CONTROL_ROOT/PUBLISHER_CONFIG_JSON" <<'JSON'
{
  "allowed_roots": [
    "queue_state",
    "ownership",
    "metrics_only_v3",
    "ensemble_oof_scores_v2",
    "checkpoints",
    "logs",
    "telemetry",
    "manifests",
    "config"
  ],
  "bucket": "TradingSystemDataset44",
  "config_hash": "03b74cea5f6f76a852371d38ff2a8ba656bdc11f277369030103eaef7141c0cb",
  "copy_mode": "rclone copy; never destructive sync",
  "credentials_included": false,
  "forbidden_markers": [
    "full_prediction",
    "full-prediction",
    "prediction_partitions",
    "holdout",
    "paper_order",
    "live_order",
    ".env",
    "credential",
    "secret",
    "rclone.conf"
  ],
  "local_run_root": "/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=<RUN_ID>",
  "max_backup_age_seconds": 1200,
  "remote_prefix": "ds24/vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=<RUN_ID>",
  "resource_overlap_gates": {
    "max_cpu_total_percent": 65,
    "max_disk_busy_percent": 70,
    "max_publisher_backlog_gb": 12,
    "min_disk_free_gb": 80,
    "min_ram_free_gb": 16
  },
  "schema_version": "ds24_vast_durable_publisher_config.v1"
}
JSON

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 write-materialized-live-configs \
  --repo-root "$DS24_SOURCE_ROOT" \
  --output-root "$DS24_RUN_ROOT/config" \
  --bootstrap-commit "$DS24_BOOTSTRAP_COMMIT"

if [ -n "${DS24_DELL_STATUS_SNAPSHOT_JSON_B64:-}" ]; then
  printf '%s' "$DS24_DELL_STATUS_SNAPSHOT_JSON_B64" | base64 -d > "$DS24_RUN_ROOT/config/dell_status_snapshot.live.json"
  export DS24_DELL_STATUS_SNAPSHOT_PATH="$DS24_RUN_ROOT/config/dell_status_snapshot.live.json"
fi
if [ -n "${DS24_MAC_STATUS_SNAPSHOT_JSON_B64:-}" ]; then
  printf '%s' "$DS24_MAC_STATUS_SNAPSHOT_JSON_B64" | base64 -d > "$DS24_RUN_ROOT/config/mac_status_snapshot.live.json"
  export DS24_MAC_STATUS_SNAPSHOT_PATH="$DS24_RUN_ROOT/config/mac_status_snapshot.live.json"
fi
: "${DS24_DELL_STATUS_SNAPSHOT_PATH:?Set DS24_DELL_STATUS_SNAPSHOT_PATH or DS24_DELL_STATUS_SNAPSHOT_JSON_B64}"
: "${DS24_MAC_STATUS_SNAPSHOT_PATH:?Set DS24_MAC_STATUS_SNAPSHOT_PATH or DS24_MAC_STATUS_SNAPSHOT_JSON_B64}"
if [ -n "${DS24_DELL_STATUS_SNAPSHOT_PATH:-}" ]; then
  python -m core.research.ml.ds24.vast_reverse_queue_r1 validate-snapshot \
    --repo-root "$DS24_SOURCE_ROOT" \
    --snapshot "$DS24_DELL_STATUS_SNAPSHOT_PATH" \
    > "$DS24_RUN_ROOT/config/dell_status_snapshot.validation.json"
fi
if [ -n "${DS24_MAC_STATUS_SNAPSHOT_PATH:-}" ]; then
  python -m core.research.ml.ds24.vast_reverse_queue_r1 validate-snapshot \
    --repo-root "$DS24_SOURCE_ROOT" \
    --snapshot "$DS24_MAC_STATUS_SNAPSHOT_PATH" \
    > "$DS24_RUN_ROOT/config/mac_status_snapshot.validation.json"
fi

echo "Downloading TradingSystemDataset44/ds24/full_data_r1 to $DS24_DATASET_ROOT"
mkdir -p "$DS24_DATASET_ROOT"
rclone copy "b2:TradingSystemDataset44/ds24/full_data_r1" "$DS24_DATASET_ROOT" \
  --transfers 16 --checkers 32 --retries 20 --low-level-retries 50 --stats 30s \
  --exclude ".env" --exclude "rclone.conf"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 verify-local-dataset \
  --dataset-root "$DS24_DATASET_ROOT" \
  --expected-count 18505 \
  --expected-bytes 47323707293 \
  --marker "$DS24_DATASET_ROOT/DATASET_COMPLETE.json" \
  > "$DS24_RUN_ROOT/manifests/local_dataset_verification.json"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 run-gpu-admission \
  --output "$DS24_RUN_ROOT/telemetry/gpu_admission.json" \
  --expected-gpu-regex "$DS24_EXPECTED_GPU_REGEX"
python -m core.research.ml.ds24.vast_gpu_live_launch_r1 validate-gpu-admission \
  --input "$DS24_RUN_ROOT/telemetry/gpu_admission.json"

cat > "$DS24_RUN_ROOT/config/accepted_reverse_order.txt" <<'EOF'
temporal_fusion_transformer market_context_encoder momentum_transformer itransformer transformer patchtst dlinear lightgbm_lambdarank lightgbm_rank_xendcg
EOF

cat > "$DS24_RUN_ROOT/publisher/publisher_loop.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
set +x
while true; do
  python -m core.research.ml.ds24.vast_gpu_live_launch_r1 publisher-once \
    --run-root "$DS24_RUN_ROOT" \
    --bucket "TradingSystemDataset44" \
    --remote-prefix "ds24/vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=$DS24_RUN_ID" || true
  sleep 300
done
SH
chmod +x "$DS24_RUN_ROOT/publisher/publisher_loop.sh"

tmux has-session -t ds24_vast_r51_publisher 2>/dev/null || \
  tmux new-session -d -s ds24_vast_r51_publisher "cd '$DS24_SOURCE_ROOT' && bash '$DS24_RUN_ROOT/publisher/publisher_loop.sh' 2>&1 | tee -a '$DS24_RUN_ROOT/logs/publisher.log'"

tmux has-session -t ds24_vast_r51_queue 2>/dev/null || \
  tmux new-session -d -s ds24_vast_r51_queue "cd '$DS24_SOURCE_ROOT' && python -m core.research.ml.ds24.vast_gpu_live_launch_r1 run-vast-reverse-queue --repo-root '$DS24_SOURCE_ROOT' --dataset-root '$DS24_DATASET_ROOT' --run-root '$DS24_RUN_ROOT' --execute-live --confirm-token 'AUTHORIZE_DS24_VAST_JUPYTER_PROXY_LIVE_LAUNCH_R1' 2>&1 | tee -a '$DS24_RUN_ROOT/logs/reverse_queue.log'"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 render-monitoring --output "$DS24_RUN_ROOT/monitoring_commands.json"
echo "DS24 Vast R51 launched in tmux. Use: tmux attach -t ds24_vast_r51_queue"
