#!/usr/bin/env bash
set -euo pipefail
ROOT="${DS24_RUN_ROOT:-/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=${DS24_RUN_ID:-latest}}"
echo "active family"; cat "$ROOT/queue_state/current_family.json" 2>/dev/null || true
echo "queue cursor"; cat "$ROOT/queue_state/queue_state.json" 2>/dev/null || true
echo "gpu"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
echo "gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
echo "cpu/ram/disk"; uptime || true; free -h || true; df -h /workspace || true
echo "checkpoint age"; find "$ROOT/checkpoints" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -20 || true
echo "latest publication"; cat "$ROOT/publisher/latest_successful_publication.json" 2>/dev/null || true
