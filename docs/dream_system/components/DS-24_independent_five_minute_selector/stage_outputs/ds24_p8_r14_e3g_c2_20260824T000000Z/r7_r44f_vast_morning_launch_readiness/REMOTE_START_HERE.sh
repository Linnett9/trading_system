#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---launch-full-queue}"
export SOURCE_ROOT="${SOURCE_ROOT:-/workspace/ds24/source}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/ds24/output}"
export QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1}"
export FULL_DATASET_MANIFEST="${FULL_DATASET_MANIFEST:-${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/06_full_partition_manifest.csv}"
export EXPECTED_FULL_DATASET_MANIFEST_SHA256="${EXPECTED_FULL_DATASET_MANIFEST_SHA256:-6bcefad7f7bc98fb929a8f49f0b02de8add348cc5d661a84b9d3fd004ae66555}"
export EXPECTED_FULL_DATASET_SCHEMA_HASH="${EXPECTED_FULL_DATASET_SCHEMA_HASH:-f7162068d0d4e06a27395c6923dc7298335d955e401ad26a2ac39bbcdeda69cb}"
export SOFT_REVIEW_MINUTES="${SOFT_REVIEW_MINUTES:-90}"
export HARD_BUDGET_USD="${HARD_BUDGET_USD:-8.40}"
export HARD_WALL_CLOCK_HOURS="${HARD_WALL_CLOCK_HOURS:-20}"
export REVIEW_GRACE_MINUTES="${REVIEW_GRACE_MINUTES:-15}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export LIGHTGBM_NUM_THREADS="${LIGHTGBM_NUM_THREADS:-8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export DS24_RANDOM_SEED="${DS24_RANDOM_SEED:-1729}"
mkdir -p "${QUEUE_ROOT}" "${OUTPUT_ROOT}"
if [[ -z "${INSTANCE_START_TIMESTAMP:-}" && -f /workspace/ds24/control/INSTANCE_START_TIMESTAMP ]]; then
  export INSTANCE_START_TIMESTAMP="$(cat /workspace/ds24/control/INSTANCE_START_TIMESTAMP)"
fi
: "${INSTANCE_START_TIMESTAMP:?Set actual Vast billing start timestamp from instance create/start}"
: "${HOURLY_COMPUTE_PRICE:?Set validated complete compute hourly price}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness}"
rm -f "${QUEUE_ROOT}/WATCHDOG_PREFLIGHT_FAILED"
tmux has-session -t ds24_r44f_r44e2_watchdog 2>/dev/null || tmux new-session -d -s ds24_r44f_r44e2_watchdog "cd '${SOURCE_ROOT}' && exec bash '${RUNTIME_ROOT}/budget_watchdog.sh'"
for _ in $(seq 1 40); do
  test -f "${QUEUE_ROOT}/WATCHDOG_ARMED" && break
  test -f "${QUEUE_ROOT}/WATCHDOG_PREFLIGHT_FAILED" && exit 12
  sleep 1
done
test -f "${QUEUE_ROOT}/WATCHDOG_ARMED" || { echo "R44E2 watchdog failed to arm before dependency install/upload/model work"; exit 13; }
date -u +%FT%TZ > "${QUEUE_ROOT}/WATCHDOG_ARMED_BEFORE_DEPENDENCY_INSTALL_UPLOAD_MODEL_WORK"
python -m pip install --disable-pip-version-check -r "${SOURCE_ROOT}/requirements.txt" >/tmp/ds24_r44f_pip_install.log 2>&1 || { cat /tmp/ds24_r44f_pip_install.log; exit 14; }
python -m core.research.ml.ds24.vast_soft_review_transition validate-full-data --repo-root "${SOURCE_ROOT}" --manifest-path "${FULL_DATASET_MANIFEST}" --expected-manifest-sha256 "${EXPECTED_FULL_DATASET_MANIFEST_SHA256}" --expected-schema-hash "${EXPECTED_FULL_DATASET_SCHEMA_HASH}" --required-predictor-count 101 > "${QUEUE_ROOT}/r44f_full_data_gate_before_benchmark.json"
if [[ "${MODE}" == "--profile-benchmark-only" ]]; then
  python -m core.research.ml.ds24.remote_tft_r44f select-hardware-profile > "${QUEUE_ROOT}/r44f_hardware_profile_selection.json"
  exit 0
fi
python -m core.research.ml.ds24.remote_tft_r44f select-hardware-profile > "${QUEUE_ROOT}/r44f_hardware_profile_selection.json"
exec bash "${RUNTIME_ROOT}/transition_to_full_queue.sh"
