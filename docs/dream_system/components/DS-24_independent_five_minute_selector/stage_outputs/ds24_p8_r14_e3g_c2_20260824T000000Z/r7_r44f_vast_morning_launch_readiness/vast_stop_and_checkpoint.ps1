param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
$ErrorActionPreference = "Stop"
if (-not $Execute) { Write-Host "[DRY RUN] Would checkpoint, flush, sync, then run vastai stop instance through R44E2 guard."; exit 0 }
ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && STOP_REASON=user_review_stop bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44e2_vast_soft_review_and_full_queue_transition/pause_queue_at_budget.sh"
