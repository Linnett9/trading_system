param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
$ErrorActionPreference = "Stop"
if (-not $Execute) { Write-Host "[DRY RUN] Would launch R44E2 guarded full queue on the same instance."; exit 0 }
ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness/REMOTE_START_HERE.sh --launch-full-queue"
