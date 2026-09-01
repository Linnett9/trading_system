param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
$ErrorActionPreference = "Stop"
if (-not $Execute) { Write-Host "[DRY RUN] Would display SMOKE_90_MINUTE_REVIEW_READY and review summaries."; exit 0 }
ssh -p $SshPort "$SshUser@$SshHost" "ls -l /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/SMOKE_90_MINUTE_REVIEW_READY /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_telemetry_summary.json /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_throughput_summary.json; cat /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_budget_status.json"
