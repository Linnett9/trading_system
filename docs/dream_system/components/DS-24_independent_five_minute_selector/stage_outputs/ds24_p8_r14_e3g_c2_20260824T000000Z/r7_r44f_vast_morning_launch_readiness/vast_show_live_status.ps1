param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
$ErrorActionPreference = "Stop"
if (-not $Execute) { Write-Host "[DRY RUN] Would show elapsed billed time, estimated spend, remaining hard-budget time, current family, completed work, GPU/CPU/RAM utilisation, throughput forecast."; exit 0 }
ssh -p $SshPort "$SshUser@$SshHost" "cat /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_budget_status.json; nvidia-smi || true; free -h || true; df -h /workspace || true"
