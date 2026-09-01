param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [Parameter(Mandatory=$true)][string]$Destination, [string]$SshUser = "root", [switch]$Execute)
$ErrorActionPreference = "Stop"
if (-not $Execute) { Write-Host "[DRY RUN] Would download checkpoints, V3 metrics, compact OOF, ensemble inputs, telemetry and throughput summaries."; exit 0 }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
if (Get-Command rsync -ErrorAction SilentlyContinue) {
  rsync -a --partial --append-verify --info=progress2 -e "ssh -p $SshPort" "$SshUser@$SshHost:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/" "$Destination/"
} else {
  scp -P $SshPort -r "$SshUser@$SshHost:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1" "$Destination/"
}
