param(
  [Parameter(Mandatory=$true)][string]$SshHost,
  [Parameter(Mandatory=$true)][int]$SshPort,
  [string]$SshUser = "root",
  [string]$RepoRoot = (Resolve-Path "..\..\..\..\..\..\..\..").Path,
  [switch]$Execute
)
$ErrorActionPreference = "Stop"
$LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TransferRoot = Join-Path $LaunchRoot "transfer"
$FilesFrom = Join-Path $TransferRoot "full_data_rsync_files_from.txt"
$Bundle = Join-Path $TransferRoot "ds24_r44f_morning_runtime_source_bundle.zip"
$BundleSha = Join-Path $TransferRoot "ds24_r44f_morning_runtime_source_bundle.sha256"
if (-not $Execute) { Write-Host "[DRY RUN] Would upload source bundle and full data with rsync/rclone/scp resume semantics."; exit 0 }
ssh -p $SshPort "$SshUser@$SshHost" "mkdir -p /workspace/ds24/source /workspace/ds24/upload /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1 && df -BG /workspace | tee /workspace/ds24/upload/free_space_before_upload.txt"
scp -P $SshPort "$Bundle" "$BundleSha" "$SshUser@$SshHost:/workspace/ds24/upload/"
ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/upload && sha256sum -c ds24_r44f_morning_runtime_source_bundle.sha256 && unzip -oq ds24_r44f_morning_runtime_source_bundle.zip -d /workspace/ds24/source"
if (Get-Command rsync -ErrorAction SilentlyContinue) {
  rsync -a --partial --append-verify --info=progress2 --files-from="$FilesFrom" -e "ssh -p $SshPort" "$RepoRoot/" "$SshUser@$SshHost:/workspace/ds24/source/"
} elseif (Get-Command rclone -ErrorAction SilentlyContinue) {
  rclone copy "$RepoRoot" ":sftp:/workspace/ds24/source" --sftp-host "$SshHost" --sftp-port "$SshPort" --sftp-user "$SshUser" --files-from "$FilesFrom" --progress --retries 8 --low-level-retries 20
} else {
  Write-Host "rsync/rclone unavailable; using bounded scp fallback with .partial resume markers."
  Get-Content -LiteralPath $FilesFrom | ForEach-Object {
    $rel = $_
    if ($rel.Trim().Length -eq 0) { return }
    $local = Join-Path $RepoRoot $rel
    $remoteDir = Split-Path "/workspace/ds24/source/$rel" -Parent
    ssh -p $SshPort "$SshUser@$SshHost" "mkdir -p '$remoteDir'"
    scp -P $SshPort "$local" "$SshUser@$SshHost:/workspace/ds24/source/$rel.partial"
    ssh -p $SshPort "$SshUser@$SshHost" "mv '/workspace/ds24/source/$rel.partial' '/workspace/ds24/source/$rel'"
  }
}
ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && python -m core.research.ml.ds24.remote_tft_r44f verify-transfer-manifest --repo-root /workspace/ds24/source --manifest-path docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness/transfer/full_data_transfer_manifest.csv"
