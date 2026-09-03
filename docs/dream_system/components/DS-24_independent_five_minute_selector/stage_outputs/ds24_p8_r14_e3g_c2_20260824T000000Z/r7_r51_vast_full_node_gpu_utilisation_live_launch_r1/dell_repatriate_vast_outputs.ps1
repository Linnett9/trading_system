param(
  [string]$RunId = "<RUN_ID>",
  [string]$Bucket = "TradingSystemDataset44",
  [string]$RemotePrefix = "ds24/vast_runs/queue=DS24_VAST_REVERSE_NINE_FAMILY_R1/run=<RUN_ID>",
  [string]$Destination = "C:\Users\Brandon\trading_system\docs\dream_system\components\DS-24_independent_five_minute_selector\stage_outputs\ds24_p8_r14_e3g_c2_20260824T000000Z\r7_r51_vast_full_node_gpu_utilisation_live_launch_r1\dell_imports",
  [int]$PollSeconds = 300,
  [switch]$Once
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $RunId -or $RunId -eq "<RUN_ID>") { throw "Set -RunId to the Vast DS24_RUN_ID." }
$remote = "b2:$Bucket/$($RemotePrefix -replace '<RUN_ID>', $RunId)"
$dest = Join-Path $Destination "run=$RunId"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$include = @(
  "--include", "metrics_only_v3/**",
  "--include", "ensemble_oof_scores_v2/**",
  "--include", "checkpoints/**",
  "--include", "models/**",
  "--include", "manifests/**",
  "--include", "queue_state/**",
  "--include", "publisher/**",
  "--include", "telemetry/**",
  "--include", "COMMITTED.json",
  "--include", "vast_output_manifest.json",
  "--exclude", "*full_prediction*",
  "--exclude", "*prediction_partitions*",
  "--exclude", "*holdout*",
  "--exclude", "*paper_order*",
  "--exclude", "*live_order*",
  "--exclude", ".env",
  "--exclude", "rclone.conf"
)
do {
  Write-Host "[$(Get-Date -Format o)] retrieving $remote -> $dest"
  & rclone copy $remote $dest @include --transfers 8 --checkers 16 --retries 20 --low-level-retries 50 --stats 30s
  python -m core.research.ml.ds24.vast_gpu_live_launch_r1 verify-repatriation --root $dest
  if ($Once) { break }
  Start-Sleep -Seconds $PollSeconds
} while ($true)
