$ErrorActionPreference = "Stop"

$Root = "C:\Users\Brandon\trading_system"
$Python = "C:\Users\Brandon\AppData\Local\Programs\Python\Python311\python.exe"
$RunId = "ds24_p8_r14_e3g_c2_r7_r27_20260826T000000Z"
$LeasePath = Join-Path $Root "docs\dream_system\components\DS-24_independent_five_minute_selector\stage_outputs\ds24_p8_r14_e3g_c2_20260824T000000Z\R7_R27_tournament_supervisor.lease.json"
$SupervisorScript = "ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py"

Set-Location -LiteralPath $Root

if (Test-Path -LiteralPath $LeasePath) {
  $lease = Get-Content -LiteralPath $LeasePath -Raw | ConvertFrom-Json
  $ownerPid = [int]($lease.pid)
  if ($ownerPid -gt 0) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue
    if ($owner) {
      $commandLine = [string]$owner.CommandLine
      $commandOk = (
        $commandLine.Contains($SupervisorScript) -and
        $commandLine.Contains("--daemon") -and
        $commandLine.Contains("--ready-family-queue-manifest") -and
        $commandLine.Contains("--cross-host-ownership-manifest") -and
        $commandLine.Contains("--retired-families-path") -and
        -not $commandLine.Contains("--admit-crashed-recoverable") -and
        -not $commandLine.Contains("--family-queue")
      )
      $hostOk = ([string]$lease.hostname) -eq $env:COMPUTERNAME
      $runOk = ([string]$lease.run_id) -eq $RunId
      $creationOk = $true
      if ($lease.process_creation_time) {
        $creationOk = $false
        try {
          $leaseCreated = [DateTimeOffset]::Parse([string]$lease.process_creation_time).UtcDateTime
          $processCreated = $owner.CreationDate.ToUniversalTime()
          $creationOk = [Math]::Abs(($leaseCreated - $processCreated).TotalSeconds) -lt 2
        } catch {
          $creationOk = $false
        }
      }
      if ($commandOk -and $hostOk -and $runOk -and $creationOk) {
        Write-Output "SUPERVISOR_ALREADY_RUNNING"
        exit 0
      }
      Write-Error "SUPERVISOR_IDENTITY_CONFLICT pid=$ownerPid commandOk=$commandOk hostOk=$hostOk runOk=$runOk creationOk=$creationOk"
      exit 20
    }
    Write-Output "STALE_SUPERVISOR_LEASE_RECOVERABLE pid=$ownerPid"
  }
}

& $Python `
  "scripts\local\ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py" `
  "--daemon" `
  "--resume" `
  "--poll-seconds" `
  "20" `
  "--max-active-model-processes" `
  "3" `
  "--max-policy-workers" `
  "3" `
  "--max-restarts-per-family" `
  "2" `
  "--admission-commit-percent" `
  "92" `
  "--max-system-commit-percent" `
  "95" `
  "--min-available-ram-gb" `
  "6" `
  "--evaluation-version" `
  "v3" `
  "--metrics-root-name" `
  "metrics_only_v3" `
  "--refit-policy" `
  "daily_session_v1" `
  "--ready-family-queue-manifest" `
  "C:\Users\Brandon\trading_system\docs\dream_system\components\DS-24_independent_five_minute_selector\stage_outputs\ds24_p8_r14_e3g_c2_20260824T000000Z\R42_ready_family_queue.json" `
  "--cross-host-ownership-manifest" `
  "C:\Users\Brandon\trading_system\docs\dream_system\components\DS-24_independent_five_minute_selector\stage_outputs\ds24_p8_r14_e3g_c2_20260824T000000Z\R44_cross_host_family_ownership.json" `
  "--retired-families-path" `
  "C:\Users\Brandon\trading_system\docs\dream_system\components\DS-24_independent_five_minute_selector\stage_outputs\ds24_p8_r14_e3g_c2_20260824T000000Z\R54_retired_families.json"
