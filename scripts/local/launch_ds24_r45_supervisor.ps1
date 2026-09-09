$ErrorActionPreference = "Stop"

$Root = "C:\Users\Brandon\trading_system"
$Python = "C:\Users\Brandon\AppData\Local\Programs\Python\Python311\python.exe"

Set-Location -LiteralPath $Root

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
  "--admit-crashed-recoverable"
