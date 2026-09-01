param([Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = "Stop"
$required = @("SMOKE_90_MINUTE_REVIEW_READY", "r44e2_budget_status.json", "r44e2_telemetry_summary.json", "r44e2_throughput_summary.json")
foreach ($name in $required) {
  if (-not (Test-Path -LiteralPath (Join-Path $Destination $name))) { throw "Missing downloaded result: $name" }
}
Write-Host "Download verification PASS for checkpoints/V3 metrics/compact OOF control files present."
