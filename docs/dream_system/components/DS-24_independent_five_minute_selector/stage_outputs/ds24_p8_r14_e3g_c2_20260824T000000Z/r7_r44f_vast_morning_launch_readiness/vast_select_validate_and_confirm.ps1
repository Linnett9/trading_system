param(
  [double]$MaximumCompleteHourlyPrice = 0.45,
  [int]$RequestedDiskGb = 250,
  [switch]$Execute
)
$ErrorActionPreference = "Stop"
$LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $LaunchRoot
function Find-RepoRoot([string]$Start) {
  $dir = Resolve-Path $Start
  while ($null -ne $dir) {
    $candidate = Join-Path $dir "core\\research\\ml\\ds24\\remote_tft_r44f.py"
    if (Test-Path -LiteralPath $candidate) { return [string]$dir }
    $parent = Split-Path -Parent $dir
    if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq [string]$dir) { break }
    $dir = $parent
  }
  throw "Could not locate repository root containing core\\research\\ml\\ds24\\remote_tft_r44f.py"
}
$RepoRoot = Find-RepoRoot $LaunchRoot
$env:PYTHONPATH = $RepoRoot + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
function Invoke-PythonModule([string[]]$Arguments) {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($null -eq $python) { throw "python executable not found on PATH." }
  & $python.Source @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}
if (-not $Execute) {
  Write-Host "[DRY RUN] Would search current Vast offers for one RTX 4090, >=24GB VRAM, >=64GB RAM, >=16 CPU, >=250GB disk, reliability>=0.98, direct_port_count>=1, complete hourly <= $MaximumCompleteHourlyPrice."
  exit 0
}
$query = "gpu_name=RTX_4090 num_gpus=1 rentable=true direct_port_count>=1 disk_space>=$RequestedDiskGb"
$raw = & vastai search offers $query --raw
if ($LASTEXITCODE -ne 0) { throw "Vast offer search failed; stopping before selection." }
if ([string]::IsNullOrWhiteSpace($raw)) { throw "Vast offer search returned no JSON; stopping before selection." }
$offersPath = Join-Path $LaunchRoot "r44f_current_vast_offers.json"
$raw | Out-File -LiteralPath $offersPath -Encoding utf8
Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "rank-offers", "--offers-json", $offersPath, "--maximum-complete-hourly-price", "$MaximumCompleteHourlyPrice", "--requested-disk-gb", "$RequestedDiskGb") | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_ranked_offers.json")
$ranked = Get-Content -LiteralPath (Join-Path $LaunchRoot "r44f_ranked_offers.json") -Raw | ConvertFrom-Json
if ($ranked.status -ne "PASS") { throw "No acceptable current RTX 4090 Vast offer." }
$offerId = [string]$ranked.selected_offer.offer_id
$token = Read-Host "To select this current offer, type SELECT_DS24_R44F_OFFER_$offerId"
Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "validate-selected-offer", "--offers-json", $offersPath, "--offer-id", $offerId, "--maximum-complete-hourly-price", "$MaximumCompleteHourlyPrice", "--confirmation-token", $token) | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_selected_offer_confirmation.json")
