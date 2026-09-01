param(
  [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
  [Parameter(Mandatory=$true)][string]$ConfirmToken,
  [int]$RequestedDiskGb = 250,
  [switch]$Execute
)
$ErrorActionPreference = "Stop"
$LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $LaunchRoot
if ($ConfirmToken -ne "CREATE_EXACTLY_ONE_DS24_R44F_GUARDED_INSTANCE") { throw "Refusing create without exact R44F confirmation token." }
function Find-RepoRoot([string]$Start) {
  $dir = Resolve-Path $Start
  while ($null -ne $dir) {
    $candidate = Join-Path $dir "core\research\ml\ds24\remote_tft_r44f.py"
    if (Test-Path -LiteralPath $candidate) { return [string]$dir }
    $parent = Split-Path -Parent $dir
    if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq [string]$dir) { break }
    $dir = $parent
  }
  throw "Could not locate repository root containing core\research\ml\ds24\remote_tft_r44f.py"
}
$RepoRoot = Find-RepoRoot $LaunchRoot
$env:PYTHONPATH = $RepoRoot + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
function Invoke-PythonModule([string[]]$Arguments) {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($null -eq $python) { throw "python executable not found on PATH." }
  & $python.Source @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}
$confirmationPath = Join-Path $LaunchRoot "r44f_selected_offer_confirmation.json"
if (-not (Test-Path -LiteralPath $confirmationPath)) { throw "Run vast_select_validate_and_confirm.ps1 first." }
$confirmation = Get-Content -LiteralPath $confirmationPath -Raw | ConvertFrom-Json
if ($confirmation.status -ne "PASS" -or $confirmation.rent_allowed -ne $true) { throw "Selected offer confirmation is not PASS." }
$offerId = [string]$confirmation.offer_id
if (-not $Execute) {
  Write-Host "[DRY RUN] Would revalidate current offer $offerId and create exactly one guarded instance."
  exit 0
}
$currentOfferPath = Join-Path $LaunchRoot "r44f_offer_revalidated_before_create.json"
$currentOfferJson = (
    & vastai --raw search offers `
        "num_gpus=1 gpu_name=RTX_4090 gpu_ram>=24 cpu_ram>=64 cpu_cores_effective>=16 disk_space>=250 reliability>=0.98 direct_port_count>=1 rentable=true dph<=0.45 cuda_vers>=12.1" `
        --on-demand `
        --storage $RequestedDiskGb `
        --limit 200 `
        -o "dlperf_usd-"
) -join "`n"

if ([string]::IsNullOrWhiteSpace($currentOfferJson)) {
    throw "Fresh broad offer revalidation returned no data."
}

[IO.File]::WriteAllText(
    $currentOfferPath,
    $currentOfferJson,
    (New-Object System.Text.UTF8Encoding($false))
)
if ($LASTEXITCODE -ne 0) { throw "Offer revalidation failed before create; no instance rented." }
Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "validate-selected-offer", "--offers-json", $currentOfferPath, "--previous-offers-json", $confirmationPath, "--offer-id", $offerId, "--maximum-complete-hourly-price", "$($confirmation.complete_hourly_price_usd)", "--confirmation-token", "SELECT_DS24_R44F_OFFER_$offerId") | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_create_offer_validation.json")
$validation = Get-Content -LiteralPath (Join-Path $LaunchRoot "r44f_create_offer_validation.json") -Raw | ConvertFrom-Json
if ($validation.status -ne "PASS") { throw "Offer disappeared or changed before create; no instance rented." }
$onStart = "mkdir -p /workspace/ds24/control /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1; date -u +%FT%TZ > /workspace/ds24/control/INSTANCE_START_TIMESTAMP; nohup bash -lc 'while true; do date -u +%FT%TZ > /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/BOOT_GUARD_ALIVE; sleep 15; done' >/workspace/ds24/control/r44f_boot_guard.log 2>&1 &"
$cmd = "vastai create instance $offerId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk $RequestedDiskGb --label ds24-r44f-morning --ssh --direct --cancel-unavail --onstart-cmd `"$onStart`""
Write-Host "Creating exactly one Vast instance. No automatic destruction is configured."
Invoke-Expression $cmd | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_create_instance_response.txt")
