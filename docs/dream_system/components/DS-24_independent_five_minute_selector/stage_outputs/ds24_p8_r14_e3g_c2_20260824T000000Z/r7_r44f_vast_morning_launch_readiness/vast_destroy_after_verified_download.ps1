param([Parameter(Mandatory=$true)][string]$InstanceId, [Parameter(Mandatory=$true)][string]$ConfirmToken, [switch]$Execute)
$ErrorActionPreference = "Stop"
if ($ConfirmToken -ne "DESTROY_DS24_R44F_AFTER_VERIFIED_DOWNLOAD") { throw "Refusing destroy until verified download and exact token." }
if ($InstanceId -notmatch '^[1-9][0-9]*$') { throw "InstanceId must be numeric." }
if (-not $Execute) { Write-Host "[DRY RUN] vastai destroy instance $InstanceId"; exit 0 }
vastai destroy instance $InstanceId
