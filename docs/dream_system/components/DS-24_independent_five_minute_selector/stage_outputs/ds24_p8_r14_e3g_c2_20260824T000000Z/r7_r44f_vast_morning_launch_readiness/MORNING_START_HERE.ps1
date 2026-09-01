param([switch]$PaidActionAcknowledged)
$ErrorActionPreference = "Stop"
$LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $LaunchRoot
Write-Host "DS24 R44F morning launch pack"
Write-Host "Step 1: .\vast_select_validate_and_confirm.ps1 -Execute"
Write-Host "Step 2: inspect r44f_selected_offer_confirmation.json"
Write-Host "Step 3: .\vast_create_guarded_instance.ps1 -ConfirmToken CREATE_EXACTLY_ONE_DS24_R44F_GUARDED_INSTANCE -Execute"
Write-Host "Step 4: upload, verify, benchmark, launch full queue, review at 90 minutes, download results"
Write-Host "This guide pauses before the first paid action. No create command is run by this script."
$typed = Read-Host "Before renting, type READY_FOR_FIRST_PAID_ACTION after reviewing the current complete hourly price"
if ($typed -ne "READY_FOR_FIRST_PAID_ACTION") {
  Write-Host "Paid create action remains blocked."
  exit 0
}
Write-Host "Paid-action acknowledgement recorded locally. Run the guarded create script only after final offer validation."
