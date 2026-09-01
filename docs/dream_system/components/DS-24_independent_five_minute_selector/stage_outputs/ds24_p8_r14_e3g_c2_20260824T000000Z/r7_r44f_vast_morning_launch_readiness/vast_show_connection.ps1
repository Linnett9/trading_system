param([Parameter(Mandatory=$true)][string]$InstanceId, [switch]$Execute)
$ErrorActionPreference = "Stop"
if ($InstanceId -notmatch '^[1-9][0-9]*$') { throw "InstanceId must be numeric." }
if (-not $Execute) { Write-Host "[DRY RUN] vastai ssh-url $InstanceId"; exit 0 }
vastai ssh-url $InstanceId | Tee-Object -FilePath "r44f_ssh_url.txt"
vastai show instance $InstanceId --raw | Tee-Object -FilePath "r44f_instance_snapshot.json"
