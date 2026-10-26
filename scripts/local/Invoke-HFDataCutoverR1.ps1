param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("Start", "Status", "Monitor", "Pause", "Resume", "Stop", "Foreground")]
    [string]$Action,
    [string]$AuditRoot = "",
    [double]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$controller = Join-Path $repositoryRoot "scripts\local\hf_data_cutover_controller.py"
if ([string]::IsNullOrWhiteSpace($AuditRoot)) {
    $AuditRoot = Join-Path $repositoryRoot "docs\audits\HF_DATA_CUTOVER_R1_20261025T154506Z"
}
$AuditRoot = (Resolve-Path -LiteralPath $AuditRoot).Path
$baseArguments = @($controller, "--repository-root", $repositoryRoot, "--audit-root", $AuditRoot)

function Start-CutoverController([bool]$RunForeground) {
    & python @baseArguments "prepare"
    $prepareExit = $LASTEXITCODE
    if ($prepareExit -eq 10) {
        Write-Output "HF-DATA-CUTOVER-R1 controller is already active."
        return
    }
    if ($prepareExit -ne 0) {
        throw "Controller start preparation failed with exit code $prepareExit"
    }
    if ($RunForeground) {
        Push-Location $repositoryRoot
        try {
            & python @baseArguments "worker"
            $workerExit = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }
        exit $workerExit
    }
    $sessionLogRoot = Join-Path $AuditRoot "controller\session_logs"
    New-Item -ItemType Directory -Force -Path $sessionLogRoot | Out-Null
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $stdout = Join-Path $sessionLogRoot ("controller_{0}.stdout.log" -f $stamp)
    $stderr = Join-Path $sessionLogRoot ("controller_{0}.stderr.log" -f $stamp)
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList (@($controller, "worker", "--repository-root", $repositoryRoot, "--audit-root", $AuditRoot)) `
        -WorkingDirectory $repositoryRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Write-Output ("HF-DATA-CUTOVER-R1 controller started pid={0}" -f $process.Id)
    Write-Output ("stdout={0}" -f $stdout)
    Write-Output ("stderr={0}" -f $stderr)
}

switch ($Action) {
    "Start" { Start-CutoverController $false }
    "Resume" { Start-CutoverController $false }
    "Foreground" { Start-CutoverController $true }
    "Status" {
        & python @baseArguments "status"
        exit $LASTEXITCODE
    }
    "Monitor" {
        & python @baseArguments "monitor" "--poll-seconds" "$PollSeconds"
        exit $LASTEXITCODE
    }
    "Pause" {
        & python @baseArguments "pause"
        exit $LASTEXITCODE
    }
    "Stop" {
        & python @baseArguments "stop"
        exit $LASTEXITCODE
    }
}
