[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ControllerRunId,
    [switch]$Resume,
    [switch]$WaitForFinaliser,
    [ValidateRange(60,86400)][int]$PollSeconds = 60,
    [Parameter(Mandatory)][string]$FinaliserManifest,
    [int]$FinaliserProcessId = 0,
    [Parameter(Mandatory)][string]$ComponentInputRoot,
    [Parameter(Mandatory)][string]$OutcomePath,
    [Parameter(Mandatory)][string]$EvaluationCutoff,
    [string]$StatePath = ""
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$arguments = @(
    'scripts/post_finaliser_pipeline.py',
    '--controller-run-id', $ControllerRunId,
    '--poll-seconds', "$PollSeconds",
    '--finaliser-manifest', $FinaliserManifest,
    '--component-input-root', $ComponentInputRoot,
    '--outcome-path', $OutcomePath,
    '--evaluation-cutoff', $EvaluationCutoff
)
if ($Resume) { $arguments += '--resume' }
if ($WaitForFinaliser) { $arguments += '--wait-for-finaliser' }
if ($FinaliserProcessId -gt 0) { $arguments += @('--finaliser-process-id', "$FinaliserProcessId") }
if ($StatePath) { $arguments += @('--state-path', $StatePath) }
& python @arguments
exit $LASTEXITCODE
