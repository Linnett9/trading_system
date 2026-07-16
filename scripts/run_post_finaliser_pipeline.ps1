[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ControllerRunId,
    [switch]$Resume,
    [switch]$WaitForFinaliser,
    [ValidateRange(60,86400)][int]$PollSeconds = 60,
    [Parameter(Mandatory)][string]$FinaliserManifest,
    [int]$FinaliserProcessId = 0,
    [string]$ComponentInputInventory = "",
    [string]$OperationalInputsOutputRoot = "",
    [string]$EvaluationCutoff = "",
    [ValidateSet('archive_validation','selector_stage_10','input_inventory','component_publication','selector_evaluation')][string]$StopAfterPhase = "",
    [string]$StatePath = ""
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$arguments = @(
    'scripts/post_finaliser_pipeline.py',
    '--controller-run-id', $ControllerRunId,
    '--poll-seconds', "$PollSeconds",
    '--finaliser-manifest', $FinaliserManifest
)
if ($ComponentInputInventory) { $arguments += @('--component-input-inventory', $ComponentInputInventory) }
if ($OperationalInputsOutputRoot) { $arguments += @('--operational-inputs-output-root', $OperationalInputsOutputRoot) }
if ($EvaluationCutoff) { $arguments += @('--evaluation-cutoff', $EvaluationCutoff) }
if ($StopAfterPhase) { $arguments += @('--stop-after-phase', $StopAfterPhase) }
if ($Resume) { $arguments += '--resume' }
if ($WaitForFinaliser) { $arguments += '--wait-for-finaliser' }
if ($FinaliserProcessId -gt 0) { $arguments += @('--finaliser-process-id', "$FinaliserProcessId") }
if ($StatePath) { $arguments += @('--state-path', $StatePath) }
& python @arguments
exit $LASTEXITCODE
