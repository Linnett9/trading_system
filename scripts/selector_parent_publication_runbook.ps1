[CmdletBinding()]
param(
    [ValidateRange(1,16)][int]$FromStage = 1,
    [ValidateRange(1,16)][int]$ThroughStage = 10,
    [switch]$Resume,
    [string]$RunId = "",
    [string]$TranscriptPath = "",
    [string]$ComponentJobId = "",
    [string]$TrainingRowsJson = "",
    [string]$PredictionRowsJson = "",
    [string]$OperationalReadinessReport = "",
    [switch]$AllowSelectorFits,
    [switch]$InitializeOnly,
    [ValidateSet('none','complete','fail')][string]$SyntheticStageOne = 'none'
)

$ErrorActionPreference = 'Stop'
if ($FromStage -gt $ThroughStage) { throw 'Invalid stage range: FromStage must not exceed ThroughStage' }
if ($ThroughStage -ge 11 -and -not $AllowSelectorFits) { throw 'Stages 11-16 require -AllowSelectorFits' }
if ($InitializeOnly -and $SyntheticStageOne -ne 'none') { throw 'InitializeOnly and SyntheticStageOne are mutually exclusive' }
if ($SyntheticStageOne -ne 'none' -and ($FromStage -ne 1 -or $ThroughStage -gt 10 -or $Resume)) { throw 'SyntheticStageOne requires a new non-resume run beginning at stage 1 and ending no later than stage 10' }
if (-not $RunId) { $RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$runRoot = Join-Path $repo "reports/ml/readiness/selector_evaluation_1c_e/runs/$RunId"
$statePath = Join-Path $runRoot 'run_state.json'
if (-not $TranscriptPath) { $TranscriptPath = Join-Path $runRoot 'transcript.txt' }

function Get-CleanSourceLineage {
    $sourceLineageText = & python -m core.research.ml.runbook_source_lineage $repo
    $sourceLineageExitCode = $LASTEXITCODE
    if (-not $sourceLineageText) { throw 'RUNBOOK_SOURCE_LINEAGE_CHECK_FAILED: no result returned' }
    $result = $sourceLineageText | ConvertFrom-Json
    if ($sourceLineageExitCode -ne 0 -or -not $result.clean_working_tree) {
        throw "RUNBOOK_SOURCE_BOUNDARY_DIRTY: $(@($result.changes) -join '; ')"
    }
    return $result
}

$sourceLineage = $null
if (-not $Resume -and -not $InitializeOnly -and $SyntheticStageOne -eq 'none') {
    $sourceLineage = Get-CleanSourceLineage
}

$stageDefinitions = @(
    @{number=1; name='canonical registry preflight'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=2; name='canonical registry publication'; mutating=$true; resumable=$true; skippable=$true; expected='complete'},
    @{number=3; name='canonical registry verification'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=4; name='daily-spine preflight'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=5; name='daily-spine publication'; mutating=$true; resumable=$true; skippable=$true; expected='READY'},
    @{number=6; name='daily-spine verification'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=7; name='selector dataset preflight'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=8; name='selector dataset rebuild'; mutating=$true; resumable=$true; skippable=$true; expected='VERIFIED'},
    @{number=9; name='selector dataset validation'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=10; name='parent gate and component readiness'; mutating=$false; resumable=$true; skippable=$false; expected='READY'},
    @{number=11; name='guarded single-component publication'; mutating=$true; resumable=$true; skippable=$false; expected='complete'},
    @{number=12; name='component inventory revalidation'; mutating=$false; resumable=$true; skippable=$false; expected='complete'},
    @{number=13; name='resume'; mutating=$false; resumable=$true; skippable=$false; expected='complete'},
    @{number=14; name='component validation'; mutating=$false; resumable=$true; skippable=$false; expected='VERIFIED_STRICT_OOS'},
    @{number=15; name='registry verification'; mutating=$false; resumable=$true; skippable=$false; expected='VERIFIED'},
    @{number=16; name='panel refreezing'; mutating=$true; resumable=$true; skippable=$true; expected='READY'}
)
$stageIO = @{
    1=@{inputs=@('registry CSV','alias CSV'); outputs=@('read-only audit result'); exit='0 verified, nonzero blocked'}
    2=@{inputs=@('verified registry CSV','verified alias CSV'); outputs=@('run-owned registry manifest'); exit='0 published, nonzero conflict'}
    3=@{inputs=@('run-owned registry manifest'); outputs=@('registry validation JSON'); exit='0 READY, nonzero blocked'}
    4=@{inputs=@('archive manifest','base artifact','enriched artifact','registry manifest'); outputs=@('run-owned spine preflight reports'); exit='0 READY, nonzero blocked'}
    5=@{inputs=@('stage-4 verified inputs'); outputs=@('versioned spine manifest','versioned feature manifest'); exit='0 published/reused, nonzero conflict'}
    6=@{inputs=@('exact stage-5 spine manifest','exact registry manifest'); outputs=@('spine validation JSON'); exit='0 READY, nonzero blocked'}
    7=@{inputs=@('exact spine, feature, registry manifests','enriched source'); outputs=@('dataset build preflight JSON'); exit='0 READY, nonzero blocked'}
    8=@{inputs=@('stage-7 verified parents'); outputs=@('immutable v2 dataset manifest'); exit='0 published/reused, nonzero conflict'}
    9=@{inputs=@('exact stage-8 dataset and parent manifests'); outputs=@('dataset validation JSON'); exit='0 READY, nonzero blocked'}
    10=@{inputs=@('exact validated dataset and parent manifests'); outputs=@('parent_gate.json','component_readiness_v2.json'); exit='0 READY/PARTIAL, nonzero BLOCKED with state preserved'}
    11=@{inputs=@('reviewed stage-10 plan','ComponentJobId','AllowSelectorFits','operational readiness'); outputs=@('one authoritative component owner'); exit='0 complete/skipped-compatible, nonzero stops immediately'}
    12=@{inputs=@('exact stage-10 parent gate and component root'); outputs=@('updated component_readiness_v2.json'); exit='0 revalidated'}
    13=@{inputs=@('current run state'); outputs=@('safe resume command'); exit='0 displayed'}
    14=@{inputs=@('run-owned component manifests'); outputs=@('lineage verification results'); exit='0 all verified, nonzero blocked'}
    15=@{inputs=@('current ML registries'); outputs=@('registry verification JSON'); exit='0 VERIFIED, nonzero blocked'}
    16=@{inputs=@('15 verified component manifests'); outputs=@('run-owned frozen panel artifacts'); exit='0 READY, nonzero blocked'}
}
foreach ($definition in $stageDefinitions) {
    $contract = $stageIO[$definition.number]
    $definition.expected_inputs = @($contract.inputs)
    $definition.expected_outputs = @($contract.outputs)
    $definition.exit_semantics = $contract.exit
}

function New-StageStateRecords([string]$sourceCommit) {
    $records = @()
    $numbers = @{}
    foreach ($definition in $stageDefinitions) {
        foreach ($requiredKey in @('number','name','mutating','resumable','skippable','expected','expected_inputs','expected_outputs','exit_semantics')) {
            if (-not $definition.ContainsKey($requiredKey)) { throw "Invalid stage definition: missing $requiredKey" }
        }
        if ($numbers.ContainsKey($definition.number)) { throw "Invalid stage definitions: duplicate stage number $($definition.number)" }
        $numbers[$definition.number] = $true
        $record = [ordered]@{}
        foreach ($key in @('number','name','mutating','resumable','skippable','expected','expected_inputs','expected_outputs','exit_semantics')) {
            $value = $definition[$key]
            if ($key -eq 'number') {
                $record.stage_number = [int]$value
            } elseif ($key -in @('expected_inputs','expected_outputs')) {
                $record[$key] = [object[]]@($value)
            } else {
                $record[$key] = $value
            }
        }
        $record.status = 'pending'
        $record.started_at = $null
        $record.completed_at = $null
        $record.exit_code = $null
        $record.error = $null
        $record.command = $null
        $record.command_history = [object[]]@()
        $record.observed_inputs = [object[]]@()
        $record.produced_outputs = [object[]]@()
        $record.reused_artifact_identities = [object[]]@()
        $record.attempt_count = 0
        $record.freshness_metadata = [ordered]@{
            run_id = $RunId
            source_commit = $sourceCommit
            parent_artifact_identities = [ordered]@{}
            stage_started_file_time_utc = $null
        }
        $records += [pscustomobject]$record
    }
    if ($records.Count -ne 16) { throw "Invalid stage definitions: expected 16 stages, found $($records.Count)" }
    return @($records)
}

function Write-State {
    $temp = "$statePath.$PID.tmp"
    $script:state | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $statePath -Force
}
function Assert-CompatibleRunState([object]$candidate) {
    $requiredFields = @(
        'stage_number','name','mutating','resumable','skippable','expected','expected_inputs','expected_outputs',
        'exit_semantics','status','started_at','completed_at','exit_code','error','command','command_history',
        'observed_inputs','produced_outputs','reused_artifact_identities','attempt_count','freshness_metadata'
    )
    if ($candidate.run_state_version -ne 'selector_parent_publication_run_state_v2' -or @($candidate.stages).Count -ne 16) {
        throw 'INCOMPATIBLE_RUN_STATE_SCHEMA: create a new RunId'
    }
    $numbers = @()
    foreach ($row in @($candidate.stages)) {
        foreach ($field in $requiredFields) {
            if ($null -eq $row.PSObject.Properties[$field]) { throw 'INCOMPATIBLE_RUN_STATE_SCHEMA: create a new RunId' }
        }
        $numbers += [int]$row.stage_number
    }
    if (($numbers -join ',') -ne ((1..16) -join ',')) { throw 'INCOMPATIBLE_RUN_STATE_SCHEMA: create a new RunId' }
}
function Stage-Row([int]$number) { return $script:state.stages | Where-Object stage_number -eq $number }
function Start-Stage([int]$number, [string]$command) {
    $row = Stage-Row $number
    if ($row.status -eq 'running') { throw "INTERRUPTED_STAGE_REQUIRES_MANUAL_REVIEW: stage $number" }
    if ($row.status -notin @('pending','failed','complete')) { throw "Invalid stage status for execution: $($row.status)" }
    $row.status = 'running'; $row.started_at = (Get-Date).ToUniversalTime().ToString('o')
    $row.completed_at = $null; $row.exit_code = $null; $row.error = $null
    $row.command = $command; $row.command_history = @($row.command_history) + $command
    $row.attempt_count = [int]$row.attempt_count + 1
    $row.freshness_metadata.stage_started_file_time_utc = (Get-Date).ToUniversalTime().ToString('o')
    Write-State
    Write-Host ("`nSTAGE {0} - {1}" -f $number, $row.name) -ForegroundColor Cyan
}
function Complete-Stage([int]$number, [int]$exitCode, [string[]]$outputs = @()) {
    $row = Stage-Row $number
    $row.status = 'complete'; $row.completed_at = (Get-Date).ToUniversalTime().ToString('o')
    $row.exit_code = $exitCode; $row.error = $null; $row.produced_outputs = [object[]]@($outputs)
    Write-State
}
function Fail-Stage([int]$number, [int]$exitCode, [string]$reason) {
    $row = Stage-Row $number
    $row.status = 'failed'; $row.completed_at = (Get-Date).ToUniversalTime().ToString('o')
    $row.exit_code = $exitCode; $row.error = $reason
    Write-State
    throw "Stage $number failed: $reason"
}
function Invoke-Checked([int]$number, [string]$command, [string[]]$outputs = @()) {
    Start-Stage $number $command
    Invoke-Expression $command
    $code = $LASTEXITCODE
    if ($code -ne 0) { Fail-Stage $number $code "command exit code $code" }
    foreach ($output in $outputs) { if (-not (Test-Path -LiteralPath $output)) { Fail-Stage $number 2 "missing expected output: $output" } }
    Complete-Stage $number 0 $outputs
}
function Require-Path([int]$number, [string]$path) {
    if (-not $path -or -not (Test-Path -LiteralPath $path)) { Fail-Stage $number 2 "required artifact missing: $path" }
}
function Can-Reuse([int]$number) {
    if (-not $Resume) { return $false }
    $row = Stage-Row $number
    if ($row.status -ne 'complete') { return $false }
    foreach ($output in @($row.produced_outputs)) { if (-not (Test-Path -LiteralPath $output)) { return $false } }
    $row.reused_artifact_identities = [object[]]@($row.produced_outputs)
    Write-State
    return $true
}
function Set-Artifact([string]$name, [string]$value) {
    $state.artifacts | Add-Member -NotePropertyName $name -NotePropertyValue $value -Force
    Write-State
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $statePath)) { throw "Resume state does not exist: $statePath" }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    Assert-CompatibleRunState $state
    if (@($state.stages | Where-Object status -eq 'running').Count -gt 0) { throw 'INTERRUPTED_STAGE_REQUIRES_MANUAL_REVIEW' }
    if (-not $InitializeOnly -and $SyntheticStageOne -eq 'none') { $sourceLineage = Get-CleanSourceLineage }
    if ($state.source_commit -ne (git rev-parse HEAD)) { throw 'Source commit changed since the selected run; start a new RunId' }
    if ($sourceLineage -and $state.source_tree_content_checksum -ne $sourceLineage.source_tree_content_checksum) { throw 'Runbook source boundary changed since the selected run; start a new RunId' }
    $state.requested_stage_range = @{from=$FromStage; through=$ThroughStage}
    $state.allow_selector_fits = [bool]$AllowSelectorFits
} else {
    if (Test-Path -LiteralPath $statePath) { throw "RunId already exists; use -Resume or choose another RunId: $RunId" }
    $sourceCommit = if ($sourceLineage) { [string]$sourceLineage.source_commit } else { git rev-parse HEAD }
    $stageStateRecords = New-StageStateRecords $sourceCommit
    $state = [ordered]@{
        run_state_version='selector_parent_publication_run_state_v2'; run_id=$RunId
        start_timestamp=(Get-Date).ToUniversalTime().ToString('o'); repository_path=$repo
        source_commit=$sourceCommit; requested_stage_range=@{from=$FromStage; through=$ThroughStage}
        clean_working_tree=if ($sourceLineage) { [bool]$sourceLineage.clean_working_tree } else { $null }
        source_tree_content_checksum=if ($sourceLineage) { [string]$sourceLineage.source_tree_content_checksum } else { $null }
        source_boundary=if ($sourceLineage) { [object[]]@($sourceLineage.source_boundary) } else { [object[]]@() }
        allow_selector_fits=[bool]$AllowSelectorFits
        stages=@($stageStateRecords)
        artifacts=@{}
    }
    New-Item -ItemType Directory -Force $runRoot | Out-Null
    Write-State
}

if ($InitializeOnly) {
    Write-Host ("INITIALIZED RUN STATE ONLY. NO STAGES EXECUTED: {0}" -f $statePath) -ForegroundColor Green
    return
}

if ($SyntheticStageOne -ne 'none') {
    $syntheticOutput = Join-Path $runRoot 'synthetic_stage_1_output.txt'
    Start-Stage 1 'synthetic read-only harness transition'
    if ($SyntheticStageOne -eq 'fail') {
        try { Fail-Stage 1 17 'synthetic stage failure' } catch { Write-Error $_; exit 17 }
    }
    Set-Content -LiteralPath $syntheticOutput -Value 'synthetic harness output; no production command executed' -Encoding UTF8
    Complete-Stage 1 0 @($syntheticOutput)
    Write-Host ("SYNTHETIC STAGE 1 COMPLETE. NO PRODUCTION COMMANDS EXECUTED: {0}" -f $statePath) -ForegroundColor Green
    return
}

$registryCsv = 'data/reference/assets/canonical_asset_registry.csv'
$aliasCsv = 'data/reference/assets/provider_symbol_aliases.csv'
$selectorConfig = 'config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml'
$registryReport = "reports/data_lineage/canonical_asset_registry_v2/run=$RunId"
$registryManifest = "$registryReport/manifest.json"
$archiveManifest = 'reports/data_lineage/canonical_daily_v2/build_manifest.json'
$baseArtifact = 'reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2/benchmark/stock_level_prediction_artifacts.parquet'
$enrichedArtifact = 'reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2/benchmark/stock_level_prediction_artifacts_enriched.parquet'
$spineRoot = "data/processed/ml/reference/canonical_daily_stock_spine_v2/run=$RunId"
$featureRoot = "data/processed/ml/features/daily_price_features_v2/run=$RunId"
$datasetRoot = "reports/ml/readiness/canonical_v2_selector_dataset_v2/run=$RunId/frozen"
$componentRoot = "reports/ml/selector_components/operational_v2/run=$RunId"
$componentLogRoot = "reports/ml/selector_components/logs/operational_v2/run=$RunId"
$freshPreflight = Join-Path $runRoot 'component_preflight_v2.json'
$parentGate = Join-Path $runRoot 'selector_parent_gate.json'
$operationalDates = Join-Path $runRoot 'selector_operational_dates.json'

Start-Transcript -LiteralPath $TranscriptPath -Append | Out-Null
try {
    foreach ($stage in $FromStage..$ThroughStage) {
        switch ($stage) {
            1 {
                $cmd = 'python -m core.research.ml.reference.canonical_assets --verify-only --registry-output "{0}" --alias-output "{1}" --parquet-output data/reference/assets/canonical_asset_registry.parquet --report-dir "{2}" --universe-path config/universes/alpaca_514_symbols.txt' -f $registryCsv, $aliasCsv, $registryReport
                Invoke-Checked 1 $cmd
            }
            2 {
                if (Can-Reuse 2) { Write-Host 'Reusing verified immutable registry publication'; break }
                $cmd = 'python -m core.research.ml.reference.canonical_assets --audit-only --registry-output "{0}" --alias-output "{1}" --parquet-output data/reference/assets/canonical_asset_registry.parquet --report-dir "{2}" --universe-path config/universes/alpaca_514_symbols.txt' -f $registryCsv, $aliasCsv, $registryReport
                Invoke-Checked 2 $cmd @($registryManifest)
                Set-Artifact 'registry_manifest' $registryManifest
            }
            3 {
                Require-Path 3 $registryManifest
                $stage3Report = "$runRoot/registry_validation.json"
                $stage3Args = @('main.py', '--mode', 'ml-registry-verify', '--config', $selectorConfig, '--artifact-manifest', $registryManifest, '--registry-run-id', $RunId, '--verification-output', $stage3Report)
                $cmd = 'python ' + (($stage3Args | ForEach-Object { '"{0}"' -f ($_ -replace '"', '\"') }) -join ' ')
                Start-Stage 3 $cmd
                & python @stage3Args
                $code = $LASTEXITCODE
                if ($code -ne 0) { Fail-Stage 3 $code "command exit code $code" }
                if (-not (Test-Path -LiteralPath $stage3Report)) { Fail-Stage 3 2 "missing expected output: $stage3Report" }
                $stage3Payload = Get-Content -LiteralPath $stage3Report -Raw | ConvertFrom-Json
                if ($stage3Payload.status -ne 'READY') { Fail-Stage 3 2 "registry verification status is not READY" }
                Complete-Stage 3 0 @($stage3Report)
            }
            4 {
                Require-Path 4 $registryManifest; Require-Path 4 $archiveManifest; Require-Path 4 $baseArtifact; Require-Path 4 $enrichedArtifact
                $cmd = 'python scripts/verify_and_register_daily_stock_spine.py --base-artifact "{0}" --enriched-artifact "{1}" --registry "{2}" --aliases "{3}" --registry-manifest "{4}" --daily-archive-manifest "{5}" --output-root "{6}" --feature-output-root "{7}" --report-root "{8}" --expected-config config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml --verify-only' -f $baseArtifact, $enrichedArtifact, $registryCsv, $aliasCsv, $registryManifest, $archiveManifest, $spineRoot, $featureRoot, "$runRoot/spine_preflight"
                Invoke-Checked 4 $cmd
            }
            5 {
                if (Can-Reuse 5) { Write-Host 'Reusing verified immutable spine publication'; break }
                $cmd = 'python scripts/verify_and_register_daily_stock_spine.py --base-artifact "{0}" --enriched-artifact "{1}" --registry "{2}" --aliases "{3}" --registry-manifest "{4}" --daily-archive-manifest "{5}" --output-root "{6}" --feature-output-root "{7}" --report-root "{8}" --expected-config config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml' -f $baseArtifact, $enrichedArtifact, $registryCsv, $aliasCsv, $registryManifest, $archiveManifest, $spineRoot, $featureRoot, "$runRoot/spine_publication"
                Start-Stage 5 $cmd; Invoke-Expression $cmd
                if ($LASTEXITCODE -ne 0) { Fail-Stage 5 $LASTEXITCODE 'daily-spine publication command failed' }
                $spines = @(Get-ChildItem -LiteralPath $spineRoot -Recurse -Filter manifest.json -File)
                $features = @(Get-ChildItem -LiteralPath $featureRoot -Recurse -Filter manifest.json -File)
                if ($spines.Count -ne 1 -or $features.Count -ne 1) { Fail-Stage 5 2 'expected exactly one run-owned spine and feature manifest' }
                Set-Artifact 'spine_manifest' $spines[0].FullName
                Set-Artifact 'feature_manifest' $features[0].FullName
                Complete-Stage 5 0 @($spines[0].FullName, $features[0].FullName)
            }
            6 {
                $spineManifest = [string]$state.artifacts.spine_manifest; Require-Path 6 $spineManifest
                $cmd = 'python main.py --mode ml-selector-spine-validate --daily-spine-manifest "{0}" --symbol-registry-manifest "{1}" --verification-output "{2}"' -f $spineManifest, $registryManifest, "$runRoot/spine_validation.json"
                Invoke-Checked 6 $cmd @("$runRoot/spine_validation.json")
            }
            7 {
                $spineManifest = [string]$state.artifacts.spine_manifest; $featureManifest = [string]$state.artifacts.feature_manifest
                Require-Path 7 $spineManifest; Require-Path 7 $featureManifest
                $cmd = 'python main.py --mode ml-selector-dataset-build-preflight --selector-source "{0}" --selector-dataset-output-root "{1}" --daily-spine-manifest "{2}" --daily-feature-manifest "{3}" --symbol-registry-manifest "{4}" --verification-output "{5}"' -f $enrichedArtifact, $datasetRoot, $spineManifest, $featureManifest, $registryManifest, "$runRoot/dataset_build_preflight.json"
                Invoke-Checked 7 $cmd @("$runRoot/dataset_build_preflight.json")
            }
            8 {
                if (Can-Reuse 8) { Write-Host 'Reusing verified immutable selector dataset'; break }
                $spineManifest = [string]$state.artifacts.spine_manifest; $featureManifest = [string]$state.artifacts.feature_manifest
                $sourceHash = (Get-FileHash $enrichedArtifact -Algorithm SHA256).Hash
                $configHash = (Get-FileHash config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml -Algorithm SHA256).Hash
                $cmd = 'python scripts/build_canonical_v2_selector_dataset.py --source "{0}" --market-root data/processed/market_data/canonical_daily_v2/full --output-root "{1}" --source-sha256 {2} --config-hash {3} --daily-spine-manifest "{4}" --daily-feature-manifest "{5}" --symbol-registry-manifest "{6}"' -f $enrichedArtifact, $datasetRoot, $sourceHash, $configHash, $spineManifest, $featureManifest, $registryManifest
                Invoke-Checked 8 $cmd @("$datasetRoot/manifest.json")
                Set-Artifact 'dataset_manifest' "$datasetRoot/manifest.json"
            }
            9 {
                $datasetManifest = [string]$state.artifacts.dataset_manifest; Require-Path 9 $datasetManifest
                $spineManifest = [string]$state.artifacts.spine_manifest; $featureManifest = [string]$state.artifacts.feature_manifest
                $cmd = 'python main.py --mode ml-selector-dataset-validate --selector-dataset-manifest "{0}" --daily-spine-manifest "{1}" --daily-feature-manifest "{2}" --symbol-registry-manifest "{3}" --verification-output "{4}"' -f $datasetManifest, $spineManifest, $featureManifest, $registryManifest, "$runRoot/dataset_validation.json"
                Invoke-Checked 9 $cmd @("$runRoot/dataset_validation.json")
            }
            10 {
                $datasetManifest = [string]$state.artifacts.dataset_manifest; Require-Path 10 $datasetManifest
                $datasetPayload = Get-Content -LiteralPath $datasetManifest -Raw | ConvertFrom-Json
                $datesPayload = [ordered]@{status='READY';publication_status='complete';validation_status='VERIFIED';selector_dataset_id=$datasetPayload.dataset_id;selector_dataset_manifest_checksum=(Get-FileHash $datasetManifest -Algorithm SHA256).Hash;row_population_checksum=$datasetPayload.row_population_checksum;available_operational_dates=@('2024-03-15','2024-09-16','2025-03-17','2025-09-15','2026-03-16')}
                $datesPayload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$operationalDates.tmp" -Encoding UTF8
                Move-Item -LiteralPath "$operationalDates.tmp" -Destination $operationalDates -Force
                $spineManifest = [string]$state.artifacts.spine_manifest; $featureManifest = [string]$state.artifacts.feature_manifest
                $gateCmd = 'python main.py --mode ml-selector-parent-gate --symbol-registry-manifest "{0}" --daily-spine-manifest "{1}" --daily-feature-manifest "{2}" --selector-dataset-manifest "{3}" --operational-dates-manifest "{4}" --approved-root "{5}" --required-operational-date 2024-03-15 --required-operational-date 2024-09-16 --required-operational-date 2025-03-17 --required-operational-date 2025-09-15 --required-operational-date 2026-03-16 --verification-output "{6}"' -f $registryManifest,$spineManifest,$featureManifest,$datasetManifest,$operationalDates,$repo,$parentGate
                $readinessCmd = 'python main.py --mode ml-selector-component-preflight --parent-gate "{0}" --selector-dataset-root "{1}" --component-output-root "{2}" --approved-component-root "{2}" --config config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml --verification-output "{3}"' -f $parentGate,$datasetRoot,$componentRoot,$freshPreflight
                Start-Stage 10 "$gateCmd; $readinessCmd"
                Invoke-Expression $gateCmd; if ($LASTEXITCODE -ne 0) { Fail-Stage 10 $LASTEXITCODE 'selector parent gate blocked' }
                Invoke-Expression $readinessCmd; $code = $LASTEXITCODE
                Require-Path 10 $parentGate; Require-Path 10 $freshPreflight
                $preflight = Get-Content -LiteralPath $freshPreflight -Raw | ConvertFrom-Json
                Set-Artifact 'component_preflight' $freshPreflight
                Set-Artifact 'parent_gate' $parentGate
                if ($code -ne 0 -or $preflight.overall_status -eq 'BLOCKED' -or $preflight.expected_component_count -ne 15) {
                    $resumeCommand = '& "{0}" -FromStage 10 -ThroughStage 10 -Resume -RunId "{1}" -TranscriptPath "{2}"' -f $PSCommandPath, $RunId, $TranscriptPath
                    Write-Host ('BLOCKED: {0}' -f ($preflight.blockers -join ', ')) -ForegroundColor Red
                    Write-Host ('Safe resume: {0}' -f $resumeCommand)
                    Fail-Stage 10 2 ('parent gate or component readiness blocked: {0}' -f $freshPreflight)
                }
                Complete-Stage 10 0 @($parentGate,$freshPreflight)
                Write-Host 'READY FOR ONE EXPLICIT COMPONENT PUBLICATION. NO CAMPAIGN HAS BEEN LAUNCHED.' -ForegroundColor Green
            }
            11 {
                if (-not $AllowSelectorFits) { Fail-Stage 11 2 'explicit -AllowSelectorFits is required' }
                if (-not $ComponentJobId -or -not $TrainingRowsJson -or -not $PredictionRowsJson) { Fail-Stage 11 2 'ComponentJobId, TrainingRowsJson, and PredictionRowsJson are required' }
                $preflightPath = [string]$state.artifacts.component_preflight; Require-Path 11 $preflightPath
                $preflight = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
                $job = @($preflight.production_plan | Where-Object job_id -eq $ComponentJobId)
                if ($job.Count -ne 1) { Fail-Stage 11 2 "expected exactly one planned job: $ComponentJobId" }
                $jobPath = Join-Path $runRoot 'selected_component_job.json'; $job[0] | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $jobPath -Encoding UTF8
                $cmd = 'python main.py --mode ml-selector-component-publish --production-plan-job "{0}" --parent-gate "{1}" --training-rows-json "{2}" --prediction-rows-json "{3}" --experiment-ledger "{4}" --verification-output "{5}"' -f $jobPath,$parentGate,$TrainingRowsJson,$PredictionRowsJson,"$runRoot/experiment_ledger.jsonl","$runRoot/component_publication.json"
                Invoke-Checked 11 $cmd @("$runRoot/component_publication.json")
            }
            12 {
                $cmd = 'python main.py --mode ml-selector-component-preflight --parent-gate "{0}" --selector-dataset-root "{1}" --component-output-root "{2}" --approved-component-root "{2}" --config config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml --verification-output "{3}"' -f $parentGate,$datasetRoot,$componentRoot,$freshPreflight
                Invoke-Checked 12 $cmd @($freshPreflight)
            }
            13 {
                $resumeFitCommand = '& "{0}" -FromStage 10 -ThroughStage 12 -Resume -RunId "{1}" -TranscriptPath "{2}" -AllowSelectorFits -ComponentJobId "{3}" -TrainingRowsJson "{4}" -PredictionRowsJson "{5}" -OperationalReadinessReport "{6}"' -f $PSCommandPath, $RunId, $TranscriptPath, $ComponentJobId, $TrainingRowsJson, $PredictionRowsJson, $OperationalReadinessReport
                Start-Stage 13 'resume guidance'; Write-Host $resumeFitCommand; Complete-Stage 13 0
            }
            14 {
                $cmd = 'Get-ChildItem "{0}" -Recurse -Filter manifest.json -File | ForEach-Object {{ python main.py --mode ml-artifact-lineage-verify --artifact-manifest $_.FullName --expected-artifact-kind BOUNDED_SELECTOR_PREDICTION --require-promotion-grade; if ($LASTEXITCODE -ne 0) {{ throw "Component validation failed: $($_.FullName)" }} }}' -f $componentRoot
                Invoke-Checked 14 $cmd
            }
            15 {
                $cmd = 'python main.py --mode ml-registry-verify --verification-output "{0}"' -f "$runRoot/ml_registry_verification.json"
                Invoke-Checked 15 $cmd @("$runRoot/ml_registry_verification.json")
            }
            16 {
                $panelRoot = "reports/ml/panels/authoritative_selector_panel_v2/run=$RunId"
                $cmd = 'python main.py --mode ml-selector-panel-resolve --selector-manifest-root "{0}" --panel-config config/selector_evaluation/selector_multi_regime_evaluation_v1.json --panel-output-root "{1}"' -f $componentRoot, $panelRoot
                Invoke-Checked 16 $cmd
            }
        }
    }
} finally {
    Stop-Transcript | Out-Null
}
