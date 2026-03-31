[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$SkipEnvSync,
    [switch]$SkipEvalExport,
    [switch]$SkipCompareModels,
    [switch]$SkipBenchmarkReport,
    [switch]$AllowExistingOutputs,
    [switch]$DisableAmp,
    [switch]$DisableTf32,
    [string]$PythonExe = "",
    [string]$Namespace = "outputs/rerun_2026q1",
    [string]$ExportRoot = "outputs/exports/rerun_2026q1",
    [string]$ComparisonRoot = "outputs/rerun_2026q1_support/comparisons",
    [string]$RegistryCsv = "results/rerun_2026q1_registry.csv",
    [string]$RegistryMd = "results/rerun_2026q1_registry.md",
    [string]$MasterPrefix = "results/rerun_2026q1_master",
    [string]$DocsReport = "docs/reports/rerun_2026q1_technical_report.md",
    [string]$ArticleTable = "Article/sections/rerun_2026q1_benchmark_table.tex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "RERUN 2026Q1 OFFICIAL BATTERY" -ForegroundColor Cyan
    Write-Host "3 datasets x 2 tasks x 3 models = 18 runs" -ForegroundColor DarkCyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("[ " + $Title + " ]") -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host ("  - " + $Message) -ForegroundColor Gray
}

function Write-Success {
    param([string]$Message)
    Write-Host ("  OK  " + $Message) -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host ("  WARN " + $Message) -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host ("  FAIL " + $Message) -ForegroundColor Red
}

function Format-Command {
    param([string[]]$Command)
    return ($Command | ForEach-Object {
        if ($_ -match "\s") {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    }) -join " "
}

function Resolve-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Resolve-BatteryPath {
    param(
        [string]$RepoRoot,
        [string]$RelativePath
    )
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RelativePath))
}

function Resolve-PythonCommand {
    param(
        [string]$RepoRoot,
        [string]$Requested
    )

    if ($Requested) {
        return $Requested
    }

    $candidates = @(
        (Join-Path $RepoRoot ".venv-win\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\bin\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        return "py"
    }

    throw "Nenhum interpretador Python encontrado. Informe -PythonExe ou crie .venv."
}

function Test-UVAvailable {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    return $null -ne $uvCommand
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Command,
        [string]$WorkingDirectory,
        [switch]$PreviewOnly
    )

    Write-Host ""
    Write-Host $Label -ForegroundColor Cyan
    Write-Host ("  cwd: " + $WorkingDirectory) -ForegroundColor DarkGray
    Write-Host ("  cmd: " + (Format-Command -Command $Command)) -ForegroundColor DarkGray

    if ($PreviewOnly) {
        Write-Host "  preview only: comando nao executado." -ForegroundColor DarkYellow
        return
    }

    Push-Location $WorkingDirectory
    try {
        & $Command[0] @($Command[1..($Command.Length - 1)])
        if ($LASTEXITCODE -ne 0) {
            throw "Comando falhou com exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

function New-RunSpec {
    param(
        [string]$Dataset,
        [string]$Task,
        [string]$Arch,
        [string]$SplitMode,
        [int]$ImgSize,
        [int]$BatchSize,
        [int]$Epochs,
        [string]$Lr,
        [string]$BackboneLr,
        [int]$WarmupEpochs,
        [int]$EarlyStopPatience
    )

    return [PSCustomObject]@{
        dataset = $Dataset
        task = $Task
        arch = $Arch
        split_mode = $SplitMode
        img_size = $ImgSize
        batch_size = $BatchSize
        epochs = $Epochs
        lr = $Lr
        backbone_lr = $BackboneLr
        warmup_epochs = $WarmupEpochs
        early_stop_patience = $EarlyStopPatience
    }
}

function Get-RunSpecs {
    $datasets = @(
        @{ name = "archive"; split = "patient" },
        @{ name = "mamografias"; split = "random" },
        @{ name = "patches_completo"; split = "random" }
    )
    $tasks = @("density", "binary")
    $architectures = @(
        @{ name = "efficientnet_b0"; img_size = 512; batch_size = 16; epochs = 30; lr = "1e-4"; backbone_lr = "1e-5"; warmup_epochs = 2; early_stop_patience = 5 },
        @{ name = "resnet50"; img_size = 512; batch_size = 16; epochs = 30; lr = "1e-4"; backbone_lr = "1e-5"; warmup_epochs = 2; early_stop_patience = 5 },
        @{ name = "vit_b_16"; img_size = 224; batch_size = 8; epochs = 30; lr = "1e-3"; backbone_lr = "1e-4"; warmup_epochs = 3; early_stop_patience = 10 }
    )

    $specs = New-Object System.Collections.Generic.List[object]
    foreach ($dataset in $datasets) {
        foreach ($task in $tasks) {
            foreach ($arch in $architectures) {
                $specs.Add((New-RunSpec `
                    -Dataset $dataset.name `
                    -Task $task `
                    -Arch $arch.name `
                    -SplitMode $dataset.split `
                    -ImgSize $arch.img_size `
                    -BatchSize $arch.batch_size `
                    -Epochs $arch.epochs `
                    -Lr $arch.lr `
                    -BackboneLr $arch.backbone_lr `
                    -WarmupEpochs $arch.warmup_epochs `
                    -EarlyStopPatience $arch.early_stop_patience))
            }
        }
    }
    return $specs
}

function Get-RunName {
    param([pscustomobject]$Spec)
    return "{0}_{1}_{2}_seed42" -f $Spec.dataset, $Spec.task, $Spec.arch
}

function Get-SeedOutdir {
    param(
        [string]$RepoRoot,
        [string]$NamespaceRoot,
        [pscustomobject]$Spec
    )
    $relative = Join-Path $NamespaceRoot (Join-Path $Spec.dataset (Join-Path $Spec.task (Join-Path $Spec.arch "seed42")))
    return Resolve-BatteryPath -RepoRoot $RepoRoot -RelativePath $relative
}

function Get-ResultsDir {
    param(
        [string]$RepoRoot,
        [string]$NamespaceRoot,
        [pscustomobject]$Spec
    )
    return Join-Path (Get-SeedOutdir -RepoRoot $RepoRoot -NamespaceRoot $NamespaceRoot -Spec $Spec) "results"
}

function Get-RunTaskClassMode {
    param([pscustomobject]$Spec)

    if ($Spec.task -eq "density") {
        return "multiclass"
    }
    return "binary"
}

function Test-CompletedResultsDir {
    param([string]$ResultsDir)

    $summaryPath = Join-Path $ResultsDir "summary.json"
    $trainHistoryPath = Join-Path $ResultsDir "train_history.csv"
    $checkpointPath = Join-Path $ResultsDir "checkpoint.pt"
    $valMetricsPath = Join-Path (Join-Path $ResultsDir "metrics") "val_metrics.json"

    if (-not (Test-Path $summaryPath) -or -not (Test-Path $trainHistoryPath) -or -not (Test-Path $checkpointPath) -or -not (Test-Path $valMetricsPath)) {
        return $false
    }

    try {
        $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }

    $finishedAtProp = $summary.PSObject.Properties["finished_at"]
    if ($null -eq $finishedAtProp) {
        return $false
    }

    if ([string]::IsNullOrWhiteSpace([string]$finishedAtProp.Value)) {
        return $false
    }

    $testFracProp = $summary.PSObject.Properties["test_frac"]
    $testFrac = 0.0
    if ($null -ne $testFracProp) {
        [double]::TryParse([string]$testFracProp.Value, [ref]$testFrac) | Out-Null
    }
    if ($testFrac -gt 0.0) {
        $testMetricsPath = Join-Path (Join-Path $ResultsDir "metrics") "test_metrics.json"
        if (-not (Test-Path $testMetricsPath)) {
            return $false
        }
    }

    return $true
}

function Resolve-RunResultsDir {
    param(
        [string]$SeedOutdir,
        [pscustomobject]$Spec
    )

    $expectedResultsDir = Join-Path $SeedOutdir "results"
    if (Test-CompletedResultsDir -ResultsDir $expectedResultsDir) {
        return $expectedResultsDir
    }

    $parentDir = Split-Path -Path $SeedOutdir -Parent
    if (-not (Test-Path $parentDir)) {
        return $expectedResultsDir
    }

    $seedLeaf = Split-Path -Path $SeedOutdir -Leaf
    $targetRunName = Get-RunName -Spec $Spec
    $targetClassMode = Get-RunTaskClassMode -Spec $Spec
    $candidates = New-Object System.Collections.Generic.List[object]

    foreach ($candidateDir in (Get-ChildItem -Path $parentDir -Directory | Where-Object { $_.Name -like ($seedLeaf + "*") })) {
        $candidateResultsDir = Join-Path $candidateDir.FullName "results"
        $candidateSummaryPath = Join-Path $candidateResultsDir "summary.json"
        if (-not (Test-CompletedResultsDir -ResultsDir $candidateResultsDir)) {
            continue
        }

        try {
            $summary = Get-Content -Path $candidateSummaryPath -Raw | ConvertFrom-Json
        }
        catch {
            continue
        }

        if ($summary.dataset -ne $Spec.dataset) {
            continue
        }
        if ($summary.arch -ne $Spec.arch) {
            continue
        }
        if ([string]$summary.classes -ne $targetClassMode) {
            continue
        }
        if ([string]$summary.tracker_run_name -and ([string]$summary.tracker_run_name -ne $targetRunName)) {
            continue
        }

        $summaryItem = Get-Item $candidateSummaryPath
        $candidates.Add([PSCustomObject]@{
            results_dir = $candidateResultsDir
            last_write  = $summaryItem.LastWriteTimeUtc
        })
    }

    if ($candidates.Count -gt 0) {
        return ($candidates | Sort-Object last_write -Descending | Select-Object -First 1).results_dir
    }

    return $expectedResultsDir
}

function Get-EvalExportManifestPath {
    param(
        [string]$RepoRoot,
        [string]$ResultsDir,
        [string]$ExportRootPath
    )

    $fullRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
    $fullResultsDir = [System.IO.Path]::GetFullPath($ResultsDir)

    if ($fullResultsDir.StartsWith($fullRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $fullResultsDir.Substring($fullRepoRoot.Length).TrimStart("\", "/")
        if ($relative) {
            $parts = $relative -split "[\\/]"
            if (($parts.Length -gt 1) -and ($parts[0] -eq "outputs")) {
                $relative = [System.IO.Path]::Combine($parts[1..($parts.Length - 1)])
            }
        }
        if ($relative) {
            return Join-Path (Join-Path $ExportRootPath $relative) "eval_export_manifest.json"
        }
    }

    return Join-Path (Join-Path $ExportRootPath (Split-Path -Path $fullResultsDir -Leaf)) "eval_export_manifest.json"
}

function Get-TrainCommand {
    param(
        [string]$PythonCmd,
        [string]$SeedOutdir,
        [string]$RegistryCsvPath,
        [string]$RegistryMdPath,
        [pscustomobject]$Spec,
        [bool]$EnableAmp,
        [bool]$EnableTf32
    )

    $command = New-Object System.Collections.Generic.List[string]
    foreach ($token in @(
        $PythonCmd,
        "-m", "mammography.commands.train",
        "--dataset", $Spec.dataset,
        "--task", $Spec.task,
        "--arch", $Spec.arch,
        "--outdir", $SeedOutdir,
        "--seed", "42",
        "--subset", "0",
        "--deterministic",
        "--pretrained",
        "--train-backbone",
        "--unfreeze-last-block",
        "--augment",
        "--class-weights", "auto",
        "--sampler-weighted",
        "--test-frac", "0.1",
        "--split-mode", $Spec.split_mode,
        "--img-size", [string]$Spec.img_size,
        "--batch-size", [string]$Spec.batch_size,
        "--epochs", [string]$Spec.epochs,
        "--lr", $Spec.lr,
        "--backbone-lr", $Spec.backbone_lr,
        "--warmup-epochs", [string]$Spec.warmup_epochs,
        "--early-stop-patience", [string]$Spec.early_stop_patience,
        "--save-val-preds",
        "--export-val-embeddings",
        "--tracker", "local",
        "--tracker-run-name", (Get-RunName -Spec $Spec),
        "--registry-csv", $RegistryCsvPath,
        "--registry-md", $RegistryMdPath,
        "--log-level", "info"
    )) {
        $command.Add([string]$token)
    }

    if ($EnableAmp) {
        $command.Add("--amp")
    }
    if ($EnableTf32) {
        $command.Add("--allow-tf32")
    }
    else {
        $command.Add("--no-allow-tf32")
    }

    return $command.ToArray()
}

function Get-EvalExportCommand {
    param(
        [string]$PythonCmd,
        [string]$ResultsDir,
        [string]$ExportRootPath,
        [pscustomobject]$Spec
    )

    return @(
        $PythonCmd,
        "-m", "mammography.commands.eval_export",
        "--run", $ResultsDir,
        "--output-dir", $ExportRootPath,
        "--run-name", (Get-RunName -Spec $Spec),
        "--no-mlflow",
        "--no-registry",
        "--log-level", "INFO"
    )
}

function Get-CompareCommand {
    param(
        [string]$PythonCmd,
        [string]$OutputDir,
        [System.Collections.Generic.List[string]]$RunDirs
    )

    $command = New-Object System.Collections.Generic.List[string]
    $command.Add($PythonCmd)
    $command.Add("-m")
    $command.Add("mammography.commands.compare_models")
    foreach ($runDir in $RunDirs) {
        $command.Add("--run")
        $command.Add($runDir)
    }
    $command.Add("--outdir")
    $command.Add($OutputDir)
    $command.Add("--export")
    $command.Add("csv,json,md,tex")
    $command.Add("--no-visualizations")
    $command.Add("--log-level")
    $command.Add("info")
    return $command.ToArray()
}

function Get-BenchmarkReportCommand {
    param(
        [string]$PythonCmd,
        [string]$NamespacePath,
        [string]$MasterPrefixPath,
        [string]$DocsReportPath,
        [string]$ArticleTablePath,
        [string]$ExportRootPath
    )

    return @(
        $PythonCmd,
        "-m", "mammography.commands.benchmark_report",
        "--namespace", $NamespacePath,
        "--output-prefix", $MasterPrefixPath,
        "--docs-report", $DocsReportPath,
        "--article-table", $ArticleTablePath,
        "--exports-search-root", $ExportRootPath,
        "--log-level", "INFO"
    )
}

$repoRoot = Resolve-RepoRoot
$namespacePath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $Namespace
$exportRootPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $ExportRoot
$comparisonRootPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $ComparisonRoot
$registryCsvPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $RegistryCsv
$registryMdPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $RegistryMd
$masterPrefixPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $MasterPrefix
$docsReportPath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $DocsReport
$articleTablePath = Resolve-BatteryPath -RepoRoot $repoRoot -RelativePath $ArticleTable
$pythonCmd = Resolve-PythonCommand -RepoRoot $repoRoot -Requested $PythonExe

$currentPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($currentPythonPath)) {
    $env:PYTHONPATH = (Join-Path $repoRoot "src")
}
else {
    $env:PYTHONPATH = ((Join-Path $repoRoot "src") + [System.IO.Path]::PathSeparator + $currentPythonPath)
}
$runSpecs = Get-RunSpecs
$totalRuns = $runSpecs.Count
$resolvedRunDirs = @{}
$enableAmp = -not $DisableAmp
$enableTf32 = -not $DisableTf32

Write-Banner
Write-Section "Mode"
if ($Execute) {
    Write-Info "Execution mode: ENABLED"
    Write-Info "This script will run the 18 official trainings."
}
else {
    Write-Info "Execution mode: PREVIEW ONLY"
    Write-Info "Nothing will be executed. Use -Execute to start the battery."
}

Write-Section "Paths"
Write-Info ("Repo root       : " + $repoRoot)
Write-Info ("Python          : " + $pythonCmd)
Write-Info ("AMP             : " + ($(if ($enableAmp) { "enabled" } else { "disabled" })))
Write-Info ("TF32            : " + ($(if ($enableTf32) { "enabled" } else { "disabled" })))
Write-Info ("Namespace       : " + $namespacePath)
Write-Info ("Export root     : " + $exportRootPath)
Write-Info ("Comparison root : " + $comparisonRootPath)
Write-Info ("Registry CSV    : " + $registryCsvPath)
Write-Info ("Registry MD     : " + $registryMdPath)

Write-Section "Run Matrix"
$runSpecs |
    Select-Object dataset, task, split_mode, arch, img_size, batch_size, epochs |
    Format-Table -AutoSize |
    Out-String |
    ForEach-Object { Write-Host $_ }

$hardCollisionPaths = @(
    $namespacePath,
    $registryCsvPath,
    $registryMdPath,
    ($masterPrefixPath + ".csv"),
    ($masterPrefixPath + ".md"),
    ($masterPrefixPath + ".json"),
    ($masterPrefixPath + ".tex")
) | Select-Object -Unique

$softCollisionPaths = @(
    $docsReportPath,
    $articleTablePath,
    $exportRootPath,
    $comparisonRootPath
) | Select-Object -Unique

$existingHardCollisions = @($hardCollisionPaths | Where-Object { Test-Path $_ })
$existingSoftCollisions = @($softCollisionPaths | Where-Object { Test-Path $_ })

if (($existingHardCollisions.Count -gt 0) -or ($existingSoftCollisions.Count -gt 0)) {
    Write-Section "Preflight Warning"
    foreach ($path in $existingHardCollisions) {
        Write-Warn ("Existing battery output detected: " + $path)
    }
    foreach ($path in $existingSoftCollisions) {
        Write-Warn ("Existing overwriteable artifact detected: " + $path)
    }
    if ($Execute -and -not $AllowExistingOutputs -and $existingHardCollisions.Count -gt 0) {
        throw "Saida oficial ja existe. Limpe/mova os artefatos ou use -AllowExistingOutputs."
    }
}

Write-Section "Execution Plan"
Write-Info ("Train runs      : " + $totalRuns)
Write-Info ("Eval exports    : " + ($(if ($SkipEvalExport) { "skipped" } else { $totalRuns })))
Write-Info ("Comparisons     : " + ($(if ($SkipCompareModels) { "skipped" } else { "6" })))
Write-Info ("Benchmark report: " + ($(if ($SkipBenchmarkReport) { "skipped" } else { "enabled" })))

if (-not $SkipEnvSync) {
    if (-not (Test-UVAvailable)) {
        if ($Execute) {
            throw "uv nao encontrado. Instale uv ou rode com -SkipEnvSync."
        }
        Write-Warn "uv nao encontrado. O passo 'uv sync --frozen' falhara se voce executar sem -SkipEnvSync."
    }
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    if (-not $SkipEnvSync) {
        Invoke-Step `
            -Label "[env] uv sync --frozen" `
            -Command @("uv", "sync", "--frozen") `
            -WorkingDirectory $repoRoot `
            -PreviewOnly:(-not $Execute)
    }

    for ($index = 0; $index -lt $runSpecs.Count; $index++) {
        $spec = $runSpecs[$index]
        $runNumber = $index + 1
        $runName = Get-RunName -Spec $spec
        $seedOutdir = Get-SeedOutdir -RepoRoot $repoRoot -NamespaceRoot $Namespace -Spec $spec
        $resultsDir = Get-ResultsDir -RepoRoot $repoRoot -NamespaceRoot $Namespace -Spec $spec
        $trainCommand = Get-TrainCommand `
            -PythonCmd $pythonCmd `
            -SeedOutdir $seedOutdir `
            -RegistryCsvPath $registryCsvPath `
            -RegistryMdPath $registryMdPath `
            -Spec $spec `
            -EnableAmp:$enableAmp `
            -EnableTf32:$enableTf32

        if ($Execute -and $AllowExistingOutputs) {
            $existingResultsDir = Resolve-RunResultsDir -SeedOutdir $seedOutdir -Spec $spec
            if (Test-CompletedResultsDir -ResultsDir $existingResultsDir) {
                Write-Warn ("Existing completed run detected for " + $runName + ": " + $existingResultsDir)
                $resultsDir = $existingResultsDir
            }
            else {
                $label = "[{0:d2}/{1:d2}] train {2} | {3} | {4}" -f $runNumber, $totalRuns, $spec.dataset, $spec.task, $spec.arch
                Invoke-Step -Label $label -Command $trainCommand -WorkingDirectory $repoRoot -PreviewOnly:(-not $Execute)
                $resultsDir = Resolve-RunResultsDir -SeedOutdir $seedOutdir -Spec $spec
            }
        }
        else {
            $label = "[{0:d2}/{1:d2}] train {2} | {3} | {4}" -f $runNumber, $totalRuns, $spec.dataset, $spec.task, $spec.arch
            Invoke-Step -Label $label -Command $trainCommand -WorkingDirectory $repoRoot -PreviewOnly:(-not $Execute)
            if ($Execute) {
                $resultsDir = Resolve-RunResultsDir -SeedOutdir $seedOutdir -Spec $spec
            }
        }

        if ($Execute -and -not (Test-CompletedResultsDir -ResultsDir $resultsDir)) {
            throw ("Run incompleto ou sem artefatos finais para " + $runName + " em " + $resultsDir)
        }

        $resolvedRunDirs[$runName] = $resultsDir

        if ((-not $Execute) -or $SkipEvalExport) {
            continue
        }

        $exportManifestPath = Get-EvalExportManifestPath `
            -RepoRoot $repoRoot `
            -ResultsDir $resultsDir `
            -ExportRootPath $exportRootPath
        if ($AllowExistingOutputs -and (Test-Path $exportManifestPath)) {
            Write-Warn ("Existing eval export detected for " + $runName + ": " + $exportManifestPath)
            continue
        }

        $exportCommand = Get-EvalExportCommand `
            -PythonCmd $pythonCmd `
            -ResultsDir $resultsDir `
            -ExportRootPath $exportRootPath `
            -Spec $spec

        Invoke-Step `
            -Label ("[{0:d2}/{1:d2}] export {2}" -f $runNumber, $totalRuns, (Get-RunName -Spec $spec)) `
            -Command $exportCommand `
            -WorkingDirectory $repoRoot
    }

    if (-not $SkipCompareModels) {
        foreach ($dataset in @("archive", "mamografias", "patches_completo")) {
            foreach ($task in @("density", "binary")) {
                $runDirs = New-Object System.Collections.Generic.List[string]
                foreach ($arch in @("efficientnet_b0", "resnet50", "vit_b_16")) {
                    $runName = "{0}_{1}_{2}_seed42" -f $dataset, $task, $arch
                    if ($resolvedRunDirs.ContainsKey($runName)) {
                        $runDirs.Add([string]$resolvedRunDirs[$runName])
                    }
                    else {
                        $runDirs.Add((Join-Path $namespacePath (Join-Path $dataset (Join-Path $task (Join-Path $arch "seed42\results")))))
                    }
                }

                $compareOutdir = Join-Path $comparisonRootPath (Join-Path $dataset $task)
                $compareCommand = Get-CompareCommand `
                    -PythonCmd $pythonCmd `
                    -OutputDir $compareOutdir `
                    -RunDirs $runDirs

                Invoke-Step `
                    -Label ("[compare] {0} | {1}" -f $dataset, $task) `
                    -Command $compareCommand `
                    -WorkingDirectory $repoRoot `
                    -PreviewOnly:(-not $Execute)
            }
        }
    }

    if (-not $SkipBenchmarkReport) {
        $benchmarkCommand = Get-BenchmarkReportCommand `
            -PythonCmd $pythonCmd `
            -NamespacePath $namespacePath `
            -MasterPrefixPath $masterPrefixPath `
            -DocsReportPath $docsReportPath `
            -ArticleTablePath $articleTablePath `
            -ExportRootPath $exportRootPath

        Invoke-Step `
            -Label "[report] benchmark-report" `
            -Command $benchmarkCommand `
            -WorkingDirectory $repoRoot `
            -PreviewOnly:(-not $Execute)
    }

    Write-Section "Done"
    if ($Execute) {
        Write-Success ("Battery finished in " + $stopwatch.Elapsed.ToString())
    }
    else {
        Write-Success "Preview finished. No command was executed."
        Write-Info "To run for real, execute:"
        Write-Host ("    powershell -ExecutionPolicy Bypass -File `"" + (Join-Path $repoRoot "tools\dev\run_rerun_2026q1.ps1") + "`" -Execute") -ForegroundColor White
    }
}
catch {
    Write-Section "Aborted"
    Write-Fail $_.Exception.Message
    exit 1
}
