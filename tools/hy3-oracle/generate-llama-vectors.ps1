[CmdletBinding()]
param(
    [switch] $VerifyOnly,
    [string] $LlamaCppSource = 'C:\tmp\lightbridge-llama-b10153',
    [string] $BuildDirectory,
    [string] $OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repository = 'https://github.com/ggml-org/llama.cpp.git'
$Release = 'b10153'
$Revision = 'b77d646751d01c0962bc203b6809e9d94f7d50b7'
$Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $Workspace 'target\hy3-llama-oracle-nmake'
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $Workspace 'target\hy3-oracle'
}
$OracleSource = Join-Path $PSScriptRoot 'llama-oracle.cpp'
$Executable = Join-Path $BuildDirectory 'bin\bridge-hy3-llama-oracle.exe'
$WeightsDirectory = Join-Path $OutputDirectory 'weights'
$BridgeDirectory = Join-Path $OutputDirectory 'bridge'
$ReportPath = Join-Path $OutputDirectory 'llama-hy3.json'

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Get-Sha256 {
    param([string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-GitText {
    param([string[]] $Arguments)
    $actual = & rtk proxy git -c "safe.directory=$LlamaCppSource" -C $LlamaCppSource @Arguments
    Assert-True ($LASTEXITCODE -eq 0) "failed to run git $($Arguments -join ' ')"
    return ($actual -join "`n").Trim()
}

function Assert-Pins {
    Assert-True (Test-Path -LiteralPath $LlamaCppSource -PathType Container) 'pinned llama.cpp checkout is missing'
    Assert-True ((Get-GitText @('rev-parse', 'HEAD')) -ceq $Revision) 'llama.cpp checkout is not at the pinned revision'
    Assert-True ((Get-GitText @('rev-parse', "refs/tags/$Release^{}")) -ceq $Revision) 'llama.cpp release tag does not resolve to the pinned revision'
    Assert-True ((Get-GitText @('remote', 'get-url', 'origin')) -ceq $Repository) 'llama.cpp origin is not the pinned repository'
    Assert-True ([string]::IsNullOrEmpty((Get-GitText @('status', '--porcelain=v1', '--untracked-files=all')))) 'llama.cpp checkout is dirty'
    $sourceManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'SOURCE.json') -Raw | ConvertFrom-Json
    Assert-True ($sourceManifest.llama_cpp.commit -ceq $Revision) 'SOURCE.json llama.cpp revision does not match the engine pin'
    Assert-True (($sourceManifest.llama_cpp.repository.TrimEnd('/') + '.git') -ceq $Repository) 'SOURCE.json llama.cpp repository does not match the engine pin'
    Assert-True ($sourceManifest.llama_cpp.release -ceq $Release) 'SOURCE.json llama.cpp release does not match the engine pin'
    Assert-True ($sourceManifest.llama_cpp.license -ceq 'MIT') 'SOURCE.json llama.cpp license is not MIT'
}

function Get-VcVars {
    $candidates = @(
        'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    )
    foreach ($candidate in $candidates) {
        $vcvarsall = Join-Path (Split-Path -Parent $candidate) 'vcvarsall.bat'
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
            (Test-Path -LiteralPath $vcvarsall -PathType Leaf)) {
            return $candidate
        }
    }
    throw 'a supported Visual Studio C++ toolchain was not found'
}

function Invoke-InMsvcEnvironment {
    param([string] $Command)
    $vcvars = Get-VcVars
    $wrapped = "call `"$vcvars`" >nul && $Command"
    & rtk cmd /d /c $wrapped
    Assert-True ($LASTEXITCODE -eq 0) "MSVC command failed: $Command"
}

function Assert-Report {
    Assert-True (Test-Path -LiteralPath $ReportPath -PathType Leaf) 'llama oracle report is missing'
    $report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    Assert-True ($report.format -ceq 'lightbridge-llama-hy3-oracle-v1') 'llama oracle report format mismatch'
    Assert-True ($report.llama_commit -ceq $Revision) 'llama oracle report revision mismatch'
    Assert-True ($report.provenance.local_oracle_sha256 -ceq (Get-Sha256 $OracleSource)) 'llama oracle source hash mismatch'
    Assert-True ($report.provenance.gguf_sha256 -ceq (Get-Sha256 (Join-Path $WeightsDirectory 'reduced-hy3.gguf'))) 'llama oracle GGUF hash mismatch'
    Assert-True (@($report.steps).Count -eq 2) 'llama oracle must contain two decode steps'
    foreach ($step in @($report.steps)) {
        Assert-True (@($step.selected_experts).Count -eq 2) 'llama oracle step must contain exact top-2 routing'
        Assert-True (@($step.logits).Count -eq 32) 'llama oracle step must contain the complete reduced vocabulary logits'
        Assert-True (@($step.probabilities).Count -eq 32) 'llama oracle step must contain the complete probability distribution'
    }
}

Assert-Pins
if ($VerifyOnly) {
    Assert-Report
    Write-Output 'Hy3 llama.cpp oracle verification passed: pin, source/model hashes, routing, logits, probabilities'
    exit 0
}

New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$configure = "rtk cmake -S `"$PSScriptRoot`" -B `"$BuildDirectory`" -G `"NMake Makefiles`" -DLLAMA_CPP_SOURCE=`"$LlamaCppSource`""
Invoke-InMsvcEnvironment $configure
Invoke-InMsvcEnvironment "rtk cmake --build `"$BuildDirectory`" --target bridge-hy3-llama-oracle"
Invoke-InMsvcEnvironment "rtk cmake --build `"$BuildDirectory`" --target bridge-hy3-full-model-oracle"

Push-Location $Workspace
try {
    & rtk cargo run -p bridge-test-model --example export_oracle_bundle -- $WeightsDirectory
    Assert-True ($LASTEXITCODE -eq 0) 'failed to export reduced Hy3 GGUF'
    & rtk cargo run -p bridge-test-model --example export_run_vectors -- $BridgeDirectory
    Assert-True ($LASTEXITCODE -eq 0) 'failed to export BRIDGE runtime vectors'
} finally {
    Pop-Location
}

& $Executable (Join-Path $WeightsDirectory 'reduced-hy3.gguf') $ReportPath 3 7
Assert-True ($LASTEXITCODE -eq 0) 'llama.cpp reduced-Hy3 graph oracle failed'

$report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
$report | Add-Member -NotePropertyName provenance -NotePropertyValue ([ordered] @{
    repository = 'https://github.com/ggml-org/llama.cpp.git'
    release = $Release
    commit = $Revision
    license = 'MIT'
    local_oracle_sha256 = (Get-Sha256 $OracleSource)
    gguf_sha256 = (Get-Sha256 (Join-Path $WeightsDirectory 'reduced-hy3.gguf'))
    command = 'rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/hy3-oracle/generate-llama-vectors.ps1'
})
$json = ($report | ConvertTo-Json -Depth 16) -replace "`r`n", "`n"
[IO.File]::WriteAllText($ReportPath, $json.TrimEnd() + "`n", (New-Object Text.UTF8Encoding($false, $true)))

Assert-Report
& rtk cmd /c python (Join-Path $PSScriptRoot 'verify-llama.py') --bridge $BridgeDirectory --llama $ReportPath
Assert-True ($LASTEXITCODE -eq 0) 'BRIDGE llama-Q8_K differential comparison failed'
Write-Output 'Hy3 llama.cpp oracle generation passed: normal graph, two-token KV reuse, routing, logits, probabilities'
