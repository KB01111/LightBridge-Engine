[CmdletBinding()]
param(
    [switch] $VerifyOnly,
    [switch] $VerifyTamperMatrix,
    [string] $LlamaCppSource = 'C:\tmp\lightbridge-llama-b10153',
    [string] $BuildDirectory = 'C:\tmp\lightbridge-quant-oracle-build'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Revision = 'b77d646751d01c0962bc203b6809e9d94f7d50b7'
$Release = 'b10153'
$Repository = 'https://github.com/ggml-org/llama.cpp.git'
$SourceUrlPrefix = 'https://github.com/ggml-org/llama.cpp/blob'
$GenerationCommand = 'rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1'
$Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Fixtures = Join-Path $Workspace 'crates\bridge-quant-layout\tests\fixtures'
$PinManifest = Join-Path $Workspace 'vendor\upstream\llama.cpp\PINNED.toml'
$LicensePath = Join-Path $Workspace 'vendor\upstream\llama.cpp\LICENSE'
$OracleSource = Join-Path $PSScriptRoot 'oracle.cpp'
$GeneratorScript = Join-Path $PSScriptRoot 'generate-vectors.ps1'
$CMakeSource = Join-Path $PSScriptRoot 'CMakeLists.txt'
$OracleExecutable = Join-Path $BuildDirectory 'bin\bridge-quant-oracle.exe'

$ExpectedFiles = [ordered] @{
    'decode-f32.input.bin'                  = 64L
    'decode-f32.output-f32le.bin'           = 64L
    'decode-iq2-s.input.bin'                = 246L
    'decode-iq2-s.output-f32le.bin'         = 3072L
    'decode-iq3-s.input.bin'                = 330L
    'decode-iq3-s.output-f32le.bin'         = 3072L
    'decode-q4-k.input.bin'                 = 432L
    'decode-q4-k.output-f32le.bin'          = 3072L
    'decode-q5-k.input.bin'                 = 528L
    'decode-q5-k.output-f32le.bin'          = 3072L
    'dot-iq2-s-q8-k.output-f32le.bin'       = 4L
    'dot-iq3-s-q8-k.output-f32le.bin'       = 4L
    'dot-q4-k-q8-k.output-f32le.bin'        = 4L
    'dot-q5-k-q8-k.output-f32le.bin'        = 4L
    'q8-k-activations.input-f32le.bin'      = 3072L
    'q8-k-activations.output-q8-k.bin'      = 876L
}

$OracleSources = @(
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml.c'
        upstream_git_blob_oid = 'a7d1fe7d94be4bee3df47f0d710fbfdb62087d1f'
        upstream_sha256 = '84b5c2608cf70f7beacdaa67d3cc4b58d34654d57d3d50268ff4b9eb83a643e0'
        external_worktree_sha256 = '3947981fc3aafd684b57e1f41548c60d87230bfa8c7ba66b741943d9f42b60d9'
        purpose = 'F32 type-trait and bit-preserving ABI identity evidence.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-quants.c'
        upstream_git_blob_oid = '1ebc50a763f16db909de37090da38cc8c0fdde94'
        upstream_sha256 = '07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d'
        external_worktree_sha256 = 'c5829ed1b4ced3970964464eaf2c985af38008fa76c91af15f4f63ef4447ab1f'
        purpose = 'Reference dequantizers and Q8_K reference quantizer.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-quants.h'
        upstream_git_blob_oid = '75188f1af180e3592f3c94bc4077989fb817c359'
        upstream_sha256 = '28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da'
        external_worktree_sha256 = 'f7676dc8160f940034cf092c6a0b27577e3677c6153cb476bb2bc78fb5f85762'
        purpose = 'Reference dequantizer and Q8_K quantizer declarations.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-impl.h'
        upstream_git_blob_oid = '62b76abbcec9e71c860ba1a99d79b501bad26b93'
        upstream_sha256 = '2ed56e264202906d107e26d08eabb242d3107b026ebfb78096fa1e5f94bdbbb8'
        external_worktree_sha256 = '165258ef041cae44c3848f783402634717d3c97f6c3b112dd543cb1bde5c561a'
        purpose = 'Pinned FP16 conversion and low-level helper declarations.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-cpu/ggml-cpu.c'
        upstream_git_blob_oid = '491316f7491252248d6f74a60440d3efa7aa6177'
        upstream_sha256 = 'f2abcaf7f627a2d8a4b7744a7128b210dad0d147fc92cb94ce9cbaed2945e84a'
        external_worktree_sha256 = 'b5e41ad21600eab8c56dfbc871df9c9e0ba23821b8ff35533ac1ec5dc42a21f5'
        purpose = 'Dispatch and type-trait evidence; excluded from the build.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-cpu/ggml-cpu-impl.h'
        upstream_git_blob_oid = '5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4'
        upstream_sha256 = 'e7008069e3e46f1db5e3d2eaafb4ddec3c7d0ece5c0454f99c5a8e33a50f20ba'
        external_worktree_sha256 = 'e35cc31320be8ba0412e8e577e54aa419e6030a4348a84097239d9d1795ec299'
        purpose = 'Scalar CPU quantization support ABI evidence.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-cpu/quants.c'
        upstream_git_blob_oid = '5e36459f8cbc5900b375d2189414307393471a6b'
        upstream_sha256 = 'a61f1011e49d05b5f99d352b158d5b8e36cf008294bb0db309f72bdd7f1d4e35'
        external_worktree_sha256 = '459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad'
        purpose = 'Directly compiled generic scalar selected-type Q8_K dot kernels.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'ggml/src/ggml-cpu/quants.h'
        upstream_git_blob_oid = '93ea7eeffe5b00ad2c612aac49b7983c12949525'
        upstream_sha256 = '918d6755b3e601ec7bb83c7dbf1d73304490651ccbb4072b0d31a2b45df751da'
        external_worktree_sha256 = '90dfb1484e17e22eb2e7824c0d5c029e678e8c6469ca176be3962db36bc06d73'
        purpose = 'Generic scalar selected-type Q8_K dot declarations.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'src/models/hy-v3.cpp'
        upstream_git_blob_oid = '47a0beaf217f19219e3e8fb8d5c35664625d7c73'
        upstream_sha256 = 'fcc0822f3291db653a3a723614f525bf96204161517abec5a56a3f5d1d8ac6c3'
        external_worktree_sha256 = '4da02af4a1428a2461dc04b5713eb9154829f385211dec9bafecd4486c670c89'
        purpose = 'Hy3 graph and selected packed-tensor execution reference.'
    },
    [pscustomobject] [ordered] @{
        upstream_path = 'tests/test-quantize-fns.cpp'
        upstream_git_blob_oid = '9510ac14ce00805e1689a8c8b16b6dd6c329911c'
        upstream_sha256 = '851f302b1f9338f2cc259f765cc36d702f13f564ee0fb6050043c7041a55c13b'
        external_worktree_sha256 = '309817cbbcb2275746fa9ee8e43c8e7f6ab1f619e91da2490e219734fe04d892'
        purpose = 'Upstream differential quantization-test methodology.'
    }
)

$AuthenticatedBuildInputs = @(
    [pscustomobject] @{
        upstream_path = 'ggml/CMakeLists.txt'
        upstream_git_blob_oid = 'a766e49ea11d2630e7d0acd6407e80d664e3f283'
        upstream_sha256 = '608661170ba6b628728c1b5e08eb7b30b257f69a8279dbe929fa77c971363d8c'
        external_worktree_sha256 = '6c8764a3012f8ce6d4bb45ac018d25c938cc70b06319eb92cf6aeae4cc63d04e'
    },
    [pscustomobject] @{
        upstream_path = 'ggml/src/CMakeLists.txt'
        upstream_git_blob_oid = '82e9480c2f240a66d2269198d5340b4e8565da3a'
        upstream_sha256 = 'b4210d3aded4f4d217209624f981ecb5c65b1592eca7e049ff14b13c60743cb5'
        external_worktree_sha256 = '9dddcad674b00a674cfb6f0929f26b9e2c6d61a0488bf064045ecbc4c7eb189c'
    },
    [pscustomobject] @{
        upstream_path = 'ggml/src/ggml-common.h'
        upstream_git_blob_oid = '83f9118da84a6a61967a5c8a04af9893130a4e95'
        upstream_sha256 = 'af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0'
        external_worktree_sha256 = 'c7a75460f797ac4406f8af4a9e318c2f0674b7b660a1c81ce5b4c019dc29a89a'
    }
)

$DecodeDefinitions = @(
    [pscustomobject] @{ id = 'decode-f32'; type = 'F32'; block_elements = 1L; block_bytes = 4L; block_count = 16L; input = 'decode-f32.input.bin'; output = 'decode-f32.output-f32le.bin'; source = 'ggml/src/ggml.c'; function = 'GGML_TYPE_F32 type traits (ABI identity)'; lines = '662-667' },
    [pscustomobject] @{ id = 'decode-q4-k'; type = 'Q4_K'; block_elements = 256L; block_bytes = 144L; block_count = 3L; input = 'decode-q4-k.input.bin'; output = 'decode-q4-k.output-f32le.bin'; source = 'ggml/src/ggml-quants.c'; function = 'get_scale_min_k4 and dequantize_row_q4_K'; lines = '880-887,1529-1551' },
    [pscustomobject] @{ id = 'decode-q5-k'; type = 'Q5_K'; block_elements = 256L; block_bytes = 176L; block_count = 3L; input = 'decode-q5-k.input.bin'; output = 'decode-q5-k.output-f32le.bin'; source = 'ggml/src/ggml-quants.c'; function = 'get_scale_min_k4 and dequantize_row_q5_K'; lines = '880-887,1731-1756' },
    [pscustomobject] @{ id = 'decode-iq2-s'; type = 'IQ2_S'; block_elements = 256L; block_bytes = 82L; block_count = 3L; input = 'decode-iq2-s.input.bin'; output = 'decode-iq2-s.output-f32le.bin'; source = 'ggml/src/ggml-quants.c'; function = 'dequantize_row_iq2_s'; lines = '2543-2571' },
    [pscustomobject] @{ id = 'decode-iq3-s'; type = 'IQ3_S'; block_elements = 256L; block_bytes = 110L; block_count = 3L; input = 'decode-iq3-s.input.bin'; output = 'decode-iq3-s.output-f32le.bin'; source = 'ggml/src/ggml-quants.c'; function = 'dequantize_row_iq3_s'; lines = '2607-2646' }
)

$DotDefinitions = @(
    [pscustomobject] @{ id = 'dot-q4-k-q8-k'; type = 'Q4_K'; input = 'decode-q4-k.input.bin'; output = 'dot-q4-k-q8-k.output-f32le.bin'; function = 'ggml_vec_dot_q4_K_q8_K_generic'; lines = '696-769' },
    [pscustomobject] @{ id = 'dot-q5-k-q8-k'; type = 'Q5_K'; input = 'decode-q5-k.input.bin'; output = 'dot-q5-k-q8-k.output-f32le.bin'; function = 'ggml_vec_dot_q5_K_q8_K_generic'; lines = '771-849' },
    [pscustomobject] @{ id = 'dot-iq2-s-q8-k'; type = 'IQ2_S'; input = 'decode-iq2-s.input.bin'; output = 'dot-iq2-s-q8-k.output-f32le.bin'; function = 'ggml_vec_dot_iq2_s_q8_K_generic'; lines = '998-1048' },
    [pscustomobject] @{ id = 'dot-iq3-s-q8-k'; type = 'IQ3_S'; input = 'decode-iq3-s.input.bin'; output = 'dot-iq3-s-q8-k.output-f32le.bin'; function = 'ggml_vec_dot_iq3_s_q8_K_generic'; lines = '1094-1148' }
)

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Equal {
    param($Actual, $Expected, [string] $Message)
    if ($Actual -cne $Expected) {
        throw "$Message (actual='$Actual', expected='$Expected')"
    }
}

function Assert-DirectPath {
    param([string] $Path, [string] $Kind)
    $item = Get-Item -LiteralPath $Path -Force
    Assert-True (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "$Path must not be a reparse point"
    if ($Kind -eq 'file') {
        Assert-True (-not $item.PSIsContainer) "$Path must be a file"
    } elseif ($Kind -eq 'directory') {
        Assert-True $item.PSIsContainer "$Path must be a directory"
    } else {
        throw "unknown path kind $Kind"
    }
}

function Assert-NoReparseAncestors {
    param([string] $Path)
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            Assert-True (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "reparse ancestor rejected: $current"
        }
        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent) {
            break
        }
        $current = $parent.FullName
    }
}

function Get-Sha256File {
    param([string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Sha256Bytes {
    param([byte[]] $Bytes)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $algorithm.ComputeHash($Bytes)
        return ([BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
}

function Read-StrictUtf8 {
    param([string] $Path)
    $bytes = [IO.File]::ReadAllBytes($Path)
    Assert-True ($bytes.Length -lt 3 -or -not ($bytes[0] -eq 0xef -and $bytes[1] -eq 0xbb -and $bytes[2] -eq 0xbf)) "$Path must not have a UTF-8 BOM"
    $encoding = New-Object Text.UTF8Encoding($false, $true)
    return $encoding.GetString($bytes)
}

function Invoke-GitBytes {
    param([string[]] $Arguments)
    $safe = ([IO.Path]::GetFullPath($LlamaCppSource)).Replace('\', '/')
    $all = @('-c', "safe.directory=$safe", '-C', $safe) + $Arguments
    foreach ($argument in $all) {
        Assert-True ($argument -notmatch '[\s"]') "authenticated git argument contains unsupported whitespace or quote: $argument"
    }
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = 'git.exe'
    $start.Arguments = $all -join ' '
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.EnvironmentVariables['GIT_OPTIONAL_LOCKS'] = '0'
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    Assert-True $process.Start() 'failed to start authenticated git'
    $memory = New-Object IO.MemoryStream
    try {
        $process.StandardOutput.BaseStream.CopyTo($memory)
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        Assert-True ($process.ExitCode -eq 0) "authenticated git failed: $stderr"
        return ,$memory.ToArray()
    } finally {
        $memory.Dispose()
        $process.Dispose()
    }
}

function Invoke-GitText {
    param([string[]] $Arguments)
    $encoding = New-Object Text.UTF8Encoding($false, $true)
    return $encoding.GetString((Invoke-GitBytes $Arguments)).TrimEnd("`r", "`n")
}

function Get-OraclePinRecords {
    param([string] $Path = $PinManifest)
    $text = Read-StrictUtf8 $Path
    $parts = [regex]::Split($text, '(?m)^\[\[oracle_sources\]\]\s*$')
    Assert-Equal ($parts.Count - 1) 10 'PINNED.toml oracle source block count'
    $records = @()
    for ($index = 1; $index -lt $parts.Count; $index++) {
        $values = @{}
        foreach ($match in [regex]::Matches($parts[$index], '(?m)^([a-z0-9_]+) = "([^"]*)"\s*$')) {
            $values[$match.Groups[1].Value] = $match.Groups[2].Value
        }
        Assert-True ($parts[$index] -match '(?m)^external_only = true\s*$') 'oracle source must be external_only'
        $records += [pscustomobject] $values
    }
    return $records
}

function Assert-UpstreamAndPins {
    param(
        [string] $ManifestPath = $PinManifest,
        [string] $RetainedLicensePath = $LicensePath
    )
    Assert-NoReparseAncestors $LlamaCppSource
    Assert-DirectPath $LlamaCppSource 'directory'
    Assert-DirectPath $ManifestPath 'file'
    Assert-DirectPath $RetainedLicensePath 'file'
    Assert-DirectPath $OracleSource 'file'
    Assert-Equal (Get-Sha256File $RetainedLicensePath) '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d' 'retained llama.cpp LICENSE SHA-256'
    $license = Read-StrictUtf8 $RetainedLicensePath
    Assert-True $license.Contains('Copyright (c) 2023-2026 The ggml authors') 'retained LICENSE copyright text'
    Assert-True $license.Contains('Permission is hereby granted, free of charge') 'retained LICENSE permission grant'
    $pinText = Read-StrictUtf8 $ManifestPath
    Assert-True ($pinText -match '(?ms)^\[upstream\].*?^license = "MIT"\s*$') 'PINNED.toml upstream MIT license'
    Assert-True ($pinText -match '(?m)^oracle_source_count = 10\s*$') 'PINNED.toml oracle source count'

    Assert-Equal (Invoke-GitText @('rev-parse', 'HEAD')) $Revision 'checkout HEAD'
    Assert-Equal (Invoke-GitText @('rev-parse', "refs/tags/$Release^{}")) $Revision 'release tag'
    Assert-Equal (Invoke-GitText @('remote', 'get-url', 'origin')) $Repository 'origin URL'
    Assert-Equal (Invoke-GitText @('status', '--porcelain=v1', '--untracked-files=all')) '' 'checkout must be clean'
    foreach ($line in (Invoke-GitText @('ls-files', '-v')).Split("`n")) {
        Assert-True $line.TrimEnd("`r").StartsWith('H ') "checkout index flag rejected: $line"
    }

    $pinRecords = @(Get-OraclePinRecords $ManifestPath)
    $pinByPath = @{}
    foreach ($pin in $pinRecords) {
        Assert-True (-not $pinByPath.ContainsKey($pin.upstream_path)) "duplicate oracle pin $($pin.upstream_path)"
        $pinByPath[$pin.upstream_path] = $pin
    }

    foreach ($source in $OracleSources) {
        $path = $source.upstream_path
        Assert-True $pinByPath.ContainsKey($path) "missing oracle pin $path"
        $pin = $pinByPath[$path]
        Assert-Equal $pin.revision $Revision "$path pinned revision"
        Assert-Equal $pin.upstream_git_blob_oid $source.upstream_git_blob_oid "$path pinned blob OID"
        Assert-Equal $pin.upstream_sha256 $source.upstream_sha256 "$path pinned blob SHA-256"
        Assert-Equal $pin.external_worktree_sha256 $source.external_worktree_sha256 "$path pinned worktree SHA-256"
        Assert-True (-not [string]::IsNullOrWhiteSpace($pin.purpose)) "$path pin purpose"

        $external = Join-Path $LlamaCppSource ($path.Replace('/', '\'))
        Assert-DirectPath $external 'file'
        Assert-Equal (Invoke-GitText @('rev-parse', "HEAD:$path")) $source.upstream_git_blob_oid "$path commit blob OID"
        Assert-Equal (Invoke-GitText @('hash-object', $path)) $source.upstream_git_blob_oid "$path worktree blob OID"
        Assert-Equal (Get-Sha256Bytes (Invoke-GitBytes @('cat-file', 'blob', "$Revision`:$path"))) $source.upstream_sha256 "$path canonical blob SHA-256"
        Assert-Equal (Get-Sha256File $external) $source.external_worktree_sha256 "$path worktree SHA-256"
    }
    foreach ($source in $AuthenticatedBuildInputs) {
        $path = $source.upstream_path
        $external = Join-Path $LlamaCppSource ($path.Replace('/', '\'))
        Assert-DirectPath $external 'file'
        Assert-Equal (Invoke-GitText @('rev-parse', "HEAD:$path")) $source.upstream_git_blob_oid "$path build-input commit blob OID"
        Assert-Equal (Invoke-GitText @('hash-object', $path)) $source.upstream_git_blob_oid "$path build-input worktree blob OID"
        Assert-Equal (Get-Sha256Bytes (Invoke-GitBytes @('cat-file', 'blob', "$Revision`:$path"))) $source.upstream_sha256 "$path build-input canonical SHA-256"
        Assert-Equal (Get-Sha256File $external) $source.external_worktree_sha256 "$path build-input worktree SHA-256"
    }
}

function Assert-HexSha256 {
    param([string] $Value, [string] $Context)
    Assert-True ($Value -cmatch '^[0-9a-f]{64}$') "$Context must be lowercase SHA-256"
}

function Assert-Properties {
    param($Object, [string[]] $Expected, [string] $Context)
    $actual = @($Object.PSObject.Properties.Name | Sort-Object)
    $wanted = @($Expected | Sort-Object)
    Assert-Equal ($actual -join '|') ($wanted -join '|') "$Context property set"
}

function Assert-Provenance {
    param($Record, [string] $Context)
    $source = $Record.source_path
    $expected = @($OracleSources | Where-Object upstream_path -ceq $source)
    Assert-Equal $expected.Count 1 "$Context source identity"
    Assert-Equal $Record.source_url "$SourceUrlPrefix/$Revision/$source" "$Context source URL"
    Assert-Equal $Record.commit $Revision "$Context commit"
    Assert-True (-not [string]::IsNullOrWhiteSpace($Record.source_function)) "$Context source function"
    Assert-True (-not [string]::IsNullOrWhiteSpace($Record.source_lines)) "$Context source lines"
    Assert-Equal $Record.upstream_git_blob_oid $expected[0].upstream_git_blob_oid "$Context blob OID"
    Assert-Equal $Record.upstream_blob_sha256 $expected[0].upstream_sha256 "$Context blob SHA-256"
    Assert-Equal $Record.generation_command $GenerationCommand "$Context generation command"
    Assert-Equal $Record.endianness 'little' "$Context endianness"
    Assert-Equal $Record.license 'MIT' "$Context license"
    Assert-Equal $Record.local_oracle_sha256 (Get-Sha256File $OracleSource) "$Context local oracle hash"
    Assert-Equal $Record.local_generator_sha256 (Get-Sha256File $GeneratorScript) "$Context local generator hash"
    Assert-Equal $Record.local_cmake_sha256 (Get-Sha256File $CMakeSource) "$Context local CMake hash"
}

function Get-FileRecordMap {
    param($Manifest)
    $map = @{}
    foreach ($record in @($Manifest.files)) {
        Assert-True ($record.path -cmatch '^[a-z0-9][a-z0-9.-]*\.bin$') "unsafe fixture basename $($record.path)"
        $key = $record.path.ToLowerInvariant()
        Assert-True (-not $map.ContainsKey($key)) "duplicate fixture record $($record.path)"
        $map[$key] = $record
    }
    return $map
}

function Assert-FileLink {
    param($Link, [hashtable] $FileMap, [string] $Context)
    $key = ([string] $Link.path).ToLowerInvariant()
    Assert-True $FileMap.ContainsKey($key) "$Context references unknown file $($Link.path)"
    $file = $FileMap[$key]
    Assert-Equal ([int64] $Link.bytes) ([int64] $file.bytes) "$Context byte cross-link"
    Assert-Equal $Link.sha256 $file.sha256 "$Context hash cross-link"
}

function Get-SignedByteValue {
    param([byte] $Byte)
    if ($Byte -le 127) {
        return [int] $Byte
    }
    return [int] $Byte - 256
}

function Assert-FiniteF32File {
    param([string] $Path, [string] $Context)
    $bytes = [IO.File]::ReadAllBytes($Path)
    Assert-True (($bytes.Length % 4) -eq 0) "$Context F32 length"
    for ($offset = 0; $offset -lt $bytes.Length; $offset += 4) {
        $value = [BitConverter]::ToSingle($bytes, $offset)
        Assert-True (-not [single]::IsNaN($value) -and -not [single]::IsInfinity($value)) "$Context contains non-finite F32"
    }
}

function Get-Q8BlockSums {
    param([byte[]] $Bytes)
    Assert-Equal $Bytes.Length 876 'Q8_K aggregate bytes'
    $sums = @()
    for ($block = 0; $block -lt 3; $block++) {
        $base = $block * 292
        $d = [BitConverter]::ToSingle($Bytes, $base)
        Assert-True (-not [single]::IsNaN($d) -and -not [single]::IsInfinity($d)) "Q8_K block $block has non-finite d"
        for ($group = 0; $group -lt 16; $group++) {
            $sum = 0
            for ($lane = 0; $lane -lt 16; $lane++) {
                $sum += Get-SignedByteValue $Bytes[$base + 4 + $group * 16 + $lane]
            }
            $stored = [BitConverter]::ToInt16($Bytes, $base + 260 + $group * 2)
            Assert-Equal $stored $sum "Q8_K block $block group $group sum"
            $sums += [int] $sum
        }
    }
    return $sums
}

function Assert-FixtureSet {
    param([string] $Directory)
    Assert-NoReparseAncestors $Directory
    Assert-DirectPath $Directory 'directory'
    $manifestPath = Join-Path $Directory 'quant-vectors.json'
    Assert-DirectPath $manifestPath 'file'
    $text = Read-StrictUtf8 $manifestPath
    Assert-True ($text.EndsWith("`n") -and -not $text.EndsWith("`n`n")) 'manifest must have exactly one final newline'
    Assert-True (-not $text.Contains("`r")) 'manifest must use LF line endings'
    $manifest = $text | ConvertFrom-Json
    Assert-Properties $manifest @('schema_version', 'upstream', 'generator', 'source_records', 'iq_tables', 'files', 'decode_vectors', 'q8_k_vectors', 'dot_vectors') 'manifest'
    Assert-Equal ([int64] $manifest.schema_version) 1L 'manifest schema version'
    Assert-Equal $manifest.upstream.repository $Repository 'manifest repository'
    Assert-Equal $manifest.upstream.release $Release 'manifest release'
    Assert-Equal $manifest.upstream.commit $Revision 'manifest revision'
    Assert-Equal $manifest.upstream.license 'MIT' 'manifest license'
    Assert-Equal $manifest.generator.generation_command $GenerationCommand 'manifest generation command'
    Assert-Equal $manifest.generator.endianness 'little' 'manifest endianness'
    Assert-Equal $manifest.generator.local_oracle_sha256 (Get-Sha256File $OracleSource) 'manifest oracle source hash'
    Assert-Equal $manifest.generator.local_generator_sha256 (Get-Sha256File $GeneratorScript) 'manifest generator script hash'
    Assert-Equal $manifest.generator.local_cmake_sha256 (Get-Sha256File $CMakeSource) 'manifest CMake source hash'

    $entries = @(Get-ChildItem -LiteralPath $Directory -Force)
    Assert-Equal $entries.Count 17 'fixture directory entry count'
    $expectedNames = @($ExpectedFiles.Keys) + @('quant-vectors.json')
    Assert-Equal (@($entries.Name | Sort-Object) -join '|') (@($expectedNames | Sort-Object) -join '|') 'hard-coded fixture inventory'
    foreach ($entry in $entries) {
        Assert-True (-not $entry.PSIsContainer) "fixture entry must be a file: $($entry.Name)"
        Assert-True (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "fixture entry must not be a reparse point: $($entry.Name)"
    }

    $files = Get-FileRecordMap $manifest
    Assert-Equal $files.Count 16 'manifest binary record count'
    foreach ($expected in $ExpectedFiles.GetEnumerator()) {
        $key = $expected.Key.ToLowerInvariant()
        Assert-True $files.ContainsKey($key) "missing manifest binary $($expected.Key)"
        $record = $files[$key]
        Assert-Equal ([int64] $record.bytes) ([int64] $expected.Value) "$($expected.Key) declared bytes"
        Assert-HexSha256 $record.sha256 "$($expected.Key) SHA-256"
        Assert-True (-not [string]::IsNullOrWhiteSpace($record.role)) "$($expected.Key) role"
        $path = Join-Path $Directory $expected.Key
        Assert-Equal ([int64] (Get-Item -LiteralPath $path).Length) ([int64] $expected.Value) "$($expected.Key) actual bytes"
        Assert-Equal (Get-Sha256File $path) $record.sha256 "$($expected.Key) content hash"
    }

    $sourceRecords = @($manifest.source_records)
    Assert-Equal $sourceRecords.Count 10 'manifest oracle source count'
    foreach ($source in $OracleSources) {
        $records = @($sourceRecords | Where-Object upstream_path -ceq $source.upstream_path)
        Assert-Equal $records.Count 1 "manifest source $($source.upstream_path)"
        Assert-Equal $records[0].revision $Revision "manifest source revision $($source.upstream_path)"
        Assert-Equal $records[0].upstream_git_blob_oid $source.upstream_git_blob_oid "manifest source OID $($source.upstream_path)"
        Assert-Equal $records[0].upstream_sha256 $source.upstream_sha256 "manifest source SHA $($source.upstream_path)"
        Assert-Equal $records[0].external_worktree_sha256 $source.external_worktree_sha256 "manifest source worktree SHA $($source.upstream_path)"
    }

    $tables = @($manifest.iq_tables)
    Assert-Equal $tables.Count 3 'IQ table record count'
    $expectedTables = @{
        'kmask_iq2xs' = [pscustomobject] @{ sha256 = '5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5'; element_type = 'u8'; element_count = 8L; bytes = 8L; lines = '509-511' }
        'iq2s_grid' = [pscustomobject] @{ sha256 = 'e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88'; element_type = 'u64'; element_count = 1024L; bytes = 8192L; lines = '758-1015' }
        'iq3s_grid' = [pscustomobject] @{ sha256 = 'bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f'; element_type = 'u32'; element_count = 512L; bytes = 2048L; lines = '1052-1117' }
    }
    $common = @($AuthenticatedBuildInputs | Where-Object upstream_path -ceq 'ggml/src/ggml-common.h')[0]
    $localCommon = Join-Path $Workspace 'vendor\upstream\llama.cpp\ggml\src\ggml-common.h'
    Assert-DirectPath $localCommon 'file'
    Assert-Equal (Get-Sha256File $localCommon) $common.upstream_sha256 'local retained ggml-common.h SHA-256'
    foreach ($table in $tables) {
        Assert-True $expectedTables.ContainsKey($table.name) "unknown IQ table $($table.name)"
        $expectedTable = $expectedTables[$table.name]
        Assert-Equal $table.serialization 'little-endian integers' "$($table.name) serialization"
        Assert-Equal $table.sha256 $expectedTable.sha256 "$($table.name) content hash"
        Assert-Equal $table.element_type $expectedTable.element_type "$($table.name) element type"
        Assert-Equal ([int64] $table.element_count) $expectedTable.element_count "$($table.name) element count"
        Assert-Equal ([int64] $table.bytes) $expectedTable.bytes "$($table.name) bytes"
        Assert-Equal $table.source_url "$SourceUrlPrefix/$Revision/ggml/src/ggml-common.h" "$($table.name) source URL"
        Assert-Equal $table.commit $Revision "$($table.name) revision"
        Assert-Equal $table.source_path 'ggml/src/ggml-common.h' "$($table.name) source path"
        Assert-Equal $table.source_function $table.name "$($table.name) source function"
        Assert-Equal $table.source_lines $expectedTable.lines "$($table.name) source lines"
        Assert-Equal $table.upstream_git_blob_oid $common.upstream_git_blob_oid "$($table.name) blob OID"
        Assert-Equal $table.upstream_blob_sha256 $common.upstream_sha256 "$($table.name) blob SHA-256"
        Assert-Equal $table.external_worktree_sha256 $common.external_worktree_sha256 "$($table.name) worktree SHA-256"
        Assert-Equal $table.local_vendored_sha256 $common.upstream_sha256 "$($table.name) local vendored SHA-256"
        Assert-Equal $table.generation_command $GenerationCommand "$($table.name) generation command"
        Assert-Equal $table.endianness 'little' "$($table.name) endianness"
        Assert-Equal $table.license 'MIT' "$($table.name) license"
        Assert-Equal $table.local_oracle_sha256 (Get-Sha256File $OracleSource) "$($table.name) local oracle SHA-256"
        Assert-Equal $table.local_generator_sha256 (Get-Sha256File $GeneratorScript) "$($table.name) local generator SHA-256"
        Assert-Equal $table.local_cmake_sha256 (Get-Sha256File $CMakeSource) "$($table.name) local CMake SHA-256"
    }

    $decode = @($manifest.decode_vectors)
    Assert-Equal $decode.Count 5 'decode vector count'
    foreach ($definition in $DecodeDefinitions) {
        $records = @($decode | Where-Object id -ceq $definition.id)
        Assert-Equal $records.Count 1 "$($definition.id) record count"
        $record = $records[0]
        Assert-Provenance $record $definition.id
        Assert-Equal $record.physical_type $definition.type "$($definition.id) type"
        Assert-Equal ([int64] $record.block_elements) $definition.block_elements "$($definition.id) block elements"
        Assert-Equal ([int64] $record.block_bytes) $definition.block_bytes "$($definition.id) block bytes"
        Assert-Equal ([int64] $record.block_count) $definition.block_count "$($definition.id) block count"
        Assert-FileLink $record.input $files "$($definition.id) input"
        Assert-FileLink $record.output $files "$($definition.id) output"
        if ($definition.type -ne 'F32') {
            Assert-Equal (@($record.cases) -join '|') 'structural|lcg|zero_scale' "$($definition.id) cases"
        }
    }

    $f32Input = [IO.File]::ReadAllBytes((Join-Path $Directory 'decode-f32.input.bin'))
    $f32Output = [IO.File]::ReadAllBytes((Join-Path $Directory 'decode-f32.output-f32le.bin'))
    Assert-Equal ([Convert]::ToBase64String($f32Output)) ([Convert]::ToBase64String($f32Input)) 'F32 ABI identity'

    $q8Records = @($manifest.q8_k_vectors)
    Assert-Equal $q8Records.Count 1 'Q8_K vector count'
    $q8 = $q8Records[0]
    Assert-Provenance $q8 'Q8_K vector'
    Assert-Equal $q8.id 'q8-k-activations' 'Q8_K ID'
    Assert-Equal ([int64] $q8.block_elements) 256L 'Q8_K block elements'
    Assert-Equal ([int64] $q8.block_bytes) 292L 'Q8_K block bytes'
    Assert-Equal ([int64] $q8.block_count) 3L 'Q8_K block count'
    Assert-FileLink $q8.input $files 'Q8_K input'
    Assert-FileLink $q8.output $files 'Q8_K output'
    Assert-FiniteF32File (Join-Path $Directory $q8.input.path) 'Q8_K activation input'
    $actualSums = @(Get-Q8BlockSums ([IO.File]::ReadAllBytes((Join-Path $Directory $q8.output.path)))
    )
    Assert-Equal $actualSums.Count 48 'Q8_K block sum count'
    Assert-Equal ($actualSums -join '|') (@($q8.block_sums) -join '|') 'Q8_K manifest block sums'

    $dots = @($manifest.dot_vectors)
    Assert-Equal $dots.Count 4 'dot vector count'
    foreach ($definition in $DotDefinitions) {
        $records = @($dots | Where-Object id -ceq $definition.id)
        Assert-Equal $records.Count 1 "$($definition.id) record count"
        $record = $records[0]
        Assert-Provenance $record $definition.id
        $n = [int64] $record.n
        Assert-Equal $n 768L "$($definition.id) n"
        Assert-True ($n -gt 0 -and ($n % 256) -eq 0 -and $n -le [int]::MaxValue) "$($definition.id) n constraints"
        Assert-FileLink $record.weight_input $files "$($definition.id) weight input"
        Assert-FileLink $record.q8_input $files "$($definition.id) Q8 input"
        Assert-FileLink $record.output $files "$($definition.id) output"
        Assert-Equal ([int64] $record.weight_input.bytes) ([int64] (($n / 256) * $record.weight_block_bytes)) "$($definition.id) weight arithmetic"
        Assert-Equal ([int64] $record.q8_input.bytes) ([int64] (($n / 256) * 292)) "$($definition.id) Q8 arithmetic"
        Assert-FiniteF32File (Join-Path $Directory $record.output.path) "$($definition.id) output"
    }
}

function New-FileLink {
    param([hashtable] $FileMap, [string] $Name)
    $file = $FileMap[$Name.ToLowerInvariant()]
    return [ordered] @{ path = $file.path; bytes = [int64] $file.bytes; sha256 = $file.sha256 }
}

function New-Provenance {
    param([string] $SourcePath, [string] $Function, [string] $Lines, [string] $OracleHash)
    $source = @($OracleSources | Where-Object upstream_path -ceq $SourcePath)
    Assert-Equal $source.Count 1 "provenance source $SourcePath"
    return [ordered] @{
        source_url = "$SourceUrlPrefix/$Revision/$SourcePath"
        commit = $Revision
        source_path = $SourcePath
        source_function = $Function
        source_lines = $Lines
        upstream_git_blob_oid = $source[0].upstream_git_blob_oid
        upstream_blob_sha256 = $source[0].upstream_sha256
        generation_command = $GenerationCommand
        endianness = 'little'
        license = 'MIT'
        local_oracle_sha256 = $OracleHash
        local_generator_sha256 = Get-Sha256File $GeneratorScript
        local_cmake_sha256 = Get-Sha256File $CMakeSource
    }
}

function Add-Provenance {
    param([System.Collections.Specialized.OrderedDictionary] $Record, [System.Collections.Specialized.OrderedDictionary] $Provenance)
    foreach ($entry in $Provenance.GetEnumerator()) {
        $Record[$entry.Key] = $entry.Value
    }
    return $Record
}

function New-Manifest {
    param([string] $Directory)
    $oracleHash = Get-Sha256File $OracleSource
    $fileRecords = @()
    $roleByName = @{
        'decode-f32.input.bin' = 'F32 ABI identity input bits'
        'decode-f32.output-f32le.bin' = 'F32 ABI identity expected bits'
        'decode-q4-k.input.bin' = 'Q4_K structural, LCG, and zero-scale packed blocks'
        'decode-q4-k.output-f32le.bin' = 'Q4_K reference decoded F32LE values'
        'decode-q5-k.input.bin' = 'Q5_K structural, LCG, and zero-scale packed blocks'
        'decode-q5-k.output-f32le.bin' = 'Q5_K reference decoded F32LE values'
        'decode-iq2-s.input.bin' = 'IQ2_S structural, LCG, and zero-scale packed blocks'
        'decode-iq2-s.output-f32le.bin' = 'IQ2_S reference decoded F32LE values'
        'decode-iq3-s.input.bin' = 'IQ3_S structural, LCG, and zero-scale packed blocks'
        'decode-iq3-s.output-f32le.bin' = 'IQ3_S reference decoded F32LE values'
        'q8-k-activations.input-f32le.bin' = 'Finite structural, LCG, and zero Q8_K activation input'
        'q8-k-activations.output-q8-k.bin' = 'Reference block_q8_K activation encoding'
        'dot-q4-k-q8-k.output-f32le.bin' = 'Generic scalar Q4_K by Q8_K dot result'
        'dot-q5-k-q8-k.output-f32le.bin' = 'Generic scalar Q5_K by Q8_K dot result'
        'dot-iq2-s-q8-k.output-f32le.bin' = 'Generic scalar IQ2_S by Q8_K dot result'
        'dot-iq3-s-q8-k.output-f32le.bin' = 'Generic scalar IQ3_S by Q8_K dot result'
    }
    foreach ($expected in $ExpectedFiles.GetEnumerator()) {
        $path = Join-Path $Directory $expected.Key
        $fileRecords += [pscustomobject] [ordered] @{
            path = $expected.Key
            role = $roleByName[$expected.Key]
            bytes = [int64] (Get-Item -LiteralPath $path).Length
            sha256 = Get-Sha256File $path
        }
    }
    $fileMap = @{}
    foreach ($file in $fileRecords) {
        $fileMap[$file.path.ToLowerInvariant()] = $file
    }

    $decodeRecords = @()
    foreach ($definition in $DecodeDefinitions) {
        $record = [ordered] @{
            id = $definition.id
            physical_type = $definition.type
            block_elements = $definition.block_elements
            block_bytes = $definition.block_bytes
            block_count = $definition.block_count
            cases = if ($definition.type -eq 'F32') { @('representative_ieee754_bits') } else { @('structural', 'lcg', 'zero_scale') }
            input = New-FileLink $fileMap $definition.input
            output = New-FileLink $fileMap $definition.output
        }
        $record = Add-Provenance $record (New-Provenance $definition.source $definition.function $definition.lines $oracleHash)
        $decodeRecords += [pscustomobject] $record
    }

    $q8Bytes = [IO.File]::ReadAllBytes((Join-Path $Directory 'q8-k-activations.output-q8-k.bin'))
    $q8Record = [ordered] @{
        id = 'q8-k-activations'
        physical_type = 'Q8_K'
        block_elements = 256L
        block_bytes = 292L
        block_count = 3L
        cases = @('structural', 'lcg', 'zero')
        input = New-FileLink $fileMap 'q8-k-activations.input-f32le.bin'
        output = New-FileLink $fileMap 'q8-k-activations.output-q8-k.bin'
        block_sums = @(Get-Q8BlockSums $q8Bytes)
    }
    $q8Record = Add-Provenance $q8Record (New-Provenance 'ggml/src/ggml-quants.c' 'nearest_int and quantize_row_q8_K_ref' '621-626,2768-2805' $oracleHash)

    $dotRecords = @()
    foreach ($definition in $DotDefinitions) {
        $weightDefinition = @($DecodeDefinitions | Where-Object type -ceq $definition.type)[0]
        $record = [ordered] @{
            id = $definition.id
            weight_type = $definition.type
            n = 768L
            nrc = 1L
            strides = [ordered] @{ bs = 0L; bx = 0L; by = 0L }
            weight_block_bytes = $weightDefinition.block_bytes
            weight_input = New-FileLink $fileMap $definition.input
            q8_input = New-FileLink $fileMap 'q8-k-activations.output-q8-k.bin'
            output = New-FileLink $fileMap $definition.output
        }
        $record = Add-Provenance $record (New-Provenance 'ggml/src/ggml-cpu/quants.c' $definition.function $definition.lines $oracleHash)
        $dotRecords += [pscustomobject] $record
    }

    $sourceRecords = @()
    foreach ($source in $OracleSources) {
        $sourceRecords += [pscustomobject] [ordered] @{
            upstream_path = $source.upstream_path
            revision = $Revision
            upstream_git_blob_oid = $source.upstream_git_blob_oid
            upstream_sha256 = $source.upstream_sha256
            external_worktree_sha256 = $source.external_worktree_sha256
            purpose = $source.purpose
        }
    }

    $common = @($AuthenticatedBuildInputs | Where-Object upstream_path -ceq 'ggml/src/ggml-common.h')[0]
    $tableDefinitions = @(
        [pscustomobject] @{ name = 'kmask_iq2xs'; element_type = 'u8'; element_count = 8L; bytes = 8L; lines = '509-511'; sha256 = '5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5' },
        [pscustomobject] @{ name = 'iq2s_grid'; element_type = 'u64'; element_count = 1024L; bytes = 8192L; lines = '758-1015'; sha256 = 'e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88' },
        [pscustomobject] @{ name = 'iq3s_grid'; element_type = 'u32'; element_count = 512L; bytes = 2048L; lines = '1052-1117'; sha256 = 'bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f' }
    )
    $iqTables = @()
    foreach ($table in $tableDefinitions) {
        $iqTables += [pscustomobject] [ordered] @{
            name = $table.name
            element_type = $table.element_type
            element_count = $table.element_count
            bytes = $table.bytes
            serialization = 'little-endian integers'
            sha256 = $table.sha256
            source_url = "$SourceUrlPrefix/$Revision/ggml/src/ggml-common.h"
            commit = $Revision
            source_path = 'ggml/src/ggml-common.h'
            source_function = $table.name
            source_lines = $table.lines
            upstream_git_blob_oid = $common.upstream_git_blob_oid
            upstream_blob_sha256 = $common.upstream_sha256
            external_worktree_sha256 = $common.external_worktree_sha256
            local_vendored_sha256 = $common.upstream_sha256
            generation_command = $GenerationCommand
            endianness = 'little'
            license = 'MIT'
            local_oracle_sha256 = $oracleHash
            local_generator_sha256 = Get-Sha256File $GeneratorScript
            local_cmake_sha256 = Get-Sha256File $CMakeSource
        }
    }

    return [pscustomobject] [ordered] @{
        schema_version = 1L
        upstream = [ordered] @{ repository = $Repository; release = $Release; commit = $Revision; license = 'MIT' }
        generator = [ordered] @{
            generation_command = $GenerationCommand
            endianness = 'little'
            local_oracle_sha256 = $oracleHash
            local_generator_sha256 = Get-Sha256File $GeneratorScript
            local_cmake_sha256 = Get-Sha256File $CMakeSource
            build = 'static ggml-base plus directly compiled ggml/src/ggml-cpu/quants.c'
            configuration = 'Release'
            floating_point = 'MSVC /fp:strict on ggml-base and bridge-quant-oracle; contraction and reassociation disabled'
            cpu_dispatch = 'disabled'
        }
        source_records = $sourceRecords
        iq_tables = $iqTables
        files = $fileRecords
        decode_vectors = $decodeRecords
        q8_k_vectors = @([pscustomobject] $q8Record)
        dot_vectors = $dotRecords
    }
}

function Write-Manifest {
    param([string] $Directory)
    $manifest = New-Manifest $Directory
    $json = $manifest | ConvertTo-Json -Depth 16
    $json = ($json -replace "`r`n", "`n").TrimEnd("`r", "`n") + "`n"
    $encoding = New-Object Text.UTF8Encoding($false, $true)
    [IO.File]::WriteAllText((Join-Path $Directory 'quant-vectors.json'), $json, $encoding)
}

function Invoke-OracleProcess {
    param([string] $Executable, [string[]] $Arguments)
    foreach ($argument in $Arguments) {
        Assert-True ($argument -notmatch '[\s"]') "oracle test argument contains unsupported whitespace or quote: $argument"
    }
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $Executable
    $start.Arguments = $Arguments -join ' '
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    Assert-True $process.Start() 'failed to start oracle'
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $result = [pscustomobject] @{ ExitCode = $process.ExitCode; Stdout = $stdout; Stderr = $stderr }
    $process.Dispose()
    return $result
}

function Invoke-ExpectedFailure {
    param([string] $Executable, [string[]] $Arguments, [string] $Sentinel, [byte[]] $ExpectedSentinel, [string] $Context)
    $before = Get-Sha256File $Sentinel
    $result = Invoke-OracleProcess $Executable $Arguments
    Assert-True ($result.ExitCode -ne 0) "$Context unexpectedly succeeded"
    Assert-Equal (Get-Sha256File $Sentinel) $before "$Context sentinel hash"
    Assert-Equal ([Convert]::ToBase64String([IO.File]::ReadAllBytes($Sentinel))) ([Convert]::ToBase64String($ExpectedSentinel)) "$Context sentinel bytes"
}

function Invoke-NegativeMatrix {
    param([string] $Executable, [string] $Generated, [string] $Directory)
    $fixtureBefore = @(
        Get-ChildItem -LiteralPath $Generated -Force |
            Sort-Object Name |
            ForEach-Object { "$($_.Name)|$($_.Length)|$(Get-Sha256File $_.FullName)|$($_.LastWriteTimeUtc.Ticks)" }
    ) -join "`n"
    New-Item -ItemType Directory -Path $Directory | Out-Null
    $sentinel = Join-Path $Directory 'sentinel.bin'
    [byte[]] $sentinelBytes = 0x4c, 0x42, 0x52, 0x47, 0xa5, 0x5a
    [IO.File]::WriteAllBytes($sentinel, $sentinelBytes)
    $empty = Join-Path $Directory 'empty.bin'
    [IO.File]::WriteAllBytes($empty, [byte[]] @())

    $positiveCases = @(
        [pscustomobject] @{ Name = 'decode-f32'; Arguments = @('decode', 'f32', '16', (Join-Path $Generated 'decode-f32.input.bin')); Expected = (Join-Path $Generated 'decode-f32.output-f32le.bin') },
        [pscustomobject] @{ Name = 'decode-q4-k'; Arguments = @('decode', 'q4-k', '768', (Join-Path $Generated 'decode-q4-k.input.bin')); Expected = (Join-Path $Generated 'decode-q4-k.output-f32le.bin') },
        [pscustomobject] @{ Name = 'decode-q5-k'; Arguments = @('decode', 'q5-k', '768', (Join-Path $Generated 'decode-q5-k.input.bin')); Expected = (Join-Path $Generated 'decode-q5-k.output-f32le.bin') },
        [pscustomobject] @{ Name = 'decode-iq2-s'; Arguments = @('decode', 'iq2-s', '768', (Join-Path $Generated 'decode-iq2-s.input.bin')); Expected = (Join-Path $Generated 'decode-iq2-s.output-f32le.bin') },
        [pscustomobject] @{ Name = 'decode-iq3-s'; Arguments = @('decode', 'iq3-s', '768', (Join-Path $Generated 'decode-iq3-s.input.bin')); Expected = (Join-Path $Generated 'decode-iq3-s.output-f32le.bin') },
        [pscustomobject] @{ Name = 'q8-k'; Arguments = @('q8', '768', (Join-Path $Generated 'q8-k-activations.input-f32le.bin')); Expected = (Join-Path $Generated 'q8-k-activations.output-q8-k.bin') },
        [pscustomobject] @{ Name = 'dot-q4-k'; Arguments = @('dot', 'q4-k', '768', (Join-Path $Generated 'decode-q4-k.input.bin'), (Join-Path $Generated 'q8-k-activations.output-q8-k.bin')); Expected = (Join-Path $Generated 'dot-q4-k-q8-k.output-f32le.bin') },
        [pscustomobject] @{ Name = 'dot-q5-k'; Arguments = @('dot', 'q5-k', '768', (Join-Path $Generated 'decode-q5-k.input.bin'), (Join-Path $Generated 'q8-k-activations.output-q8-k.bin')); Expected = (Join-Path $Generated 'dot-q5-k-q8-k.output-f32le.bin') },
        [pscustomobject] @{ Name = 'dot-iq2-s'; Arguments = @('dot', 'iq2-s', '768', (Join-Path $Generated 'decode-iq2-s.input.bin'), (Join-Path $Generated 'q8-k-activations.output-q8-k.bin')); Expected = (Join-Path $Generated 'dot-iq2-s-q8-k.output-f32le.bin') },
        [pscustomobject] @{ Name = 'dot-iq3-s'; Arguments = @('dot', 'iq3-s', '768', (Join-Path $Generated 'decode-iq3-s.input.bin'), (Join-Path $Generated 'q8-k-activations.output-q8-k.bin')); Expected = (Join-Path $Generated 'dot-iq3-s-q8-k.output-f32le.bin') }
    )
    foreach ($positive in $positiveCases) {
        $output = Join-Path $Directory "$($positive.Name).positive.bin"
        $result = Invoke-OracleProcess $Executable (@($positive.Arguments) + @($output))
        Assert-Equal $result.ExitCode 0 "$($positive.Name) positive CLI result: $($result.Stderr)"
        Assert-Equal (Get-Sha256File $output) (Get-Sha256File $positive.Expected) "$($positive.Name) positive CLI hash"
    }

    function Write-Mutant {
        param([string] $Name, [byte[]] $Bytes)
        $path = Join-Path $Directory $Name
        [IO.File]::WriteAllBytes($path, $Bytes)
        return $path
    }

    $q4 = [IO.File]::ReadAllBytes((Join-Path $Generated 'decode-q4-k.input.bin'))
    $q4Short = Write-Mutant 'q4-short.bin' $q4[0..($q4.Length - 2)]
    $q4LongBytes = New-Object byte[] ($q4.Length + 1)
    [Array]::Copy($q4, $q4LongBytes, $q4.Length)
    $q4LongBytes[$q4.Length] = 0x7f
    $q4Long = Write-Mutant 'q4-long.bin' $q4LongBytes
    $q4BadDBytes = [byte[]] $q4.Clone()
    $q4BadDBytes[0] = 0x00
    $q4BadDBytes[1] = 0x7e
    $q4BadD = Write-Mutant 'q4-nan-d.bin' $q4BadDBytes
    $q4BadDMinBytes = [byte[]] $q4.Clone()
    $q4BadDMinBytes[2] = 0x00
    $q4BadDMinBytes[3] = 0x7e
    $q4BadDMin = Write-Mutant 'q4-nan-dmin.bin' $q4BadDMinBytes

    $activation = [IO.File]::ReadAllBytes((Join-Path $Generated 'q8-k-activations.input-f32le.bin'))
    $activationShort = Write-Mutant 'activation-short.bin' $activation[0..($activation.Length - 2)]
    $activationLongBytes = New-Object byte[] ($activation.Length + 1)
    [Array]::Copy($activation, $activationLongBytes, $activation.Length)
    $activationLong = Write-Mutant 'activation-long.bin' $activationLongBytes
    $activationNanBytes = [byte[]] $activation.Clone()
    $activationNanBytes[0] = 0x45
    $activationNanBytes[1] = 0x23
    $activationNanBytes[2] = 0xc1
    $activationNanBytes[3] = 0x7f
    $activationNan = Write-Mutant 'activation-nan.bin' $activationNanBytes

    $q8 = [IO.File]::ReadAllBytes((Join-Path $Generated 'q8-k-activations.output-q8-k.bin'))
    $q8Short = Write-Mutant 'q8-short.bin' $q8[0..($q8.Length - 2)]
    $q8LongBytes = New-Object byte[] ($q8.Length + 1)
    [Array]::Copy($q8, $q8LongBytes, $q8.Length)
    $q8Long = Write-Mutant 'q8-long.bin' $q8LongBytes
    $q8BadDBytes = [byte[]] $q8.Clone()
    $q8BadDBytes[0] = 0x45
    $q8BadDBytes[1] = 0x23
    $q8BadDBytes[2] = 0xc1
    $q8BadDBytes[3] = 0x7f
    $q8BadD = Write-Mutant 'q8-nan-d.bin' $q8BadDBytes
    $q8BadSumBytes = [byte[]] $q8.Clone()
    $q8BadSumBytes[260] = $q8BadSumBytes[260] -bxor 1
    $q8BadSum = Write-Mutant 'q8-bad-sum.bin' $q8BadSumBytes

    $validQ4 = Join-Path $Generated 'decode-q4-k.input.bin'
    $validActivation = Join-Path $Generated 'q8-k-activations.input-f32le.bin'
    $validQ8 = Join-Path $Generated 'q8-k-activations.output-q8-k.bin'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '768', $q4Short, $sentinel) $sentinel $sentinelBytes 'decode one-byte-short'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '768', $q4Long, $sentinel) $sentinel $sentinelBytes 'decode one-byte-long'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '768', $q4BadD, $sentinel) $sentinel $sentinelBytes 'decode non-finite d'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '768', $q4BadDMin, $sentinel) $sentinel $sentinelBytes 'decode non-finite dmin'
    Invoke-ExpectedFailure $Executable @('decode', 'unknown', '768', $validQ4, $sentinel) $sentinel $sentinelBytes 'decode unknown type'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '0', $validQ4, $sentinel) $sentinel $sentinelBytes 'decode n zero'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '255', $validQ4, $sentinel) $sentinel $sentinelBytes 'decode n below one block'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '257', $validQ4, $sentinel) $sentinel $sentinelBytes 'decode invalid block n'
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '256', $empty, $sentinel) $sentinel $sentinelBytes 'decode empty input'
    Invoke-ExpectedFailure $Executable @('q8', '768', $activationShort, $sentinel) $sentinel $sentinelBytes 'Q8 one-byte-short'
    Invoke-ExpectedFailure $Executable @('q8', '768', $activationLong, $sentinel) $sentinel $sentinelBytes 'Q8 one-byte-long'
    Invoke-ExpectedFailure $Executable @('q8', '768', $activationNan, $sentinel) $sentinel $sentinelBytes 'Q8 non-finite activation'
    Invoke-ExpectedFailure $Executable @('q8', '257', $validActivation, $sentinel) $sentinel $sentinelBytes 'Q8 invalid n'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $q4Short, $validQ8, $sentinel) $sentinel $sentinelBytes 'dot short weight'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $q4Long, $validQ8, $sentinel) $sentinel $sentinelBytes 'dot long weight'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $validQ4, $q8Short, $sentinel) $sentinel $sentinelBytes 'dot short Q8'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $validQ4, $q8Long, $sentinel) $sentinel $sentinelBytes 'dot long Q8'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $validQ4, $q8BadD, $sentinel) $sentinel $sentinelBytes 'dot non-finite Q8 d'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '768', $validQ4, $q8BadSum, $sentinel) $sentinel $sentinelBytes 'dot inconsistent Q8 sum'
    Invoke-ExpectedFailure $Executable @('dot', 'f32', '768', $validQ4, $validQ8, $sentinel) $sentinel $sentinelBytes 'dot unsupported type'
    Invoke-ExpectedFailure $Executable @('dot', 'q4-k', '2147483648', $validQ4, $validQ8, $sentinel) $sentinel $sentinelBytes 'dot n above INT_MAX'
    Invoke-ExpectedFailure $Executable @('unknown-operation', $validQ4, $sentinel) $sentinel $sentinelBytes 'unknown operation'

    $hardlink = Join-Path $Directory 'q4-output-hardlink.bin'
    New-Item -ItemType HardLink -Path $hardlink -Target $validQ4 | Out-Null
    $q4Before = [IO.File]::ReadAllBytes($validQ4)
    Invoke-ExpectedFailure $Executable @('decode', 'q4-k', '768', $validQ4, $hardlink) $hardlink $q4Before 'output hardlink alias'
    Assert-Equal ([Convert]::ToBase64String([IO.File]::ReadAllBytes($validQ4))) ([Convert]::ToBase64String($q4Before)) 'hardlink alias source bytes'

    $nonEmptySnapshot = @(
        Get-ChildItem -LiteralPath $Generated -Force |
            Sort-Object Name |
            ForEach-Object { "$($_.Name)|$($_.Length)|$(Get-Sha256File $_.FullName)" }
    ) -join "`n"
    $result = Invoke-OracleProcess $Executable @('generate', $Generated)
    Assert-True ($result.ExitCode -ne 0) 'generate into non-empty staging unexpectedly succeeded'
    $afterSnapshot = @(
        Get-ChildItem -LiteralPath $Generated -Force |
            Sort-Object Name |
            ForEach-Object { "$($_.Name)|$($_.Length)|$(Get-Sha256File $_.FullName)" }
    ) -join "`n"
    Assert-Equal $afterSnapshot $nonEmptySnapshot 'non-empty generation mutation'
    $fixtureAfter = @(
        Get-ChildItem -LiteralPath $Generated -Force |
            Sort-Object Name |
            ForEach-Object { "$($_.Name)|$($_.Length)|$(Get-Sha256File $_.FullName)|$($_.LastWriteTimeUtc.Ticks)" }
    ) -join "`n"
    Assert-Equal $fixtureAfter $fixtureBefore 'malformed CLI matrix mutated generated fixture tree'
}

function Remove-ControlledDirectory {
    param([string] $Path, [string] $RequiredPrefix)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $full = [IO.Path]::GetFullPath($Path)
    Assert-True ($full.StartsWith($RequiredPrefix, [StringComparison]::OrdinalIgnoreCase)) "refusing to remove uncontrolled directory $full"
    Assert-DirectPath $full 'directory'
    Remove-Item -LiteralPath $full -Recurse -Force
}

function Assert-ControlledExistingFixtureSet {
    param([string] $Directory)
    Assert-NoReparseAncestors $Directory
    Assert-DirectPath $Directory 'directory'
    $entries = @(Get-ChildItem -LiteralPath $Directory -Force)
    $expectedNames = @($ExpectedFiles.Keys) + @('quant-vectors.json')
    Assert-Equal $entries.Count 17 'replacement-source fixture entry count'
    Assert-Equal (@($entries.Name | Sort-Object) -join '|') (@($expectedNames | Sort-Object) -join '|') 'replacement-source exact inventory'
    foreach ($entry in $entries) {
        Assert-True (-not $entry.PSIsContainer) "replacement-source entry must be a file: $($entry.Name)"
        Assert-True (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "replacement-source entry must not be a reparse point: $($entry.Name)"
    }
    $manifest = (Read-StrictUtf8 (Join-Path $Directory 'quant-vectors.json')) | ConvertFrom-Json
    Assert-Equal $manifest.upstream.repository $Repository 'replacement-source repository'
    Assert-Equal $manifest.upstream.release $Release 'replacement-source release'
    Assert-Equal $manifest.upstream.commit $Revision 'replacement-source revision'
    Assert-Equal $manifest.upstream.license 'MIT' 'replacement-source license'
    Assert-Equal $manifest.generator.local_oracle_sha256 (Get-Sha256File $OracleSource) 'replacement-source oracle hash'
    $fileMap = Get-FileRecordMap $manifest
    Assert-Equal $fileMap.Count 16 'replacement-source file record count'
    foreach ($expected in $ExpectedFiles.GetEnumerator()) {
        $key = $expected.Key.ToLowerInvariant()
        Assert-True $fileMap.ContainsKey($key) "replacement-source missing $($expected.Key)"
        $record = $fileMap[$key]
        Assert-Equal ([int64] $record.bytes) ([int64] $expected.Value) "replacement-source $($expected.Key) declared bytes"
        $path = Join-Path $Directory $expected.Key
        Assert-Equal ([int64] (Get-Item -LiteralPath $path).Length) ([int64] $expected.Value) "replacement-source $($expected.Key) actual bytes"
        Assert-Equal (Get-Sha256File $path) $record.sha256 "replacement-source $($expected.Key) hash"
    }
}

function Publish-Fixtures {
    param(
        [string] $Staging,
        [string] $Destination = $Fixtures,
        [switch] $InjectPreSwapFailure
    )
    $parent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Assert-NoReparseAncestors $parent
    $parentPrefix = [IO.Path]::GetFullPath($parent) + [IO.Path]::DirectorySeparatorChar
    $publish = "$Destination.publish-$([guid]::NewGuid().ToString('N'))"
    $backup = "$Destination.backup-$([guid]::NewGuid().ToString('N'))"
    $hadExisting = Test-Path -LiteralPath $Destination
    $movedExisting = $false
    try {
        New-Item -ItemType Directory -Path $publish | Out-Null
        foreach ($entry in Get-ChildItem -LiteralPath $Staging -Force) {
            [IO.File]::Copy($entry.FullName, (Join-Path $publish $entry.Name), $false)
        }
        Assert-FixtureSet $publish
        if ($InjectPreSwapFailure) {
            throw 'injected pre-swap publication failure'
        }

        if ($hadExisting) {
            Assert-ControlledExistingFixtureSet $Destination
            Move-Item -LiteralPath $Destination -Destination $backup
            $movedExisting = $true
        }
        try {
            Move-Item -LiteralPath $publish -Destination $Destination
            Assert-FixtureSet $Destination
        } catch {
            if (Test-Path -LiteralPath $Destination) {
                Remove-ControlledDirectory $Destination $parentPrefix
            }
            if ($movedExisting -and (Test-Path -LiteralPath $backup)) {
                Move-Item -LiteralPath $backup -Destination $Destination
                $movedExisting = $false
            }
            throw
        }
        if ($movedExisting) {
            Remove-ControlledDirectory $backup $parentPrefix
            $movedExisting = $false
        }
    } finally {
        if (Test-Path -LiteralPath $publish) {
            Remove-ControlledDirectory $publish $parentPrefix
        }
    }
}

function Get-FixtureTreeSnapshot {
    param([string] $Directory)
    return @(
        Get-ChildItem -LiteralPath $Directory -Force |
            Sort-Object Name |
            ForEach-Object { "$($_.Name)|$($_.Length)|$(Get-Sha256File $_.FullName)|$($_.LastWriteTimeUtc.Ticks)" }
    ) -join "`n"
}

function Invoke-PublicationFailureTest {
    param([string] $Staging)
    $root = "C:\tmp\lightbridge-quant-oracle-publish-test-$([guid]::NewGuid().ToString('N'))"
    $destination = Join-Path $root 'fixtures'
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    try {
        foreach ($entry in Get-ChildItem -LiteralPath $Staging -Force) {
            [IO.File]::Copy($entry.FullName, (Join-Path $destination $entry.Name), $false)
        }
        Assert-FixtureSet $destination
        $before = Get-FixtureTreeSnapshot $destination
        Assert-Rejected {
            Publish-Fixtures $Staging $destination -InjectPreSwapFailure
        } 'injected pre-swap publication failure'
        Assert-Equal (Get-FixtureTreeSnapshot $destination) $before 'publication failure canonical-tree mutation'
        $siblings = @(Get-ChildItem -LiteralPath $root -Force)
        Assert-Equal $siblings.Count 1 'publication failure orphan sibling count'
        Assert-Equal $siblings[0].Name 'fixtures' 'publication failure surviving entry'
    } finally {
        Remove-ControlledDirectory $root 'C:\tmp\lightbridge-quant-oracle-publish-test-'
    }
}

function Write-JsonObject {
    param($Object, [string] $Path)
    $json = ($Object | ConvertTo-Json -Depth 16)
    $json = ($json -replace "`r`n", "`n").TrimEnd("`r", "`n") + "`n"
    $encoding = New-Object Text.UTF8Encoding($false, $true)
    [IO.File]::WriteAllText($Path, $json, $encoding)
}

function New-TamperCase {
    param([string] $Root, [string] $Name)
    $case = Join-Path $Root $Name
    $fixtureCopy = Join-Path $case 'fixtures'
    New-Item -ItemType Directory -Path $fixtureCopy -Force | Out-Null
    foreach ($entry in Get-ChildItem -LiteralPath $Fixtures -Force) {
        [IO.File]::Copy($entry.FullName, (Join-Path $fixtureCopy $entry.Name), $false)
    }
    $pinCopy = Join-Path $case 'PINNED.toml'
    $licenseCopy = Join-Path $case 'LICENSE'
    [IO.File]::Copy($PinManifest, $pinCopy, $false)
    [IO.File]::Copy($LicensePath, $licenseCopy, $false)
    return [pscustomobject] @{ Root = $case; Fixtures = $fixtureCopy; Pin = $pinCopy; License = $licenseCopy }
}

function Assert-Rejected {
    param([scriptblock] $Action, [string] $Context)
    $rejected = $false
    try {
        & $Action
    } catch {
        $rejected = $true
    }
    Assert-True $rejected "$Context tamper was not rejected"
}

function Get-LiveVerificationSnapshot {
    $rows = @()
    foreach ($entry in Get-ChildItem -LiteralPath $Fixtures -Force | Sort-Object Name) {
        $rows += "fixture/$($entry.Name)|$($entry.Length)|$(Get-Sha256File $entry.FullName)|$($entry.LastWriteTimeUtc.Ticks)"
    }
    foreach ($path in @($PinManifest, $LicensePath, $OracleSource)) {
        $entry = Get-Item -LiteralPath $path
        $rows += "$($entry.FullName)|$($entry.Length)|$(Get-Sha256File $entry.FullName)|$($entry.LastWriteTimeUtc.Ticks)"
    }
    return $rows -join "`n"
}

function Invoke-VerifyTamperMatrix {
    Assert-True ([BitConverter]::IsLittleEndian) 'tamper verification supports little-endian hosts only'
    Assert-UpstreamAndPins
    Assert-FixtureSet $Fixtures
    $liveBefore = Get-LiveVerificationSnapshot
    $root = "C:\tmp\lightbridge-quant-oracle-tamper-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $root | Out-Null
    try {
        $case = New-TamperCase $root 'missing'
        Remove-Item -LiteralPath (Join-Path $case.Fixtures 'decode-q4-k.input.bin') -Force
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'missing fixture'

        $case = New-TamperCase $root 'extra'
        [IO.File]::WriteAllBytes((Join-Path $case.Fixtures 'extra.bin'), [byte[]] @(1, 2, 3))
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'extra fixture'

        $case = New-TamperCase $root 'hash'
        $path = Join-Path $case.Fixtures 'decode-q4-k.input.bin'
        $bytes = [IO.File]::ReadAllBytes($path)
        $bytes[20] = $bytes[20] -bxor 1
        [IO.File]::WriteAllBytes($path, $bytes)
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'fixture hash alteration'

        $case = New-TamperCase $root 'length'
        $path = Join-Path $case.Fixtures 'decode-q5-k.input.bin'
        $bytes = [IO.File]::ReadAllBytes($path)
        $longer = New-Object byte[] ($bytes.Length + 1)
        [Array]::Copy($bytes, $longer, $bytes.Length)
        [IO.File]::WriteAllBytes($path, $longer)
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'fixture length alteration'

        $case = New-TamperCase $root 'malformed-json'
        [IO.File]::WriteAllText((Join-Path $case.Fixtures 'quant-vectors.json'), "{`n", (New-Object Text.UTF8Encoding($false, $true)))
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'malformed JSON'

        $case = New-TamperCase $root 'duplicate'
        $path = Join-Path $case.Fixtures 'quant-vectors.json'
        $manifest = (Read-StrictUtf8 $path) | ConvertFrom-Json
        $manifest.files[1].path = $manifest.files[0].path
        Write-JsonObject $manifest $path
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'duplicate fixture record'

        $case = New-TamperCase $root 'traversal'
        $path = Join-Path $case.Fixtures 'quant-vectors.json'
        $manifest = (Read-StrictUtf8 $path) | ConvertFrom-Json
        $manifest.files[0].path = '../escape.bin'
        Write-JsonObject $manifest $path
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'traversal fixture record'

        $case = New-TamperCase $root 'wrong-source'
        $path = Join-Path $case.Fixtures 'quant-vectors.json'
        $manifest = (Read-StrictUtf8 $path) | ConvertFrom-Json
        $manifest.decode_vectors[0].source_path = 'ggml/src/not-authenticated.c'
        $manifest.decode_vectors[0].source_url = "$SourceUrlPrefix/$Revision/ggml/src/not-authenticated.c"
        Write-JsonObject $manifest $path
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'wrong vector source'

        $case = New-TamperCase $root 'q8-bsums'
        $q8Path = Join-Path $case.Fixtures 'q8-k-activations.output-q8-k.bin'
        $q8Bytes = [IO.File]::ReadAllBytes($q8Path)
        $q8Bytes[260] = $q8Bytes[260] -bxor 1
        [IO.File]::WriteAllBytes($q8Path, $q8Bytes)
        $q8Hash = Get-Sha256File $q8Path
        $path = Join-Path $case.Fixtures 'quant-vectors.json'
        $manifest = (Read-StrictUtf8 $path) | ConvertFrom-Json
        @($manifest.files | Where-Object path -ceq 'q8-k-activations.output-q8-k.bin')[0].sha256 = $q8Hash
        $manifest.q8_k_vectors[0].output.sha256 = $q8Hash
        foreach ($dot in @($manifest.dot_vectors)) {
            $dot.q8_input.sha256 = $q8Hash
        }
        Write-JsonObject $manifest $path
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'Q8_K block sum alteration'

        $case = New-TamperCase $root 'dot-n'
        $path = Join-Path $case.Fixtures 'quant-vectors.json'
        $manifest = (Read-StrictUtf8 $path) | ConvertFrom-Json
        $manifest.dot_vectors[0].n = 256
        Write-JsonObject $manifest $path
        Assert-Rejected { Assert-FixtureSet $case.Fixtures } 'dot n alteration'

        $case = New-TamperCase $root 'license'
        [IO.File]::AppendAllText($case.License, "`nforged`n", (New-Object Text.UTF8Encoding($false, $true)))
        Assert-Rejected { Assert-UpstreamAndPins $case.Pin $case.License } 'retained LICENSE alteration'

        $case = New-TamperCase $root 'pin'
        $pinText = Read-StrictUtf8 $case.Pin
        $pinText = $pinText.Replace(
            'upstream_git_blob_oid = "a7d1fe7d94be4bee3df47f0d710fbfdb62087d1f"',
            'upstream_git_blob_oid = "0000000000000000000000000000000000000000"')
        [IO.File]::WriteAllText($case.Pin, $pinText, (New-Object Text.UTF8Encoding($false, $true)))
        Assert-Rejected { Assert-UpstreamAndPins $case.Pin $case.License } 'PINNED source identity alteration'

        Assert-Equal (Get-LiveVerificationSnapshot) $liveBefore 'tamper matrix live-tree mutation'
        Write-Output 'VerifyOnly tamper matrix passed: 12/12 missing, extra, hash, length, JSON, duplicate, traversal, source, Q8 sums, dot n, LICENSE, and PINNED alterations rejected'
    } finally {
        Remove-ControlledDirectory $root 'C:\tmp\lightbridge-quant-oracle-tamper-'
    }
}

function Invoke-VerifyOnly {
    Assert-True ([BitConverter]::IsLittleEndian) 'quantization fixtures support little-endian hosts only'
    Assert-UpstreamAndPins
    Assert-FixtureSet $Fixtures
    Write-Output 'quant-oracle verification passed: pin, exact 16-binary inventory, hashes, cross-links, Q8 sums, finite activations/dots'
}

if ($VerifyOnly) {
    Invoke-VerifyOnly
    exit 0
}

if ($VerifyTamperMatrix) {
    Invoke-VerifyTamperMatrix
    exit 0
}

Assert-True ([BitConverter]::IsLittleEndian) 'quantization fixture generation supports little-endian hosts only'
Assert-UpstreamAndPins

& rtk cmake -S $PSScriptRoot -B $BuildDirectory -G 'Visual Studio 18 2026' -A x64 "-DLLAMA_CPP_SOURCE=$LlamaCppSource" -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DGGML_CPU=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF -DGGML_CCACHE=OFF -DGGML_NATIVE=OFF
Assert-True ($LASTEXITCODE -eq 0) 'CMake configure failed'
& rtk cmake --build $BuildDirectory --config Release --target bridge-quant-oracle
Assert-True ($LASTEXITCODE -eq 0) 'oracle build failed'
Assert-DirectPath $OracleExecutable 'file'

$stage = "C:\tmp\lightbridge-quant-oracle-stage-$([guid]::NewGuid().ToString('N'))"
$negative = "C:\tmp\lightbridge-quant-oracle-negative-$([guid]::NewGuid().ToString('N'))"
try {
    New-Item -ItemType Directory -Path $stage | Out-Null
    Assert-NoReparseAncestors $stage
    $result = Invoke-OracleProcess $OracleExecutable @('generate', $stage)
    Assert-Equal $result.ExitCode 0 "oracle generation failed: $($result.Stderr)"
    Invoke-NegativeMatrix $OracleExecutable $stage $negative
    Write-Manifest $stage
    Assert-FixtureSet $stage
    Invoke-PublicationFailureTest $stage
    Publish-Fixtures $stage
    Invoke-VerifyOnly
    Write-Output 'quant-oracle generation passed: hardened scalar build, malformed-input matrix, staged fixture publication, verification'
} finally {
    Remove-ControlledDirectory $negative 'C:\tmp\lightbridge-quant-oracle-negative-'
    Remove-ControlledDirectory $stage 'C:\tmp\lightbridge-quant-oracle-stage-'
}
