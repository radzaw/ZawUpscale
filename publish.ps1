# Builds self-contained single-file executables of ZawUpscale.
# Usage: ./publish.ps1 [-Runtimes win-x64,linux-x64,osx-arm64] [-FrameworkDependent] [-Version 1.2.0]
# linux-x64-cuda is linux-x64 with ONNX Runtime's CUDA build, for NVIDIA GPUs (much larger; needs CUDA 12 and cuDNN 9).
param(
    [string[]]$Runtimes = @('win-x64', 'linux-x64', 'linux-arm64', 'osx-arm64'),
    [switch]$FrameworkDependent,
    [string]$Version = $env:VERSION
)
$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'src/ZawUpscale/ZawUpscale.csproj'
$extra = @()
if ($Version) { $extra += "-p:Version=$Version" }
foreach ($target in $Runtimes) {
    $rid = $target -replace '-cuda$', ''
    $cuda = @()
    if ($rid -ne $target) { $cuda += '-p:ZawCuda=true' }
    $out = Join-Path $PSScriptRoot "dist/$target"
    Write-Host "Publishing $target -> $out" -ForegroundColor Cyan
    dotnet publish $project -c Release -r $rid -o $out `
        --self-contained $(if ($FrameworkDependent) { 'false' } else { 'true' }) `
        -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:DebugType=none @cuda @extra
    if ($LASTEXITCODE -ne 0) { throw "Publishing $target failed" }
}
