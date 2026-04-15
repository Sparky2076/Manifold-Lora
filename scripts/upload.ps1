# 仅上传 DistilBERT + distilbert_autogrid + 根模块 + scripts（与 upload.sh 一致）
# 用法: .\scripts\upload.ps1  或  pwsh -File scripts/upload.ps1

$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$Remote = "${Server}:~/${RemoteDir}/"
Write-Host "上传到 $Remote （增量；默认排除 results/）"

scp "$ProjectDir/optimizers.py", `
    "$ProjectDir/lora.py", `
    "$ProjectDir/mlora.py" `
    $Remote
if (Test-Path "$ProjectDir/requirements.txt") { scp "$ProjectDir/requirements.txt" $Remote }

function Sync-TreeIncremental {
    param(
        [string]$SourceDir,
        [string]$RemoteSubDir
    )
    $source = (Resolve-Path $SourceDir).Path
    $items = Get-ChildItem -Path $source -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/](results|__pycache__)([\\/]|$)' -and $_.Extension -ne '.pyc'
    }
    foreach ($it in $items) {
        $rel = $it.FullName.Substring($source.Length).TrimStart('\','/')
        $relDir = Split-Path -Parent $rel
        $remoteDirUnix = if ([string]::IsNullOrWhiteSpace($relDir)) { "$RemoteSubDir" } else { "$RemoteSubDir/" + ($relDir -replace '\\','/') }
        ssh $Server "mkdir -p ~/$RemoteDir/$remoteDirUnix" | Out-Null
        scp $it.FullName "${Server}:~/${RemoteDir}/${remoteDirUnix}/"
    }
}

Sync-TreeIncremental "${ProjectDir}/distilbert" "distilbert"
Sync-TreeIncremental "${ProjectDir}/distilbert_autogrid" "distilbert_autogrid"

$uploadOnly = @(
    "$ProjectDir/scripts/upload.sh",
    "$ProjectDir/scripts/upload.ps1",
    "$ProjectDir/scripts/pull_results.sh",
    "$ProjectDir/scripts/pull_results.ps1",
    "$ProjectDir/scripts/refresh_results_and_publish.sh",
    "$ProjectDir/scripts/refresh_results_and_publish.ps1",
    "$ProjectDir/scripts/commit_and_push.sh",
    "$ProjectDir/scripts/server_submit_distilbert_grid.sh",
    "$ProjectDir/scripts/server_submit_distilbert_grid_force.sh",
    "$ProjectDir/scripts/kill_distilbert_grid_bjobs.sh"
)
foreach ($f in $uploadOnly) {
    if (Test-Path $f) {
        scp $f "${Server}:~/${RemoteDir}/scripts/"
    }
}

Write-Host "上传完成。"
