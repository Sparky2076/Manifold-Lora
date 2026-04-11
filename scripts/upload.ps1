# 仅上传 DistilBERT + distilbert_autogrid + 根模块 + scripts（与 upload.sh 一致）
# 用法: .\scripts\upload.ps1  或  pwsh -File scripts/upload.ps1

$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$Remote = "${Server}:~/${RemoteDir}/"
Write-Host "上传到 $Remote （DistilBERT + distilbert_autogrid + 根模块 + scripts）"

scp "$ProjectDir/optimizers.py", `
    "$ProjectDir/lora.py", `
    "$ProjectDir/mlora.py" `
    $Remote
if (Test-Path "$ProjectDir/requirements.txt") { scp "$ProjectDir/requirements.txt" $Remote }

scp -r "${ProjectDir}/distilbert" "${Server}:~/${RemoteDir}/"
scp -r "${ProjectDir}/distilbert_autogrid" "${Server}:~/${RemoteDir}/"

$uploadOnly = @(
    "$ProjectDir/scripts/upload.sh",
    "$ProjectDir/scripts/upload.ps1",
    "$ProjectDir/scripts/commit_and_push.sh",
    "$ProjectDir/scripts/server_submit_distilbert_grid.sh"
)
foreach ($f in $uploadOnly) {
    if (Test-Path $f) {
        scp $f "${Server}:~/${RemoteDir}/scripts/"
    }
}

Write-Host "上传完成。"
