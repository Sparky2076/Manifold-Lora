# 在本地 PowerShell 运行，上传代码到服务器
# 用法: .\scripts\upload.ps1  或  pwsh -File scripts/upload.ps1

$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$Remote = "${Server}:~/${RemoteDir}/"
Write-Host "上传到 $Remote"

scp "$ProjectDir/main.py", `
    "$ProjectDir/models.py", `
    "$ProjectDir/utils.py", `
    "$ProjectDir/optimizers.py", `
    "$ProjectDir/lora.py", `
    "$ProjectDir/mlora.py", `
    "$ProjectDir/requirements.txt" `
    $Remote

$scripts = @(
    "$ProjectDir/scripts/submit_bsub.sh",
    "$ProjectDir/scripts/run_train_bsub.sh",
    "$ProjectDir/scripts/watch_metrics.sh"
)
foreach ($f in $scripts) {
    if (Test-Path $f) {
        scp $f "${Server}:~/${RemoteDir}/scripts/"
    }
}

Write-Host "上传完成"
