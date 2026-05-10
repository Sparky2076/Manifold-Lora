# 拉回 DistilBERT 最优超参 20 epoch 终局目录（与 pull_distilbert_best_20ep.sh 一致）
# 用法:
#   .\scripts\pull_distilbert_best_20ep.ps1
#   $env:PULL_WHICH="lora"   # 或 mlora、both(默认)

$ErrorActionPreference = "Stop"

$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.221" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$Which = if ($env:PULL_WHICH) { $env:PULL_WHICH } else { "both" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$Dist = Join-Path $ProjectDir "distilbert"
New-Item -ItemType Directory -Path $Dist -Force | Out-Null

function Pull-One {
    param([string]$Name)
    $remote = "${Server}:~/${RemoteDir}/distilbert/${Name}"
    Write-Host "拉取 $remote -> $Dist"
    scp -r $remote $Dist
}

switch ($Which) {
    "lora" { Pull-One "results_final_best_lora_20ep" }
    "mlora" { Pull-One "results_final_best_mlora_20ep" }
    "both" {
        Pull-One "results_final_best_lora_20ep"
        Pull-One "results_final_best_mlora_20ep"
    }
    default { throw "PULL_WHICH 应为 lora | mlora | both" }
}

Write-Host "拉取完成。"
