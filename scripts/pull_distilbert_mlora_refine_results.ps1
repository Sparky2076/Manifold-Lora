# 拉回 mLoRA 第二轮。汇总在服务器: cd ~/Manifold-Lora; python -m distilbert_autogrid.aggregate_results --results-root distilbert_autogrid/results_mlora_refine --allow-incomplete
$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dst = Join-Path $Root "distilbert_autogrid\results_mlora_refine"
New-Item -ItemType Directory -Path $Dst -Force | Out-Null
if (Get-Command rsync -ErrorAction SilentlyContinue) {
    rsync -avz "${Server}:~/${RemoteDir}/distilbert_autogrid/results_mlora_refine/" ($Dst.TrimEnd('\') + "/")
} else {
    scp -r "${Server}:~/${RemoteDir}/distilbert_autogrid/results_mlora_refine" (Split-Path $Dst)
}
