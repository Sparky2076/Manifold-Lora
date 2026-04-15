# 从服务器拉回 DistilBERT 网格汇总结果到本地（Windows PowerShell）
# 用法:
#   .\scripts\pull_results.ps1
#   $env:SERVER="wangxiao@202.121.138.196"; .\scripts\pull_results.ps1
#   $env:REMOTE_DIR="Manifold-Lora"; .\scripts\pull_results.ps1

$ErrorActionPreference = "Stop"

$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LocalResults = Join-Path $ProjectDir "distilbert_autogrid/results"

New-Item -ItemType Directory -Path $LocalResults -Force | Out-Null

Write-Host "从 $Server 拉取结果到 $LocalResults"

function Try-Scp {
    param(
        [string]$RemoteFile,
        [string]$LocalDir
    )
    try {
        scp "$Server`:~/$RemoteDir/$RemoteFile" $LocalDir
        return $true
    } catch {
        return $false
    }
}

# 必拉: summary + missing
scp "$Server`:~/$RemoteDir/distilbert_autogrid/results/summary.csv" $LocalResults
scp "$Server`:~/$RemoteDir/distilbert_autogrid/results/missing_runs.csv" $LocalResults

# 分析报告: 新路径 results/；若不存在则回退旧路径 docs/
$ok = Try-Scp "distilbert_autogrid/results/distilbert_grid_analysis.md" $LocalResults
if (-not $ok) {
    Write-Host "results 下未找到分析报告，尝试旧路径 docs/ ..."
    $ok = Try-Scp "docs/distilbert_grid_analysis.md" $LocalResults
}
if (-not $ok) {
    Write-Warning "未找到 distilbert_grid_analysis.md（results/ 和 docs/ 均不存在）"
}

# 快照文档可选
$null = Try-Scp "distilbert_autogrid/results/distilbert_grid_snapshot.md" $LocalResults

Write-Host "拉取完成。"
