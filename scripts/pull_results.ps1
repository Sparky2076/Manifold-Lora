# 从服务器拉回 DistilBERT 网格汇总结果到本地（Windows PowerShell）
# 用法:
#   .\scripts\pull_results.ps1
#   $env:SERVER="wangxiao@202.121.138.221"; .\scripts\pull_results.ps1
#   $env:REMOTE_DIR="Manifold-Lora"; .\scripts\pull_results.ps1
# mLoRA:
#   $env:RESULTS_REL="distilbert_autogrid/results_mlora"; .\scripts\pull_results.ps1

$ErrorActionPreference = "Stop"

$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.221" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ResultsRel = if ($env:RESULTS_REL) { $env:RESULTS_REL } else { "distilbert_autogrid/results" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LocalResults = Join-Path $ProjectDir $ResultsRel

New-Item -ItemType Directory -Path $LocalResults -Force | Out-Null

Write-Host "从 $Server`:~/$RemoteDir/$ResultsRel 拉取结果到 $LocalResults"

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

# 必拉: summary；missing 可选
scp "$Server`:~/$RemoteDir/$ResultsRel/summary.csv" $LocalResults
if (-not (Try-Scp "$ResultsRel/missing_runs.csv" $LocalResults)) {
    Write-Warning "远端无 missing_runs.csv，跳过（可先在服务器跑 aggregate_results 生成）。"
}

# 分析报告: 新路径 results/；若不存在则回退旧路径 docs/
$ok = Try-Scp "$ResultsRel/distilbert_grid_analysis.md" $LocalResults
if (-not $ok) {
    Write-Host "results 下未找到分析报告，尝试旧路径 docs/ ..."
    $ok = Try-Scp "docs/distilbert_grid_analysis.md" $LocalResults
}
if (-not $ok) {
    Write-Warning "未找到 distilbert_grid_analysis.md（results/ 和 docs/ 均不存在）"
}

# 快照文档可选
$null = Try-Scp "$ResultsRel/distilbert_grid_snapshot.md" $LocalResults

Write-Host "拉取完成。"
