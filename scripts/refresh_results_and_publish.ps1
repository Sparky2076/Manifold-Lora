# 一键：拉取服务器汇总文件 -> 校验是否跑满375 -> 本地汇总/分析 -> git 提交并推送
# 用法:
#   .\scripts\refresh_results_and_publish.ps1
#   $env:COMMIT_MSG="update full 375 results"; .\scripts\refresh_results_and_publish.ps1
# 可选:
#   $env:SERVER / $env:REMOTE_DIR 传给 pull_results.ps1
#   $env:ALLOW_INCOMPLETE=1       允许未跑满时继续（默认严格要求 375）

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "==> [1/5] 拉取服务器结果到本地"
& "$ScriptDir/pull_results.ps1"

$strictArgs = @()
if ($env:ALLOW_INCOMPLETE -eq "1") {
    $strictArgs += "--allow-incomplete"
}

Write-Host "==> [2/5] 本地汇总（默认要求跑满 375）"
python -m distilbert_autogrid.aggregate_results @strictArgs

Write-Host "==> [3/5] 本地分析（默认要求跑满 375）"
python -m distilbert_autogrid.analyze_results @strictArgs

Write-Host "==> [4/5] git 状态"
git status -sb

Write-Host "==> [5/5] 提交并推送"
git add `
    distilbert_autogrid/results/summary.csv `
    distilbert_autogrid/results/missing_runs.csv `
    distilbert_autogrid/results/distilbert_grid_analysis.md `
    distilbert_autogrid/results/distilbert_grid_snapshot.md

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "没有结果文件变化，跳过 commit/push。"
    exit 0
}

$msg = if ($env:COMMIT_MSG) { $env:COMMIT_MSG } else { "Update distilbert grid results (summary/missing/analysis)" }
git commit -m "$msg"
git push
Write-Host "完成。"
