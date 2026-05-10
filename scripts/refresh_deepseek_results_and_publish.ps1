$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir
& "$ScriptDir/pull_deepseek_results.ps1"
$strictArgs = @()
if ($env:ALLOW_INCOMPLETE -eq "1") { $strictArgs += "--allow-incomplete" }
python -m deepseek_autogrid.aggregate_results @strictArgs
python -m deepseek_autogrid.analyze_results @strictArgs
git add deepseek_autogrid/results/summary.csv deepseek_autogrid/results/missing_runs.csv deepseek_autogrid/results/deepseek_grid_analysis.md deepseek_autogrid/results/deepseek_grid_snapshot.md
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) { Write-Host "No deepseek result changes."; exit 0 }
$msg = if ($env:COMMIT_MSG) { $env:COMMIT_MSG } else { "Update deepseek grid results" }
git commit --trailer "Made-with: Cursor" -m "$msg"
git push
Write-Host "Done."
