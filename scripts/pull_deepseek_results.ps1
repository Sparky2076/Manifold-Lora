$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.221" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "Manifold-Lora" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LocalResults = if ($env:LOCAL_RESULTS) { $env:LOCAL_RESULTS } else { Join-Path $ProjectDir "deepseek_autogrid/results" }
New-Item -ItemType Directory -Force -Path $LocalResults | Out-Null
scp "$Server`:~/$RemoteDir/deepseek_autogrid/results/summary.csv" $LocalResults
scp "$Server`:~/$RemoteDir/deepseek_autogrid/results/missing_runs.csv" $LocalResults 2>$null
scp "$Server`:~/$RemoteDir/deepseek_autogrid/results/deepseek_grid_analysis.md" $LocalResults 2>$null
Write-Host "Done. pulled into $LocalResults"
