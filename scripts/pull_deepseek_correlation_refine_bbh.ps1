# Pull bbh_eval.json for the stable Top-5 correlation-refine runs (see README list B).
# Usage (repo root): .\scripts\pull_deepseek_correlation_refine_bbh.ps1
# Override: $env:SERVER="user@202.121.138.196"; $env:REMOTE_BASE="/nfsshare/home/wangxiao/Manifold-Lora/deepseek_autogrid/results_correlation_refine"

$ErrorActionPreference = "Stop"
$Server = if ($env:SERVER) { $env:SERVER } else { "wangxiao@202.121.138.196" }
$RemoteBase = if ($env:REMOTE_BASE) {
    $env:REMOTE_BASE.TrimEnd("/")
} else {
    "/nfsshare/home/wangxiao/Manifold-Lora/deepseek_autogrid/results_correlation_refine"
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LocalBase = Join-Path $ProjectDir "deepseek_autogrid/results_correlation_refine"

$Runs = @(
    "lr_1p0673e-03_r32_a16_st500_wd_1p0000e-02",
    "lr_2p0000e-04_r64_a32_st500_wd_1p0000e-02",
    "lr_4p6203e-04_r64_a16_st500_wd_1p0000e-02",
    "lr_5p6961e-04_r32_a32_st500_wd_1p0000e-02",
    "lr_7p0224e-04_r64_a32_st500_wd_1p0000e-02"
)

Write-Host "Pulling bbh_eval.json from ${Server}:${RemoteBase}/ -> $LocalBase"
foreach ($r in $Runs) {
    $dst = Join-Path $LocalBase $r
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $src = "${Server}:${RemoteBase}/${r}/bbh_eval.json"
    Write-Host "  $src"
    scp $src $dst
}

Write-Host "Done. Regenerate leaderboard:"
Write-Host "  python scripts/summarize_deepseek_bbh_results.py --results-root deepseek_autogrid/results_correlation_refine --summary-csv deepseek_autogrid/results_correlation_refine/summary.csv"
