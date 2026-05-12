# DeepSeek SFT correlation-refine grid (48 runs)

## What this is

- **Grid**: `deepseek_autogrid.config_correlation_refine` — dense **lr** (12 log-spaced points between 2e-4 and 2e-3), **r** in {32, 64}, **alpha** in {16, 32}, **weight_decay** fixed **0.01** → **48** combinations.
- **Training**: `DeepSeek-R1-Distill-Qwen-1.5B`, `alpaca_train_1k`, `SFT_VAL_RATIO=0.2`, `MAX_STEPS=500`, `EVAL_EVERY=100`, LoRA `default`.
- **Where it ran**: compute cluster (login `202.121.138.196`), results under `.../deepseek_autogrid/results_correlation_refine/`.
- **Checked in here**: lightweight **`summary.csv`** + **`deepseek_grid_analysis.md`** only (per-run `test_sft.csv` / `sft_lora_state.pt` stay on cluster or sync separately; a few sample run dirs may exist in git for debugging only).

## Summary stats (`summary.csv`)

- **Rows**: 48, all **`status=ok`**.
- **Metric**: `best_eval_perplexity` (validation) — min **3.922**, max **9.789**, mean **4.569**, median **4.278**.

## BBH Top-K (two options)

### A) Default pipeline (`pick_topk_deepseek_runs.py`): by **`best_eval_perplexity` ascending**, `TOP_K=10`

Use on server (paths as on cluster):

```bash
export SUMMARY_CSV="$PWD/deepseek_autogrid/results_correlation_refine/summary.csv"
export RESULTS_ROOT="$PWD/deepseek_autogrid/results_correlation_refine"
TOP_K=10 bash scripts/server_submit_deepseek_bbh_topk_from_summary.sh
```

**Run directory names (Top-10, same order as script):**

1. `lr_2p0000e-03_r64_a32_st500_wd_1p0000e-02`
2. `lr_1p6223e-03_r32_a32_st500_wd_1p0000e-02`
3. `lr_1p0673e-03_r32_a16_st500_wd_1p0000e-02`
4. `lr_2p0000e-04_r64_a32_st500_wd_1p0000e-02`
5. `lr_1p6223e-03_r64_a32_st500_wd_1p0000e-02`
6. `lr_4p6203e-04_r64_a16_st500_wd_1p0000e-02`
7. `lr_3p0398e-04_r32_a32_st500_wd_1p0000e-02`
8. `lr_1p6223e-03_r64_a16_st500_wd_1p0000e-02`
9. `lr_5p6961e-04_r32_a32_st500_wd_1p0000e-02`
10. `lr_7p0224e-04_r64_a32_st500_wd_1p0000e-02`

Several of these have **very large** `last_eval_perplexity` vs best (training unstable after the best checkpoint). They are still valid “best ppl” winners for the default picker.

### B) Recommended if BBH budget is tight: **stable** runs (`last_eval_perplexity / best_eval_perplexity <= 1.15`), **Top-8 by best ppl**

1. `lr_1p0673e-03_r32_a16_st500_wd_1p0000e-02`
2. `lr_2p0000e-04_r64_a32_st500_wd_1p0000e-02`
3. `lr_4p6203e-04_r64_a16_st500_wd_1p0000e-02`
4. `lr_5p6961e-04_r32_a32_st500_wd_1p0000e-02`
5. `lr_7p0224e-04_r64_a32_st500_wd_1p0000e-02`
6. `lr_3p7476e-04_r32_a32_st500_wd_1p0000e-02`
7. `lr_1p0673e-03_r32_a32_st500_wd_1p0000e-02`
8. `lr_2p4657e-04_r64_a32_st500_wd_1p0000e-02`

For a **Top-5 BBH** subset, use rows **1–5** from list B.

## Regenerate checked-in summaries

From repo root (with full run dirs present):

```bash
python -m deepseek_autogrid.aggregate_results \
  --config-module deepseek_autogrid.config_correlation_refine \
  --results-root deepseek_autogrid/results_correlation_refine
python -m deepseek_autogrid.analyze_results \
  --config-module deepseek_autogrid.config_correlation_refine \
  --summary deepseek_autogrid/results_correlation_refine/summary.csv
```
