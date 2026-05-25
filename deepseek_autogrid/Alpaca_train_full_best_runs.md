# Full Alpaca（`alpaca_train_full`）Top-K 定稿：最佳 LoRA / mLoRA（按 **最低验证 PPL**，非终点 PPL）

定稿目录：

- LoRA：`deepseek_autogrid/results_final/`
- mLoRA：`deepseek_autogrid/results_mlora_final/`

**选型规则**：在每个 `summary.csv` 里 `status=ok` 的行中，取 **`best_eval_perplexity` 最小** 的一条，即训练过程中 **验证集 perplexity 的全局最低点**（对应列 `best_iteration`），**不使用** `last_eval_perplexity`（终点可能更差）。

## 重新汇总与分析

```bash
python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_final \
  --results-root deepseek_autogrid/results_final --discover-run-dirs
python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_final \
  --results-root deepseek_autogrid/results_mlora_final --discover-run-dirs

python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_final \
  --summary deepseek_autogrid/results_final/summary.csv \
  --output deepseek_autogrid/results_final/deepseek_grid_analysis.md
python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_final \
  --summary deepseek_autogrid/results_mlora_final/summary.csv \
  --output deepseek_autogrid/results_mlora_final/deepseek_grid_analysis.md
```

## 收敛图（验证 PPL）

```bash
pip install matplotlib   # 若尚未安装
python -m deepseek_autogrid.plot_alpaca_full_best_convergence
```

输出：`deepseek_autogrid/figures/alpaca_train_full_best_lora_mlora_val_ppl.png`。

## 快照（生成后见各目录 `summary.csv` 首行）

当前仓库快照中（按 `best_eval_perplexity` 排序后第一行）：

| 适配器 | best min val PPL | last val PPL | lr | r | α | wd | run dir |
|--------|------------------|--------------|----|---|---|-----|---------|
| LoRA | 3.700816 | 3.700816 | 1.6223e-03 | 32 | 32 | 0.01 | `lr_1p6223e-03_r32_a32_st2600_wd_1p0000e-02` |
| mLoRA | 3.708931 | 3.708931 | 2e-03 | 64 | 64 | 0.01 | `lr_2p0000e-03_r64_a64_st2600_wd_1p0000e-02` |

（若你本地重跑汇总后数值变化，以 `summary.csv` 为准；上表仅为文档时的仓库状态。）
