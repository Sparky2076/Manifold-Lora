# DistilBERT LoRA 网格结果快照

本页为某次拉取结果后运行 `python -m distilbert_autogrid.aggregate_results` 的汇总说明；**完整逐轮曲线仍仅在本地**（各子目录 `train.csv` / `test.csv` 由 `distilbert_autogrid/results/.gitignore` 忽略，避免仓库体积过大）。

## 汇总表（已入库）

| 文件 | 说明 |
|------|------|
| [`distilbert_autogrid/results/summary.csv`](../distilbert_autogrid/results/summary.csv) | 每组一行：`lr, r, alpha, weight_decay, best_val_acc, …`，`metrics_dir` 为仓库内相对路径 |
| [`docs/distilbert_grid_analysis.md`](distilbert_grid_analysis.md) | **自动分析报告**（均值/分组/Top15；由 `python -m distilbert_autogrid.analyze_results` 生成） |

本次快照：**246** 条有效组合（`status=ok`），验证集准确率按 `best_val_acc` 在 `aggregate_results` 中已排序。

## 当前最佳（本快照）

- **best_val_acc ≈ 0.9163**
- **超参**：`lr = 3e-4`，`lora_r = 32`，`lora_alpha = 16`，`weight_decay = 0`，`epochs = 3`（与 `config.py` 网格一致）
- **对应目录**：`distilbert_autogrid/results/lr_3p0000e-04_r32_a16_ep3_wd_0p0000e00/`

头部若干行与 `3e-4` 学习率、中等 rank/alpha 的配置表现接近；更细对比可直接打开 `summary.csv` 排序筛选。

## 如何更新快照

在仓库根、本地已有 `distilbert_autogrid/results/<run_name>/` 时：

```bash
python -m distilbert_autogrid.aggregate_results
```

再按需提交 `summary.csv`、运行 `python -m distilbert_autogrid.analyze_results` 更新 [`distilbert_grid_analysis.md`](distilbert_grid_analysis.md)，并提交本说明（或只更新 CSV）。
