# DistilBERT 分类实验结果目录（与 `deepseek/results/` 形式对称）

| 子目录 | 说明 |
|--------|------|
| `tuning_logs/` | 各轮调参 Markdown 记录（lr 网格、JobID、最优 accuracy 等） |
| `final_loRA/` / `final_mLoRA/` | 选定超参后的长训；存放 `train.csv`、`test.csv` 与 README |
| 其他子目录 | 按日期或实验名归档的完整运行 |

单次 `distilbert/scripts/submit_bsub.sh` 且未改 `METRICS_DIR` 时，默认写入 **`distilbert/results/`** 根下的 `train.csv`、`test.csv`。
