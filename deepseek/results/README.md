# DeepSeek SFT 实验结果目录（与仓库根目录 `results/` 结构对应）

| 子目录 | 对应 DistilBERT 侧 | 说明 |
|--------|-------------------|------|
| `tuning_logs/` | `results/tuning_logs/` | 各轮调参 Markdown 记录（lr 网格、JobID、最优 eval loss 等） |
| `final_sft/` | `results/final_loRA/` / `final_mLoRA/` | 选定超参后的长训；存放 `train_sft.csv`、`test_sft.csv` 与 README |
| `sft_grid/` | （网格时按子目录归档） | `gs_lr_deepseek_sft.sh` 自动写入：`sft_grid/<preset>_lr_*/` |

单次 `submit_bsub_sft.sh` 且未改 `METRICS_DIR` 时，默认写入 **`deepseek/results/`** 根下的 `train_sft.csv`、`test_sft.csv`。
