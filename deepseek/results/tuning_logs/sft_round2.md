# DeepSeek SFT Round 2（learning rate grid, stable run）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft.sh`
- 预设：`SFT_PRESET=testing_alpaca_small`
- 训练轮次：`EPOCHS=5`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | 备注 |
|------:|---:|------|------|
| 313772 | `1e-5` | DONE | 指标正常收敛，best eval loss = `14.2498` |
| 313773 | `2e-5` | DONE | 指标正常收敛，best eval loss = `10.7875` |
| 313774 | `5e-5` | DONE | 指标显著更低，best eval loss = `0.0922` |

## 结论（本轮）

- 3 组任务均成功完成（`DONE`），无 NaN 全程崩溃。
- 按 `eval loss` 指标，本轮最优学习率为 `5e-5`。
- 结果文件位于：`deepseek/results/sft_grid/testing_alpaca_small_lr_*/train_sft.csv` 与 `test_sft.csv`。
