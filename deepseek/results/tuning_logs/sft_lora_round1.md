# DeepSeek SFT LoRA 第一轮（learning rate grid，结果待回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`（以服务器实际提交参数为准）
- 训练轮次：`EPOCHS=20`（以服务器实际提交参数为准）
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | 备注 |
|------:|---:|------|----------------|------|
| 313783 | `1e-5` | RUN |  | 运行中，结果待回填 |
| 313784 | `2e-5` | RUN |  | 运行中，结果待回填 |
| 313785 | `5e-5` | RUN |  | 运行中，结果待回填 |

## 回填说明

- 作业结束后，补充每个 Job 的 `best eval loss`、最佳 epoch 与简要结论。
- 指标路径：`deepseek/results/sft_grid/<preset>_lr_*/train_sft.csv`、`test_sft.csv`。
