# DeepSeek SFT mLoRA 第二轮（learning rate grid v2，结果待回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft_mlora_v2.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`
- 训练轮次：`EPOCHS=20`
- LoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | 备注 |
|------:|---:|------|----------------|------|
| 313852 | `5e-5` | RUN |  | 运行中，结果待回填 |
| 313853 | `8e-5` | RUN |  | 运行中，结果待回填 |
| 313854 | `1e-4` | PEND |  | 排队中，等待调度 |

## 回填说明

- 作业结束后，补充每个 Job 的 `best eval loss`、最佳 epoch 与简要结论。
- 指标路径：`deepseek/results/sft_grid_v2/<preset>_mlora_v2_lr_*/train_sft.csv`、`test_sft.csv`。
