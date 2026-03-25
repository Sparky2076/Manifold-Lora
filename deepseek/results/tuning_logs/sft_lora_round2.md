# DeepSeek SFT LoRA 第二轮（learning rate grid v2，结果待回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft_v2.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`
- 训练轮次：`EPOCHS=12`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | best epoch | best eval ppl | 备注 |
|------:|---:|------|----------------|-----------:|--------------:|------|
| 313849 | `1.5e-5` | DONE | 1.6294 | 12 | 5.10 | `EPOCHS=12` 收敛稳定 |
| 313850 | `2e-5` | DONE | 1.6198 | 12 | 5.05 | 相比 `1.5e-5` 更优 |
| 313851 | `2.5e-5` | DONE | 1.6172 | 9 | 5.04 | 本轮 LoRA 最优；后续 epoch 略回升 |

## 回填说明

- 作业结束后，补充每个 Job 的 `best eval loss`、最佳 epoch 与简要结论。
- 指标路径：`deepseek/results/sft_grid_v2/<preset>_lora_v2_lr_*/train_sft.csv`、`test_sft.csv`。

## 结论（本轮）

- **最优**：LoRA `lr=2.5e-5`（Job `313851`），best eval loss **1.6172**（epoch **9**，ppl **5.04**）。
- **次优**：LoRA `lr=2e-5`（Job `313850`），best eval loss **1.6198**（epoch **12**，ppl **5.05**）。
