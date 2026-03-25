# DeepSeek SFT mLoRA 第二轮（learning rate grid v2，结果待回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft_mlora_v2.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`
- 训练轮次：`EPOCHS=20`
- LoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | best epoch | best eval ppl | 备注 |
|------:|---:|------|----------------|-----------:|--------------:|------|
| 313852 | `5e-5` | DONE | 1.7363 | 20 | 5.68 | 收敛较慢，20 epoch 持续下降 |
| 313853 | `8e-5` | DONE | 1.6847 | 20 | 5.39 | 本轮 mLoRA 最优 |
| 313854 | `1e-4` | DONE | 1.6665 | 20 | 5.29 | 相比 `8e-5` 更优，且持续下降 |

## 回填说明

- 作业结束后，补充每个 Job 的 `best eval loss`、最佳 epoch 与简要结论。
- 指标路径：`deepseek/results/sft_grid_v2/<preset>_mlora_v2_lr_*/train_sft.csv`、`test_sft.csv`。

## 结论（本轮）

- **最优**：mLoRA `lr=1e-4`（Job `313854`），best eval loss **1.6665**（epoch **20**，ppl **5.29**）。
- 与 LoRA 第二轮最优（1.6172）相比，**本轮 mLoRA 整体偏弱**（loss 更高）。
