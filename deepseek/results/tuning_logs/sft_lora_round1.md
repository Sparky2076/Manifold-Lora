# DeepSeek SFT LoRA 第一轮（learning rate grid，结果已回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`（以服务器实际提交参数为准）
- 训练轮次：`EPOCHS=20`（以服务器实际提交参数为准）
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | 备注 |
|------:|---:|------|----------------|------|
| 313783 | `1e-5` | DONE | `1.6253` | best epoch=20，收敛稳定但较慢 |
| 313784 | `2e-5` | DONE | `1.6173` | best epoch=9，整体最稳健 |
| 313785 | `5e-5` | DONE | `1.6158` | best epoch=5，后续明显过拟合 |

## 简要结论

- 最优单点来自 `5e-5`（`1.6158 @ epoch 5`），但后续过拟合明显。
- 工程默认建议 `2e-5`，在效果接近最优的同时稳定性更好。
- 指标路径：`deepseek/results/sft_grid/<preset>_lr_*/train_sft.csv`、`test_sft.csv`。
