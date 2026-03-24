# DeepSeek SFT mLoRA 第一轮（learning rate grid，结果已回填）

- 日期：2026-03-24
- 任务：`deepseek/scripts/gs_lr_deepseek_sft_mlora.sh`
- 预设：`SFT_PRESET=alpaca_train_1k`（以服务器实际提交参数为准）
- 训练轮次：`EPOCHS=20`（以服务器实际提交参数为准）
- LoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | 状态 | best eval loss | 备注 |
|------:|---:|------|----------------|------|
| 313801 | `1e-5` | DONE | `2.1269` | best epoch=20，收敛慢且效果较弱 |
| 313802 | `2e-5` | DONE | `1.9153` | best epoch=20，显著优于 1e-5 |
| 313803 | `5e-5` | DONE | `1.7363` | best epoch=20，本轮最优且仍在下降 |

## 简要结论

- 第一轮 mLoRA 下，`5e-5` 最优，且到 epoch 20 仍未见明显过拟合。
- mLoRA 整体仍弱于同轮 LoRA（LoRA 最优约 `1.6158`）。
- 指标路径：`deepseek/results/sft_grid/alpaca_train_1k_mlora_lr_*/train_sft.csv`、`test_sft.csv`。
