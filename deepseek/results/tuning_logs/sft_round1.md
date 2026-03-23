# DeepSeek SFT Round 1（learning rate grid）

- 日期：2026-03-09
- 模型：DeepSeek SFT（`deepseek/scripts/gs_lr_deepseek_sft.sh`）
- 预设：`SFT_PRESET=testing_alpaca_small`
- 统一配置：
  - `EPOCHS=5`
  - `BATCH_SIZE=2`
  - `GRAD_ACCUM_STEPS=8`
  - `MAX_LENGTH=512`
  - `LORA_TYPE=default`
  - `LORA_R=8`
  - `LORA_ALPHA=16`
  - `LORA_DROPOUT=0.05`

## 提交记录

| JobID | lr | 输出目录 | 状态 |
|------:|---:|----------|------|
| 313554 | `1e-5` | `deepseek/results/sft_grid/testing_alpaca_small_lr_1e_5` | submitted |
| 313555 | `2e-5` | `deepseek/results/sft_grid/testing_alpaca_small_lr_2e_5` | submitted |
| 313556 | `5e-5` | `deepseek/results/sft_grid/testing_alpaca_small_lr_5e_5` | submitted |

## 备注

- 网格已成功提交到 `gpu` 队列。
- 训练结束后补充每个 Job 的最佳 `eval loss` / `eval ppl` 与结论。
