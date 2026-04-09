# DeepSeek SFT mLoRA 第三轮（步数预算，提交后回填 JobID）

- 日期：**2026-04-08**
- 目标：固定 round2 最优 `lr=1.1e-4`，扫 `max_steps`（**12000 / 15000 / 18000**），观察 mLoRA 长训是否继续追赶 LoRA
- 提交脚本：`deepseek/scripts/gs_sft_mlora_round3_steps.sh`

## 统一训练配置（本轮）

- 同 round2：`mlora, r=8, alpha=16, dropout=0.05`，单卡，`TORCH_DTYPE=float32`，`BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- `LR=1.1e-4`（默认可 `LR=...` 覆盖）
- `EPOCHS=20`（仅上限，脚本默认；可按集群情况覆盖）

## Job 提交记录（待提交后回填 JobID 与指标）

| JobID | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|----------:|------|----------------|----------:|--------------:|------|
|  | 12000 |  |  |  |  |  |
|  | 15000 |  |  |  |  |  |
|  | 18000 |  |  |  |  |  |

## 指标位置

- `deepseek/results/sft_grid_round3/mix_chat_real_300k_mlora_lr_1p1e_4_s12000/` 等（`s15000`、`s18000`）
