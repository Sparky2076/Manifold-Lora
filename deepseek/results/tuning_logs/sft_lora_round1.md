# DeepSeek SFT LoRA 第一轮（coarse grid，结果待回填）

- 日期：**2026-04-08**
- 目标：在 `mix_chat_real_300k` 上粗筛 LoRA 学习率区间（单卡并发友好）
- 提交脚本：`deepseek/scripts/gs_sft_lora_grid3_coarse.sh`

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`
- 验证集比例：`SFT_VAL_RATIO=0.02`
- 预算控制：`MAX_STEPS=3000`（`EPOCHS=8` 仅作上限）
- 并行方式：**单卡**（`NGPU=1`, `NPROC_PER_NODE=1`，非 DDP）
- `TORCH_DTYPE=float32`
- `BATCH_SIZE=2`（每卡）
- `GRAD_ACCUM_STEPS=8`
- `MAX_LENGTH=512`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录（待回填）

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 316967 | `1e-5` | 3000 | RUN |  |  |  | 2026-04-08 提交 |
| 316968 | `2e-5` | 3000 | RUN |  |  |  | 2026-04-08 提交 |
| 316969 | `3e-5` | 3000 | RUN |  |  |  | 2026-04-08 提交 |

## 指标位置

- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_lora_lr_*_s3000/train_sft.csv`
- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_lora_lr_*_s3000/test_sft.csv`

