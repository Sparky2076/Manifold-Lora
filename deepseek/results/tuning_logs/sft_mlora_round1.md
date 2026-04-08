# DeepSeek SFT mLoRA 第一轮（coarse grid，结果待回填）

- 日期：**2026-04-08**
- 目标：在 `mix_chat_real_300k` 上粗筛 mLoRA 学习率区间（单卡并发友好）
- 提交脚本：`deepseek/scripts/gs_sft_mlora_grid3_coarse.sh`

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
- LoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 316970 | `5e-5` | 3000 | DONE | 1.4199 | 3000 | 4.14 | 粗筛基线；收敛正常 |
| 316971 | `8e-5` | 3000 | DONE | 1.3972 | 3000 | 4.04 | 优于 `5e-5` |
| 316972 | `1e-4` | 3000 | DONE | 1.3870 | 3000 | 4.00 | 本轮 mLoRA 最优 |

## 指标位置

- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_mlora_lr_*_s3000/train_sft.csv`
- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_mlora_lr_*_s3000/test_sft.csv`

## 本轮结论（mLoRA）

- 在 `max_steps=3000` 的单卡粗筛下，mLoRA 也呈现随学习率增大而改善：`5e-5 > 8e-5 > 1e-4`（loss 由 1.4199 降至 1.3870）。
- **当前最优点：`lr=1e-4`（Job 316972）**，建议下一轮围绕 `1e-4` 细化（如 `9e-5 / 1e-4 / 1.1e-4`），并将 `max_steps` 提升到 6000-10000 验证稳定性。

