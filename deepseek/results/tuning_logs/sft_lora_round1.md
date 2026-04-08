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

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 316967 | `1e-5` | 3000 | DONE | 1.3698 | 3000 | 3.93 | 粗筛基线；收敛正常 |
| 316968 | `2e-5` | 3000 | DONE | 1.3439 | 3000 | 3.83 | 明显优于 `1e-5` |
| 316969 | `3e-5` | 3000 | DONE | 1.3306 | 3000 | 3.78 | 本轮 LoRA 最优 |

## 指标位置

- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_lora_lr_*_s3000/train_sft.csv`
- `deepseek/results/sft_grid_coarse/mix_chat_real_300k_lora_lr_*_s3000/test_sft.csv`

## 本轮结论（LoRA）

- 在 `max_steps=3000` 的单卡粗筛下，LoRA 呈现随学习率增大而持续改善：`1e-5 > 2e-5 > 3e-5`（loss 由 1.3698 降至 1.3306）。
- **当前最优点：`lr=3e-5`（Job 316969）**，建议下一轮围绕 `3e-5` 做细化（如 `2.5e-5 / 3e-5 / 3.5e-5`），并把 `max_steps` 提升到 6000-10000 验证稳定性。

