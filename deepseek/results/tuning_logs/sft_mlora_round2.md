# DeepSeek SFT mLoRA 第二轮（fine grid，结果待回填）

- 日期：**2026-04-08**
- 目标：围绕 round1 最优学习率 `1e-4` 做细化搜索
- 提交脚本：`deepseek/scripts/gs_sft_mlora_round2_fine.sh`

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`
- 验证集比例：`SFT_VAL_RATIO=0.02`
- 预算控制：`MAX_STEPS=6000`（`EPOCHS=12` 仅作上限）
- 并行方式：**单卡**（`NGPU=1`, `NPROC_PER_NODE=1`，非 DDP）
- `TORCH_DTYPE=float32`
- `BATCH_SIZE=2`（每卡）
- `GRAD_ACCUM_STEPS=8`
- `MAX_LENGTH=512`
- LoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录（待回填）

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317093 | `9e-5` | 6000 | RUN |  |  |  | 2026-04-08 提交 |
| 317094 | `1e-4` | 6000 | RUN |  |  |  | 2026-04-08 提交 |
| 317095 | `1.1e-4` | 6000 | RUN |  |  |  | 2026-04-08 提交 |

## 指标位置

- `deepseek/results/sft_grid_round2/mix_chat_real_300k_mlora_lr_*_s6000/train_sft.csv`
- `deepseek/results/sft_grid_round2/mix_chat_real_300k_mlora_lr_*_s6000/test_sft.csv`

