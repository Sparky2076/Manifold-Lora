# DeepSeek SFT LoRA 第五轮（固定 max_steps=6000，lr 细扫，指标待回填）

- 日期：**2026-04-09**
- 目标：在 `max_steps=6000` 下细扫 `lr`：`3.75e-5 / 4e-5 / 4.25e-5`
- 提交脚本：`deepseek/scripts/gs_sft_lora_round5_lr.sh`

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`，`SFT_VAL_RATIO=0.02`
- 预算：`MAX_STEPS=6000`（`EPOCHS=16` 仅上限）
- 单卡：`NGPU=1`，`NPROC_PER_NODE=1`
- `TORCH_DTYPE=float32`，`BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录（指标待回填）

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317883 | `3.75e-5` | 6000 | RUN |  |  |  | 2026-04-09 提交 |
| 317884 | `4e-5` | 6000 | RUN |  |  |  | 2026-04-09 提交 |
| 317885 | `4.25e-5` | 6000 | RUN |  |  |  | 2026-04-09 提交 |

## 指标位置

- `deepseek/results/sft_grid_round5/mix_chat_real_300k_lora_lr_*_s6000/train_sft.csv`
- `deepseek/results/sft_grid_round5/mix_chat_real_300k_lora_lr_*_s6000/test_sft.csv`
