# DeepSeek SFT mLoRA 第四轮（固定 max_steps=10000，lr 细扫，指标待回填）

- 日期：**2026-04-09**
- 说明：按轮次编号，本轮为 **mLoRA round 4**（在 round1 粗筛、round2 细扫、round3 步数 sweep 之后的固定步数 lr 细扫）。指标目录名为 `sft_grid_round5/`，与已提交作业路径一致。
- 目标：`max_steps=10000` 下细扫 `lr`：`1.1e-4 / 1.3e-4 / 1.5e-4`
- 提交脚本：`deepseek/scripts/gs_sft_mlora_round4_lr.sh`（`gs_sft_mlora_round5_lr.sh` 为兼容转发）

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`，`SFT_VAL_RATIO=0.02`
- 预算：`MAX_STEPS=10000`（`EPOCHS=20` 仅上限）
- 单卡：`NGPU=1`，`NPROC_PER_NODE=1`
- `TORCH_DTYPE=float32`，`BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- mLoRA：`mlora, r=8, alpha=16, dropout=0.05`

## Job 提交记录（指标待回填）

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317886 | `1.1e-4` | 10000 | RUN |  |  |  | 2026-04-09 提交 |
| 317887 | `1.3e-4` | 10000 | RUN |  |  |  | 2026-04-09 提交 |
| 317888 | `1.5e-4` | 10000 | RUN |  |  |  | 2026-04-09 提交 |

## 指标位置

- `deepseek/results/sft_grid_round5/mix_chat_real_300k_mlora_lr_*_s10000/train_sft.csv`
- `deepseek/results/sft_grid_round5/mix_chat_real_300k_mlora_lr_*_s10000/test_sft.csv`
