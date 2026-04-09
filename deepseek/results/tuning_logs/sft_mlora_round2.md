# DeepSeek SFT mLoRA 第二轮（fine grid，结果已回填）

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

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317093 | `9e-5` | 6000 | DONE | 1.3605 | 6000 | 3.90 |  |
| 317094 | `1e-4` | 6000 | DONE | 1.3559 | 6000 | 3.88 |  |
| 317095 | `1.1e-4` | 6000 | DONE | 1.3520 | 6000 | 3.87 | 本轮 mLoRA 最优 |

## 指标位置

- `deepseek/results/sft_grid_round2/mix_chat_real_300k_mlora_lr_*_s6000/train_sft.csv`
- `deepseek/results/sft_grid_round2/mix_chat_real_300k_mlora_lr_*_s6000/test_sft.csv`

## 本轮结论（mLoRA round2）

- `9e-5 → 1e-4 → 1.1e-4` 单调变好（`1.3605 → 1.3559 → 1.3520`），与 round1 粗筛方向一致。
- **当前 mLoRA 推荐单点：`lr=1.1e-4`（Job 317095）**。
- 相对 round1 同 lr：`1e-4` 从 3000 step 的 **1.3870** 降到 6000 step 的 **1.3559**，说明 **加长步数对 mLoRA 同样有效**。
- 与 LoRA round2 最优（`1.3038 @ 3.5e-5, 6000`）相比，mLoRA 仍落后约 **0.05** eval loss；第三轮用固定 `1.1e-4` 扫更长 `max_steps` 验证是否继续缩小差距。

## 下一轮

- 见 `deepseek/scripts/gs_sft_mlora_round3_steps.sh` 与 `sft_mlora_round3.md`。
