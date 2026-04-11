# DeepSeek SFT LoRA 第四轮（固定步数，学习率细扫，结果已回填）

- 日期：**2026-04-09**
- 目标：固定 `max_steps=10000`，细扫 `lr`（围绕 `3.5e-5`）
- 提交脚本：`deepseek/scripts/gs_sft_lora_round4_lr.sh`

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`
- 验证集比例：`SFT_VAL_RATIO=0.02`
- 预算控制：`MAX_STEPS=10000`（`EPOCHS=16` 仅作上限）
- 并行：**单卡**（`NGPU=1`, `NPROC_PER_NODE=1`）
- `TORCH_DTYPE=float32`
- `BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317532 | `3.25e-5` | 10000 | DONE | 1.2908 | 10000 | 3.64 |  |
| 317533 | `3.5e-5` | 10000 | DONE | 1.2893 | 10000 | 3.63 | 与 round3 同 lr@10k 一致 |
| 317534 | `3.75e-5` | 10000 | DONE | 1.2879 | 10000 | 3.63 | **本轮（10k 固定）lr 最优** |

## 指标位置

- `deepseek/results/sft_grid_round4/mix_chat_real_300k_lora_lr_*_s10000/train_sft.csv`
- `deepseek/results/sft_grid_round4/mix_chat_real_300k_lora_lr_*_s10000/test_sft.csv`

## 本轮结论（LoRA round4）

- 固定 `max_steps=10000` 时，`3.25e-5 → 3.5e-5 → 3.75e-5` **eval 单调变好**（`1.2908 → 1.2893 → 1.2879`），与此前「lr 略上调仍有益」的趋势一致。
- **本轮（10k 步）最优 lr：`3.75e-5`（Job 317534）**。
- 与 **round3** 对比：同 `lr=3.5e-5` 时，`10000` step 为 **1.2893**，与 round3 记录一致；**`max_steps=12000` 仍为更强配置（`eval_loss=1.2843`）**，优于本轮最优 `3.75e-5 @ 10000`（`1.2879`）。因此 **全局更优仍是「`3.5e-5` + `max_steps=12000`」**；若继续押 `3.75e-5`，建议再跑 **`max_steps=12000`（或 15000）单点** 验证是否超过 1.2843。
