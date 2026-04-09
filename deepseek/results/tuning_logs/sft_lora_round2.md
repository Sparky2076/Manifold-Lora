# DeepSeek SFT LoRA 第二轮（fine grid，结果已回填）

- 日期：**2026-04-08**
- 目标：围绕 round1 最优学习率 `3e-5` 做细化搜索
- 提交脚本：`deepseek/scripts/gs_sft_lora_round2_fine.sh`

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
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|----------:|--------------:|------|
| 317029 | `2.5e-5` | 6000 | DONE | 1.3122 | 6000 | 3.71 |  |
| 317030 | `3e-5` | 6000 | DONE | 1.3075 | 6000 | 3.70 |  |
| 317031 | `3.5e-5` | 6000 | DONE | 1.3038 | 6000 | 3.68 | 本轮 LoRA 最优 |

## 指标位置

- `deepseek/results/sft_grid_round2/mix_chat_real_300k_lora_lr_*_s6000/train_sft.csv`
- `deepseek/results/sft_grid_round2/mix_chat_real_300k_lora_lr_*_s6000/test_sft.csv`

## 本轮结论（LoRA round2）

- 在 `max_steps=6000` 下，`2.5e-5 → 3e-5 → 3.5e-5` 单调变好（`1.3122 → 1.3075 → 1.3038`），与 round1 粗筛趋势一致。
- **当前 LoRA 推荐单点：`lr=3.5e-5`（Job 317031）**；相对 round1 最优 `3e-5 @ 3000`（`1.3306`），6000 step 下 `3e-5` 已降至 `1.3075`，说明拉长步数有效。

## 下一步建议

1. **主训/复验**：固定 `lr=3.5e-5`，将 `MAX_STEPS` 提到 `8000` 或 `10000`，看 `eval_loss` 是否继续下降或进入平台期；若明显回升再考虑略降学习率或加 dropout。
2. **可选细扫**：若仍想压最后一点，可在 `3.25e-5 / 3.5e-5 / 3.75e-5` 再跑一轮短训（如 4000 steps）快速对比；否则优先把时间花在更长步数 + 单一最优 lr 上。
3. **与 mLoRA 对比**：mLoRA round1 最优约 `1.3870 @ 1e-4, 3000 steps`；同预算下 **LoRA 仍明显更优**。待 mLoRA round2（317093–317095）跑完后可再比一轮。
