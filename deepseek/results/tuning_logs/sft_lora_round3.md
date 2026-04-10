# DeepSeek SFT LoRA 第三轮（步数预算，结果已回填）

- 日期：**2026-04-08**
- 目标：固定 `lr=3.5e-5`，扫 `max_steps`，观察 eval 是否继续改善或平台期
- 提交脚本：`deepseek/scripts/gs_sft_lora_round3_steps.sh`

## 统一训练配置（本轮）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 数据预设：`SFT_PRESET=mix_chat_real_300k`
- 验证集比例：`SFT_VAL_RATIO=0.02`
- 学习率：**固定** `LR=3.5e-5`（round2 最优）
- 步数：`STEPS_LIST=8000 10000 12000`（以实际提交为准）
- `EPOCHS=16`（仅上限，训练由 `max_steps` 截断）
- 并行：**单卡**（`NGPU=1`, `NPROC_PER_NODE=1`）
- `TORCH_DTYPE=float32`
- `BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

脚本为先 `echo` 再 `bsub`，故 JobID 与步数对应为：第一轮提交 → 8000，第二轮 → 10000，第三轮 → 12000。

| JobID | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|----------:|------|----------------|----------:|--------------:|------|
| 317152 | 8000 | DONE | 1.2955 | 8000 | 3.65 |  |
| 317153 | 10000 | DONE | 1.2893 | 10000 | 3.63 |  |
| 317154 | 12000 | DONE | 1.2843 | 12000 | 3.61 | 本轮最优 |

## 指标位置

- `deepseek/results/sft_grid_round3/mix_chat_real_300k_lora_lr_3p5e_5_s8000/`
- `deepseek/results/sft_grid_round3/mix_chat_real_300k_lora_lr_3p5e_5_s10000/`
- `deepseek/results/sft_grid_round3/mix_chat_real_300k_lora_lr_3p5e_5_s12000/`

## 本轮结论（LoRA round3）

- 固定 `lr=3.5e-5` 时，`8000 → 10000 → 12000` 步 eval **持续下降**（`1.2955 → 1.2893 → 1.2843`），未出现平台或反弹迹象。
- 相对 round2 同 lr（`6000` step 的 **1.3038**），继续加长训练到 **12000** 可再降约 **0.02** loss。
- **当前 LoRA 推荐单点：`lr=3.5e-5` + `max_steps=12000`（Job 317154）**；若算力允许，可再试 `15000` 单点确认是否仍下降或已接近平台。
