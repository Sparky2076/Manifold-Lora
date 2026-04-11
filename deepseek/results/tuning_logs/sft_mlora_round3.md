# DeepSeek SFT mLoRA 第三轮（步数预算，结果已回填）

- 日期：**2026-04-08**
- 目标：固定 round2 最优 `lr=1.1e-4`，扫 `max_steps`（**12000 / 15000 / 18000**），观察 mLoRA 长训是否继续追赶 LoRA
- 提交脚本：`deepseek/scripts/gs_sft_mlora_round3_steps.sh`

## 统一训练配置（本轮）

- 同 round2：`mlora, r=8, alpha=16, dropout=0.05`，单卡，`TORCH_DTYPE=float32`，`BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- `LR=1.1e-4`（默认可 `LR=...` 覆盖）
- `EPOCHS=20`（仅上限，脚本默认；可按集群情况覆盖）

## Job 提交记录

| JobID | max_steps | 状态 | best eval loss | best step | best eval ppl | 备注 |
|------:|----------:|------|----------------|----------:|--------------:|------|
| 317180 | 12000 | DONE | 1.3264 | 12000 | 3.77 |  |
| 317181 | 15000 | DONE | 1.3187 | 15000 | 3.74 |  |
| 317182 | 18000 | DONE | 1.3123 | 18000 | 3.71 | 本轮 mLoRA 最优 |

## 指标位置

- `deepseek/results/sft_grid_round3/mix_chat_real_300k_mlora_lr_1p1e_4_s12000/` 等（`s15000`、`s18000`）

## 本轮结论（mLoRA round3）

- 固定 `lr=1.1e-4` 时，`12000 → 15000 → 18000` 的 **eval loss 单调下降**（`1.3264 → 1.3187 → 1.3123`），说明 **加长训练对 mLoRA 仍有效**，尚未在本轮三个点上看到平台或反弹。
- 相对 round2 同 lr、**6000 step** 的 **1.3520**，拉到 **18000** 约再降 **0.04** loss，支持你之前「mLoRA 可能需要更多 step」的判断。
- 与 **LoRA** 同数据上的当前强点（`lr=3.5e-5`，**`eval_loss=1.2843 @ max_steps=12000`**）相比，mLoRA 在 **18000 step** 仍高约 **0.028**；若继续加 step，需看 `test_sft.csv` 曲线是否仍降或开始过拟合。
- **当前 mLoRA 推荐单点（本轮内）**：`lr=1.1e-4` + **`max_steps=18000`（Job 317182）**。

## 下一轮

- 固定步数 lr 细扫（**mLoRA round 4**）：`deepseek/scripts/gs_sft_mlora_round4_lr.sh`，日志 `sft_mlora_round4.md`。
