# DeepSeek SFT LoRA 大集第一轮（scheme A，结果待回填）

- 日期：**2026-04-07**（本表为当前有效记录；此前因 OOM / Hub / 混数据 schema 等失败的提交已从表中移除，以本批为准。）
- 任务：`deepseek/scripts/gs_lr_deepseek_sft_big_v1.sh`
- 预设：`SFT_PRESET=mix_chat_real_300k`
- 验证集比例：`SFT_VAL_RATIO=0.02`
- 预算控制：`MAX_STEPS in {2000, 4000}`（`EPOCHS=99` 仅作上限）
- 训练：**多卡 DDP**（脚本默认 `NGPU=2`、`NPROC_PER_NODE=2`，`torchrun`）
- LoRA：`default, r=8, alpha=16, dropout=0.05`

## Job 提交记录

| JobID | lr | max_steps | 状态 | best eval loss | best epoch | best eval ppl | 备注 |
|------:|---:|----------:|------|----------------|-----------:|--------------:|------|
| 316725 | `2e-5` | 2000 | RUN |  |  |  | 2026-04-07 14:22 提交 |
| 316726 | `2e-5` | 4000 | RUN |  |  |  | 2026-04-07 14:22 提交 |
| 316727 | `2.5e-5` | 2000 | RUN |  |  |  | 2026-04-07 14:22 提交 |
| 316728 | `2.5e-5` | 4000 | RUN |  |  |  | 2026-04-07 14:22 提交 |

## 回填说明

- 作业结束后，补充每个 Job 的 `best eval loss`、最佳 epoch、`best eval ppl` 与简要结论。
- 指标路径：`deepseek/results/sft_grid_big/mix_chat_real_300k_lora_big_lr_*_steps_*/train_sft.csv`、`test_sft.csv`。
