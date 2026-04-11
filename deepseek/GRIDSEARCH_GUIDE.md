# DeepSeek（SFT）超参设置与 Grid Search 指南

本文回答：

- **怎么跑 grid search（哪些脚本）**
- **网格间隔多大、都搜了哪些参数**
- **grid search 跑多少 step/epoch**
- **当前最优配置是什么**
- **正式运行建议用哪套超参**

> 目标任务：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 在 `mix_chat_real_300k` 上做 SFT（LoRA / mLoRA）。

## 1. 训练入口与提交方式（LSF）

- **提交脚本（申请资源）**：`deepseek/scripts/submit_bsub_sft.sh`
- **计算节点启动训练**：`deepseek/scripts/run_deepseek_sft_bsub.sh`
- **训练主程序**：`python -m deepseek.main_sft`

关键环境变量：

- **`MAX_STEPS`**：训练预算的“硬阀门”（优先于 `EPOCHS`）。
- **`EPOCHS`**：外层上限（一般设置为 12–20 即可）。
- **`NGPU` / `NPROC_PER_NODE`**：多卡 DDP 用（`NPROC_PER_NODE>1` 时走 `torchrun`）。

## 2. 我们的 Grid Search 策略（分阶段）

### 2.1 粗筛（coarse）

目的：快速定位学习率大致区间（单卡并发友好）。

- LoRA coarse：`deepseek/scripts/gs_sft_lora_grid3_coarse.sh`
- mLoRA coarse：`deepseek/scripts/gs_sft_mlora_grid3_coarse.sh`

默认设置：

- **`MAX_STEPS=3000`**，`EPOCHS=8`
- `TORCH_DTYPE=float32`，`BATCH_SIZE=2`，`GRAD_ACCUM_STEPS=8`，`MAX_LENGTH=512`
- `r=8, alpha=16, dropout=0.05`

coarse 的学习率网格：

- **LoRA**：`1e-5 / 2e-5 / 3e-5`
- **mLoRA**：`5e-5 / 8e-5 / 1e-4`

### 2.2 细扫（fine）

目的：围绕 coarse 最优点做 3 点细化，并把预算提升到 **6000**。

- LoRA round2：`deepseek/scripts/gs_sft_lora_round2_fine.sh`（`2.5e-5 / 3e-5 / 3.5e-5`，`MAX_STEPS=6000`）
- mLoRA round2：`deepseek/scripts/gs_sft_mlora_round2_fine.sh`（`9e-5 / 1e-4 / 1.1e-4`，`MAX_STEPS=6000`）

### 2.3 预算 sweep（只调 step，不调 lr）

目的：验证“步数是否仍在带来收益”，判断平台期。

- LoRA round3：`deepseek/scripts/gs_sft_lora_round3_steps.sh`（固定 `lr=3.5e-5`，扫 `8000/10000/12000`）
- mLoRA round3：`deepseek/scripts/gs_sft_mlora_round3_steps.sh`（固定 `lr=1.1e-4`，扫 `12000/15000/18000`）

### 2.4 固定 step 的 lr 细扫（round4）

目的：当步数已经明确/固定后，再压最后一点 lr。

- LoRA round4：`deepseek/scripts/gs_sft_lora_round4_lr.sh`
  - 默认：固定 `MAX_STEPS=10000`
  - lr：`3.25e-5 / 3.5e-5 / 3.75e-5`

### 2.5 下一轮固定步数对照（LoRA round5 / mLoRA round4）

- LoRA round5：`deepseek/scripts/gs_sft_lora_round5_lr.sh`
  - 默认：固定 `MAX_STEPS=6000`，lr：`3.75e-5 / 4e-5 / 4.25e-5`
- mLoRA round4：`deepseek/scripts/gs_sft_mlora_round4_lr.sh`
  - 默认：固定 `MAX_STEPS=10000`，lr：`1.1e-4 / 1.3e-4 / 1.5e-4`
  - 兼容别名：`deepseek/scripts/gs_sft_mlora_round5_lr.sh`（转发到 round4 脚本）

## 3. 网格间隔与“都搜了哪些参数”

当前调参阶段主要搜索：

- **学习率 `lr`**（最核心）
- **训练预算 `max_steps`**（判断平台期）

其余参数在阶段内保持固定，确保可比性：

- LoRA/mLoRA：`r=8, alpha=16, dropout=0.05`
- 数据：`SFT_PRESET=mix_chat_real_300k`, `SFT_VAL_RATIO=0.02`
- 训练：`BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=8`, `MAX_LENGTH=512`, `TORCH_DTYPE=float32`

> 如需扩大搜索维度（例如 `dropout`、`r/alpha`），建议在 lr+step 稳定后再做，避免维度爆炸。

## 4. 当前最优配置（截至已完成实验）

以 `eval_loss`（越低越好）为准：

- **LoRA**：
  - round2（6000 step）最佳：`lr=3.5e-5`，`eval_loss=1.3038`
  - round3（步数 sweep）最佳：`lr=3.5e-5 + max_steps=12000`，`eval_loss=1.2843`（**当前全局最优**）
  - round4（固定 `max_steps=10000`，lr 细扫）最佳：`lr=3.75e-5`，`eval_loss=1.2879`（在 10k 步约束下优于 `3.25e-5/3.5e-5`，但仍略逊于 round3 的 12k 点）
  - 详见：`deepseek/results/tuning_logs/sft_lora_round2.md`、`sft_lora_round3.md`、`sft_lora_round4.md`

- **mLoRA**：
  - round2（6000 step）最佳：`lr=1.1e-4`，`eval_loss=1.3520`
  - round3（步数 sweep）最佳：`lr=1.1e-4 + max_steps=18000`，`eval_loss=1.3123`
  - 详见：`deepseek/results/tuning_logs/sft_mlora_round2.md`、`sft_mlora_round3.md`

结论（当前）：

- **同任务设置下 LoRA 仍优于 mLoRA**（以各自当前最优比：约 `1.2843` vs `1.3123`，差距约 **0.028**）；mLoRA 通过更长 `max_steps` 明显缩小了与早期 mLoRA 的差距，但仍未反超 LoRA。

## 5. Grid Search 时跑了多少 step/epoch？

- coarse：`MAX_STEPS=3000`（`EPOCHS=8` 仅上限）
- fine：`MAX_STEPS=6000`（`EPOCHS=12` 仅上限）
- step sweep：LoRA `8000/10000/12000`；mLoRA `12000/15000/18000`
- 固定 step 的 lr sweep：`MAX_STEPS=10000`

## 6. 正式运行（推荐）

若你现在要“正式跑一个最强 LoRA 版本”（在现阶段参数空间内）：

- **LoRA**：`lr=3.5e-5, r=8, alpha=16, dropout=0.05`
- **max_steps**：建议先用 **`12000`**（已验证更优）；若后续仍单调下降，可继续加到 15000–18000 观察平台

可选：若你希望进一步验证 round4 的 `3.75e-5` 是否能在更长步数上超过 `3.5e-5 @ 12000`，建议单独提交 **`lr=3.75e-5, max_steps=12000`**（或 15000）做对照。

执行方式参考对应脚本：

- 固定 lr + 扫 step：`deepseek/scripts/gs_sft_lora_round3_steps.sh`
- 固定 step + 扫 lr：`deepseek/scripts/gs_sft_lora_round4_lr.sh`

