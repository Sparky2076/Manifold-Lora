# DistilBERT 超参设置与 Grid Search 指南（SST-2）

本文回答：

- **DistilBERT 的超参怎么设置**
- **怎么跑 grid search（哪些脚本）**
- **网格间隔多大、都搜了哪些参数**
- **grid search 跑多少 step/epoch**
- **当前最优配置是什么**
- **正式运行建议用哪套超参**

> 目标任务：`distilbert-base-uncased` 在 GLUE `sst2` 上做分类训练，对比 LoRA vs mLoRA。

## 1. 训练入口与提交方式（LSF）

- **提交脚本**：`distilbert/scripts/submit_bsub.sh`
  - 固定申请 1 卡：`NGPU=1`
- **计算节点执行脚本**：`distilbert/scripts/run_train_bsub.sh`
- **训练主程序**：`python -m distilbert.main`

数据与模型固定为：

- `--dataset_name glue --dataset_config sst2 --text_field sentence`
- `--max_length 128`

## 2. Grid Search 跑法（脚本）

### 2.1 LoRA 学习率网格

- 脚本：`distilbert/scripts/gs_lr_lora.sh`
- 网格：`LR_LIST=(2.5e-5 3e-5 3.5e-5)`
- 预算：`EPOCHS=20`
- LoRA 固定配置：`default, r=8, alpha=16, dropout=0.05`

### 2.2 mLoRA 学习率网格

- 脚本：`distilbert/scripts/gs_lr_mlora.sh`
- 网格：`LR_LIST=(1.2e-4 1.5e-4 2e-4)`
- 预算：`EPOCHS=20`
- mLoRA 固定配置：`mlora, r=8, alpha=16, dropout=0.05`

## 3. 网格间隔与搜索参数

当前 distilbert 调参主要搜索：

- **学习率 `lr`**

其余参数统一固定（确保可比性）：

- 训练：`epochs=20, batch_size=4, grad_accum_steps=8, max_length=128`
- 优化：`weight_decay=0.01, max_grad_norm=1.0`
- LoRA：`r=8, alpha=16, dropout=0.05`

> distilbert 的脚本是“按 epoch 控制预算”，不像 deepseek 主要用 `max_steps`。

## 4. Grid Search 跑了多少 step/epoch？

- distilbert 的调参默认 **都跑 `20 epochs`**（详见两个 `gs_lr_*.sh`）。
- step 数量由数据集大小与 batch/accum 决定，不在脚本里显式固定。

## 5. 当前最优配置（来自 tuning logs）

以 `Best eval acc`（越高越好）为准。

### 5.1 LoRA

从 `distilbert/results/tuning_logs/lora_round3.md`：

- best eval acc ≈ **0.9106**
- 候选 lr 在 **`2.5e-5` 与 `3e-5`** 附近形成平台

推荐：

- **LoRA 正式配置**：`lr=2.5e-5`（更保守稳定）或 `3e-5`（与部分轮次保持一致）

### 5.2 mLoRA

从 `distilbert/results/tuning_logs/mlora_round4.md`：

- best eval acc ≈ **0.9071**（`lr=2e-4`）
- 整体仍略低于 LoRA 最优（≈0.9106）

推荐：

- mLoRA 若要对比：`lr=2e-4`（当前最优点），但**最终更推荐 LoRA**。

## 6. 正式运行建议（DistilBERT + SST2）

若你只需要一套“最终训练配置”：

- **优先 LoRA**：
  - `lora_type=default, r=8, alpha=16, dropout=0.05`
  - `lr=2.5e-5`（或 `3e-5`）
  - `epochs=20, batch_size=4, grad_accum_steps=8, max_length=128`

提交参考：

- 直接跑某个 lr：修改 `LR=...` 并运行 `distilbert/scripts/submit_bsub.sh`
- 或复用网格脚本：`distilbert/scripts/gs_lr_lora.sh`

