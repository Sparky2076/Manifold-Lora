## 调参日志 - mLoRA 第 2 轮（DistilBERT + SST2，lr 精调）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，`text_field=sentence`
- 训练配置（本轮统一）：
  - `epochs = 20`
  - `batch_size = 4`
  - `grad_accum_steps = 8`
  - `max_length = 128`
  - `lr` 为本轮网格搜索变量，取值：`{3e-5, 5e-5, 7e-5, 1e-4}`
  - `weight_decay = 0.01`
  - `max_grad_norm = 1.0`
- LoRA 配置（本轮统一）：
  - `lora_type = mlora`（`mlora.py`）
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - 只作用于 attention 里的 Linear（`attention_only=True`）
- 提交脚本：`scripts/gs_lr_mlora.sh`（内部调用 `scripts/submit_bsub.sh` → `scripts/run_train_bsub.sh`）

### lr 网格结果（mLoRA，第 2 轮）——等待实验完成

| JobID  | lr    | Best eval acc | Best epoch | 备注                     |
|--------|-------|---------------|------------|--------------------------|
| 308213 | 3e-5  | 0.8635        | 18         | 明显好于 5e-6，但偏低    |
| 308214 | 5e-5  | 0.8750        | 12         | 继续提升                 |
| 308215 | 7e-5  | 0.8842        | 19         | 表现更好，但接近饱和     |
| 308216 | 1e-4  | 0.8933        | 11         | 本轮最优，仍略低于 LoRA  |

### 初步结论（mLoRA 第 2 轮）

1. 在更大的学习率范围内，mLoRA 的 best eval acc 随 lr 从 3e-5 增大到 1e-4 基本**单调提升**，最佳出现在 **lr = 1e-4（0.8933）**。  
2. 与 LoRA 第 2 轮最佳（0.9106 @ lr=3e-5）相比，**同等配置下 mLoRA 仍略逊一筹**，但已经明显好于其自身第 1 轮的 5e-5（0.8807）。  
3. 若后续主要目标是“效果最优”，推荐优先采用 LoRA（lr≈3e-5）；若想研究 mLoRA 特性，可在 lr≈1e-4 附近进一步加密或尝试更长 epoch。  

