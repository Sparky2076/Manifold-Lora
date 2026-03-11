## 调参日志 - mLoRA 第 1 轮（DistilBERT + SST2，lr 网格）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，`text_field=sentence`
- 训练配置（本轮统一）：
  - `epochs = 20`
  - `batch_size = 4`
  - `grad_accum_steps = 8`
  - `max_length = 128`
  - `lr` 为本轮网格搜索变量，取值：`{5e-6, 1e-5, 2e-5, 5e-5}`
  - `weight_decay = 0.01`
  - `max_grad_norm = 1.0`
- LoRA 配置（本轮统一）：
  - `lora_type = mlora`（`mlora.py`）
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - 只作用于 attention 里的 Linear（`attention_only=True`）
- 提交脚本：`scripts/gs_lr_mlora.sh`（内部调用 `scripts/submit_bsub.sh` → `scripts/run_train_bsub.sh`）

### lr 网格结果（mLoRA，第 1 轮）——等待实验完成

| JobID  | lr    | Best eval acc | Best epoch | 备注                 |
|--------|-------|---------------|------------|----------------------|
| 308107 | 5e-6  | 0.5596        | 16         | 收敛很慢，精度较低   |
| 308108 | 1e-5  | 0.8314        | 13         | 明显好于 5e-6        |
| 308109 | 2e-5  | 0.8498        | 16         | 继续提升，但有限     |
| 308110 | 5e-5  | 0.8807        | 18         | 本轮最优，仍低于 LoRA |

### 初步结论（mLoRA）

1. 很小的学习率（\(5\times10^{-6}\)）下，mLoRA 收敛极慢，20 epoch 只到约 0.56。  
2. 随着学习率增大到 \(1\times10^{-5}, 2\times10^{-5}, 5\times10^{-5}\)，best eval acc 持续提升，在 \(5\times10^{-5}\) 达到本轮最优 0.8807。  
3. 与 LoRA 第 1 轮最佳（0.9048 @ lr≈2e-5/5e-5）相比，**当前同配置下 mLoRA 略逊一筹，但趋势显示 mLoRA 可能需要更偏大的 lr 才能充分发挥。**

后续 mLoRA 调参可以围绕本轮最优附近进一步加密，例如第二轮候选 lr：`[3e-5, 5e-5, 7e-5, 1e-4]`，其余配置不变。

