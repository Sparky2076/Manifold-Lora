## 调参日志 - LoRA 第 1 轮（DistilBERT + SST2，lr 网格）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，`text_field=sentence`
- 训练配置（本轮统一）：
  - `epochs = 20`
  - `batch_size = 4`
  - `grad_accum_steps = 8`
  - `max_length = 128`
  - `lr` 为本轮网格搜索变量
  - `weight_decay = 0.01`
  - `max_grad_norm = 1.0`
- LoRA 配置（本轮统一）：
  - `lora_type = default`（`lora.py`）
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - 只作用于 attention 里的 Linear（`attention_only=True`）
- 提交脚本：`scripts/gs_lr_lora.sh`（内部调用 `scripts/submit_bsub.sh` → `scripts/run_train_bsub.sh`）

### lr 网格结果（LoRA，第 1 轮）

| JobID  | lr    | Best eval acc | Best epoch | 备注 |
|--------|-------|---------------|------------|------|
| 308004 | 5e-6  | 0.8830        | 16         | 收敛较慢，精度略低 |
| 308005 | 1e-5  | 0.8922        | 16         | 明显优于 5e-6 |
| 308006 | 2e-5  | **0.9048**    | 19         | 本轮最优，曲线较稳定 |
| 308007 | 5e-5  | **0.9048**    | 9          | 峰值与 2e-5 持平，后期有轻微波动 |

### 初步结论（LoRA）

1. 在当前配置下，**过小的学习率（5e-6）收敛慢，最终精度明显偏低**。  
2. `1e-5` 明显好于 `5e-6`，但仍略低于较大的 `2e-5` / `5e-5`。  
3. `2e-5` 与 `5e-5` 的 **最佳 eval acc 都是 0.9048**，但：
   - `2e-5` 在后期更平稳；
   - `5e-5` 在较早 epoch 即达到较高精度，但曲线略有抖动（有轻微过拟合风险）。

综合考虑稳定性和精度，**后续 LoRA 调参可以优先围绕 `lr ≈ 2e-5` 做更细的微调**，例如：

- 第二轮 LoRA 建议的 lr 网格（示例）：`[1.5e-5, 2e-5, 3e-5, 4e-5]`，其余配置不变。

