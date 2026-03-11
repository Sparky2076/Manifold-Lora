## 调参日志 - LoRA 第 2 轮（DistilBERT + SST2，lr 精调）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，`text_field=sentence`
- 训练配置（本轮统一）：
  - `epochs = 20`
  - `batch_size = 4`
  - `grad_accum_steps = 8`
  - `max_length = 128`
  - `lr` 为本轮网格搜索变量，取值：`{1.5e-5, 2e-5, 3e-5, 4e-5}`
  - `weight_decay = 0.01`
  - `max_grad_norm = 1.0`
- LoRA 配置（本轮统一）：
  - `lora_type = default`（`lora.py`）
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - 只作用于 attention 里的 Linear（`attention_only=True`）
- 提交脚本：`scripts/gs_lr_lora.sh`（内部调用 `scripts/submit_bsub.sh` → `scripts/run_train_bsub.sh`）

### lr 网格结果（LoRA，第 2 轮）——等待实验完成

| JobID  | lr      | Best eval acc | Best epoch | 备注                   |
|--------|---------|---------------|------------|------------------------|
| 308187 | 1.5e-5  | 0.9037        | 12         | 较好，但略低于更大 lr  |
| 308188 | 2e-5    | 0.9071        | 6          | 明显优于 1.5e-5        |
| 308189 | 3e-5    | **0.9106**    | 11         | 本轮最优，且较稳定      |
| 308190 | 4e-5    | 0.9094        | 8          | 接近最优，略逊于 3e-5   |

### 初步结论（LoRA 第 2 轮）

1. 在本轮精调中，最佳点出现在 **lr = 3e-5（Job 308189, best acc = 0.9106）**，2e-5 和 4e-5 也表现接近，但略低。  
2. 相比第 1 轮的最佳（0.9048 @ lr≈2e-5/5e-5），**将 lr 提升到 3e-5 带来了小幅但稳定的提升**。  
3. 综合两轮结果，当前推荐的 LoRA 超参为：  
   - lr ≈ **3e-5**  
   - 其余保持：`epochs = 20（或更高用于最终模型）`, `batch_size = 4`, `grad_acc = 8`, `r = 8`, `alpha = 16`, `dropout = 0.05`。


