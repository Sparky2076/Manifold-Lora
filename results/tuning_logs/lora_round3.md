## 调参日志 - LoRA 第 3 轮（DistilBERT + SST2，lr 精调 2）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，`text_field=sentence`
- 训练配置（本轮统一）：
  - `epochs = 20`
  - `batch_size = 4`
  - `grad_accum_steps = 8`
  - `max_length = 128`
  - `lr` 为本轮网格搜索变量，取值：`{2.5e-5, 3e-5, 3.5e-5}`
  - `weight_decay = 0.01`
  - `max_grad_norm = 1.0`
- LoRA 配置（本轮统一）：
  - `lora_type = default`（`lora.py`）
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - 只作用于 attention 里的 Linear（`attention_only=True`）
- 提交脚本：`scripts/gs_lr_lora.sh`（内部调用 `scripts/submit_bsub.sh` → `scripts/run_train_bsub.sh`）

### lr 网格结果（LoRA，第 3 轮）——等待实验完成

| JobID  | lr      | Best eval acc | Best epoch | 备注 |
|--------|---------|---------------|------------|------|
| 308229 | 2.5e-5  | TODO          | TODO       |      |
| 308230 | 3e-5    | TODO          | TODO       |      |
| 308231 | 3.5e-5  | TODO          | TODO       |      |

