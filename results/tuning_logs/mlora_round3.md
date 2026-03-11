## 调参日志 - mLoRA 第 3 轮（DistilBERT + SST2，lr 精调 2）

- 日期：2026-03-11
- 模型 / 数据：`distilbert-base-uncased` + GLUE `sst2`，`text_field=sentence`
- 统一训练配置：`epochs = 20, batch_size = 4, grad_accum_steps = 8, max_length = 128`
- mLoRA 统一配置：`lora_type = mlora (mlora.py), r = 8, alpha = 16, dropout = 0.05`

### lr 网格结果（mLoRA，第 3 轮）——等待实验完成

| JobID  | lr     | Best eval acc | Best epoch | 备注 |
|--------|--------|---------------|------------|------|
| 308232 | 8e-5   | TODO          | TODO       |      |
| 308233 | 1e-4   | TODO          | TODO       |      |
| 308234 | 1.2e-4 | TODO          | TODO       |      |

