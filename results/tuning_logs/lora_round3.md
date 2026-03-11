## 调参日志 - LoRA 第 3 轮（DistilBERT + SST2，lr 精调 2）

- 日期：2026-03-11
- 模型 / 数据：`distilbert-base-uncased` + GLUE `sst2`，`text_field=sentence`
- 统一训练配置：同前两轮，lr 网格为 `{2.5e-5, 3e-5, 3.5e-5}`
- LoRA 统一配置：同前两轮

### lr 网格结果（LoRA，第 3 轮）——等待实验完成

| JobID  | lr      | Best eval acc | Best epoch | 备注 |
|--------|---------|---------------|------------|------|
| 308229 | 2.5e-5  | TODO          | TODO       |      |
| 308230 | 3e-5    | TODO          | TODO       |      |
| 308231 | 3.5e-5  | TODO          | TODO       |      |

