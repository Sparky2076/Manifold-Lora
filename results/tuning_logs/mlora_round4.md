## 调参日志 - mLoRA 第 4 轮（DistilBERT + SST2，lr 上调）

- 日期：2026-03-11
- 模型 / 数据：`distilbert-base-uncased` + GLUE `sst2`，`text_field=sentence`
- 统一训练配置：`epochs = 20, batch_size = 4, grad_accum_steps = 8, max_length = 128`
- mLoRA 统一配置：`lora_type = mlora (mlora.py), r = 8, alpha = 16, dropout = 0.05`

### lr 网格结果（mLoRA，第 4 轮）

| JobID  | lr     | Best eval acc | Best epoch | 备注                         |
|--------|--------|---------------|------------|------------------------------|
| 308303 | 1.2e-4 | 0.8991        | 15         | 与 round3 最优接近          |
| 308304 | 1.5e-4 | 0.8979        | 18         | 略低，可能开始不稳定         |
| 308305 | 2e-4   | 0.9071        | 13         | 本轮最优，但仍低于 LoRA 最优 |

### 初步结论（mLoRA 第 4 轮）

在 1.2e-4 → 2e-4 区间，mLoRA 的 best eval acc 仍随 lr 上调有小幅提升，但到 2e-4（0.9071）仍略低于 LoRA 的约 0.9106，且曲线开始略显不稳。综合四轮结果，在本任务和当前配置下 **LoRA 明显优于 mLoRA**；mLoRA 若需对比，可选 `lr ≈ 1.2e-4–2e-4`，但作为最终推荐更偏向使用 LoRA 的 `lr ≈ 2.5e-5–3e-5`。

