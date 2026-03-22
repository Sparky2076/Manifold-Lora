## 调参日志 - mLoRA 第 3 轮（DistilBERT + SST2，lr 精调 2）

- 日期：2026-03-11
- 模型 / 数据：`distilbert-base-uncased` + GLUE `sst2`，`text_field=sentence`
- 统一训练配置：`epochs = 20, batch_size = 4, grad_accum_steps = 8, max_length = 128`
- mLoRA 统一配置：`lora_type = mlora (mlora.py), r = 8, alpha = 16, dropout = 0.05`

### lr 网格结果（mLoRA，第 3 轮）

| JobID  | lr     | Best eval acc | Best epoch | 备注                          |
|--------|--------|---------------|------------|-------------------------------|
| 308232 | 8e-5   | 0.8911        | 19         | 接近第 2 轮 1e-4 水平         |
| 308233 | 1e-4   | 0.8922        | 9          | 略优于 8e-5，提升有限         |
| 308234 | 1.2e-4 | 0.9025        | 20         | 本轮最优，仍低于 LoRA 最优    |

### 初步结论（mLoRA 第 3 轮）

mLoRA 在 8e-5 → 1.2e-4 区间内随 lr 增大继续提升，最佳 0.9025 @ 1.2e-4，但仍低于 LoRA 的 ~0.9106。三轮综合：**同配置下 LoRA 优于 mLoRA**；mLoRA 推荐 lr ≈ 1.2e-4，不再建议继续 lr 网格迭代。
