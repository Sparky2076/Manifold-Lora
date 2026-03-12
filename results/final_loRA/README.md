# 最终 LoRA 训练（50 epoch）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`，数据集：GLUE `sst2`
- 训练：`epochs=50, batch_size=4, grad_accum_steps=8, max_length=128, lr=2.5e-5`
- LoRA：`lora_type=default, r=8, alpha=16, dropout=0.05`
- 服务器任务：Job **308295**，指标目录 `~/Manifold-Lora/results_final_loRA/`，Best eval acc ≈ **0.9071 @ epoch 14**

跑完后将服务器上 `results_final_loRA/train.csv`、`results_final_loRA/test.csv` 拷入此目录并提交。
