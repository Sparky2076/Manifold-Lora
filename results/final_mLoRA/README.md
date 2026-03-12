# 最终 mLoRA 训练（50 epoch）

- 日期：2026-03-11
- 模型：`distilbert-base-uncased`，数据集：GLUE `sst2`
- 训练：`epochs=50, batch_size=4, grad_accum_steps=8, max_length=128, lr=1.2e-4`
- mLoRA：`lora_type=mlora, r=8, alpha=16, dropout=0.05`
- 服务器任务：Job **308296(bkilled for mlora_round_4)**，指标目录 `~/Manifold-Lora/results_final_mlora/`

跑完后将服务器上 `results_final_mlora/train.csv`、`results_final_mlora/test.csv` 拷入此目录并提交。
