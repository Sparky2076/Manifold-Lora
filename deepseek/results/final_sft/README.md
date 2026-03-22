# 最终 DeepSeek SFT 训练（长 epoch）

与 `results/final_loRA/README.md` 形式一致，用于归档「选定 lr 等超参后」的完整一次训练。

- 日期：（填写）
- 模型：（如 `DeepSeek-R1-Distill-Qwen-1.5B` 本地路径或 Hub id）
- 数据：`SFT_PRESET` / Hub 数据集名
- 训练：`epochs=, batch_size=, grad_accum_steps=, max_length=, lr=`
- LoRA：`lora_type=, r=, alpha=, dropout=`
- 服务器任务：Job **（填写）**，指标目录（填写），Best eval loss ≈ **（填写）@ epoch （填写）**

跑完后将服务器上对应目录的 `train_sft.csv`、`test_sft.csv` 拷入本目录并提交 GitHub。
