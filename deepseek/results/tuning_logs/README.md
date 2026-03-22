# DeepSeek SFT 调参日志

与 `distilbert/results/tuning_logs/` 用法相同：每轮网格或对比实验可在此新增 `sft_round1.md` 等，记录：

- 日期、模型、`SFT_PRESET` / 数据集
- 统一训练配置（epoch、batch、LoRA 等）
- 表格：JobID、lr、Best eval loss、Best epoch、备注
