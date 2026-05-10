# DeepSeek SFT + 全因子网格（LoRA / mLoRA）

本目录用于 DeepSeek 系列模型的 SFT 训练。当前实现默认使用历史方案中的 `alpaca_train_1k` + `SFT_VAL_RATIO=0.2`，并与 `deepseek_autogrid/` 配套完成自动补齐网格。

## 单次训练（本机）

```bash
python -m deepseek.main_sft \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --sft_preset alpaca_train_1k --sft_val_ratio 0.2 \
  --max_steps 500 --eval_every 100 \
  --lora_type default --metrics_dir deepseek/results/smoke
```

输出文件：
- `train_sft.csv`（`iteration,train_loss,train_perplexity`）
- `test_sft.csv`（`iteration,eval_loss,eval_perplexity`）
- `run_meta.json`

## 服务器提交（单次）

```bash
bash deepseek/scripts/submit_bsub_sft.sh
```

可用环境变量覆盖：`MAX_STEPS`、`EVAL_EVERY`、`LORA_TYPE`、`LORA_R`、`LORA_ALPHA`、`WEIGHT_DECAY`、`METRICS_DIR` 等。

## 网格（与 DistilBERT 一致）

- LoRA: `bash scripts/server_submit_deepseek_grid.sh`
- mLoRA: `bash scripts/server_submit_deepseek_grid_mlora.sh`

说明见 [`deepseek_autogrid/README.md`](../deepseek_autogrid/README.md)。
