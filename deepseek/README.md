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
- **`sft_lora_state.pt`**：仅含训练得到的 `lora_A`/`lora_B` 键，供后续 **合并为 HF**（不再使用 `lora_adapter.pt` 命名）。若未写出任何 LoRA 张量，训练进程会以非零退出码失败。
- **`model_merged_hf/`**：**不在**默认 SFT 结束时生成；在跑 BBH 或显式导出时由下面命令写入。

## 网格后 Top-K：先合并再 BBH（推荐）

1. 跑完网格并 `aggregate_results` 后，对 Top-K 目录生成合并权重：

```bash
SUMMARY_CSV=deepseek_autogrid/results/summary.csv TOP_K=10 \
  bash scripts/export_merged_deepseek_topk.sh
```

或单个 run：

```bash
python -m deepseek.export_merged_hf --metrics_dir deepseek_autogrid/results/<run_name> --trust_remote_code
```

2. 再对同一目录跑 BBH（仅 lm-eval，要求已有 `model_merged_hf/config.json`）：

```bash
export SKIP_MERGE=1
export METRICS_DIR=deepseek_autogrid/results/<run_name>
bash deepseek/scripts/run_deepseek_bbh_bsub.sh
```

一步提交 Top-K 合并 + BBH 作业：见 `scripts/server_submit_deepseek_bbh_topk_from_summary.sh`（默认先 `export_merged` 再 `SKIP_MERGE=1` 递交）。

## BBH（lm-eval）

依赖：`pip install lm-eval`（及 `transformers`、`torch`；建议 `lm_eval[hf]`）。

单机：若尚未合并，会先合并再评测；若已合并则只跑评测：

```bash
python -m deepseek.eval_bbh --metrics_dir deepseek_autogrid/results/<run_name> --trust_remote_code
```

仅评测已有合并目录：

```bash
python -m deepseek.eval_bbh --metrics_dir <run_dir> --merged_hf_dir <run_dir>/model_merged_hf --trust_remote_code
```

集群：设置 `METRICS_DIR`（或 `ADAPTER_PATH`）后：

```bash
bash deepseek/scripts/submit_bsub_bbh.sh
```

`SKIP_MERGE=1`、`FORCE_REMERGE=1`、`MERGED_HF_DIR` 可经 `run_deepseek_bbh_bsub.sh` 传给 `eval_bbh`。

## 服务器提交（单次 SFT）

```bash
bash deepseek/scripts/submit_bsub_sft.sh
```

可用环境变量覆盖：`MAX_STEPS`、`EVAL_EVERY`、`LORA_TYPE`、`LORA_R`、`LORA_ALPHA`、`WEIGHT_DECAY`、`METRICS_DIR` 等。

## 网格入口

- LoRA: `bash scripts/server_submit_deepseek_grid.sh`
- mLoRA: `bash scripts/server_submit_deepseek_grid_mlora.sh`

说明见 [`deepseek_autogrid/README.md`](../deepseek_autogrid/README.md)。
