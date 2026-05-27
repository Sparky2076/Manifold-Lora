# Qwen3-0.6B（仅 MMLU 数据管线）

**训练数据**一律为 **`knowledge_mc_mix`**（ARC-Easy / ARC-Challenge / SciQ）。LoRA：**`lora_alpha = 2 × lora_r`**（不再对 α 独立网格）。  
**weight_decay**：`0.0, 0.001, 0.01`（3 档）。超参筛选仍以 **tinyMMLU → summary_mmlu → 全量 MMLU** 为准。

与 [`deepseek_autogrid`](../deepseek_autogrid/README.md) 共用 [`run_grid_bsub.sh`](../deepseek_autogrid/run_grid_bsub.sh)。

- **底座**：`Qwen/Qwen3-0.6B`；**dtype**：默认 `bfloat16`；**`TRUST_REMOTE_CODE=1`**  
- **LoRA 目标**：`q_proj,...,down_proj`（可 `LORA_TARGETS` 覆盖）  
- **`config_mmlu.py`**：`config.py` 的兼容 shim

## 网格规模（约）

| 阶段 | LR 档 × r 档 × WD 档 | 组合数 |
|------|---------------------|--------|
| 粗网格 | 5 × 3 × 3 | **45** |
| 细网格 | 8 × 2 × 3 | **48** |

（细网格：`r ∈ {32, 64}` → `α ∈ {64, 128}`。）

## 目录与结果

| 阶段 | 脚本 | 输出 |
|------|------|------|
| 粗网格 LoRA | `run_lora_grid_bsub.sh` | `qwen3_autogrid/results_mmlu/` |
| 粗网格 mLoRA | `run_mlora_grid_bsub.sh` | `qwen3_autogrid/results_mmlu_mlora/` |
| 细网格 LoRA | `run_refine_grid_bsub.sh` | `qwen3_autogrid/results_mmlu_refine/` |
| 细网格 mLoRA | `run_mlora_refine_grid_bsub.sh` | `qwen3_autogrid/results_mmlu_mlora_refine/` |

## 登录节点：缓存 → smoke → Phase D（nohup 粗网格）

1. **`scripts/cluster_hf_cache_env.sh`**：未设置时在登录节点与工作节点上使用同一套默认路径  
   **`HF_HOME` / `HF_DATASETS_CACHE`**（若存在 **`/nfsshare/home/$USER`** 则用其下 **`.cache/huggingface`**）；并默认 **`HF_HUB_DOWNLOAD_TIMEOUT=300`**。  
   可选：若 etag 校验易卡住，可手动 **`export HF_HUB_ETAG_TIMEOUT=120`**（或按需）。
2. **`scripts/cache_hf_qwen3_login.sh`**：**镜像优先**（默认 **`HF_ENDPOINT=https://hf-mirror.com`**，除非已显式设定）；失败后回退官方。**`FORCE_OFFICIAL_HF_ONLY=1`** 时仅用官方站。  
   预缓存里 tokenizer 使用 **`use_fast=False`**，降低镜像损坏 fast tokenizer 的概率（训练侧 **`models_sft`** 同步时同理）。
3. **`scripts/cluster_qwen3_login_lora_pipeline.sh`** / **`scripts/cluster_qwen3_smoke_then_coarse_grid.sh`**：Phase B 缓存 → Phase C **`knowledge_mc_mix`** smoke → 成功后才 **Phase D nohup** 投递 **`server_submit_qwen3_grid.sh`**。**`bsub`** 作业通过 **`deepseek/scripts/submit_bsub_sft.sh`** 带入 **`HF_HOME`** 与 **`HF_DATASETS_CACHE`**，避免计算节点重复全量下载（共享 NFS 典型）。

一键命令见下一节。

## 一键：冒烟通过后再投粗网格

```bash
bash scripts/cluster_qwen3_smoke_then_coarse_grid.sh
```

等价于「cache + smoke 成功后才 `SUBMIT_GRID_AFTER_SMOKE=1`」；冒烟失败会直接 **exit**，**不会** `nohup` 网格。

也可手动：

```bash
SUBMIT_GRID_AFTER_SMOKE=1 bash scripts/cluster_qwen3_login_lora_pipeline.sh
```

`SUBMIT_MMLU_GRID_AFTER_SMOKE=1` 与 `SUBMIT_GRID_AFTER_SMOKE=1` 等价（兼容旧名）。

**预缓存**：`bash scripts/cache_hf_qwen3_login.sh`  

**tinyMMLU 扫描（与当前粗网格 run 名一致）**：`bash scripts/server_submit_qwen3_mmlu_coarse_eval.sh`

## 本机汇总

```bash
python -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config

python -m deepseek_autogrid.analyze_results \
  --config-module qwen3_autogrid.config \
  --summary qwen3_autogrid/results_mmlu/summary.csv

python -m deepseek_autogrid.aggregate_mmlu_results --results-root qwen3_autogrid/results_mmlu
python scripts/join_sft_mmlu_summary.py \
  --sft-summary qwen3_autogrid/results_mmlu/summary.csv \
  --mmlu-summary qwen3_autogrid/results_mmlu/mmlu_summary.csv \
  --output qwen3_autogrid/results_mmlu/summary_mmlu.csv

# 细网格
python -m deepseek_autogrid.aggregate_results \
  --config-module qwen3_autogrid.config_refine \
  --results-root qwen3_autogrid/results_mmlu_refine
python -m deepseek_autogrid.analyze_results \
  --config-module qwen3_autogrid.config_refine \
  --summary qwen3_autogrid/results_mmlu_refine/summary.csv
```

定稿评测：**全量 MMLU** → `scripts/server_submit_qwen3_mmlu_topk_from_summary.sh`；**BBH** → `scripts/server_submit_qwen3_bbh_topk_from_summary.sh`；**GSM8K Top-K** → `scripts/server_submit_qwen3_gsm8k_top3_refine.sh`；一键 **MMLU+BBH+GSM8K**：`scripts/server_submit_qwen3_final_benchmark_topk.sh`。

## mLoRA 三阶段管线（与 LoRA 同网格 / 同数据）

超参与 LoRA 相同（粗 45 + 细 48，`knowledge_mc_mix`，`alpha=2*r`）。共用 `deepseek_autogrid/run_grid_bsub.sh`，通过 **`LORA_TYPE=mlora`** 与 **`RESULTS_ROOT`** 区分目录。

| 阶段 | 集群入口 | 结果目录 |
|------|----------|----------|
| 粗网格 SFT | `bash scripts/cluster_qwen3_mlora_smoke_then_coarse_grid.sh` | `results_mmlu_mlora/` |
| 细网格 SFT | `bash scripts/cluster_qwen3_mlora_refine_smoke_then_grid.sh` | `results_mmlu_mlora_refine/` |
| tinyMMLU | `LORA_TYPE=mlora bash scripts/server_submit_qwen3_mmlu_refine_eval.sh` | 各 run 下 `mmlu_eval.json` |
| Top-5 全量 MMLU | `bash scripts/cluster_qwen3_mmlu_mlora_full_top5_submit.sh` | `summary_mmlu.csv` + `topk5_plot_bundle/` |

**监控（集群）**

```bash
tail -f ~/Manifold-Lora/qwen3_mmlu_mlora_grid_submit.log
LORA_TYPE=mlora python3 ~/Manifold-Lora/scripts/_cluster_summarize_qwen3_complete.py
bjobs | grep qwen3_mmlu_grid_mlora
bash scripts/grid_submitter_status.sh
```

**本机汇总（粗/细）**

```bash
python -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config \
  --results-root qwen3_autogrid/results_mmlu_mlora
python -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config_refine \
  --results-root qwen3_autogrid/results_mmlu_mlora_refine
python -m deepseek_autogrid.aggregate_mmlu_results --results-root qwen3_autogrid/results_mmlu_mlora_refine
python scripts/join_sft_mmlu_summary.py \
  --sft-summary qwen3_autogrid/results_mmlu_mlora_refine/summary.csv \
  --mmlu-summary qwen3_autogrid/results_mmlu_mlora_refine/mmlu_summary.csv \
  --output qwen3_autogrid/results_mmlu_mlora_refine/summary_mmlu.csv
```

补投未完成粗网格：`bash scripts/cluster_nohup_qwen3_grid_mlora_rerun_incomplete.sh`（勿杀 RUN/PEND 训练 job）。

节流向：`GRID_MAX_PEND`、`GRID_MAX_RUN`、`SUBMIT_SLEEP_SEC`。

**遗留目录**：旧的粗网格目录（例如按 90 组或自由 α 跑出来的 run）可能与当前 `alpha=2*r` 命名 **不一致**，`aggregate_results` 只会收录与当前 `iter_grid()` 匹配的 run 名。
