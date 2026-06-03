# Qwen3-0.6B 调参结果（MMLU 数据管线）

## 实验设置

| 项 | 值 |
|----|-----|
| 底座 | **`Qwen/Qwen3-0.6B`**（bfloat16，eager attention） |
| 训练数据 | **`knowledge_mc_mix`**（ARC-Easy / ARC-Challenge / SciQ） |
| 适配器 | LoRA 与 mLoRA；**`lora_alpha = 2 × lora_r`**（α 不独立网格） |
| weight_decay | `{0, 0.001, 0.01}` |
| 步数 | **500 step**，每 **100 step** eval；验证比例 **10%** |
| 粗网格 | 5 lr × 3 r × 3 wd = **45** 组 |
| 细网格 | 8 lr × 2 r × 3 wd = **48** 组（r∈{32,64} → α∈{64,128}） |

## 验证指标（两阶段）

1. **SFT 验证 PPL** — `summary.csv` 中 `best_eval_perplexity`（全程最低）
2. **tinyMMLU** — 各 run 目录下 `mmlu_eval.json`；汇总为 `mmlu_summary.csv`，合并 SFT 为 `summary_mmlu.csv`

## 结果目录

| 目录 | 阶段 | 方法 | 主要文件 |
|------|------|------|----------|
| [`results_mmlu/`](results_mmlu/) | 粗网格 | LoRA | `summary.csv`, `deepseek_grid_analysis.md` |
| [`results_mmlu_refine/`](results_mmlu_refine/) | 细网格 | LoRA | 上列 + `summary_mmlu.csv`, `summary_topk_full_mmlu.csv`, `topk5_plot_bundle/` |
| [`results_mmlu_mlora/`](results_mmlu_mlora/) | 粗网格 | mLoRA | `summary.csv`, `summary_mmlu.csv`, `mmlu_summary.csv` |
| [`results_mmlu_mlora_refine/`](results_mmlu_mlora_refine/) | 细网格 | mLoRA | 同上 |

## 粗网格要点（LoRA，`results_mmlu/`，45/45 完成）

| 排名 | best PPL | 超参 |
|------|----------|------|
| 1 | **1.0344** | `lr=2e-5, r=64, α=128, wd=0` |
| 2 | 1.0345 | `lr=2e-5, r=64, α=128, wd=0.01` |

> 极低 lr（`2e-7`）组合 PPL 发散至 5–88，见 analysis 尾部。

## 细网格要点（LoRA，`results_mmlu_refine/`，48/48 完成）

| 排名 | best PPL | tinyMMLU（best） | 超参 |
|------|----------|------------------|------|
| 1 | **1.0352** | 0.4968 | `lr=3e-4, r=32, α=64, wd=0.001` |
| 2 | 1.0352 | — | `lr=2e-4, r=64, α=128, wd=0.001` |

分析：[`results_mmlu_refine/deepseek_grid_analysis.md`](results_mmlu_refine/deepseek_grid_analysis.md)

## LoRA vs mLoRA 对比

详见 [`mlora_vs_lora_comparison.md`](mlora_vs_lora_comparison.md)（2026-05-28 汇总）：

| 指标 | LoRA | mLoRA | 结论 |
|------|------|-------|------|
| 细网格 best PPL | **1.0352** | 1.0353 | LoRA 略优 |
| 细网格 best tinyMMLU | 49.68% | **54.28%** | mLoRA **+4.6 pp** |
| 细网格 Top-5 tinyMMLU mean | 48.85% | **53.97%** | mLoRA **+5.1 pp** |
| Spearman ρ（48 组配对） | — | — | **0.846**（排序高度一致） |

## 最终 benchmark（全量 MMLU）

文件：[`results_mmlu_refine/summary_topk_full_mmlu.csv`](results_mmlu_refine/summary_topk_full_mmlu.csv)

| 项 | 值 |
|----|-----|
| 选取 | 细网格 tinyMMLU **Top-5** LoRA run |
| LoRA Top-5 **全量 MMLU mean** | **0.4543**（5/5 完成） |
| 最佳单 run full MMLU | **0.4666** — `lr=3e-4, r=32, α=64, wd=0.001` |
| mLoRA Top-5 full MMLU | 投递中/未完成（对比 md 中标记 pending） |

可视化 bundle：[`results_mmlu_refine/topk5_plot_bundle/`](results_mmlu_refine/topk5_plot_bundle/)

- `best_run_sft_trend.png` — Top run 验证 PPL 曲线
- `best_run_mmlu_acc_trend.png` — MMLU 准确率趋势
- `sft_curves/` — Top-5 部分 run 的 train/test SFT CSV
- `summary_topk.csv`, `topk_manifest.json`

## 尚未纳入本归档的 benchmark

以下在集群脚本中定义，**结果 CSV 未 push 到本仓库**：

- **BBH** Top-K（`server_submit_qwen3_bbh_topk_from_summary.sh`）
- **GSM8K** Top-3 refine（`server_submit_qwen3_gsm8k_top3_refine.sh`）

## 文件列说明（`summary.csv`）

| 列 | 含义 |
|----|------|
| `best_eval_perplexity` | 验证集最低 PPL |
| `best_iteration` | 达到最优 PPL 的 step |
| `last_eval_perplexity` | 第 500 step 的 PPL |
| `sft_preset` | 固定为 `knowledge_mc_mix` |
| `lora_type` | `default`（LoRA）或 mLoRA 标识 |
| `status` | `ok` 表示 500 step 完整且指标有限 |

## 备注

- 不含 checkpoint（`sft_lora_state.pt`）与训练代码。
- 粗网格 LoRA **未**跑 tinyMMLU；mLoRA 粗网格有完整 tinyMMLU（45/45）。
