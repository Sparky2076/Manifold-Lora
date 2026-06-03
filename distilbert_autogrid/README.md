# DistilBERT 调参结果（LoRA / mLoRA）

## 实验设置

| 项 | 值 |
|----|-----|
| 底座 | `distilbert-base-uncased` |
| 任务 | GLUE **SST-2** 二分类 |
| 适配器 | LoRA 与 mLoRA（`lora_type` 列区分） |
| 搜索空间 | `lr × lora_r × lora_alpha × weight_decay`（Adam β1/β2 固定 0.9 / 0.999） |
| 粗网格 epoch | 3（约 **375** 组 LoRA；mLoRA 同空间） |
| 验证 | 训练集 20% 划分验证集；指标 **`best_val_acc`**（全程最高验证准确率） |

## 结果目录

| 目录 | 阶段 | 组合数（summary 行） | 内容 |
|------|------|----------------------|------|
| [`results/`](results/) | LoRA **粗网格** | 361 ok / 375 设计 | `summary.csv`、`distilbert_grid_analysis.md`、`missing_runs.csv` |
| [`results_mlora/`](results_mlora/) | mLoRA **粗网格** | 375 ok | 同上格式 |
| [`results_refine/`](results_refine/) | LoRA **细网格**（lr/epoch 加密） | 58 组 | 仅 `summary.csv`（无逐 run 曲线） |
| [`results_mlora_refine/`](results_mlora_refine/) | mLoRA **细网格** | 56 组 | 每组含 `run_meta.json`、`train.csv`、`test.csv`（8 epoch；含 1 组 ep20 定稿探针） |

## 粗网格要点（LoRA，`results/`）

- **最优验证准确率**：**0.9163** — `lr=3e-4, r=32, α=16, wd=0`
- **lr 边际**：`3e-4` 档 mean acc ≈ 0.90；`3e-3` / `3e-7` 明显更差
- 详见 [`results/distilbert_grid_analysis.md`](results/distilbert_grid_analysis.md)

## mLoRA 粗网格要点（`results_mlora/`）

- **最优验证准确率**：**0.9117** — `lr=3e-3, r=32, α=64, wd=0.01`（与 LoRA 最优同档 acc，但 lr 更高）
- 详见 [`results_mlora/distilbert_grid_analysis.md`](results_mlora/distilbert_grid_analysis.md)

## 最终 benchmark（测试集）

定稿 run 在仓库根目录 [`distilbert_final_results/`](../distilbert_final_results/)：

| 方法 | 超参（来自网格 Top） | 训练 | 测试集末次 acc |
|------|----------------------|------|----------------|
| LoRA | `lr=3e-4, r=32, α=16, wd=0`, **20 epoch** | [`lora_test.csv`](../distilbert_final_results/lora_test.csv) | **0.9025** |
| mLoRA | `lr=3e-3, r=32, α=64, wd=0.01`, **20 epoch** | [`mlora_test.csv`](../distilbert_final_results/mlora_test.csv) | **0.9014** |

> 测试曲线按固定 step 记录；末行 iteration=42080 对应 20 epoch 结束。

## 如何阅读单个 run

以 `results_mlora_refine/lr_3p0000e-03_r32_a64_ep20_wd_1p0000e-02/` 为例：

- `run_meta.json` — 超参与 `metrics_dir`
- `train.csv` — 列：`iteration`, `train_loss`, `val_loss`, `val_accuracy`
- `test.csv` — 列：`iteration`, `test_loss`, `test_accuracy`

## 备注

- 本目录 **不含** checkpoint（`.pt`）与提交脚本；仅 CSV / Markdown 汇总。
- 极高学习率（如 `3e-3`）在部分组合上不稳定，分析 md 中可见 acc 下尾。
