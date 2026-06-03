# DeepSeek 调参结果（LoRA / mLoRA）

## 实验设置

| 项 | 值 |
|----|-----|
| 底座 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| 训练数据 | **`alpaca_train_1k`**（阶段一/二/相关性 refine）；**`alpaca_train_full`**（定稿 Top-K，2600 step） |
| 格式 | Chat SFT；验证比例 **20%**（定稿阶段同） |
| 搜索空间 | `lr × lora_r × lora_alpha × weight_decay` |
| 短网格步数 | **500 step**，每 **100 step** eval |
| 验证指标 | **`best_eval_perplexity`**（全程最低验证 PPL，非末次 PPL） |

## 结果目录一览

| 目录 | 阶段 | LoRA 类型 | 说明 |
|------|------|-----------|------|
| [`results/`](results/) | **粗网格** | LoRA | 90 组；最优 PPL **3.891**（`lr=2e-3, r=64, α=16`） |
| [`results_refine/`](results_refine/) | **细网格** | LoRA | 独立 lr 加密；与粗网格目录不覆盖 |
| [`results_correlation_refine/`](results_correlation_refine/) | **相关性驱动 refine** | LoRA | 由粗网格 Spearman/η² 自动生成密网格 |
| [`results_mlora/`](results_mlora/) | **粗网格** | mLoRA | 90 组 summary |
| [`results_mlora_correlation_refine/`](results_mlora_correlation_refine/) | 相关性 refine | mLoRA | summary + analysis |
| [`results_final/`](results_final/) | **全量 Alpaca 定稿 Top-3** | LoRA | 2600 step；含 3 组完整 `train_sft.csv` / `test_sft.csv` |
| [`results_mlora_final/`](results_mlora_final/) | **全量 Alpaca 定稿 Top-3** | mLoRA | 同上，3 组曲线 |
| [`figures/`](figures/) | 可视化 | LoRA + mLoRA | 定稿最优 run 验证 PPL 收敛对比图 |

关联目录（仓库根）：

| 目录 | 说明 |
|------|------|
| [`deepseek/results_final_lora_long_st1200/`](../deepseek/results_final_lora_long_st1200/) | 粗网格最优组合 **1200 step** 长训曲线 |
| [`deepseek/results_final_mlora_long_st1500/`](../deepseek/results_final_mlora_long_st1500/) | mLoRA 代表组合 **1500 step** 长训曲线 |
| [`deepseek_bbh_autogrid/results/`](../deepseek_bbh_autogrid/results/) | BBH 管线相关 SFT 网格（142 组 summary） |

## 粗网格要点（`results/`）

- **Top-1**：PPL **3.891** — `lr=2e-3, r=64, α=16, wd=0.01`
- **稳定 lr 带**：`2e-4` 档 multiple runs PPL ≈ 3.9–4.0
- **失效区**：`2e-7` 与学习率过高组合 PPL 爆炸（见 analysis 尾部）
- 分析：[`results/deepseek_grid_analysis.md`](results/deepseek_grid_analysis.md)

## 定稿 Top-K（全量 Alpaca，`results_final/` / `results_mlora_final/`）

选型规则见 [`Alpaca_train_full_best_runs.md`](Alpaca_train_full_best_runs.md)：在 `summary.csv` 的 `status=ok` 行中取 **`best_eval_perplexity` 最小**。

| 方法 | 最优 PPL | 代表 run |
|------|----------|----------|
| LoRA | **3.7008** | `lr≈1.62e-3, r=32, α=32, wd=0.01` |
| mLoRA | （见 `results_mlora_final/summary.csv`） | 3 组完整曲线已归档 |

每组 run 目录含：

- `run_meta.json` — 超参与步数
- `train_sft.csv` / `test_sft.csv` — 列含 `iteration`, `eval_loss`, `eval_perplexity`

## 下游 benchmark

### BBH

- SFT 阶段汇总：[`deepseek_bbh_autogrid/results/summary.csv`](../deepseek_bbh_autogrid/results/summary.csv)（500 step 短网格，含 post-peak 发散指标）
- **合并 LoRA → lm-eval BBH** 的完整 benchmark CSV **未**纳入本归档（体积与路径在集群）；定稿 Top-K 合并与 BBH 投递在开发仓库脚本中完成

### 长训稳定性

- LoRA 1200 step：[`deepseek/results_final_lora_long_st1200/`](../deepseek/results_final_lora_long_st1200/) — 粗网格最优超参延伸训练
- mLoRA 1500 step：[`deepseek/results_final_mlora_long_st1500/`](../deepseek/results_final_mlora_long_st1500/)

## 文件说明

| 文件 | 含义 |
|------|------|
| `summary.csv` | 全网格汇总；关键列 `best_eval_perplexity`, `best_iteration`, `status` |
| `deepseek_grid_analysis.md` | 排名、按 lr/r/wd 分组统计 |
| `topk_source.json` | 定稿 Top-K 来源（`results_final/`） |
| `deepseek_grid_snapshot.md` | 粗网格快照说明（历史参考） |

## 备注

- checkpoint（`sft_lora_state.pt`、`model_merged_hf/`）**未**上传 GitHub。
- 定稿 analysis 含 **post-peak** 列：衡量最优点之后是否发散（越接近 1 越稳）。
