# Manifold-Lora — 调参实验结果归档

本仓库 **仅收录** DistilBERT、DeepSeek、Qwen3 三条 LoRA / mLoRA 超参网格的**训练、验证与 benchmark 结果**（CSV、汇总表、分析 Markdown、曲线图）。不含训练代码与集群脚本；完整开发环境在本地/集群私有副本中维护。

| 模型 | 说明文档 | 主要验证指标 | 下游 benchmark |
|------|----------|--------------|----------------|
| **DistilBERT** | [distilbert_autogrid/README.md](distilbert_autogrid/README.md) | SST-2 验证集 `best_val_acc` | GLUE SST-2 测试集（20 epoch 定稿 run） |
| **DeepSeek** | [deepseek_autogrid/README.md](deepseek_autogrid/README.md) | Alpaca 验证集 `best_eval_perplexity` | BBH（Top-K 合并模型）；长训定稿曲线 |
| **Qwen3-0.6B** | [qwen3_autogrid/README.md](qwen3_autogrid/README.md) | SFT 验证 PPL + tinyMMLU | 全量 MMLU Top-5（LoRA 已完成） |

## 目录结构

```
distilbert_autogrid/   # DistilBERT 粗/细/mLoRA 网格汇总与部分 run 曲线
distilbert_final_results/  # 最优超参 20 epoch 测试集曲线
deepseek_autogrid/     # DeepSeek 多阶段网格 + 定稿 Top-3 曲线
deepseek/              # 长步数定稿 run（LoRA 1200 step / mLoRA 1500 step）
deepseek_bbh_autogrid/ # BBH 相关网格 SFT 汇总
qwen3_autogrid/        # Qwen3 MMLU 管线粗/细网格 + LoRA vs mLoRA 对比
```

## 文件约定

- **`summary.csv`**：全网格一行一组合，含最优/末次验证指标与 `status`。
- **`*_grid_analysis.md`**：由汇总脚本自动生成的排名与分组统计。
- **`train*.csv` / `test*.csv`**：定稿或代表性 run 的训练/验证/测试逐步曲线（**不含** checkpoint `.pt`）。
- **`run_meta.json`**：该 run 的超参与路径元数据。
