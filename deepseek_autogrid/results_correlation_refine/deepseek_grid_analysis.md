# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-12 08:15:19Z
- **有效行数**：48（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 3.9222 |
| max | 9.7886 |
| mean | 4.5688 |
| median | 4.2779 |

## Top 组合（按 perplexity 越低越好）

| rank | best_eval_perplexity | lr | r | alpha | weight_decay |
|------|----------------------|----|---|-------|--------------|
| 1 | 3.9222 | 0.002 | 64 | 32.0 | 0.01 |
| 2 | 3.9260 | 0.0016222617 | 32 | 32.0 | 0.01 |
| 3 | 3.9393 | 0.0010673398 | 32 | 16.0 | 0.01 |
| 4 | 3.9440 | 0.0002 | 64 | 32.0 | 0.01 |
| 5 | 3.9580 | 0.0016222617 | 64 | 32.0 | 0.01 |
| 6 | 3.9724 | 0.0004620259 | 64 | 16.0 | 0.01 |
| 7 | 3.9837 | 0.0003039822 | 32 | 32.0 | 0.01 |
| 8 | 3.9914 | 0.0016222617 | 64 | 16.0 | 0.01 |
| 9 | 3.9927 | 0.0005696072 | 32 | 32.0 | 0.01 |
| 10 | 4.0069 | 0.0007022383 | 64 | 32.0 | 0.01 |
| 11 | 4.0140 | 0.0003747635 | 32 | 32.0 | 0.01 |
| 12 | 4.0208 | 0.0013158664 | 64 | 32.0 | 0.01 |
| 13 | 4.0222 | 0.0010673398 | 32 | 32.0 | 0.01 |
| 14 | 4.0293 | 0.0004620259 | 32 | 16.0 | 0.01 |
| 15 | 4.0483 | 0.0003039822 | 64 | 32.0 | 0.01 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 0.0016222617 | 4 | 3.9260 | 3.9939 |
| lr | 0.0003747635 | 4 | 4.0140 | 4.1049 |
| lr | 0.0003039822 | 4 | 3.9837 | 4.1380 |
| lr | 0.0010673398 | 4 | 3.9393 | 4.1717 |
| lr | 0.0002 | 4 | 3.9440 | 4.4041 |
| lr | 0.002 | 4 | 3.9222 | 4.4363 |
| lr | 0.0005696072 | 4 | 3.9927 | 4.4871 |
| lr | 0.0002465693 | 4 | 4.1610 | 4.6733 |
| lr | 0.0008657523 | 4 | 4.4792 | 4.7086 |
| lr | 0.0013158664 | 4 | 4.0208 | 4.9844 |
| lr | 0.0004620259 | 4 | 3.9724 | 5.1438 |
| lr | 0.0007022383 | 4 | 4.0069 | 5.5796 |
| weight_decay | 0.01 | 48 | 3.9222 | 4.5688 |
| lora_r | 64 | 24 | 3.9222 | 4.3983 |
| lora_r | 32 | 24 | 3.9260 | 4.7393 |
| lora_alpha | 32.0 | 24 | 3.9222 | 4.4107 |
| lora_alpha | 16.0 | 24 | 3.9393 | 4.7269 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```