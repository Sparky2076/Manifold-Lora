# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-28 13:43:33Z
- **有效行数**：45（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 1.0410 |
| max | 121.5956 |
| mean | 62.6511 |
| median | 75.6674 |

## Top 组合（按 perplexity 越低越好）

| rank | best_eval_perplexity | lr | r | alpha | weight_decay |
|------|----------------------|----|---|-------|--------------|
| 1 | 1.0410 | 0.002 | 16 | 32.0 | 0.0 |
| 2 | 1.0410 | 0.002 | 16 | 32.0 | 0.01 |
| 3 | 1.0410 | 0.002 | 16 | 32.0 | 0.001 |
| 4 | 1.0475 | 0.002 | 32 | 64.0 | 0.0 |
| 5 | 1.0475 | 0.002 | 32 | 64.0 | 0.01 |
| 6 | 1.0475 | 0.002 | 32 | 64.0 | 0.001 |
| 7 | 1.0486 | 0.002 | 64 | 128.0 | 0.0 |
| 8 | 1.0486 | 0.002 | 64 | 128.0 | 0.01 |
| 9 | 1.0486 | 0.002 | 64 | 128.0 | 0.001 |
| 10 | 1.0486 | 0.0002 | 64 | 128.0 | 0.0 |
| 11 | 1.0486 | 0.0002 | 64 | 128.0 | 0.01 |
| 12 | 1.0486 | 0.0002 | 64 | 128.0 | 0.001 |
| 13 | 1.0554 | 0.0002 | 32 | 64.0 | 0.0 |
| 14 | 1.0554 | 0.0002 | 32 | 64.0 | 0.01 |
| 15 | 1.0554 | 0.0002 | 32 | 64.0 | 0.001 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 0.002 | 9 | 1.0410 | 1.0457 |
| lr | 0.0002 | 9 | 1.0486 | 1.0580 |
| lr | 2e-05 | 9 | 10.5613 | 67.9603 |
| lr | 2e-06 | 9 | 121.5956 | 121.5956 |
| lr | 2e-07 | 9 | 121.5956 | 121.5956 |
| weight_decay | 0.0 | 15 | 1.0410 | 62.6511 |
| weight_decay | 0.01 | 15 | 1.0410 | 62.6511 |
| weight_decay | 0.001 | 15 | 1.0410 | 62.6511 |
| lora_r | 64 | 15 | 1.0486 | 51.1700 |
| lora_r | 32 | 15 | 1.0475 | 64.1923 |
| lora_r | 16 | 15 | 1.0410 | 72.5909 |
| lora_alpha | 128.0 | 15 | 1.0486 | 51.1700 |
| lora_alpha | 64.0 | 15 | 1.0475 | 64.1923 |
| lora_alpha | 32.0 | 15 | 1.0410 | 72.5909 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```