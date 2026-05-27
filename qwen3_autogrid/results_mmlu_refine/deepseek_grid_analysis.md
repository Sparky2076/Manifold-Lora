# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-27 11:01:15Z
- **有效行数**：48（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 1.0352 |
| max | 1.1430 |
| mean | 1.0864 |
| median | 1.0659 |

## Top 组合（按 perplexity 越低越好）

| rank | best_eval_perplexity | lr | r | alpha | weight_decay |
|------|----------------------|----|---|-------|--------------|
| 1 | 1.0352 | 0.0002 | 64 | 128.0 | 0.01 |
| 2 | 1.0352 | 0.0002 | 64 | 128.0 | 0.001 |
| 3 | 1.0373 | 0.0002 | 32 | 64.0 | 0.0 |
| 4 | 1.0373 | 0.0002 | 32 | 64.0 | 0.01 |
| 5 | 1.0373 | 0.0002 | 32 | 64.0 | 0.001 |
| 6 | 1.0380 | 0.0003 | 32 | 64.0 | 0.001 |
| 7 | 1.0402 | 0.0003 | 32 | 64.0 | 0.0 |
| 8 | 1.0402 | 0.0003 | 32 | 64.0 | 0.01 |
| 9 | 1.0409 | 0.0002 | 64 | 128.0 | 0.0 |
| 10 | 1.0422 | 0.0004 | 32 | 64.0 | 0.01 |
| 11 | 1.0438 | 0.0003 | 64 | 128.0 | 0.01 |
| 12 | 1.0440 | 0.0004 | 32 | 64.0 | 0.0 |
| 13 | 1.0440 | 0.0004 | 32 | 64.0 | 0.001 |
| 14 | 1.0445 | 0.0006 | 32 | 64.0 | 0.0 |
| 15 | 1.0446 | 0.0006 | 32 | 64.0 | 0.001 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 0.0002 | 6 | 1.0352 | 1.0372 |
| lr | 0.0003 | 6 | 1.0380 | 1.0429 |
| lr | 0.0004 | 6 | 1.0422 | 1.0444 |
| lr | 0.0006 | 6 | 1.0445 | 1.0539 |
| lr | 0.001 | 6 | 1.0615 | 1.0988 |
| lr | 0.0015 | 6 | 1.1244 | 1.1359 |
| lr | 0.002 | 6 | 1.1376 | 1.1390 |
| lr | 0.003 | 6 | 1.1379 | 1.1391 |
| weight_decay | 0.001 | 16 | 1.0352 | 1.0857 |
| weight_decay | 0.01 | 16 | 1.0352 | 1.0863 |
| weight_decay | 0.0 | 16 | 1.0373 | 1.0872 |
| lora_r | 32 | 24 | 1.0373 | 1.0806 |
| lora_r | 64 | 24 | 1.0352 | 1.0922 |
| lora_alpha | 64.0 | 24 | 1.0373 | 1.0806 |
| lora_alpha | 128.0 | 24 | 1.0352 | 1.0922 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```