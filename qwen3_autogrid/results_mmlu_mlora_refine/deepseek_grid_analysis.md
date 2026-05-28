# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-28 13:43:33Z
- **有效行数**：48（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 1.0353 |
| max | 1.0502 |
| mean | 1.0392 |
| median | 1.0380 |

## Top 组合（按 perplexity 越低越好）

| rank | best_eval_perplexity | lr | r | alpha | weight_decay |
|------|----------------------|----|---|-------|--------------|
| 1 | 1.0353 | 0.003 | 32 | 64.0 | 0.0 |
| 2 | 1.0353 | 0.003 | 32 | 64.0 | 0.01 |
| 3 | 1.0353 | 0.003 | 32 | 64.0 | 0.001 |
| 4 | 1.0354 | 0.0015 | 64 | 128.0 | 0.0 |
| 5 | 1.0354 | 0.0015 | 64 | 128.0 | 0.01 |
| 6 | 1.0354 | 0.0015 | 64 | 128.0 | 0.001 |
| 7 | 1.0359 | 0.002 | 32 | 64.0 | 0.0 |
| 8 | 1.0359 | 0.002 | 32 | 64.0 | 0.01 |
| 9 | 1.0359 | 0.002 | 32 | 64.0 | 0.001 |
| 10 | 1.0359 | 0.001 | 64 | 128.0 | 0.0 |
| 11 | 1.0359 | 0.001 | 64 | 128.0 | 0.01 |
| 12 | 1.0359 | 0.001 | 64 | 128.0 | 0.001 |
| 13 | 1.0363 | 0.0015 | 32 | 64.0 | 0.0 |
| 14 | 1.0363 | 0.0015 | 32 | 64.0 | 0.01 |
| 15 | 1.0363 | 0.0015 | 32 | 64.0 | 0.001 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 0.0015 | 6 | 1.0354 | 1.0359 |
| lr | 0.002 | 6 | 1.0359 | 1.0362 |
| lr | 0.001 | 6 | 1.0359 | 1.0367 |
| lr | 0.003 | 6 | 1.0353 | 1.0373 |
| lr | 0.0006 | 6 | 1.0374 | 1.0380 |
| lr | 0.0004 | 6 | 1.0388 | 1.0401 |
| lr | 0.0003 | 6 | 1.0402 | 1.0424 |
| lr | 0.0002 | 6 | 1.0439 | 1.0470 |
| weight_decay | 0.0 | 16 | 1.0353 | 1.0392 |
| weight_decay | 0.01 | 16 | 1.0353 | 1.0392 |
| weight_decay | 0.001 | 16 | 1.0353 | 1.0392 |
| lora_r | 64 | 24 | 1.0354 | 1.0384 |
| lora_r | 32 | 24 | 1.0353 | 1.0400 |
| lora_alpha | 128.0 | 24 | 1.0354 | 1.0384 |
| lora_alpha | 64.0 | 24 | 1.0353 | 1.0400 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```