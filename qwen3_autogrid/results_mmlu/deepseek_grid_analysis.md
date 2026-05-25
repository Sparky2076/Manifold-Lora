# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-25 13:51:23Z
- **有效行数**：45（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 1.0344 |
| max | 88.0608 |
| mean | 9.8025 |
| median | 1.0465 |

## Top 组合（按 perplexity 越低越好）

| rank | best_eval_perplexity | lr | r | alpha | weight_decay |
|------|----------------------|----|---|-------|--------------|
| 1 | 1.0344 | 2e-05 | 64 | 128.0 | 0.0 |
| 2 | 1.0345 | 2e-05 | 64 | 128.0 | 0.01 |
| 3 | 1.0348 | 2e-05 | 64 | 128.0 | 0.001 |
| 4 | 1.0348 | 0.0002 | 16 | 32.0 | 0.0 |
| 5 | 1.0348 | 0.0002 | 16 | 32.0 | 0.001 |
| 6 | 1.0360 | 2e-05 | 32 | 64.0 | 0.0 |
| 7 | 1.0360 | 2e-05 | 32 | 64.0 | 0.01 |
| 8 | 1.0362 | 0.0002 | 16 | 32.0 | 0.01 |
| 9 | 1.0362 | 2e-05 | 32 | 64.0 | 0.001 |
| 10 | 1.0373 | 0.0002 | 32 | 64.0 | 0.0 |
| 11 | 1.0373 | 0.0002 | 32 | 64.0 | 0.001 |
| 12 | 1.0374 | 2e-05 | 16 | 32.0 | 0.0 |
| 13 | 1.0376 | 2e-05 | 16 | 32.0 | 0.01 |
| 14 | 1.0379 | 0.0002 | 32 | 64.0 | 0.01 |
| 15 | 1.0379 | 2e-05 | 16 | 32.0 | 0.001 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 2e-05 | 9 | 1.0344 | 1.0361 |
| lr | 0.0002 | 9 | 1.0348 | 1.0380 |
| lr | 2e-06 | 9 | 1.0410 | 1.0513 |
| lr | 0.002 | 9 | 1.1352 | 1.1528 |
| lr | 2e-07 | 9 | 5.1962 | 44.7344 |
| weight_decay | 0.001 | 15 | 1.0348 | 9.7629 |
| weight_decay | 0.0 | 15 | 1.0344 | 9.8111 |
| weight_decay | 0.01 | 15 | 1.0345 | 9.8335 |
| lora_r | 64 | 15 | 1.0344 | 1.9003 |
| lora_r | 32 | 15 | 1.0360 | 9.1061 |
| lora_r | 16 | 15 | 1.0348 | 18.4012 |
| lora_alpha | 128.0 | 15 | 1.0344 | 1.9003 |
| lora_alpha | 64.0 | 15 | 1.0360 | 9.1061 |
| lora_alpha | 32.0 | 15 | 1.0348 | 18.4012 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```