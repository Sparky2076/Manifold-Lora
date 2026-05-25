# DeepSeek 网格结果分析

由 `python -m deepseek_autogrid.analyze_results` 自动生成。

- **生成时间（UTC）**：2026-05-25 05:58:53Z
- **有效行数**：3（status=ok）

## 整体指标（best_eval_perplexity）

| 统计量 | 值 |
|--------|-----|
| min | 3.7089 |
| max | 3.9054 |
| mean | 3.7859 |
| median | 3.7434 |

## Top 组合（按 perplexity 越低越好）

| rank | best_ppl | last/best | tail_mean/best | lr | r | alpha | wd |
|------|----------|-----------|----------------|----|---|-------|-----|
| 1 | 3.7089 | 1.0 | 1.0 | 0.002 | 64 | 64.0 | 0.01 |
| 2 | 3.7434 | 1.0 | 1.0 | 0.002 | 64 | 32.0 | 0.01 |
| 3 | 3.9054 | 1.0 | 1.0 | 0.0013158664 | 64 | 64.0 | 0.01 |

## Post-peak（最优之后又发散了吗？）

- **post_peak_last_over_best**：最后一次 eval 的 ppl / 全程最优；**接近 1 好**，明显 **>1** 表示终点比最低点差。
- **post_peak_tail_mean_over_best**：达到最优 perplexity 的那次 eval **之后**，所有 eval 点的平均 ppl / 最优 ppl；**接近 1** 表示最优后整体未漂，**明显 >1** 表示最优后持续变差。

### 尾部发散最明显（tail_mean/best 降序，最多 12 行）

| tail/best | last/best | best_ppl | lr | r | alpha | wd |
|-------------|-----------|----------|----|---|-------|-----|
| 1.0000 | 1.0 | 3.7089 | 0.002 | 64 | 64.0 | 0.01 |
| 1.0000 | 1.0 | 3.7434 | 0.002 | 64 | 32.0 | 0.01 |
| 1.0000 | 1.0 | 3.9054 | 0.0013158664 | 64 | 64.0 | 0.01 |

## 分组统计（mean / min）

| group | key | n | min | mean |
|------|-----|---|-----|------|
| lr | 0.002 | 2 | 3.7089 | 3.7261 |
| lr | 0.0013158664 | 1 | 3.9054 | 3.9054 |
| weight_decay | 0.01 | 3 | 3.7089 | 3.7859 |
| lora_r | 64 | 3 | 3.7089 | 3.7859 |
| lora_alpha | 32.0 | 1 | 3.7434 | 3.7434 |
| lora_alpha | 64.0 | 2 | 3.7089 | 3.8072 |

## 更新方式

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
# 细网格（config_refine + results_refine）：
# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine
# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv
```