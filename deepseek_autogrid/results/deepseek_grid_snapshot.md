# DeepSeek 网格结果快照

本目录保存 DeepSeek 网格汇总文件（轻量结果）。逐 run 的大文件默认忽略。

## 文件

- `summary.csv`
- `missing_runs.csv`
- `deepseek_grid_analysis.md`

## 更新

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
```
