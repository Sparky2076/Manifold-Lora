# DeepSeek 全因子网格（LoRA / mLoRA）

当前网格默认配置：

- 数据：`alpaca_train_1k`
- 验证比例：`SFT_VAL_RATIO=0.2`
- 步数：`MAX_STEPS=1500`，`EVAL_EVERY=100`
- 参数网格：`lr(5) × r(5) × alpha(5) × wd(3) = 375` 组（每种 LoRA 类型）

## 服务器提交（默认 nohup）

LoRA:

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
nohup bash scripts/server_submit_deepseek_grid.sh > deepseek_grid_submit.log 2>&1 &
tail -f deepseek_grid_submit.log
```

mLoRA:

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
nohup bash scripts/server_submit_deepseek_grid_mlora.sh > deepseek_grid_mlora_submit.log 2>&1 &
tail -f deepseek_grid_mlora_submit.log
```

## 自动补齐缺失

检测：

```bash
python -m deepseek_autogrid.fill_missing_runs
```

仅补交缺失：

```bash
python -m deepseek_autogrid.fill_missing_runs --submit-bsub
```

## 汇总与分析

默认要求全部组合完成（否则报错退出）：

```bash
python -m deepseek_autogrid.aggregate_results
python -m deepseek_autogrid.analyze_results
```

阶段性检查可加 `--allow-incomplete`。

## 结果目录

- LoRA: `deepseek_autogrid/results/`
- mLoRA: `deepseek_autogrid/results_mlora/`

建议在各自目录下保留：
- `summary.csv`
- `missing_runs.csv`
- `deepseek_grid_analysis.md`
