# DeepSeek 全因子网格（LoRA / mLoRA）

当前网格默认配置：

- 数据：`alpaca_train_1k`
- 验证比例：`SFT_VAL_RATIO=0.2`
- 步数：`MAX_STEPS=500`（默认，约为原 1500 的 1/3，缩短单 job），`EVAL_EVERY=100`
- 参数网格（粗略）：`lr(3) × r(3) × alpha(3) × wd(2) = 54` 组（每种 LoRA 类型）

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

## 队列节流（默认）

`run_grid_bsub.sh` 默认 **`GRID_MAX_PEND=1`**：仅当本账号 **`PEND=0`** 时才再 `bsub` 下一单，减轻站点「Pending 上限 / User permission denied」。若仍偶发拒绝，脚本会**等待后重试**，不会整段退出。可调：`GRID_MAX_RUN`、`GRID_MAX_PEND`、`GRID_POLL_SEC`、`SUBMIT_SLEEP_SEC`（关闭 PEND 限制：`GRID_MAX_PEND=0`，不推荐）。

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
