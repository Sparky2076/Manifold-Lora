# DeepSeek 全因子网格（LoRA / mLoRA）

当前网格默认配置：

- 数据：`alpaca_train_1k`
- 验证比例：`SFT_VAL_RATIO=0.2`
- 步数：`MAX_STEPS=500`（默认，约为原 1500 的 1/3，缩短单 job），`EVAL_EVERY=100`
- 参数网格（粗略）：`lr(5) × r(3) × alpha(3) × wd(2) = 90` 组（每种 LoRA 类型）

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

`server_submit_deepseek_grid_mlora.sh` 内会 **`export GRID_RESUME=1`**，避免继承此前 shell 里误留的 **`GRID_RESUME=0`** 导致整网反复重交。若你**刻意**要全量重跑，请用 `GRID_RESUME=0 bash scripts/server_submit_deepseek_grid.sh`，并自行 `export LORA_TYPE=mlora` 与 `RESULTS_ROOT=.../results_mlora`。

## 找不到「提交脚本」进程？

- **`bjobs` 里的 JOBID** 是 **LSF 训练作业**；**网格提交循环**是登录节点上的 **bash 进程**，两者不是同一个号。
- **`pgrep` 要在启动 `nohup` 的那台登录节点上执行**（例如 `bjobs` 里 `FROM_HOST=mgtgpu01` 则到 `mgtgpu01` 上 `pgrep`）。换一台登录节点常会「查不到」。
- 更新脚本后，提交循环会写 **`deepseek_autogrid/.grid_submitter.pid`**（LoRA）或 **`.grid_submitter_mlora.pid`**（mLoRA）；或运行 **`bash scripts/grid_submitter_status.sh`**。

## `GRID_RESUME=0`（全量重跑）注意

设为 `0` 时会对**每个组合都再 `bsub` 一次**（无视已有 `test_sft.csv`）。脚本在**第一轮递交结束并等队列排空后就会退出**，不会 endless 重复整网；若你曾用旧脚本看到「90 组已满仍在不停提交」，多半是旧逻辑在第二轮又把 90 组交了一遍——请 `git pull` 更新。

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

### 本仓库已随代码提交的 LoRA 网格产物（数据说明）

当前默认网格为 **90 组**（`lr×r×alpha×wd`，`MAX_STEPS=500`）。仓库内 **`deepseek_autogrid/results/summary.csv`** 为聚合后的每组合一行指标（含 `best_eval_perplexity`、`metrics_dir`、`status` 等）；**`deepseek_autogrid/results/deepseek_grid_analysis.md`** 为由 `python -m deepseek_autogrid.analyze_results` 生成的 Markdown 统计与 Top 组合说明。逐 run 的大目录仍由 `.gitignore` 忽略，仅保留轻量汇总与说明文件。`aggregate_results.py` 仅收录与 `config.iter_grid()` 目录名一致的 run，避免旧实验目录再次混入 `summary.csv`。
