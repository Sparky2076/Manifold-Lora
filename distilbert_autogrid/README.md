# DistilBERT LoRA 全因子网格（`lr × r × alpha × weight_decay`）

优化器：**只扫** `weight_decay ∈ {0, 0.01, 0.1}`；`adam_beta1` / `adam_beta2` **固定**为 `config.py` 中的 `ADAM_BETA1_FIXED` / `ADAM_BETA2_FIXED`（默认 0.9 / 0.999）。

## 配置

[`config.py`](config.py) 定义 `LR_LIST`、`R_LIST`、`ALPHA_LIST`、`WEIGHT_DECAY_LIST`、`EPOCHS_DEFAULT` 与 `RESULTS_ROOT`（默认 `distilbert_autogrid/results/`）。缩小或扩大网格**只改此文件**。

当前全量约为 **375** 组（5×5×5×3）；极高学习率（如 `3e-3`）可能不稳定。

## 本地顺序跑满网格

在**仓库根**执行：

```bash
python -m distilbert_autogrid.run_grid
```

- 干跑（只打印命令）：`python -m distilbert_autogrid.run_grid --dry-run`
- 烟测 1 次：`python -m distilbert_autogrid.run_grid --max-attempts 1`
- 直到 1 次成功：`python -m distilbert_autogrid.run_grid --max-runs 1`

每个组合目录含 `run_meta.json`、`train.csv`、`test.csv`。

## LSF：每个组合一个作业（`bsub`）

### 一键（服务器端完整流程）

在**仓库根**（会先 `sed`、再按需探测 `CONDA_ROOT`，最后调用 `run_grid_bsub.sh`）：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"   # 建议显式设置
bash scripts/server_submit_distilbert_grid.sh
```

脚本说明见 [`scripts/server_submit_distilbert_grid.sh`](../scripts/server_submit_distilbert_grid.sh) 顶部注释（含 **tmux** 用法）。

### 手动分步

在**仓库根**：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
bash distilbert_autogrid/run_grid_bsub.sh
```

### Conda（LSF 计算节点必看）

批作业里常**没有**交互式 shell 的 `PATH`，`conda info --base` 可能失败，导致找不到 `conda.sh`。请先确认本机 conda 根目录（其下应有 `etc/profile.d/conda.sh`），在**提交网格或单次训练前**执行一次：

```bash
export CONDA_ROOT="$HOME/miniconda3"   # 或 anaconda3、mambaforge 等，按你机器实际路径改
# 可选：环境名不是 torch 时
# export CONDA_ENV_NAME=torch
```

再运行 `bash distilbert_autogrid/run_grid_bsub.sh` 或 `bash distilbert/scripts/submit_bsub.sh`。`submit_bsub.sh` 会把 `CONDA_ROOT` / `CONDA_BASE` / `CONDA_ENV_NAME` 传给作业。

若仍出现 `importlib_metadata` / conda 插件报错，在登录节点尝试：`conda update -n base conda`，或在 base 环境 `pip install -U "importlib-metadata>=6"`。

### SSH 易断线：用 `tmux`（推荐）

大批量提交时，用 `tmux` 包住会话，断网/关终端不会杀掉提交循环：

```bash
tmux new -s grid
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
bash distilbert_autogrid/run_grid_bsub.sh
```

- **暂时离开（detach）**：`Ctrl+B`，松开后按 `D`。
- **重新连上后恢复**：`tmux attach -t grid`
- 列出会话：`tmux ls`

不用 `tmux` 时也可用：`nohup bash distilbert_autogrid/run_grid_bsub.sh > run_grid_bsub.log 2>&1 &`（日志看 `tail -f run_grid_bsub.log`）。

### 断点续交（自动跳过已跑完的组合）

`run_grid_bsub.sh` 默认 **`GRID_RESUME=1`**：若某目录下已有 **`test.csv`**，且数据行数（去掉表头）**≥ `EPOCHS`**，则认为该组合已完整跑完，**不再 `bsub`**。脚本中断或 SSH 断开后，**重新执行同一条** `bash distilbert_autogrid/run_grid_bsub.sh` 即可从缺口继续提交。

- **强制全部重交**（忽略已有结果）：`GRID_RESUME=0 bash distilbert_autogrid/run_grid_bsub.sh`

脚本对 `iter_grid()` 中每一组调用一次 `distilbert/scripts/submit_bsub.sh`，各作业 `METRICS_DIR` 指向 `distilbert_autogrid/results/<run_name>/`。可选环境变量（覆盖默认）：

| 变量 | 含义 |
|------|------|
| `RESULTS_ROOT` | 结果根目录（默认 `distilbert_autogrid/results`） |
| `EPOCHS` | 训练轮数（默认见 `config.py`） |
| `GRID_RESUME` | `1`（默认）跳过已完整完成的组合；`0` 全部提交 |
| `QUEUE` | LSF 队列名（默认 `gpu`，在 `submit_bsub.sh`） |
| `LORA_TYPE` | `default` 或 `mlora`（默认 `default`） |
| `LORA_DROPOUT` | LoRA dropout（默认 `0.05`） |
| `BATCH_SIZE` | batch（默认 `4`） |
| `GRAD_ACCUM_STEPS` | 梯度累积（默认 `8`） |
| `CONDA_ROOT` / `CONDA_BASE` | Conda 安装根目录（LSF 批作业建议设置，见上文） |
| `CONDA_ENV_NAME` | 要激活的环境名（默认 `torch`） |

## 汇总

网格跑完后，在仓库根：

```bash
python -m distilbert_autogrid.aggregate_results
```

生成 `distilbert_autogrid/results/summary.csv`（按 `best_val_acc` 降序；指标来自各子目录 `test.csv`）。

## 上传到集群

`scripts/upload.sh` / `upload.ps1` **仅**同步 **`distilbert/`**、**`distilbert_autogrid/`** 与根目录 `lora.py` / `mlora.py` / `optimizers.py` 及 `scripts/`（体积小）；提交网格前执行即可。

## 提交到 GitHub

见仓库根 [`scripts/commit_and_push.sh`](../scripts/commit_and_push.sh)：在 Git Bash 中执行，按提示检查 `git status`、填写提交说明并 `git push`。请勿将大规模 `results/` 下的 CSV 误提交（`distilbert_autogrid/results/.gitignore` 已忽略运行产物）。
