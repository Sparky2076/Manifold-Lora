# DistilBERT LoRA 全因子网格（`lr × r × alpha × weight_decay`）

优化器：**只扫** `weight_decay ∈ {0, 0.01, 0.1}`；`adam_beta1` / `adam_beta2` **固定**为 [`config.py`](config.py) 中的 `ADAM_BETA1_FIXED` / `ADAM_BETA2_FIXED`（默认 0.9 / 0.999）。

> **仓库入口**：根目录 [README.md](../README.md) 仅作总览；**上传服务器、`bsub`、tmux、`GRID_RESUME`、停止与队列** 等操作说明以**本文档为准**。

---

## 配置

[`config.py`](config.py) 定义 `LR_LIST`、`R_LIST`、`ALPHA_LIST`、`WEIGHT_DECAY_LIST`、`EPOCHS_DEFAULT` 与 `RESULTS_ROOT`（默认 `distilbert_autogrid/results/`）。缩小或扩大网格**只改此文件**。

当前全量约为 **375** 组（5×5×5×3）；极高学习率（如 `3e-3`）可能不稳定。

---

## 本机上传与拉回结果

**上传**（仅 DistilBERT 与网格：`distilbert/`、`distilbert_autogrid/`、根目录 `lora.py` / `mlora.py` / `optimizers.py`、`scripts/`）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

PowerShell：`.\scripts\upload.ps1`。

**拉回 CSV**（IP 换成你的服务器；`<run_name>` 在服务器上 `ls distilbert_autogrid/results` 查看）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp -r "wangxiao@202.121.138.196:~/Manifold-Lora/distilbert_autogrid/results/<run_name>" distilbert_autogrid/results/
```

---

## 本地顺序跑满网格（不占 LSF）

在**仓库根**：

```bash
python -m distilbert_autogrid.run_grid
```

- 干跑：`python -m distilbert_autogrid.run_grid --dry-run`
- 烟测 1 次：`python -m distilbert_autogrid.run_grid --max-attempts 1`
- 直到 1 次成功：`python -m distilbert_autogrid.run_grid --max-runs 1`

每个组合目录含 `run_meta.json`、`train.csv`、`test.csv`。

---

## 集群 LSF：每个组合一个作业（`bsub`）

### 一键（推荐）

在**仓库根**（`sed` → 可选探测 `CONDA_ROOT` → `run_grid_bsub.sh`）：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"   # 建议显式设置，与计算节点一致
bash scripts/server_submit_distilbert_grid.sh
```

脚本头部说明：[`scripts/server_submit_distilbert_grid.sh`](../scripts/server_submit_distilbert_grid.sh)。

### 手动分步

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
export CONDA_ROOT="$HOME/miniconda3"
bash distilbert_autogrid/run_grid_bsub.sh
```

### Conda（计算节点必看）

批作业里常无交互式 `PATH`，`conda info --base` 可能失败。请先确认 conda 根目录（含 `etc/profile.d/conda.sh`）：

```bash
export CONDA_ROOT="$HOME/miniconda3"
# export CONDA_ENV_NAME=torch   # 若环境名不是 torch
```

再运行 `server_submit_*.sh` / `run_grid_bsub.sh` / `distilbert/scripts/submit_bsub.sh`。`submit_bsub.sh` 会把 `CONDA_ROOT` / `CONDA_BASE` / `CONDA_ENV_NAME` 传给作业。

若出现 `importlib_metadata` / conda 插件报错：登录节点执行 `conda update -n base conda` 或 `pip install -U "importlib-metadata>=6"`（在 base 环境）。

---

## tmux：新建会话、重连、离开（推荐）

大批量提交时用 **tmux**，SSH 断开后**提交循环一般会继续跑**。

| 操作 | 命令或按键 |
|------|------------|
| **新建会话并起名 `grid`** | `tmux new -s grid` |
| **断线后再连上（恢复同一窗口）** | **`tmux attach -t grid`** |
| 列出会话 | `tmux ls` |
| **暂时离开（detach，脚本继续跑）** | 先 **`Ctrl+B`**，**松开**后，再按 **小写 `d`**（勿在普通 shell 里单独输入 `d` 回车） |

在 tmux 里跑续交或强交示例：

```bash
tmux new -s grid
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
bash scripts/server_submit_distilbert_grid.sh          # 默认续跑 skip
# 或： bash scripts/server_submit_distilbert_grid_force.sh   # GRID_RESUME=0 全部重交
```

不用 tmux 时：`nohup bash distilbert_autogrid/run_grid_bsub.sh > run_grid_bsub.log 2>&1 &`（`tail -f run_grid_bsub.log`）。

---

## 网格提交脚本：续跑 vs 强交 vs 停止

| 场景 | 命令 | 说明 |
|------|------|------|
| **接着交 / 不重跑已完成的**（默认） | `bash scripts/server_submit_distilbert_grid.sh` 或 `bash distilbert_autogrid/run_grid_bsub.sh` | 默认 **`GRID_RESUME=1`**：某目录 `test.csv` 已有 ≥ `EPOCHS` 行则 **skip**。 |
| **强制全部重交** | `bash scripts/server_submit_distilbert_grid_force.sh` | 固定 **`GRID_RESUME=0`**。等价：`GRID_RESUME=0 bash scripts/server_submit_distilbert_grid.sh`。 |
| **计算节点 Conda** | `export CONDA_ROOT=...` | 见上文；与 [`distilbert/scripts/run_train_bsub.sh`](../distilbert/scripts/run_train_bsub.sh) 一致。 |

**停止「自动 bsub」脚本**：在跑 `server_submit_*.sh` 或 `run_grid_bsub.sh` 的终端按 **`Ctrl+C`** → 只结束**提交循环**，**不会**取消已提交的作业。

**取消队列里的作业**：`bjobs` 查看 JobID，再 **`bkill <JOBID>`**（或集群允许的批量命令，如 `bkill 0`，以你们 LSF 说明为准）。

**Pending 节流**：出现 `Pending job threshold reached. Retrying in 60 seconds...` 为站点对同时 **PEND** 作业数限制，属正常；集群侧会重试；若仍出现 **`User permission denied`**，需减少同时占用队列的作业数。

**自节流（`run_grid_bsub.sh` 内建）**：每次 `bsub` **前**用 `bjobs` 数你的 `RUN` / `PEND`，**超过上限则先 sleep 再试**。

含义（数字即「最多允许几」，直观）：

| 变量 | 含义 |
|------|------|
| `GRID_MAX_RUN` | 默认 `0`（关闭）。设为 **`5`** 表示：**RUN 数 > 5** 时暂停（即 **RUN≤5** 才继续交）→ 同时 **最多约 5 个 RUN**。 |
| `GRID_MAX_PEND` | 默认 `0`（关闭）。设为 **`1`** 表示：**PEND 数 > 1** 时暂停（即 **PEND≤1**）→ **最多 1 个在排队**。 |
| `GRID_POLL_SEC` | 轮询间隔秒数，默认 `30`。 |

（旧版曾写「`GRID_MAX_PEND=2`」是因为当时用 **`>=` 阈值**；现已改为 **`>`，数字=「最多允许几个」，故最多 1 个 PEND 应设 **`GRID_MAX_PEND=1`。**）

示例（**最多约 5 个 RUN、最多 1 个 PEND** + **tmux**，SSH 断线后提交循环一般仍继续）：

```bash
tmux new -s grid
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
export GRID_MAX_RUN=5
export GRID_MAX_PEND=1
export GRID_POLL_SEC=30
bash scripts/server_submit_distilbert_grid.sh
```

- **暂时离开（detach）**：`Ctrl+B`，松开后按 **小写 `d`**  
- **断线后再连上**：`tmux attach -t grid`  
- **列出会话**：`tmux ls`

---

### 断点续交与 `GRID_RESUME`

`run_grid_bsub.sh` 默认 **`GRID_RESUME=1`**：已有完整 `test.csv` 的组合不再 `bsub`。中断后重新执行同一命令即可从缺口继续。

可选环境变量：

| 变量 | 含义 |
|------|------|
| `RESULTS_ROOT` | 结果根目录（默认 `distilbert_autogrid/results`） |
| `EPOCHS` | 训练轮数（默认见 `config.py`） |
| `GRID_RESUME` | `1`（默认）跳过已完成；`0` 全部提交 |
| `QUEUE` | LSF 队列（默认 `gpu`，见 `submit_bsub.sh`） |
| `LORA_TYPE` | `default` 或 `mlora` |
| `LORA_DROPOUT` | 默认 `0.05` |
| `BATCH_SIZE` | 默认 `4` |
| `GRAD_ACCUM_STEPS` | 默认 `8` |
| `CONDA_ROOT` / `CONDA_BASE` | Conda 根目录 |
| `CONDA_ENV_NAME` | 默认 `torch` |
| `GRID_MAX_RUN` | 见上文「自节流」；`0` 关闭 |
| `GRID_MAX_PEND` | 见上文；`0` 关闭 |
| `GRID_POLL_SEC` | 自节流轮询间隔，默认 `30` |

---

## 汇总

```bash
python -m distilbert_autogrid.aggregate_results
```

生成 `distilbert_autogrid/results/summary.csv`（按 `best_val_acc` 降序）。

---

## 实时看指标

```bash
cd ~/Manifold-Lora
bash distilbert/scripts/watch_metrics.sh
# 网格子目录：
# METRICS_DIR=distilbert_autogrid/results/<run_name> bash distilbert/scripts/watch_metrics.sh
```

---

## 仓库内脚本索引

| 路径 | 说明 |
|------|------|
| `distilbert_autogrid/run_grid_bsub.sh` | LSF 全因子网格，每组合一作业 |
| `distilbert_autogrid/run_grid.py` | 本地顺序跑网格 |
| `distilbert_autogrid/aggregate_results.py` | 生成 `summary.csv` |
| `distilbert/scripts/submit_bsub.sh` | 单次分类 `bsub` |
| `scripts/upload.sh`、`upload.ps1` | 本机上传到集群（仅 DistilBERT 相关） |
| `scripts/server_submit_distilbert_grid.sh` | 服务器：`sed` + `CONDA_ROOT` + `run_grid_bsub.sh` |
| `scripts/server_submit_distilbert_grid_force.sh` | 同上，`GRID_RESUME=0` |
| `scripts/commit_and_push.sh` | 交互式推 GitHub |

查看任务：`bjobs`；日志：`JOBID.out`、`JOBID.err`。

---

## 单次训练（非网格）

```bash
bash distilbert/scripts/submit_bsub.sh
```

说明见 [distilbert/README.md](../distilbert/README.md)。

---

## 提交到 GitHub

[`scripts/commit_and_push.sh`](../scripts/commit_and_push.sh)。勿将大规模 `results/` CSV 提交入库（[`results/.gitignore`](results/.gitignore)）。
