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
scp wangxiao@202.121.138.221:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp -r "wangxiao@202.121.138.221:~/Manifold-Lora/distilbert_autogrid/results/<run_name>" distilbert_autogrid/results/
```

一键拉回汇总文件（`summary.csv` / `missing_runs.csv` / `distilbert_grid_analysis.md`）：

```bash
bash scripts/pull_results.sh
# 或 PowerShell: .\scripts\pull_results.ps1
```

## 本地一键“拉取→汇总→分析→上传 GitHub”

推荐固定用以下脚本，避免漏步骤：

```bash
bash scripts/refresh_results_and_publish.sh
# 或 PowerShell: .\scripts\refresh_results_and_publish.ps1
```

脚本顺序：

1. 调用 `pull_results` 拉回 `summary.csv` / `missing_runs.csv` / `distilbert_grid_analysis.md`
2. 运行 `aggregate_results`（默认严格：`ok` 必须达到 375）
3. 运行 `analyze_results`（默认严格：`ok` 必须达到 375）
4. 仅 `git add` 结果汇总文件（不包含大规模逐 run CSV）
5. 自动 `git commit` + `git push`

可选环境变量：

| 变量 | 含义 |
|------|------|
| `SERVER` / `REMOTE_DIR` | 远程服务器与目录（传给 `pull_results`） |
| `COMMIT_MSG` | 提交说明（默认 `Update distilbert grid results (summary/missing/analysis)`） |
| `ALLOW_INCOMPLETE=1` | 允许未满 375 时继续（默认不建议） |

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

### mLoRA 全同网格（与 LoRA 同参数空间）

直接用包装脚本（默认写入 `distilbert_autogrid/results_mlora/`，避免覆盖 LoRA 结果）。与下文「默认后台方式：nohup（推荐）」一节一致，建议 **`nohup`** 后台提交（日志文件名与 LoRA 的 `grid_submit.log` 区分）：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
export GRID_MAX_RUN=5
export GRID_MAX_PEND=1
export GRID_POLL_SEC=30
nohup bash scripts/server_submit_distilbert_grid_mlora.sh > grid_mlora_submit.log 2>&1 &
tail -f grid_mlora_submit.log
```

等价于手动设置（同样建议 `nohup`）：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
export LORA_TYPE=mlora
export RESULTS_ROOT=distilbert_autogrid/results_mlora
export GRID_MAX_RUN=5
export GRID_MAX_PEND=1
export GRID_POLL_SEC=30
nohup bash scripts/server_submit_distilbert_grid.sh > grid_mlora_submit.log 2>&1 &
tail -f grid_mlora_submit.log
```

### 网格筛完后：最优超参 20 epoch（终局复现）

`aggregate_results` 写出的 `summary.csv` 已按 **`best_val_acc` 降序** 排序，**第一行数据**即当前最优组合。下面脚本从对应 `summary.csv` 读首行超参，各提交 **一个** `bsub`，**`EPOCHS=20`**：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"

bash scripts/server_submit_distilbert_best_lora_20ep.sh    # LoRA → distilbert/results_final_best_lora_20ep/
bash scripts/server_submit_distilbert_best_mlora_20ep.sh    # mLoRA → distilbert/results_final_best_mlora_20ep/
```

等价：`bash scripts/server_submit_distilbert_best_20ep.sh lora` / `... mlora`。跑完后用 `bjobs`、`JOBID.out` / `.err` 查看；指标在各自 `METRICS_DIR` 下的 `train.csv` / `test.csv`。若要把终局目录拉回本机，可 `scp -r` 上述目录（体积大时勿整库 `git add`）。

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

## 默认后台方式：nohup（推荐）

默认统一使用 **`nohup`** 跑提交循环，SSH 断开后进程仍在：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
export GRID_MAX_RUN=5
export GRID_MAX_PEND=1
export GRID_POLL_SEC=30
nohup bash scripts/server_submit_distilbert_grid.sh > grid_submit.log 2>&1 &
tail -f grid_submit.log
```

断线后进程仍在；勿关登录节点上该进程（勿 `kill`）。查看：`tail -f grid_submit.log`。

### 可选：tmux / screen

如果你更习惯会话管理，`tmux` / `screen` 仍可用（非默认）：

- `tmux new -s grid`，重连：`tmux attach -t grid`
- `screen -S grid`，重连：`screen -r grid`

### nohup 场景：查找并停止 grid 提交脚本

`nohup` 方式下，先定位仍在跑的 `run_grid_bsub.sh`：

```bash
pgrep -af run_grid_bsub
```

常见会看到两条（父/子 bash，来自脚本里的管道 `while`），可用下面命令确认：

```bash
ps -o pid,ppid,cmd -p <PID1>,<PID2>
```

优先结束**父进程**（`PPID` 不是另一个 `run_grid_bsub` 的那条）：

```bash
kill <PARENT_PID>
pgrep -af run_grid_bsub
```

若仍残留，再 `kill <CHILD_PID>`（必要时 `kill -9`）。

> 这只会停止「继续 `bsub` 的提交循环」，不会取消已进 LSF 队列的训练作业。
> 若要撤队列中的网格作业，请用：`bash scripts/kill_distilbert_grid_bjobs.sh --yes`

---

## 网格提交脚本：续跑 vs 强交 vs 停止

| 场景 | 命令 | 说明 |
|------|------|------|
| **接着交 / 不重跑已完成的**（默认） | `bash scripts/server_submit_distilbert_grid.sh` 或 `bash distilbert_autogrid/run_grid_bsub.sh` | 默认 **`GRID_RESUME=1`**：某目录 `test.csv` 已有 ≥ `EPOCHS` 行则 **skip**。 |
| **强制全部重交** | `bash scripts/server_submit_distilbert_grid_force.sh` | 固定 **`GRID_RESUME=0`**。等价：`GRID_RESUME=0 bash scripts/server_submit_distilbert_grid.sh`。 |
| **计算节点 Conda** | `export CONDA_ROOT=...` | 见上文；与 [`distilbert/scripts/run_train_bsub.sh`](../distilbert/scripts/run_train_bsub.sh) 一致。 |

**停止「自动 bsub」脚本**：在跑 `server_submit_*.sh` 或 `run_grid_bsub.sh` 的终端按 **`Ctrl+C`** → 只结束**提交循环**，**不会**取消已提交的作业。

**取消队列里的作业**：可手动 `bjobs` → `bkill <JOBID>`，或用仓库提供的批量脚本（见下节）。

**Pending 节流**：出现 `Pending job threshold reached. Retrying in 60 seconds...` 为站点对同时 **PEND** 作业数限制，属正常；集群侧会重试；若仍出现 **`User permission denied`**，需减少同时占用队列的作业数。

**自节流（`run_grid_bsub.sh` 内建）**：每次 `bsub` **前**用 `bjobs` 数你的 `RUN` / `PEND`，**超过上限则先 sleep 再试**。

含义（数字即「最多允许几」，直观）：

| 变量 | 含义 |
|------|------|
| `GRID_MAX_RUN` | 默认 `0`（关闭）。设为 **`5`**：**RUN 数 > 5** 时暂停（即最多 **5 个 RUN**；5 RUN + 0 PEND 时仍可再交 1 个进 PEND，凑满「5 跑 + 1 等」）。 |
| `GRID_MAX_PEND` | 默认 `0`（关闭）。设为 **`1`**：**PEND 数 ≥ 1** 时暂停 → **只有 PEND=0 时才 `bsub`**，避免连续提交把 PEND 堆成 2（与「最多 1 个在等」一致）。 |
| `GRID_POLL_SEC` | 轮询间隔秒数，默认 `30`。 |

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

## 批量终止网格 LSF 作业：`scripts/kill_distilbert_grid_bjobs.sh`

按作业名前缀 **`distilbert_grid`**（与 `run_grid_bsub.sh` 里 `JOB_NAME=distilbert_grid_<组合>`、`bsub -J` 一致）筛选**当前用户**未结束作业，再 **`bkill`**。在**登录节点**、仓库根目录执行：

```bash
cd ~/Manifold-Lora
bash scripts/kill_distilbert_grid_bjobs.sh
```

- 默认会先打印待杀的 JobID，**需输入 `y` 确认**。
- **不询问直接杀**：`bash scripts/kill_distilbert_grid_bjobs.sh --yes` 或 `GRID_KILL_YES=1 bash scripts/kill_distilbert_grid_bjobs.sh`。

| 环境变量 | 含义 |
|----------|------|
| `GRID_JOB_PREFIX` | 默认 `distilbert_grid`；只杀 `JOB_NAME` 以该字符串**开头**的作业 |
| `GRID_KILL_YES` | 设为 `1` 时等同 `--yes` |

**与「停提交脚本」的区别**：

| 操作 | 作用 |
|------|------|
| `kill` / `Ctrl+C` 结束 `run_grid_bsub.sh` | 只停**本机循环**，**不会**撤队列里已 `bsub` 的作业 |
| `kill_distilbert_grid_bjobs.sh` | 撤 **LSF 里**仍排队或运行中的网格训练作业 |

**依赖**：需要 `bjobs -o 'jobid job_name'`（常见 LSF 9+）。若报错，请手动：`bjobs -u $USER -w` 找到 `distilbert_grid` 相关行，再 `bkill <JOBID>`。

---

### 断点续交与 `GRID_RESUME`

`run_grid_bsub.sh` 默认 **`GRID_RESUME=1`**：已有完整 `test.csv` 的组合不再 `bsub`。中断后重新执行同一命令即可从缺口继续。

脚本还会在每轮提交后等待当前 `distilbert_grid_*` 作业跑完，并再次扫描未完成组合；若有失败/缺失会继续补交，直到全部完成（或达到 `GRID_MAX_PASSES`）。

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
| `EXCLUDE_HOSTS` | 默认 `gpu17`；传给 `submit_bsub.sh` 作为主机过滤（逗号分隔，如 `gpu17,gpu18`） |
| `CONDA_ROOT` / `CONDA_BASE` | Conda 根目录 |
| `CONDA_ENV_NAME` | 默认 `torch` |
| `GRID_MAX_RUN` | 见上文「自节流」；`0` 关闭 |
| `GRID_MAX_PEND` | 见上文；`0` 关闭 |
| `GRID_POLL_SEC` | 自节流轮询间隔，默认 `30` |
| `SUBMIT_SLEEP_SEC` | 两次 `bsub` 之间的间隔秒数，默认 **`180`（3 分钟）**，减轻同节点 GPU 扎堆；临时加快可设 `SUBMIT_SLEEP_SEC=30` |
| `GRID_MAX_PASSES` | 自动补交循环最大轮数，默认 `0`（直到全部完成才结束） |

---

## 精确补齐缺失组合（375 对照）

当你想明确知道「还缺哪些组合」并只补交缺失项时：

```bash
python -m distilbert_autogrid.fill_missing_runs
```

会生成：`distilbert_autogrid/results/missing_runs.csv`，按 `config.py` 全网格（375）对照当前 `results/`，输出每条缺失记录的 `reason`（如 `missing_dir`、`missing_test`、`incomplete_test_rows`）。

当前常见状态是已完成约 246 组、缺失约 129 组；可先执行上面的检测命令确认实时缺口数量。

仅提交缺失项（不重交已完成）：

```bash
python -m distilbert_autogrid.fill_missing_runs --submit-bsub
```

如需后台自动补齐（`nohup`）：

```bash
cd ~/Manifold-Lora
nohup python -m distilbert_autogrid.fill_missing_runs --submit-bsub > fill_missing.log 2>&1 &
tail -f fill_missing.log
```

- 脚本会按 `test.csv` 行数对照 `EPOCHS` 判定完成度；**已完成的 246 组不会再次提交**。
- 提交仍走 `distilbert/scripts/submit_bsub.sh`，会继承 `EXCLUDE_HOSTS`（默认避开 `gpu17`）。
- 可配节流：`--grid-max-run` / `--grid-max-pend` / `--grid-poll-sec`。
- 提交间隔：`--submit-sleep-sec`（默认取环境变量 `SUBMIT_SLEEP_SEC`，当前默认 180 秒）。

补交跑完后再汇总并合并到已有结果：

```bash
python -m distilbert_autogrid.aggregate_results
python -m distilbert_autogrid.analyze_results
```

`summary.csv` 会自动包含旧结果 + 新补齐结果（按 `metrics_dir` 扫描聚合）。

> 现在 `aggregate_results` / `analyze_results` 默认会在 **`ok < 375`** 时报错退出，防止你误生成“未跑满”的最终汇总；若仅做阶段性检查，显式加 `--allow-incomplete`。

---

## 汇总

```bash
python -m distilbert_autogrid.aggregate_results
python -m distilbert_autogrid.analyze_results
```

生成 `distilbert_autogrid/results/summary.csv`（按 `best_val_acc` 降序），并写 **`distilbert_autogrid/results/distilbert_grid_analysis.md`**（分组均值/Top15 等）。仓库中可跟踪 `summary.csv` 与分析文档（见 `results/.gitignore`）；逐组 `train.csv`/`test.csv` 仍默认不入库。快照说明见 [`results/distilbert_grid_snapshot.md`](results/distilbert_grid_snapshot.md)。

若仅需查看中间结果（尚未跑满 375）：

```bash
python -m distilbert_autogrid.aggregate_results --allow-incomplete
python -m distilbert_autogrid.analyze_results --allow-incomplete
```

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
| `distilbert_autogrid/fill_missing_runs.py` | 对照 375 全网格生成 `missing_runs.csv`，并可仅补交缺失项 |
| `distilbert_autogrid/aggregate_results.py` | 生成 `summary.csv` |
| `distilbert_autogrid/analyze_results.py` | 从 `summary.csv` 生成 `results/distilbert_grid_analysis.md` |
| `distilbert/scripts/submit_bsub.sh` | 单次分类 `bsub` |
| `scripts/upload.sh`、`upload.ps1` | 本机上传到集群（仅 DistilBERT 相关） |
| `scripts/kill_distilbert_grid_bjobs.sh` | 按作业名前缀 `distilbert_grid*` 批量 `bkill` |
| `scripts/server_submit_distilbert_grid.sh` | 服务器：`sed` + `CONDA_ROOT` + `run_grid_bsub.sh` |
| `scripts/server_submit_distilbert_grid_force.sh` | 同上，`GRID_RESUME=0` |
| `scripts/server_submit_distilbert_grid_mlora.sh` | 服务器 mLoRA 网格（默认 `LORA_TYPE=mlora`，`RESULTS_ROOT=results_mlora`）；**推荐** `nohup … > grid_mlora_submit.log 2>&1 &`（见「mLoRA 全同网格」） |
| `scripts/server_submit_distilbert_best_20ep.sh` | 从 `summary.csv` 首行读最优超参，单作业 **20 epoch**（参数：`lora` / `mlora`） |
| `scripts/server_submit_distilbert_best_lora_20ep.sh` | 同上，LoRA 便捷封装 |
| `scripts/server_submit_distilbert_best_mlora_20ep.sh` | 同上，mLoRA 便捷封装 |
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
