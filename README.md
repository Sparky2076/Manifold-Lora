## Manifold-Lora (个人实验 Fork)

本仓库当前以 **DistilBERT 文本分类 + LoRA 全因子网格**（`distilbert/`、`distilbert_autogrid/`）为主，在学校 GPU 集群（SSH + `bsub`）上提交与汇总。**`deepseek/`** 目录已清空，仅保留占位说明，便于后续重写 **DeepSeek 网格 / SFT**（见 [`deepseek/README.md`](deepseek/README.md)）。

训练**复用**根目录 `lora.py`、`mlora.py`、`optimizers.py`。

| 流水线 | 入口 | 提交脚本 | 指标（默认） |
|--------|------|----------|----------------|
| **DistilBERT 分类** | `python -m distilbert.main` | 单次：`distilbert/scripts/submit_bsub.sh`；网格：`distilbert_autogrid/run_grid_bsub.sh` | `distilbert/results/` 或 `distilbert_autogrid/results/<run_name>/` |

**说明**：[distilbert/README.md](distilbert/README.md)、[distilbert_autogrid/README.md](distilbert_autogrid/README.md)；Git 辅助：[scripts/commit_and_push.sh](scripts/commit_and_push.sh)。

---

### 0. 上传服务器、提交网格、拉回结果

**① 本机上传**（**仅** DistilBERT 与网格相关：`distilbert/`、`distilbert_autogrid/`、根目录 `lora.py` / `mlora.py` / `optimizers.py`、`scripts/`）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

PowerShell：`.\scripts\upload.ps1`。

**② 服务器：一键提交（推荐）或手动**

一键（含 `sed`、可选自动探测 `CONDA_ROOT`、再调 `run_grid_bsub.sh`）：

```bash
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"   # 建议显式设置，与计算节点一致
bash scripts/server_submit_distilbert_grid.sh
```

**强制全部重交**（`GRID_RESUME=0`，不跳过已有 `test.csv`），建议同样包在 **tmux** 里：

```bash
tmux new -s grid
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
bash scripts/server_submit_distilbert_grid_force.sh
```

（等价：`GRID_RESUME=0 bash scripts/server_submit_distilbert_grid.sh`。）

或手动：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
export CONDA_ROOT="$HOME/miniconda3"
bash distilbert_autogrid/run_grid_bsub.sh
```

全因子网格（每组合一个 `bsub`；建议 `tmux`、续跑见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md)）。

单次训练：`bash distilbert/scripts/submit_bsub.sh`。

跑完后汇总：`python -m distilbert_autogrid.aggregate_results`。

**③ 本机拉回 CSV**（IP 换成你的服务器）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp -r "wangxiao@202.121.138.196:~/Manifold-Lora/distilbert_autogrid/results/<run_name>" distilbert_autogrid/results/
```

#### 网格提交脚本：该用哪个、tmux、怎么停

| 场景 | 用什么 | 说明 |
|------|--------|------|
| **接着交 / 不重跑已完成的**（默认） | `bash scripts/server_submit_distilbert_grid.sh` | 内部调用 `run_grid_bsub.sh`，默认 **`GRID_RESUME=1`**：某目录 `test.csv` 已有 ≥ `EPOCHS` 行则 **skip**，不再 `bsub`。也可直接：`bash distilbert_autogrid/run_grid_bsub.sh`（需自行 `sed` 时见上表「手动」）。 |
| **强制全部重交**（覆盖重跑每一格） | `bash scripts/server_submit_distilbert_grid_force.sh` | 固定 **`GRID_RESUME=0`**，不跳过。等价：`GRID_RESUME=0 bash scripts/server_submit_distilbert_grid.sh`。 |
| **计算节点 Conda** | `export CONDA_ROOT="$HOME/miniconda3"` | 批作业里常需显式设置（路径下须有 `etc/profile.d/conda.sh`），与 [`distilbert/scripts/run_train_bsub.sh`](distilbert/scripts/run_train_bsub.sh) 一致。 |

**tmux（推荐，SSH 断线后提交循环一般仍继续）**：

```bash
tmux new -s grid
cd ~/Manifold-Lora
export CONDA_ROOT="$HOME/miniconda3"
bash scripts/server_submit_distilbert_grid.sh          # 续跑 skip
# 或： bash scripts/server_submit_distilbert_grid_force.sh   # 全部重交
# 离开会话：Ctrl+B，松手后再按 D     重连：tmux attach -t grid     列表：tmux ls
```

**停止提交脚本**：在运行 `bash scripts/server_submit_*.sh` 或 `run_grid_bsub.sh` 的终端按 **`Ctrl+C`** → 只结束**本机上的提交循环**，**之后不会再自动 bsub**。已在队列里或运行中的作业 **不会** 因此被取消。

**取消已提交的作业**：`bjobs` 查看 JobID，再 **`bkill <JOBID>`**（或集群提供的批量命令）。详见集群 LSF 文档。

**Pending 节流**：若出现 `Pending job threshold reached. Retrying in 60 seconds...`，为站点对同时 **PEND** 作业数限制，属正常；脚本会等待后**继续**提交，无需改仓库脚本。

更细的网格参数、汇总命令见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md)。

---

### 1. 实时看指标（DistilBERT）

```bash
cd ~/Manifold-Lora
bash distilbert/scripts/watch_metrics.sh
# 网格子目录：METRICS_DIR=distilbert_autogrid/results/<run_name> bash distilbert/scripts/watch_metrics.sh
```

---

### 2. DistilBERT 默认配置要点

- 模型：`distilbert-base-uncased`；数据：GLUE `sst2`。
- 单次 `run_train_bsub` 默认 `epochs=50` 等可被环境变量覆盖；指标：`train.csv`、`test.csv`（列名见 `distilbert/main.py`）。

---

### 3. DeepSeek（占位）

旧版 **`deepseek/`** 代码与实验记录已移除。规划与任务形态可参考 **[docs/DEEPSEEK_FINETUNE_PLAN.md](docs/DEEPSEEK_FINETUNE_PLAN.md)**；新流水线将写在 **`deepseek/`**（占位见 [deepseek/README.md](deepseek/README.md)）。

---

### 4. 超参网格脚本索引

| 脚本 | 说明 |
|------|------|
| `distilbert_autogrid/run_grid_bsub.sh` | LSF：全因子网格，每组合一作业；**续跑/强交/tmux/停止** 见上文 **§0「网格提交脚本」**；参数细节见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md) |
| `distilbert_autogrid/run_grid.py` | 本地顺序跑网格 |
| `distilbert_autogrid/aggregate_results.py` | 生成 `summary.csv` |
| `distilbert/scripts/submit_bsub.sh` | 单次分类作业 |
| `scripts/upload.sh`、`upload.ps1` | **仅**上传 DistilBERT 网格相关文件（见上文） |
| `scripts/server_submit_distilbert_grid.sh` | 服务器端：`sed` + 可选探测 `CONDA_ROOT` + `run_grid_bsub.sh` |
| `scripts/server_submit_distilbert_grid_force.sh` | 同上但 **`GRID_RESUME=0`**（强制全部重交）；注释含 tmux |
| `scripts/commit_and_push.sh` | 交互式提交推送 GitHub |

服务器上提交前 **`sed`** 与 **§0** 一致；查看任务：`bjobs`，日志：`JOBID.out` / `JOBID.err`。

---

### 5. 本机试跑 DistilBERT（非 bsub）

```bash
python -m distilbert.main --model_name distilbert-base-uncased ^
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 --metrics_dir distilbert/results
```

详见 [distilbert/README.md](distilbert/README.md)。
