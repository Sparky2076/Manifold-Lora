## Manifold-Lora (个人实验 Fork)

基于原始 Manifold-Lora 项目的个人实验仓库，用于在 DistilBERT、DeepSeek-1.5B 等模型上做 LoRA / mLoRA 微调，并方便在学校 GPU 集群（SSH + `bsub`）上提交和监控任务。

本仓库主要包含两条流水线（形式对齐，便于对照）：

| 流水线 | 入口 | 提交脚本 | 指标 CSV（默认目录） |
|--------|------|----------|----------------------|
| **DistilBERT 文本分类** | `distilbert/main.py`（`python -m distilbert.main`） | 单次：`distilbert/scripts/submit_bsub.sh`；**全因子网格（每组合一作业）**：`distilbert_autogrid/run_grid_bsub.sh` | 单次：**`distilbert/results/`**；网格：**`distilbert_autogrid/results/<run_name>/`**（见该目录说明） |
| **DeepSeek 指令微调 SFT** | `deepseek/main_sft.py`（`python -m deepseek.main_sft`） | `deepseek/scripts/submit_bsub_sft.sh` | `deepseek/results/train_sft.csv`、`test_sft.csv`（或 `METRICS_DIR`） |

- **DistilBERT**：**`distilbert/`**（训练代码与 `scripts/`）；**`distilbert_autogrid/`**（网格配置、`run_grid` / `run_grid_bsub.sh`、`aggregate_results`）。
- **DeepSeek**：独立目录 **`deepseek/`**（代码、`deepseek/scripts/`、**`deepseek/results/`**）；两条线均**复用**根目录 `lora.py` / `mlora.py` / `optimizers.py`。

**DistilBERT 说明：[distilbert/README.md](distilbert/README.md)；网格与汇总：[distilbert_autogrid/README.md](distilbert_autogrid/README.md)。Git 提交/推送辅助：[scripts/commit_and_push.sh](scripts/commit_and_push.sh)。DeepSeek 见：[deepseek/README.md](deepseek/README.md)。**

---

### 0. 上传代码、提交任务与下载结果（SSH 服务器全流程）

改完代码后：**本机上传 → SSH 上 `sed` 换行 → `bsub` 提交 → 本机 `scp` 拉回 CSV**。下面按两条流水线分别写，**步骤编号与命令形式与 DistilBERT 保持一致**。

#### 0.1 DistilBERT 分类（`distilbert/`，`python -m distilbert.main`）

**① 本机上传到服务器**（本地 **Git Bash**，会提示输入服务器密码）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

`upload.sh` 会同步 **`distilbert/`**、**`distilbert_autogrid/`**、**`deepseek/`** 及根目录共享模块。

**② 服务器上修正脚本换行并提交**

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
```

- **全因子网格（推荐，每个超参组合一个 `bsub` 作业）**：

```bash
bash distilbert_autogrid/run_grid_bsub.sh
```

长时提交建议包在 **`tmux`** 里（断线不杀进程），以及 **`run_grid_bsub.sh` 默认会跳过 `test.csv` 已写满 `EPOCHS` 行的组合**，中断后同命令可续交。详见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md)（含 `tmux attach`、`GRID_RESUME=0` 强制全交）。

网格定义在 **`distilbert_autogrid/config.py`**；结果在 **`distilbert_autogrid/results/<run_name>/`**。跑完后汇总：`python -m distilbert_autogrid.aggregate_results`。

- **单次训练**（烟测或自定义 `METRICS_DIR`）：

```bash
bash distilbert/scripts/submit_bsub.sh
```

默认 **`METRICS_DIR=~/Manifold-Lora/distilbert/results`**。查看任务：`bjobs`；日志：`cat JOBID.out`、`cat JOBID.err`。

**③ 本机拉回 CSV**（将 IP 换成你的服务器；路径按实际 `METRICS_DIR` 修改）

单次默认目录示例：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/test.csv distilbert/results/
```

网格某一组合（将 `<run_name>` 换成实际目录名，在服务器上 `ls distilbert_autogrid/results` 查看）：

```bash
scp -r "wangxiao@202.121.138.196:~/Manifold-Lora/distilbert_autogrid/results/<run_name>" distilbert_autogrid/results/
```

---

#### 0.2 DeepSeek 指令微调 SFT（与 0.1 步骤一一对应，**文件均在 `deepseek/`**）

**① 本机上传**（与 0.1 相同；含 **`distilbert/`**、**`distilbert_autogrid/`**、**`deepseek/`**）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

**② 服务器：`sed` + 提交**

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
bash deepseek/scripts/submit_bsub_sft.sh
```

- 默认 **`METRICS_DIR=~/Manifold-Lora/deepseek/results`**，指标在 **`deepseek/results/train_sft.csv`**、**`test_sft.csv`**。
- 本地模型路径示例：`export MODEL_NAME=".../snapshots/<哈希>"` 后执行同上 `bash deepseek/scripts/submit_bsub_sft.sh`。

**③ 本机拉回 SFT 的 CSV**

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/train_sft.csv deepseek/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/test_sft.csv deepseek/results/
```

网格结果在服务器 **`deepseek/results/sft_grid/<预设>_lr_*/`**。学习率网格：`bash deepseek/scripts/gs_lr_deepseek_sft.sh`。

**更细的说明（监控、目录与 DistilBERT 的对应关系）→ [deepseek/README.md](deepseek/README.md)。**

---

### 1. 实时查看指标（与 CSV 形式对应）

| 脚本 | 适用流水线 | 监控文件（默认目录） |
|------|------------|----------------------|
| `distilbert/scripts/watch_metrics.sh` | DistilBERT 分类 | `METRICS_DIR` 下 `train.csv`、`test.csv`（默认 **`distilbert/results/`**；网格子目录同理） |
| `deepseek/scripts/watch_metrics_sft.sh` | DeepSeek SFT | **`deepseek/results/`** 下 `train_sft.csv`、`test_sft.csv` |

**DistilBERT（服务器）**：

```bash
ssh wangxiao@202.121.138.196
cd ~/Manifold-Lora
bash distilbert/scripts/watch_metrics.sh
# 或网格某一跑：
# METRICS_DIR=distilbert_autogrid/results/lr_3p0000e-03_r8_a8_ep3_wd_0p0000e+00 bash distilbert/scripts/watch_metrics.sh
```

**DeepSeek SFT（服务器）**：

```bash
cd ~/Manifold-Lora
bash deepseek/scripts/watch_metrics_sft.sh
```

指标在网格子目录时：

```bash
METRICS_DIR=deepseek/results/sft_grid/testing_alpaca_small_lr_2e_5 bash deepseek/scripts/watch_metrics_sft.sh
```

按 **Ctrl+C** 退出监控，不影响正在运行的 `bsub` 任务。

---

### 2. DistilBERT + LoRA 微调说明（默认配置，与脚本一致）

- 模型：`distilbert-base-uncased`
- 数据集：GLUE `sst2`，字段 `sentence`
- 训练超参（见 `distilbert/scripts/run_train_bsub.sh` 和 `distilbert/main.py`）：
  - `epochs = 50`（可由环境变量 `EPOCHS` 覆盖）
  - `batch_size = 4`、`grad_accum_steps = 8`
  - `lr = 1e-5`、`max_length = 128`
  - LoRA：`r=8, alpha=16, dropout=0.05`
- 指标文件：
  - 每次运行开始会**清空并重写**表头
  - `train.csv`：`iteration,train_loss,train_accuracy`
  - `test.csv`：`iteration,test_loss,test_accuracy`

---

### 3. DeepSeek SFT + LoRA（独立目录）

配置、默认超参、结果目录 **`deepseek/results/`**（含 `tuning_logs/`、`final_sft/`、`sft_grid/`）的说明见 **[deepseek/README.md](deepseek/README.md)** 与 **[docs/DEEPSEEK_FINETUNE_PLAN.md](docs/DEEPSEEK_FINETUNE_PLAN.md)**。

> **不推荐**：用 DistilBERT 分类流水线跑 DeepSeek；标准做法为 **`python -m deepseek.main_sft`**（在仓库根执行）。

---

### 4. 超参网格（Grid Search）

**本机仍只负责 `bash scripts/upload.sh`**（见第 0 节）；网格在**服务器**上执行。

#### 4.1 相关脚本说明

| 脚本 | 说明 |
|------|------|
| `distilbert_autogrid/run_grid_bsub.sh` | **DistilBERT 全因子网格**：`lr×r×alpha×weight_decay`，**每个组合一个 `bsub` 作业**；环境变量 **`GRID_RESUME`**（续跑跳过已完成）、**`tmux`** 防 SSH 断线（见 [`distilbert_autogrid/README.md`](distilbert_autogrid/README.md)） |
| `distilbert_autogrid/run_grid.py` | 同上网格的**本地顺序**执行（不占 LSF） |
| `distilbert_autogrid/aggregate_results.py` | 扫描各子目录 `test.csv`，写 `summary.csv` |
| `deepseek/scripts/gs_lr_deepseek_sft.sh` | DeepSeek SFT：`lr` 多组，结果 **`deepseek/results/sft_grid/`** |
| `deepseek/scripts/submit_bsub_sft.sh` | SFT 单 job；默认 `METRICS_DIR=$PROJECT_DIR/deepseek/results` |
| `deepseek/scripts/run_deepseek_sft_bsub.sh` | 计算节点执行 `python -m deepseek.main_sft` |
| `distilbert/scripts/submit_bsub.sh` | DistilBERT **单次**分类；转发 `EPOCHS`、`LR`、`LORA_*` 等 |
| `scripts/upload.sh`、`scripts/upload.ps1` | 上传 **`distilbert/`**、**`distilbert_autogrid/`**、**`deepseek/`** + 根目录共享模块 |
| `scripts/commit_and_push.sh` | 交互式 `git add` / `commit` / `push`（推 GitHub 时用） |

#### 4.2 在服务器上提交网格

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
```

- **DistilBERT 全因子网格**：`bash distilbert_autogrid/run_grid_bsub.sh`
- **DeepSeek SFT**：`bash deepseek/scripts/gs_lr_deepseek_sft.sh`

#### 4.3 查看任务与指标

```bash
bjobs
bjobs -d
cat JOBID.out
cat JOBID.err
```

- DistilBERT 单次：`tail -20 distilbert/results/train.csv` 等；网格：`tail -20 distilbert_autogrid/results/<run_name>/test.csv`
- SFT：`tail -20 deepseek/results/train_sft.csv` 等（或 **`deepseek/results/sft_grid/某目录/`** 下）

#### 4.4 保存结果到本机（与 0.1 / 0.2 一致）

分类（示例）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/test.csv distilbert/results/
```

SFT 默认目录 **`deepseek/results/`**：

```powershell
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/train_sft.csv deepseek/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/test_sft.csv deepseek/results/
```

DeepSeek 网格结果在 **`deepseek/results/sft_grid/`**。DistilBERT 全因子网格结果在 **`distilbert_autogrid/results/`**（勿将大体积 CSV 误提交 Git，见该目录 `.gitignore`）。

---

### 5. 本机直接试跑 SFT（可选，非 bsub）

在**仓库根目录**执行：

```bash
python -m deepseek.main_sft --trust_remote_code --device_map auto --torch_dtype float16 ^
  --sft_preset testing_alpaca_small --epochs 3 --batch_size 2 --max_length 512 --lr 2e-5 ^
  --metrics_dir deepseek/results
```

无 GPU 或勿在登录节点长跑时，请只用第 **0.2** 节的 `bsub` 流程。

---

### 6. 本机直接试跑 DistilBERT 分类（可选，非 bsub）

在**仓库根目录**执行：

```bash
python -m distilbert.main --model_name distilbert-base-uncased ^
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 --metrics_dir distilbert/results
```

更细的说明见 **[distilbert/README.md](distilbert/README.md)**。
