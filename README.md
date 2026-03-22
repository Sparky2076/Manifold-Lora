## Manifold-Lora (个人实验 Fork)

基于原始 Manifold-Lora 项目的个人实验仓库，用于在 DistilBERT、DeepSeek-1.5B 等模型上做 LoRA / mLoRA 微调，并方便在学校 GPU 集群（SSH + `bsub`）上提交和监控任务。

本仓库主要包含两条流水线（形式对齐，便于对照）：

| 流水线 | 入口 | 提交脚本 | 指标 CSV（默认目录） |
|--------|------|----------|----------------------|
| **DistilBERT 文本分类** | `main.py` | `scripts/submit_bsub.sh` | `train.csv`、`test.csv`（项目根或 `METRICS_DIR`） |
| **DeepSeek 指令微调 SFT** | `deepseek/main_sft.py`（`python -m deepseek.main_sft`） | `deepseek/scripts/submit_bsub_sft.sh` | `deepseek/results/train_sft.csv`、`test_sft.csv`（或 `METRICS_DIR`） |

- **DistilBERT**：代码与脚本在仓库**根目录**（`main.py`、`scripts/`、`results/`）。
- **DeepSeek**：独立目录 **`deepseek/`**（代码、`deepseek/scripts/`、**`deepseek/results/`**），与 `results/` 并列；训练仍**复用**根目录 `lora.py` / `mlora.py` / `optimizers.py`（**不改** `main.py` / `models.py` / `utils.py`）。

**DeepSeek 全流程、网格、监控的完整说明见：[deepseek/README.md](deepseek/README.md)。**

---

### 0. 上传代码、提交任务与下载结果（SSH 服务器全流程）

改完代码后：**本机上传 → SSH 上 `sed` 换行 → `bsub` 提交 → 本机 `scp` 拉回 CSV**。下面按两条流水线分别写，**步骤编号与命令形式与 DistilBERT 保持一致**。

#### 0.1 DistilBERT 分类（`main.py`）

**① 本机上传到服务器**（本地 **Git Bash**，会提示输入服务器密码）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

**② 服务器上修正脚本换行并提交训练任务**（SSH 登录后执行）：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh
bash scripts/submit_bsub.sh
```

- 第一句 `sed`：把 Windows 上传的 `.sh` 从 CRLF 改为 LF，避免 `$'\r': command not found` / `set: pipefail`。
- 第二句：提交单卡任务，默认 **DistilBERT + GLUE SST2**。查看任务：`bjobs`；日志：`cat JOBID.out`、`cat JOBID.err`。

**③ 本机保存训练结果 CSV**（本地 **PowerShell** 或 **Git Bash**；将 `196` 换成当前学校服务器 IP，如 `197`）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/train.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test.csv .
```

```bash
cd /d/GitHub_Code/Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/train.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test.csv .
```

若任务里设置了 `METRICS_DIR` 为子目录（如网格搜索），把远端路径改成该目录下的 `train.csv` / `test.csv`。

---

#### 0.2 DeepSeek 指令微调 SFT（与 0.1 步骤一一对应，**文件均在 `deepseek/`**）

**① 本机上传**（与 0.1 相同；`upload.sh` 会 **`scp -r deepseek/`** 整包上传）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

**② 服务器：`sed` + 提交**

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh deepseek/scripts/*.sh
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
| `scripts/watch_metrics.sh` | DistilBERT `main.py` | 项目根 `train.csv`、`test.csv` |
| `deepseek/scripts/watch_metrics_sft.sh` | DeepSeek SFT | **`deepseek/results/`** 下 `train_sft.csv`、`test_sft.csv` |

**DistilBERT（服务器）**：

```bash
ssh wangxiao@202.121.138.196
cd ~/Manifold-Lora
bash scripts/watch_metrics.sh
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
- 训练超参（见 `scripts/run_train_bsub.sh` 和 `main.py`）：
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

> **不推荐**：用 `main.py` + GLUE 分类跑 DeepSeek；标准做法为 **`python -m deepseek.main_sft`**（在仓库根执行）。

---

### 4. 学习率网格搜索（Grid Search）

**本机仍只负责 `bash scripts/upload.sh`**（见第 0 节）；网格在**服务器**上执行。

#### 4.1 相关脚本说明

| 脚本 | 说明 |
|------|------|
| `scripts/gs_lr_lora.sh` | DistilBERT + LoRA：`lr` 多组 × 固定 epoch，多个 job |
| `scripts/gs_lr_mlora.sh` | DistilBERT + mLoRA：同上 |
| `deepseek/scripts/gs_lr_deepseek_sft.sh` | DeepSeek SFT 网格：`lr` 多组，结果写入 **`deepseek/results/sft_grid/`** |
| `deepseek/scripts/submit_bsub_sft.sh` | SFT 单 job；默认 `METRICS_DIR=$PROJECT_DIR/deepseek/results` |
| `deepseek/scripts/run_deepseek_sft_bsub.sh` | 计算节点执行 `python -m deepseek.main_sft` |
| `scripts/submit_bsub.sh` | DistilBERT 分类；转发 `EPOCHS`、`LR`、`LORA_*` 等 |
| `scripts/upload.sh`、`scripts/upload.ps1` | 上传根目录分类相关 `.py` + **`scp -r deepseek/`** |

#### 4.2 在服务器上提交网格

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh deepseek/scripts/*.sh
```

- **LoRA**：`bash scripts/gs_lr_lora.sh`
- **mLoRA**：`bash scripts/gs_lr_mlora.sh`
- **DeepSeek SFT**：`bash deepseek/scripts/gs_lr_deepseek_sft.sh`

#### 4.3 查看任务与指标

```bash
bjobs
bjobs -d
cat JOBID.out
cat JOBID.err
```

- 分类：`tail -20 train.csv`、`tail -20 test.csv`（或你的 `METRICS_DIR`）
- SFT：`tail -20 deepseek/results/train_sft.csv` 等（或 **`deepseek/results/sft_grid/某目录/`** 下）

#### 4.4 保存结果到本机（与 0.1 / 0.2 一致）

分类（示例）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/train.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test.csv .
```

SFT 默认目录 **`deepseek/results/`**：

```powershell
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/train_sft.csv deepseek/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/test_sft.csv deepseek/results/
```

网格结果在 **`deepseek/results/sft_grid/`**；归档时请放入本仓库 **`deepseek/results/`** 下对应子目录（与 DistilBERT 使用根目录 **`results/`** 对称）。

---

### 5. 本机直接试跑 SFT（可选，非 bsub）

在**仓库根目录**执行：

```bash
python -m deepseek.main_sft --trust_remote_code --device_map auto --torch_dtype float16 ^
  --sft_preset testing_alpaca_small --epochs 3 --batch_size 2 --max_length 512 --lr 2e-5 ^
  --metrics_dir deepseek/results
```

无 GPU 或勿在登录节点长跑时，请只用第 **0.2** 节的 `bsub` 流程。
