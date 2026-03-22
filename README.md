## Manifold-Lora (个人实验 Fork)

基于原始 Manifold-Lora 项目的个人实验仓库，用于在 DistilBERT、DeepSeek-1.5B 等模型上做 LoRA / mLoRA 微调，并方便在学校 GPU 集群（SSH + `bsub`）上提交和监控任务。

本仓库主要包含两条流水线（形式对齐，便于对照）：

| 流水线 | 入口 | 提交脚本 | 指标 CSV（默认目录） |
|--------|------|----------|----------------------|
| **DistilBERT 文本分类** | `main.py` | `scripts/submit_bsub.sh` | `train.csv`、`test.csv`（项目根或 `METRICS_DIR`） |
| **DeepSeek 指令微调 SFT** | `main_sft.py` | `scripts/submit_bsub_sft.sh` | `train_sft.csv`、`test_sft.csv`（项目根或 `METRICS_DIR`） |

共用：`lora.py` / `mlora.py`、`optimizers.py`；SFT 另含 `models_sft.py`、`utils_sft.py`（**不修改**原 `models.py` / `utils.py` / `main.py`）。

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

#### 0.2 DeepSeek 指令微调 SFT（`main_sft.py`，与 0.1 步骤一一对应）

**① 本机上传到服务器**（与 0.1 相同，`upload.sh` / `upload.ps1` 已包含 `main_sft.py`、`models_sft.py`、`utils_sft.py` 及 SFT 相关 `.sh`）：

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

**② 服务器上修正脚本换行并提交 SFT 任务**：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh
bash scripts/submit_bsub_sft.sh
```

- 默认：**DeepSeek-R1-Distill-Qwen-1.5B** + 小指令集预设 `testing_alpaca_small`（可在 `scripts/run_deepseek_sft_bsub.sh` 或环境变量里改 `MODEL_NAME`、`SFT_PRESET`）。
- 使用 **本地 HF 缓存模型路径**时，与原先 DistilBERT 流程类似，在服务器设 `MODEL_NAME` 再提交，例如：
  ```bash
  export MODEL_NAME="/nfsshare/home/wangxiao/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/<SNAP哈希>"
  bash scripts/submit_bsub_sft.sh
  ```

**③ 本机保存 SFT 指标 CSV**（默认写在项目根目录；列名与分类不同，文件名为 `train_sft.csv` / `test_sft.csv`）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/train_sft.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test_sft.csv .
```

网格搜索时，每个学习率对应目录在服务器 `~/Manifold-Lora/results/sft_grid/` 下，例如：

```powershell
scp wangxiao@202.121.138.196:~/Manifold-Lora/results/sft_grid/testing_alpaca_small_lr_1e_5/train_sft.csv ./results/sft_grid/
```

（请先在本机建好 `results/sft_grid/` 等目录，或改成你希望的路径。）

**学习率网格（与 4.2 节 DistilBERT 网格形式一致，在服务器执行）**：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh
bash scripts/gs_lr_deepseek_sft.sh
```

各 job 的 `METRICS_DIR` 为 `results/sft_grid/<预设>_lr_<学习率>/`，内含 `train_sft.csv`、`test_sft.csv`。

---

### 1. 实时查看指标（与 CSV 形式对应）

| 脚本 | 适用流水线 | 监控文件 |
|------|------------|----------|
| `scripts/watch_metrics.sh` | DistilBERT `main.py` | `train.csv`、`test.csv`（项目根） |
| `scripts/watch_metrics_sft.sh` | DeepSeek SFT `main_sft.py` | `train_sft.csv`、`test_sft.csv` |

**DistilBERT（服务器）**：

```bash
ssh wangxiao@202.121.138.196
cd ~/Manifold-Lora
bash scripts/watch_metrics.sh
```

**DeepSeek SFT（服务器）**：

```bash
cd ~/Manifold-Lora
bash scripts/watch_metrics_sft.sh
```

指标在子目录时：

```bash
METRICS_DIR=results/sft_grid/testing_alpaca_small_lr_2e_5 bash scripts/watch_metrics_sft.sh
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

### 3. DeepSeek SFT + LoRA 说明（与第 2 节形式对齐）

- 模型：默认 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（`scripts/run_deepseek_sft_bsub.sh` 中 `MODEL_NAME`，建议服务器改用本地 snapshot 路径）
- 数据：**Hub 小指令集预设**（`--sft_preset`），默认 `testing_alpaca_small`；可选 `alpaca_gpt4_500`、`alpaca_train_500`、`alpaca_train_1k`（见 `utils_sft.py`）
- 训练超参（见 `scripts/run_deepseek_sft_bsub.sh` / `main_sft.py`）：
  - `epochs = 5`、`batch_size = 2`、`grad_accum_steps = 8`
  - `max_length = 512`、`lr = 2e-5`（网格脚本可覆盖 `LR`）
  - `torch_dtype = float16`、`device_map = auto`（与单卡大模型常见设定一致）
  - LoRA：`r=8, alpha=16, dropout=0.05`，目标模块含 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,...`
- 指标文件（**不与**分类的 `train.csv` 混用）：
  - `train_sft.csv`：`iteration,train_loss,train_perplexity`
  - `test_sft.csv`：`iteration,eval_loss,eval_perplexity`
- 任务背景与更多数据集说明：**[docs/DEEPSEEK_FINETUNE_PLAN.md](docs/DEEPSEEK_FINETUNE_PLAN.md)**

> **不推荐**：用 `main.py` + GLUE 分类任务跑 DeepSeek（生成模型与分类头不匹配）。若必须试跑，需自行改参；标准做法请用本节 **SFT + `main_sft.py`**。

---

### 4. 学习率网格搜索（Grid Search）

**本机仍只负责 `bash scripts/upload.sh`**（见第 0 节）；网格在**服务器**上执行。

#### 4.1 相关脚本说明

| 脚本 | 说明 |
|------|------|
| `scripts/gs_lr_lora.sh` | DistilBERT + LoRA：`lr` 多组 × 固定 epoch，多个 job |
| `scripts/gs_lr_mlora.sh` | DistilBERT + mLoRA：同上 |
| `scripts/gs_lr_deepseek_sft.sh` | DeepSeek SFT：默认 `lr` 为 `1e-5 2e-5 5e-5`，结果写入 `results/sft_grid/.../` |
| `scripts/submit_bsub.sh` | 分类任务；转发 `EPOCHS`、`LR`、`LORA_*` 等 |
| `scripts/submit_bsub_sft.sh` | SFT 任务；转发 `EPOCHS`、`LR`、`LORA_*`、`SFT_PRESET`、`SFT_DATASET`、`MAX_LENGTH` 等 |
| `scripts/upload.sh`、`scripts/upload.ps1` | 上传核心 `.py` 与上述 `.sh`（含 SFT 与 `watch_metrics_sft.sh`） |

#### 4.2 在服务器上提交网格

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh
```

- **LoRA**：`bash scripts/gs_lr_lora.sh`
- **mLoRA**：`bash scripts/gs_lr_mlora.sh`
- **DeepSeek SFT**：`bash scripts/gs_lr_deepseek_sft.sh`

#### 4.3 查看任务与指标

```bash
bjobs
bjobs -d
cat JOBID.out
cat JOBID.err
```

- 分类：`tail -20 train.csv`、`tail -20 test.csv`（或你的 `METRICS_DIR`）
- SFT：`tail -20 train_sft.csv`、`tail -20 test_sft.csv`（或 `results/sft_grid/某目录/` 下）

#### 4.4 保存结果到本机（与 0.1 / 0.2 一致）

分类（示例）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/train.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test.csv .
```

SFT 单次任务写在项目根时：

```powershell
scp wangxiao@202.121.138.196:~/Manifold-Lora/train_sft.csv .
scp wangxiao@202.121.138.196:~/Manifold-Lora/test_sft.csv .
```

网格结果按需 `scp` 整个 `results/sft_grid/` 下某一子目录或打包归档后提交到本仓库 `results/`。

---

### 5. 本机直接试跑 SFT（可选，非 bsub）

```bash
python main_sft.py --trust_remote_code --device_map auto --torch_dtype float16 \
  --sft_preset testing_alpaca_small --epochs 3 --batch_size 2 --max_length 512 --lr 2e-5
```

无 GPU 或勿在登录节点长跑时，请只用第 **0.2** 节的 `bsub` 流程。
