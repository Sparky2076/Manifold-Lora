# DeepSeek 指令微调（SFT）— 独立目录说明

本目录与 **`distilbert/`（DistilBERT 分类）** 完全分离：

- **代码**：`deepseek/main_sft.py`、`models_sft.py`、`utils_sft.py`（训练仍**复用**仓库根的 `lora.py` / `mlora.py` / `optimizers.py`）
- **脚本**：`deepseek/scripts/*.sh`
- **结果**：`deepseek/results/`（结构与 `distilbert/results/` 对应，见下表）

运行训练时**工作目录必须是仓库根** `Manifold-Lora/`，内部通过 `python -m deepseek.main_sft` 调用。

---

## 与 `distilbert/results/` 结构对应

| 本目录 | DistilBERT 分类（`distilbert/`） |
|--------|----------------------------------|
| `deepseek/results/tuning_logs/` | `distilbert/results/tuning_logs/` |
| `deepseek/results/final_sft/` | `distilbert/results/final_loRA/`、`distilbert/results/final_mLoRA/` |
| `deepseek/results/sft_grid/` | 网格实验按子目录保存（类似 `distilbert/results/` 下按实验分子文件夹的做法） |

---

## 0. 上传、提交、下载（与根 README §0 形式一致）

### ① 本机上传到服务器（Git Bash）

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

`upload.sh` / `upload.ps1` 会递归上传 **`distilbert/`** 与 **`deepseek/`** 整包，以及根目录 `lora.py` / `mlora.py` / `optimizers.py`。

### ② 服务器：修正换行并提交

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh deepseek/scripts/*.sh
bash deepseek/scripts/submit_bsub_sft.sh
```

- 默认 **`METRICS_DIR=$PWD/deepseek/results`**，指标为 `deepseek/results/train_sft.csv`、`test_sft.csv`。
- 查看任务：`bjobs`；日志：`cat JOBID.out` / `cat JOBID.err`。

### ③ 本机拉回 CSV（PowerShell 示例）

单次任务（默认写入 `deepseek/results/`）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/train_sft.csv deepseek/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/test_sft.csv deepseek/results/
```

网格某一格：

```powershell
scp wangxiao@202.121.138.196:~/Manifold-Lora/deepseek/results/sft_grid/testing_alpaca_small_lr_2e_5/train_sft.csv deepseek/results/sft_grid/
```

（IP 按学校服务器实际修改。）

---

## 1. 实时监控指标

```bash
cd ~/Manifold-Lora
bash deepseek/scripts/watch_metrics_sft.sh
```

子目录：

```bash
METRICS_DIR=deepseek/results/sft_grid/testing_alpaca_small_lr_2e_5 bash deepseek/scripts/watch_metrics_sft.sh
```

---

## 2. 默认配置说明（与根 README §2 形式对齐）

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（建议服务器改为本地 `snapshots/...` 路径）
- 数据：脚本默认 `SFT_PRESET=alpaca_train_1k`（`tatsu-lab/alpaca` 前 1000 条）；验证集比例默认 `SFT_VAL_RATIO=0.2`（可用环境变量覆盖）
- 超参：`run_deepseek_sft_bsub.sh` 默认 `EPOCHS=20`；其余见 `main_sft.py` 中 `argparse`
- 指标列：
  - `train_sft.csv`：`iteration,train_loss,train_perplexity`
  - `test_sft.csv`：`iteration,eval_loss,eval_perplexity`

---

## 3. 学习率网格（Grid Search）

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' deepseek/scripts/*.sh
bash deepseek/scripts/gs_lr_deepseek_sft.sh
```

每个 `lr` 一个 job，输出目录：`deepseek/results/sft_grid/<preset>_lr_<安全化lr>/`。

第二轮（v2）建议：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' deepseek/scripts/*.sh

# LoRA v2（围绕 2e-5 细化）
bash deepseek/scripts/gs_lr_deepseek_sft_v2.sh

# mLoRA v2（向更高学习率继续探索）
bash deepseek/scripts/gs_lr_deepseek_sft_mlora_v2.sh
```

v2 输出目录统一写到 `deepseek/results/sft_grid_v2/`，便于与第一轮分开对比。

---

## 3.1 多卡数据并行（DDP）

本仓库 SFT 使用 **PyTorch DDP**（`torchrun`，每进程绑定一卡）。**张量并行**（Megatron / DeepSpeed TP 切分单层）未实现，若需要需另行集成。

- 环境变量：`NPROC_PER_NODE`（与 `NGPU`、`torchrun` 一致；`submit_bsub_sft.sh` 默认 `NPROC_PER_NODE=$NGPU`）
- LSF：`bsub -n` 默认 **等于 `NGPU`**（四卡即 `-n 4`）。若集群要求更多 CPU 槽，可设 **`NCORES`**（例如 `NGPU=4 NCORES=8`）
- `run_deepseek_sft_bsub.sh`：当 `NPROC_PER_NODE>1` 时自动 `torchrun --standalone --nproc_per_node=...`，`main_sft.py` 内会 `init_process_group`、使用 `DistributedSampler`，梯度累积步配合 `no_sync`。
- 大集网格 `gs_lr_deepseek_sft_big_v1.sh` / `gs_lr_deepseek_sft_mlora_big_v1.sh` 默认 **`NGPU=2`**（可按集群改成 4、8 等）。

单卡行为不变：`NPROC_PER_NODE=1` 时仍用 `python -m deepseek.main_sft`，可用 `device_map=auto`。

---

## 4. 本机试跑（非 bsub）

在仓库根目录执行：

```bash
cd D:\GitHub_Code\Manifold-Lora
python -m deepseek.main_sft --trust_remote_code --device_map auto --torch_dtype float32 ^
  --sft_preset alpaca_train_1k --sft_val_ratio 0.2 --epochs 20 --batch_size 2 --metrics_dir deepseek/results
```

更多背景：**[../docs/DEEPSEEK_FINETUNE_PLAN.md](../docs/DEEPSEEK_FINETUNE_PLAN.md)**
