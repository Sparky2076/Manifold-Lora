# DistilBERT 文本分类 + LoRA / mLoRA — 独立目录说明

本目录与 **`deepseek/`** 形式对齐，与 DeepSeek SFT **完全分离**：

- **代码**：`distilbert/main.py`、`models.py`、`utils.py`（训练**复用**仓库根的 `lora.py` / `mlora.py` / `optimizers.py`）
- **脚本**：`distilbert/scripts/*.sh`
- **结果**：`distilbert/results/`（`train.csv`、`test.csv`、实验子目录、`tuning_logs/`、`final_loRA/`、`final_mLoRA/` 等）

运行训练时**工作目录必须是仓库根** `Manifold-Lora/`，通过 **`python -m distilbert.main`** 调用。

---

## 与 `deepseek/results/` 结构对应（概念上）

| 本目录 | DeepSeek SFT |
|--------|----------------|
| `distilbert/results/tuning_logs/` | `deepseek/results/tuning_logs/` |
| `distilbert/results/final_loRA/`、`final_mLoRA/` | `deepseek/results/final_sft/` |
| 网格实验子目录 | `deepseek/results/sft_grid/` |

---

## 0. 上传、提交、下载（与根 README §0 形式一致）

### ① 本机上传到服务器（Git Bash）

```bash
cd /d/GitHub_Code/Manifold-Lora
bash scripts/upload.sh
```

`upload.sh` / `upload.ps1` 会递归上传整个 **`distilbert/`** 目录及根目录共享的 `lora.py` / `mlora.py` / `optimizers.py` 等。

### ② 服务器：修正换行并提交

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh deepseek/scripts/*.sh
bash distilbert/scripts/submit_bsub.sh
```

- 默认 **`METRICS_DIR=$PWD/distilbert/results`**，指标为 `distilbert/results/train.csv`、`test.csv`。
- 查看任务：`bjobs`；日志：`cat JOBID.out` / `cat JOBID.err`。

### ③ 本机拉回 CSV（PowerShell 示例）

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/test.csv distilbert/results/
```

（IP 按学校服务器实际修改。）

---

## 1. 实时监控指标

```bash
cd ~/Manifold-Lora
bash distilbert/scripts/watch_metrics.sh
```

子目录：

```bash
METRICS_DIR=distilbert/results/某实验子目录 bash distilbert/scripts/watch_metrics.sh
```

---

## 2. 默认配置说明（与根 README §2 形式对齐）

- 模型：`distilbert-base-uncased`
- 数据：GLUE `sst2`，字段 `sentence`
- 超参：见 `distilbert/scripts/run_train_bsub.sh` 与 `main.py` 中 `argparse` 默认值
- 指标列：
  - `train.csv`：`iteration,train_loss,train_accuracy`
  - `test.csv`：`iteration,test_loss,test_accuracy`

---

## 3. 学习率网格（Grid Search）

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' distilbert/scripts/*.sh
bash distilbert/scripts/gs_lr_lora.sh
# 或
bash distilbert/scripts/gs_lr_mlora.sh
```

---

## 4. 本机试跑（非 bsub）

在仓库根目录执行：

```bash
cd D:\GitHub_Code\Manifold-Lora
python -m distilbert.main --model_name distilbert-base-uncased ^
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 --metrics_dir distilbert/results
```
