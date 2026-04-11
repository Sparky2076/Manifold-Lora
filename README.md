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

**② 服务器：换行 + 提交**

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
```

全因子网格（每组合一个 `bsub`；建议 `tmux`、续跑见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md)）：

```bash
export CONDA_ROOT="$HOME/miniconda3"   # 按你机器修改
bash distilbert_autogrid/run_grid_bsub.sh
```

单次训练：`bash distilbert/scripts/submit_bsub.sh`。

跑完后汇总：`python -m distilbert_autogrid.aggregate_results`。

**③ 本机拉回 CSV**（IP 换成你的服务器）：

```powershell
cd D:\GitHub_Code\Manifold-Lora
scp wangxiao@202.121.138.196:~/Manifold-Lora/distilbert/results/train.csv distilbert/results/
scp -r "wangxiao@202.121.138.196:~/Manifold-Lora/distilbert_autogrid/results/<run_name>" distilbert_autogrid/results/
```

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
| `distilbert_autogrid/run_grid_bsub.sh` | LSF：全因子网格，每组合一作业；`GRID_RESUME`、`tmux` 见 [distilbert_autogrid/README.md](distilbert_autogrid/README.md) |
| `distilbert_autogrid/run_grid.py` | 本地顺序跑网格 |
| `distilbert_autogrid/aggregate_results.py` | 生成 `summary.csv` |
| `distilbert/scripts/submit_bsub.sh` | 单次分类作业 |
| `scripts/upload.sh`、`upload.ps1` | **仅**上传 DistilBERT 网格相关文件（见上文） |
| `scripts/commit_and_push.sh` | 交互式提交推送 GitHub |

服务器上提交前 **`sed`** 与 **§0** 一致；查看任务：`bjobs`，日志：`JOBID.out` / `JOBID.err`。

---

### 5. 本机试跑 DistilBERT（非 bsub）

```bash
python -m distilbert.main --model_name distilbert-base-uncased ^
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 --metrics_dir distilbert/results
```

详见 [distilbert/README.md](distilbert/README.md)。
