# DistilBERT 文本分类 + LoRA / mLoRA

训练**复用**仓库根目录的 `lora.py`、`mlora.py`、`optimizers.py`（与将重建的 **`deepseek/`** 占位目录独立）。

## 目录

| 路径 | 作用 |
|------|------|
| `main.py`、`models.py`、`utils.py` | 训练入口与数据/模型 |
| `scripts/run_train_bsub.sh` | 计算节点执行（由 `submit_bsub.sh` 调度） |
| `scripts/submit_bsub.sh` | 提交**单个** LSF 作业（`bsub`） |
| `scripts/watch_metrics.sh` | 轮询 `METRICS_DIR` 下 `train.csv` / `test.csv` 尾部 |
| `../scripts/kill_distilbert_grid_bjobs.sh` | 按作业名前缀批量 `bkill` 网格作业（与网格 README 一致） |
| `results/` | 单次默认输出（见该目录下 `README.md`） |

工作目录必须是仓库根：`python -m distilbert.main`。

### 近期行为（与集群排障相关）

- **`main.py`**：在加载大模型前会对 CUDA 做短探测；若报 `busy/unavailable`，可按环境变量重试（默认多次、间隔约 20s）：`TRAIN_CUDA_RETRY`、`TRAIN_CUDA_RETRY_SEC`。
- **`scripts/run_train_bsub.sh`**：作业开头会打印 `CUDA_VISIBLE_DEVICES` 与 `nvidia-smi -L`，便于对照 `.err`。
- **全因子网格**：`run_grid_bsub.sh` 默认两次 `bsub` 之间 **`sleep` 15 分钟**（`SUBMIT_SLEEP_SEC=900`，减轻同节点 GPU 扎堆）；要改快交可临时设 `SUBMIT_SLEEP_SEC=30` 等。

## 全因子网格（推荐）

**每个超参组合一个 `bsub` 作业**，网格定义只在 `distilbert_autogrid/config.py`：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
bash distilbert_autogrid/run_grid_bsub.sh
```

`tmux`、断点续交（`GRID_RESUME`）与 **`nohup`** 说明见 [distilbert_autogrid/README.md](../distilbert_autogrid/README.md)。

本地顺序跑（不占队列）：`python -m distilbert_autogrid.run_grid`（同上文档）。

## 单次训练（对比 / 烟测）

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
bash distilbert/scripts/submit_bsub.sh
```

默认 `METRICS_DIR=$PWD/distilbert/results`。可用环境变量覆盖：`EPOCHS`、`LR`、`LORA_R`、`LORA_ALPHA`、`WEIGHT_DECAY`、`METRICS_DIR` 等（见 `submit_bsub.sh`）。

## 本机试跑（非 bsub）

```bash
python -m distilbert.main --model_name distilbert-base-uncased \
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 \
  --metrics_dir distilbert/results
```

上传服务器、拉回 CSV 见 [distilbert_autogrid/README.md](../distilbert_autogrid/README.md)；**`scripts/upload.sh` 仅上传** DistilBERT 与 `distilbert_autogrid/`（及根目录共享模块），并包含 **`scripts/kill_distilbert_grid_bjobs.sh`**。
