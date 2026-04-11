# DistilBERT 文本分类 + LoRA / mLoRA

与 **`deepseek/`** 独立；训练**复用**仓库根目录的 `lora.py`、`mlora.py`、`optimizers.py`。

## 目录

| 路径 | 作用 |
|------|------|
| `main.py`、`models.py`、`utils.py` | 训练入口与数据/模型 |
| `scripts/run_train_bsub.sh` | 计算节点执行（由 `submit_bsub.sh` 调度） |
| `scripts/submit_bsub.sh` | 提交**单个** LSF 作业（`bsub`） |
| `scripts/watch_metrics.sh` | 轮询 `METRICS_DIR` 下 `train.csv` / `test.csv` 尾部 |
| `results/` | 单次默认输出（见该目录下 `README.md`） |

工作目录必须是仓库根：`python -m distilbert.main`。

## 全因子网格（推荐）

**每个超参组合一个 `bsub` 作业**，网格定义只在 `distilbert_autogrid/config.py`：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
bash distilbert_autogrid/run_grid_bsub.sh
```

`tmux`、断点续交（`GRID_RESUME`）与 **`nohup`** 说明见 [distilbert_autogrid/README.md](../distilbert_autogrid/README.md)。

本地顺序跑（不占队列）：`python -m distilbert_autogrid.run_grid`（同上文档）。

## 单次训练（对比 / 烟测）

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
bash distilbert/scripts/submit_bsub.sh
```

默认 `METRICS_DIR=$PWD/distilbert/results`。可用环境变量覆盖：`EPOCHS`、`LR`、`LORA_R`、`LORA_ALPHA`、`WEIGHT_DECAY`、`METRICS_DIR` 等（见 `submit_bsub.sh`）。

## 本机试跑（非 bsub）

```bash
python -m distilbert.main --model_name distilbert-base-uncased \
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 \
  --metrics_dir distilbert/results
```

上传服务器、拉回 CSV 等流程与仓库根 [README.md](../README.md) 一致；DistilBERT 网格请同步上传 **`distilbert_autogrid/`**（见 `scripts/upload.sh`）。
