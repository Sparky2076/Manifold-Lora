## Manifold-Lora (个人实验 Fork)

本仓库以 **DistilBERT 文本分类 + LoRA 全因子网格**（`distilbert/`、`distilbert_autogrid/`）为主，在学校 GPU 集群（SSH + `bsub`）上提交与汇总。训练**复用**根目录 `lora.py`、`mlora.py`、`optimizers.py`。

| 流水线 | 入口 | 提交 | 指标（默认） |
|--------|------|------|----------------|
| **DistilBERT 分类** | `python -m distilbert.main` | 单次：`distilbert/scripts/submit_bsub.sh`；网格：`distilbert_autogrid/run_grid_bsub.sh` | `distilbert/results/` 或 `distilbert_autogrid/results/<run_name>/` |

**`deepseek/`** 已清空占位，便于后续新 DeepSeek 网格（见 [`deepseek/README.md`](deepseek/README.md)）。

---

### 文档与操作入口

| 说明 | 链接 |
|------|------|
| **上传服务器、`bsub`、tmux（含 `tmux attach -t grid`）、`CONDA_ROOT` + `GRID_MAX_RUN` / `GRID_MAX_PEND`、续跑/强交、`Ctrl+C` / `bkill`** | **[distilbert_autogrid/README.md](distilbert_autogrid/README.md)**（集群操作以该文档为准） |
| DistilBERT 目录与单次训练 | [distilbert/README.md](distilbert/README.md) |
| DeepSeek SFT 与全因子网格（LoRA/mLoRA） | [deepseek_autogrid/README.md](deepseek_autogrid/README.md) |
| Git 提交/推送辅助 | [scripts/commit_and_push.sh](scripts/commit_and_push.sh) |
| DeepSeek 规划（实现待补） | [docs/DEEPSEEK_FINETUNE_PLAN.md](docs/DEEPSEEK_FINETUNE_PLAN.md) |

---

### 本机试跑 DistilBERT（非 bsub）

```bash
python -m distilbert.main --model_name distilbert-base-uncased ^
  --dataset_name glue --dataset_config sst2 --epochs 1 --batch_size 4 --metrics_dir distilbert/results
```

详见 [distilbert/README.md](distilbert/README.md)。
