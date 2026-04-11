# DistilBERT LoRA 全因子网格（`lr × r × alpha × weight_decay`）

优化器：**只扫** `weight_decay ∈ {0, 0.01, 0.1}`；`adam_beta1` / `adam_beta2` **固定**为 `config.py` 中的 `ADAM_BETA1_FIXED` / `ADAM_BETA2_FIXED`（默认 0.9 / 0.999）。

## 配置

[`config.py`](config.py) 定义 `LR_LIST`、`R_LIST`、`ALPHA_LIST`、`WEIGHT_DECAY_LIST`、`EPOCHS_DEFAULT` 与 `RESULTS_ROOT`（默认 `distilbert_autogrid/results/`）。缩小或扩大网格**只改此文件**。

当前全量约为 **375** 组（5×5×5×3）；极高学习率（如 `3e-3`）可能不稳定。

## 本地顺序跑满网格

在**仓库根**执行：

```bash
python -m distilbert_autogrid.run_grid
```

- 干跑（只打印命令）：`python -m distilbert_autogrid.run_grid --dry-run`
- 烟测 1 次：`python -m distilbert_autogrid.run_grid --max-attempts 1`
- 直到 1 次成功：`python -m distilbert_autogrid.run_grid --max-runs 1`

每个组合目录含 `run_meta.json`、`train.csv`、`test.csv`。

## LSF：每个组合一个作业（`bsub`）

在**仓库根**：

```bash
cd ~/Manifold-Lora
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh
bash distilbert_autogrid/run_grid_bsub.sh
```

脚本对 `iter_grid()` 中每一组调用一次 `distilbert/scripts/submit_bsub.sh`，各作业 `METRICS_DIR` 指向 `distilbert_autogrid/results/<run_name>/`。可选环境变量（覆盖默认）：

| 变量 | 含义 |
|------|------|
| `RESULTS_ROOT` | 结果根目录（默认 `distilbert_autogrid/results`） |
| `EPOCHS` | 训练轮数（默认见 `config.py`） |
| `QUEUE` | LSF 队列名（默认 `gpu`，在 `submit_bsub.sh`） |
| `LORA_TYPE` | `default` 或 `mlora`（默认 `default`） |
| `LORA_DROPOUT` | LoRA dropout（默认 `0.05`） |
| `BATCH_SIZE` | batch（默认 `4`） |
| `GRAD_ACCUM_STEPS` | 梯度累积（默认 `8`） |

## 汇总

网格跑完后，在仓库根：

```bash
python -m distilbert_autogrid.aggregate_results
```

生成 `distilbert_autogrid/results/summary.csv`（按 `best_val_acc` 降序；指标来自各子目录 `test.csv`）。

## 上传到集群

`scripts/upload.sh` / `upload.ps1` 会同步 **`distilbert/`**、**`distilbert_autogrid/`** 与 **`deepseek/`** 及根目录共享模块；提交网格前请整包上传。

## 提交到 GitHub

见仓库根 [`scripts/commit_and_push.sh`](../scripts/commit_and_push.sh)：在 Git Bash 中执行，按提示检查 `git status`、填写提交说明并 `git push`。请勿将大规模 `results/` 下的 CSV 误提交（`distilbert_autogrid/results/.gitignore` 已忽略运行产物）。
