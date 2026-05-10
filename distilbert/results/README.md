# DistilBERT 单次训练输出目录

未设置 `METRICS_DIR` 时，`distilbert/scripts/submit_bsub.sh` 默认将 `train.csv`、`test.csv` 写到这里。

**全因子超参网格**（每个组合一个 LSF 作业）使用 **`distilbert_autogrid/`**，结果在 `distilbert_autogrid/results/<run_name>/`。说明见 [../../distilbert_autogrid/README.md](../../distilbert_autogrid/README.md)。
