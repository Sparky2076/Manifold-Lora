## Manifold-Lora (个人实验 Fork)

基于原始 Manifold-Lora 项目的个人实验仓库，用于在 DeepSeek-1.5B 等模型上做 LoRA / mLoRA 微调，并方便在学校 GPU 集群上提交和监控任务。

本仓库当前主要增加了两块内容：
- **LoRA / mLoRA 在 DeepSeek 上的 dtype 适配**（在 `lora.py` / `mlora.py` 中）
- **训练过程中实时查看 `train.csv` / `test.csv` 的监控脚本 `scripts/watch_metrics.sh`**

---

### 1. 实时查看 train/test 指标：`scripts/watch_metrics.sh`

脚本路径：`scripts/watch_metrics.sh`

功能：
- 每 5 秒刷新一次本目录下 `train.csv` 和 `test.csv` 的最后 5 行
- 便于在 bsub 提交的训练任务运行时，实时观察 loss / accuracy 变化

#### 使用方法（在服务器上）

1. SSH 登录服务器：

```bash
ssh wangxiao@202.121.138.196
```

2. 进入项目目录并运行脚本：

```bash
cd ~/Manifold-Lora
```
```bash
bash scripts/watch_metrics.sh
```

- 终端会每 5 秒刷新一次 `train.csv` / `test.csv` 尾部信息
- 按 **Ctrl+C** 即可退出脚本，不会影响正在运行的 bsub 训练任务

---

### 2. DeepSeek-1.5B 调参示例（bsub 提交）

下面是一个更保守、更稳定的 bsub 提交命令示例，只调整了超参数，**没有改任何 Python 代码逻辑**。

在登录节点 `mgtgpu01` 上执行：

```bash
cd ~/Manifold-Lora

SNAP=$(ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ | head -1)
MODEL_PATH="/nfsshare/home/wangxiao/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/$SNAP"

bsub -J manifold_lora_tune \
  -q gpu \
  -n 1 \
  -R "rusage[mem=32768]" \
  -gpu "num=1" \
  -o "%J.out" \
  -e "%J.err" \
  "source ~/miniconda3/etc/profile.d/conda.sh; conda activate torch; cd ~/Manifold-Lora; \
   python main.py \
     --model_name \"$MODEL_PATH\" \
     --dataset_name glue --dataset_config sst2 --text_field sentence \
     --epochs 1 --batch_size 4 --max_length 128 \
     --grad_accum_steps 8 \
     --lr 1e-5"
```

要点：
- **只调整了参数**：`batch_size=4`、`grad_accum_steps=8`、`lr=1e-5`，其余使用当前代码中的默认逻辑（包括 LoRA 配置）。
- 任务在 `gpu` 队列的 GPU 节点上运行，避免在登录节点 CPU 上用 Half 精度导致的 `"addmm_impl_cpu_" not implemented for 'Half'` 错误。

训练运行后，可以在另一个 SSH 终端中使用 `watch_metrics.sh` 监控 `train.csv` / `test.csv`：

```bash
ssh wangxiao@202.121.138.196
cd ~/Manifold-Lora
bash scripts/watch_metrics.sh
```

