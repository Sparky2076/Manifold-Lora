# DeepSeek 微调方案：任务与数据集选择

当前仓库的 `main.py` 是针对**文本分类**（如 GLUE SST-2）设计的：使用 `AutoModelForSequenceClassification`、交叉熵损失、准确率指标。这类设定适合 BERT/DistilBERT，**不适合**直接用于 DeepSeek 这类**生成式因果语言模型**。下面说明原因、常见做法，并给出在 Hugging Face 上可用的数据集与实现方案。

---

## 1. 为什么「文本分类」不适合 DeepSeek？

| 维度 | 当前流水线（分类） | DeepSeek 常见用法 |
|------|---------------------|-------------------|
| 任务形态 | 单句 → 离散标签（如 0/1） | 指令/问题 → 一段**生成**的回复 |
| 模型头 | `ForSequenceClassification`（分类头） | `ForCausalLM`（下一个 token 预测） |
| 损失 | 交叉熵 on logits vs label | 语言模型损失（只对「回答」部分算 loss） |
| 数据格式 | `sentence` + `label` | `instruction` +（可选）`input` + `output` |
| 评估 | accuracy / F1 | 生成质量、BLEU/ROUGE、或人工/模型打分 |

DeepSeek 是 decoder-only 的因果 LM，没有为「多分类」设计的 head；强行用分类任务会浪费其生成能力，且你之前遇到的 padding token、device_map 等问题也多半来自「把生成模型当分类用」的错配。因此更合理的做法是：**用指令跟随 / 对话 / 问答类任务 + 因果 LM 的 SFT**。

---

## 2. 课题组 / 业界一般用什么任务和数据集微调 DeepSeek？

常见有几类：

- **指令跟随（Instruction Following）**：给一条指令（+ 可选输入），模型生成一段回答。数据格式多为 `instruction / input / output` 或多轮对话。
- **对话 / 客服（Chat）**：多轮 user/assistant，ShareGPT、OpenAssistant 等。
- **推理 / CoT（Chain-of-Thought）**：带推理步骤的问答，适合 DeepSeek-R1 这类强调推理的模型。
- **摘要 / 翻译**：长文→短文、外文→中文等。

常见**数据来源**包括：

- **Alpaca**：约 5.2 万条 instruction-input-output，英文为主，格式统一，非常适合做 SFT 入门。
- **ShareGPT / 对话格式**：多轮对话，很多中文+英文混合。
- **Dolly、OpenAssistant、FLAN** 等：指令/问答/多任务。
- **领域数据**：医疗 CoT、代码、数学等，按需选。

### Hugging Face 上课题组微调 DeepSeek 的常见任务与数据集（检索整理）

根据 Hugging Face 博客、已微调模型页和社区用法，课题组/社区微调 DeepSeek 时常见搭配如下：

| 任务类型 | 常见数据集（HF） | 说明 |
|----------|------------------|------|
| **指令跟随** | `tatsu-lab/alpaca`、`databricks-dolly-15k` | 通用 instruction/input/output，SFT 最常用 |
| **推理 / CoT** | `paloalma/Reasoning-DeepSeek-R1-Distilled-1.4M-Alpaca-V2`、自建合成推理数据 | 1.4M 条 Alpaca 格式推理数据，或 Synthetic Data Generator 生成 |
| **代码** | 自建 Python 题+解答、或 CodeAlpaca 等 | 常用「问题 + 带推理步骤的代码解答」做 SFT |
| **数学** | `openai/gsm8k`、MATH 等 | 数学推理，评估多用准确率 |
| **多轮对话** | ShareGPT、OpenAssistant、Belle | 多轮 user/assistant，只对 assistant 回复算 loss |

**已微调模型示例（HF 上可见）**：

- `sweatSmile/DeepSeek-R1-Distill-Qwen-1.5B-Alpaca-Instruct`：用 **Alpaca 指令数据**微调的 1.5B 蒸馏版。
- Hugging Face 官方博客：用 **合成推理数据**（Synthetic Data Generator + DeepSeek 生成 Python 解题）微调 DeepSeek-R1，任务为「代码推理」。
- 不少 DeepSeek-R1-Distill-Llama-8B / V3 的 finetune 模型：任务多为 **指令跟随、对话、数学/代码**。

**结论**：做方案 A 时，**任务**优先选「指令跟随」或「推理/代码」；**数据集**优先用 `tatsu-lab/alpaca` 或 `databricks-dolly-15k` 跑通流程，再按需加推理/代码类数据。

下面只列 **Hugging Face 上可直接用的、和 DeepSeek 微调最相关**的数据集与格式，便于你直接写脚本或对接到现有代码。

---

## 3. Hugging Face 上适合 DeepSeek 的数据集（推荐）

### 3.1 指令跟随（Instruction Following）— 首选

| 数据集 | HF 路径 | 说明 | 规模 |
|--------|---------|------|------|
| **Alpaca** | `tatsu-lab/alpaca` | 经典指令数据，instruction / input / output，英文 | ~52k |
| **Alpaca-GPT4 (ShareGPT 格式)** | `abhinand/alpaca-gpt4-sharegpt` | Alpaca 转成多轮对话格式 | ~52k |
| **Dolly** | `databricks-dolly-15k` | 指令跟随，多类型任务 | 15k |
| **OpenAssistant** | `OpenAssistant/oasst1` | 多轮对话，多语言 | 约 8k+ |

**Alpaca 字段示例**（`tatsu-lab/alpaca`）：

- `instruction`：任务描述  
- `input`：可选上下文（可为空）  
- `output`：期望模型生成的回答  
- 不少版本还有 `text`：已拼好的 `"### Instruction: ... ### Input: ... ### Response: ..."` 一整段，方便直接做 SFT。

训练方式：把「Instruction + Input」当作 prompt，只对「Response」部分的 token 算 LM loss（prompt 部分 mask 掉），即标准 SFT。

#### 小规模指令数据（调试用、省算力）

若不想一上来用 52k / 15k 全量，可选用下面**体量更小**的 Hub 数据集，或对大集做切片：

| 数据集 | HF 路径 | 约规模 | 说明 |
|--------|---------|--------|------|
| **testing_alpaca_small** | `HuggingFaceH4/testing_alpaca_small` | ~200 条 | HF 侧测试用小 Alpaca，适合**跑通 pipeline** |
| **alpaca-gpt4-500** | `levulinh/alpaca-gpt4-500` 或 `chargoddard/alpaca-gpt4-500` | 500 条 | Alpaca/GPT4 风格，instruction/input/output |
| **python-instruct-1k** | `harryng4869/python-instruct-1k` | ~1k 条 | 偏 **Python 编程**指令 |
| **tinyAlpacaEval** | `tinyBenchmarks/tinyAlpacaEval` | ~100 条 | 偏**评测/基准**，也可极小样本试训 |
| **alpaca-small（演示）** | `jamesargent/alpaca-small` | 仅数条 | 仅适合**格式调试**，不适合当真训练 |

**通用做法（任意大集变小）**：仍用 `tatsu-lab/alpaca` 或 `databricks/databricks-dolly-15k`，在代码里只取前 N 条，例如：

```python
from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca", split="train[:500]")  # 只训 500 条
# 或
ds = load_dataset("tatsu-lab/alpaca")["train"].select(range(1000))
```

注意：样本太少时 loss 会降、但**泛化差**，适合验证代码与流程；正式实验仍建议至少数千条或全量。

### 3.2 对话 / 多轮（Chat）

| 数据集 | HF 路径 | 说明 |
|--------|---------|------|
| **ShareGPT** 风格 | 如 `anon8231489123/ShareGPT_Vicuna_unfiltered` | 多轮 user/assistant，中英混合 |
| **Belle** | `BelleGroup/train_0.5M_CN` 等 | 中文指令/对话 |

格式一般是 `conversations`: `[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]`，需要你在数据层转成「只对 assistant 回复算 loss」的序列。

### 3.3 推理 / 数学 / 代码（偏 DeepSeek-R1）

若你用的是 **DeepSeek-R1**（推理型），可以再加：

- **GSM8K**：`openai/gsm8k`，数学推理  
- **MATH**、**CodeAlpaca-20k** 等  

这些通常也是「问题 + 解答」或「instruction + output」，和 Alpaca 用法类似，只是领域不同。

---

## 4. 实现方案（两种思路）

### 方案 A：在本仓库内增加「SFT 分支」（已实现，**不修改** `main.py` / `utils.py` / `models.py`）

本仓库已用**独立文件**实现方案 A，与 DistilBERT 分类流水线并存：

| 路径 | 作用 |
|------|------|
| `deepseek/main_sft.py` | SFT 入口（仓库根执行 `python -m deepseek.main_sft`）；训练循环对齐 `main.py`，指标为 **eval loss / perplexity**。 |
| `deepseek/models_sft.py` | `AutoModelForCausalLM` + Tokenizer。 |
| `deepseek/utils_sft.py` | Hub 小指令集、Alpaca/Dolly 字段、`labels=-100` 掩码 prompt。 |
| `deepseek/results/` | 与根目录 `results/` 结构对应：`tuning_logs/`、`final_sft/`、`sft_grid/`。 |
| `lora.py` / `mlora.py`（根目录） | **复用** LoRA / mLoRA。 |
| `optimizers.py`（根目录） | **复用** AdamW。 |

**数据预设**（`--sft_preset`）：见 `deepseek.utils_sft.SFT_DATASET_PRESETS`。

**本地 / 服务器单卡示例**（工作目录为仓库根）：

```bash
python -m deepseek.main_sft --trust_remote_code --device_map auto --torch_dtype float16 \
  --sft_preset testing_alpaca_small --epochs 3 --batch_size 2 --max_length 512 --lr 2e-5 \
  --metrics_dir deepseek/results
```

**LSF**：`deepseek/scripts/submit_bsub_sft.sh`（`run_deepseek_sft_bsub.sh` 调 `python -m deepseek.main_sft`）。**网格**：`deepseek/scripts/gs_lr_deepseek_sft.sh` → **`deepseek/results/sft_grid/`**。

**SSH 全流程**：根目录 **[README.md](../README.md) §0.2**、**[deepseek/README.md](../deepseek/README.md)**。

**推荐起步**：先用 `testing_alpaca_small` 或 `alpaca_gpt4_500` 跑通；再上全量 `tatsu-lab/alpaca`（`--sft_dataset tatsu-lab/alpaca --sft_split train[:5000]` 等）。

---

### 方案 B：用现成框架跑 DeepSeek SFT（最快跑通）

如果希望**尽快**在标准任务上微调 DeepSeek，可以直接用已经支持「Alpaca / ShareGPT + CausalLM + LoRA」的框架，例如：

- **LLaMA-Factory**：支持 Alpaca/ShareGPT 等格式、LoRA、多卡，文档里有明确的数据准备说明。  
- **TRL**（Hugging Face）：`SFTTrainer` 支持 instruction 格式和 CausalLM。  
- **Unsloth**：针对显存优化，适合单卡/小卡跑 1.5B~7B。

这些框架一般要求数据是 **Alpaca 三列**（instruction/input/output）或 **ShareGPT 对话 JSON**，和上面 3.1、3.2 的表完全对应；你只需选一个 HF 数据集（如 `tatsu-lab/alpaca`），按框架文档转成其要求的 JSON/parquet 即可。

---

## 5. 小结与推荐

- **文本分类（如 SST-2）不适合 DeepSeek**：应使用「指令跟随 / 对话 / 问答」类任务 + 因果 LM 的 SFT。  
- **常用数据集**：  
  - 首选 **`tatsu-lab/alpaca`**（指令跟随，约 52k，英文）；  
  - 备选 **`databricks-dolly-15k`**、**`OpenAssistant/oasst1`**、ShareGPT/Belle 等。  
- **实现上**：  
  - 若希望继续在本仓库玩 LoRA/mlora：按**方案 A** 加 SFT 分支 + Alpaca 数据 + CausalLM。  
  - 若优先「快速在标准设定下微调 DeepSeek」：用 **方案 B**（LLaMA-Factory / TRL / Unsloth）+ 同一批 Hugging Face 数据集。

方案 A 的代码入口为 **`deepseek/main_sft.py`**（`python -m deepseek.main_sft`）；分类仍只用根目录 **`main.py`**。
