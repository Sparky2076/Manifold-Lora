# utils_sft.py — 小体量指令数据集 + SFT 标签构造（独立于根目录 utils.py 的 GLUE 分类）
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, interleave_datasets, load_dataset


@dataclass
class SFTDataConfig:
    """Hub 小指令集预设见 deepseek/README.md 与 docs/DEEPSEEK_FINETUNE_PLAN.md"""

    dataset_id: str = "HuggingFaceH4/testing_alpaca_small"
    dataset_config: Optional[str] = None
    split_train: str = "train"
    val_ratio: float = 0.1
    max_samples: Optional[int] = None
    max_length: int = 512
    num_workers: int = 0
    seed: int = 42


SFT_DATASET_PRESETS: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    "testing_alpaca_small": ("HuggingFaceH4/testing_alpaca_small", None, None),
    "alpaca_gpt4_500": ("levulinh/alpaca-gpt4-500", None, None),
    "alpaca_train_500": ("tatsu-lab/alpaca", None, "train[:500]"),
    "alpaca_train_1k": ("tatsu-lab/alpaca", None, "train[:1000]"),
    # 大集混合：ShareGPT 对话 + 指令覆盖 + 中文补充（按 build_sft_dataloaders 内的 mix 逻辑构建）
    "mix_chat_real_300k": ("__mix_chat_real_300k__", None, None),
}


def _alpaca_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or example.get("response") or "").strip()
    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    return prompt, out


def _dolly_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    inst = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    resp = (example.get("response") or "").strip()
    if ctx:
        prompt = f"### Instruction:\n{inst}\n\n### Context:\n{ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    return prompt, resp


def _sharegpt_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    ShareGPT 风格：example["conversations"] = [{"from": "system|human|gpt", "value": "..."}...]
    这里取“最后一个 gpt 回复”为监督目标，prompt 包含其之前的全部上下文。
    """

    conv = example.get("conversations") or []
    if not isinstance(conv, list) or len(conv) == 0:
        return "### Response:\n", ""

    turns: List[Tuple[str, str]] = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        role = (t.get("from") or t.get("role") or "").strip().lower()
        val = (t.get("value") or t.get("content") or "").strip()
        if not val and role != "system":
            continue
        if role in ("system",):
            turns.append(("system", val))
        elif role in ("human", "user"):
            turns.append(("user", val))
        elif role in ("gpt", "assistant"):
            turns.append(("assistant", val))

    # 找最后一个 assistant turn
    last_assistant_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx == -1:
        return "### Response:\n", ""

    response = turns[last_assistant_idx][1]
    ctx = turns[:last_assistant_idx]
    lines: List[str] = []
    for role, text in ctx:
        if role == "system":
            if text:
                lines.append(f"### System:\n{text}\n")
        elif role == "user":
            lines.append(f"### User:\n{text}\n")
        else:
            lines.append(f"### Assistant:\n{text}\n")

    lines.append("### Assistant:\n")
    prompt = "\n".join(lines)
    return prompt, response


def _detect_format(ds_sample: Dict[str, Any]) -> str:
    keys = set(ds_sample.keys())
    if "conversations" in keys and isinstance(ds_sample.get("conversations"), list):
        return "sharegpt"
    if "instruction" in keys and ("output" in keys or "response" in keys):
        return "alpaca_like"
    if "instruction" in keys and "context" in keys and "response" in keys:
        return "dolly"
    if "text" in keys and len(keys) <= 4:
        return "text_only"
    return "alpaca_like"


def _encode_sft(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Dict[str, List[int]]:
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        r_ids = r_ids + [tokenizer.eos_token_id]

    # 防止标签全为 -100：至少保留 1 个 response token 参与监督
    if not r_ids:
        fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if fallback_id is None:
            fallback_id = 0
        r_ids = [fallback_id]
    # CausalLM 内部会做 shift，至少需要 2 个监督 token 才能产生有效 loss
    if len(r_ids) == 1:
        r_ids = r_ids + [r_ids[0]]

    if len(p_ids) + len(r_ids) > max_length:
        keep_resp = min(len(r_ids), max(1, max_length))
        keep_prompt = max_length - keep_resp
        p_ids = p_ids[: max(0, keep_prompt)]
        r_ids = r_ids[:keep_resp]

    input_ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids

    # 二次保险：若仍超长则按 max_length 截断，但优先保留末尾监督 token
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {"input_ids": input_ids, "labels": labels}


def build_sft_dataloaders(
    tokenizer,
    cfg: SFTDataConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    def _try_load_first_available(
        candidates: List[Tuple[str, Optional[str], str]],
        tag: str,
    ):
        last_err = None
        for did, dcfg, dsplit in candidates:
            if not did:
                continue
            try:
                if dcfg:
                    ds_local = load_dataset(did, dcfg, split=dsplit, trust_remote_code=True)
                else:
                    ds_local = load_dataset(did, split=dsplit, trust_remote_code=True)
                print(f"[SFT][mix] {tag} source: {did} split={dsplit}")
                return ds_local
            except Exception as e:  # pragma: no cover - runtime fallback path
                last_err = e
                print(f"[Warn][SFT][mix] failed loading {tag} source={did}: {type(e).__name__}: {e}")
                continue
        raise RuntimeError(f"No available dataset source for {tag}. last_error={last_err}")

    # 内置 mix preset：拼一个 300k 规模的“聊天+指令+中文”混合数据
    if cfg.dataset_id == "__mix_chat_real_300k__":
        # 比例：OpenHermes 50%（指令覆盖）+ Nectar-ShareGPT 30%（更像真实 chat）+ COIG 20%（中文补充）
        # 支持环境变量覆盖主源；并内置候选回退，避免单一 HF 仓库不可达时直接失败。
        hermes_primary = os.environ.get("SFT_MIX_HERMES_DATASET", "teknium/OpenHermes-2.5")
        nectar_primary = os.environ.get("SFT_MIX_NECTAR_DATASET", "PhilipMay/Nectar-ShareGPT-clean")
        coig_primary = os.environ.get("SFT_MIX_COIG_DATASET", "BAAI/COIG")

        hermes = _try_load_first_available(
            [
                (hermes_primary, None, "train"),
                ("Open-Orca/SlimOrca", None, "train"),
            ],
            "hermes_like",
        )
        nectar = _try_load_first_available(
            [
                (nectar_primary, None, "train"),
                ("berkeley-nest/Nectar", None, "train"),
            ],
            "nectar_like",
        )
        coig = _try_load_first_available(
            [
                (coig_primary, None, "train"),
                ("BelleGroup/train_1M_CN", None, "train"),
            ],
            "zh_like",
        )

        n_total = 300_000
        n_hermes = int(n_total * 0.50)
        n_nectar = int(n_total * 0.30)
        n_coig = n_total - n_hermes - n_nectar

        hermes = hermes.shuffle(seed=cfg.seed).select(range(min(n_hermes, len(hermes))))
        nectar = nectar.shuffle(seed=cfg.seed).select(range(min(n_nectar, len(nectar))))
        coig = coig.shuffle(seed=cfg.seed).select(range(min(n_coig, len(coig))))

        # interleave 让训练 batch 更像真实混合分布
        ds = interleave_datasets(
            [hermes, nectar, coig],
            probabilities=[0.50, 0.30, 0.20],
            seed=cfg.seed,
            stopping_strategy="all_exhausted",
        )

        # 保证恰好 n_total（若某个子集不足，会少于 n_total；这里截断到可用长度）
        if len(ds) > n_total:
            ds = ds.select(range(n_total))
    else:
        split = cfg.split_train
        if cfg.dataset_config:
            ds = load_dataset(
                cfg.dataset_id,
                cfg.dataset_config,
                split=split,
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                cfg.dataset_id,
                split=split,
                trust_remote_code=True,
            )

    if cfg.max_samples is not None and cfg.max_samples > 0:
        n = min(cfg.max_samples, len(ds))
        ds = ds.select(range(n))

    sample = ds[0]
    fmt = _detect_format(sample)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        if fmt == "sharegpt":
            prompt, response = _sharegpt_prompt_response(example)
        elif fmt == "dolly":
            prompt, response = _dolly_prompt_response(example)
        elif fmt == "text_only" and example.get("text"):
            full = example["text"].strip()
            mid = len(full) // 2
            prompt, response = full[:mid], full[mid:]
        else:
            prompt, response = _alpaca_prompt_response(example)
        enc = _encode_sft(tokenizer, prompt, response, cfg.max_length)
        return enc

    ds = ds.map(preprocess, remove_columns=ds.column_names, desc="SFT tokenize")

    n = len(ds)
    if n <= 1:
        train_ds, eval_ds = ds, ds
    else:
        test_n = max(1, min(n - 1, ceil(n * cfg.val_ratio)))
        split_ds = ds.train_test_split(test_size=test_n, seed=cfg.seed)
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            ids = x["input_ids"]
            lab = x["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader
