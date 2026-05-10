"""BBH via lm-eval: either use existing ``model_merged_hf`` or build it from ``sft_lora_state.pt`` + ``run_meta.json``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from deepseek.merge_lora_export import merge_and_export_hf, merged_hf_ready, resolve_lora_state_path


def _bbh_scores_from_lm_eval(results: dict) -> tuple[float, dict[str, float]]:
    """Return (mean_acc, per_subtask_acc) from lm_eval ``results`` dict."""
    rmap = results.get("results") or {}
    sub_tasks = {
        k: v
        for k, v in rmap.items()
        if k.startswith("bbh_") and isinstance(v, dict) and "acc,none" in v
    }
    accs: dict[str, float] = {}
    if sub_tasks:
        for k, v in sub_tasks.items():
            try:
                accs[k] = float(v["acc,none"])
            except (TypeError, ValueError):
                continue
    if accs:
        return sum(accs.values()) / len(accs), accs
    v = rmap.get("bbh")
    if isinstance(v, dict) and "acc,none" in v:
        try:
            x = float(v["acc,none"])
            return x, {"bbh": x}
        except (TypeError, ValueError):
            pass
    return float("nan"), {}


def main() -> int:
    p = argparse.ArgumentParser(description="BBH eval (lm-eval): merged HF or merge from SFT run dir.")
    p.add_argument("--metrics_dir", type=Path, required=True, help="SFT run directory (for output paths + meta)")
    p.add_argument(
        "--merged_hf_dir",
        type=Path,
        default=None,
        help="Use this merged HF dir for lm-eval only (default: <metrics_dir>/model_merged_hf if complete).",
    )
    p.add_argument(
        "--force_remerge",
        action="store_true",
        help="Rebuild merged HF from sft_lora_state.pt even if model_merged_hf already exists.",
    )
    p.add_argument("--output_json", type=Path, default=None, help="Default: <metrics_dir>/bbh_eval.json")
    p.add_argument("--model_name", type=str, default=None, help="Override base HF id when merging")
    p.add_argument("--tasks", type=str, default="bbh", help="Comma-separated lm-eval task names")
    p.add_argument("--num_fewshot", type=int, default=3)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0, help="If >0, passed to lm_eval as doc limit per task (smoke tests).")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--dry_merge_only", action="store_true", help="Only merge+save HF weights; skip lm_eval.")
    args = p.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    meta_path = metrics_dir / "run_meta.json"
    if not meta_path.is_file():
        print(f"[eval_bbh] missing {meta_path}", file=sys.stderr)
        return 2

    merged_dir = (args.merged_hf_dir or (metrics_dir / "model_merged_hf")).resolve()
    need_merge = args.force_remerge or not merged_hf_ready(merged_dir)

    if need_merge:
        if resolve_lora_state_path(metrics_dir) is None:
            print(
                f"[eval_bbh] missing LoRA snapshot (expected {metrics_dir}/sft_lora_state.pt "
                "or legacy lora_adapter.pt); cannot merge.",
                file=sys.stderr,
            )
            return 2
        try:
            merged_dir = merge_and_export_hf(
                metrics_dir,
                merged_dir,
                model_name=args.model_name,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=args.torch_dtype,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[eval_bbh] merge failed: {e}", file=sys.stderr)
            return 2

    if not merged_hf_ready(merged_dir):
        print(f"[eval_bbh] merged HF not ready at {merged_dir}", file=sys.stderr)
        return 2

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = args.model_name or meta.get("model_name") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    out_json = (args.output_json or (metrics_dir / "bbh_eval.json")).resolve()
    if args.dry_merge_only:
        payload = {"bbh_mean_acc": None, "note": "dry_merge_only", "merged_hf_dir": str(merged_dir)}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval_bbh] wrote {out_json}")
        return 0

    try:
        import lm_eval
    except ImportError:
        print("[eval_bbh] pip install lm-eval lm_eval[hf] (and torch/transformers) on the eval node.", file=sys.stderr)
        return 2

    simple_evaluate = getattr(lm_eval, "simple_evaluate", None)
    if simple_evaluate is None:
        from lm_eval import evaluator as _ev

        simple_evaluate = _ev.simple_evaluate

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    merged_s = merged_dir.as_posix()
    model_args = f"pretrained={merged_s},trust_remote_code={str(bool(args.trust_remote_code)).lower()}"
    kwargs = dict(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    if args.limit and args.limit > 0:
        kwargs["limit"] = args.limit
    if torch.cuda.is_available():
        kwargs["device"] = "cuda"

    print(f"[eval_bbh] lm_eval tasks={task_list} model_args={model_args}", flush=True)
    results = simple_evaluate(**kwargs)
    mean_acc, task_acc = _bbh_scores_from_lm_eval(results)

    payload = {
        "metrics_dir": str(metrics_dir),
        "model_name": model_name,
        "merged_hf_dir": str(merged_dir),
        "tasks": task_list,
        "bbh_mean_acc": mean_acc,
        "bbh_task_acc": task_acc,
        "lm_eval_versions": results.get("versions"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval_bbh] bbh_mean_acc={mean_acc:.6f} wrote {out_json}")
    return 0 if mean_acc == mean_acc else 3


if __name__ == "__main__":
    raise SystemExit(main())
