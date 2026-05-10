"""CLI: merge SFT LoRA snapshot into base HF weights (no lm-eval)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepseek.merge_lora_export import merge_and_export_hf


def main() -> int:
    p = argparse.ArgumentParser(description="Export merged HF weights from DeepSeek SFT run_dir.")
    p.add_argument("--metrics_dir", type=Path, required=True, help="Run dir with run_meta.json + sft_lora_state.pt")
    p.add_argument("--merged_hf_dir", type=Path, default=None, help="Default: <metrics_dir>/model_merged_hf")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None)
    args = p.parse_args()

    try:
        out = merge_and_export_hf(
            args.metrics_dir,
            args.merged_hf_dir,
            model_name=args.model_name,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[export_merged_hf] {e}", file=sys.stderr)
        return 2
    print(f"[export_merged_hf] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
