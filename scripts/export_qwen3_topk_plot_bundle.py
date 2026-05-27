#!/usr/bin/env python3
"""Export Top-K summary slice + SFT curve paths for plotting (train_sft.csv / test_sft.csv)."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _pick_runs(summary_csv: Path, top_k: int, metric: str, sort: str, lora_type: str) -> list[str]:
    script = Path(__file__).resolve().parent / "pick_topk_deepseek_runs.py"
    cmd = [
        sys.executable,
        str(script),
        "--summary-csv",
        str(summary_csv),
        "--top-k",
        str(top_k),
        "--require-status-ok",
        "--lora-type",
        lora_type,
        "--metric",
        metric,
        "--sort",
        sort,
    ]
    out = subprocess.check_output(cmd, text=True)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Bundle Top-K SFT curves + summary slice for plots.")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--metric", type=str, default="mmlu_mean_acc")
    p.add_argument("--sort", type=str, default="desc")
    p.add_argument("--lora-type", type=str, default="default")
    args = p.parse_args()

    runs = _pick_runs(args.summary_csv, args.top_k, args.metric, args.sort, args.lora_type)
    if not runs:
        print("[export_topk_plot] no runs matched", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "sft_curves"
    curves_dir.mkdir(exist_ok=True)

    rows = list(csv.DictReader(args.summary_csv.open(newline="", encoding="utf-8")))
    run_set = set(runs)
    picked_rows = []
    for r in rows:
        md = (r.get("metrics_dir") or "").replace("\\", "/").rstrip("/")
        name = md.split("/")[-1] if md else ""
        if name in run_set:
            picked_rows.append(r)

    summary_out = out_dir / "summary_topk.csv"
    if picked_rows:
        with summary_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(picked_rows[0].keys()))
            w.writeheader()
            w.writerows(picked_rows)

    manifest: list[dict[str, str]] = []
    root = args.results_root.resolve()
    for run in runs:
        md = root / run
        entry = {"run_name": run, "metrics_dir": str(md)}
        for label, fname in (("train_sft", "train_sft.csv"), ("test_sft", "test_sft.csv")):
            src = md / fname
            dst = curves_dir / f"{run}_{label}.csv"
            entry[f"{label}_csv"] = str(src)
            if src.is_file():
                shutil.copy2(src, dst)
                entry[f"{label}_copy"] = str(dst)
        for label, fname in (
            ("mmlu_eval_full", "mmlu_eval_full.json"),
            ("mmlu_eval_full_progress", "mmlu_eval_full_progress.csv"),
            ("mmlu_eval_tiny", "mmlu_eval.json"),
        ):
            pth = md / fname
            entry[label] = str(pth) if pth.is_file() else ""
        manifest.append(entry)

    manifest_path = out_dir / "topk_manifest.json"
    manifest_path.write_text(json.dumps({"runs": runs, "entries": manifest}, indent=2), encoding="utf-8")
    print(f"[export_topk_plot] runs={len(runs)} out={out_dir} summary={summary_out}")
    for run in runs:
        print(f"  {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
