#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deepseek_autogrid.config import PROJECT_ROOT, RESULTS_ROOT, grid_size


def read_meta(run_dir: Path) -> dict | None:
    p = run_dir / "run_meta.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def best_from_test(path: Path):
    if not path.is_file():
        return None, None, None
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                it = int(row["iteration"])
                loss = float(row["eval_loss"])
                ppl = float(row["eval_perplexity"])
            except (KeyError, ValueError):
                continue
            rows.append((it, loss, ppl))
    if not rows:
        return None, None, None
    best = min(rows, key=lambda x: x[2])
    return best[2], best[1], best[0]


def aggregate() -> int:
    p = argparse.ArgumentParser(description="Build DeepSeek summary.csv from results")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()

    root = args.results_root.resolve()
    out = args.output or (root / "summary.csv")

    fields = [
        "lr",
        "lora_r",
        "lora_alpha",
        "weight_decay",
        "max_steps",
        "eval_every",
        "sft_preset",
        "sft_val_ratio",
        "lora_type",
        "best_eval_perplexity",
        "best_eval_loss",
        "best_iteration",
        "last_eval_perplexity",
        "last_eval_loss",
        "last_iteration",
        "metrics_dir",
        "status",
    ]

    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_csv = run_dir / "test_sft.csv"
        meta = read_meta(run_dir)
        best_ppl, best_loss, best_it = best_from_test(test_csv)

        last_it = last_loss = last_ppl = None
        if test_csv.is_file():
            with test_csv.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        last_it = int(row["iteration"])
                        last_loss = float(row["eval_loss"])
                        last_ppl = float(row["eval_perplexity"])
                    except (KeyError, ValueError):
                        continue

        if test_csv.is_file() and best_ppl is not None:
            status = "ok"
        elif meta and not test_csv.is_file():
            status = "incomplete"
        elif not meta and not test_csv.is_file():
            continue
        else:
            status = "parse_error"

        rows.append(
            {
                "lr": meta.get("lr") if meta else "",
                "lora_r": meta.get("lora_r") if meta else "",
                "lora_alpha": meta.get("lora_alpha") if meta else "",
                "weight_decay": meta.get("weight_decay") if meta else "",
                "max_steps": meta.get("max_steps") if meta else "",
                "eval_every": meta.get("eval_every") if meta else "",
                "sft_preset": meta.get("sft_preset") if meta else "",
                "sft_val_ratio": meta.get("sft_val_ratio") if meta else "",
                "lora_type": meta.get("lora_type") if meta else "",
                "best_eval_perplexity": best_ppl if best_ppl is not None else "",
                "best_eval_loss": best_loss if best_loss is not None else "",
                "best_iteration": best_it if best_it is not None else "",
                "last_eval_perplexity": last_ppl if last_ppl is not None else "",
                "last_eval_loss": last_loss if last_loss is not None else "",
                "last_iteration": last_it if last_it is not None else "",
                "metrics_dir": run_dir.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "status": status,
            }
        )

    rows.sort(key=lambda x: (x["best_eval_perplexity"] == "", float(x["best_eval_perplexity"] or 1e18)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = sum(1 for r in rows if r.get("status") == "ok")
    expected = grid_size()
    print(f"Wrote {len(rows)} rows to {out} (ok={ok_rows}, expected={expected})")
    if not args.allow_incomplete and ok_rows < expected:
        print(
            f"[aggregate_results] ERROR: grid incomplete (ok={ok_rows} < expected={expected}). "
            "Finish missing runs first or pass --allow-incomplete."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(aggregate())
