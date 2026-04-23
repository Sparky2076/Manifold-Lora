#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_summary_map(summary_csv: Path) -> dict[str, dict[str, str]]:
    if not summary_csv.is_file():
        return {}
    rows = list(csv.DictReader(summary_csv.open(newline="", encoding="utf-8")))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        md = (r.get("metrics_dir") or "").replace("\\", "/").rstrip("/")
        run_name = md.split("/")[-1] if md else ""
        if run_name:
            out[run_name] = r
    return out


def _iter_eval_json(results_root: Path):
    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        p = run_dir / "bbh_eval.json"
        if p.is_file():
            yield run_dir, p


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize DeepSeek BBH eval outputs into ranked CSV/Markdown.")
    p.add_argument("--results-root", type=Path, required=True, help="Root containing run dirs and bbh_eval.json")
    p.add_argument("--summary-csv", type=Path, default=None, help="Optional training summary.csv for side metrics")
    p.add_argument("--output-csv", type=Path, default=None, help="Default: <results-root>/bbh_leaderboard.csv")
    p.add_argument("--output-md", type=Path, default=None, help="Default: <results-root>/bbh_leaderboard.md")
    args = p.parse_args()

    root = args.results_root.resolve()
    out_csv = (args.output_csv or (root / "bbh_leaderboard.csv")).resolve()
    out_md = (args.output_md or (root / "bbh_leaderboard.md")).resolve()
    summary_map = _load_summary_map(args.summary_csv.resolve()) if args.summary_csv else {}

    rows: list[dict[str, object]] = []
    for run_dir, eval_json in _iter_eval_json(root):
        payload = _read_json(eval_json)
        if not payload:
            continue
        run_name = run_dir.name
        mean_acc = payload.get("bbh_mean_acc")
        if mean_acc is None:
            continue
        task_acc = payload.get("bbh_task_acc") or {}
        n_tasks = len(task_acc) if isinstance(task_acc, dict) else 0
        tr = summary_map.get(run_name, {})
        rows.append(
            {
                "run_name": run_name,
                "bbh_mean_acc": float(mean_acc),
                "bbh_tasks": n_tasks,
                "best_eval_perplexity": tr.get("best_eval_perplexity", ""),
                "post_peak_last_over_best": tr.get("post_peak_last_over_best", ""),
                "post_peak_tail_mean_over_best": tr.get("post_peak_tail_mean_over_best", ""),
                "bbh_eval_json": str(eval_json),
            }
        )

    rows.sort(key=lambda r: float(r["bbh_mean_acc"]), reverse=True)

    fieldnames = [
        "run_name",
        "bbh_mean_acc",
        "bbh_tasks",
        "best_eval_perplexity",
        "post_peak_last_over_best",
        "post_peak_tail_mean_over_best",
        "bbh_eval_json",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# DeepSeek BBH Leaderboard",
        "",
        f"- results_root: `{root}`",
        f"- evaluated runs: **{len(rows)}**",
        "",
        "| rank | run_name | bbh_mean_acc | tasks | best_eval_perplexity | post_peak_last_over_best | post_peak_tail_mean_over_best |",
        "|------|----------|--------------|-------|----------------------|--------------------------|-------------------------------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['run_name']} | {float(r['bbh_mean_acc']):.6f} | {r['bbh_tasks']} | "
            f"{r['best_eval_perplexity']} | {r['post_peak_last_over_best']} | {r['post_peak_tail_mean_over_best']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
