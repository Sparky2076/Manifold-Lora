#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from deepseek_autogrid.config import RESULTS_ROOT, grid_size


def _f(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Build Markdown analysis from DeepSeek summary.csv")
    p.add_argument("--summary", type=Path, default=RESULTS_ROOT / "summary.csv")
    p.add_argument("--output", type=Path, default=RESULTS_ROOT / "deepseek_grid_analysis.md")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = grid_size()
    ok_rows = [r for r in rows if r.get("status") == "ok" and _f(r.get("best_eval_perplexity")) is not None]
    if not args.allow_incomplete and len(ok_rows) < expected:
        raise SystemExit(f"Incomplete grid: ok rows {len(ok_rows)} < expected {expected}.")
    if not ok_rows:
        raise SystemExit("No valid ok rows in summary.")

    ppl = [_f(r["best_eval_perplexity"]) for r in ok_rows]
    ppl = [x for x in ppl if x is not None]
    top = sorted(ok_rows, key=lambda r: _f(r["best_eval_perplexity"]) or 1e18)[:15]

    def _group(col):
        g = defaultdict(list)
        for r in ok_rows:
            v = _f(r["best_eval_perplexity"])
            if v is None:
                continue
            g[str(r.get(col, ""))].append(v)
        out = []
        for k, vals in sorted(g.items(), key=lambda kv: statistics.mean(kv[1])):
            out.append((k, len(vals), min(vals), statistics.mean(vals)))
        return out

    lines = [
        "# DeepSeek 网格结果分析",
        "",
        "由 `python -m deepseek_autogrid.analyze_results` 自动生成。",
        "",
        f"- **生成时间（UTC）**：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z",
        f"- **有效行数**：{len(ok_rows)}（status=ok）",
        "",
        "## 整体指标（best_eval_perplexity）",
        "",
        "| 统计量 | 值 |",
        "|--------|-----|",
        f"| min | {min(ppl):.4f} |",
        f"| max | {max(ppl):.4f} |",
        f"| mean | {statistics.mean(ppl):.4f} |",
        f"| median | {statistics.median(ppl):.4f} |",
        "",
        "## Top 组合（按 perplexity 越低越好）",
        "",
        "| rank | best_eval_perplexity | lr | r | alpha | weight_decay |",
        "|------|----------------------|----|---|-------|--------------|",
    ]
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | {_f(r['best_eval_perplexity']):.4f} | {r.get('lr','')} | {r.get('lora_r','')} | {r.get('lora_alpha','')} | {r.get('weight_decay','')} |"
        )

    lines += ["", "## 分组统计（mean / min）", "", "| group | key | n | min | mean |", "|------|-----|---|-----|------|"]
    for gname, col in [("lr", "lr"), ("weight_decay", "weight_decay"), ("lora_r", "lora_r"), ("lora_alpha", "lora_alpha")]:
        for k, n, mn, avg in _group(col):
            lines.append(f"| {gname} | {k} | {n} | {mn:.4f} | {avg:.4f} |")

    lines += ["", "## 更新方式", "", "```bash", "python -m deepseek_autogrid.aggregate_results", "python -m deepseek_autogrid.analyze_results", "```"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
