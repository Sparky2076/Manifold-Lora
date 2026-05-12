#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from deepseek_autogrid.grid_config import load_grid_config


def _f(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Build Markdown analysis from DeepSeek summary.csv")
    p.add_argument(
        "--config-module",
        type=str,
        default=None,
        help="Must match aggregate (default: env DEEPSEEK_GRID_CONFIG_MODULE or deepseek_autogrid.config).",
    )
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()

    cfg = load_grid_config(args.config_module)
    rr = cfg.RESULTS_ROOT
    summary_path = args.summary or (rr / "summary.csv")
    output_path = args.output or (rr / "deepseek_grid_analysis.md")

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = cfg.grid_size()
    ok_rows = [r for r in rows if r.get("status") == "ok" and _f(r.get("best_eval_perplexity")) is not None]
    if not args.allow_incomplete and len(ok_rows) < expected:
        raise SystemExit(f"Incomplete grid: ok rows {len(ok_rows)} < expected {expected}.")
    if not ok_rows:
        raise SystemExit("No valid ok rows in summary.")

    ppl = [_f(r["best_eval_perplexity"]) for r in ok_rows]
    ppl = [x for x in ppl if x is not None]
    top = sorted(ok_rows, key=lambda r: _f(r["best_eval_perplexity"]) or 1e18)[:15]

    def _pk(r, key: str) -> str:
        v = (r.get(key) or "").strip()
        return v if v else "-"

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
        "| rank | best_ppl | last/best | tail_mean/best | lr | r | alpha | wd |",
        "|------|----------|-----------|----------------|----|---|-------|-----|",
    ]
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | {_f(r['best_eval_perplexity']):.4f} | {_pk(r, 'post_peak_last_over_best')} | {_pk(r, 'post_peak_tail_mean_over_best')} | "
            f"{r.get('lr','')} | {r.get('lora_r','')} | {r.get('lora_alpha','')} | {r.get('weight_decay','')} |"
        )

    drift_rows = []
    for r in ok_rows:
        t = _f(r.get("post_peak_tail_mean_over_best"))
        if t is not None:
            drift_rows.append((t, r))
    drift_rows.sort(key=lambda x: -x[0])
    lines += [
        "",
        "## Post-peak（最优之后又发散了吗？）",
        "",
        "- **post_peak_last_over_best**：最后一次 eval 的 ppl / 全程最优；**接近 1 好**，明显 **>1** 表示终点比最低点差。",
        "- **post_peak_tail_mean_over_best**：达到最优 perplexity 的那次 eval **之后**，所有 eval 点的平均 ppl / 最优 ppl；**接近 1** 表示最优后整体未漂，**明显 >1** 表示最优后持续变差。",
        "",
        "### 尾部发散最明显（tail_mean/best 降序，最多 12 行）",
        "",
        "| tail/best | last/best | best_ppl | lr | r | alpha | wd |",
        "|-------------|-----------|----------|----|---|-------|-----|",
    ]
    if drift_rows:
        for t, r in drift_rows[:12]:
            lines.append(
                f"| {t:.4f} | {_pk(r, 'post_peak_last_over_best')} | {_f(r['best_eval_perplexity']):.4f} | "
                f"{r.get('lr','')} | {r.get('lora_r','')} | {r.get('lora_alpha','')} | {r.get('weight_decay','')} |"
            )
    else:
        lines.append("| （无 post_peak 列：请重新运行 `aggregate_results` 生成 summary） | | | | | | |")

    lines += ["", "## 分组统计（mean / min）", "", "| group | key | n | min | mean |", "|------|-----|---|-----|------|"]
    for gname, col in [("lr", "lr"), ("weight_decay", "weight_decay"), ("lora_r", "lora_r"), ("lora_alpha", "lora_alpha")]:
        for k, n, mn, avg in _group(col):
            lines.append(f"| {gname} | {k} | {n} | {mn:.4f} | {avg:.4f} |")

    lines += [
        "",
        "## 更新方式",
        "",
        "```bash",
        "python -m deepseek_autogrid.aggregate_results",
        "python -m deepseek_autogrid.analyze_results",
        "# 细网格（config_refine + results_refine）：",
        "# python -m deepseek_autogrid.aggregate_results --config-module deepseek_autogrid.config_refine --results-root deepseek_autogrid/results_refine",
        "# python -m deepseek_autogrid.analyze_results --config-module deepseek_autogrid.config_refine --summary deepseek_autogrid/results_refine/summary.csv",
        "```",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
