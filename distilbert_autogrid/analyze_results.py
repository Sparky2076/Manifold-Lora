#!/usr/bin/env python3
"""Read distilbert_autogrid/results/summary.csv and write results/distilbert_grid_analysis.md."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from distilbert_autogrid.config import RESULTS_ROOT, grid_size


def _f(x: str) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing summary: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Markdown analysis from summary.csv")
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULTS_ROOT / "summary.csv",
        help="Path to summary.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RESULTS_ROOT / "distilbert_grid_analysis.md",
        help="Output Markdown path",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow generating analysis even when summary does not cover full grid",
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    rows = load_rows(summary_path)
    expected = grid_size()
    ok_total = sum(1 for r in rows if r.get("status") == "ok")
    if not args.allow_incomplete and ok_total < expected:
        raise SystemExit(
            f"Incomplete grid: ok rows {ok_total} < expected {expected}. "
            "Finish missing runs first, or pass --allow-incomplete."
        )

    ok = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        acc = _f(r.get("best_val_acc", ""))
        if acc is None:
            continue
        ok.append(r)

    if not ok:
        raise SystemExit("No status=ok rows with best_val_acc in summary.")

    accs = [float(_f(r["best_val_acc"]) or 0) for r in ok]

    def by_lr(r: dict) -> str:
        v = _f(r.get("lr", ""))
        return f"{v:.4e}" if v is not None else "nan"

    def by_wd(r: dict) -> str:
        v = _f(r.get("weight_decay", ""))
        return f"{v:.4e}" if v is not None else "nan"

    # group stats: key -> list of acc
    g_lr: dict[str, list[float]] = defaultdict(list)
    g_wd: dict[str, list[float]] = defaultdict(list)
    g_r: dict[str, list[float]] = defaultdict(list)
    g_a: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        a = float(_f(r["best_val_acc"]) or 0)
        g_lr[by_lr(r)].append(a)
        g_wd[by_wd(r)].append(a)
        g_r[str(int(float(r["lora_r"]))) if r.get("lora_r") else "?"].append(a)
        g_a[str(int(float(r["lora_alpha"]))) if r.get("lora_alpha") else "?"].append(a)

    def agg(groups: dict[str, list[float]]) -> list[tuple[str, int, float, float]]:
        out = []
        for k, vals in sorted(groups.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True):
            out.append((k, len(vals), max(vals), statistics.mean(vals)))
        return out

    top_n = 15
    sorted_ok = sorted(ok, key=lambda r: float(_f(r["best_val_acc"]) or 0), reverse=True)[:top_n]

    lines: list[str] = [
        "# DistilBERT LoRA 网格结果分析",
        "",
        f"由 `python -m distilbert_autogrid.analyze_results` 根据 [`summary.csv`](summary.csv) 自动生成。",
        "",
        f"- **生成时间（UTC）**：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z",
        f"- **有效行数**：{len(ok)}（`status=ok` 且 `best_val_acc` 可解析）",
        "",
        "## 整体指标（验证集 best_val_acc）",
        "",
        "| 统计量 | 值 |",
        "|--------|-----|",
        f"| min | {min(accs):.4f} |",
        f"| max | {max(accs):.4f} |",
        f"| mean | {statistics.mean(accs):.4f} |",
        f"| median | {statistics.median(accs):.4f} |",
        "",
        "## Top 组合（按 best_val_acc）",
        "",
        "| rank | best_val_acc | lr | r | alpha | weight_decay |",
        "|------|--------------|-----|---|-------|--------------|",
    ]
    for i, r in enumerate(sorted_ok, 1):
        acc = float(_f(r["best_val_acc"]) or 0)
        lr = r.get("lr", "")
        lines.append(
            f"| {i} | {acc:.4f} | {lr} | {r.get('lora_r', '')} | {r.get('lora_alpha', '')} | {r.get('weight_decay', '')} |"
        )

    lines += [
        "",
        "## 按学习率 lr（组内 mean / max，行数 n）",
        "",
        "| lr | n | max | mean |",
        "|----|---|-----|------|",
    ]
    for k, n, mx, mn in agg(g_lr):
        lines.append(f"| {k} | {n} | {mx:.4f} | {mn:.4f} |")

    lines += [
        "",
        "## 按 weight_decay",
        "",
        "| weight_decay | n | max | mean |",
        "|--------------|---|-----|------|",
    ]
    for k, n, mx, mn in agg(g_wd):
        lines.append(f"| {k} | {n} | {mx:.4f} | {mn:.4f} |")

    lines += [
        "",
        "## 按 LoRA rank（r）",
        "",
        "| r | n | max | mean |",
        "|---|---|-----|------|",
    ]
    for k, n, mx, mn in agg(g_r):
        lines.append(f"| {k} | {n} | {mx:.4f} | {mn:.4f} |")

    lines += [
        "",
        "## 按 LoRA alpha（α）",
        "",
        "| alpha | n | max | mean |",
        "|-------|---|-----|------|",
    ]
    for k, n, mx, mn in agg(g_a):
        lines.append(f"| {k} | {n} | {mx:.4f} | {mn:.4f} |")

    lines += [
        "",
        "## 解读提示（非因果结论）",
        "",
        "- 表中为 **验证集** `best_val_acc`，来自各 run 的 `test.csv` 按 epoch/步聚合后的最优值。",
        "- **mean 高**表示该维度取值在网格内整体较稳；**max** 反映该维度下能达到的峰值。",
        "- 超参之间存在交互，请以 **Top 表** 与 `summary.csv` 全表为准做最终选型。",
        "",
        "## 更新方式",
        "",
        "```bash",
        "python -m distilbert_autogrid.aggregate_results",
        "python -m distilbert_autogrid.analyze_results",
        "```",
        "",
        "然后提交 `summary.csv`（若变更）与本文件。",
        "",
    ]

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
