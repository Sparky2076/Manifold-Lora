#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from deepseek_autogrid.config import MAX_STEPS_DEFAULT, PROJECT_ROOT, RESULTS_ROOT, iter_grid, run_dir_name


def _rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return max(0, len(path.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    except OSError:
        return 0


def _wait_slot(max_run: int, max_pend: int, poll_sec: int) -> None:
    if max_run <= 0 and max_pend <= 0:
        return
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        return
    while True:
        p = subprocess.run(["bjobs", "-u", user], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return
        run_n = pend_n = 0
        for line in p.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3:
                run_n += 1 if parts[2].startswith("RUN") else 0
                pend_n += 1 if parts[2].startswith("PEND") else 0
        if (max_run <= 0 or run_n <= max_run) and (max_pend <= 0 or pend_n < max_pend):
            return
        time.sleep(max(1, poll_sec))


def main() -> int:
    p = argparse.ArgumentParser(description="Find/submit missing DeepSeek grid runs")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--submit-bsub", action="store_true")
    p.add_argument("--submit-sleep-sec", type=int, default=int(os.environ.get("SUBMIT_SLEEP_SEC", "180") or "180"))
    p.add_argument("--grid-max-run", type=int, default=int(os.environ.get("GRID_MAX_RUN", "0") or "0"))
    p.add_argument("--grid-max-pend", type=int, default=int(os.environ.get("GRID_MAX_PEND", "0") or "0"))
    p.add_argument("--grid-poll-sec", type=int, default=int(os.environ.get("GRID_POLL_SEC", "30") or "30"))
    args = p.parse_args()

    root = args.results_root.resolve()
    out_csv = args.output or (root / "missing_runs.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    total = complete = 0
    for lr, r, a, wd in iter_grid():
        total += 1
        name = run_dir_name(lr, r, a, args.max_steps, wd)
        run_dir = root / name
        rows = _rows(run_dir / "test_sft.csv")
        if rows >= 1:
            complete += 1
            continue
        if not run_dir.is_dir():
            reason = "missing_dir"
        elif not (run_dir / "test_sft.csv").is_file():
            reason = "missing_test"
        else:
            reason = "incomplete_test_rows"
        missing.append(
            {
                "run_name": name,
                "lr": str(lr),
                "lora_r": str(r),
                "lora_alpha": str(a),
                "weight_decay": str(wd),
                "test_rows_found": str(rows),
                "reason": reason,
                "metrics_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["run_name", "lr", "lora_r", "lora_alpha", "weight_decay", "test_rows_found", "reason", "metrics_dir"]
        )
        w.writeheader()
        w.writerows(missing)
    print(f"[deepseek fill_missing] total={total} complete={complete} missing={len(missing)} -> {out_csv}")

    if not args.submit_bsub:
        return 0

    submit_sh = PROJECT_ROOT / "deepseek" / "scripts" / "submit_bsub_sft.sh"
    for i, row in enumerate(missing, 1):
        _wait_slot(args.grid_max_run, args.grid_max_pend, args.grid_poll_sec)
        env = os.environ.copy()
        env["JOB_NAME"] = f"deepseek_grid_{row['run_name']}"
        env["LR"] = row["lr"]
        env["LORA_R"] = row["lora_r"]
        env["LORA_ALPHA"] = row["lora_alpha"]
        env["WEIGHT_DECAY"] = row["weight_decay"]
        env["MAX_STEPS"] = str(args.max_steps)
        env["METRICS_DIR"] = str((root / row["run_name"]).resolve())
        print(f"[deepseek fill_missing] submit {i}/{len(missing)}: {row['run_name']}")
        subprocess.run(["bash", str(submit_sh)], cwd=str(PROJECT_ROOT), env=env, check=False)
        if i < len(missing):
            time.sleep(max(0, args.submit_sleep_sec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
