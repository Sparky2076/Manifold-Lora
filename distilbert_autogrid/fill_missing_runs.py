#!/usr/bin/env python3
"""Find missing/incomplete grid runs, write missing_runs.csv, optionally submit only missing combos."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from distilbert_autogrid.config import EPOCHS_DEFAULT, PROJECT_ROOT, RESULTS_ROOT, iter_grid, run_dir_name


def _test_rows(test_csv: Path) -> int:
    if not test_csv.is_file():
        return 0
    try:
        # Keep exactly the same completeness rule as run_grid_bsub.sh:
        # data rows = total lines - header line.
        lines = test_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
        return max(0, len(lines) - 1)
    except Exception:
        return 0


def _wait_slot(max_run: int, max_pend: int, poll_sec: int) -> None:
    if max_run <= 0 and max_pend <= 0:
        return
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        return
    while True:
        run_n = 0
        pend_n = 0
        p = subprocess.run(["bjobs", "-u", user], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return
        for line in p.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3:
                st = parts[2]
                if st.startswith("RUN"):
                    run_n += 1
                if st.startswith("PEND"):
                    pend_n += 1
        need_wait = False
        if max_run > 0 and run_n > max_run:
            need_wait = True
        if max_pend > 0 and pend_n >= max_pend:
            need_wait = True
        if not need_wait:
            return
        print(
            f"[fill_missing] throttle: RUN={run_n} PEND={pend_n} "
            f"(wait if RUN>{max_run} or PEND>={max_pend}); sleep {poll_sec}s ...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(1, poll_sec))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff expected full grid vs results dirs, write missing_runs.csv, optionally submit only missing runs."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-root>/missing_runs.csv)",
    )
    parser.add_argument(
        "--submit-bsub",
        action="store_true",
        help="Submit only missing/incomplete combos via distilbert/scripts/submit_bsub.sh",
    )
    parser.add_argument(
        "--submit-sleep-sec",
        type=int,
        default=int(os.environ.get("SUBMIT_SLEEP_SEC", "180") or "180"),
        help="Sleep seconds between submissions when --submit-bsub is enabled",
    )
    parser.add_argument(
        "--grid-max-run",
        type=int,
        default=int(os.environ.get("GRID_MAX_RUN", "0") or "0"),
        help="Throttle: wait while RUN > this value (0=off)",
    )
    parser.add_argument(
        "--grid-max-pend",
        type=int,
        default=int(os.environ.get("GRID_MAX_PEND", "0") or "0"),
        help="Throttle: wait while PEND >= this value (0=off)",
    )
    parser.add_argument(
        "--grid-poll-sec",
        type=int,
        default=int(os.environ.get("GRID_POLL_SEC", "30") or "30"),
        help="Throttle poll interval (seconds)",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_csv = args.output or (results_root / "missing_runs.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, str]] = []
    total = 0
    complete = 0
    for lr, r, alpha, wd in iter_grid():
        total += 1
        name = run_dir_name(lr, r, alpha, args.epochs, wd)
        run_dir = results_root / name
        meta = run_dir / "run_meta.json"
        test_csv = run_dir / "test.csv"
        rows = _test_rows(test_csv)
        done = rows >= args.epochs
        if done:
            complete += 1
            continue
        if not run_dir.is_dir():
            reason = "missing_dir"
        elif not meta.is_file() and not test_csv.is_file():
            reason = "missing_meta_and_test"
        elif not test_csv.is_file():
            reason = "missing_test"
        else:
            reason = "incomplete_test_rows"
        missing.append(
            {
                "run_name": name,
                "lr": str(lr),
                "lora_r": str(r),
                "lora_alpha": str(alpha),
                "weight_decay": str(wd),
                "epochs_required": str(args.epochs),
                "test_rows_found": str(rows),
                "reason": reason,
                "metrics_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "lr",
                "lora_r",
                "lora_alpha",
                "weight_decay",
                "epochs_required",
                "test_rows_found",
                "reason",
                "metrics_dir",
            ],
        )
        w.writeheader()
        w.writerows(missing)

    print(
        f"[fill_missing] total={total} complete={complete} missing={len(missing)} -> {out_csv}",
        flush=True,
    )

    if not args.submit_bsub:
        return 0

    submit_sh = PROJECT_ROOT / "distilbert" / "scripts" / "submit_bsub.sh"
    if not submit_sh.is_file():
        print(f"[fill_missing] submit script not found: {submit_sh}", file=sys.stderr)
        return 2

    for i, row in enumerate(missing, 1):
        _wait_slot(args.grid_max_run, args.grid_max_pend, args.grid_poll_sec)
        env = os.environ.copy()
        env["JOB_NAME"] = f"distilbert_grid_{row['run_name']}"
        env["LR"] = row["lr"]
        env["LORA_R"] = row["lora_r"]
        env["LORA_ALPHA"] = row["lora_alpha"]
        env["WEIGHT_DECAY"] = row["weight_decay"]
        env["EPOCHS"] = str(args.epochs)
        env["METRICS_DIR"] = str((results_root / row["run_name"]).resolve())
        print(
            f"[fill_missing] submit {i}/{len(missing)}: {row['run_name']} "
            f"(reason={row['reason']}, rows={row['test_rows_found']})",
            flush=True,
        )
        p = subprocess.run(["bash", str(submit_sh)], cwd=str(PROJECT_ROOT), env=env, check=False)
        if p.returncode != 0:
            print(f"[fill_missing] submit failed: {row['run_name']} code={p.returncode}", file=sys.stderr)
        if i < len(missing):
            time.sleep(max(0, args.submit_sleep_sec))

    print("[fill_missing] submit loop done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
