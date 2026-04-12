#!/usr/bin/env bash
# 终止本用户在 LSF 上提交的 DistilBERT 网格作业（JOB_NAME 前缀 distilbert_grid_*，与 run_grid_bsub.sh 一致）
# 用法: bash scripts/kill_distilbert_grid_bjobs.sh
#       bash scripts/kill_distilbert_grid_bjobs.sh --yes   # 不询问直接 bkill
# 环境: GRID_JOB_PREFIX=distilbert_grid  GRID_KILL_YES=1 等同 --yes
#
# 若 bjobs -o 不支持，可手动: bjobs -u $USER -w | 找 distilbert_grid 行，再 bkill <JOBID>

set -euo pipefail

PREFIX="${GRID_JOB_PREFIX:-distilbert_grid}"
YES=0
if [[ "${1:-}" == "--yes" ]] || [[ "${GRID_KILL_YES:-}" == "1" ]]; then
  YES=1
fi

u="${USER:-${LOGNAME:-}}"
if [[ -z "$u" ]]; then
  echo "USER empty" >&2
  exit 1
fi
if ! command -v bjobs >/dev/null 2>&1 || ! command -v bkill >/dev/null 2>&1; then
  echo "Need bjobs and bkill (LSF) in PATH." >&2
  exit 1
fi

out=$(bjobs -u "$u" -o 'jobid job_name' 2>/dev/null) || true
if [[ -z "$out" ]]; then
  echo "No unfinished jobs (or bjobs failed)."
  exit 0
fi
if ! echo "$out" | head -1 | grep -qiE 'jobid|JOBID'; then
  echo "bjobs -o 'jobid job_name' not supported on this cluster. Manual: bjobs -u $u -w then bkill <JOBID>" >&2
  exit 2
fi

ids=()
while read -r jid name _; do
  [[ -z "${jid:-}" ]] && continue
  [[ "$jid" =~ ^[0-9]+$ ]] || continue
  [[ "$name" == "${PREFIX}"* ]] && ids+=("$jid")
done < <(echo "$out" | tail -n +2)

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "No jobs matching ${PREFIX}* for user ${u}."
  exit 0
fi

echo "Will bkill: ${ids[*]}"
if [[ "$YES" -eq 0 ]]; then
  read -r -p "Confirm? [y/N] " ans
  [[ "${ans:-}" == [yY]* ]] || { echo "Aborted."; exit 0; }
fi

bkill "${ids[@]}"
echo "bkill sent for ${#ids[@]} job(s)."
