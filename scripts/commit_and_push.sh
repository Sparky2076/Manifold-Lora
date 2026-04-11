#!/usr/bin/env bash
# 将本仓库改动提交并推送到 origin（GitHub）。在仓库根、Git Bash 中执行。
#
# 用法:
#   bash scripts/commit_and_push.sh
#   bash scripts/commit_and_push.sh "feat(distilbert): 网格迁移至 distilbert_autogrid"
#
# 若未传参数，会打开默认编辑器填写提交说明（与 git commit 行为一致）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误：当前目录不是 git 仓库（期望在 $ROOT）。"
  exit 1
fi

echo "=== git status（简要）==="
git status -sb

echo ""
echo "=== 建议纳入本次 DistilBERT / 网格相关的路径（可按需增删）==="
echo "  distilbert/"
echo "  distilbert_autogrid/"
echo "  scripts/upload.sh scripts/upload.ps1 scripts/commit_and_push.sh"
echo "  README.md"
echo ""
read -r -p "是否继续执行 git add（上述目录 + 根目录 README 与 scripts）？[y/N] " ok || true
if [[ "${ok:-}" != "y" && "${ok:-}" != "Y" ]]; then
  echo "已取消。你可手动: git add -p && git commit && git push"
  exit 0
fi

git add README.md scripts/upload.sh scripts/upload.ps1 scripts/commit_and_push.sh
git add distilbert/ distilbert_autogrid/ 2>/dev/null || true

echo ""
echo "=== 暂存区 diff 统计 ==="
git diff --cached --stat || true

MSG="${1:-}"
if [[ -n "$MSG" ]]; then
  git commit -m "$MSG"
else
  echo ""
  echo "未提供提交说明，将启动编辑器（空消息会中止提交）。"
  git commit
fi

echo ""
echo "=== 推送到 origin 当前分支 ==="
BR="$(git branch --show-current)"
read -r -p "执行: git push -u origin \"$BR\" ? [y/N] " pushok || true
if [[ "${pushok:-}" == "y" || "${pushok:-}" == "Y" ]]; then
  git push -u origin "$BR"
  echo "完成。"
else
  echo "已跳过 push。稍后请执行: git push -u origin $BR"
fi
