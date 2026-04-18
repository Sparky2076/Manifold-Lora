#!/usr/bin/env bash
# 便捷封装：mLoRA 最优（results_mlora/summary.csv 首行）+ 20 epoch
exec "$(cd "$(dirname "$0")" && pwd)/server_submit_distilbert_best_20ep.sh" mlora
