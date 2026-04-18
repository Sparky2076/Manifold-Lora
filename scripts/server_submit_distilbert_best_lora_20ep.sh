#!/usr/bin/env bash
# 便捷封装：LoRA 最优（results/summary.csv 首行）+ 20 epoch
exec "$(cd "$(dirname "$0")" && pwd)/server_submit_distilbert_best_20ep.sh" lora
