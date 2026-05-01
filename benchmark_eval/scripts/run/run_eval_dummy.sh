#!/usr/bin/env bash

# 简单示例脚本：运行统一 EEG-to-Text benchmark 评估（使用 dummy 模型）
# 使用方式（在项目根目录 f:\Files\代码复现\benchmark 下）：
#   bash benchmark_eval/run_eval_dummy.sh
# 请根据你的实际数据路径和 Python 解释器路径进行修改。

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${PROJECT_ROOT}/unified_dataset_example.pkl"  # TODO: 修改为你的统一数据集路径
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results/dummy"

mkdir -p "${OUTPUT_DIR}"

python -m benchmark_eval.eval_runner \
  --data-path "${DATA_PATH}" \
  --phase test \
  --output-dir "${OUTPUT_DIR}" \
  --model-name dummy \
  --batch-size 8 \
  --num-workers 0 \
  --resume
