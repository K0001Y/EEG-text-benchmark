#!/usr/bin/env bash

# GLIM 模型评估脚本
# 使用方式（在项目根目录 /root/autodl-tmp/benchmark 下）：
#   bash benchmark_eval/scripts/run_eval_glim.sh

set -e

# 设置 HF 镜像站点（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

PROJECT_ROOT="/root/autodl-tmp/benchmark"
DATA_PATH="${PROJECT_ROOT}/benchmark_eval/data/unified_zuco.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_eval/test_outputs/eval_glim"
CHECKPOINT_PATH="${PROJECT_ROOT}/models/GLIM-main/checkpoints/glim-zuco-epoch=199-step=49600.ckpt"

echo "=============================================="
echo "GLIM Model Evaluation"
echo "=============================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Path: ${DATA_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

python -m benchmark_eval.evaluation.eval_runner \
  --data-path "${DATA_PATH}" \
  --phase test \
  --output-dir "${OUTPUT_DIR}" \
  --model-name glim \
  --model-checkpoint "${CHECKPOINT_PATH}" \
  --batch-size 8 \
  --num-workers 0 \
  --resume

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
