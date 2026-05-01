#!/usr/bin/env bash
# =============================================================================
# EEG2Text 完整评估流程脚本
#
# 流程：
#   Step 1 - 生成统一数据集 unified_zuco.pkl（从 ZuCo .mat 文件）
#   Step 2 - 运行 EEG2Text 模型推理，生成 predictions.jsonl（自动在 screen 中运行）
#   Step 3 - 从 predictions.jsonl 离线计算 BLEU/ROUGE/WER 指标
#
# 用法（在项目根目录下执行）：
#   bash benchmark_eval/scripts/shell/run_eeg2text_pipeline.sh [选项]
#
# 选项：
#   --skip-build     跳过 Step 1（已有 unified_zuco.pkl 时使用）
#   --skip-eval      跳过 Step 2（已有 predictions.jsonl 时使用）
#   --skip-metrics   跳过 Step 3
#   --phase PHASE    数据集分割，默认 test
#   --batch-size N   推理 batch size，默认 4
#   --help           显示帮助
# =============================================================================

set -euo pipefail

# ============================
# 可配置路径（根据实际情况修改）
# ============================

# 自动推断项目根目录（本脚本位于 benchmark_eval/scripts/shell/ 下）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ZuCo 数据集根目录（包含 task1-SR/task2-NR/task3-TSR/task2-NR-2.0 子目录）
ZUCO_ROOT="${PROJECT_ROOT}/models/EEG2Text-main/dataset/ZuCo"

# 要处理的任务列表（逗号分隔）
TASKS="task1-SR,task2-NR,task3-TSR,task2-NR-2.0"

# 统一数据集输出路径
UNIFIED_PKL="${PROJECT_ROOT}/benchmark_eval/data/unified_zuco.pkl"

# EEG2Text 微调 checkpoint 路径
CHECKPOINT="${PROJECT_ROOT}/models/EEG2Text-main/checkpoints/decoding/last/task1_task2_finetune_BrainTranslator_skipstep1_b4_20_30_5e-05_1e-06_unique_sent-pretrain_robert-braintranslator.pt"

# 评估输出目录
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_eval/test_outputs/eval_eeg2text"

# 实时日志文件（eval 推理）
EVAL_LOG="${PROJECT_ROOT}/benchmark_eval/test_outputs/eval_eeg2text_run.log"

# Step 1 参数
MAX_LEN=58
DIM=105
EEG_TYPE="GD"

# ============================
# 命令行参数解析
# ============================

SKIP_BUILD=false
SKIP_EVAL=false
SKIP_METRICS=false
PHASE="test"
BATCH_SIZE=4

print_help() {
    grep "^#" "$0" | grep -v "^#!/" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)    SKIP_BUILD=true;      shift ;;
        --skip-eval)     SKIP_EVAL=true;       shift ;;
        --skip-metrics)  SKIP_METRICS=true;    shift ;;
        --phase)         PHASE="$2";           shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";      shift 2 ;;
        --help|-h)       print_help ;;
        *) echo "[WARN] 未知参数: $1，忽略" >&2; shift ;;
    esac
done

# ============================
# 工具函数
# ============================

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ============================
# 前置检查
# ============================

log "项目根目录: ${PROJECT_ROOT}"
log "Phase: ${PHASE} | BatchSize: ${BATCH_SIZE}"

[[ -d "${PROJECT_ROOT}/benchmark_eval" ]] || die "请在项目根目录下执行本脚本"

mkdir -p "$(dirname "${UNIFIED_PKL}")"
mkdir -p "${OUTPUT_DIR}"

# ============================
# Step 1: 生成统一数据集
# ============================

if [[ "${SKIP_BUILD}" == "true" ]]; then
    log "=== [Step 1] 跳过数据集生成（--skip-build）==="
    [[ -f "${UNIFIED_PKL}" ]] || die "UNIFIED_PKL 不存在: ${UNIFIED_PKL}，请先生成或去掉 --skip-build"
else
    log "=== [Step 1] 生成统一数据集 ==="
    log "ZuCo root: ${ZUCO_ROOT}"
    log "Tasks:     ${TASKS}"
    log "Output:    ${UNIFIED_PKL}"

    [[ -d "${ZUCO_ROOT}" ]] || die "ZuCo 根目录不存在: ${ZUCO_ROOT}"

    python3 benchmark_eval/data_processing/build_unified_dataset.py \
        --zuco-root "${ZUCO_ROOT}" \
        --tasks "${TASKS}" \
        --output "${UNIFIED_PKL}" \
        --max-len "${MAX_LEN}" \
        --dim "${DIM}" \
        --eeg-type "${EEG_TYPE}"

    log "Step 1 完成 -> ${UNIFIED_PKL}"
fi

# ============================
# Step 2: EEG2Text 模型推理
# ============================

if [[ "${SKIP_EVAL}" == "true" ]]; then
    log "=== [Step 2] 跳过模型推理（--skip-eval）==="
    [[ -f "${OUTPUT_DIR}/predictions.jsonl" ]] || die "predictions.jsonl 不存在，请先运行评估或去掉 --skip-eval"
else
    log "=== [Step 2] EEG2Text 模型推理 ==="
    log "Checkpoint: ${CHECKPOINT}"
    log "Output dir: ${OUTPUT_DIR}"
    log "Log file:   ${EVAL_LOG}"

    [[ -f "${CHECKPOINT}" ]] || die "Checkpoint 不存在: ${CHECKPOINT}"

    SCREEN_NAME="eval_eeg2text_$$"
    log "在 screen 会话 '${SCREEN_NAME}' 中启动推理（可用 screen -r ${SCREEN_NAME} 查看进度）"

    EVAL_CMD="TRANSFORMERS_OFFLINE=1 python3 benchmark_eval/evaluation/eval_runner.py \
        --data-path ${UNIFIED_PKL} \
        --phase ${PHASE} \
        --output-dir ${OUTPUT_DIR} \
        --model-name eeg2text \
        --model-checkpoint ${CHECKPOINT} \
        --batch-size ${BATCH_SIZE} \
        --resume \
        2>&1 | tee ${EVAL_LOG}"

    screen -dmS "${SCREEN_NAME}" bash -c "cd ${PROJECT_ROOT} && ${EVAL_CMD}; echo '[DONE] eval finished'"

    log "等待推理完成（每 30 秒检查一次状态）..."
    while screen -list | grep -q "${SCREEN_NAME}"; do
        sleep 30
        TOTAL=$(python3 -c "
import json, os
p = '${OUTPUT_DIR}/state.json'
if os.path.exists(p):
    s = json.load(open(p))
    print(f\"{s.get('next_index',0)}/{s.get('total','?')}\")" 2>/dev/null || echo "?/?")
        log "  进度: ${TOTAL}"
    done

    log "Step 2 完成"

    # 检查 predictions.jsonl 是否生成
    [[ -f "${OUTPUT_DIR}/predictions.jsonl" ]] || die "predictions.jsonl 未生成，请检查日志: ${EVAL_LOG}"
    PRED_COUNT=$(wc -l < "${OUTPUT_DIR}/predictions.jsonl")
    log "  预测条数: ${PRED_COUNT}"
fi

# ============================
# Step 3: 计算评估指标
# ============================

if [[ "${SKIP_METRICS}" == "true" ]]; then
    log "=== [Step 3] 跳过指标计算（--skip-metrics）==="
else
    log "=== [Step 3] 计算 BLEU / ROUGE / WER 指标 ==="

    python3 benchmark_eval/scripts/analysis/compute_eeg2text_metrics.py \
        --pred-path "${OUTPUT_DIR}/predictions.jsonl" \
        --output "${OUTPUT_DIR}/metrics.json"

    log "Step 3 完成 -> ${OUTPUT_DIR}/metrics.json"

    # 打印最终结果
    log ""
    log "=========================================="
    log "EEG2Text 评估结果（phase=${PHASE}）"
    log "=========================================="
    python3 -c "
import json
m = json.load(open('${OUTPUT_DIR}/metrics.json'))
ov = m['overall']
print(f'  BLEU-1:  {ov[\"bleu1\"]:.2f}%')
print(f'  BLEU-2:  {ov[\"bleu2\"]:.2f}%')
print(f'  BLEU-3:  {ov[\"bleu3\"]:.2f}%')
print(f'  BLEU-4:  {ov[\"bleu4\"]:.2f}%')
print(f'  ROUGE-1: {ov[\"rouge1\"]:.2f}%')
print(f'  ROUGE-2: {ov[\"rouge2\"]:.2f}%')
print(f'  ROUGE-L: {ov[\"rougeL\"]:.2f}%')
print(f'  WER:     {ov[\"wer\"]:.2f}%')
print()
by_task = m.get('grouped', {}).get('by_task', {})
if by_task:
    print('  -- 按任务分组 --')
    for task, tm in sorted(by_task.items()):
        n = m.get('task_sample_counts', {}).get(task, '?')
        print(f'  {task}: BLEU-1={tm[\"bleu1\"]:.2f}%  ROUGE-1={tm[\"rouge1\"]:.2f}%  WER={tm[\"wer\"]:.2f}%')
"
fi

log "=========================================="
log "EEG2Text 全流程完成"
log "  统一数据集: ${UNIFIED_PKL}"
log "  预测结果:   ${OUTPUT_DIR}/predictions.jsonl"
log "  评估指标:   ${OUTPUT_DIR}/metrics.json"
log "=========================================="
