#!/usr/bin/env bash
# =============================================================================
# 总控脚本：重新运行 A 线 + B 线（含跨 session 测试 / 可视化 / 显著性检验）
#
# 步骤：
#   Step A   - 诊断线 A：validate_eeg_signal.py（含 A1d/A2/A3 跨 session）
#   Step B   - 检索评估 4 模型 × 4 噪声条件 → 落盘 embeddings.npz
#   Step V   - visualize_b_embeddings.py 降维可视化 V1-V6
#   Step S   - run_significance_tests.py 显著性检验
#
# 用法：
#   bash benchmark_eval/scripts/shell/run_all_experiments.sh [选项]
#
# 选项：
#   --skip-a / --skip-b / --skip-viz / --skip-sig  分别跳过对应步骤
#   --only-model MODEL                             B 线只跑指定模型（可重复或逗号分隔）
#   --only-noise NOISE                             B 线只跑指定噪声（可重复或逗号分隔）
#   --help
# =============================================================================

set -u  # 注意这里不用 -e，以保证单个模型失败不会中断整体

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_PATH="${PROJECT_ROOT}/benchmark_eval/data/unified_zuco.pkl"
LINE_A_OUT="${PROJECT_ROOT}/benchmark_eval/test_outputs/dataset_validity"
LINE_B_ROOT="${PROJECT_ROOT}/benchmark_eval/test_outputs/line_b"
RESULTS_DIR="${PROJECT_ROOT}/benchmark_eval/test_outputs"

# checkpoint 路径
CKPT_CET_MAE="${PROJECT_ROOT}/models/CET-MAE/checkpoints/decoding/cet_mae_benchmark_best.pt"
CKPT_EEG_TO_TEXT="${PROJECT_ROOT}/models/EEG-To-Text-main/checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b32_20_30_5e-05_5e-07_unique_sent_EEG.pt"
CKPT_EEG2TEXT="${PROJECT_ROOT}/models/EEG2Text-main/checkpoints/decoding/best/task1_task2_finetune_BrainTranslator_skipstep1_b4_20_30_5e-05_1e-06_unique_sent-pretrain_robert-braintranslator.pt"
CKPT_GLIM="${PROJECT_ROOT}/models/GLIM-main/checkpoints/glim-zuco-epoch=199-step=49600.ckpt"

ALL_MODELS=("cet_mae" "eeg_to_text" "eeg2text" "glim")
ALL_NOISES=("real" "gaussian" "shuffle" "zero")

SKIP_A=false
SKIP_B=false
SKIP_VIZ=false
SKIP_SIG=false
ONLY_MODELS=()
ONLY_NOISES=()

print_help() { grep "^#" "$0" | grep -v "^#!/" | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-a)         SKIP_A=true;   shift ;;
        --skip-b)         SKIP_B=true;   shift ;;
        --skip-viz)       SKIP_VIZ=true; shift ;;
        --skip-sig)       SKIP_SIG=true; shift ;;
        --only-model)     IFS=',' read -r -a tmp <<< "$2"; ONLY_MODELS+=("${tmp[@]}"); shift 2 ;;
        --only-noise)     IFS=',' read -r -a tmp <<< "$2"; ONLY_NOISES+=("${tmp[@]}"); shift 2 ;;
        --help|-h)        print_help ;;
        *) echo "[WARN] 未知参数: $1，忽略" >&2; shift ;;
    esac
done

# 筛选模型 / 噪声
if [[ ${#ONLY_MODELS[@]} -eq 0 ]]; then MODELS=("${ALL_MODELS[@]}"); else MODELS=("${ONLY_MODELS[@]}"); fi
if [[ ${#ONLY_NOISES[@]} -eq 0 ]]; then NOISES=("${ALL_NOISES[@]}"); else NOISES=("${ONLY_NOISES[@]}"); fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
banner() { echo; echo "==============================================================================="; log "$*"; echo "==============================================================================="; }

banner "项目根目录: ${PROJECT_ROOT}"
log "数据集:  ${DATA_PATH}"
log "A 线:    ${LINE_A_OUT}"
log "B 线:    ${LINE_B_ROOT}/{model}/{noise}"
log "模型:    ${MODELS[*]}"
log "噪声:    ${NOISES[*]}"

[[ -f "${DATA_PATH}" ]] || { log "[ERROR] unified_zuco.pkl 不存在: ${DATA_PATH}"; exit 1; }

# ============================================================================
# Step A：诊断线 A
# ============================================================================
if [[ "${SKIP_A}" == true ]]; then
    banner "Step A 跳过（--skip-a）"
else
    banner "Step A - 诊断线 A: validate_eeg_signal.py"
    mkdir -p "${LINE_A_OUT}"
    python3 benchmark_eval/scripts/diagnostics/validate_eeg_signal.py \
        --data-path "${DATA_PATH}" \
        --output-dir "${LINE_A_OUT}" 2>&1 | tee "${LINE_A_OUT}/run_all.log"
    A_STATUS=${PIPESTATUS[0]}
    if [[ ${A_STATUS} -ne 0 ]]; then
        log "[ERROR] Step A 失败 (exit=${A_STATUS})，但继续下一步"
    else
        log "Step A 完成"
    fi
fi

# ============================================================================
# Step B：4 模型 × 4 噪声条件检索
# ============================================================================
if [[ "${SKIP_B}" == true ]]; then
    banner "Step B 跳过（--skip-b）"
else
    banner "Step B - 检索评估（${#MODELS[@]} 模型 × ${#NOISES[@]} 噪声）"
    for model in "${MODELS[@]}"; do
        case "${model}" in
            cet_mae)     SCRIPT="benchmark_eval/scripts/retrieval/run_cet_mae_retrieval.py";     CKPT="${CKPT_CET_MAE}";     EXTRA="" ;;
            eeg_to_text) SCRIPT="benchmark_eval/scripts/retrieval/run_eeg_to_text_retrieval.py"; CKPT="${CKPT_EEG_TO_TEXT}"; EXTRA="" ;;
            eeg2text)    SCRIPT="benchmark_eval/scripts/retrieval/run_eeg2text_retrieval.py";    CKPT="${CKPT_EEG2TEXT}";    EXTRA="" ;;
            glim)        SCRIPT="benchmark_eval/scripts/retrieval/run_glim_retrieval.py";        CKPT="${CKPT_GLIM}";        EXTRA="" ;;
            *)           log "[WARN] 未知模型 ${model}，跳过"; continue ;;
        esac
        if [[ ! -f "${CKPT}" ]]; then
            log "[ERROR] ${model} checkpoint 不存在: ${CKPT}，跳过"
            continue
        fi
        for noise in "${NOISES[@]}"; do
            OUTDIR="${LINE_B_ROOT}/${model}/${noise}"
            mkdir -p "${OUTDIR}"
            banner "[B] ${model} / ${noise} → ${OUTDIR}"
            TRANSFORMERS_OFFLINE=1 python3 "${SCRIPT}" \
                --data-path "${DATA_PATH}" \
                --model-checkpoint "${CKPT}" \
                --output-dir "${OUTDIR}" \
                --noise-type "${noise}" \
                ${EXTRA} 2>&1 | tee "${OUTDIR}/run.log"
            RC=${PIPESTATUS[0]}
            if [[ ${RC} -ne 0 ]]; then
                log "[ERROR] ${model}/${noise} 失败 (exit=${RC})，继续下一组"
            else
                log "[B] ${model}/${noise} 完成"
            fi
        done
    done
    banner "Step B 全部完成"
fi

# ============================================================================
# Step V：B 线降维可视化
# ============================================================================
if [[ "${SKIP_VIZ}" == true ]]; then
    banner "Step V 跳过（--skip-viz）"
else
    banner "Step V - 降维可视化 visualize_b_embeddings.py"
    # 跨模型 V3
    python3 benchmark_eval/scripts/analysis/visualize_b_embeddings.py \
        --results-dir "${RESULTS_DIR}" --viz v3 2>&1 | tee "${LINE_B_ROOT}/viz_v3.log"
    for model in "${MODELS[@]}"; do
        banner "[V] 单模型可视化 - ${model}"
        python3 benchmark_eval/scripts/analysis/visualize_b_embeddings.py \
            --results-dir "${RESULTS_DIR}" --model "${model}" --viz all \
            2>&1 | tee "${LINE_B_ROOT}/viz_${model}.log"
    done
    banner "Step V 完成"
fi

# ============================================================================
# Step S：B 线显著性检验
# ============================================================================
if [[ "${SKIP_SIG}" == true ]]; then
    banner "Step S 跳过（--skip-sig）"
else
    banner "Step S - 显著性检验 run_significance_tests.py"
    python3 benchmark_eval/scripts/analysis/run_significance_tests.py \
        --results-dir "${RESULTS_DIR}" 2>&1 | tee "${LINE_B_ROOT}/significance.log"
    banner "Step S 完成"
fi

banner "全流程完成"
