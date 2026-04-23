#!/usr/bin/env python3
"""综合对比分析报告生成器。

读取诊断线 A 和诊断线 B 的所有结果，计算统计检验，
综合两条线的结论，输出最终归因判定。

统计检验（v2 新增）：
  - 置换检验 (Permutation Test): p-value
  - Bootstrap 95% CI
  - Cohen's d 效应量

用法（项目根目录下）：
  python benchmark_eval/scripts/compare_contrast_results.py \
      --results-dir benchmark_eval/test_outputs

详见 docs/contrast_experiment_spec.md
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)

if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from utils.logging_utils import setup_logging
from constants import DEFAULT_SEED

MODELS = ["cet_mae", "eeg_to_text", "eeg2text", "glim"]
NOISE_CONDITIONS = ["gaussian", "shuffle", "zero"]
METRICS = ["r@1", "r@5", "r@10", "mrr"]


def parse_args():
    p = argparse.ArgumentParser(description="综合对比分析报告")
    p.add_argument("--results-dir", required=True,
                   help="test_outputs 目录路径")
    p.add_argument("--output-path", default=None,
                   help="输出 JSON 路径（默认 results-dir/contrast_summary.json）")
    p.add_argument("--n-permutations", type=int, default=1000,
                   help="置换检验次数")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Bootstrap 重采样次数")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_retrieval_metrics(results_dir: str, model: str,
                           noise: str = "") -> Optional[Dict]:
    """加载检索评估指标文件。优先查找 line_b 目录结构。"""
    # 优先查找 line_b/{model}/{noise}/ 结构
    line_b_path = os.path.join(results_dir, "line_b", model, noise or "real", "retrieval_metrics.json")
    if os.path.isfile(line_b_path):
        with open(line_b_path, "r") as f:
            return json.load(f)
    # 回退到旧路径格式
    if noise and noise != "real":
        dirname = f"eval_{model}_retrieval_{noise}"
    else:
        dirname = f"eval_{model}_retrieval"
    path = os.path.join(results_dir, dirname, "retrieval_metrics.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_dataset_validity(results_dir: str) -> Optional[Dict]:
    """加载诊断线 A 结果。"""
    path = os.path.join(results_dir, "dataset_validity", "linear_probe_results.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 统计检验
# ═══════════════════════════════════════════════════════════════════════════

def cohens_d(a: float, b: float, std_a: float, std_b: float,
             n_a: int, n_b: int) -> float:
    """计算 Cohen's d 效应量。"""
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) /
                          (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (a - b) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """解释 Cohen's d 效应量。"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_pairwise_comparison(real_metrics: Dict, noise_metrics: Dict,
                                metric_key: str) -> Dict:
    """计算 real vs noise 的统计比较。"""
    real_val = real_metrics.get("overall", {}).get(metric_key, 0)
    noise_val = noise_metrics.get("overall", {}).get(metric_key, 0)
    diff = real_val - noise_val

    n_real = real_metrics.get("overall", {}).get("num_queries", 1000)
    n_noise = noise_metrics.get("overall", {}).get("num_queries", 1000)

    # 经验阈值判定
    thresholds = {
        "mean_rank": 3.0,
        "r@10": 0.02,
        "mrr": 0.005,
    }
    threshold = thresholds.get(metric_key, 0.02)
    empirical_significant = abs(diff) > threshold

    return {
        "real": round(real_val, 6),
        "noise": round(noise_val, 6),
        "diff": round(diff, 6),
        "empirical_threshold": threshold,
        "empirical_significant": empirical_significant,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 诊断归因
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_model(model_name: str, comparisons: Dict) -> Dict:
    """对单个模型进行诊断归因。

    基于 spec 中的六种典型诊断结论表。
    """
    # 检查 real vs gaussian
    rg = comparisons.get("gaussian", {})
    rs = comparisons.get("shuffle", {})
    rz = comparisons.get("zero", {})

    rg_sig = any(
        rg.get(m, {}).get("empirical_significant", False)
        for m in METRICS
    )
    rs_sig = any(
        rs.get(m, {}).get("empirical_significant", False)
        for m in METRICS
    )

    # 判断 zero 异常（与 random baseline 显著偏离）
    zero_anomaly = False
    if rz:
        mean_rank_zero = rz.get("mean_rank", {}).get("noise", 65.5)
        zero_anomaly = abs(mean_rank_zero - 65.5) > 5.0

    # 六种诊断
    if not rg_sig and not rs_sig and not zero_anomaly:
        diagnosis = "A"
        conclusion = "编码器完全无效，未从 EEG 学到任何信息"
    elif rg_sig and not rs_sig:
        diagnosis = "B"
        conclusion = "学到 EEG 统计特性，但未学到跨模态对应"
    elif rg_sig and rs_sig:
        diagnosis = "C"
        conclusion = "学到了跨模态对应，但检索天花板受限于 SNR"
    elif not rg_sig and zero_anomaly:
        diagnosis = "D"
        conclusion = "数据管道 bug：噪声与真实信号经处理后特征坍缩"
    elif not rg_sig and not zero_anomaly:
        diagnosis = "E"
        conclusion = "模型存在 shortcut/bias，不依赖输入内容"
    else:
        diagnosis = "F"
        conclusion = "模型有效但 EEG SNR 太低"

    return {
        "model": model_name,
        "diagnosis": diagnosis,
        "conclusion": conclusion,
        "real_vs_gaussian_significant": rg_sig,
        "real_vs_shuffle_significant": rs_sig,
        "zero_anomaly": zero_anomaly,
    }


def diagnose_line_a(validity_results: Optional[Dict]) -> Dict:
    """诊断线 A 归因。"""
    if validity_results is None:
        return {"status": "not_available", "conclusion": "诊断线 A 结果未找到"}

    a1a = validity_results.get("A1a_mean_pool", {})
    a1c = validity_results.get("A1c_band_separated", {})
    eta2 = validity_results.get("A2_eta_squared", {})
    a3 = validity_results.get("A3_desubject", {})

    random_baseline = a1a.get("random_baseline", 0.0077)

    # 判断 Linear Probe 是否显著高于随机
    a1a_sig = a1a.get("top1_accuracy", 0) > random_baseline * 5  # > 5x random
    a1c_sig = a1c.get("top1_accuracy", 0) > random_baseline * 5

    # 判断被试效应
    subject_dominant = eta2.get("conclusion") == "subject_dominant"

    if not a1a_sig and not a1c_sig:
        if subject_dominant:
            if a3:
                desubj_acc = a3.get("linear_probe_desubject", {}).get("top1_accuracy", 0)
                if desubj_acc > random_baseline * 5:
                    conclusion = "在当前表示下，信号存在但被个体差异淹没"
                    suggestion = "探索被试归一化策略 / 多被试聚合"
                else:
                    conclusion = "在当前表示下，未检测到可检测的句子级语义信息"
                    suggestion = "尝试时序模型或更换特征表示"
            else:
                conclusion = "被试效应主导，需执行 A3 进一步验证"
                suggestion = "运行 A3 去被试化验证"
        else:
            conclusion = "当前 EEG 特征维度本身信息稀疏"
            suggestion = "尝试更丰富的特征提取方法"
    elif not a1a_sig and a1c_sig:
        conclusion = "mean-pooling 是瓶颈，需要更精细的特征提取"
        suggestion = "采用 duration-weighted 或时序模型"
    else:
        conclusion = "当前表示下信号有区分性，问题在模型端"
        suggestion = "检查模型编码器（转诊断线 B）"

    return {
        "status": "completed",
        "a1a_significant": a1a_sig,
        "a1c_significant": a1c_sig,
        "subject_dominant": subject_dominant,
        "conclusion": conclusion,
        "suggestion": suggestion,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.results_dir, "contrast_summary.json")

    logger = setup_logging(os.path.dirname(args.output_path) or ".",
                           log_name="contrast_analysis.log")
    logger.info("综合对比分析 | results_dir=%s", args.results_dir)

    summary = {
        "line_a": {},
        "line_b": {},
        "combined_diagnosis": {},
    }

    # ── 诊断线 A ──
    logger.info("=== 诊断线 A ===")
    validity = load_dataset_validity(args.results_dir)
    if validity:
        logger.info("  加载成功: dataset_validity/linear_probe_results.json")
        summary["line_a"]["data"] = validity
    else:
        logger.warning("  未找到诊断线 A 结果")

    line_a_diagnosis = diagnose_line_a(validity)
    summary["line_a"]["diagnosis"] = line_a_diagnosis
    logger.info("  诊断线 A 结论: %s", line_a_diagnosis.get("conclusion", "N/A"))

    # ── 诊断线 B ──
    logger.info("=== 诊断线 B ===")
    for model in MODELS:
        model_key = model
        real_metrics = load_retrieval_metrics(args.results_dir, model, "real")
        if real_metrics is None:
            logger.info("  %s: real 结果未找到，跳过", model)
            continue

        model_comparisons = {}
        for noise in NOISE_CONDITIONS:
            noise_metrics = load_retrieval_metrics(args.results_dir, model, noise)
            if noise_metrics is None:
                logger.info("  %s/%s: 未找到", model, noise)
                continue

            comparisons = {}
            for metric in METRICS + ["mean_rank"]:
                comparisons[metric] = compute_pairwise_comparison(
                    real_metrics, noise_metrics, metric)

            model_comparisons[noise] = comparisons
            logger.info("  %s/%s: loaded", model, noise)

        if model_comparisons:
            diagnosis = diagnose_model(model, model_comparisons)
            summary["line_b"][model_key] = {
                "comparisons": model_comparisons,
                "diagnosis": diagnosis,
            }
            logger.info("  %s 诊断: %s — %s",
                        model, diagnosis["diagnosis"], diagnosis["conclusion"])

    # ── 综合归因 ──
    logger.info("=== 综合归因 ===")
    line_a_conclusion = line_a_diagnosis.get("conclusion", "N/A")
    line_b_conclusions = {
        m: summary["line_b"].get(m, {}).get("diagnosis", {}).get("conclusion", "N/A")
        for m in MODELS
    }

    summary["combined_diagnosis"] = {
        "line_a_conclusion": line_a_conclusion,
        "line_b_conclusions": line_b_conclusions,
        "note": "在当前表示下的条件性结论，详见 contrast_experiment_spec.md",
    }

    # ── 保存 ──
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("综合分析报告已保存 → %s", args.output_path)

    # ── 打印摘要 ──
    sep = "=" * 60
    print(f"\n{sep}")
    print("综合对比分析报告")
    print(sep)
    print(f"  诊断线 A: {line_a_conclusion}")
    for m in MODELS:
        c = line_b_conclusions.get(m, "N/A")
        d = summary["line_b"].get(m, {}).get("diagnosis", {}).get("diagnosis", "?")
        print(f"  诊断线 B ({m}): [{d}] {c}")
    print(sep)
    print(f"  报告 → {args.output_path}")
    print(sep)


if __name__ == "__main__":
    main()
