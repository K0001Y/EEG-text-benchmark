"""统一数据集验证脚本。

用于验证统一数据集的完整性和质量。
"""

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "..")
from utils.logging_utils import setup_logging, get_logger


def validate_unified_dataset(data_path: str, logger=None) -> Dict[str, Any]:
    """验证统一数据集的完整性和质量。

    Args:
        data_path: 统一数据集 pickle 文件路径
        logger: 日志记录器

    Returns:
        验证结果统计字典
    """
    if logger is None:
        logger = get_logger("validate_dataset")

    logger.info("Loading dataset from %s", data_path)

    with open(data_path, "rb") as f:
        samples = pickle.load(f)

    stats = {
        "total_samples": len(samples),
        "by_task": Counter(),
        "by_phase": Counter(),
        "by_subject": Counter(),
        "by_dataset": Counter(),
        "eeg_shape": None,
        "nan_count": 0,
        "inf_count": 0,
        "missing_text": 0,
        "missing_meta": 0,
        "phase_leakage": False,
        "text_uid_issues": [],
    }

    if not samples:
        logger.error("Dataset is empty!")
        return stats

    # 检查第一个样本的形状
    first_eeg = samples[0].get("eeg")
    if first_eeg is not None:
        stats["eeg_shape"] = np.array(first_eeg).shape
        logger.info("EEG shape: %s", stats["eeg_shape"])

    # 按 text_uid 分组检查
    text_uid_groups = defaultdict(list)

    for i, sample in enumerate(samples):
        # 检查必需字段
        if "eeg" not in sample or sample["eeg"] is None:
            logger.warning("Sample %d: missing EEG data", i)
            continue

        if "input_text" not in sample or not sample["input_text"]:
            stats["missing_text"] += 1
            logger.warning("Sample %d: missing input_text", i)

        if "meta" not in sample:
            stats["missing_meta"] += 1
            logger.warning("Sample %d: missing meta", i)
            continue

        meta = sample["meta"]

        # 统计分布
        task = meta.get("task", "unknown")
        phase = sample.get("phase", "unknown")
        subject = meta.get("subject", "unknown")
        dataset = meta.get("dataset", "unknown")

        stats["by_task"][task] += 1
        stats["by_phase"][phase] += 1
        stats["by_subject"][subject] += 1
        stats["by_dataset"][dataset] += 1

        # 检查 NaN/Inf
        eeg_array = np.array(sample["eeg"])
        if np.isnan(eeg_array).any():
            stats["nan_count"] += 1
        if np.isinf(eeg_array).any():
            stats["inf_count"] += 1

        # 按 text_uid 分组
        text = sample.get("input_text", "")
        text_uid_groups[text].append({
            "idx": i,
            "phase": phase,
            "subject": subject,
            "task": task,
        })

    # 检查数据泄露：同一句子的不同记录是否出现在不同 phase
    for text, records in text_uid_groups.items():
        phases = set(r["phase"] for r in records)
        if len(phases) > 1:
            stats["phase_leakage"] = True
            stats["text_uid_issues"].append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "phases": list(phases),
                "count": len(records),
            })

    # 输出统计结果
    logger.info("=" * 60)
    logger.info("Dataset Validation Report")
    logger.info("=" * 60)
    logger.info("Total samples: %d", stats["total_samples"])
    logger.info("Samples with NaN: %d", stats["nan_count"])
    logger.info("Samples with Inf: %d", stats["inf_count"])
    logger.info("Samples missing text: %d", stats["missing_text"])
    logger.info("Samples missing meta: %d", stats["missing_meta"])
    logger.info("-" * 60)
    logger.info("By Task: %s", dict(stats["by_task"]))
    logger.info("By Phase: %s", dict(stats["by_phase"]))
    logger.info("By Dataset: %s", dict(stats["by_dataset"]))
    logger.info("By Subject: %s", dict(stats["by_subject"]))
    logger.info("-" * 60)

    if stats["phase_leakage"]:
        logger.error("PHASE LEAKAGE DETECTED!")
        logger.error("Found %d texts appearing in multiple phases", len(stats["text_uid_issues"]))
        for issue in stats["text_uid_issues"][:5]:  # 只显示前5个
            logger.error("  Text: %s, Phases: %s, Count: %d", issue["text"], issue["phases"], issue["count"])
    else:
        logger.info("No phase leakage detected.")

    logger.info("=" * 60)

    return stats


def check_label_coverage(samples: List[Dict], logger=None) -> Dict[str, Any]:
    """检查标签覆盖情况。

    Args:
        samples: 样本列表
        logger: 日志记录器

    Returns:
        标签覆盖统计
    """
    if logger is None:
        logger = get_logger("validate_dataset")

    label_stats = {
        "task1_with_sentiment": 0,
        "task1_without_sentiment": 0,
        "task2_with_relation": 0,
        "task2_without_relation": 0,
        "task3_with_relation": 0,
        "task3_without_relation": 0,
    }

    for sample in samples:
        meta = sample.get("meta", {})
        task = meta.get("task", "")

        if task == "task1-SR":
            if "sentiment_label" in meta:
                label_stats["task1_with_sentiment"] += 1
            else:
                label_stats["task1_without_sentiment"] += 1

        elif task == "task2-NR":
            if "relation_label" in meta:
                label_stats["task2_with_relation"] += 1
            else:
                label_stats["task2_without_relation"] += 1

        elif task == "task3-TSR":
            if "relation_label" in meta:
                label_stats["task3_with_relation"] += 1
            else:
                label_stats["task3_without_relation"] += 1

    logger.info("Label Coverage:")
    logger.info("  Task1 (SR) with sentiment_label: %d / %d",
                label_stats["task1_with_sentiment"],
                label_stats["task1_with_sentiment"] + label_stats["task1_without_sentiment"])
    logger.info("  Task2 (NR) with relation_label: %d / %d",
                label_stats["task2_with_relation"],
                label_stats["task2_with_relation"] + label_stats["task2_without_relation"])
    logger.info("  Task3 (TSR) with relation_label: %d / %d",
                label_stats["task3_with_relation"],
                label_stats["task3_with_relation"] + label_stats["task3_without_relation"])

    return label_stats


def main():
    parser = argparse.ArgumentParser(description="Validate unified EEG-text dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to unified dataset pickle file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save validation report")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger = get_logger("validate_dataset")

    # 验证数据集
    stats = validate_unified_dataset(args.data_path, logger)

    # 加载样本检查标签
    with open(args.data_path, "rb") as f:
        samples = pickle.load(f)
    check_label_coverage(samples, logger)

    # 保存验证报告
    import json
    report_path = f"{args.output_dir}/validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # 转换 Counter 为 dict 以便 JSON 序列化
        report = {
            k: (dict(v) if isinstance(v, Counter) else v)
            for k, v in stats.items()
        }
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Validation report saved to %s", report_path)

    # 返回退出码
    if stats["nan_count"] > 0 or stats["phase_leakage"]:
        logger.error("Validation FAILED!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
