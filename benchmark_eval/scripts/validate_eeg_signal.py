#!/usr/bin/env python3
"""诊断线 A：原始数据集有效性验证。

在当前特征表示下，从数据层面验证 ZuCo EEG 是否包含可检测的句子级语义信息。
包含以下实验：
  A1a: Linear Probe — Mean-Pool 基线
  A1b: Linear Probe — Duration-Weighted Pool
  A1c: Linear Probe — Band-Separated
  A2:  被试效应 vs 句子效应分析（余弦相似度分组 + η² 方差分解 + t-SNE）
  A2-band: 频带级 η² 分析
  A3:  去被试化信号恢复验证（条件执行：仅在 A2 确认被试效应显著时执行）

纯 CPU 运行，依赖 sklearn / scipy / matplotlib。

用法（项目根目录下）：
  python benchmark_eval/scripts/validate_eeg_signal.py \
      --data-path benchmark_eval/data/unified_zuco.pkl

详见 docs/contrast_experiment_spec.md（v2 修订版）
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)
PROJ_ROOT = os.path.dirname(BENCH_DIR)

if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from data_processing.dataset import UnifiedDataset
from utils.logging_utils import setup_logging
from constants import EEG_CHANNELS, EEG_BANDS, EEG_WORD_DIM, DEFAULT_SEED

# 频带名称
BAND_NAMES = ["theta1", "theta2", "alpha1", "alpha2",
              "beta1", "beta2", "gamma1", "gamma2"]


def parse_args():
    p = argparse.ArgumentParser(description="诊断线 A：EEG 数据有效性验证")
    p.add_argument("--data-path", required=True, help="unified_zuco.pkl 路径")
    p.add_argument("--output-dir", default=None,
                   help="输出目录（默认 benchmark_eval/test_outputs/dataset_validity）")
    p.add_argument("--skip-a3", action="store_true", help="跳过 A3 去被试化验证")
    p.add_argument("--skip-tsne", action="store_true", help="跳过 t-SNE 可视化（节省时间）")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════════════════

def collect_samples(data_path: str, phase: str = "test"):
    """从 unified_zuco.pkl 加载样本，提取 EEG 特征和元数据。"""
    ds = UnifiedDataset(data_path, phase=phase)
    eeg_list, text_list, subject_list, task_list, sentence_id_list = [], [], [], [], []

    # 构建句子文本 → 唯一 ID 映射
    unique_texts = []
    text_to_id = {}

    for i in range(len(ds)):
        sample = ds[i]
        # 获取 EEG（优先 eeg_word_norm1d）
        eeg_key = "eeg_word_norm1d" if "eeg_word_norm1d" in sample else "eeg"
        eeg = sample[eeg_key].numpy()  # (MAX_LEN, 840)
        mask_key = "mask_word" if "mask_word" in sample else "mask"
        mask = sample[mask_key].numpy()  # (MAX_LEN,)

        text = sample["reference_text"]
        meta = sample.get("meta", {})
        subject = meta.get("subject", "unknown")
        task = meta.get("task", "unknown")

        if text not in text_to_id:
            text_to_id[text] = len(unique_texts)
            unique_texts.append(text)

        eeg_list.append((eeg, mask))
        text_list.append(text)
        subject_list.append(subject)
        task_list.append(task)
        sentence_id_list.append(text_to_id[text])

    return {
        "eeg_list": eeg_list,
        "text_list": text_list,
        "subject_list": subject_list,
        "task_list": task_list,
        "sentence_id_list": sentence_id_list,
        "unique_texts": unique_texts,
        "n_classes": len(unique_texts),
    }


def collect_train_samples(data_path: str):
    """加载 train 集样本。"""
    return collect_samples(data_path, phase="train")


def extract_features(eeg_list, variant="mean_pool"):
    """从 EEG 词级序列提取句级特征向量。

    Args:
        eeg_list: [(eeg, mask), ...], eeg shape (MAX_LEN, 840), mask shape (MAX_LEN,)
        variant: "mean_pool" | "band_separated"

    Returns:
        features: np.ndarray, shape (N, 840)
    """
    feats = []
    for eeg, mask in eeg_list:
        valid_len = int(mask.sum())
        if valid_len == 0:
            valid_len = 1
        eeg_valid = eeg[:valid_len]  # (valid_len, 840)

        if variant == "mean_pool":
            feat = eeg_valid.mean(axis=0)  # (840,)
        elif variant == "band_separated":
            # 按频带结构保持：reshape → (valid_len, 8, 105) → mean → flatten
            reshaped = eeg_valid.reshape(valid_len, EEG_BANDS, EEG_CHANNELS)
            feat = reshaped.mean(axis=0).flatten()  # (840,)
        else:
            feat = eeg_valid.mean(axis=0)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# A1: Linear Probe（多版本）
# ═══════════════════════════════════════════════════════════════════════════

def run_linear_probe(train_feats, train_labels, test_feats, test_labels,
                     n_classes, variant_name, logger):
    """运行 Linear Probe 实验：sklearn LogisticRegression 130 类分类。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    logger.info("=== A1 Linear Probe: %s ===", variant_name)
    logger.info("  train: %d samples, test: %d samples, %d classes",
                len(train_feats), len(test_feats), n_classes)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats)
    X_test = scaler.transform(test_feats)

    # 逻辑回归（无非线性）
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs",
        random_state=DEFAULT_SEED, n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    # 预测概率
    proba = clf.predict_proba(X_test)  # (N, n_classes)
    predictions = clf.predict(X_test)

    # 计算 Top-1 / Top-5 / Top-10 accuracy
    top1 = float(np.mean(predictions == test_labels))

    top5, top10 = 0.0, 0.0
    for i, true_label in enumerate(test_labels):
        sorted_classes = np.argsort(proba[i])[::-1]
        if true_label in sorted_classes[:5]:
            top5 += 1
        if true_label in sorted_classes[:10]:
            top10 += 1
    top5 /= len(test_labels)
    top10 /= len(test_labels)

    random_baseline = 1.0 / n_classes

    result = {
        "variant": variant_name,
        "top1_accuracy": round(top1, 6),
        "top5_accuracy": round(top5, 6),
        "top10_accuracy": round(top10, 6),
        "random_baseline": round(random_baseline, 6),
        "n_train": len(train_feats),
        "n_test": len(test_feats),
        "n_classes": n_classes,
    }

    logger.info("  Top-1: %.4f%% (random: %.4f%%)", top1 * 100, random_baseline * 100)
    logger.info("  Top-5: %.4f%%", top5 * 100)
    logger.info("  Top-10: %.4f%%", top10 * 100)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# A2: 被试效应 vs 句子效应分析
# ═══════════════════════════════════════════════════════════════════════════

def run_cosine_similarity_analysis(features, sentence_ids, subject_ids, task_ids, logger):
    """余弦相似度分组对比（v2 修正版）。

    分三组计算余弦相似度：
      - 同句异被试：反映句子语义效应
      - 同被试异句：反映被试个体特征
      - 异句异被试：基线
    """
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("=== A2: 余弦相似度分组对比 ===")
    n = len(features)

    # 计算全体余弦相似度矩阵
    cos_matrix = cosine_similarity(features)  # (N, N)

    same_sent_diff_subj = []
    same_subj_diff_sent = []
    diff_sent_diff_subj = []

    for i in range(n):
        for j in range(i + 1, n):
            cos_val = cos_matrix[i, j]
            same_sent = (sentence_ids[i] == sentence_ids[j])
            same_subj = (subject_ids[i] == subject_ids[j])

            if same_sent and not same_subj:
                same_sent_diff_subj.append(cos_val)
            elif same_subj and not same_sent:
                same_subj_diff_sent.append(cos_val)
            elif not same_sent and not same_subj:
                diff_sent_diff_subj.append(cos_val)

    def _stats(arr, name):
        a = np.array(arr)
        s = {
            "count": len(a),
            "mean": round(float(a.mean()), 6) if len(a) > 0 else None,
            "std": round(float(a.std()), 6) if len(a) > 0 else None,
            "median": round(float(np.median(a)), 6) if len(a) > 0 else None,
        }
        logger.info("  %s: n=%d, mean=%.6f, std=%.6f",
                     name, s["count"], s["mean"] or 0, s["std"] or 0)
        return s

    result = {
        "same_sent_diff_subj": _stats(same_sent_diff_subj, "同句异被试"),
        "same_subj_diff_sent": _stats(same_subj_diff_sent, "同被试异句"),
        "diff_sent_diff_subj": _stats(diff_sent_diff_subj, "异句异被试"),
    }

    # 判定排序
    means = {
        "same_sent_diff_subj": result["same_sent_diff_subj"]["mean"] or 0,
        "same_subj_diff_sent": result["same_subj_diff_sent"]["mean"] or 0,
        "diff_sent_diff_subj": result["diff_sent_diff_subj"]["mean"] or 0,
    }
    sorted_groups = sorted(means.items(), key=lambda x: x[1], reverse=True)
    result["ranking"] = [g[0] for g in sorted_groups]
    logger.info("  排序: %s", " > ".join(
        f"{g[0]}({g[1]:.4f})" for g in sorted_groups))

    return result


def run_eta_squared_analysis(features, sentence_ids, subject_ids, logger):
    """方差分解：η² 分析（v2 修正版）。

    对 EEG 特征的每个维度做 two-way ANOVA 变体（手动计算 η²）。
    """
    logger.info("=== A2: η² 方差分解 ===")

    n_features = features.shape[1]
    eta2_sentence = []
    eta2_subject = []

    # 编码因子
    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)

    unique_sent = np.unique(sent_arr)
    unique_subj = np.unique(subj_arr)
    grand_mean_all = features.mean(axis=0)  # (D,)

    for d in range(n_features):
        y = features[:, d]
        grand_mean = y.mean()
        ss_total = np.sum((y - grand_mean) ** 2)

        if ss_total < 1e-12:
            eta2_sentence.append(0.0)
            eta2_subject.append(0.0)
            continue

        # SS_sentence
        ss_sent = 0.0
        for s in unique_sent:
            mask = sent_arr == s
            n_s = mask.sum()
            if n_s > 0:
                ss_sent += n_s * (y[mask].mean() - grand_mean) ** 2

        # SS_subject
        ss_subj = 0.0
        for s in unique_subj:
            mask = subj_arr == s
            n_s = mask.sum()
            if n_s > 0:
                ss_subj += n_s * (y[mask].mean() - grand_mean) ** 2

        eta2_sentence.append(ss_sent / ss_total)
        eta2_subject.append(ss_subj / ss_total)

    eta2_sent_arr = np.array(eta2_sentence)
    eta2_subj_arr = np.array(eta2_subject)

    result = {
        "eta2_sentence": {
            "mean": round(float(eta2_sent_arr.mean()), 6),
            "median": round(float(np.median(eta2_sent_arr)), 6),
            "std": round(float(eta2_sent_arr.std()), 6),
        },
        "eta2_subject": {
            "mean": round(float(eta2_subj_arr.mean()), 6),
            "median": round(float(np.median(eta2_subj_arr)), 6),
            "std": round(float(eta2_subj_arr.std()), 6),
        },
        "n_features": n_features,
        "n_sentences": len(unique_sent),
        "n_subjects": len(unique_subj),
    }

    logger.info("  η²(句子): mean=%.6f, median=%.6f",
                result["eta2_sentence"]["mean"], result["eta2_sentence"]["median"])
    logger.info("  η²(被试): mean=%.6f, median=%.6f",
                result["eta2_subject"]["mean"], result["eta2_subject"]["median"])

    # 判定
    ratio = result["eta2_subject"]["median"] / max(result["eta2_sentence"]["median"], 1e-12)
    if ratio > 3:
        result["conclusion"] = "subject_dominant"
        logger.info("  结论: 被试效应主导 (η²_subj/η²_sent = %.1f)", ratio)
    elif ratio > 0.5:
        result["conclusion"] = "comparable"
        logger.info("  结论: 两者效应相当")
    else:
        result["conclusion"] = "sentence_dominant"
        logger.info("  结论: 句子效应主导")

    return result


def run_band_level_eta_squared(features, sentence_ids, subject_ids, logger):
    """A2-band：频带级 η² 分析（v2 新增）。

    对 8 个频带分别计算 η²_sentence 和 η²_subject。
    """
    logger.info("=== A2-band: 频带级 η² 分析 ===")

    n_samples = features.shape[0]
    # 重塑为 (N, 8, 105)
    reshaped = features.reshape(n_samples, EEG_BANDS, EEG_CHANNELS)

    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)
    unique_sent = np.unique(sent_arr)
    unique_subj = np.unique(subj_arr)

    band_results = {}
    for b, band_name in enumerate(BAND_NAMES):
        band_feat = reshaped[:, b, :]  # (N, 105)
        eta2_sent_list, eta2_subj_list = [], []

        for d in range(EEG_CHANNELS):
            y = band_feat[:, d]
            grand_mean = y.mean()
            ss_total = np.sum((y - grand_mean) ** 2)

            if ss_total < 1e-12:
                eta2_sent_list.append(0.0)
                eta2_subj_list.append(0.0)
                continue

            ss_sent = sum(
                m.sum() * (y[m].mean() - grand_mean) ** 2
                for s in unique_sent if (m := (sent_arr == s)).sum() > 0
            )
            ss_subj = sum(
                m.sum() * (y[m].mean() - grand_mean) ** 2
                for s in unique_subj if (m := (subj_arr == s)).sum() > 0
            )

            eta2_sent_list.append(ss_sent / ss_total)
            eta2_subj_list.append(ss_subj / ss_total)

        band_results[band_name] = {
            "eta2_sentence_median": round(float(np.median(eta2_sent_list)), 6),
            "eta2_subject_median": round(float(np.median(eta2_subj_list)), 6),
            "eta2_sentence_mean": round(float(np.mean(eta2_sent_list)), 6),
            "eta2_subject_mean": round(float(np.mean(eta2_subj_list)), 6),
        }
        logger.info("  %s: η²_sent=%.6f, η²_subj=%.6f (median)",
                     band_name,
                     band_results[band_name]["eta2_sentence_median"],
                     band_results[band_name]["eta2_subject_median"])

    return band_results


def run_tsne_visualization(features, sentence_ids, subject_ids, task_ids,
                           output_dir, logger):
    """t-SNE 降维可视化（v2 增强：多 perplexity）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib 或 sklearn 未安装，跳过 t-SNE 可视化")
        return

    logger.info("=== A2: t-SNE 可视化 ===")

    # PCA 降维到 50 维
    pca = PCA(n_components=min(50, features.shape[1]), random_state=DEFAULT_SEED)
    pca_feats = pca.fit_transform(features)
    logger.info("  PCA: %d → %d 维（解释方差: %.2f%%）",
                features.shape[1], pca_feats.shape[1],
                pca.explained_variance_ratio_.sum() * 100)

    perplexities = [5, 30, 50]
    color_configs = [
        ("subject", subject_ids, "按被试 ID 着色"),
        ("sentence", sentence_ids, "按句子 ID 着色"),
        ("task", task_ids, "按 task 着色"),
    ]

    for perp in perplexities:
        logger.info("  t-SNE perplexity=%d ...", perp)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=DEFAULT_SEED,
                     n_iter=1000)
        tsne_feats = tsne.fit_transform(pca_feats)

        for color_name, color_ids, title_suffix in color_configs:
            # 仅为默认 perplexity=30 输出所有着色方案
            # 其他 perplexity 仅输出按被试着色
            if perp != 30 and color_name != "subject":
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            unique_ids = sorted(set(color_ids))

            # 限制颜色数量
            if len(unique_ids) > 20:
                # 太多类别时只用前 20 种颜色
                cmap = plt.cm.get_cmap("tab20", min(len(unique_ids), 20))
            else:
                cmap = plt.cm.get_cmap("tab20", len(unique_ids))

            for idx, uid in enumerate(unique_ids):
                mask = [cid == uid for cid in color_ids]
                pts = tsne_feats[mask]
                c = cmap(idx % 20)
                label = str(uid) if len(unique_ids) <= 20 else None
                ax.scatter(pts[:, 0], pts[:, 1], c=[c], s=5, alpha=0.6, label=label)

            ax.set_title(f"t-SNE (perplexity={perp}) - {title_suffix}")
            if len(unique_ids) <= 20:
                ax.legend(fontsize=6, markerscale=3, loc="best")

            fname = f"tsne_by_{color_name}_p{perp}.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("    → %s", fname)


# ═══════════════════════════════════════════════════════════════════════════
# A3: 去被试化信号恢复验证
# ═══════════════════════════════════════════════════════════════════════════

def run_desubject_analysis(data_path, output_dir, logger):
    """A3：去被试化验证（v2 修正数据泄漏）。

    步骤 1：在 train 集计算 per-subject μ/σ（严禁使用 test 集）
    步骤 2：应用 z-score 归一化
    步骤 3：重新运行 Linear Probe
    另：被试聚合检索（同句多被试平均 → 130×130 方阵检索）
    """
    logger.info("=== A3: 去被试化验证 ===")

    # 加载 train 和 test
    train_data = collect_samples(data_path, phase="train")
    test_data = collect_samples(data_path, phase="test")

    train_feats = extract_features(train_data["eeg_list"], "mean_pool")
    test_feats = extract_features(test_data["eeg_list"], "mean_pool")

    # 步骤 1：在 train 集计算 per-subject μ/σ
    train_subjects = train_data["subject_list"]
    unique_subjects = sorted(set(train_subjects))
    subject_stats = {}

    for subj in unique_subjects:
        mask = [s == subj for s in train_subjects]
        subj_feats = train_feats[mask]
        subject_stats[subj] = {
            "mean": subj_feats.mean(axis=0),
            "std": subj_feats.std(axis=0),
        }

    # 步骤 2：归一化
    def normalize(feats, subjects):
        normed = np.zeros_like(feats)
        for i, subj in enumerate(subjects):
            if subj in subject_stats:
                mu = subject_stats[subj]["mean"]
                sigma = subject_stats[subj]["std"]
                sigma = np.where(sigma < 1e-8, 1.0, sigma)
                normed[i] = (feats[i] - mu) / sigma
            else:
                normed[i] = feats[i]  # 未知被试不归一化
        return normed

    train_normed = normalize(train_feats, train_data["subject_list"])
    test_normed = normalize(test_feats, test_data["subject_list"])

    # 步骤 3：重新运行 Linear Probe
    lp_result = run_linear_probe(
        train_normed, np.array(train_data["sentence_id_list"]),
        test_normed, np.array(test_data["sentence_id_list"]),
        test_data["n_classes"], "A3_desubject_mean_pool", logger,
    )

    # 被试聚合检索（分组交叉验证）：
    # 将被试对半分为 group_A / group_B，各自聚合后做 EEG-vs-EEG 跨组检索。
    # 避免自检索（sim[i][i]=1 的平凡结果）。
    logger.info("--- A3: 被试聚合检索（分组交叉验证）---")
    n_classes = test_data["n_classes"]

    unique_subjects_test = sorted(set(test_data["subject_list"]))
    half = max(1, len(unique_subjects_test) // 2)
    group_a_subj = set(unique_subjects_test[:half])
    group_b_subj = set(unique_subjects_test[half:])
    logger.info("  group_A: %d subjects, group_B: %d subjects",
                len(group_a_subj), len(group_b_subj))

    agg_a: dict = {}
    agg_b: dict = {}
    for i, sid in enumerate(test_data["sentence_id_list"]):
        subj = test_data["subject_list"][i]
        if subj in group_a_subj:
            agg_a.setdefault(sid, []).append(test_feats[i])
        else:
            agg_b.setdefault(sid, []).append(test_feats[i])

    # 只保留两组都有数据的句子
    common_sids = sorted(set(agg_a.keys()) & set(agg_b.keys()))
    if len(common_sids) == 0:
        logger.warning("  两组无公共句子，跳过聚合检索")
        agg_result = {"error": "no_common_sentences"}
    else:
        vec_a = np.array([np.mean(agg_a[s], axis=0) for s in common_sids], dtype=np.float32)
        vec_b = np.array([np.mean(agg_b[s], axis=0) for s in common_sids], dtype=np.float32)

        # L2 归一化后余弦相似度
        vec_a /= np.linalg.norm(vec_a, axis=1, keepdims=True).clip(min=1e-8)
        vec_b /= np.linalg.norm(vec_b, axis=1, keepdims=True).clip(min=1e-8)
        sim = vec_a @ vec_b.T   # (M, M), query=group_A, candidate=group_B

        n_common = len(common_sids)
        ranks = []
        for i in range(n_common):
            order = np.argsort(sim[i])[::-1]
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks, dtype=np.float32)

        agg_result = {
            "n_sentences": n_common,
            "n_group_a_subjects": len(group_a_subj),
            "n_group_b_subjects": len(group_b_subj),
            "r@1":  round(float((ranks <= 1).mean()),  6),
            "r@5":  round(float((ranks <= 5).mean()),  6),
            "r@10": round(float((ranks <= 10).mean()), 6),
            "mrr":  round(float((1.0 / ranks).mean()), 6),
            "mean_rank":   round(float(ranks.mean()), 2),
            "median_rank": round(float(np.median(ranks)), 2),
        }
        logger.info("  聚合检索(分组): R@1=%.4f%%, MRR=%.4f, Mean Rank=%.1f",
                    agg_result["r@1"] * 100, agg_result["mrr"], agg_result["mean_rank"])

    return {
        "linear_probe_desubject": lp_result,
        "aggregated_retrieval": agg_result,
        "n_subjects_with_stats": len(subject_stats),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(BENCH_DIR, "test_outputs", "dataset_validity")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir, log_name="validate_eeg_signal.log")
    logger.info("诊断线 A：EEG 数据有效性验证")
    logger.info("Args: %s", vars(args))

    all_results = {}

    # ── 数据加载 ──
    logger.info("加载数据...")
    train_data = collect_samples(args.data_path, phase="train")
    test_data = collect_samples(args.data_path, phase="test")
    logger.info("Train: %d samples, Test: %d samples, %d unique sentences",
                len(train_data["eeg_list"]), len(test_data["eeg_list"]),
                test_data["n_classes"])

    # ── A1a: Mean-Pool Linear Probe ──
    train_feats_mp = extract_features(train_data["eeg_list"], "mean_pool")
    test_feats_mp = extract_features(test_data["eeg_list"], "mean_pool")
    lp_a1a = run_linear_probe(
        train_feats_mp, np.array(train_data["sentence_id_list"]),
        test_feats_mp, np.array(test_data["sentence_id_list"]),
        test_data["n_classes"], "A1a_mean_pool", logger,
    )
    all_results["A1a_mean_pool"] = lp_a1a

    # ── A1c: Band-Separated Linear Probe ──
    train_feats_bs = extract_features(train_data["eeg_list"], "band_separated")
    test_feats_bs = extract_features(test_data["eeg_list"], "band_separated")
    lp_a1c = run_linear_probe(
        train_feats_bs, np.array(train_data["sentence_id_list"]),
        test_feats_bs, np.array(test_data["sentence_id_list"]),
        test_data["n_classes"], "A1c_band_separated", logger,
    )
    all_results["A1c_band_separated"] = lp_a1c

    # ── A1b: Duration-Weighted Pool ──
    # 注：需要 fixation duration 数据；如不可用，使用 mean_pool 代替并标注
    # 当前 UnifiedDataset 未包含 fixation duration 字段，标注为 fallback
    logger.info("=== A1b: Duration-Weighted Pool ===")
    logger.info("  注意: UnifiedDataset 当前未包含 fixation duration 字段")
    logger.info("  使用 mean_pool 作为 fallback（等效于 A1a）")
    all_results["A1b_duration_weighted"] = {
        "variant": "A1b_duration_weighted",
        "status": "fallback_to_mean_pool",
        "note": "fixation duration 数据不可用，需在 build_unified_dataset.py 中添加",
        **lp_a1a,
    }
    all_results["A1b_duration_weighted"]["variant"] = "A1b_duration_weighted"

    # ── A2: 被试效应分析 ──
    cosine_result = run_cosine_similarity_analysis(
        test_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], test_data["task_list"], logger,
    )
    all_results["A2_cosine_similarity"] = cosine_result

    eta2_result = run_eta_squared_analysis(
        test_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
    )
    all_results["A2_eta_squared"] = eta2_result

    # ── A2-band: 频带级 η² 分析 ──
    band_eta2 = run_band_level_eta_squared(
        test_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
    )
    all_results["A2_band_eta_squared"] = band_eta2

    # ── A2: t-SNE 可视化 ──
    if not args.skip_tsne:
        run_tsne_visualization(
            test_feats_mp, test_data["sentence_id_list"],
            test_data["subject_list"], test_data["task_list"],
            args.output_dir, logger,
        )

    # ── A3: 去被试化验证（条件执行） ──
    if not args.skip_a3 and eta2_result.get("conclusion") == "subject_dominant":
        logger.info("η² 分析表明被试效应主导，执行 A3 去被试化验证")
        a3_result = run_desubject_analysis(args.data_path, args.output_dir, logger)
        all_results["A3_desubject"] = a3_result
    elif not args.skip_a3:
        logger.info("η² 分析未表明被试效应主导，仍执行 A3 作为参考")
        a3_result = run_desubject_analysis(args.data_path, args.output_dir, logger)
        all_results["A3_desubject"] = a3_result
    else:
        logger.info("跳过 A3（--skip-a3）")

    # ── 保存结果 ──
    out_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("结果已保存 → %s", out_path)

    # ── 保存频带级 η² ──
    band_path = os.path.join(args.output_dir, "band_level_eta_squared.json")
    with open(band_path, "w", encoding="utf-8") as f:
        json.dump(band_eta2, f, ensure_ascii=False, indent=2)
    logger.info("频带 η² 已保存 → %s", band_path)

    # ── 保存被试效应分析 ──
    subj_path = os.path.join(args.output_dir, "subject_effect_analysis.json")
    subj_analysis = {
        "cosine_similarity": cosine_result,
        "eta_squared": eta2_result,
    }
    with open(subj_path, "w", encoding="utf-8") as f:
        json.dump(subj_analysis, f, ensure_ascii=False, indent=2, default=str)
    logger.info("被试效应分析已保存 → %s", subj_path)

    # ── 综合打印 ──
    sep = "=" * 60
    print(f"\n{sep}")
    print("诊断线 A：EEG 数据有效性验证结果")
    print(sep)
    for key in ["A1a_mean_pool", "A1c_band_separated"]:
        r = all_results[key]
        print(f"  {r['variant']}: Top-1={r['top1_accuracy']*100:.2f}%, "
              f"Top-5={r['top5_accuracy']*100:.2f}%, "
              f"Top-10={r['top10_accuracy']*100:.2f}%")
    print(f"  随机基线: {all_results['A1a_mean_pool']['random_baseline']*100:.2f}%")
    print(f"\n  η²(句子) median: {eta2_result['eta2_sentence']['median']:.6f}")
    print(f"  η²(被试) median: {eta2_result['eta2_subject']['median']:.6f}")
    print(f"  结论: {eta2_result.get('conclusion', 'N/A')}")

    if "A3_desubject" in all_results:
        a3 = all_results["A3_desubject"]
        agg = a3["aggregated_retrieval"]
        print(f"\n  去被试化 Linear Probe Top-1: "
              f"{a3['linear_probe_desubject']['top1_accuracy']*100:.2f}%")
        print(f"  被试聚合检索: R@1={agg['r@1']*100:.2f}%, "
              f"MRR={agg['mrr']:.4f}, Mean Rank={agg['mean_rank']:.1f}")

    print(sep)
    print(f"  输出目录: {args.output_dir}")
    print(sep)
    logger.info("诊断线 A 完成")


if __name__ == "__main__":
    main()
