#!/usr/bin/env python3
"""诊断线 A：原始数据集有效性验证（v3 规范，三组信号并行 + LOSO 5折CV）。

在当前特征表示下，从数据层面验证 ZuCo EEG 是否包含可检测的句子级语义信息。
包含以下实验：
  A1a: Linear Probe — Mean-Pool 基线（三组信号：词级EEG / 句级EEG / 高斯噪声）
  A1b: Linear Probe — Duration-Weighted Pool（词级用nfixations加权；句级/噪声fallback）
  A1c: Linear Probe — Band-Separated（词级band_sep；句级/噪声与A1a等价）
  A2:  被试效应 vs 句子效应分析（余弦相似度分组 + η² 方差分解 + t-SNE）
  A2-band: 频带级 η² 分析
  A3:  去被试化信号恢复验证（无条件执行：LOSO框架下per-subject z-score + 被试聚合检索）

所有实验仅在 test 集上进行（约130句 / 1858条 / 14被试）。
A1/A3 使用 LOSO 5折CV（StratifiedGroupKFold），无被试泄露。
纯 CPU 运行，依赖 sklearn / scipy / matplotlib。

用法（项目根目录下）：
  python benchmark_eval/scripts/validate_eeg_signal.py \
      --data-path benchmark_eval/data/unified_zuco.pkl

详见 docs/detail/experiment_A_details.md（v3 修订版）
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
from constants import EEG_CHANNELS, EEG_BANDS, EEG_WORD_DIM, MAX_LEN, DEFAULT_SEED

# 频带名称
BAND_NAMES = ["theta1", "theta2", "alpha1", "alpha2",
              "beta1", "beta2", "gamma1", "gamma2"]


def parse_args():
    p = argparse.ArgumentParser(description="诊断线 A：EEG 数据有效性验证（v3）")
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
    """从 unified_zuco.pkl 加载 test 集样本，提取 EEG 特征和元数据。

    v3 变更：
    - 固定只加载 test 集（A1/A3 诊断实验只用 test 集做 LOSO CV）
    - 新增读取 sent_eeg_raw 和 nfixations_word
    """
    ds = UnifiedDataset(data_path, phase=phase)
    eeg_list = []
    text_list = []
    subject_list = []
    task_list = []
    sentence_id_list = []
    sent_eeg_list = []
    nfixations_list = []

    # 构建句子文本 -> 唯一 ID 映射
    unique_texts = []
    text_to_id = {}

    for i in range(len(ds)):
        sample = ds[i]
        # 获取词级 EEG（优先 eeg_word_norm1d）
        eeg_key = "eeg_word_norm1d" if "eeg_word_norm1d" in sample else "eeg"
        eeg = sample[eeg_key].numpy()  # (MAX_LEN, 840)
        mask_key = "mask_word" if "mask_word" in sample else "mask"
        mask = sample[mask_key].numpy()  # (MAX_LEN,)

        # 获取句级 EEG
        if "sent_eeg_raw" in sample:
            sent_eeg = sample["sent_eeg_raw"].numpy()  # (840,)
        else:
            sent_eeg = np.zeros(EEG_WORD_DIM, dtype=np.float32)

        # 获取词级注视次数
        if "nfixations_word" in sample:
            nfix = sample["nfixations_word"].numpy()  # (MAX_LEN,)
        else:
            nfix = None

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
        sent_eeg_list.append(sent_eeg)
        nfixations_list.append(nfix)

    return {
        "eeg_list": eeg_list,
        "text_list": text_list,
        "subject_list": subject_list,
        "task_list": task_list,
        "sentence_id_list": sentence_id_list,
        "sent_eeg_list": sent_eeg_list,
        "nfixations_list": nfixations_list,
        "unique_texts": unique_texts,
        "n_classes": len(unique_texts),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 特征提取（三组信号）
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(eeg_list, variant="mean_pool"):
    """从词级 EEG 序列提取句级特征向量。

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
            # 按频带结构保持：reshape -> (valid_len, 8, 105) -> mean -> flatten
            reshaped = eeg_valid.reshape(valid_len, EEG_BANDS, EEG_CHANNELS)
            feat = reshaped.mean(axis=0).flatten()  # (840,)
        else:
            feat = eeg_valid.mean(axis=0)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def extract_weighted_features(eeg_list, nfixations_list):
    """Duration-Weighted Pool：用 nfixations 加权 mean-pool。

    Args:
        eeg_list: [(eeg, mask), ...], eeg shape (MAX_LEN, 840), mask shape (MAX_LEN,)
        nfixations_list: [np.ndarray (MAX_LEN,) or None, ...]

    Returns:
        features: np.ndarray, shape (N, 840)
    """
    feats = []
    for (eeg, mask), nfix in zip(eeg_list, nfixations_list):
        valid_len = int(mask.sum())
        if valid_len == 0:
            valid_len = 1
        eeg_valid = eeg[:valid_len]  # (valid_len, 840)

        if nfix is not None:
            nfix_valid = nfix[:valid_len]
            weights = nfix_valid.copy()
            # 确保权重非负且不全为零
            weights = np.maximum(weights, 0.0)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(valid_len, dtype=np.float32) / valid_len
            # 加权均值: (valid_len,) @ (valid_len, 840) -> (840,)
            feat = (weights[:, np.newaxis] * eeg_valid).sum(axis=0)
        else:
            # fallback: 无 nfixations 数据时用普通 mean_pool
            feat = eeg_valid.mean(axis=0)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def extract_sent_features(sent_eeg_list):
    """从 sent_eeg_raw 提取句级特征（逐样本 z-score 归一化）。

    Args:
        sent_eeg_list: [np.ndarray (840,), ...]

    Returns:
        features: np.ndarray, shape (N, 840)
    """
    feats = []
    for s in sent_eeg_list:
        s = np.asarray(s, dtype=np.float32)
        std = s.std()
        if std > 1e-8:
            feat = (s - s.mean()) / std
        else:
            feat = np.zeros_like(s)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def generate_noise_features(N, dim=EEG_WORD_DIM, base_seed=DEFAULT_SEED):
    """生成高斯噪声特征（每样本独立种子，保证可复现）。

    Args:
        N: 样本数
        dim: 特征维度（默认840）
        base_seed: 基础种子（每样本种子 = base_seed + i）

    Returns:
        features: np.ndarray, shape (N, dim)
    """
    feats = []
    for i in range(N):
        rng = np.random.default_rng(base_seed + i)
        feat = rng.standard_normal(dim).astype(np.float32)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# A1: LOSO 5折CV Linear Probe
# ═══════════════════════════════════════════════════════════════════════════

def run_loso_linear_probe(X, y, groups, n_classes, variant_name, logger):
    """LOSO 5折交叉验证 Linear Probe。

    使用 StratifiedGroupKFold(n_splits=5) 按被试分组，保证无被试泄露。
    每折: StandardScaler fit on train -> transform test, LogisticRegression fit/predict。

    Args:
        X: 特征矩阵 (N, 840)
        y: 标签数组 (N,)
        groups: 被试分组数组 (N,)
        n_classes: 类别数
        variant_name: 实验变体名（用于日志）
        logger: 日志器

    Returns:
        dict: {"mean_top1", "std_top1", "mean_top5", "std_top5",
               "mean_top10", "std_top10", "random_baseline", "n_classes",
               "n_samples", "folds": [...]}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold

    logger.info("=== A1 LOSO Linear Probe: %s ===", variant_name)
    logger.info("  N=%d, n_classes=%d, n_groups=%d",
                X.shape[0], n_classes, len(np.unique(groups)))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # LogisticRegression
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=DEFAULT_SEED,
        )
        clf.fit(X_train_scaled, y_train)

        # 预测概率
        proba = clf.predict_proba(X_test_scaled)  # (n_test, C)
        predictions = clf.predict(X_test_scaled)

        # Top-1 / Top-5 / Top-10 accuracy
        top1 = float(np.mean(predictions == y_test))
        top5_count, top10_count = 0, 0
        for i, true_label in enumerate(y_test):
            sorted_classes = np.argsort(proba[i])[::-1]
            if true_label in sorted_classes[:5]:
                top5_count += 1
            if true_label in sorted_classes[:10]:
                top10_count += 1
        top5 = top5_count / len(y_test)
        top10 = top10_count / len(y_test)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "top1": round(top1, 6),
            "top5": round(top5, 6),
            "top10": round(top10, 6),
        })
        logger.info("  Fold %d: train=%d, test=%d, Top-1=%.4f%%, Top-5=%.4f%%, Top-10=%.4f%%",
                     fold_idx, len(train_idx), len(test_idx),
                     top1 * 100, top5 * 100, top10 * 100)

    # 汇总: 5折均值 +/- std
    top1_vals = [f["top1"] for f in fold_results]
    top5_vals = [f["top5"] for f in fold_results]
    top10_vals = [f["top10"] for f in fold_results]

    result = {
        "variant": variant_name,
        "mean_top1": round(float(np.mean(top1_vals)), 6),
        "std_top1": round(float(np.std(top1_vals)), 6),
        "mean_top5": round(float(np.mean(top5_vals)), 6),
        "std_top5": round(float(np.std(top5_vals)), 6),
        "mean_top10": round(float(np.mean(top10_vals)), 6),
        "std_top10": round(float(np.std(top10_vals)), 6),
        "random_baseline": round(1.0 / n_classes, 6),
        "n_classes": n_classes,
        "n_samples": X.shape[0],
        "folds": fold_results,
    }

    logger.info("  LOSO 5-fold mean: Top-1=%.4f%%(+/-%.4f), Top-5=%.4f%%(+/-%.4f), "
                "Top-10=%.4f%%(+/-%.4f)",
                result["mean_top1"] * 100, result["std_top1"] * 100,
                result["mean_top5"] * 100, result["std_top5"] * 100,
                result["mean_top10"] * 100, result["std_top10"] * 100)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# A2: 被试效应 vs 句子效应分析
# ═══════════════════════════════════════════════════════════════════════════

def run_cosine_similarity_analysis(features, sentence_ids, subject_ids, logger):
    """余弦相似度分组对比。

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
    """方差分解: eta^2 分析。

    对 EEG 特征的每个维度做 two-way ANOVA 变体（手动计算 eta^2）。
    """
    logger.info("=== A2: eta^2 方差分解 ===")

    n_features = features.shape[1]
    eta2_sentence = []
    eta2_subject = []

    # 编码因子
    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)

    unique_sent = np.unique(sent_arr)
    unique_subj = np.unique(subj_arr)

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

    logger.info("  eta^2(句子): mean=%.6f, median=%.6f",
                result["eta2_sentence"]["mean"], result["eta2_sentence"]["median"])
    logger.info("  eta^2(被试): mean=%.6f, median=%.6f",
                result["eta2_subject"]["mean"], result["eta2_subject"]["median"])

    # 判定
    ratio = result["eta2_subject"]["median"] / max(result["eta2_sentence"]["median"], 1e-12)
    if ratio > 3:
        result["conclusion"] = "subject_dominant"
        logger.info("  结论: 被试效应主导 (eta^2_subj/eta^2_sent = %.1f)", ratio)
    elif ratio > 0.5:
        result["conclusion"] = "comparable"
        logger.info("  结论: 两者效应相当")
    else:
        result["conclusion"] = "sentence_dominant"
        logger.info("  结论: 句子效应主导")

    return result


def run_band_level_eta_squared(features, sentence_ids, subject_ids, logger):
    """A2-band: 频带级 eta^2 分析。

    对 8 个频带分别计算 eta^2_sentence 和 eta^2_subject。
    """
    logger.info("=== A2-band: 频带级 eta^2 分析 ===")

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
        logger.info("  %s: eta^2_sent=%.6f, eta^2_subj=%.6f (median)",
                     band_name,
                     band_results[band_name]["eta2_sentence_median"],
                     band_results[band_name]["eta2_subject_median"])

    return band_results


def run_tsne_visualization(features, sentence_ids, subject_ids, task_ids,
                           output_dir, logger):
    """t-SNE 降维可视化（多 perplexity）。"""
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
    logger.info("  PCA: %d -> %d 维（解释方差: %.2f%%）",
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
                     max_iter=1000)
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
            logger.info("    -> %s", fname)


# ═══════════════════════════════════════════════════════════════════════════
# A3: 去被试化信号恢复验证（LOSO框架，无数据泄漏）
# ═══════════════════════════════════════════════════════════════════════════

def run_desubject_analysis(test_feats, y, groups, n_classes,
                           test_feats_noise, logger):
    """A3: 去被试化验证（v3: LOSO框架下per-subject z-score，无数据泄漏）。

    包含两个子实验:
      A3-LP: LOSO框架下去被试化 Linear Probe
      A3-Retrieval: 被试聚合检索（分组交叉验证）+ 噪声对照

    Args:
        test_feats: 测试集词级 mean_pool 特征 (N, 840)
        y: 句子标签 (N,)
        groups: 被试分组 (N,)
        n_classes: 类别数
        test_feats_noise: 噪声特征 (N, 840)，用于对照
        logger: 日志器
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold

    logger.info("=== A3: 去被试化验证（LOSO框架） ===")

    # ── A3-LP: LOSO 框架下 per-subject z-score + Linear Probe ──
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(test_feats, y, groups)):
        X_train, X_test = test_feats[train_idx], test_feats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # 在 train fold 内部计算 per-subject mu/sigma
        train_subj_set = set(groups_train)
        subject_stats = {}
        for subj in train_subj_set:
            subj_mask = groups_train == subj
            subj_feats = X_train[subj_mask]
            subject_stats[subj] = {
                "mean": subj_feats.mean(axis=0),
                "std": subj_feats.std(axis=0),
            }

        # 对 train/test fold 分别应用 per-subject z-score
        def _normalize_by_subject(feats, feats_groups):
            normed = np.zeros_like(feats)
            for i, subj in enumerate(feats_groups):
                if subj in subject_stats:
                    mu = subject_stats[subj]["mean"]
                    sigma = subject_stats[subj]["std"]
                    sigma = np.where(sigma < 1e-8, 1.0, sigma)
                    normed[i] = (feats[i] - mu) / sigma
                else:
                    normed[i] = feats[i]  # 未在train fold中出现的被试不归一化
            return normed

        X_train_normed = _normalize_by_subject(X_train, groups_train)
        X_test_normed = _normalize_by_subject(X_test, groups[test_idx])

        # StandardScaler + LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_normed)
        X_test_scaled = scaler.transform(X_test_normed)

        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=DEFAULT_SEED,
        )
        clf.fit(X_train_scaled, y_train)

        proba = clf.predict_proba(X_test_scaled)
        predictions = clf.predict(X_test_scaled)

        top1 = float(np.mean(predictions == y_test))
        top5_count, top10_count = 0, 0
        for i, true_label in enumerate(y_test):
            sorted_classes = np.argsort(proba[i])[::-1]
            if true_label in sorted_classes[:5]:
                top5_count += 1
            if true_label in sorted_classes[:10]:
                top10_count += 1
        top5 = top5_count / len(y_test)
        top10 = top10_count / len(y_test)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "top1": round(top1, 6),
            "top5": round(top5, 6),
            "top10": round(top10, 6),
        })
        logger.info("  A3-LP Fold %d: Top-1=%.4f%%, Top-5=%.4f%%, Top-10=%.4f%%",
                     fold_idx, top1 * 100, top5 * 100, top10 * 100)

    # 汇总 A3-LP
    top1_vals = [f["top1"] for f in fold_results]
    top5_vals = [f["top5"] for f in fold_results]
    top10_vals = [f["top10"] for f in fold_results]

    a3_lp_result = {
        "variant": "A3_desubject_mean_pool_LOSO",
        "mean_top1": round(float(np.mean(top1_vals)), 6),
        "std_top1": round(float(np.std(top1_vals)), 6),
        "mean_top5": round(float(np.mean(top5_vals)), 6),
        "std_top5": round(float(np.std(top5_vals)), 6),
        "mean_top10": round(float(np.mean(top10_vals)), 6),
        "std_top10": round(float(np.std(top10_vals)), 6),
        "random_baseline": round(1.0 / n_classes, 6),
        "n_classes": n_classes,
        "n_samples": test_feats.shape[0],
        "folds": fold_results,
    }

    logger.info("  A3-LP LOSO mean: Top-1=%.4f%%(+/-%.4f)",
                a3_lp_result["mean_top1"] * 100, a3_lp_result["std_top1"] * 100)

    # ── A3-Retrieval: 被试聚合检索（分组交叉验证） ──
    # 对 EEG 和噪声各运行一次
    eeg_retrieval = _run_aggregated_retrieval(
        test_feats, y, groups, logger, source="word_eeg")
    noise_retrieval = _run_aggregated_retrieval(
        test_feats_noise, y, groups, logger, source="noise")

    return {
        "linear_probe_desubject": a3_lp_result,
        "aggregated_retrieval": eeg_retrieval,
        "aggregated_retrieval_noise": noise_retrieval,
    }


def _run_aggregated_retrieval(features, sentence_ids, subject_ids, logger, source="eeg"):
    """A3-Retrieval: 被试聚合检索（分组交叉验证）。

    将被试对半分为 group_A / group_B，各自按句子聚合后做跨组检索。
    避免 self-retrieval (sim[i][i]=1 的平凡结果)。

    Args:
        features: (N, 840) 特征矩阵
        sentence_ids: (N,) 句子ID
        subject_ids: (N,) 被试ID
        logger: 日志器
        source: "word_eeg" 或 "noise"，用于日志

    Returns:
        dict: 检索结果
    """
    logger.info("--- A3-Retrieval: 被试聚合检索 [%s] ---", source)

    unique_subjects = sorted(set(subject_ids))
    half = max(1, len(unique_subjects) // 2)
    group_a_subj = set(unique_subjects[:half])
    group_b_subj = set(unique_subjects[half:])
    logger.info("  [%s] group_A: %d subjects, group_B: %d subjects",
                source, len(group_a_subj), len(group_b_subj))

    agg_a: dict = {}
    agg_b: dict = {}
    for i, sid in enumerate(sentence_ids):
        subj = subject_ids[i]
        if subj in group_a_subj:
            agg_a.setdefault(sid, []).append(features[i])
        else:
            agg_b.setdefault(sid, []).append(features[i])

    # 只保留两组都有数据的句子
    common_sids = sorted(set(agg_a.keys()) & set(agg_b.keys()))
    if len(common_sids) == 0:
        logger.warning("  [%s] 两组无公共句子，跳过聚合检索", source)
        return {"error": "no_common_sentences", "source": source}

    vec_a = np.array([np.mean(agg_a[s], axis=0) for s in common_sids], dtype=np.float32)
    vec_b = np.array([np.mean(agg_b[s], axis=0) for s in common_sids], dtype=np.float32)

    # L2 归一化后余弦相似度
    vec_a_norm = np.linalg.norm(vec_a, axis=1, keepdims=True).clip(min=1e-8)
    vec_b_norm = np.linalg.norm(vec_b, axis=1, keepdims=True).clip(min=1e-8)
    vec_a = vec_a / vec_a_norm
    vec_b = vec_b / vec_b_norm
    sim = vec_a @ vec_b.T  # (M, M), query=group_A, candidate=group_B

    n_common = len(common_sids)
    ranks = []
    for i in range(n_common):
        order = np.argsort(sim[i])[::-1]
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.float32)

    result = {
        "source": source,
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
    logger.info("  [%s] 聚合检索: R@1=%.4f%%, MRR=%.4f, Mean Rank=%.1f",
                source, result["r@1"] * 100, result["mrr"], result["mean_rank"])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(BENCH_DIR, "test_outputs", "dataset_validity")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir, log_name="validate_eeg_signal.log")
    logger.info("诊断线 A：EEG 数据有效性验证（v3）")
    logger.info("Args: %s", vars(args))

    all_results = {}

    # ── 数据加载（仅 test 集）──
    logger.info("加载 test 数据...")
    test_data = collect_samples(args.data_path, phase="test")
    N = len(test_data["eeg_list"])
    n_classes = test_data["n_classes"]
    y = np.array(test_data["sentence_id_list"])
    groups = np.array(test_data["subject_list"])
    logger.info("Test: %d samples, %d unique sentences, %d unique subjects",
                N, n_classes, len(np.unique(groups)))

    # ── 提取三组信号特征 ──
    # 词级 EEG (mean_pool)
    word_feats_mp = extract_features(test_data["eeg_list"], "mean_pool")
    # 词级 EEG (band_separated)
    word_feats_bs = extract_features(test_data["eeg_list"], "band_separated")
    # 词级 EEG (duration-weighted)
    word_feats_dw = extract_weighted_features(
        test_data["eeg_list"], test_data["nfixations_list"])
    # 句级 EEG
    sent_feats = extract_sent_features(test_data["sent_eeg_list"])
    # 高斯噪声
    noise_feats = generate_noise_features(N, dim=EEG_WORD_DIM, base_seed=args.seed)

    # ──────────────────────────────────────────────────────────────────────
    # A1a: Mean-Pool Linear Probe（三组信号并行）
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("A1a: Mean-Pool Linear Probe (三组信号)")
    logger.info("=" * 60)

    a1a_word = run_loso_linear_probe(
        word_feats_mp, y, groups, n_classes, "A1a_word_eeg_mean_pool", logger)
    a1a_sent = run_loso_linear_probe(
        sent_feats, y, groups, n_classes, "A1a_sent_eeg", logger)
    a1a_noise = run_loso_linear_probe(
        noise_feats, y, groups, n_classes, "A1a_noise", logger)

    all_results["A1a"] = {
        "word_eeg": a1a_word,
        "sent_eeg": a1a_sent,
        "noise": a1a_noise,
    }

    # ──────────────────────────────────────────────────────────────────────
    # A1b: Duration-Weighted Pool Linear Probe（三组信号）
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("A1b: Duration-Weighted Pool Linear Probe")
    logger.info("=" * 60)

    # 词级: 用 nfixations 加权
    has_nfix = test_data["nfixations_list"][0] is not None
    if has_nfix:
        a1b_word = run_loso_linear_probe(
            word_feats_dw, y, groups, n_classes,
            "A1b_word_eeg_duration_weighted", logger)
    else:
        logger.info("  nfixations_word 不可用，fallback 到 A1a 词级结果")
        a1b_word = dict(a1a_word)
        a1b_word["variant"] = "A1b_word_eeg_duration_weighted"
        a1b_word["status"] = "fallback_to_mean_pool"

    # 句级/噪声: 无时间轴可加权，fallback 到 A1a 结果
    a1b_sent = dict(a1a_sent)
    a1b_sent["variant"] = "A1b_sent_eeg"
    a1b_sent["status"] = "fallback_to_a1a"

    a1b_noise = dict(a1a_noise)
    a1b_noise["variant"] = "A1b_noise"
    a1b_noise["status"] = "fallback_to_a1a"

    all_results["A1b"] = {
        "word_eeg": a1b_word,
        "sent_eeg": a1b_sent,
        "noise": a1b_noise,
    }

    # ──────────────────────────────────────────────────────────────────────
    # A1c: Band-Separated Linear Probe（三组信号）
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("A1c: Band-Separated Linear Probe")
    logger.info("=" * 60)

    # 词级: band_separated
    a1c_word = run_loso_linear_probe(
        word_feats_bs, y, groups, n_classes,
        "A1c_word_eeg_band_separated", logger)

    # 句级: band_sep 与 mean_pool 数值等价（已是 (840,) 向量）
    a1c_sent = dict(a1a_sent)
    a1c_sent["variant"] = "A1c_sent_eeg"
    a1c_sent["note"] = "band_sep_equiv_to_a1a_for_sent_eeg"

    # 噪声: band_sep 与 mean_pool 数值等价
    a1c_noise = dict(a1a_noise)
    a1c_noise["variant"] = "A1c_noise"
    a1c_noise["note"] = "band_sep_equiv_to_a1a_for_noise"

    all_results["A1c"] = {
        "word_eeg": a1c_word,
        "sent_eeg": a1c_sent,
        "noise": a1c_noise,
    }

    # ──────────────────────────────────────────────────────────────────────
    # A2: 被试效应 vs 句子效应分析（EEG + 噪声对照）
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("A2: 被试效应 vs 句子效应分析")
    logger.info("=" * 60)

    # EEG 余弦相似度
    cosine_eeg = run_cosine_similarity_analysis(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger)
    # 噪声余弦相似度
    cosine_noise = run_cosine_similarity_analysis(
        noise_feats, test_data["sentence_id_list"],
        test_data["subject_list"], logger)

    all_results["A2_cosine_similarity"] = {
        "eeg": cosine_eeg,
        "noise": cosine_noise,
    }

    # EEG eta^2
    eta2_eeg = run_eta_squared_analysis(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger)
    # 噪声 eta^2
    eta2_noise = run_eta_squared_analysis(
        noise_feats, test_data["sentence_id_list"],
        test_data["subject_list"], logger)

    all_results["A2_eta_squared"] = {
        "eeg": eta2_eeg,
        "noise": eta2_noise,
    }

    # ── A2-band: 频带级 eta^2（仅 EEG）──
    band_eta2 = run_band_level_eta_squared(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger)
    all_results["A2_band_eta_squared"] = band_eta2

    # ── A2: t-SNE 可视化 ──
    if not args.skip_tsne:
        run_tsne_visualization(
            word_feats_mp, test_data["sentence_id_list"],
            test_data["subject_list"], test_data["task_list"],
            args.output_dir, logger,
        )

    # ──────────────────────────────────────────────────────────────────────
    # A3: 去被试化验证（无条件执行）
    # ──────────────────────────────────────────────────────────────────────
    if not args.skip_a3:
        logger.info("=" * 60)
        logger.info("A3: 去被试化信号恢复验证（无条件执行）")
        logger.info("=" * 60)

        a3_result = run_desubject_analysis(
            word_feats_mp, y, groups, n_classes,
            noise_feats, logger)
        all_results["A3_desubject"] = a3_result
    else:
        logger.info("跳过 A3 (--skip-a3)")

    # ── 保存结果 ──
    out_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("结果已保存 -> %s", out_path)

    # ── 保存频带级 eta^2 ──
    band_path = os.path.join(args.output_dir, "band_level_eta_squared.json")
    with open(band_path, "w", encoding="utf-8") as f:
        json.dump(band_eta2, f, ensure_ascii=False, indent=2)
    logger.info("频带 eta^2 已保存 -> %s", band_path)

    # ── 保存被试效应分析 ──
    subj_path = os.path.join(args.output_dir, "subject_effect_analysis.json")
    subj_analysis = {
        "cosine_similarity": {"eeg": cosine_eeg, "noise": cosine_noise},
        "eta_squared": {"eeg": eta2_eeg, "noise": eta2_noise},
    }
    with open(subj_path, "w", encoding="utf-8") as f:
        json.dump(subj_analysis, f, ensure_ascii=False, indent=2, default=str)
    logger.info("被试效应分析已保存 -> %s", subj_path)

    # ── 综合打印 ──
    sep = "=" * 60
    print(f"\n{sep}")
    print("诊断线 A：EEG 数据有效性验证结果（v3 三组信号）")
    print(sep)

    # A1 结果
    for a1_key in ["A1a", "A1b", "A1c"]:
        a1 = all_results[a1_key]
        print(f"\n  {a1_key}:")
        for src in ["word_eeg", "sent_eeg", "noise"]:
            r = a1[src]
            status = r.get("status", "")
            note = r.get("note", "")
            suffix = f" [{status}]" if status else (f" [{note}]" if note else "")
            print(f"    {src}: Top-1={r['mean_top1']*100:.2f}%(+/-{r['std_top1']*100:.2f})"
                  f"{suffix}")

    random_bl = all_results["A1a"]["word_eeg"]["random_baseline"]
    print(f"\n  随机基线: {random_bl*100:.2f}%")

    # A2 结果
    print(f"\n  eta^2(句子) median [EEG]:  {eta2_eeg['eta2_sentence']['median']:.6f}")
    print(f"  eta^2(被试) median [EEG]:  {eta2_eeg['eta2_subject']['median']:.6f}")
    print(f"  eta^2(句子) median [噪声]: {eta2_noise['eta2_sentence']['median']:.6f}")
    print(f"  eta^2(被试) median [噪声]: {eta2_noise['eta2_subject']['median']:.6f}")
    print(f"  结论: {eta2_eeg.get('conclusion', 'N/A')}")

    # A3 结果
    if "A3_desubject" in all_results:
        a3 = all_results["A3_desubject"]
        a3lp = a3["linear_probe_desubject"]
        agg = a3["aggregated_retrieval"]
        agg_noise = a3["aggregated_retrieval_noise"]
        print(f"\n  A3-LP 去被试化 Top-1: {a3lp['mean_top1']*100:.2f}%(+/-{a3lp['std_top1']*100:.2f})")
        if "error" not in agg:
            print(f"  A3 聚合检索 [EEG]:  R@1={agg['r@1']*100:.2f}%, "
                  f"MRR={agg['mrr']:.4f}, Mean Rank={agg['mean_rank']:.1f}")
        if "error" not in agg_noise:
            print(f"  A3 聚合检索 [噪声]: R@1={agg_noise['r@1']*100:.2f}%, "
                  f"MRR={agg_noise['mrr']:.4f}, Mean Rank={agg_noise['mean_rank']:.1f}")

    print(sep)
    print(f"  输出目录: {args.output_dir}")
    print(sep)
    logger.info("诊断线 A 完成")


if __name__ == "__main__":
    main()
