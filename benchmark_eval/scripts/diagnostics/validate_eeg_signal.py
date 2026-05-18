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
  python benchmark_eval/scripts/diagnostics/validate_eeg_signal.py \
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
BENCH_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
PROJ_ROOT = os.path.dirname(BENCH_DIR)

if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from data_processing.dataset import UnifiedDataset
from utils.logging_utils import setup_logging
from constants import EEG_CHANNELS, EEG_BANDS, EEG_WORD_DIM, MAX_LEN, DEFAULT_SEED

# 显著性检验封装模块（轻量延迟导入式）
try:
    from evaluation.significance import (
        wilcoxon_paired, binomial_vs_baseline, holm_bonferroni,
        permutation_eta, mannwhitney_u,
    )
    from evaluation.embedding_io import save_significance_json
    _SIG_OK = True
except Exception:
    _SIG_OK = False

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
    session_list = []
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
        session = meta.get("session", "session_unknown")

        if text not in text_to_id:
            text_to_id[text] = len(unique_texts)
            unique_texts.append(text)

        eeg_list.append((eeg, mask))
        text_list.append(text)
        subject_list.append(subject)
        task_list.append(task)
        session_list.append(session)
        sentence_id_list.append(text_to_id[text])
        sent_eeg_list.append(sent_eeg)
        nfixations_list.append(nfix)

    return {
        "eeg_list": eeg_list,
        "text_list": text_list,
        "subject_list": subject_list,
        "task_list": task_list,
        "session_list": session_list,
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
# A1d: 跨 Session 可分性 Linear Probe
# ═══════════════════════════════════════════════════════════════════════════

def run_session_probe(features_dict, subject_ids, session_ids, task_ids,
                      logger):
    """A1d: 跨 Session 可分性 Linear Probe（逐被试 StratifiedKFold 5 折）。

    对每位同时出现在两个 session 的被试单独建模，输入特征分别取
    词级 EEG / 句级 EEG / 高斯噪声，5 折均值 accuracy；全体被试再求 mean/std。
    额外输出 task1-SR 内部对照，以剥离 task 对 session 的混淆效应。

    Args:
        features_dict: {"word_eeg": (N,840), "sent_eeg": (N,840), "noise": (N,840)}
        subject_ids: list[str]，长度 N
        session_ids: list[str]，长度 N
        task_ids: list[str]，长度 N
        logger: 日志器

    Returns:
        dict: {"overall": {src: {...}}, "task1_sr": {src: {...}}}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    logger.info("=== A1d: 跨 Session Linear Probe ===")

    subj_arr = np.array(subject_ids)
    sess_arr = np.array(session_ids)
    task_arr = np.array(task_ids)

    def _probe_scope(scope_name, base_mask):
        logger.info("  --- scope=%s ---", scope_name)
        # 找出在该 scope 内同时出现两个 session 的被试
        dual_subjects = []
        for p in sorted(set(subj_arr[base_mask])):
            p_mask = base_mask & (subj_arr == p)
            p_sess = set(sess_arr[p_mask]) - {"session_unknown"}
            if len(p_sess) >= 2:
                dual_subjects.append(p)
        logger.info("    [%s] dual-session subjects: %d",
                    scope_name, len(dual_subjects))
        if not dual_subjects:
            return {src: {"error": "no_dual_session_subject", "n_subjects": 0}
                    for src in features_dict.keys()}

        scope_result: Dict[str, Any] = {}
        for src_name, X in features_dict.items():
            per_subject = []
            baselines: List[float] = []
            for p in dual_subjects:
                p_mask = (base_mask & (subj_arr == p)
                          & (sess_arr != "session_unknown"))
                if p_mask.sum() < 10:
                    continue
                X_p = X[p_mask]
                y_p = (sess_arr[p_mask] == "session_2").astype(int)
                n1 = int((y_p == 0).sum())
                n2 = int((y_p == 1).sum())
                if n1 < 2 or n2 < 2:
                    continue
                baseline = max(n1, n2) / (n1 + n2)
                baselines.append(baseline)

                n_splits = min(5, min(n1, n2))
                skf = StratifiedKFold(
                    n_splits=n_splits, shuffle=True,
                    random_state=DEFAULT_SEED,
                )
                fold_accs = []
                for tr_idx, te_idx in skf.split(X_p, y_p):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_p[tr_idx])
                    X_te = scaler.transform(X_p[te_idx])
                    clf = LogisticRegression(
                        max_iter=1000, solver="lbfgs",
                        random_state=DEFAULT_SEED,
                    )
                    clf.fit(X_tr, y_p[tr_idx])
                    pred = clf.predict(X_te)
                    fold_accs.append(float((pred == y_p[te_idx]).mean()))
                per_subject.append({
                    "subject": str(p),
                    "n": int(p_mask.sum()),
                    "n_session_1": n1,
                    "n_session_2": n2,
                    "baseline": round(baseline, 6),
                    "mean_acc": round(float(np.mean(fold_accs)), 6),
                    "std_acc": round(float(np.std(fold_accs)), 6),
                    "n_splits": n_splits,
                })

            if per_subject:
                accs_all = [s["mean_acc"] for s in per_subject]
                bl_mean = float(np.mean(baselines)) if baselines else 0.0
                scope_result[src_name] = {
                    "per_subject": per_subject,
                    "n_subjects": len(per_subject),
                    "mean_acc": round(float(np.mean(accs_all)), 6),
                    "std_acc": round(float(np.std(accs_all)), 6),
                    "mean_baseline": round(bl_mean, 6),
                    "delta_vs_baseline": round(
                        float(np.mean(accs_all)) - bl_mean, 6),
                }
                logger.info(
                    "    [%s] %s: n_subj=%d, mean_acc=%.4f(+/-%.4f), "
                    "baseline=%.4f, delta=%.4f",
                    scope_name, src_name, len(per_subject),
                    scope_result[src_name]["mean_acc"],
                    scope_result[src_name]["std_acc"],
                    bl_mean,
                    scope_result[src_name]["delta_vs_baseline"],
                )
            else:
                scope_result[src_name] = {
                    "error": "insufficient_data", "n_subjects": 0,
                }
                logger.warning("    [%s] %s: insufficient_data",
                               scope_name, src_name)
        return scope_result

    overall_mask = np.ones(len(sess_arr), dtype=bool)
    task1_mask = (task_arr == "task1-SR")
    return {
        "overall": _probe_scope("overall", overall_mask),
        "task1_sr": _probe_scope("task1_sr", task1_mask),
    }


# ═══════════════════════════════════════════════════════════════════════════
# A2: 被试效应 vs 句子效应分析
# ═══════════════════════════════════════════════════════════════════════════

def run_cosine_similarity_analysis(features, sentence_ids, subject_ids, logger,
                                   session_ids=None):
    """余弦相似度分组对比（含跨 session 分组）。

    基础三组：
      - 同句异被试：句子语义效应
      - 同被试异句：被试个体特征
      - 异句异被试：基线
    Session 扩展（仅当 session_ids 提供且包含两个已知 session 时输出）：
      - 同被试异句同 session
      - 同被试异句跨 session
    """
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("=== A2: 余弦相似度分组对比 ===")
    n = len(features)

    # 计算全体余弦相似度矩阵
    cos_matrix = cosine_similarity(features)  # (N, N)

    same_sent_diff_subj = []
    same_subj_diff_sent = []
    same_subj_diff_sent_same_session = []
    same_subj_diff_sent_cross_session = []
    diff_sent_diff_subj = []

    has_session = session_ids is not None

    for i in range(n):
        for j in range(i + 1, n):
            cos_val = cos_matrix[i, j]
            same_sent = (sentence_ids[i] == sentence_ids[j])
            same_subj = (subject_ids[i] == subject_ids[j])

            if same_sent and not same_subj:
                same_sent_diff_subj.append(cos_val)
            elif same_subj and not same_sent:
                same_subj_diff_sent.append(cos_val)
                if has_session:
                    sa, sb = session_ids[i], session_ids[j]
                    if sa != "session_unknown" and sb != "session_unknown":
                        if sa == sb:
                            same_subj_diff_sent_same_session.append(cos_val)
                        else:
                            same_subj_diff_sent_cross_session.append(cos_val)
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
    if has_session:
        result["same_subj_diff_sent_same_session"] = _stats(
            same_subj_diff_sent_same_session, "同被试异句同session")
        result["same_subj_diff_sent_cross_session"] = _stats(
            same_subj_diff_sent_cross_session, "同被试异句跨session")

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

    # Session 维度解读
    if has_session:
        same_s = result["same_subj_diff_sent_same_session"].get("mean")
        cross_s = result["same_subj_diff_sent_cross_session"].get("mean")
        if same_s is not None and cross_s is not None:
            delta = same_s - cross_s
            result["session_delta_same_minus_cross"] = round(float(delta), 6)
            logger.info(
                "  Session 对比: same_session=%.6f, cross_session=%.6f, delta=%.6f",
                same_s, cross_s, delta,
            )

    return result


def run_eta_squared_analysis(features, sentence_ids, subject_ids, logger,
                             session_ids=None):
    """方差分解: eta^2 分析（启 session 因素时为三因素）。

    对 EEG 特征的每个维度手动计算每个因子的组间方差占总方差的比例（独立考察）。
    若提供 session_ids，额外输出 eta^2_session（排除 `session_unknown` 后仅三类：
    其实际有值类别为两个 session）。
    """
    logger.info("=== A2: eta^2 方差分解 ===")

    n_features = features.shape[1]
    eta2_sentence = []
    eta2_subject = []
    eta2_session = []

    # 编码因子
    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)

    unique_sent = np.unique(sent_arr)
    unique_subj = np.unique(subj_arr)

    has_session = session_ids is not None
    if has_session:
        sess_arr = np.array(session_ids)
        # 只在已知 session 的样本上计算 session 因子（整体方差由它们汇总）
        sess_mask_known = sess_arr != "session_unknown"
        unique_session = [s for s in np.unique(sess_arr) if s != "session_unknown"]
        n_known_session = int(sess_mask_known.sum())
        if len(unique_session) < 2 or n_known_session < 10:
            has_session = False  # 不足以做 session 分解
            logger.warning(
                "Session factor skipped: unique_session=%d, n_known=%d",
                len(unique_session), n_known_session,
            )

    for d in range(n_features):
        y = features[:, d]
        grand_mean = y.mean()
        ss_total = np.sum((y - grand_mean) ** 2)

        if ss_total < 1e-12:
            eta2_sentence.append(0.0)
            eta2_subject.append(0.0)
            if has_session:
                eta2_session.append(0.0)
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

        # SS_session（仅已知 session）
        if has_session:
            y_k = y[sess_mask_known]
            if y_k.size == 0:
                eta2_session.append(0.0)
                continue
            grand_k = y_k.mean()
            ss_total_k = np.sum((y_k - grand_k) ** 2)
            if ss_total_k < 1e-12:
                eta2_session.append(0.0)
                continue
            ss_sess = 0.0
            for s in unique_session:
                m = (sess_arr == s) & sess_mask_known
                n_s = m.sum()
                if n_s > 0:
                    ss_sess += n_s * (y[m].mean() - grand_k) ** 2
            eta2_session.append(ss_sess / ss_total_k)

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
    if has_session:
        eta2_sess_arr = np.array(eta2_session)
        result["eta2_session"] = {
            "mean": round(float(eta2_sess_arr.mean()), 6),
            "median": round(float(np.median(eta2_sess_arr)), 6),
            "std": round(float(eta2_sess_arr.std()), 6),
        }
        result["n_known_session_samples"] = int((sess_arr != "session_unknown").sum())

    logger.info("  eta^2(句子): mean=%.6f, median=%.6f",
                result["eta2_sentence"]["mean"], result["eta2_sentence"]["median"])
    logger.info("  eta^2(被试): mean=%.6f, median=%.6f",
                result["eta2_subject"]["mean"], result["eta2_subject"]["median"])
    if has_session:
        logger.info("  eta^2(session): mean=%.6f, median=%.6f",
                    result["eta2_session"]["mean"], result["eta2_session"]["median"])

    # 判定 r_subj_vs_sent
    ratio = result["eta2_subject"]["median"] / max(result["eta2_sentence"]["median"], 1e-12)
    result["r_subj_vs_sent"] = round(float(ratio), 4)
    if ratio > 3:
        result["conclusion"] = "subject_dominant"
        logger.info("  结论: 被试效应主导 (eta^2_subj/eta^2_sent = %.1f)", ratio)
    elif ratio > 0.5:
        result["conclusion"] = "comparable"
        logger.info("  结论: 两者效应相当")
    else:
        result["conclusion"] = "sentence_dominant"
        logger.info("  结论: 句子效应主导")

    # 判定 r_session_vs_sent
    if has_session:
        ratio_s = result["eta2_session"]["median"] / max(result["eta2_sentence"]["median"], 1e-12)
        result["r_session_vs_sent"] = round(float(ratio_s), 4)
        if ratio_s > 3:
            result["session_conclusion"] = "session_dominant"
        elif ratio_s > 0.5:
            result["session_conclusion"] = "session_comparable_to_sentence"
        else:
            result["session_conclusion"] = "session_weak"
        logger.info("  r_session_vs_sent=%.3f → %s",
                    ratio_s, result["session_conclusion"])

    return result


def run_band_level_eta_squared(features, sentence_ids, subject_ids, logger,
                               session_ids=None):
    """A2-band: 频带级 eta^2 分析。

    对 8 个频带分别计算 eta^2_sentence、eta^2_subject，可选加 eta^2_session。
    """
    logger.info("=== A2-band: 频带级 eta^2 分析 ===")

    n_samples = features.shape[0]
    # 重塑为 (N, 8, 105)
    reshaped = features.reshape(n_samples, EEG_BANDS, EEG_CHANNELS)

    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)
    unique_sent = np.unique(sent_arr)
    unique_subj = np.unique(subj_arr)

    has_session = session_ids is not None
    if has_session:
        sess_arr = np.array(session_ids)
        sess_mask_known = sess_arr != "session_unknown"
        unique_session = [s for s in np.unique(sess_arr) if s != "session_unknown"]
        if len(unique_session) < 2 or sess_mask_known.sum() < 10:
            has_session = False

    band_results = {}
    for b, band_name in enumerate(BAND_NAMES):
        band_feat = reshaped[:, b, :]  # (N, 105)
        eta2_sent_list, eta2_subj_list, eta2_sess_list = [], [], []

        for d in range(EEG_CHANNELS):
            y = band_feat[:, d]
            grand_mean = y.mean()
            ss_total = np.sum((y - grand_mean) ** 2)

            if ss_total < 1e-12:
                eta2_sent_list.append(0.0)
                eta2_subj_list.append(0.0)
                if has_session:
                    eta2_sess_list.append(0.0)
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

            if has_session:
                y_k = y[sess_mask_known]
                grand_k = y_k.mean()
                ss_total_k = np.sum((y_k - grand_k) ** 2)
                if ss_total_k < 1e-12:
                    eta2_sess_list.append(0.0)
                    continue
                ss_sess = 0.0
                for s in unique_session:
                    mm = (sess_arr == s) & sess_mask_known
                    n_s = mm.sum()
                    if n_s > 0:
                        ss_sess += n_s * (y[mm].mean() - grand_k) ** 2
                eta2_sess_list.append(ss_sess / ss_total_k)

        entry = {
            "eta2_sentence_median": round(float(np.median(eta2_sent_list)), 6),
            "eta2_subject_median": round(float(np.median(eta2_subj_list)), 6),
            "eta2_sentence_mean": round(float(np.mean(eta2_sent_list)), 6),
            "eta2_subject_mean": round(float(np.mean(eta2_subj_list)), 6),
        }
        if has_session:
            entry["eta2_session_median"] = round(float(np.median(eta2_sess_list)), 6)
            entry["eta2_session_mean"] = round(float(np.mean(eta2_sess_list)), 6)
        band_results[band_name] = entry

        if has_session:
            logger.info(
                "  %s: eta^2_sent=%.6f, eta^2_subj=%.6f, eta^2_sess=%.6f (median)",
                band_name,
                entry["eta2_sentence_median"],
                entry["eta2_subject_median"],
                entry["eta2_session_median"],
            )
        else:
            logger.info("  %s: eta^2_sent=%.6f, eta^2_subj=%.6f (median)",
                         band_name,
                         entry["eta2_sentence_median"],
                         entry["eta2_subject_median"])

    return band_results


def run_tsne_visualization(features, sentence_ids, subject_ids, task_ids,
                           output_dir, logger, session_ids=None):
    """t-SNE 降维可视化（多 perplexity）。

    着色方案：
      - 按被试：perplexity ∈ {5, 30, 50}
      - 按句子 / task ：仅 perplexity=30
      - 按 session（可选）：仅 perplexity=30
    """
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
    if session_ids is not None:
        color_configs.append(("session", session_ids, "按 session 着色"))

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

            # 限制颜色数量；兼容 matplotlib 3.9+（plt.cm.get_cmap 已废弃）
            n_colors = max(1, min(len(unique_ids), 20))
            try:
                import matplotlib as _mpl
                cmap = _mpl.colormaps.get_cmap("tab20").resampled(n_colors)
            except Exception:
                cmap = plt.get_cmap("tab20", n_colors)

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
# A3-SessionRetrieval: 同被试跨 Session 聚合检索
# ═══════════════════════════════════════════════════════════════════════════

def _session_retrieve_core(features, sent_arr, sess_arr, idx_mask, label,
                           min_common=5):
    """按 session 聚合后跨 session 检索的核心流程。

    Args:
        features: (N, 840)
        sent_arr: np.ndarray，句子 ID
        sess_arr: np.ndarray，字符串 session
        idx_mask: bool mask，参与聚合的样本下标
        label: 用于记录的标签
        min_common: 少于该数量的公共句子视为不可检索
    """
    idxs = np.where(idx_mask)[0]
    if len(idxs) == 0:
        return {"error": "empty_mask", "label": label}
    agg1: dict = {}
    agg2: dict = {}
    for i in idxs:
        s = sent_arr[i]
        if sess_arr[i] == "session_1":
            agg1.setdefault(s, []).append(features[i])
        elif sess_arr[i] == "session_2":
            agg2.setdefault(s, []).append(features[i])
    common = sorted(set(agg1.keys()) & set(agg2.keys()))
    M = len(common)
    if M < min_common:
        return {"error": "too_few_common_sentences",
                "n_common": M, "label": label}
    vec_a = np.array([np.mean(agg1[s], axis=0) for s in common],
                     dtype=np.float32)
    vec_b = np.array([np.mean(agg2[s], axis=0) for s in common],
                     dtype=np.float32)
    na = np.linalg.norm(vec_a, axis=1, keepdims=True).clip(min=1e-8)
    nb = np.linalg.norm(vec_b, axis=1, keepdims=True).clip(min=1e-8)
    vec_a = vec_a / na
    vec_b = vec_b / nb
    sim = vec_a @ vec_b.T  # (M, M)
    ranks = []
    for i in range(M):
        order = np.argsort(sim[i])[::-1]
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks_arr = np.array(ranks, dtype=np.float32)
    return {
        "label": label,
        "n_sentences": M,
        "r@1":  round(float((ranks_arr <= 1).mean()),  6),
        "r@5":  round(float((ranks_arr <= 5).mean()),  6),
        "r@10": round(float((ranks_arr <= 10).mean()), 6),
        "mrr":  round(float((1.0 / ranks_arr).mean()), 6),
        "mean_rank":   round(float(ranks_arr.mean()), 2),
        "median_rank": round(float(np.median(ranks_arr)), 2),
        "random_baseline": round(1.0 / M, 6),
    }


def run_session_retrieval(features, sentence_ids, subject_ids, session_ids,
                          task_ids, logger, source="word_eeg"):
    """A3-SessionRetrieval: 同被试跨 Session 聚合检索。

    包含三个视图：
      - aggregated_all      ：全体被试按 session 内并聚合（包含所有 task）
      - aggregated_task1_sr ：仅 task1-SR 内部聚合（剥离 task 混淆）
      - per_subject         ：对每位双 session 被试单独执行聚合检索，再求均值
    """
    logger.info("--- A3-SessionRetrieval: 同被试跨 session 聚合检索 [%s] ---",
                source)

    sent_arr = np.array(sentence_ids)
    subj_arr = np.array(subject_ids)
    sess_arr = np.array(session_ids)
    task_arr = np.array(task_ids)
    known_mask = (sess_arr != "session_unknown")

    results: Dict[str, Any] = {"source": source}

    # 全体聚合
    results["aggregated_all"] = _session_retrieve_core(
        features, sent_arr, sess_arr, known_mask, "aggregated_all")
    # task1-SR 内部对照
    t1_mask = known_mask & (task_arr == "task1-SR")
    results["aggregated_task1_sr"] = _session_retrieve_core(
        features, sent_arr, sess_arr, t1_mask, "aggregated_task1_sr")

    # 被试内版本
    per_subject = []
    for p in sorted(set(subj_arr)):
        p_mask = known_mask & (subj_arr == p)
        r = _session_retrieve_core(
            features, sent_arr, sess_arr, p_mask, f"subject_{p}")
        if "error" not in r:
            r["subject"] = str(p)
            per_subject.append(r)
    if per_subject:
        r1 = [r["r@1"] for r in per_subject]
        r5 = [r["r@5"] for r in per_subject]
        mrr_l = [r["mrr"] for r in per_subject]
        results["per_subject"] = {
            "subjects": per_subject,
            "n_subjects": len(per_subject),
            "r@1_mean": round(float(np.mean(r1)), 6),
            "r@1_std":  round(float(np.std(r1)), 6),
            "r@5_mean": round(float(np.mean(r5)), 6),
            "mrr_mean": round(float(np.mean(mrr_l)), 6),
        }
        logger.info(
            "  [%s] per-subject: n=%d, R@1=%.4f(+/-%.4f), R@5=%.4f, MRR=%.4f",
            source, len(per_subject),
            results["per_subject"]["r@1_mean"],
            results["per_subject"]["r@1_std"],
            results["per_subject"]["r@5_mean"],
            results["per_subject"]["mrr_mean"],
        )
    else:
        results["per_subject"] = {"error": "no_dual_session_subject",
                                   "n_subjects": 0}
        logger.warning("  [%s] per-subject: no_dual_session_subject", source)

    for key in ["aggregated_all", "aggregated_task1_sr"]:
        r = results[key]
        if "error" not in r:
            logger.info(
                "  [%s] %s: M=%d, R@1=%.4f%% (baseline=%.4f%%), MRR=%.4f",
                source, key, r["n_sentences"],
                r["r@1"] * 100, r["random_baseline"] * 100, r["mrr"],
            )
        else:
            logger.info("  [%s] %s: %s (n_common=%s)", source, key,
                        r["error"], r.get("n_common"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# A 线显著性检验：统一调度
# ═══════════════════════════════════════════════════════════════════════════

def _folds_topk(a1_entry: Dict[str, Any], k: str) -> List[float]:
    """从 A1 某一组的 fold 结果中抽取 top1/top5/top10 列表。"""
    folds = (a1_entry or {}).get("folds", []) or []
    return [float(f.get(k, float("nan"))) for f in folds]


def _line_a_eta_significance(feats, sentence_ids, subject_ids,
                             n_dims: int = 5, n_perm: int = 200, seed: int = 42):
    """维度级 eta^2 对比（sentence vs subject）+ permutation 抽样。"""
    feats = np.asarray(feats)
    n_features = int(feats.shape[1])
    sent_arr = np.asarray(sentence_ids)
    subj_arr = np.asarray(subject_ids)

    eta_sent = np.zeros(n_features)
    eta_subj = np.zeros(n_features)
    u_sent = np.unique(sent_arr)
    u_subj = np.unique(subj_arr)
    for d in range(n_features):
        y = feats[:, d]
        gm = y.mean()
        ss_total = float(np.sum((y - gm) ** 2))
        if ss_total < 1e-12:
            continue
        ss_sent = 0.0
        for s in u_sent:
            m = (sent_arr == s)
            n_s = int(m.sum())
            if n_s > 0:
                ss_sent += n_s * (y[m].mean() - gm) ** 2
        ss_subj = 0.0
        for s in u_subj:
            m = (subj_arr == s)
            n_s = int(m.sum())
            if n_s > 0:
                ss_subj += n_s * (y[m].mean() - gm) ** 2
        eta_sent[d] = ss_sent / ss_total
        eta_subj[d] = ss_subj / ss_total

    result: Dict[str, Any] = {}
    try:
        result["subj_vs_sent_wilcoxon"] = wilcoxon_paired(
            eta_subj.tolist(), eta_sent.tolist())
    except Exception as exc:
        result["subj_vs_sent_wilcoxon"] = {"error": str(exc)}

    rng = np.random.default_rng(seed)
    dims = rng.choice(n_features, size=min(n_dims, n_features), replace=False)
    samples = []
    for d in dims:
        y = feats[:, int(d)].astype(np.float64)
        try:
            p_sent = permutation_eta(y, sent_arr.tolist(),
                                     n_perm=n_perm, seed=seed + int(d))
            p_subj = permutation_eta(y, subj_arr.tolist(),
                                     n_perm=n_perm, seed=seed + int(d) + 1)
            samples.append({
                "dim": int(d),
                "sentence": p_sent,
                "subject": p_subj,
            })
        except Exception as exc:
            samples.append({"dim": int(d), "error": str(exc)})
    result["permutation_sample"] = samples
    return result


def run_line_a_significance(*, all_results: Dict[str, Any],
                            word_feats_mp, noise_feats,
                            sentence_ids, subject_ids,
                            logger) -> Dict[str, Any]:
    """A 线显著性检验统一调度器（全部委托 evaluation.significance 执行）。

    覆盖：
      - A1a/A1b/A1c：5 折配对 Wilcoxon（word/sent/noise 两两对比）+
        vs_random 二项检验（three sources × top-1）。
      - A2_cosine：观察性统计（仅记录 mean/delta）。
      - A2_eta：维度级 subj vs sent 配对 Wilcoxon + permutation 抽样（5 维）。
      - A3_lp：去被试化 LP 与 A1a word_eeg 的 5 折配对 Wilcoxon。
      - A3_session_retrieval：r@K vs 随机基线 k/M 的二项检验。
      - correction：A1 word_vs_noise 9 组（3 variant × 3 top-k）Holm-Bonferroni。
    """
    results: Dict[str, Any] = {"correction": {}}
    a1_word_vs_noise: List[Tuple[str, Optional[float]]] = []

    # ── A1a / A1b / A1c ──
    for a1_key in ("A1a", "A1b", "A1c"):
        a1 = all_results.get(a1_key) or {}
        if not a1:
            continue
        entry: Dict[str, Any] = {}
        for tag, src_a, src_b in (
            ("word_vs_noise", "word_eeg", "noise"),
            ("sent_vs_noise", "sent_eeg", "noise"),
            ("word_vs_sent",  "word_eeg", "sent_eeg"),
        ):
            for k in ("top1", "top5", "top10"):
                xa = _folds_topk(a1.get(src_a), k)
                xb = _folds_topk(a1.get(src_b), k)
                if len(xa) == len(xb) and len(xa) >= 3:
                    try:
                        res = wilcoxon_paired(xa, xb)
                    except Exception as exc:
                        res = {"error": str(exc)}
                    entry[f"{tag}.{k}"] = res
                    if tag == "word_vs_noise":
                        a1_word_vs_noise.append(
                            (f"{a1_key}.{tag}.{k}", res.get("p")))

        for src in ("word_eeg", "sent_eeg", "noise"):
            r = a1.get(src) or {}
            mt1 = r.get("mean_top1")
            n = r.get("n_samples")
            baseline = r.get("random_baseline")
            if mt1 is not None and n and baseline:
                try:
                    k_succ = int(round(float(mt1) * int(n)))
                    entry[f"vs_random.{src}.top1"] = binomial_vs_baseline(
                        k_succ, int(n), float(baseline))
                except Exception as exc:
                    entry[f"vs_random.{src}.top1"] = {"error": str(exc)}
        results[a1_key] = entry

    # ── A2 余弦：observational ──
    a2c = all_results.get("A2_cosine_similarity") or {}
    if a2c:
        eeg_same = (a2c.get("eeg") or {}).get("same_sent_diff_subj", {}).get("mean")
        eeg_diff = (a2c.get("eeg") or {}).get("diff_sent_diff_subj", {}).get("mean")
        noi_same = (a2c.get("noise") or {}).get("same_sent_diff_subj", {}).get("mean")
        delta = None
        if eeg_same is not None and eeg_diff is not None:
            delta = float(eeg_same) - float(eeg_diff)
        results["A2_cosine"] = {
            "note": "observational; statistic_only",
            "eeg_same_sent_mean": eeg_same,
            "eeg_diff_sent_mean": eeg_diff,
            "eeg_delta_same_minus_diff": delta,
            "noise_same_sent_mean": noi_same,
        }

    # ── A2 eta：维度级 Wilcoxon + permutation 抽样 ──
    try:
        if word_feats_mp is not None and len(word_feats_mp) == len(sentence_ids) \
                and len(word_feats_mp) == len(subject_ids):
            results["A2_eta"] = _line_a_eta_significance(
                word_feats_mp, sentence_ids, subject_ids)
    except Exception as exc:
        results["A2_eta"] = {"error": str(exc)}

    # ── A3-LP：vs A1a word_eeg 配对 Wilcoxon ──
    a3 = all_results.get("A3_desubject") or {}
    a3_lp = a3.get("linear_probe_desubject") if isinstance(a3, dict) else None
    a1a_word = (all_results.get("A1a") or {}).get("word_eeg")
    if a3_lp and a1a_word:
        a3_entry: Dict[str, Any] = {}
        for k in ("top1", "top5", "top10"):
            xa = _folds_topk(a3_lp, k)
            xb = _folds_topk(a1a_word, k)
            if len(xa) == len(xb) and len(xa) >= 3:
                try:
                    a3_entry[f"a3lp_vs_a1a_word.{k}"] = wilcoxon_paired(xa, xb)
                except Exception as exc:
                    a3_entry[f"a3lp_vs_a1a_word.{k}"] = {"error": str(exc)}
        results["A3_lp"] = a3_entry

    # ── A3-SessionRetrieval：r@K vs k/M 二项检验 ──
    a3sr = all_results.get("A3_session_retrieval") or {}
    if a3sr:
        entry_sr: Dict[str, Any] = {}
        for src_key in ("eeg", "noise"):
            src_res = a3sr.get(src_key) or {}
            for agg in ("aggregated_all", "aggregated_task1_sr"):
                r = src_res.get(agg) or {}
                M = r.get("n_sentences")
                if not M:
                    continue
                for k in (1, 5, 10):
                    # S-8: 跳过 k >= M 的情况（baseline=k/M >= 1 无意义）
                    if k >= M:
                        entry_sr[f"{src_key}.{agg}.r@{k}_vs_random"] = {
                            "skipped": True,
                            "reason": f"k={k} >= M={M}, baseline >= 1.0"}
                        continue
                    rk_val = r.get(f"r@{k}")
                    if rk_val is None:
                        continue
                    try:
                        k_succ = int(round(float(rk_val) * int(M)))
                        baseline = float(k) / float(M)
                        entry_sr[f"{src_key}.{agg}.r@{k}_vs_random"] = (
                            binomial_vs_baseline(k_succ, int(M), baseline))
                    except Exception as exc:
                        entry_sr[f"{src_key}.{agg}.r@{k}_vs_random"] = {
                            "error": str(exc)}
        results["A3_session_retrieval"] = entry_sr

    # ── Holm-Bonferroni：A1 word_vs_noise 9 组 ──
    valid = [(lbl, p) for lbl, p in a1_word_vs_noise if p is not None]
    if valid:
        labels = [v[0] for v in valid]
        pvals = [float(v[1]) for v in valid]
        try:
            holm = holm_bonferroni(pvals, alpha=0.05)
            results["correction"]["a1_word_vs_noise_holm"] = {
                "method": holm.get("method", "holm_bonferroni"),
                "labels": labels,
                "p_raw": pvals,
                "p_adj": holm.get("adjusted", []),
                "reject_at_0.05": holm.get("significant", []),
                "n_tests": holm.get("n_tests", len(pvals)),
                "alpha": holm.get("alpha", 0.05),
            }
        except Exception as exc:
            results["correction"]["a1_word_vs_noise_holm"] = {"error": str(exc)}

    logger.info("A 线显著性检验完成，产出 key 数=%d", len(results))
    return results


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

    # A1d: 跨 Session Linear Probe（同被试 EEG / 句级 EEG / 噪声）
    logger.info("=" * 60)
    logger.info("A1d: 跨 Session Linear Probe")
    logger.info("=" * 60)

    a1d_result = run_session_probe(
        features_dict={
            "word_eeg": word_feats_mp,
            "sent_eeg": sent_feats,
            "noise": noise_feats,
        },
        subject_ids=test_data["subject_list"],
        session_ids=test_data["session_list"],
        task_ids=test_data["task_list"],
        logger=logger,
    )
    all_results["A1d_cross_session"] = a1d_result

    # ──────────────────────────────────────────────────────────────────────
    # A2: 被试效应 vs 句子效应分析（EEG + 噪声对照）
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("A2: 被试效应 vs 句子效应分析")
    logger.info("=" * 60)

    # EEG 余弦相似度
    cosine_eeg = run_cosine_similarity_analysis(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
        session_ids=test_data["session_list"])
    # 噪声余弦相似度
    cosine_noise = run_cosine_similarity_analysis(
        noise_feats, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
        session_ids=test_data["session_list"])

    all_results["A2_cosine_similarity"] = {
        "eeg": cosine_eeg,
        "noise": cosine_noise,
    }

    # EEG eta^2
    eta2_eeg = run_eta_squared_analysis(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
        session_ids=test_data["session_list"])
    # 噪声 eta^2
    eta2_noise = run_eta_squared_analysis(
        noise_feats, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
        session_ids=test_data["session_list"])

    # task1-SR 内部 eta^2 对照（剥离 task 对 session 的混淆）
    task_arr_full = np.array(test_data["task_list"])
    t1_mask = (task_arr_full == "task1-SR")
    if t1_mask.sum() >= 10:
        sess_list_arr = np.array(test_data["session_list"])
        sent_list_arr = np.array(test_data["sentence_id_list"])
        subj_list_arr = np.array(test_data["subject_list"])
        eta2_eeg_t1 = run_eta_squared_analysis(
            word_feats_mp[t1_mask],
            sent_list_arr[t1_mask].tolist(),
            subj_list_arr[t1_mask].tolist(),
            logger,
            session_ids=sess_list_arr[t1_mask].tolist(),
        )
        eta2_eeg["task1_sr_control"] = eta2_eeg_t1
    else:
        eta2_eeg["task1_sr_control"] = {"error": "insufficient_task1_sr_samples"}

    all_results["A2_eta_squared"] = {
        "eeg": eta2_eeg,
        "noise": eta2_noise,
    }

    # ── A2-band: 频带级 eta^2（仅 EEG）──
    band_eta2 = run_band_level_eta_squared(
        word_feats_mp, test_data["sentence_id_list"],
        test_data["subject_list"], logger,
        session_ids=test_data["session_list"])
    all_results["A2_band_eta_squared"] = band_eta2

    # ── A2: t-SNE 可视化 ──
    if not args.skip_tsne:
        try:
            run_tsne_visualization(
                word_feats_mp, test_data["sentence_id_list"],
                test_data["subject_list"], test_data["task_list"],
                args.output_dir, logger,
                session_ids=test_data["session_list"],
            )
        except Exception as _e:
            logger.warning("A2 t-SNE 可视化失败，跳过：%s", _e)

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

        # A3-SessionRetrieval: 同被试跨 Session 聚合检索（EEG + 噪声对照）
        logger.info("-" * 60)
        logger.info("A3-SessionRetrieval: 同被试跨 Session 聚合检索")
        logger.info("-" * 60)
        session_retrieval_eeg = run_session_retrieval(
            word_feats_mp, test_data["sentence_id_list"],
            test_data["subject_list"], test_data["session_list"],
            test_data["task_list"], logger, source="word_eeg",
        )
        session_retrieval_noise = run_session_retrieval(
            noise_feats, test_data["sentence_id_list"],
            test_data["subject_list"], test_data["session_list"],
            test_data["task_list"], logger, source="noise",
        )
        all_results["A3_session_retrieval"] = {
            "eeg": session_retrieval_eeg,
            "noise": session_retrieval_noise,
        }
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

    # ── 显著性检验（统一封装调用）──
    if _SIG_OK:
        try:
            sig_results = run_line_a_significance(
                all_results=all_results,
                word_feats_mp=word_feats_mp,
                noise_feats=noise_feats,
                sentence_ids=test_data["sentence_id_list"],
                subject_ids=test_data["subject_list"],
                logger=logger,
            )
            sig_path = save_significance_json(args.output_dir, sig_results)
            logger.info("显著性检验已保存 -> %s", sig_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("显著性检验失败：%s", exc)
    else:
        logger.warning("evaluation.significance 未加载，跳过显著性检验")

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

    # A1d 跨 session 结果
    if "A1d_cross_session" in all_results:
        print("\n  A1d 跨 Session Linear Probe:")
        for scope_key in ["overall", "task1_sr"]:
            scope = all_results["A1d_cross_session"].get(scope_key, {})
            print(f"    scope={scope_key}:")
            for src in ["word_eeg", "sent_eeg", "noise"]:
                r = scope.get(src, {})
                if "error" in r:
                    print(f"      {src}: {r['error']}")
                else:
                    print(
                        f"      {src}: n_subj={r['n_subjects']}, "
                        f"acc={r['mean_acc']*100:.2f}%(+/-{r['std_acc']*100:.2f}), "
                        f"baseline={r['mean_baseline']*100:.2f}%, "
                        f"delta={r['delta_vs_baseline']*100:+.2f}%"
                    )

    # A2 结果
    print(f"\n  eta^2(句子) median [EEG]:  {eta2_eeg['eta2_sentence']['median']:.6f}")
    print(f"  eta^2(被试) median [EEG]:  {eta2_eeg['eta2_subject']['median']:.6f}")
    if "eta2_session" in eta2_eeg:
        print(f"  eta^2(session) median [EEG]: {eta2_eeg['eta2_session']['median']:.6f}"
              f"  (r_session_vs_sent={eta2_eeg.get('r_session_vs_sent')})")
    print(f"  eta^2(句子) median [噪声]: {eta2_noise['eta2_sentence']['median']:.6f}")
    print(f"  eta^2(被试) median [噪声]: {eta2_noise['eta2_subject']['median']:.6f}")
    print(f"  结论: {eta2_eeg.get('conclusion', 'N/A')}"
          f"  / session: {eta2_eeg.get('session_conclusion', 'N/A')}")

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

    # A3-SessionRetrieval 结果
    if "A3_session_retrieval" in all_results:
        print("\n  A3-SessionRetrieval 同被试跨 Session 聚合检索:")
        for source_key in ["eeg", "noise"]:
            sr = all_results["A3_session_retrieval"].get(source_key, {})
            for agg_key in ["aggregated_all", "aggregated_task1_sr"]:
                r = sr.get(agg_key, {})
                if "error" in r:
                    print(f"    [{source_key}] {agg_key}: {r['error']}"
                          f" (n_common={r.get('n_common')})")
                else:
                    print(
                        f"    [{source_key}] {agg_key}: M={r['n_sentences']}, "
                        f"R@1={r['r@1']*100:.2f}% (baseline={r['random_baseline']*100:.2f}%), "
                        f"MRR={r['mrr']:.4f}"
                    )
            ps = sr.get("per_subject", {})
            if "error" not in ps and ps.get("n_subjects", 0) > 0:
                print(
                    f"    [{source_key}] per_subject: n={ps['n_subjects']}, "
                    f"R@1 mean={ps['r@1_mean']*100:.2f}%(+/-{ps['r@1_std']*100:.2f}), "
                    f"MRR mean={ps['mrr_mean']:.4f}"
                )

    print(sep)
    print(f"  输出目录: {args.output_dir}")
    print(sep)
    logger.info("诊断线 A 完成")


if __name__ == "__main__":
    main()
