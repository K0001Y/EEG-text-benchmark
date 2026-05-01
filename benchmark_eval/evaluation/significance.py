"""显著性检验统一封装模块。

封装两份细节文档中约定的 7 类检验方法，统一返回格式
`{"test": 方法名, "statistic": ..., "p": ..., "effect": {...}, "ci95": [...], "n": ..., "extra": {...}}`，
方便 A/B 两线直接拼装至 `significance_tests.json`。

覆盖方法：
  - wilcoxon_paired       : 小样本配对 Wilcoxon 符号秩 + Cohen's d_z + 配对 bootstrap CI
  - mannwhitney_u         : Mann-Whitney U + 秩移动效应量 r + Cohen's d + bootstrap CI
  - binomial_vs_baseline  : 二项检验 + Clopper-Pearson CI
  - bootstrap_mean_diff   : 按 query 有放回重采样，均值差 95% CI（可推导 p）
  - permutation_retrieval : 固定表示打乱 gt_idx 1000 次，返回 R@K null 分布与 p
  - ks_vs_uniform         : rank 分布 vs 随机均匀 Kolmogorov-Smirnov
  - friedman_nemenyi      : 多组配对 Friedman + 事后 Nemenyi + Kendall's W
  - kruskal_dunn          : Kruskal-Wallis + Dunn 事后 + eta^2_H
  - permutation_eta       : 方差分解 eta^2 标签置换 null 分布

以及多重比较校正：
  - holm_bonferroni(pvals)
  - bh_fdr(pvals)

所有函数均无副作用、无网络依赖、纯 CPU。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 常量与工具
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_N_PERM = 1000
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ═══════════════════════════════════════════════════════════════════════════
# 效应量
# ═══════════════════════════════════════════════════════════════════════════

def cohens_dz_paired(a: np.ndarray, b: np.ndarray) -> float:
    """配对样本 Cohen's d_z = mean(diff) / std(diff, ddof=1)."""
    diff = _as_array(a) - _as_array(b)
    if diff.size < 2:
        return float("nan")
    s = diff.std(ddof=1)
    if s < 1e-12:
        return 0.0
    return float(diff.mean() / s)


def cohens_d_indep(a: np.ndarray, b: np.ndarray) -> float:
    """独立样本 Cohen's d（pooled std）。"""
    a, b = _as_array(a), _as_array(b)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) /
                       (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def rank_biserial_r(u_stat: float, n_a: int, n_b: int) -> float:
    """Mann-Whitney U 对应的秩移动效应量 r = 1 - 2U/(n_a*n_b)."""
    if n_a <= 0 or n_b <= 0:
        return float("nan")
    return float(1.0 - (2.0 * u_stat) / (n_a * n_b))


def interpret_effect(d: float) -> str:
    d = abs(d)
    if math.isnan(d):
        return "unknown"
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════════════════
# 1) Wilcoxon 符号秩 + 配对 bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════

def wilcoxon_paired(a: Sequence[float], b: Sequence[float],
                    alternative: str = "two-sided",
                    n_boot: int = DEFAULT_N_BOOT,
                    seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """小样本配对 Wilcoxon 符号秩检验（文档约定 5 折 / 14 被试等场景）。"""
    from scipy.stats import wilcoxon

    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape or a.size < 2:
        return {"test": "wilcoxon", "error": "invalid_input", "n": int(a.size)}

    diff = a - b
    try:
        res = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        stat = _safe_float(res.statistic)
        p = _safe_float(res.pvalue)
    except ValueError:
        # 所有差值为 0
        return {"test": "wilcoxon", "statistic": float("nan"), "p": 1.0,
                "n": int(a.size), "effect": {"cohens_dz": 0.0}, "ci95": [0.0, 0.0]}

    dz = cohens_dz_paired(a, b)

    # 配对 bootstrap 均值差 95% CI
    rng = _rng(seed)
    n = diff.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "test": "wilcoxon",
        "statistic": stat,
        "p": p,
        "n": int(n),
        "effect": {"cohens_dz": dz, "label": interpret_effect(dz),
                   "mean_diff": _safe_float(diff.mean())},
        "ci95": [_safe_float(lo), _safe_float(hi)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2) Mann-Whitney U（大样本独立比较）
# ═══════════════════════════════════════════════════════════════════════════

def mannwhitney_u(a: Sequence[float], b: Sequence[float],
                  alternative: str = "two-sided",
                  n_boot: int = DEFAULT_N_BOOT,
                  seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """独立样本 Mann-Whitney U + 秩移动 r + Cohen's d + 均值差 CI。"""
    from scipy.stats import mannwhitneyu

    a = _as_array(a)
    b = _as_array(b)
    if a.size < 2 or b.size < 2:
        return {"test": "mannwhitney_u", "error": "insufficient_samples",
                "n_a": int(a.size), "n_b": int(b.size)}

    res = mannwhitneyu(a, b, alternative=alternative)
    u = _safe_float(res.statistic)
    p = _safe_float(res.pvalue)
    r = rank_biserial_r(u, a.size, b.size)
    d = cohens_d_indep(a, b)

    # bootstrap 均值差 95% CI（独立样本分别重采样）
    rng = _rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        a_r = a[rng.integers(0, a.size, size=a.size)]
        b_r = b[rng.integers(0, b.size, size=b.size)]
        boots[i] = a_r.mean() - b_r.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "test": "mannwhitney_u",
        "statistic": u,
        "p": p,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "effect": {"rank_biserial_r": r, "cohens_d": d, "label": interpret_effect(d),
                   "mean_diff": _safe_float(a.mean() - b.mean())},
        "ci95": [_safe_float(lo), _safe_float(hi)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3) 二项检验 vs 随机基线 + Clopper-Pearson CI
# ═══════════════════════════════════════════════════════════════════════════

def binomial_vs_baseline(k: int, n: int, p0: float,
                         alternative: str = "greater") -> Dict[str, Any]:
    """精确二项检验：观察命中数 k / 样本量 n，零假设为 p0。

    文档场景：R@K vs K/M、A1 Top-K vs 1/130 等。
    """
    from scipy.stats import binomtest

    if n <= 0:
        return {"test": "binomial", "error": "n<=0"}
    res = binomtest(k=int(k), n=int(n), p=float(p0), alternative=alternative)
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    observed = k / n
    return {
        "test": "binomial",
        "statistic": int(k),
        "p": _safe_float(res.pvalue),
        "n": int(n),
        "baseline_p0": float(p0),
        "observed": _safe_float(observed),
        "effect": {"delta_vs_baseline": _safe_float(observed - p0)},
        "ci95": [_safe_float(ci.low), _safe_float(ci.high)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4) Bootstrap mean diff（按 query 有放回）
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float],
                        n_boot: int = DEFAULT_N_BOOT,
                        seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """配对样本均值差的 bootstrap 95% CI；p 由 CI 是否跨 0 推导。"""
    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape or a.size < 2:
        return {"test": "bootstrap_mean_diff", "error": "invalid_input"}
    diff = a - b
    rng = _rng(seed)
    n = diff.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # 双尾经验 p：(# boots 跨过 0) / n_boot
    prop = (boots > 0).mean() if diff.mean() >= 0 else (boots < 0).mean()
    p_emp = max(2.0 * (1.0 - prop), 1.0 / n_boot)
    return {
        "test": "bootstrap_mean_diff",
        "statistic": _safe_float(diff.mean()),
        "p": _safe_float(p_emp),
        "n": int(n),
        "effect": {"mean_diff": _safe_float(diff.mean()),
                   "cohens_dz": cohens_dz_paired(a, b)},
        "ci95": [_safe_float(lo), _safe_float(hi)],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5) Permutation test：检索指标（gt_idx 打乱）
# ═══════════════════════════════════════════════════════════════════════════

def _retrieval_metrics_from_sim(sim: np.ndarray, gt_idx: np.ndarray,
                                ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """从相似度矩阵 (N, M) 和真值下标 (N,) 计算 R@K / MRR / mean_rank。"""
    N = sim.shape[0]
    # argsort descending
    order = np.argsort(-sim, axis=1)
    # rank of gt for each row
    ranks = np.empty(N, dtype=np.int64)
    for i in range(N):
        # position of gt_idx[i] in order[i]
        ranks[i] = int(np.where(order[i] == gt_idx[i])[0][0]) + 1
    out: Dict[str, float] = {f"r@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / ranks).mean())
    out["mean_rank"] = float(ranks.mean())
    return out


def permutation_retrieval(sim: np.ndarray, gt_idx: Sequence[int],
                          ks: Sequence[int] = (1, 5, 10),
                          n_perm: int = DEFAULT_N_PERM,
                          seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """固定相似度矩阵，打乱 gt_idx → null R@K / MRR 分布。

    Args:
        sim: (N_query, M_candidate) 余弦相似度矩阵
        gt_idx: (N_query,) 真值候选下标
    """
    sim = np.asarray(sim, dtype=np.float32)
    gt_idx = np.asarray(gt_idx, dtype=np.int64)
    N, M = sim.shape

    observed = _retrieval_metrics_from_sim(sim, gt_idx, ks=ks)

    rng = _rng(seed)
    keys = list(observed.keys())
    null_vals: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_perm):
        perm = rng.permutation(N)
        gt_perm = gt_idx[perm]
        m = _retrieval_metrics_from_sim(sim, gt_perm, ks=ks)
        for k in keys:
            null_vals[k].append(m[k])

    out: Dict[str, Any] = {"test": "permutation_retrieval", "n_perm": int(n_perm),
                           "n_query": int(N), "n_candidate": int(M),
                           "observed": observed, "null": {}}
    for k, vals in null_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        obs = observed[k]
        # 单尾 p（观察值 >= null 的占比；对 mean_rank 需反向）
        if k == "mean_rank":
            p = float((arr <= obs).mean())
        else:
            p = float((arr >= obs).mean())
        p = max(p, 1.0 / n_perm)  # 下限
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out["null"][k] = {
            "mean": _safe_float(arr.mean()),
            "std": _safe_float(arr.std(ddof=1)),
            "ci95": [_safe_float(lo), _safe_float(hi)],
            "p": p,
            "z_deviation": _safe_float((obs - arr.mean()) / (arr.std(ddof=1) + 1e-12)),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6) KS 检验：rank 分布 vs 均匀
# ═══════════════════════════════════════════════════════════════════════════

def ks_vs_uniform(ranks: Sequence[int], M: int) -> Dict[str, Any]:
    """排名序列 vs 离散均匀 [1, M] 的 Kolmogorov-Smirnov 检验。"""
    from scipy.stats import kstest

    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size < 2 or M <= 1:
        return {"test": "ks_uniform", "error": "insufficient_samples"}
    # 连续化：假设排名均匀分布于 [1, M]
    res = kstest(ranks, "uniform", args=(1, M))
    return {
        "test": "ks_uniform",
        "statistic": _safe_float(res.statistic),
        "p": _safe_float(res.pvalue),
        "n": int(ranks.size),
        "M": int(M),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7) Friedman + Nemenyi（多组配对对比）
# ═══════════════════════════════════════════════════════════════════════════

def friedman_nemenyi(matrix: np.ndarray,
                     group_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Friedman 检验 + 事后 Nemenyi + Kendall's W。

    Args:
        matrix: (n_blocks, n_groups) — 每行是一个 block（如折/被试），列是比较对象（如条件/模型）。
    """
    from scipy.stats import friedmanchisquare

    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 2:
        return {"test": "friedman", "error": "invalid_shape", "shape": list(X.shape)}

    n_blocks, n_groups = X.shape
    groups = list(group_names) if group_names is not None else [f"g{i}" for i in range(n_groups)]

    cols = [X[:, j] for j in range(n_groups)]
    res = friedmanchisquare(*cols)
    stat = _safe_float(res.statistic)
    p = _safe_float(res.pvalue)

    # Kendall's W = chi2 / (n_blocks * (k - 1))
    k = n_groups
    w = stat / max(n_blocks * (k - 1), 1)

    out: Dict[str, Any] = {
        "test": "friedman",
        "statistic": stat,
        "p": p,
        "n_blocks": int(n_blocks),
        "n_groups": int(n_groups),
        "kendalls_w": _safe_float(w),
        "groups": groups,
    }

    # 事后 Nemenyi（尝试 scikit-posthocs；否则手写简化版）
    try:
        import scikit_posthocs as sp  # type: ignore
        posthoc = sp.posthoc_nemenyi_friedman(X)
        pdict: Dict[str, Any] = {}
        for i in range(k):
            for j in range(i + 1, k):
                pdict[f"{groups[i]}_vs_{groups[j]}"] = _safe_float(posthoc.iat[i, j])
        out["nemenyi"] = pdict
    except ImportError:
        # 简化版：基于平均秩差 + CD = q_alpha * sqrt(k(k+1) / (6 N))
        ranks = np.apply_along_axis(_tied_rank, 1, X)
        avg_rank = ranks.mean(axis=0)
        # q_{0.05} 取 Nemenyi 表近似值（k<=10）
        q_alpha_table = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
                         6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q = q_alpha_table.get(k, 3.164)
        cd = q * math.sqrt(k * (k + 1) / (6.0 * n_blocks))
        pairs = {}
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(avg_rank[i] - avg_rank[j])
                pairs[f"{groups[i]}_vs_{groups[j]}"] = {
                    "avg_rank_diff": _safe_float(diff),
                    "critical_difference": _safe_float(cd),
                    "significant": bool(diff > cd),
                }
        out["nemenyi_cd"] = {"cd_alpha_0.05": _safe_float(cd), "pairs": pairs,
                             "avg_ranks": {g: _safe_float(r) for g, r in zip(groups, avg_rank)}}
    return out


def _tied_rank(row: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(row)


# ═══════════════════════════════════════════════════════════════════════════
# 8) Kruskal-Wallis + Dunn 事后
# ═══════════════════════════════════════════════════════════════════════════

def kruskal_dunn(groups: Dict[str, Sequence[float]],
                 dunn_correction: str = "bh") -> Dict[str, Any]:
    """Kruskal-Wallis + Dunn 事后（多重校正）+ eta^2_H 效应量。"""
    from scipy.stats import kruskal

    keys = list(groups.keys())
    arrs = [np.asarray(groups[k], dtype=np.float64) for k in keys]
    arrs = [a for a in arrs if a.size > 0]
    if len(arrs) < 2 or sum(a.size for a in arrs) < 3:
        return {"test": "kruskal", "error": "insufficient_samples"}

    res = kruskal(*arrs)
    H = _safe_float(res.statistic)
    p = _safe_float(res.pvalue)
    N = sum(a.size for a in arrs)
    k = len(arrs)
    eta2_h = (H - k + 1) / (N - k) if N > k else float("nan")

    out: Dict[str, Any] = {
        "test": "kruskal",
        "statistic": H,
        "p": p,
        "n_groups": k,
        "n_total": int(N),
        "effect": {"eta2_H": _safe_float(eta2_h), "label": interpret_effect(eta2_h)},
        "groups": {k_: int(a.size) for k_, a in zip(keys, arrs)},
    }

    # Dunn 事后
    try:
        import scikit_posthocs as sp  # type: ignore
        import pandas as pd
        flat = np.concatenate(arrs)
        labels = np.concatenate([[k_] * a.size for k_, a in zip(keys, arrs)])
        df = pd.DataFrame({"val": flat, "grp": labels})
        mapping = {"bh": "fdr_bh", "holm": "holm", "bonferroni": "bonferroni"}
        method = mapping.get(dunn_correction, "fdr_bh")
        posthoc = sp.posthoc_dunn(df, val_col="val", group_col="grp", p_adjust=method)
        pdict = {}
        for i, gi in enumerate(keys):
            for j, gj in enumerate(keys):
                if j <= i:
                    continue
                pdict[f"{gi}_vs_{gj}"] = _safe_float(posthoc.loc[gi, gj])
        out["dunn_posthoc"] = {"method": method, "pairs": pdict}
    except ImportError:
        out["dunn_posthoc"] = {"error": "scikit-posthocs_not_installed"}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 9) Permutation test for eta^2
# ═══════════════════════════════════════════════════════════════════════════

def permutation_eta(feature_dim: np.ndarray, labels: Sequence[Any],
                    n_perm: int = DEFAULT_N_PERM,
                    seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """给定单维度特征向量与类别标签，计算原始 eta^2 及打乱标签后的 null 分布。

    eta^2 = SS_between / SS_total。
    """
    x = _as_array(feature_dim)
    labels = np.asarray(labels)
    if x.size != labels.size or x.size < 3:
        return {"test": "permutation_eta", "error": "invalid_input"}

    obs = _eta_squared(x, labels)

    rng = _rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        null[i] = _eta_squared(x, perm_labels)
    p = float((null >= obs).mean())
    p = max(p, 1.0 / n_perm)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {
        "test": "permutation_eta",
        "observed": _safe_float(obs),
        "null_mean": _safe_float(null.mean()),
        "null_std": _safe_float(null.std(ddof=1)),
        "null_ci95": [_safe_float(lo), _safe_float(hi)],
        "p": _safe_float(p),
        "n_perm": int(n_perm),
    }


def _eta_squared(x: np.ndarray, labels: np.ndarray) -> float:
    """单因素 eta^2 = SS_between / SS_total。"""
    gm = x.mean()
    ss_total = float(((x - gm) ** 2).sum())
    if ss_total < 1e-12:
        return 0.0
    uniq, inv = np.unique(labels, return_inverse=True)
    ss_between = 0.0
    for k in range(len(uniq)):
        mask = inv == k
        if mask.sum() == 0:
            continue
        gmi = x[mask].mean()
        ss_between += mask.sum() * (gmi - gm) ** 2
    return float(ss_between / ss_total)


# ═══════════════════════════════════════════════════════════════════════════
# 多重比较校正
# ═══════════════════════════════════════════════════════════════════════════

def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni 校正。返回排序后校正 p 及是否显著。"""
    p = _as_array(pvals)
    n = p.size
    if n == 0:
        return {"method": "holm_bonferroni", "adjusted": [], "significant": [], "alpha": alpha}
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    prev = 0.0
    for rank, idx in enumerate(order):
        factor = n - rank
        cur = min(1.0, p[idx] * factor)
        cur = max(cur, prev)  # 非递减
        adj[idx] = cur
        prev = cur
    sig = adj < alpha
    return {
        "method": "holm_bonferroni",
        "alpha": alpha,
        "alpha_adjusted_for_first": alpha / n,
        "adjusted": [_safe_float(x) for x in adj.tolist()],
        "significant": [bool(s) for s in sig.tolist()],
        "n_tests": int(n),
    }


def bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Hochberg FDR 校正。"""
    p = _as_array(pvals)
    n = p.size
    if n == 0:
        return {"method": "bh_fdr", "adjusted": [], "significant": [], "alpha": alpha}
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        cur = min(1.0, p[idx] * n / (rank + 1))
        cur = min(cur, prev)
        adj[idx] = cur
        prev = cur
    sig = adj < alpha
    return {
        "method": "bh_fdr",
        "alpha": alpha,
        "adjusted": [_safe_float(x) for x in adj.tolist()],
        "significant": [bool(s) for s in sig.tolist()],
        "n_tests": int(n),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 便捷 API：一组对比的打包
# ═══════════════════════════════════════════════════════════════════════════

def compare_pair(
    name_a: str, ranks_a: Sequence[int],
    name_b: str, ranks_b: Sequence[int],
    metric_ks: Sequence[int] = (1, 5, 10),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """B 线常用：两条件 per-query rank 对比（例如 real_vs_gaussian）。

    同时返回 Wilcoxon（rank 水平） + 派生 R@K/MRR 的 bootstrap mean-diff。
    """
    ra = np.asarray(ranks_a, dtype=np.float64)
    rb = np.asarray(ranks_b, dtype=np.float64)
    out: Dict[str, Any] = {
        "pair": f"{name_a}_vs_{name_b}",
        "n": int(min(ra.size, rb.size)),
    }
    out["wilcoxon_rank"] = wilcoxon_paired(ra, rb, n_boot=n_boot, seed=seed)
    # 每个 R@K 的逐样本 0/1 序列差
    for k in metric_ks:
        hits_a = (ra <= k).astype(np.float64)
        hits_b = (rb <= k).astype(np.float64)
        out[f"r@{k}_delta"] = bootstrap_mean_diff(hits_a, hits_b, n_boot=n_boot, seed=seed)
    # MRR 差
    out["mrr_delta"] = bootstrap_mean_diff(1.0 / ra, 1.0 / rb, n_boot=n_boot, seed=seed)
    # mean rank 差
    out["mean_rank_delta"] = bootstrap_mean_diff(ra, rb, n_boot=n_boot, seed=seed)
    return out


__all__ = [
    # 单项检验
    "wilcoxon_paired", "mannwhitney_u", "binomial_vs_baseline",
    "bootstrap_mean_diff", "permutation_retrieval",
    "ks_vs_uniform", "friedman_nemenyi", "kruskal_dunn", "permutation_eta",
    # 效应量
    "cohens_dz_paired", "cohens_d_indep", "rank_biserial_r", "interpret_effect",
    # 多重校正
    "holm_bonferroni", "bh_fdr",
    # 组合 API
    "compare_pair",
    # 常量
    "DEFAULT_N_PERM", "DEFAULT_N_BOOT", "DEFAULT_SEED",
]
