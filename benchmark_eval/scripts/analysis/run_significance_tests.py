#!/usr/bin/env python3
"""B 线显著性检验主脚本（跨噪声条件 + 跨模型汇总）。

读取 `test_outputs/line_b/{model}/{real,gaussian,shuffle,zero}/embeddings.npz`
中保存的 per-query ranks 向量，按照 `docs/detail/experiment_B_details.md` 的
规范执行：

  1. 每个模型：6 条件对 × 3 指标（R@1/R@5/R@10/MRR/mean_rank） Wilcoxon + bootstrap
  2. Permutation test（固定表示打乱 gt_idx）
  3. R@K vs 随机基线 K/M 的 binomial test
  4. rank 分布 vs 离散均匀 Kolmogorov-Smirnov
  5. 分组异质性：Kruskal-Wallis + Dunn（subject / task / dataset）
  6. 跨模型 Friedman + Nemenyi（4 模型 × per-query rank 矩阵）
  7. Holm-Bonferroni + BH-FDR 多重校正

输出：
  - 每个模型的 `significance_tests.json`（放在该模型目录下）
  - `test_outputs/line_b/significance_summary.json` （跨模型汇总）

用法：
  python benchmark_eval/scripts/analysis/run_significance_tests.py \
      --results-dir benchmark_eval/test_outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from evaluation.embedding_io import (
    load_embeddings, resolve_line_b_dir, save_significance_json,
    verify_l2_normalized,
)
from evaluation.significance import (
    compare_pair, permutation_retrieval, binomial_vs_baseline,
    ks_vs_uniform, kruskal_dunn, friedman_nemenyi,
    holm_bonferroni, bh_fdr, DEFAULT_N_PERM, DEFAULT_N_BOOT,
)
from utils.logging_utils import setup_logging

MODELS = ("cet_mae", "eeg_to_text", "eeg2text", "glim")
NOISE_CONDITIONS = ("real", "gaussian", "shuffle", "zero")
PAIR_ORDER = (
    ("real", "gaussian"),
    ("real", "shuffle"),
    ("real", "zero"),
    ("gaussian", "shuffle"),
    ("gaussian", "zero"),
    ("shuffle", "zero"),
)
METRICS_FOR_CORRECTION = ("r@1", "r@5", "r@10", "mrr", "mean_rank")
# 全局多重校正：len(MODELS) × len(PAIR_ORDER) × len(METRICS_FOR_CORRECTION)
# = 4 × 6 × 5 = 120
N_GLOBAL_TESTS = len(MODELS) * len(PAIR_ORDER) * len(METRICS_FOR_CORRECTION)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="test_outputs/ 根路径")
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM,
                   help="permutation test 次数")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                   help="bootstrap 重采样次数")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip-permutation", action="store_true",
                   help="跳过 permutation_retrieval（加速）")
    p.add_argument("--skip-grouped", action="store_true",
                   help="跳过 Kruskal-Wallis 分组检验")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 加载 + 基础对比
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_embeddings(results_dir: str, model: str) -> Dict[str, Dict[str, Any]]:
    """返回 {noise: payload_dict}（缺失的 noise 直接跳过并记录）。

    S-11: 加载后校验 v_eeg / v_text 是否 L2 归一化，未归一化则警告。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for noise in NOISE_CONDITIONS:
        target_dir = resolve_line_b_dir(results_dir, model, noise)
        emb_path = os.path.join(target_dir, "embeddings.npz")
        if not os.path.isfile(emb_path):
            continue
        payload = load_embeddings(emb_path)
        # L2 归一化校验（仅警告，不强制改写——保持原始数据真相）
        if "v_eeg" in payload:
            verify_l2_normalized(
                np.asarray(payload["v_eeg"], dtype=np.float32),
                name=f"{model}/{noise}/v_eeg")
        if "v_text" in payload:
            verify_l2_normalized(
                np.asarray(payload["v_text"], dtype=np.float32),
                name=f"{model}/{noise}/v_text")
        out[noise] = payload
        out[noise]["_path"] = emb_path
    return out


def _align_ranks_by_query(
    emb_by_noise: Dict[str, Dict[str, Any]],
    key_fields: Tuple[str, ...] = ("subjects", "sentence_ids", "tasks"),
) -> Tuple[List[Tuple[str, ...]], Dict[str, np.ndarray]]:
    """在多个 noise 条件间按行位置对齐 query，保留所有重复样本，保证严格配对。

    前提：同一模型在 4 种 noise 条件下由 UnifiedDataset 同序生成，embeddings.npz
    的行顺序与 (subject, sentence_id, task) + row_idx 完全一致。本函数对此前提
    进行强校验；若不一致则抛出 AssertionError。

    历史 Bug（2026-05 修复 / S-1）：旧实现使用 `set(keys)` 做交集去重，导致
    1858 条样本被坍缩到 54 个唯一三元组，配对检验功效损失约 97%。新实现改为
    按行索引对齐，保留全部 1858 条样本。

    Returns:
        query_keys: 每行的 (subject, sentence_id, task) + row_idx 键（长度 N）
        ranks_by_noise: {noise: aligned_ranks (N,)}
    """
    def _make_keys(payload):
        parts = [np.asarray(payload[f], dtype=object).tolist() for f in key_fields]
        return [tuple(str(x) for x in row) for row in zip(*parts)]

    noise_list = list(emb_by_noise.keys())
    keys_per_noise = {n: _make_keys(emb_by_noise[n]) for n in noise_list}
    ref_noise = noise_list[0]
    ref_keys = keys_per_noise[ref_noise]
    N = len(ref_keys)

    # 强校验：所有 noise 行数一致且键序列完全相同
    for n in noise_list[1:]:
        assert len(keys_per_noise[n]) == N, (
            f"row count mismatch: {ref_noise}={N} vs {n}={len(keys_per_noise[n])}"
        )
        assert keys_per_noise[n] == ref_keys, (
            f"row order mismatch between noise='{ref_noise}' and noise='{n}'; "
            "UnifiedDataset 生成顺序不一致，需排查采样逻辑。"
        )

    # 行内位置当作第四个键以区分 (subject, sent_id, task) 三元组下的重复样本
    augmented_keys = [k + (str(i),) for i, k in enumerate(ref_keys)]
    out: Dict[str, np.ndarray] = {}
    for n in noise_list:
        out[n] = np.asarray(emb_by_noise[n]["ranks"], dtype=np.int64)
    return augmented_keys, out


# ═══════════════════════════════════════════════════════════════════════════
# 单模型显著性检验
# ═══════════════════════════════════════════════════════════════════════════

def _pairwise_block(ranks_by_noise: Dict[str, np.ndarray],
                    n_boot: int, alpha: float, logger) -> Dict[str, Any]:
    """6 条件对 × 多指标的配对对比，含 Holm-Bonferroni 校正。"""
    pair_results: Dict[str, Any] = {}
    pvals_for_correction: List[float] = []
    p_keys: List[Tuple[str, str]] = []

    for (a, b) in PAIR_ORDER:
        if a not in ranks_by_noise or b not in ranks_by_noise:
            continue
        logger.info("  pair %s_vs_%s ...", a, b)
        res = compare_pair(a, ranks_by_noise[a], b, ranks_by_noise[b],
                           n_boot=n_boot)
        pair_results[f"{a}_vs_{b}"] = res
        # 收集显著性校正用 p 值
        for metric in METRICS_FOR_CORRECTION:
            key = f"{metric}_delta" if metric != "mrr" else "mrr_delta"
            # wilcoxon_rank 的 p 也进入校正
            if metric == "r@1":
                pvals_for_correction.append(float(res["wilcoxon_rank"].get("p", 1.0)))
                p_keys.append((f"{a}_vs_{b}", "wilcoxon_rank"))
            block = res.get(key, {})
            pvals_for_correction.append(float(block.get("p", 1.0)))
            p_keys.append((f"{a}_vs_{b}", key))

    # Holm-Bonferroni：单模型 18 组内部校正
    holm = holm_bonferroni(pvals_for_correction, alpha=alpha)
    for (pair_key, metric_key), adj, sig in zip(
            p_keys, holm["adjusted"], holm["significant"]):
        block = pair_results[pair_key].get(metric_key, {})
        block["p_adjusted_holm"] = adj
        block["significant_holm"] = bool(sig)
    pair_results["_correction"] = {
        "method": "holm_bonferroni",
        "alpha": alpha,
        "n_tests": holm["n_tests"],
        "alpha_first": holm["alpha_adjusted_for_first"],
    }
    return pair_results


def _vs_random_block(emb_by_noise: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """每个噪声条件：R@K vs 随机基线 K/M 的二项检验。"""
    out: Dict[str, Any] = {}
    for noise, payload in emb_by_noise.items():
        ranks = np.asarray(payload["ranks"], dtype=np.int64)
        N = ranks.size
        M = int(payload.get("n_candidate", payload["v_text"].shape[0]))
        row: Dict[str, Any] = {"n": N, "M": M}
        for k in (1, 5, 10):
            hits = int((ranks <= k).sum())
            p0 = k / M
            row[f"r@{k}"] = binomial_vs_baseline(hits, N, p0)
        out[noise] = row
    return out


def _rank_distribution_block(emb_by_noise: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for noise, payload in emb_by_noise.items():
        ranks = np.asarray(payload["ranks"], dtype=np.int64)
        M = int(payload.get("n_candidate", payload["v_text"].shape[0]))
        out[noise] = ks_vs_uniform(ranks, M)
    return out


def _grouped_block(emb_by_noise: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """每个 noise 条件：按 subject / task / dataset 做 Kruskal-Wallis + Dunn。"""
    out: Dict[str, Any] = {}
    for noise, payload in emb_by_noise.items():
        ranks = np.asarray(payload["ranks"], dtype=np.float64)
        row: Dict[str, Any] = {}
        for group_field in ("subjects", "tasks", "datasets"):
            if group_field not in payload:
                continue
            labels = np.asarray(payload[group_field], dtype=object)
            groups: Dict[str, List[float]] = {}
            for r, lb in zip(ranks, labels):
                groups.setdefault(str(lb), []).append(float(r))
            if len(groups) < 2:
                continue
            row[group_field] = kruskal_dunn(groups)
        out[noise] = row
    return out


def _permutation_block(emb_by_noise: Dict[str, Dict[str, Any]],
                       n_perm: int, logger) -> Dict[str, Any]:
    """每个 noise 条件：固定 v_eeg/v_text 打乱 gt_idx 的 null 分布。"""
    out: Dict[str, Any] = {}
    for noise, payload in emb_by_noise.items():
        v_eeg = np.asarray(payload["v_eeg"], dtype=np.float32)
        v_text = np.asarray(payload["v_text"], dtype=np.float32)
        gt_idx = np.asarray(payload["gt_idx"], dtype=np.int64)
        # 限制候选池很大时的耗时
        if gt_idx.size > 5000:
            logger.info("  [%s] skip permutation (N=%d too large)", noise, gt_idx.size)
            out[noise] = {"skipped": True, "reason": "N>5000"}
            continue
        logger.info("  [%s] permutation_retrieval n_perm=%d ...", noise, n_perm)
        sim = v_eeg @ v_text.T
        out[noise] = permutation_retrieval(sim, gt_idx, ks=(1, 5, 10),
                                           n_perm=n_perm)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 跨模型 Friedman + Nemenyi
# ═══════════════════════════════════════════════════════════════════════════

def _cross_model_friedman(
    all_model_data: Dict[str, Dict[str, Dict[str, Any]]],
    noise: str,
    logger,
) -> Dict[str, Any]:
    """对齐 4 模型的 real/noise 条件 per-query ranks，做 Friedman + Nemenyi。"""
    if len(all_model_data) < 2:
        return {"error": "fewer_than_2_models"}

    # 跨模型按行位置对齐：UnifiedDataset 测试集顺序在所有模型间一致
    # 历史 Bug（2026-05 修复 / S-1）：旧实现用 set 去重导致 1858 → 54。
    def _keys(payload, fields=("subjects", "sentence_ids", "tasks")):
        parts = [np.asarray(payload[f], dtype=object).tolist() for f in fields]
        return [tuple(str(x) for x in row) for row in zip(*parts)]

    model_keys: Dict[str, List[Tuple[str, ...]]] = {}
    model_ranks: Dict[str, np.ndarray] = {}
    for model, emb_by_noise in all_model_data.items():
        if noise not in emb_by_noise:
            continue
        payload = emb_by_noise[noise]
        model_keys[model] = _keys(payload)
        model_ranks[model] = np.asarray(payload["ranks"], dtype=np.int64)

    if len(model_keys) < 2:
        return {"error": f"noise_{noise}_missing_in_most_models"}

    # 强校验：所有模型行数一致且键序列完全相同
    models_ordered = list(model_keys.keys())
    ref_model = models_ordered[0]
    ref_keys = model_keys[ref_model]
    N = len(ref_keys)
    order_ok = True
    for m in models_ordered[1:]:
        if len(model_keys[m]) != N or model_keys[m] != ref_keys:
            order_ok = False
            break

    if order_ok:
        matrix = np.zeros((N, len(models_ordered)), dtype=np.float64)
        for j, m in enumerate(models_ordered):
            matrix[:, j] = model_ranks[m]
        n_common = N
    else:
        # 退化路径：按三元组 + 组内出现顺序的完整键对齐，仍保留重复样本
        def _augment(keys: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
            cnt: Dict[Tuple[str, ...], int] = {}
            out: List[Tuple[str, ...]] = []
            for k in keys:
                i = cnt.get(k, 0)
                out.append(k + (str(i),))
                cnt[k] = i + 1
            return out

        aug = {m: _augment(model_keys[m]) for m in models_ordered}
        common_keys = set(aug[ref_model])
        for m in models_ordered[1:]:
            common_keys &= set(aug[m])
        if len(common_keys) < 2:
            return {"error": "no_common_queries", "n_common": len(common_keys)}
        common_keys_sorted = sorted(common_keys)
        matrix = np.zeros((len(common_keys_sorted), len(models_ordered)), dtype=np.float64)
        for j, m in enumerate(models_ordered):
            key_to_pos = {k: i for i, k in enumerate(aug[m])}
            idxs = [key_to_pos[k] for k in common_keys_sorted]
            matrix[:, j] = model_ranks[m][idxs]
        n_common = len(common_keys_sorted)

    logger.info("  Friedman[noise=%s]: shape=%s models=%s",
                noise, matrix.shape, models_ordered)
    fr = friedman_nemenyi(matrix, group_names=models_ordered)
    fr["noise"] = noise
    fr["n_common_queries"] = int(n_common)
    return fr


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    line_b_root = os.path.join(args.results_dir, "line_b")
    os.makedirs(line_b_root, exist_ok=True)

    logger = setup_logging(line_b_root, log_name="significance_tests.log")
    logger.info("B 线显著性检验 | results=%s n_perm=%d n_boot=%d",
                args.results_dir, args.n_perm, args.n_boot)

    all_model_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ── 逐模型：加载 + 单模型检验 ──
    for model in MODELS:
        logger.info("=" * 60)
        logger.info("Model: %s", model)
        logger.info("=" * 60)
        emb_by_noise = _load_model_embeddings(args.results_dir, model)
        if not emb_by_noise:
            logger.warning("  skip: no embeddings.npz found for %s", model)
            continue
        logger.info("  loaded noises: %s", list(emb_by_noise.keys()))
        all_model_data[model] = emb_by_noise

        # pairwise
        _, ranks_by_noise = _align_ranks_by_query(emb_by_noise)
        if len(ranks_by_noise) >= 2:
            pairwise = _pairwise_block(ranks_by_noise, args.n_boot, args.alpha, logger)
        else:
            pairwise = {"error": "single_noise_condition_only"}

        vs_random = _vs_random_block(emb_by_noise)
        ks_block = _rank_distribution_block(emb_by_noise)
        grouped = {} if args.skip_grouped else _grouped_block(emb_by_noise)
        perm_block = ({} if args.skip_permutation
                      else _permutation_block(emb_by_noise, args.n_perm, logger))

        payload = {
            "model": model,
            "alpha": args.alpha,
            "alpha_adjusted_global": args.alpha / N_GLOBAL_TESTS,
            "n_noises_loaded": len(emb_by_noise),
            "pairwise": pairwise,
            "vs_random_baseline": vs_random,
            "rank_distribution": ks_block,
            "grouped": grouped,
            "permutation_retrieval": perm_block,
        }
        # 保存到该模型目录（首个 noise 的目录）——约定保存到 real，否则第一个
        preferred_noise = "real" if "real" in emb_by_noise else next(iter(emb_by_noise))
        out_dir = resolve_line_b_dir(args.results_dir, model, preferred_noise)
        save_significance_json(out_dir, payload)
        logger.info("  → %s/significance_tests.json", out_dir)
        # S-10: 索引副本只在嵌套布局（line_b/{model}/{noise}）下生成到 line_b/{model}/
        # flat 布局（eval_{model}_retrieval/）的 dirname 会指向 results_root，
        # 容易把摘要写到根目录污染 line_b/significance_summary.json，需显式校验。
        model_root_expected = os.path.join(args.results_dir, "line_b", model)
        out_dir_abs = os.path.abspath(out_dir)
        if os.path.abspath(model_root_expected) != out_dir_abs and \
                out_dir_abs.startswith(os.path.abspath(model_root_expected) + os.sep):
            save_significance_json(model_root_expected, payload)
            logger.info("  → %s/significance_tests.json (索引副本)",
                        model_root_expected)
        else:
            logger.info("  跳过索引副本（flat 布局或路径不在 line_b/%s 下）", model)

    # ── 跨模型 Friedman + Nemenyi ──
    logger.info("=" * 60)
    logger.info("Cross-model Friedman + Nemenyi")
    logger.info("=" * 60)
    cross_summary: Dict[str, Any] = {
        "models": list(all_model_data.keys()),
        "friedman_by_noise": {},
        "bh_fdr_global": None,
    }
    for noise in NOISE_CONDITIONS:
        cross_summary["friedman_by_noise"][noise] = _cross_model_friedman(
            all_model_data, noise, logger)

    # ── 全局 BH-FDR 校正（72 组跨模型对比的 p 值集合）──
    global_pvals: List[float] = []
    global_keys: List[str] = []
    for model, emb_by_noise in all_model_data.items():
        path = resolve_line_b_dir(args.results_dir, model,
                                  "real" if "real" in emb_by_noise else next(iter(emb_by_noise)))
        sig_path = os.path.join(path, "significance_tests.json")
        if not os.path.isfile(sig_path):
            continue
        with open(sig_path, "r", encoding="utf-8") as f:
            sig_data = json.load(f)
        pw = sig_data.get("pairwise", {})
        for pair_key, pair_block in pw.items():
            if pair_key.startswith("_"):
                continue
            for metric in METRICS_FOR_CORRECTION:
                key = f"{metric}_delta" if metric != "mrr" else "mrr_delta"
                metric_block = pair_block.get(key, {})
                p = metric_block.get("p")
                if p is None:
                    continue
                global_pvals.append(float(p))
                global_keys.append(f"{model}/{pair_key}/{metric}")

    if global_pvals:
        fdr = bh_fdr(global_pvals, alpha=args.alpha)
        cross_summary["bh_fdr_global"] = {
            "method": fdr["method"],
            "alpha": fdr["alpha"],
            "n_tests": fdr["n_tests"],
            "per_test": [
                {"key": k, "p": p, "p_adj": adj, "significant": bool(sig)}
                for k, p, adj, sig in zip(global_keys, global_pvals,
                                          fdr["adjusted"], fdr["significant"])
            ],
        }

    summary_path = os.path.join(line_b_root, "significance_summary.json")
    save_significance_json(line_b_root, cross_summary,
                           filename="significance_summary.json")
    logger.info("Cross-model summary → %s", summary_path)

    # ── 打印摘要 ──
    sep = "=" * 60
    print(f"\n{sep}")
    print("B 线显著性检验摘要")
    print(sep)
    print(f"  模型数: {len(all_model_data)}")
    for model in all_model_data:
        noises = list(all_model_data[model].keys())
        print(f"    {model}: noises={noises}")
    print(f"\n  跨模型 Friedman:")
    for noise, fr in cross_summary["friedman_by_noise"].items():
        if "error" in fr:
            print(f"    [{noise}] skip: {fr['error']}")
            continue
        print(f"    [{noise}] chi2={fr.get('statistic', float('nan')):.3f}  "
              f"p={fr.get('p', float('nan')):.4g}  W={fr.get('kendalls_w', float('nan')):.3f}")
    print(sep)
    print(f"  Summary → {summary_path}")
    print(sep)


if __name__ == "__main__":
    main()
