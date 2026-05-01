#!/usr/bin/env python3
"""B 线降维可视化主脚本（V1-V6）。

按 `docs/detail/experiment_B_details.md` 第 3 节的规划，读取
`test_outputs/line_b/{model}/{noise}/embeddings.npz` 中的嵌入向量，
统一执行 PCA(50) → t-SNE(2) 降维（perplexity=30，random_state=42），
产出 V1-V6 六种可视化图。

输出默认汇总到 `test_outputs/tsne_b/`。

用法：
  # 单模型 V1/V4/V5/V6 + 跨模型 V3 + 单模型 V2 合集
  python benchmark_eval/scripts/visualize_b_embeddings.py \
      --results-dir benchmark_eval/test_outputs \
      --model cet_mae --viz all

  # 指定子集
  python benchmark_eval/scripts/visualize_b_embeddings.py \
      --results-dir benchmark_eval/test_outputs --viz v3

  # 跨模型 V3 则不需要 --model
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from evaluation.embedding_io import load_embeddings, resolve_line_b_dir
from evaluation.visualization import (
    reduce_pca_tsne, plot_scatter_colored, plot_cross_modal,
    DEFAULT_PCA_DIM, DEFAULT_SEED,
)
from utils.logging_utils import setup_logging

MODELS = ("cet_mae", "eeg_to_text", "eeg2text", "glim")
NOISE_CONDITIONS = ("real", "gaussian", "shuffle", "zero")
VIZ_CHOICES = ("v1", "v2", "v3", "v4", "v5", "v6", "all")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="test_outputs/ 根路径")
    p.add_argument("--model", default=None, choices=list(MODELS) + [None],
                   help="目标模型（V1/V2/V4/V5/V6 需要；V3 跨模型可省略）")
    p.add_argument("--viz", default="all", choices=list(VIZ_CHOICES),
                   help="要生成的可视化种类")
    p.add_argument("--output-dir", default=None,
                   help="输出目录（默认 results-dir/tsne_b）")
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-points", type=int, default=4000,
                   help="采样上限（过大时随机下采样，防止 t-SNE OOM）")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════

def _downsample(X: np.ndarray, *parallel, max_n: int, seed: int):
    """对 X 做 min(max_n, len(X)) 采样，同步对并行数组。"""
    n = X.shape[0]
    if n <= max_n:
        return X, list(parallel)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return X[idx], [np.asarray(p)[idx] for p in parallel]


def _load_all_noises(results_dir: str, model: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for noise in NOISE_CONDITIONS:
        target_dir = resolve_line_b_dir(results_dir, model, noise)
        path = os.path.join(target_dir, "embeddings.npz")
        if not os.path.isfile(path):
            continue
        out[noise] = load_embeddings(path)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# V1：单模型四条件 EEG 嵌入对比
# ═══════════════════════════════════════════════════════════════════════════

def viz_v1(model: str, emb_by_noise: Dict[str, Dict[str, Any]],
           output_dir: str, perplexity: int, seed: int, max_points: int,
           logger) -> Optional[str]:
    if len(emb_by_noise) < 2:
        logger.info("V1[%s] skip: <2 noise conditions", model)
        return None
    Xs: List[np.ndarray] = []
    labels: List[str] = []
    for noise, payload in emb_by_noise.items():
        v = np.asarray(payload["v_eeg"], dtype=np.float32)
        Xs.append(v)
        labels.extend([noise] * v.shape[0])
    X = np.concatenate(Xs, axis=0)
    labels_arr = np.asarray(labels)

    X, (labels_arr,) = _downsample(X, labels_arr, max_n=max_points, seed=seed)
    logger.info("V1[%s] X=%s 4-cond tSNE ...", model, X.shape)
    coords = reduce_pca_tsne(X, perplexity=perplexity, seed=seed)

    out_path = os.path.join(output_dir, f"tsne_{model}_four_conditions_p{perplexity}.png")
    plot_scatter_colored(
        coords=coords, color_ids=list(labels_arr),
        output_path=out_path,
        title=f"V1 | {model} | 4 噪声条件 EEG 向量",
        legend=True,
    )
    logger.info("V1 → %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# V2：EEG + text 跨模态联合降维
# ═══════════════════════════════════════════════════════════════════════════

def viz_v2(model: str, emb_by_noise: Dict[str, Dict[str, Any]],
           output_dir: str, perplexity: int, seed: int, max_points: int,
           logger) -> Optional[str]:
    if "real" not in emb_by_noise:
        logger.info("V2[%s] skip: real missing", model)
        return None
    payload = emb_by_noise["real"]
    v_eeg = np.asarray(payload["v_eeg"], dtype=np.float32)
    v_text = np.asarray(payload["v_text"], dtype=np.float32)
    gt_idx = np.asarray(payload["gt_idx"], dtype=np.int64)

    # 按 max_points 采样 EEG（text 池通常较小）
    if v_eeg.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(v_eeg.shape[0], size=max_points, replace=False)
        idx.sort()
        v_eeg = v_eeg[idx]
        gt_idx = gt_idx[idx]

    combined = np.concatenate([v_eeg, v_text], axis=0)
    n_eeg = v_eeg.shape[0]
    logger.info("V2[%s] combined=%s tSNE ...", model, combined.shape)
    coords = reduce_pca_tsne(combined, perplexity=perplexity, seed=seed)
    coords_eeg, coords_text = coords[:n_eeg], coords[n_eeg:]

    # 真值对齐连线：随机抽 100 条
    rng = np.random.default_rng(seed)
    n_pair = min(100, n_eeg)
    pair_eeg = rng.choice(n_eeg, size=n_pair, replace=False)
    pairs = [(int(i), int(gt_idx[i])) for i in pair_eeg]

    out_path = os.path.join(output_dir, f"tsne_{model}_cross_modal_p{perplexity}.png")
    plot_cross_modal(coords_eeg, coords_text, out_path,
                     title=f"V2 | {model} | EEG + text 联合 t-SNE",
                     same_pairs=pairs)
    logger.info("V2 → %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# V3：跨模型同条件对比
# ═══════════════════════════════════════════════════════════════════════════

def viz_v3(results_dir: str, output_dir: str, perplexity: int,
           seed: int, max_points: int, logger) -> Optional[str]:
    """固定 real，拼接 4 模型的 V_eeg，按模型染色。"""
    Xs: List[np.ndarray] = []
    labels: List[str] = []
    for model in MODELS:
        emb_path = os.path.join(resolve_line_b_dir(results_dir, model, "real"),
                                "embeddings.npz")
        if not os.path.isfile(emb_path):
            logger.info("V3 skip %s: missing embeddings.npz", model)
            continue
        payload = load_embeddings(emb_path)
        v = np.asarray(payload["v_eeg"], dtype=np.float32)
        Xs.append(v)
        labels.extend([model] * v.shape[0])

    if len(Xs) < 2:
        logger.info("V3 skip: fewer than 2 models loaded")
        return None

    X = np.concatenate(Xs, axis=0)
    labels_arr = np.asarray(labels)
    X, (labels_arr,) = _downsample(X, labels_arr, max_n=max_points, seed=seed)
    logger.info("V3 X=%s 4-model tSNE ...", X.shape)
    coords = reduce_pca_tsne(X, perplexity=perplexity, seed=seed)

    out_path = os.path.join(output_dir, f"tsne_cross_model_real_p{perplexity}.png")
    plot_scatter_colored(
        coords=coords, color_ids=list(labels_arr),
        output_path=out_path,
        title=f"V3 | 跨模型 real 对比",
        legend=True,
    )
    logger.info("V3 → %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# V4：分组维度染色（subject / task / dataset）
# ═══════════════════════════════════════════════════════════════════════════

def viz_v4(model: str, emb_by_noise: Dict[str, Dict[str, Any]],
           output_dir: str, perplexity: int, seed: int, max_points: int,
           logger) -> List[str]:
    if "real" not in emb_by_noise:
        logger.info("V4[%s] skip: real missing", model)
        return []
    payload = emb_by_noise["real"]
    v = np.asarray(payload["v_eeg"], dtype=np.float32)
    subjects = np.asarray(payload["subjects"], dtype=object)
    tasks = np.asarray(payload["tasks"], dtype=object)
    datasets = np.asarray(payload["datasets"], dtype=object)

    v, (subjects, tasks, datasets) = _downsample(
        v, subjects, tasks, datasets, max_n=max_points, seed=seed)
    logger.info("V4[%s] X=%s tSNE ...", model, v.shape)
    coords = reduce_pca_tsne(v, perplexity=perplexity, seed=seed)

    outs: List[str] = []
    for cname, ids in (("subject", subjects), ("task", tasks), ("dataset", datasets)):
        out_path = os.path.join(output_dir, f"tsne_{model}_by_{cname}_p{perplexity}.png")
        plot_scatter_colored(coords=coords, color_ids=list(ids),
                             output_path=out_path,
                             title=f"V4 | {model} | colored by {cname}")
        logger.info("V4 → %s", out_path)
        outs.append(out_path)
    return outs


# ═══════════════════════════════════════════════════════════════════════════
# V5：shuffle 专项诊断
# ═══════════════════════════════════════════════════════════════════════════

def viz_v5(model: str, emb_by_noise: Dict[str, Dict[str, Any]],
           output_dir: str, perplexity: int, seed: int, max_points: int,
           logger) -> Optional[str]:
    if "real" not in emb_by_noise or "shuffle" not in emb_by_noise:
        logger.info("V5[%s] skip: need real+shuffle", model)
        return None
    real = emb_by_noise["real"]
    shuf = emb_by_noise["shuffle"]
    v_real = np.asarray(real["v_eeg"], dtype=np.float32)
    v_shuf = np.asarray(shuf["v_eeg"], dtype=np.float32)
    gt_real = np.asarray(real["gt_idx"], dtype=np.int64)
    gt_shuf = np.asarray(shuf["gt_idx"], dtype=np.int64)

    # 对齐：假设 real 与 shuffle 的样本顺序一致（正常情况下数据集 collate 是确定的）
    n = min(v_real.shape[0], v_shuf.shape[0])
    v_real, v_shuf, gt_real, gt_shuf = v_real[:n], v_shuf[:n], gt_real[:n], gt_shuf[:n]

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        idx.sort()
        v_real, v_shuf, gt_real, gt_shuf = (
            v_real[idx], v_shuf[idx], gt_real[idx], gt_shuf[idx]
        )
        n = idx.size

    combined = np.concatenate([v_real, v_shuf], axis=0)
    cond_labels = ["real"] * n + ["shuffle"] * n
    logger.info("V5[%s] combined=%s tSNE ...", model, combined.shape)
    coords = reduce_pca_tsne(combined, perplexity=perplexity, seed=seed)

    # 两张图：(a) 按条件染色 (b) 按真值句子染色
    out_paths = []
    p1 = os.path.join(output_dir, f"tsne_{model}_shuffle_diag_by_cond_p{perplexity}.png")
    plot_scatter_colored(coords=coords, color_ids=cond_labels,
                         output_path=p1,
                         title=f"V5 | {model} | real vs shuffle",
                         legend=True)
    out_paths.append(p1)
    logger.info("V5 (by cond) → %s", p1)

    # 按 gt 染色：取 top-K 最常见句子作为高亮；其余置为 "other"
    sent_labels = np.concatenate([gt_real.astype(str), gt_shuf.astype(str)])
    uniq, counts = np.unique(sent_labels, return_counts=True)
    topk = uniq[np.argsort(-counts)[:10]]
    sent_labels2 = np.where(np.isin(sent_labels, topk), sent_labels, "other")
    p2 = os.path.join(output_dir, f"tsne_{model}_shuffle_diag_by_gt_p{perplexity}.png")
    plot_scatter_colored(coords=coords, color_ids=list(sent_labels2),
                         output_path=p2,
                         title=f"V5 | {model} | real+shuffle colored by gt_id (top-10)",
                         legend=True)
    out_paths.append(p2)
    logger.info("V5 (by gt) → %s", p2)

    return p2


# ═══════════════════════════════════════════════════════════════════════════
# V6：zero 响应图
# ═══════════════════════════════════════════════════════════════════════════

def viz_v6(model: str, emb_by_noise: Dict[str, Dict[str, Any]],
           output_dir: str, perplexity: int, seed: int, max_points: int,
           logger) -> Optional[str]:
    if "zero" not in emb_by_noise:
        logger.info("V6[%s] skip: zero missing", model)
        return None
    payload = emb_by_noise["zero"]
    v = np.asarray(payload["v_eeg"], dtype=np.float32)
    subjects = np.asarray(payload["subjects"], dtype=object)

    v, (subjects,) = _downsample(v, subjects, max_n=max_points, seed=seed)

    # 相似度统计：平均两两余弦（已归一化则内积）
    # 用小样本估计，避免 N^2 爆炸
    k = min(500, v.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(v.shape[0], size=k, replace=False)
    sub = v[idx]
    sim = sub @ sub.T
    tri = sim[np.triu_indices(k, k=1)]
    mean_sim = float(tri.mean())
    logger.info("V6[%s] mean pairwise cosine ~ %.4f (n=%d)", model, mean_sim, k)

    logger.info("V6[%s] X=%s tSNE ...", model, v.shape)
    coords = reduce_pca_tsne(v, perplexity=perplexity, seed=seed)
    out_path = os.path.join(output_dir, f"tsne_{model}_zero_response_p{perplexity}.png")
    plot_scatter_colored(
        coords=coords, color_ids=list(subjects),
        output_path=out_path,
        title=f"V6 | {model} | zero EEG 响应（mean cos={mean_sim:.3f}）",
    )
    logger.info("V6 → %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "tsne_b")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir, log_name="visualize_b_embeddings.log")
    logger.info("B 线降维可视化 | viz=%s model=%s", args.viz, args.model)

    selected = {args.viz} if args.viz != "all" else {"v1", "v2", "v3", "v4", "v5", "v6"}

    # 单模型相关可视化
    single_model_vs = selected & {"v1", "v2", "v4", "v5", "v6"}
    if single_model_vs:
        models = [args.model] if args.model else list(MODELS)
        for model in models:
            emb_by_noise = _load_all_noises(args.results_dir, model)
            if not emb_by_noise:
                logger.warning("skip %s: no embeddings.npz", model)
                continue
            logger.info("Model %s loaded noises: %s", model, list(emb_by_noise.keys()))
            if "v1" in selected:
                viz_v1(model, emb_by_noise, output_dir,
                       args.perplexity, args.seed, args.max_points, logger)
            if "v2" in selected:
                viz_v2(model, emb_by_noise, output_dir,
                       args.perplexity, args.seed, args.max_points, logger)
            if "v4" in selected:
                viz_v4(model, emb_by_noise, output_dir,
                       args.perplexity, args.seed, args.max_points, logger)
            if "v5" in selected:
                viz_v5(model, emb_by_noise, output_dir,
                       args.perplexity, args.seed, args.max_points, logger)
            if "v6" in selected:
                viz_v6(model, emb_by_noise, output_dir,
                       args.perplexity, args.seed, args.max_points, logger)

    if "v3" in selected:
        viz_v3(args.results_dir, output_dir,
               args.perplexity, args.seed, args.max_points, logger)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"B 线降维可视化完成 → {output_dir}")
    print(sep)


if __name__ == "__main__":
    main()
