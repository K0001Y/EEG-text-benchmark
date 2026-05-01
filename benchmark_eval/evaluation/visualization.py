"""降维可视化统一封装模块。

文档约定：PCA(50) → t-SNE(2)，perplexity ∈ {5, 30, 50}，`random_state=42`，
按 `subject_id / sentence_id / task / session` 四种方案染色。

本模块提供无副作用的纯函数 / 薄状态类：
  - reduce_pca_tsne(X, perplexity, ...) → coords(N,2)
  - reduce_pca_tsne_multi(X, perplexities=(5,30,50)) → {perp: coords}
  - plot_scatter_colored(coords, color_ids, ax|output_path, title, ...)
  - plot_scatter_paired_lines(coords, pairs, output_path, ...)
  - save_multi_color_tsne(X, output_dir, prefix, colors_dict, perplexities, ...)

所有绘图函数内部统一使用 matplotlib Agg 后端（HPC / 无显示环境兼容）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # 延迟导入 matplotlib，保持模块可在无 GUI 环境 import
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _MPL_OK = False


DEFAULT_PCA_DIM = 50
DEFAULT_PERPLEXITIES: Tuple[int, ...] = (5, 30, 50)
DEFAULT_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# PCA → t-SNE
# ═══════════════════════════════════════════════════════════════════════════

def reduce_pca_tsne(
    X: np.ndarray,
    perplexity: int = 30,
    pca_dim: int = DEFAULT_PCA_DIM,
    seed: int = DEFAULT_SEED,
    max_iter: int = 1000,
) -> np.ndarray:
    """单次 PCA(→pca_dim) + t-SNE(→2) 降维，返回 (N, 2)。"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError(f"reduce_pca_tsne: invalid shape {X.shape}")

    # NaN/Inf 清理：部分模型在 zero/shuffle 条件下输出可能存在非有限值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed)
        Xp = pca.fit_transform(X)
    else:
        Xp = X

    # t-SNE perplexity 不能 >= n_samples / 3
    eff_perp = max(2, min(perplexity, (X.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=eff_perp,
        random_state=seed,
        init="pca",
        max_iter=max_iter,
    )
    return np.asarray(tsne.fit_transform(Xp), dtype=np.float32)


def reduce_pca_tsne_multi(
    X: np.ndarray,
    perplexities: Sequence[int] = DEFAULT_PERPLEXITIES,
    pca_dim: int = DEFAULT_PCA_DIM,
    seed: int = DEFAULT_SEED,
) -> Dict[int, np.ndarray]:
    """对多组 perplexity 批量降维。共享 PCA 结果以节省算力。"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError(f"reduce_pca_tsne_multi: invalid shape {X.shape}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed)
        Xp = pca.fit_transform(X)
    else:
        Xp = X

    out: Dict[int, np.ndarray] = {}
    for perp in perplexities:
        eff_perp = max(2, min(int(perp), (X.shape[0] - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=seed,
                    init="pca", max_iter=1000)
        out[int(perp)] = np.asarray(tsne.fit_transform(Xp), dtype=np.float32)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 绘图原子
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_mpl() -> None:
    if not _MPL_OK:
        raise RuntimeError("matplotlib 未安装，无法绘图")


def _pick_cmap(n: int):
    # matplotlib 3.9+ 废弃 plt.cm.get_cmap，改用 matplotlib.colormaps.get_cmap
    try:
        import matplotlib as _mpl
        _get = _mpl.colormaps.get_cmap
        if n <= 10:
            return _get("tab10").resampled(n)
        if n <= 20:
            return _get("tab20").resampled(n)
        return _get("hsv").resampled(n)
    except Exception:
        if n <= 10:
            return plt.get_cmap("tab10", n)
        if n <= 20:
            return plt.get_cmap("tab20", n)
        return plt.get_cmap("hsv", n)


def plot_scatter_colored(
    coords: np.ndarray,
    color_ids: Sequence[Any],
    output_path: Optional[str] = None,
    title: str = "",
    ax=None,
    s: float = 8.0,
    alpha: float = 0.6,
    legend: Optional[bool] = None,
    max_legend: int = 20,
    dpi: int = 150,
) -> None:
    """按类别 id 染色的散点图。legend=None 时自动（<=20 类显示）。"""
    _ensure_mpl()
    coords = np.asarray(coords)
    color_ids = list(color_ids)
    if coords.shape[0] != len(color_ids):
        raise ValueError("coords 与 color_ids 长度不一致")

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
        own_fig = True

    unique_ids = sorted(set(color_ids), key=lambda x: str(x))
    cmap = _pick_cmap(len(unique_ids))
    id_to_color = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_ids)}

    ids_arr = np.asarray(color_ids)
    for uid in unique_ids:
        mask = ids_arr == uid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[id_to_color[uid]], s=s, alpha=alpha,
                   label=str(uid) if legend is not False else None)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    show_legend = (legend is True) or (legend is None and len(unique_ids) <= max_legend)
    if show_legend:
        ax.legend(fontsize=6, markerscale=2, loc="best",
                  ncol=min(3, (len(unique_ids) + 9) // 10))

    if own_fig and output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_scatter_paired_lines(
    coords: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    pair_labels: Optional[Sequence[Any]] = None,
    output_path: Optional[str] = None,
    title: str = "",
    group_colors: Optional[Dict[Any, Any]] = None,
    s: float = 25.0,
    alpha: float = 0.85,
    line_alpha: float = 0.25,
    dpi: int = 150,
) -> None:
    """带同组连线的散点图（用于展示同句子/同被试在两视图下的位置对应）。

    coords: (N, 2)
    pairs: [(i, j), ...] 每对需要连线的索引
    """
    _ensure_mpl()
    coords = np.asarray(coords)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    # 连线
    for (i, j) in pairs:
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="gray", alpha=line_alpha, linewidth=0.6)

    # 散点（按 pair_labels 染色）
    if pair_labels is not None:
        labels = list(pair_labels)
        unique = sorted(set(labels), key=lambda x: str(x))
        cmap = _pick_cmap(len(unique))
        for i, uid in enumerate(unique):
            mask = [l == uid for l in labels]
            pts = coords[mask]
            c = group_colors.get(uid) if group_colors else cmap(i % cmap.N)
            ax.scatter(pts[:, 0], pts[:, 1], c=[c], s=s, alpha=alpha,
                       label=str(uid))
        if len(unique) <= 20:
            ax.legend(fontsize=6, markerscale=2, loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=s, alpha=alpha)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 批量接口：一次产出多 perplexity × 多染色
# ═══════════════════════════════════════════════════════════════════════════

def save_multi_color_tsne(
    X: np.ndarray,
    output_dir: str,
    prefix: str,
    color_configs: Dict[str, Sequence[Any]],
    perplexities: Sequence[int] = DEFAULT_PERPLEXITIES,
    default_perp_for_all: int = 30,
    seed: int = DEFAULT_SEED,
    dpi: int = 150,
    logger=None,
) -> Dict[str, str]:
    """为单组特征产出 `{prefix}_by_{color}_p{perp}.png` 系列图。

    约定（与 A 线 `validate_eeg_signal.py` 保持一致）：
      - perplexity in {5, 30, 50}：仅 perplexity=default（30）对所有染色出图；
      - 其余 perplexity 只输出第一个染色键（通常是 subject）。
    """
    _ensure_mpl()
    os.makedirs(output_dir, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    coords_by_perp = reduce_pca_tsne_multi(X, perplexities=perplexities, seed=seed)

    color_keys = list(color_configs.keys())
    files: Dict[str, str] = {}
    for perp, coords in coords_by_perp.items():
        for i, cname in enumerate(color_keys):
            # perp != default 时只画第一个染色方案
            if perp != default_perp_for_all and i != 0:
                continue
            fname = f"{prefix}_by_{cname}_p{perp}.png"
            fpath = os.path.join(output_dir, fname)
            plot_scatter_colored(
                coords=coords,
                color_ids=color_configs[cname],
                output_path=fpath,
                title=f"{prefix} | t-SNE (perp={perp}) | colored by {cname}",
                dpi=dpi,
            )
            files[f"{cname}_p{perp}"] = fpath
            if logger:
                logger.info("  -> %s", fname)
    return files


def save_before_after_grid(
    X_before: np.ndarray,
    X_after: np.ndarray,
    output_dir: str,
    prefix: str,
    color_configs: Dict[str, Sequence[Any]],
    perplexity: int = 30,
    seed: int = DEFAULT_SEED,
    dpi: int = 150,
) -> Dict[str, str]:
    """A3-LP 去被试化前/后并排图（2 × len(color_configs)）。"""
    _ensure_mpl()
    os.makedirs(output_dir, exist_ok=True)
    X_before = np.asarray(X_before, dtype=np.float32)
    X_after = np.asarray(X_after, dtype=np.float32)

    coords_before = reduce_pca_tsne(X_before, perplexity=perplexity, seed=seed)
    coords_after = reduce_pca_tsne(X_after, perplexity=perplexity, seed=seed)

    files = {}
    ncols = 2
    nrows = len(color_configs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = np.asarray(axes).reshape(1, -1)
    for r, (cname, ids) in enumerate(color_configs.items()):
        plot_scatter_colored(coords_before, ids, ax=axes[r, 0],
                             title=f"before | {cname}")
        plot_scatter_colored(coords_after, ids, ax=axes[r, 1],
                             title=f"after | {cname}")
    fname = f"{prefix}_before_after_p{perplexity}.png"
    fpath = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    files["before_after"] = fpath
    return files


# ═══════════════════════════════════════════════════════════════════════════
# B 线可视化支持：V1-V6 的基础原子
# ═══════════════════════════════════════════════════════════════════════════

def plot_cross_modal(
    coords_eeg: np.ndarray,
    coords_text: np.ndarray,
    output_path: str,
    title: str = "cross-modal alignment",
    same_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    dpi: int = 150,
) -> None:
    """EEG / Text 两套 2D 坐标拼接到同一张图（异色异形）。

    same_pairs: [(eeg_idx, text_idx)] 连线展示真值对齐。
    """
    _ensure_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    # 合并坐标以统一尺度（注意：双模态坐标不在同一空间，这里只是拼图布局）
    ax.scatter(coords_text[:, 0], coords_text[:, 1],
               c="tab:red", marker="x", s=18, alpha=0.55, label="text")
    ax.scatter(coords_eeg[:, 0], coords_eeg[:, 1],
               c="tab:blue", marker="o", s=14, alpha=0.55, label="eeg")
    if same_pairs:
        for (ei, ti) in same_pairs:
            ax.plot([coords_eeg[ei, 0], coords_text[ti, 0]],
                    [coords_eeg[ei, 1], coords_text[ti, 1]],
                    color="gray", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "reduce_pca_tsne", "reduce_pca_tsne_multi",
    "plot_scatter_colored", "plot_scatter_paired_lines", "plot_cross_modal",
    "save_multi_color_tsne", "save_before_after_grid",
    "DEFAULT_PCA_DIM", "DEFAULT_PERPLEXITIES", "DEFAULT_SEED",
]
