"""B 线嵌入向量落盘 I/O 统一封装。

根据 `docs/detail/experiment_B_details.md` 的约定，每个 `eval_*_retrieval*/`
目录都需要产出 `embeddings.npz`（降维可视化用）与 per-query ranks（显著性检验用）。

字段约定：
  - v_eeg        : (N, D) float32  — EEG 编码向量（L2 归一化）
  - v_text       : (M, D) float32  — 文本编码向量（L2 归一化）
  - gt_idx       : (N,)  int64     — 每条 EEG query 对应的真值文本下标
  - subjects     : (N,)  object    — subject_id
  - tasks        : (N,)  object    — task 名
  - datasets     : (N,)  object    — dataset 名（ZuCo1/ZuCo2）
  - sentence_ids : (N,)  object    — 可选，句子 id
  - sessions     : (N,)  object    — 可选，session 标签
  - ranks        : (N,)  int64     — 对应模型在本 noise 条件下的 per-query rank
  - noise_type   : str             — real / gaussian / shuffle / zero
  - model_name   : str             — cet_mae / eeg_to_text / eeg2text / glim
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np


def save_embeddings(
    output_dir: str,
    v_eeg: Any,
    v_text: Any,
    gt_idx: Sequence[int],
    meta_list: Sequence[Dict[str, Any]],
    noise_type: str,
    model_name: str,
    ranks: Optional[Sequence[int]] = None,
    unique_texts: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "embeddings.npz",
) -> str:
    """将 B 线检索评估的嵌入 + 元信息一次性写入 `embeddings.npz`。

    Args:
        output_dir: 输出目录
        v_eeg, v_text: torch.Tensor 或 np.ndarray
        gt_idx: 真值下标（长度 N）
        meta_list: [{"subject": ..., "task": ..., "dataset": ..., "sentence_id": ..., "session": ...}]
        noise_type: real / gaussian / shuffle / zero
        model_name: 模型名
        ranks: 每条 query 的排名（可选，未提供时可由 v_eeg @ v_text.T 重算）
        unique_texts: 候选文本池
        extra: 额外字段
    Returns:
        写入的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    v_eeg_np = _to_numpy(v_eeg, dtype=np.float32)
    v_text_np = _to_numpy(v_text, dtype=np.float32)
    gt_idx_np = np.asarray(list(gt_idx), dtype=np.int64)

    subjects = np.array([m.get("subject", "unknown") for m in meta_list], dtype=object)
    tasks = np.array([m.get("task", "unknown") for m in meta_list], dtype=object)
    datasets = np.array([m.get("dataset", "unknown") for m in meta_list], dtype=object)
    sentence_ids = np.array([m.get("sentence_id", "") for m in meta_list], dtype=object)
    sessions = np.array([m.get("session", "") for m in meta_list], dtype=object)

    # ranks：优先用外部传入；否则重算
    if ranks is None:
        ranks_np = _compute_ranks(v_eeg_np, v_text_np, gt_idx_np)
    else:
        ranks_np = np.asarray(list(ranks), dtype=np.int64)

    payload: Dict[str, Any] = dict(
        v_eeg=v_eeg_np,
        v_text=v_text_np,
        gt_idx=gt_idx_np,
        ranks=ranks_np,
        subjects=subjects,
        tasks=tasks,
        datasets=datasets,
        sentence_ids=sentence_ids,
        sessions=sessions,
        noise_type=np.array(str(noise_type)),
        model_name=np.array(str(model_name)),
        n_query=np.array(int(gt_idx_np.size)),
        n_candidate=np.array(int(v_text_np.shape[0])),
    )
    if unique_texts is not None:
        payload["unique_texts"] = np.array(list(unique_texts), dtype=object)
    if extra:
        for k, v in extra.items():
            if k in payload:
                continue
            payload[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v

    out_path = os.path.join(output_dir, filename)
    np.savez(out_path, **payload)
    return out_path


def load_embeddings(path: str) -> Dict[str, Any]:
    """读取 `embeddings.npz`，自动把 0-d str/int 还原为 Python 原生类型。"""
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        arr = data[k]
        # 0-d object / str / int 恢复
        if arr.ndim == 0:
            val = arr.item()
            out[k] = val
        else:
            out[k] = arr
    return out


def _to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    """torch.Tensor / np.ndarray / list → np.ndarray."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _compute_ranks(v_eeg: np.ndarray, v_text: np.ndarray,
                   gt_idx: np.ndarray) -> np.ndarray:
    """从向量直接算 ranks（余弦相似度）。"""
    # v_eeg / v_text 已 L2 归一化时直接做内积即余弦相似度
    sim = v_eeg @ v_text.T
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(v_eeg.shape[0], dtype=np.int64)
    for i in range(v_eeg.shape[0]):
        pos = np.where(order[i] == gt_idx[i])[0]
        ranks[i] = int(pos[0]) + 1 if pos.size > 0 else v_text.shape[0]
    return ranks


# ═══════════════════════════════════════════════════════════════════════════
# 目录布线约定：test_outputs/line_b/{model}/{noise}/embeddings.npz
# ═══════════════════════════════════════════════════════════════════════════

def resolve_line_b_dir(results_root: str, model: str, noise: str = "real") -> str:
    """返回 `test_outputs/line_b/{model}/{noise}` 路径，兼容旧 flat 布局。"""
    primary = os.path.join(results_root, "line_b", model, noise)
    if os.path.isdir(primary):
        return primary
    # 回退到 flat 布局
    if noise == "real":
        return os.path.join(results_root, f"eval_{model}_retrieval")
    return os.path.join(results_root, f"eval_{model}_retrieval_{noise}")


def save_significance_json(output_dir: str, payload: Dict[str, Any],
                           filename: str = "significance_tests.json") -> str:
    """显著性检验结果统一落盘接口（保持类型原生，避免 numpy float 序列化问题）。"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return f
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, ensure_ascii=False, indent=2)
    return out_path


__all__ = [
    "save_embeddings", "load_embeddings",
    "resolve_line_b_dir", "save_significance_json",
]
