"""Retrieval evaluation utilities shared across model-specific scripts.

This module collects common helpers used by all EEG-to-text retrieval
scripts (CET-MAE, EEG2Text, EEG-To-Text, GLIM) to avoid code duplication.
"""

from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked mean pooling.

    Args:
        hidden: Tensor of shape (B, L, D).
        mask: Optional tensor of shape (B, L) where 1=valid, 0=padding.
              If None, averages over all time steps.

    Returns:
        Tensor of shape (B, D).
    """
    if mask is None:
        return hidden.mean(1)
    m = mask.float().unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode_texts(
    encoder,
    tokenizer,
    texts: List[str],
    device,
    bs: int = 64,
    logger=None,
    log_interval: int = 10,
) -> torch.Tensor:
    """Encode a list of texts with an encoder + tokenizer → L2-normalized vectors.

    Args:
        encoder: Text encoder model (e.g. BART encoder, T5 encoder).
        tokenizer: Corresponding tokenizer.
        texts: List of text strings.
        device: torch device.
        bs: Batch size.
        logger: Optional logger for progress reporting.
        log_interval: Log every N batches.

    Returns:
        Tensor of shape (N, D) with L2-normalized vectors.
    """
    vecs = []
    total = (len(texts) + bs - 1) // bs
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        tok = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        with torch.no_grad():
            out = encoder(input_ids=ids, attention_mask=attn)
            v = F.normalize(mean_pool(out.last_hidden_state, attn), dim=-1)
        vecs.append(v.cpu())
        if logger and (i // bs + 1) % log_interval == 0:
            logger.info("  text enc %d/%d", i // bs + 1, total)
    return torch.cat(vecs, 0)


def retrieval_metrics(eeg_vecs, text_vecs, gt_idx, ks=(1, 5, 10)):
    """Cosine-similarity ranking → R@K + MRR.

    Args:
        eeg_vecs: Tensor (N_eeg, D).
        text_vecs: Tensor (N_text, D).
        gt_idx: List[int] mapping each EEG query to its ground-truth text index.
        ks: Tuple of K values for Recall@K.

    Returns:
        (metrics_dict, ranks_tensor)
    """
    sim = eeg_vecs @ text_vecs.T  # (N_eeg, N_texts)
    ranks = []
    for i, g in enumerate(gt_idx):
        order = torch.argsort(sim[i], descending=True)
        rank = (order == g).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    r = torch.tensor(ranks, dtype=torch.float)
    m = {f"r@{k}": float((r <= k).float().mean()) for k in ks}
    m["mrr"] = float((1.0 / r).mean())
    return m, r


def grouped_metrics(eeg_vecs, text_vecs, gt_idx, meta_list, ks=(1, 5, 10)):
    """Compute retrieval metrics grouped by task / subject / dataset.

    Args:
        eeg_vecs: Tensor (N_eeg, D).
        text_vecs: Tensor (N_text, D).
        gt_idx: List[int] ground-truth indices.
        meta_list: List[dict] with keys 'task', 'subject', 'dataset'.
        ks: Tuple of K values.

    Returns:
        Dict with keys 'by_task', 'by_subject', 'by_dataset'.
    """

    def _by(field):
        grps = defaultdict(list)
        for i, m in enumerate(meta_list):
            grps[m.get(field, "unknown")].append(i)
        out = {}
        for gval, idxs in grps.items():
            gm, gr = retrieval_metrics(
                eeg_vecs[idxs], text_vecs, [gt_idx[i] for i in idxs], ks
            )
            out[gval] = {
                "sample_count": len(idxs),
                "metrics": gm,
                "mean_rank": float(gr.mean()),
                "median_rank": float(gr.median()),
            }
        return out

    return {
        "by_task": _by("task"),
        "by_subject": _by("subject"),
        "by_dataset": _by("dataset"),
    }
