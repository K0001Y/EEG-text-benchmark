#!/usr/bin/env python3
"""EEG-To-Text (BrainTranslator) 编码器 EEG-文本检索评估。

EEG 编码路径：
  eeg_word_norm1d (B, L, 840)
    → additional_encoder (TransformerEncoder ×6, d_model=840)
    → fc1 + ReLU   → (B, L, 1024)
    → mean_pool    → (B, 1024)  [L2 归一化]

文本编码路径：
  text → BartTokenizer → BART Encoder (pretrained.model.encoder)
       → mean_pool → (B, 1024)  [L2 归一化]

两者处于同一 1024 维空间（BART encoder 输出维度），
BrainTranslator 通过端到端训练将 EEG 对齐到该空间。

用法（项目根目录下）：
  python benchmark_eval/scripts/run_eeg_to_text_retrieval.py \
      --data-path benchmark_eval/data/unified_zuco.pkl \
      --model-checkpoint models/EEG-To-Text-main/checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b32_20_30_5e-05_5e-07_unique_sent_EEG.pt \
      --output-dir benchmark_eval/test_outputs/eval_eeg_to_text_retrieval \
      --phase test
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 路径 ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)
PROJ_ROOT = os.path.dirname(BENCH_DIR)
EEG2TEXT_DIR = os.path.join(PROJ_ROOT, "models", "EEG-To-Text-main")

for _p in [BENCH_DIR, EEG2TEXT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_processing.dataset import UnifiedDataset, custom_collate_fn
from utils.logging_utils import setup_logging, get_logger
from wrappers.eeg_to_text_wrapper import EEGToTextWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", default="test")
    p.add_argument("--model-type", default="bart", choices=["bart", "t5"])
    p.add_argument("--eeg-batch-size", type=int, default=32)
    p.add_argument("--text-batch-size", type=int, default=64)
    return p.parse_args()


# ── 工具函数 ──────────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 加权平均池化。mask: 1=有效, 0=填充. (B,L,D) → (B,D)"""
    m = mask.float().unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode_texts(bart_encoder, tokenizer, texts: List[str],
                 device, bs=64, logger=None):
    """BART Encoder 编码文本列表 → L2 归一化向量 (N, 1024)"""
    vecs = []
    total = (len(texts) + bs - 1) // bs
    for i in range(0, len(texts), bs):
        batch = texts[i: i + bs]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        with torch.no_grad():
            out = bart_encoder(input_ids=ids, attention_mask=attn)
            v = F.normalize(mean_pool(out.last_hidden_state, attn), dim=-1)
        vecs.append(v.cpu())
        if logger and (i // bs + 1) % 10 == 0:
            logger.info("  text enc %d/%d", i // bs + 1, total)
    return torch.cat(vecs, 0)


def encode_eegs(brain_translator, eeg_batches, mask_batches, device, logger=None):
    """BrainTranslator additional_encoder + fc1 编码 EEG → L2 归一化向量 (N, 1024)

    编码路径：
      EEG (B, L, 840) → additional_encoder → fc1 + ReLU → (B, L, 1024)
                       → mean_pool         → (B, 1024)
    """
    vecs = []
    total = len(eeg_batches)
    for bi, (eeg_t, mask_t) in enumerate(zip(eeg_batches, mask_batches)):
        eeg = eeg_t.to(device)
        mask = mask_t.to(device)
        mask_invert = (1 - mask).bool()   # src_key_padding_mask: True=padding
        with torch.no_grad():
            # addin_forward: additional_encoder → fc1 + ReLU → (B, L, 1024)
            emb = brain_translator.addin_forward(eeg, mask_invert)
            v = F.normalize(mean_pool(emb, mask), dim=-1)
        vecs.append(v.cpu())
        if logger and (bi + 1) % 20 == 0:
            logger.info("  eeg enc %d/%d", bi + 1, total)
    return torch.cat(vecs, 0)


def retrieval_metrics(eeg_vecs, text_vecs, gt_idx, ks=(1, 5, 10)):
    """余弦相似度排名 → R@K + MRR。返回 (metrics_dict, ranks_tensor)"""
    sim = eeg_vecs @ text_vecs.T   # (N_eeg, N_texts)
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
    """按 task / subject / dataset 分组"""
    def _by(field):
        grps = defaultdict(list)
        for i, m in enumerate(meta_list):
            grps[m.get(field, "unknown")].append(i)
        out = {}
        for gval, idxs in grps.items():
            gm, gr = retrieval_metrics(eeg_vecs[idxs], text_vecs,
                                       [gt_idx[i] for i in idxs], ks)
            out[gval] = {"sample_count": len(idxs), "metrics": gm,
                         "mean_rank": float(gr.mean()),
                         "median_rank": float(gr.median())}
        return out
    return {"by_task": _by("task"), "by_subject": _by("subject"),
            "by_dataset": _by("dataset")}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="retrieval_eval.log")
    logger.info("EEG-To-Text Retrieval Eval | args=%s", vars(args))

    # 1. 数据集
    ds = UnifiedDataset(args.data_path, phase=args.phase)
    logger.info("Dataset: %d samples (phase=%s)", len(ds), args.phase)

    # 2. 加载模型（复用现有 Wrapper）
    logger.info("Loading EEGToTextWrapper...")
    wrapper = EEGToTextWrapper(
        model_checkpoint=args.model_checkpoint,
        model_type=args.model_type,
    )
    device = wrapper.device
    brain_translator = wrapper.model
    tokenizer = wrapper.tokenizer

    # 获取 BART encoder（用于文本编码）
    # BartForConditionalGeneration → .model (BartModel) → .encoder (BartEncoder)
    bart_encoder = brain_translator.pretrained.model.encoder
    bart_encoder.eval()
    logger.info("Model on device: %s", device)

    # 3. 收集数据
    logger.info("Collecting samples...")
    dl = DataLoader(ds, batch_size=args.eeg_batch_size, shuffle=False,
                    num_workers=0, collate_fn=custom_collate_fn)

    ref_texts, meta_list, eeg_bufs, mask_bufs = [], [], [], []
    for batch in dl:
        for i in range(len(batch["idx"])):
            ref_texts.append(batch["reference_text"][i])
            meta_list.append(dict(batch["meta"][i]))
        # EEG-To-Text 使用 eeg_word_norm1d + mask_word
        eeg_bufs.append(batch.get("eeg_word_norm1d", batch["eeg"]))
        mask_bufs.append(batch.get("mask_word", batch["mask"]))

    # 候选文本池（去重）
    unique_texts = list(dict.fromkeys(ref_texts))
    t2i = {t: i for i, t in enumerate(unique_texts)}
    gt_idx = [t2i[t] for t in ref_texts]
    N, M = len(ref_texts), len(unique_texts)
    logger.info("Queries: %d | Candidates: %d | Random R@1 ≈ %.4f%%",
                N, M, 100.0 / M)

    # 4. 编码文本
    logger.info("Encoding %d texts (bs=%d)...", M, args.text_batch_size)
    text_vecs = encode_texts(bart_encoder, tokenizer, unique_texts, device,
                             bs=args.text_batch_size, logger=logger)
    logger.info("text_vecs: %s", tuple(text_vecs.shape))

    # 5. 编码 EEG
    logger.info("Encoding %d EEGs (bs=%d)...", N, args.eeg_batch_size)
    eeg_vecs = encode_eegs(brain_translator, eeg_bufs, mask_bufs, device, logger=logger)
    logger.info("eeg_vecs: %s", tuple(eeg_vecs.shape))

    # 6. 计算指标
    logger.info("Computing retrieval metrics...")
    overall, ranks = retrieval_metrics(eeg_vecs, text_vecs, gt_idx)
    overall.update({
        "num_queries": N,
        "candidate_pool_size": M,
        "random_baseline_r@1": round(1.0 / M, 6),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(ranks.median()),
    })
    grp = grouped_metrics(eeg_vecs, text_vecs, gt_idx, meta_list)

    # 7. 保存
    result = {"overall": overall, "grouped": grp}
    out_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 8. 打印
    sep = "=" * 60
    print(f"\n{sep}")
    print("EEG-TO-TEXT (BrainTranslator) RETRIEVAL RESULTS")
    print(sep)
    print(f"  R@1          = {overall['r@1']:.4f}  ({overall['r@1']*100:.2f}%)")
    print(f"  R@5          = {overall['r@5']:.4f}  ({overall['r@5']*100:.2f}%)")
    print(f"  R@10         = {overall['r@10']:.4f}  ({overall['r@10']*100:.2f}%)")
    print(f"  MRR          = {overall['mrr']:.4f}")
    print(f"  Mean Rank    = {overall['mean_rank']:.1f}")
    print(f"  Median Rank  = {overall['median_rank']:.1f}")
    print(f"  # Queries    = {N}")
    print(f"  # Candidates = {M}")
    print(f"  Random R@1   ≈ {overall['random_baseline_r@1']*100:.4f}%")
    print(sep)
    print(f"  Saved → {out_path}")
    print(sep)
    logger.info("Done. Results → %s", out_path)


if __name__ == "__main__":
    main()
