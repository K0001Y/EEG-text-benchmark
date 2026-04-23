#!/usr/bin/env python3
"""CET-MAE 编码器 EEG-文本检索评估。

评估流程：
  1. 构建候选文本池：测试集所有唯一参考文本
  2. BART t_branch_encoder 编码文本 → 文本向量（与 CET-MAE 对比学习对齐空间一致）
  3. CET-MAE encoder（e_branch→fc_eeg→unify_branch）编码 EEG → EEG 向量
  4. 余弦相似度矩阵 → 排名 → R@1/R@5/R@10/MRR

用法（项目根目录下）：
  python benchmark_eval/scripts/run_cet_mae_retrieval.py \
      --data-path benchmark_eval/data/unified_zuco.pkl \
      --model-checkpoint models/CET-MAE/checkpoints/decoding/cet_mae_benchmark_best.pt \
      --output-dir benchmark_eval/test_outputs/eval_cet_mae_retrieval \
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
CET_MAE_DIR = os.path.join(PROJ_ROOT, "models", "CET-MAE")

for _p in [BENCH_DIR, CET_MAE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_processing.dataset import UnifiedDataset, custom_collate_fn
from utils.logging_utils import setup_logging, get_logger
from wrappers.cet_mae_wrapper import CETMAEWrapper


NOISE_TYPES = ("real", "gaussian", "shuffle", "zero")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", default="test")
    p.add_argument("--noise-type", default="real", choices=NOISE_TYPES,
                   help="噪声条件: real(默认)/gaussian/shuffle/zero")
    p.add_argument("--eeg-batch-size", type=int, default=32)
    p.add_argument("--text-batch-size", type=int, default=64)
    return p.parse_args()


# ── 工具函数 ──────────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 加权平均池化。mask: 1=有效, 0=填充. shape (B,L,D) → (B,D)"""
    m = mask.float().unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode_texts(model, tokenizer, texts: List[str], device, bs=64, logger=None):
    """BART t_branch_encoder 编码文本列表 → L2 归一化向量 (N, 1024)"""
    vecs = []
    for i in range(0, len(texts), bs):
        batch = texts[i: i + bs]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        with torch.no_grad():
            out = model.t_branch_encoder(input_ids=ids, attention_mask=attn)
            v = F.normalize(mean_pool(out.last_hidden_state, attn), dim=-1)
        vecs.append(v.cpu())
        if logger and (i // bs + 1) % 10 == 0:
            logger.info("  text enc %d/%d", i // bs + 1, (len(texts) + bs - 1) // bs)
    return torch.cat(vecs, 0)


def encode_eegs(model, eeg_batches, mask_batches, device, logger=None):
    """CET-MAE encoder 编码 EEG 批次列表 → L2 归一化向量 (N, 1024)"""
    vecs = []
    for bi, (eeg_t, mask_t) in enumerate(zip(eeg_batches, mask_batches)):
        eeg = eeg_t.to(device)
        mask = mask_t.to(device)
        inv = (1 - mask).bool()
        with torch.no_grad():
            x = eeg + model.pos_embed_e(eeg)
            x = model.e_branch(x, src_key_padding_mask=inv)
            x = model.act(model.fc_eeg(x))
            x = model.unify_branch(x, src_key_padding_mask=inv, modality="e")
            v = F.normalize(mean_pool(x, mask), dim=-1)
        vecs.append(v.cpu())
        if logger and (bi + 1) % 20 == 0:
            logger.info("  eeg enc %d/%d", bi + 1, len(eeg_batches))
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
                         "mean_rank": float(gr.mean()), "median_rank": float(gr.median())}
        return out
    return {"by_task": _by("task"), "by_subject": _by("subject"),
            "by_dataset": _by("dataset")}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # 噪声条件自动添加输出目录后缀
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, log_name="retrieval_eval.log")
    logger.info("CET-MAE Retrieval Eval | args=%s", vars(args))

    # 1. 数据集（根据噪声条件配置）
    ds_kwargs = dict(data_path=args.data_path, phase=args.phase)
    if args.noise_type == "gaussian":
        ds_kwargs.update(noise_mode=True, noise_type="gaussian")
    elif args.noise_type == "zero":
        ds_kwargs.update(noise_mode=True, noise_type="zero")
    elif args.noise_type == "shuffle":
        ds_kwargs.update(shuffle_mode=True)
    ds = UnifiedDataset(**ds_kwargs)
    logger.info("Dataset: %d samples (phase=%s, noise=%s)",
                len(ds), args.phase, args.noise_type)

    # 2. 加载模型
    logger.info("Loading CET-MAE wrapper...")
    wrapper = CETMAEWrapper(model_checkpoint=args.model_checkpoint)
    device = wrapper.device
    model = wrapper.model
    tok = wrapper.tokenizer
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
        eeg_bufs.append(batch["eeg_word_norm2d"])
        mask_bufs.append(batch["mask_word_with_sent"])

    # 候选文本池（去重）
    unique_texts = list(dict.fromkeys(ref_texts))
    t2i = {t: i for i, t in enumerate(unique_texts)}
    gt_idx = [t2i[t] for t in ref_texts]
    N, M = len(ref_texts), len(unique_texts)
    logger.info("Queries: %d | Candidate pool: %d | Random R@1 ≈ %.4f%%",
                N, M, 100.0 / M)

    # 4. 编码文本
    logger.info("Encoding %d texts (bs=%d)...", M, args.text_batch_size)
    text_vecs = encode_texts(model, tok, unique_texts, device,
                             bs=args.text_batch_size, logger=logger)
    logger.info("text_vecs: %s", tuple(text_vecs.shape))

    # 5. 编码 EEG
    logger.info("Encoding %d EEGs (bs=%d)...", N, args.eeg_batch_size)
    eeg_vecs = encode_eegs(model, eeg_bufs, mask_bufs, device, logger=logger)
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
    out_path = os.path.join(output_dir, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 8. 打印
    sep = "=" * 60
    print(f"\n{sep}")
    print("CET-MAE ENCODER RETRIEVAL RESULTS")
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
