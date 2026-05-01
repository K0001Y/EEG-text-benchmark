#!/usr/bin/env python3
"""EEG2Text (ShallowNet+Transformer) 编码器 EEG-文本检索评估。

EEG 编码路径（正确格式：原始时序）：
  spectro pickle rawData (105, time) → .T → (time, 105) → pad → (24000, 105)
    → BrainTranslator.forward() → (B, 957, 1024)
    → mean_pool → (B, 1024)  [L2 归一化]

注意：eeg_spectro (374, 65) 格式与 pretrain-robert 权重不兼容（shallownet 需要105通道），
此脚本直接从原始 spectro pickle 读取 rawData 并以正确的 (24000, 105) 格式送入模型。

文本编码路径：
  text → BartTokenizer → BART Encoder (text_decoder.model.encoder)
       → mean_pool → (B, 1024)  [L2 归一化]

用法（项目根目录下）：
  python benchmark_eval/scripts/run_eeg2text_retrieval.py \
      --data-path benchmark_eval/data/unified_zuco.pkl \
      --model-checkpoint models/EEG2Text-main/checkpoints/decoding/best/...pt \
      --output-dir benchmark_eval/test_outputs/eval_eeg2text_retrieval \
      --phase test
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 路径 ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)
PROJ_ROOT = os.path.dirname(BENCH_DIR)
EEG2TEXT_DIR = os.path.join(PROJ_ROOT, "models", "EEG2Text-main")

for _p in [BENCH_DIR, EEG2TEXT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_processing.dataset import UnifiedDataset, custom_collate_fn, _generate_derangement
from utils.logging_utils import setup_logging, get_logger
from wrappers.eeg2text_wrapper import EEG2TextWrapper

# 原始 spectro pickle 路径映射
SPECTRO_PICKLE_PATHS = {
    "task1-SR":    "dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-spectro.pickle",
    "task2-NR":    "dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle",
    "task3-TSR":   "dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-spectro.pickle",
    "task2-NR-2.0": "dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-spectro.pickle",
}

MAX_RAW_LEN = 24000   # BrainTranslator shallownet 期望的时间步数
RAW_CHANNELS = 105    # EEG 通道数
NOISE_TYPES = ("real", "gaussian", "shuffle", "zero")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", default="test")
    p.add_argument("--noise-type", default="real", choices=NOISE_TYPES,
                   help="噪声条件: real(默认)/gaussian/shuffle/zero")
    p.add_argument("--eeg-batch-size", type=int, default=16)
    p.add_argument("--text-batch-size", type=int, default=64)
    return p.parse_args()


# ── 工具函数 ──────────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """mask 加权平均池化。mask=None 时对所有时间步平均。(B,L,D) → (B,D)"""
    if mask is None:
        return hidden.mean(1)
    m = mask.float().unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def build_lookup_table(logger) -> Dict[Tuple, np.ndarray]:
    """从所有 spectro pickle 构建 (task, subject, idx) → rawData 查找表。

    rawData 形状: (105, time_steps)
    """
    lut: Dict[Tuple, np.ndarray] = {}
    for task_name, rel_path in SPECTRO_PICKLE_PATHS.items():
        pkl_path = os.path.join(EEG2TEXT_DIR, rel_path)
        if not os.path.isfile(pkl_path):
            logger.warning("Spectro pickle not found: %s", pkl_path)
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")  # {subject: [dict, ...]}
        for subj, sent_list in data.items():
            for idx, sent_obj in enumerate(sent_list):
                if sent_obj is None:
                    continue
                raw = sent_obj.get("sentence_level_EEG", {}).get("rawData")
                if raw is not None:
                    lut[(task_name, subj, idx)] = raw
        logger.info("  %s: %d entries", task_name, sum(1 for k in lut if k[0] == task_name))
    logger.info("Lookup table total: %d entries", len(lut))
    return lut


def raw_to_tensor(raw: np.ndarray) -> torch.Tensor:
    """rawData (105, time) → (24000, 105) 归一化 tensor。"""
    eeg = raw.T.astype(np.float32)            # (time, 105)
    time_steps = eeg.shape[0]
    if time_steps < MAX_RAW_LEN:
        pad = np.zeros((MAX_RAW_LEN - time_steps, RAW_CHANNELS), dtype=np.float32)
        eeg = np.concatenate([eeg, pad], axis=0)
    else:
        eeg = eeg[:MAX_RAW_LEN]
    t = torch.from_numpy(eeg)                 # (24000, 105)
    # 2D z-score (与 data_masked_raw_robert.normalize_2d 一致)
    mean = t.mean()
    std = t.std()
    if std > 1e-8:
        t = (t - mean) / std
    return t


def encode_eegs(brain_translator, raw_list: List[Optional[np.ndarray]],
                device, bs=16, logger=None,
                noise_type: str = "real", noise_seed: int = 42) -> torch.Tensor:
    """BrainTranslator.forward → mean_pool → L2 normalize → (N, 1024)

    三层架构实现层：在编码阶段应用噪声（与 UnifiedDataset 协调层种子策略一致）。
    - gaussian: 使用 seed=noise_seed+idx 生成 N(0,1) 噪声替代 rawData
    - zero: 全零张量替代 rawData
    - real/shuffle: 使用真实数据（shuffle 在收集阶段已重排）
    """
    vecs = []
    total_batches = (len(raw_list) + bs - 1) // bs
    for i in range(0, len(raw_list), bs):
        batch_raw = raw_list[i: i + bs]
        tensors = []
        for j, r in enumerate(batch_raw):
            sample_idx = i + j
            if noise_type == "zero":
                tensors.append(torch.zeros(MAX_RAW_LEN, RAW_CHANNELS))
            elif noise_type == "gaussian":
                rng = np.random.default_rng(noise_seed + sample_idx)
                noise = rng.normal(0.0, 1.0, (MAX_RAW_LEN, RAW_CHANNELS)).astype(np.float32)
                tensors.append(torch.from_numpy(noise))
            else:  # real 或 shuffle（shuffle 已在收集阶段重排）
                if r is not None:
                    tensors.append(raw_to_tensor(r))
                else:
                    tensors.append(torch.zeros(MAX_RAW_LEN, RAW_CHANNELS))
        eeg = torch.stack(tensors).to(device)   # (B, 24000, 105)
        with torch.no_grad():
            emb = brain_translator(eeg)          # (B, 957, 1024)
            v = F.normalize(mean_pool(emb), dim=-1)
        vecs.append(v.cpu())
        if logger and (i // bs + 1) % 10 == 0:
            logger.info("  eeg enc %d/%d", i // bs + 1, total_batches)
    return torch.cat(vecs, 0)


def encode_texts(bart_encoder, tokenizer, texts: List[str],
                 device, bs=64, logger=None) -> torch.Tensor:
    """BART Encoder 编码文本 → L2 归一化向量 (N, 1024)"""
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
        if logger and (i // bs + 1) % 5 == 0:
            logger.info("  text enc %d/%d", i // bs + 1, total)
    return torch.cat(vecs, 0)


def retrieval_metrics(eeg_vecs, text_vecs, gt_idx, ks=(1, 5, 10)):
    sim = eeg_vecs @ text_vecs.T
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
    # 噪声条件：不再自动追加输出目录后缀（由调用方负责指定完整路径）
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, log_name="retrieval_eval.log")
    logger.info("EEG2Text Retrieval Eval | args=%s", vars(args))

    # 1. 数据集
    ds = UnifiedDataset(args.data_path, phase=args.phase)
    logger.info("Dataset: %d samples (phase=%s, noise=%s)",
                len(ds), args.phase, args.noise_type)

    # 2. 构建 rawData 查找表（对于 gaussian/zero 仍需查找表以确定样本索引）
    logger.info("Building spectro pickle lookup table...")
    lut = build_lookup_table(logger)

    # 协调层：shuffle 时查询 UnifiedDataset 的权威 permutation
    shuffle_perm = None
    if args.noise_type == "shuffle":
        # 生成与 UnifiedDataset 一致的 derangement
        shuffle_perm = _generate_derangement(len(ds), seed=42)
        logger.info("Shuffle mode: using derangement permutation (seed=42)")

    # 3. 加载模型
    logger.info("Loading EEG2TextWrapper...")
    wrapper = EEG2TextWrapper(model_checkpoint=args.model_checkpoint)
    device = wrapper.device
    brain_translator = wrapper.model
    bart_encoder = wrapper.text_decoder.model.encoder
    tokenizer = wrapper.tokenizer
    brain_translator.eval()
    bart_encoder.eval()
    logger.info("Model on device: %s", device)

    # 4. 收集样本
    logger.info("Collecting samples...")
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=0, collate_fn=custom_collate_fn)

    ref_texts, meta_list, raw_list = [], [], []
    miss_count = 0
    # 收集所有样本的 rawData
    all_raw_data = []  # 用于 shuffle 重排
    for batch in dl:
        for i in range(len(batch["idx"])):
            meta = dict(batch["meta"][i])
            ref_texts.append(batch["reference_text"][i])
            meta_list.append(meta)
            key = (meta.get("task"), meta.get("subject"),
                   meta.get("sentence_index"))
            raw = lut.get(key)
            if raw is None:
                miss_count += 1
            all_raw_data.append(raw)

    # 三层架构实现层：shuffle 时按协调层 permutation 重排 rawData
    if shuffle_perm is not None:
        raw_list = [all_raw_data[shuffle_perm[i]] for i in range(len(all_raw_data))]
        logger.info("Applied shuffle permutation to raw_list")
    else:
        raw_list = all_raw_data

    if miss_count:
        logger.warning("Missing rawData for %d/%d samples (using zeros)", miss_count, len(raw_list))

    unique_texts = list(dict.fromkeys(ref_texts))
    t2i = {t: i for i, t in enumerate(unique_texts)}
    gt_idx = [t2i[t] for t in ref_texts]
    N, M = len(ref_texts), len(unique_texts)
    logger.info("Queries: %d | Candidates: %d | Random R@1 ≈ %.4f%%",
                N, M, 100.0 / M)

    # 5. 编码文本
    logger.info("Encoding %d texts...", M)
    text_vecs = encode_texts(bart_encoder, tokenizer, unique_texts, device,
                             bs=args.text_batch_size, logger=logger)
    logger.info("text_vecs: %s", tuple(text_vecs.shape))

    # 6. 编码 EEG（三层架构实现层：在编码阶段应用噪声）
    logger.info("Encoding %d EEGs (bs=%d, noise=%s)...", N, args.eeg_batch_size, args.noise_type)
    eeg_vecs = encode_eegs(brain_translator, raw_list, device,
                            bs=args.eeg_batch_size, logger=logger,
                            noise_type=args.noise_type, noise_seed=42)
    logger.info("eeg_vecs: %s", tuple(eeg_vecs.shape))

    # 7. 计算指标
    logger.info("Computing retrieval metrics...")
    overall, ranks = retrieval_metrics(eeg_vecs, text_vecs, gt_idx)
    overall.update({
        "num_queries": N,
        "candidate_pool_size": M,
        "random_baseline_r@1": round(1.0 / M, 6),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(ranks.median()),
        "missing_raw_count": miss_count,
    })
    grp = grouped_metrics(eeg_vecs, text_vecs, gt_idx, meta_list)

    # 8. 保存指标
    result = {"overall": overall, "grouped": grp}
    out_path = os.path.join(output_dir, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 8b. 落盘嵌入向量
    try:
        from evaluation.embedding_io import save_embeddings
        emb_path = save_embeddings(
            output_dir=output_dir,
            v_eeg=eeg_vecs, v_text=text_vecs,
            gt_idx=gt_idx, meta_list=meta_list,
            noise_type=args.noise_type, model_name="eeg2text",
            ranks=ranks.to(int).tolist(),
            unique_texts=unique_texts,
        )
        logger.info("Embeddings → %s", emb_path)
    except Exception as _emb_err:  # pragma: no cover
        logger.warning("save_embeddings failed: %s", _emb_err)

    sep = "=" * 60
    print(f"\n{sep}")
    print("EEG2TEXT (BrainTranslator pretrain-robert) RETRIEVAL RESULTS")
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
    print(f"  Missing EEG  = {miss_count}")
    print(sep)
    print(f"  Saved → {out_path}")
    print(sep)
    logger.info("Done. Results → %s", out_path)


if __name__ == "__main__":
    main()
