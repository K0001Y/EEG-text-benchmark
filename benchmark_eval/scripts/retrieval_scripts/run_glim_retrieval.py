#!/usr/bin/env python3
"""GLIM 模型 EEG-文本检索评估。

GLIM 明确使用 CLIP 损失训练 EEG-文本对齐，检索评估具有最直接意义。

EEG 编码路径：
  eeg_word_raw (B, MAX_LEN, 840)
    → convert_to_glim_format → (B, 1280, 128)
    → eeg_encoder(eeg, mask, prompt_embed) → (B, 96, 256)
    → aligner.embed_eeg → (B, 1024)  [L2 归一化]

文本编码路径：
  text → T5Tokenizer → T5 Encoder (text_model.get_encoder())
       → (B, L, 1024)
       → aligner.embed_text → (B, 1024)  [L2 归一化]

注意：EEG 向量和文本向量均经过 aligner 的 cross-attention 压缩（q_x/q_y），
      与 GLIM 训练时完全一致，因此该检索评估最能反映模型的真实对齐能力。

用法（项目根目录下）：
  python benchmark_eval/scripts/run_glim_retrieval.py \
      --data-path benchmark_eval/data/unified_zuco.pkl \
      --model-checkpoint models/GLIM-main/checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
      --output-dir benchmark_eval/test_outputs/eval_glim_retrieval \
      --phase test
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 路径 ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(THIS_DIR)
PROJ_ROOT = os.path.dirname(BENCH_DIR)

if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from data_processing.dataset import UnifiedDataset, custom_collate_fn
from utils.logging_utils import setup_logging, get_logger
from wrappers.glim_wrapper import GLIMWrapper


NOISE_TYPES = ("real", "gaussian", "shuffle", "zero")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", default="test")
    p.add_argument("--noise-type", default="real", choices=NOISE_TYPES,
                   help="噪声条件: real(默认)/gaussian/shuffle/zero")
    p.add_argument("--text-model-id", default="google/flan-t5-large")
    p.add_argument("--eeg-batch-size", type=int, default=16)
    p.add_argument("--text-batch-size", type=int, default=32)
    return p.parse_args()


# ── 编码函数 ──────────────────────────────────────────────────────────────

def encode_texts(glim_model, texts: List[str], device, bs=32, logger=None) -> torch.Tensor:
    """T5 Encoder + aligner.embed_text → L2 归一化 (N, 1024)

    完全复现 GLIM 内部的文本编码路径：
      tokenize → text_model.get_encoder() → aligner.embed_text → L2 normalize
    """
    tokenizer = glim_model.tokenizer
    text_encoder = glim_model.text_model.get_encoder()
    aligner = glim_model.aligner

    vecs = []
    total = (len(texts) + bs - 1) // bs
    for i in range(0, len(texts), bs):
        batch = texts[i: i + bs]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=96)
        ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        with torch.no_grad():
            out = text_encoder(input_ids=ids, attention_mask=attn, return_dict=True)
            hidden = out.last_hidden_state.float()   # (B, L, 1024), bfloat16→float32
            # aligner.embed_text: cross-attention with q_y → (B, 1024)
            y_emb = aligner.embed_text(hidden, attn)
            v = F.normalize(y_emb, dim=-1)
        vecs.append(v.cpu())
        if logger and (i // bs + 1) % 5 == 0:
            logger.info("  text enc %d/%d", i // bs + 1, total)
    return torch.cat(vecs, 0)


def encode_eegs(wrapper: GLIMWrapper, eeg_batches, mask_batches,
                meta_batches, device, logger=None) -> torch.Tensor:
    """EEG → GLIM encoder → aligner.embed_eeg → L2 归一化 (N, 1024)

    与 GLIMWrapper.encode_eeg 等价，但直接返回 eeg_emb_vector（L2 归一化）。
    """
    glim_model = wrapper.model
    aligner = glim_model.aligner
    vecs = []
    total = len(eeg_batches)

    for bi, (eeg_t, mask_t, meta_list) in enumerate(zip(eeg_batches, mask_batches, meta_batches)):
        eeg_input = eeg_t
        mask_input = mask_t
        with torch.no_grad():
            # 复用 wrapper 已有的格式转换和 prompt 提取逻辑
            glim_eeg, glim_mask = wrapper._convert_to_glim_format(eeg_input, mask_input)
            glim_eeg = glim_eeg.to(device)
            glim_mask = glim_mask.to(device)

            prompts = wrapper._extract_prompts_from_meta(meta_list, eeg_input.size(0))
            prompt_ids = glim_model.p_embedder.encode(prompts, device=device)
            prompt_embed = glim_model.p_embedder(prompt_ids, glim_model.eval_pembed)

            eeg_hiddens, _ = glim_model.eeg_encoder(glim_eeg, glim_mask, prompt_embed)
            _, eeg_emb_vector = aligner.embed_eeg(eeg_hiddens)

            # eeg_emb_vector 可能因 batch=1 导致 squeeze 出错，确保形状 (B, E)
            if eeg_emb_vector.dim() == 1:
                eeg_emb_vector = eeg_emb_vector.unsqueeze(0)

            v = F.normalize(eeg_emb_vector, dim=-1)
        vecs.append(v.cpu())
        if logger and (bi + 1) % 20 == 0:
            logger.info("  eeg enc %d/%d", bi + 1, total)
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
    # 噪声条件自动添加输出目录后缀
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, log_name="retrieval_eval.log")
    logger.info("GLIM Retrieval Eval | args=%s", vars(args))

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
    logger.info("Loading GLIMWrapper...")
    wrapper = GLIMWrapper(
        model_checkpoint=args.model_checkpoint,
        text_model_id=args.text_model_id,
    )
    device = wrapper.device
    glim_model = wrapper.model
    glim_model.eval()
    logger.info("Model on device: %s", device)

    # 3. 收集样本
    logger.info("Collecting samples...")
    dl = DataLoader(ds, batch_size=args.eeg_batch_size, shuffle=False,
                    num_workers=0, collate_fn=custom_collate_fn)

    ref_texts, meta_list = [], []
    eeg_bufs, mask_bufs, meta_bufs = [], [], []

    for batch in dl:
        for i in range(len(batch["idx"])):
            ref_texts.append(batch["reference_text"][i])
            meta_list.append(dict(batch["meta"][i]))
        meta_bufs.append([dict(batch["meta"][i]) for i in range(len(batch["idx"]))])
        eeg_bufs.append(batch.get("eeg_word_raw", batch.get("eeg_raw", batch["eeg"])))
        mask_bufs.append(batch.get("mask_word", batch.get("mask")))

    unique_texts = list(dict.fromkeys(ref_texts))
    t2i = {t: i for i, t in enumerate(unique_texts)}
    gt_idx = [t2i[t] for t in ref_texts]
    N, M = len(ref_texts), len(unique_texts)
    logger.info("Queries: %d | Candidates: %d | Random R@1 ≈ %.4f%%",
                N, M, 100.0 / M)

    # 4. 编码文本（通过 aligner.embed_text，与训练一致）
    logger.info("Encoding %d texts (bs=%d)...", M, args.text_batch_size)
    text_vecs = encode_texts(glim_model, unique_texts, device,
                             bs=args.text_batch_size, logger=logger)
    logger.info("text_vecs: %s", tuple(text_vecs.shape))

    # 5. 编码 EEG（通过 eeg_encoder + aligner.embed_eeg）
    logger.info("Encoding %d EEGs (bs=%d)...", N, args.eeg_batch_size)
    eeg_vecs = encode_eegs(wrapper, eeg_bufs, mask_bufs, meta_bufs, device, logger=logger)
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

    # 7. 保存指标
    result = {"overall": overall, "grouped": grp}
    out_path = os.path.join(output_dir, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 7b. 落盘嵌入向量
    try:
        from evaluation.embedding_io import save_embeddings
        emb_path = save_embeddings(
            output_dir=output_dir,
            v_eeg=eeg_vecs, v_text=text_vecs,
            gt_idx=gt_idx, meta_list=meta_list,
            noise_type=args.noise_type, model_name="glim",
            ranks=ranks.to(int).tolist(),
            unique_texts=unique_texts,
        )
        logger.info("Embeddings → %s", emb_path)
    except Exception as _emb_err:  # pragma: no cover
        logger.warning("save_embeddings failed: %s", _emb_err)

    sep = "=" * 60
    print(f"\n{sep}")
    print("GLIM (CLIP-aligned) RETRIEVAL RESULTS")
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
