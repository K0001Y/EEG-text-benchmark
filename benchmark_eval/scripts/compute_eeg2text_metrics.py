"""
EEG2Text 离线指标计算脚本

从已生成的 predictions.jsonl 计算 BLEU / ROUGE / WER 指标。
不依赖网络（无需下载 BERTScore 模型），使用本地实现。

用法：
    python3 benchmark_eval/scripts/compute_eeg2text_metrics.py \
        --pred-path benchmark_eval/test_outputs/eval_eeg2text/predictions.jsonl \
        --output   benchmark_eval/test_outputs/eval_eeg2text/metrics.json
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge


# ---------------------------------------------------------------------------
# 本地 WER 实现（动态规划编辑距离，无网络依赖）
# ---------------------------------------------------------------------------

def _edit_distance(r: list, h: list) -> int:
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[m]


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """计算语料级 WER（总编辑距离 / 总参考词数）。"""
    total_errors = 0
    total_words = 0
    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()
        if not r_words:
            continue
        total_errors += _edit_distance(r_words, h_words)
        total_words += len(r_words)
    return total_errors / total_words if total_words > 0 else 0.0


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_metrics(refs: list[str], preds: list[str]) -> dict:
    """计算 BLEU-1/2/3/4, ROUGE-1/2/L, WER（均以百分比表示）。"""
    if not refs:
        return {k: 0.0 for k in ["bleu1", "bleu2", "bleu3", "bleu4",
                                  "rouge1", "rouge2", "rougeL", "wer"]}

    ref_tokens = [[r.split()] for r in refs]
    hyp_tokens = [p.split() for p in preds]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0))
        b2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0))
        b3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0))
        b4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    r1 = r2 = rl = 0.0
    try:
        rouge = Rouge()
        rr = rouge.get_scores(preds, refs, avg=True, ignore_empty=True)
        r1 = float(rr.get("rouge-1", {}).get("f", 0.0))
        r2 = float(rr.get("rouge-2", {}).get("f", 0.0))
        rl = float(rr.get("rouge-l", {}).get("f", 0.0))
    except Exception:
        pass

    wer = corpus_wer(refs, preds)

    return {
        "bleu1":  round(b1 * 100, 2),
        "bleu2":  round(b2 * 100, 2),
        "bleu3":  round(b3 * 100, 2),
        "bleu4":  round(b4 * 100, 2),
        "rouge1": round(r1 * 100, 2),
        "rouge2": round(r2 * 100, 2),
        "rougeL": round(rl * 100, 2),
        "wer":    round(wer * 100, 2),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_predictions(pred_path: str) -> list[dict]:
    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_metrics(label: str, n: int, m: dict) -> None:
    print(f"\n  [{label}]  (n={n})")
    print(f"    BLEU-1:  {m['bleu1']:.2f}%")
    print(f"    BLEU-2:  {m['bleu2']:.2f}%")
    print(f"    BLEU-3:  {m['bleu3']:.2f}%")
    print(f"    BLEU-4:  {m['bleu4']:.2f}%")
    print(f"    ROUGE-1: {m['rouge1']:.2f}%")
    print(f"    ROUGE-2: {m['rouge2']:.2f}%")
    print(f"    ROUGE-L: {m['rougeL']:.2f}%")
    print(f"    WER:     {m['wer']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 predictions.jsonl 离线计算评估指标")
    parser.add_argument("--pred-path", type=str,
                        default="benchmark_eval/test_outputs/eval_eeg2text/predictions.jsonl",
                        help="predictions.jsonl 路径")
    parser.add_argument("--output", type=str,
                        default="benchmark_eval/test_outputs/eval_eeg2text/metrics.json",
                        help="metrics.json 输出路径")
    args = parser.parse_args()

    records = load_predictions(args.pred_path)
    print(f"Loaded {len(records)} predictions from {args.pred_path}")

    refs  = [r["reference"]  for r in records]
    preds = [r["prediction"] for r in records]

    # ---- 整体指标 ----
    print("\n=== Overall Metrics ===")
    overall = compute_metrics(refs, preds)
    print_metrics("Overall", len(records), overall)

    # ---- 按 task 分组 ----
    by_task: dict = defaultdict(lambda: {"refs": [], "preds": []})
    for r in records:
        t = r.get("meta", {}).get("task", "unknown")
        by_task[t]["refs"].append(r["reference"])
        by_task[t]["preds"].append(r["prediction"])

    print("\n=== Per-Task Metrics ===")
    task_metrics = {}
    task_counts = {}
    for task, data in sorted(by_task.items()):
        m = compute_metrics(data["refs"], data["preds"])
        task_metrics[task] = m
        task_counts[task] = len(data["refs"])
        print_metrics(task, task_counts[task], m)

    # ---- 保存 ----
    full_results = {
        "overall": overall,
        "grouped": {"by_task": task_metrics},
        "num_samples": len(records),
        "task_sample_counts": task_counts,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
