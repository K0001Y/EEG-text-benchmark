from collections import Counter
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from evaluate import load


@lru_cache(maxsize=1)
def _get_wer_metric():
    """缓存 WER 指标加载，避免重复加载。"""
    return load("wer")


@lru_cache(maxsize=1)
def _get_bertscore_metric():
    """缓存 BERTScore 指标加载。"""
    return load("bertscore")


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    """计算 n-gram 计数。"""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_corpus_metrics(references: Iterable[str], predictions: Iterable[str]) -> Dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE (1/2/L), WER and BERTScore over the whole corpus.

    参考 EEG-To-Text eval_decoding.py 中的评估方式：
    - BLEU: 使用 nltk 的 corpus_bleu，分别计算 1/2/3/4-gram
    - ROUGE: 使用 python-rouge 的 Rouge.get_scores（取 F1 值）
    - WER: 使用 evaluate.load("wer")
    - BERTScore: 使用 evaluate.load("bertscore")
    """
    refs_list = list(references)
    preds_list = list(predictions)

    if not refs_list:
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "wer": 0.0,
            "bertscore": 0.0,
        }

    # BLEU（corpus_bleu）
    # 参考格式：[[ref_tokens], ...], [hyp_tokens, ...]
    ref_tokens = [[r.split()] for r in refs_list]
    hyp_tokens = [p.split() for p in preds_list]

    weights_list = [
        (1.0,),
        (0.5, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    bleu_scores: Dict[str, float] = {}
    for n, w in enumerate(weights_list, start=1):
        try:
            score = corpus_bleu(ref_tokens, hyp_tokens, weights=w)
        except Exception:
            score = 0.0
        bleu_scores[f"bleu{n}"] = float(score)

    # ROUGE（使用 Rouge 库，取 F1 分数）
    rouge1_f = 0.0
    rouge2_f = 0.0
    rougel_f = 0.0
    try:
        rouge = Rouge()
        rouge_result = rouge.get_scores(preds_list, refs_list, avg=True, ignore_empty=True)
        if isinstance(rouge_result, dict):
            rouge1_f = float(rouge_result.get("rouge-1", {}).get("f", 0.0))
            rouge2_f = float(rouge_result.get("rouge-2", {}).get("f", 0.0))
            rougel_f = float(rouge_result.get("rouge-l", {}).get("f", 0.0))
    except Exception:
        pass

    # WER（使用缓存的 evaluate.load("wer")）
    wer_score = 0.0
    try:
        wer_metric = _get_wer_metric()
        wer_score = float(wer_metric.compute(predictions=preds_list, references=refs_list))
    except Exception:
        pass

    # BERTScore
    bertscore_f = 0.0
    try:
        bertscore_metric = _get_bertscore_metric()
        results = bertscore_metric.compute(
            predictions=preds_list,
            references=refs_list,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
        bertscore_f = float(np.mean(results["f1"]))
    except Exception:
        pass

    metrics = dict(bleu_scores)
    metrics["rouge1"] = rouge1_f
    metrics["rouge2"] = rouge2_f
    metrics["rougeL"] = rougel_f
    metrics["wer"] = wer_score
    metrics["bertscore"] = bertscore_f
    return metrics


def compute_grouped_metrics(
    predictions: List[Dict[str, any]],
    group_by: str = "task"
) -> Dict[str, Dict[str, float]]:
    """按指定字段分组计算指标。

    Args:
        predictions: 预测结果列表，每个元素是包含 reference、prediction、meta 的字典
        group_by: 分组字段，支持 "task"、"subject"、"dataset"

    Returns:
        分组后的指标字典，格式为 {group_value: {metric_name: metric_value}}
    """
    from collections import defaultdict

    # 按 group_by 字段分组
    grouped = defaultdict(lambda: {"references": [], "predictions": []})

    for pred in predictions:
        meta = pred.get("meta", {})
        group_value = meta.get(group_by, "unknown")

        grouped[group_value]["references"].append(pred["reference"])
        grouped[group_value]["predictions"].append(pred["prediction"])

    # 为每个组计算指标
    results = {}
    for group_value, data in grouped.items():
        metrics = compute_corpus_metrics(data["references"], data["predictions"])
        results[group_value] = metrics

    return results


def compute_all_grouped_metrics(predictions: List[Dict[str, any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """计算所有分组维度的指标（task、subject、dataset）。

    Args:
        predictions: 预测结果列表

    Returns:
        包含所有分组维度的指标字典
    """
    return {
        "by_task": compute_grouped_metrics(predictions, group_by="task"),
        "by_subject": compute_grouped_metrics(predictions, group_by="subject"),
        "by_dataset": compute_grouped_metrics(predictions, group_by="dataset"),
    }
