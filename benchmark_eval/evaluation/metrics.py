"""指标计算模块。

修复内容（v2）：
- C-2：BERTScore 添加离线降级，失败时返回 NaN 而非 0.0
- H-4：缩小异常捕获范围，添加 logger.warning 记录具体错误
- H-7：分组指标添加 sample_count 字段
- M-5：所有计算失败统一返回 float('nan')
- M-6：类型标注修复（any → Any）
"""

import logging
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from evaluate import load


logger = logging.getLogger("metrics")


@lru_cache(maxsize=1)
def _get_wer_metric():
    """缓存 WER 指标加载，避免重复加载。"""
    return load("wer")


@lru_cache(maxsize=1)
def _get_bertscore_metric():
    """缓存 BERTScore 指标加载。"""
    return load("bertscore")


def _nan_metrics() -> Dict[str, float]:
    """返回全 NaN 的指标字典（表示计算未执行或数据为空）。"""
    return {
        "bleu1": float("nan"),
        "bleu2": float("nan"),
        "bleu3": float("nan"),
        "bleu4": float("nan"),
        "rouge1": float("nan"),
        "rouge2": float("nan"),
        "rougeL": float("nan"),
        "wer": float("nan"),
        "bertscore": float("nan"),
    }


def compute_corpus_metrics(
    references: Iterable[str],
    predictions: Iterable[str],
) -> Dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-1/2/L, WER and BERTScore over the whole corpus.

    计算失败时返回 float('nan') 而非 0.0，区分"计算值为零"与"计算失败"。

    - BLEU:       使用 nltk corpus_bleu
    - ROUGE:      使用 python-rouge（F1）
    - WER:        使用 evaluate.load("wer")
    - BERTScore:  使用 evaluate.load("bertscore")，离线降级返回 NaN
    """
    refs_list = list(references)
    preds_list = list(predictions)

    if not refs_list:
        return _nan_metrics()

    # --- BLEU ---
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
            score = float(corpus_bleu(ref_tokens, hyp_tokens, weights=w))
        except ZeroDivisionError:
            logger.warning("BLEU-%d: ZeroDivisionError (no valid n-grams), returning NaN", n)
            score = float("nan")
        except Exception as exc:
            logger.warning("BLEU-%d computation failed: %s", n, exc)
            score = float("nan")
        bleu_scores[f"bleu{n}"] = score

    # --- ROUGE ---
    rouge1_f = rouge2_f = rougel_f = float("nan")
    try:
        rouge = Rouge()
        rouge_result = rouge.get_scores(preds_list, refs_list, avg=True, ignore_empty=True)
        if isinstance(rouge_result, dict):
            rouge1_f = float(rouge_result.get("rouge-1", {}).get("f", float("nan")))
            rouge2_f = float(rouge_result.get("rouge-2", {}).get("f", float("nan")))
            rougel_f = float(rouge_result.get("rouge-l", {}).get("f", float("nan")))
    except Exception as exc:
        logger.warning("ROUGE computation failed: %s", exc)

    # --- WER ---
    wer_score = float("nan")
    try:
        wer_metric = _get_wer_metric()
        wer_score = float(wer_metric.compute(predictions=preds_list, references=refs_list))
    except Exception as exc:
        logger.warning("WER computation failed: %s", exc)

    # --- BERTScore（离线降级）---
    bertscore_f = float("nan")
    try:
        bertscore_metric = _get_bertscore_metric()
        results = bertscore_metric.compute(
            predictions=preds_list,
            references=refs_list,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        bertscore_f = float(np.mean(results["f1"]))
    except (OSError, ValueError) as exc:
        # OSError：网络不可达或模型文件不存在（离线环境）
        # ValueError：模型下载失败
        logger.warning(
            "BERTScore unavailable (likely offline environment): %s. "
            "Returning NaN. Set compute_bertscore: false in config to suppress.",
            exc,
        )
    except Exception as exc:
        logger.warning("BERTScore computation failed: %s", exc)

    metrics = dict(bleu_scores)
    metrics["rouge1"] = rouge1_f
    metrics["rouge2"] = rouge2_f
    metrics["rougeL"] = rougel_f
    metrics["wer"] = wer_score
    metrics["bertscore"] = bertscore_f
    return metrics


def compute_grouped_metrics(
    predictions: List[Dict[str, Any]],
    group_by: str = "task",
) -> Dict[str, Dict[str, Any]]:
    """按指定字段分组计算指标，并附带各组样本数（H-7）。

    Args:
        predictions: 预测结果列表，每个元素包含 reference、prediction、meta
        group_by:    分组字段（"task"、"subject"、"dataset"）

    Returns:
        {group_value: {"metrics": {...}, "sample_count": N}}
    """
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"references": [], "predictions": []}
    )

    for pred in predictions:
        meta = pred.get("meta", {})
        group_value = meta.get(group_by, "unknown")
        grouped[group_value]["references"].append(pred["reference"])
        grouped[group_value]["predictions"].append(pred["prediction"])

    results: Dict[str, Dict[str, Any]] = {}
    for group_value, data in grouped.items():
        n = len(data["references"])
        group_metrics = compute_corpus_metrics(data["references"], data["predictions"])
        results[group_value] = {
            "sample_count": n,
            "metrics": group_metrics,
        }

    return results


def compute_all_grouped_metrics(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """计算所有分组维度的指标（task、subject、dataset）。

    Returns:
        {
            "by_task":    {task_name:    {"sample_count": N, "metrics": {...}}},
            "by_subject": {subject_name: {"sample_count": N, "metrics": {...}}},
            "by_dataset": {dataset_name: {"sample_count": N, "metrics": {...}}},
        }
    """
    return {
        "by_task": compute_grouped_metrics(predictions, group_by="task"),
        "by_subject": compute_grouped_metrics(predictions, group_by="subject"),
        "by_dataset": compute_grouped_metrics(predictions, group_by="dataset"),
    }
