"""指标计算模块。

修复内容（v2）：
- C-2：BERTScore 添加离线降级，失败时返回 NaN 而非 0.0
- H-4：缩小异常捕获范围，添加 logger.warning 记录具体错误
- H-7：分组指标添加 sample_count 字段
- M-5：所有计算失败统一返回 float('nan')
- M-6：类型标注修复（any → Any）
- OFFLINE：WER 使用本地实现，BERTScore 在离线环境跳过
- MIRROR：BERTScore 下载优先使用 hf-mirror.com 镜像
- CACHE：BERTScore 模型在进程内只加载一次（模块级缓存）
"""

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


logger = logging.getLogger("metrics")

# ── BERTScore 模块级缓存（进程内只初始化一次）──────────────────────────────
_bertscore_metric = None   # bert_score 模块对象
_bertscore_available: bool | None = None  # None=未检测, True/False=已检测


def _get_bertscore_metric():
    """获取 BERTScore 计算函数（直接使用 bert_score 包，不经过 evaluate.load）。

    - 直接调用 bert_score.score()，无需从 Hub 下载 metric 脚本
    - 优先使用本地 HF 缓存（已下载的 DeBERTa 模型），不再依赖网络
    - 若 bert_score 未安装或模型不存在则返回 None
    """
    global _bertscore_metric, _bertscore_available

    if _bertscore_available is False:
        return None
    if _bertscore_metric is not None:
        return _bertscore_metric

    try:
        import bert_score as _bs  # noqa: F401
        _bertscore_metric = _bs
        _bertscore_available = True
        logger.info("BERTScore: using bert_score package directly (no evaluate.load)")
        return _bertscore_metric
    except ImportError:
        logger.warning("BERTScore skipped: bert_score package not installed.")
        _bertscore_available = False
        return None


def _compute_wer_local(references: List[str], predictions: List[str]) -> float:
    """本地 WER 实现，不依赖 evaluate 库（离线兼容）。

    WER = (S + D + I) / N，其中 S=替换, D=删除, I=插入, N=参考词总数
    使用标准动态规划 Levenshtein 距离算法。
    """
    total_errors = 0
    total_ref_len = 0

    for ref, hyp in zip(references, predictions):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        r, h = len(ref_tokens), len(hyp_tokens)

        if r == 0:
            total_errors += h
            continue

        # DP 矩阵
        d = [[0] * (h + 1) for _ in range(r + 1)]
        for i in range(r + 1):
            d[i][0] = i
        for j in range(h + 1):
            d[0][j] = j
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

        total_errors += d[r][h]
        total_ref_len += r

    return total_errors / total_ref_len if total_ref_len > 0 else float("nan")


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
        wer_score = _compute_wer_local(refs_list, preds_list)
    except Exception as exc:
        logger.warning("WER computation failed: %s", exc)

    # --- BERTScore（直接使用 bert_score 包，从本地 HF 缓存加载模型）---
    bertscore_f = float("nan")
    bertscore_pkg = _get_bertscore_metric()
    if bertscore_pkg is not None:
        try:
            # 直接传本地路径，避免任何网络请求（绕过 HuggingFace name resolution）
            import os as _os, glob as _glob
            _snapshot_dir = _os.path.expanduser(
                "~/.cache/huggingface/hub/models--microsoft--deberta-xlarge-mnli/snapshots"
            )
            _snapshots = sorted(_glob.glob(_os.path.join(_snapshot_dir, "*")))
            _model_path = _snapshots[-1] if _snapshots else "microsoft/deberta-xlarge-mnli"
            logger.info("BERTScore model path: %s", _model_path)

            _, _, F1 = bertscore_pkg.score(
                preds_list,
                refs_list,
                lang="en",
                model_type=_model_path,
                verbose=False,
            )
            bertscore_f = float(F1.mean().item())
        except (OSError, ValueError) as exc:
            logger.warning(
                "BERTScore unavailable (model not found): %s. Returning NaN.", exc
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
