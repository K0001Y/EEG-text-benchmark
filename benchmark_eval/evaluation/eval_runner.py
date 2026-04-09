"""统一评估执行器（v2）。

修复内容：
- H-2：新增 set_seed()，覆盖 random / numpy / torch / torch.cuda / cudnn
- H-3：DataLoader 使用固定 generator，保证 worker 随机性一致
- A-4：统一输出 JSON schema（"overall" + "grouped" 键）
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_processing.dataset import UnifiedDataset, custom_collate_fn
from utils.logging_utils import setup_logging, get_logger
from evaluation.metrics import compute_corpus_metrics, compute_all_grouped_metrics
from evaluation.model_wrappers import build_model_wrapper, BenchmarkModelWrapper
from constants import DEFAULT_SEED


STATE_FILENAME = "state.json"
PRED_FILENAME = "predictions.jsonl"
METRIC_FILENAME = "metrics.json"


# ---------------------------------------------------------------------------
# H-2：统一随机种子设置
# ---------------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """统一设置所有随机性来源的种子，确保实验可复现。

    覆盖：random / numpy / torch / torch.cuda / cudnn.deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified EEG-to-Text benchmark evaluation runner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to unified dataset pickle file")
    parser.add_argument("--phase", type=str, default="test", help="Dataset split: train/val/test")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store logs and results")
    parser.add_argument("--model-name", type=str, default="dummy",
                        help="Model wrapper name: dummy/eeg_to_text/eeg2text/cet_mae/glim")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (optional)")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (0 recommended for reproducibility)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing state.json and predictions.jsonl if present")
    # 噪声实验参数
    parser.add_argument("--noise-experiment", action="store_true", help="Run noise control experiment")
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_SEED, help="Random seed for noise generation")
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        choices=["gaussian", "uniform", "zero", "shuffle"],
                        help="Type of noise")
    parser.add_argument("--noise-mean", type=float, default=0.0, help="Mean for Gaussian noise")
    parser.add_argument("--noise-std", type=float, default=1.0,
                        help="Std for Gaussian noise or range for uniform noise")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 状态持久化（断点恢复）
# ---------------------------------------------------------------------------

def load_state(state_path: str) -> Dict[str, Any]:
    if not os.path.isfile(state_path):
        return {"next_index": 0, "total": None}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: str, state: Dict[str, Any]) -> None:
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def append_predictions(pred_path: str, records: List[Dict[str, Any]]) -> None:
    with open(pred_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def load_predictions(pred_path: str) -> Dict[int, Dict[str, Any]]:
    """加载已有预测，按 idx 建索引。"""
    results: Dict[int, Dict[str, Any]] = {}
    if not os.path.isfile(pred_path):
        return results
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = int(rec["idx"])
            results[idx] = rec
    return results


# ---------------------------------------------------------------------------
# 核心评估函数
# ---------------------------------------------------------------------------

def run_evaluation(
    model: BenchmarkModelWrapper,
    dataset: UnifiedDataset,
    args: argparse.Namespace,
    output_subdir: str = "",
) -> Dict[str, Any]:
    """运行单次评估（正常或噪声模式）。

    输出 schema（A-4 统一）：
      {
        "overall":      {metric_name: value, ...},
        "grouped":      {"by_task": {...}, "by_subject": {...}, "by_dataset": {...}},
        "failed_count": N,
        "num_samples":  M,
      }
    """
    if output_subdir:
        output_dir = os.path.join(args.output_dir, output_subdir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger("eval_runner")
    logger.info("Running evaluation in %s", output_dir)

    total = len(dataset)

    state_path = os.path.join(output_dir, STATE_FILENAME)
    state = load_state(state_path)
    if not args.resume:
        state = {"next_index": 0, "total": total}
    else:
        if state.get("total") is None:
            state["total"] = total
        if state["total"] != total:
            logger.warning(
                "Total samples changed (state=%d, current=%d), resetting progress.",
                state["total"], total
            )
            state = {"next_index": 0, "total": total}

    next_index = int(state.get("next_index", 0))
    if next_index >= total:
        logger.info(
            "State says all samples already processed (next_index=%d, total=%d).",
            next_index, total
        )

    pred_path = os.path.join(output_dir, PRED_FILENAME)
    existing_predictions = load_predictions(pred_path) if args.resume else {}

    if not args.resume:
        if os.path.isfile(pred_path):
            logger.info("Not resuming: removing existing prediction file: %s", pred_path)
            os.remove(pred_path)
        existing_predictions = {}

    # H-3：DataLoader 使用固定 generator
    if next_index < total:
        indices = list(range(next_index, total))
        subset = Subset(dataset, indices)
        generator = torch.Generator().manual_seed(args.seed)
        dataloader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            generator=generator,
        )
    else:
        dataloader = []  # type: ignore

    processed = next_index
    failed_indices: List[int] = []
    failed_errors: List[str] = []

    for batch in dataloader:
        try:
            idx_list: List[int] = batch["idx"].tolist()
            eeg = batch["eeg"]
            mask = batch["mask"]

            meta_batch: List[Dict[str, Any]] = []
            for i in range(len(idx_list)):
                meta = dict(batch["meta"][i])
                meta["input_text"] = batch["input_text"][i]
                meta_batch.append(meta)

            preds = model.generate_text(eeg, mask, meta_batch, batch=batch)

            records: List[Dict[str, Any]] = []
            for local_i, idx in enumerate(idx_list):
                ref = batch["reference_text"][local_i]
                pred = preds[local_i]
                records.append({
                    "idx": int(idx),
                    "reference": ref,
                    "prediction": pred,
                    "meta": meta_batch[local_i],
                })

            append_predictions(pred_path, records)
            processed = idx_list[-1] + 1
            state["next_index"] = processed
            state["total"] = total
            save_state(state_path, state)
            logger.info("Processed up to index %d / %d", processed, total)

        except Exception as e:
            idx_list_str = batch["idx"].tolist() if "idx" in batch else "unknown"
            logger.exception(
                "Error while processing batch (indices: %s), skipping this batch.", idx_list_str
            )
            failed_indices.extend(idx_list_str if isinstance(idx_list_str, list) else [])
            failed_errors.append(str(e))

    if failed_indices:
        logger.warning("=" * 60)
        logger.warning("Evaluation completed with %d failed samples", len(failed_indices))
        logger.warning("Failed indices (first 20): %s", failed_indices[:20])
        failed_path = os.path.join(output_dir, "failed_samples.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump({
                "failed_count": len(failed_indices),
                "failed_indices": failed_indices,
                "errors": failed_errors[:20],
            }, f, indent=2, ensure_ascii=False)
        logger.warning("Failed samples record saved to %s", failed_path)
        logger.warning("=" * 60)

    # 重新加载所有预测
    all_preds = load_predictions(pred_path)
    if not all_preds:
        logger.warning("No predictions found, skip metric computation.")
        return {
            "overall": {},
            "grouped": {},
            "failed_count": len(failed_indices),
            "num_samples": 0,
        }

    ordered_indices = sorted(all_preds.keys())
    references: List[str] = []
    predictions: List[str] = []
    for idx in ordered_indices:
        rec = all_preds[idx]
        references.append(rec["reference"])
        predictions.append(rec["prediction"])

    metrics = compute_corpus_metrics(references, predictions)
    all_preds_list = list(all_preds.values())
    grouped_metrics = compute_all_grouped_metrics(all_preds_list)

    # A-4：统一 schema
    full_results: Dict[str, Any] = {
        "overall": metrics,
        "grouped": grouped_metrics,
        "failed_count": len(failed_indices),
        "num_samples": len(all_preds),
    }

    metrics_path = os.path.join(output_dir, METRIC_FILENAME)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    logger.info("Finished evaluation. Metrics saved to %s", metrics_path)
    logger.info("Overall Metrics: %s", metrics)

    return full_results


# ---------------------------------------------------------------------------
# 噪声实验对比
# ---------------------------------------------------------------------------

def compare_normal_vs_noise(
    normal_results: Dict[str, Any],
    noise_results: Dict[str, Any],
) -> Dict[str, Any]:
    """对比正常和噪声实验结果，计算相对下降。"""
    comparison: Dict[str, Any] = {}

    normal_metrics = normal_results.get("overall", {})
    noise_metrics = noise_results.get("overall", {})

    metrics_to_compare = ["bleu1", "bleu2", "bleu4", "rougeL", "wer"]

    for metric in metrics_to_compare:
        if metric in normal_metrics and metric in noise_metrics:
            normal_val = normal_metrics[metric]
            noise_val = noise_metrics[metric]

            if isinstance(normal_val, float) and np.isnan(normal_val):
                continue
            if isinstance(noise_val, float) and np.isnan(noise_val):
                continue

            if normal_val != 0:
                relative_drop = (normal_val - noise_val) / normal_val * 100
            else:
                relative_drop = 0.0

            comparison[metric] = {
                "normal": normal_val,
                "noise": noise_val,
                "absolute_diff": normal_val - noise_val,
                "relative_drop_%": relative_drop,
            }

    if comparison:
        drops = [
            v["relative_drop_%"]
            for v in comparison.values()
            if isinstance(v, dict) and "relative_drop_%" in v
        ]
        avg_drop = float(np.mean(drops)) if drops else 0.0
        comparison["summary"] = {
            "average_relative_drop_%": avg_drop,
            "is_eeg_dependent": avg_drop > 10.0,
            "conclusion": (
                "Model appears to use EEG signal" if avg_drop > 10.0
                else "WARNING: Model may not be using EEG signal effectively"
            ),
        }

    return comparison


def print_noise_experiment_summary(comparison: Dict[str, Any], logger) -> None:
    """打印噪声实验总结。"""
    logger.info("\n" + "=" * 60)
    logger.info("NOISE EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary = comparison.get("summary", {})
    logger.info("Average relative drop: %.2f%%", summary.get("average_relative_drop_%", 0))
    logger.info("Conclusion: %s", summary.get("conclusion", "N/A"))

    logger.info("\nDetailed metrics:")
    for metric, values in comparison.items():
        if metric != "summary" and isinstance(values, dict):
            logger.info(
                "  %s: normal=%.4f, noise=%.4f, drop=%.2f%%",
                metric,
                values.get("normal", 0),
                values.get("noise", 0),
                values.get("relative_drop_%", 0),
            )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir)
    logger.info("Starting evaluation with args: %s", vars(args))

    # H-2：在入口处统一设置全局随机种子
    set_seed(args.seed)
    logger.info("Global random seed set to %d", args.seed)

    model_kwargs: Dict[str, Any] = {}
    if args.model_checkpoint:
        model_kwargs["model_checkpoint"] = args.model_checkpoint
    if args.pretrain_checkpoint:
        model_kwargs["pretrain_checkpoint"] = args.pretrain_checkpoint
    model = build_model_wrapper(args.model_name, **model_kwargs)

    logger.info("=" * 60)
    logger.info("Running NORMAL evaluation...")
    logger.info("=" * 60)

    normal_dataset = UnifiedDataset(args.data_path, phase=args.phase)
    normal_results = run_evaluation(model, normal_dataset, args, output_subdir="")

    if args.noise_experiment:
        logger.info("\n" + "=" * 60)
        logger.info("Running NOISE evaluation (noise_type=%s, seed=%d)...", args.noise_type, args.noise_seed)
        logger.info("=" * 60)

        noise_dataset = UnifiedDataset(
            args.data_path,
            phase=args.phase,
            noise_mode=True,
            noise_type=args.noise_type,
            noise_seed=args.noise_seed,
            noise_mean=args.noise_mean,
            noise_std=args.noise_std,
        )

        original_resume = args.resume
        args.resume = False
        noise_results = run_evaluation(model, noise_dataset, args, output_subdir="noise")
        args.resume = original_resume

        logger.info("\n" + "=" * 60)
        logger.info("Comparing normal vs noise results...")
        logger.info("=" * 60)

        comparison = compare_normal_vs_noise(normal_results, noise_results)
        print_noise_experiment_summary(comparison, logger)

        comparison_path = os.path.join(args.output_dir, "noise_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump({
                "normal": normal_results.get("overall", {}),
                "noise": noise_results.get("overall", {}),
                "comparison": comparison,
            }, f, ensure_ascii=False, indent=2)
        logger.info("Comparison results saved to %s", comparison_path)

    logger.info("\n" + "=" * 60)
    logger.info("All evaluations completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
