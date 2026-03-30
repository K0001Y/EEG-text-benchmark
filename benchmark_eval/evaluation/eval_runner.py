import argparse
import json
import os
import sys
from typing import Any, Dict, List

from torch.utils.data import DataLoader, Subset

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_processing.dataset import UnifiedDataset, custom_collate_fn
from utils.logging_utils import setup_logging, get_logger
from evaluation.metrics import compute_corpus_metrics, compute_all_grouped_metrics
from evaluation.model_wrappers import build_model_wrapper


STATE_FILENAME = "state.json"
PRED_FILENAME = "predictions.jsonl"
METRIC_FILENAME = "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified EEG-to-Text benchmark evaluation runner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to unified dataset pickle file")
    parser.add_argument("--phase", type=str, default="test", help="Dataset split: train/val/test")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store logs and results")
    parser.add_argument("--model-name", type=str, default="dummy", help="Model wrapper name, e.g. dummy/eeg_to_text/eeg2text/cet_mae/glim")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--pretrain-checkpoint", type=str, default=None, help="Path to pretrained encoder checkpoint (optional)")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state.json and predictions.jsonl if present")
    return parser.parse_args()


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
    # 以追加方式写入，每条一行 JSON，方便中断后继续
    with open(pred_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def load_predictions(pred_path: str) -> Dict[int, Dict[str, Any]]:
    """加载已有预测，按 idx 建索引，用于在恢复或最终计算指标时使用。"""
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir)
    logger.info("Starting evaluation with args: %s", vars(args))

    # 加载数据集
    dataset = UnifiedDataset(args.data_path, phase=args.phase)
    total = len(dataset)
    logger.info("Loaded dataset from %s, phase=%s, total samples=%d", args.data_path, args.phase, total)

    # 加载或初始化 state
    state_path = os.path.join(args.output_dir, STATE_FILENAME)
    state = load_state(state_path)
    if not args.resume:
        state = {"next_index": 0, "total": total}
    else:
        if state.get("total") is None:
            state["total"] = total
        if state["total"] != total:
            logger.warning("Total samples changed (state=%d, current=%d), resetting progress.", state["total"], total)
            state = {"next_index": 0, "total": total}

    next_index = int(state.get("next_index", 0))
    if next_index >= total:
        logger.info("State says all samples already processed (next_index=%d, total=%d).", next_index, total)

    # 预测结果文件
    pred_path = os.path.join(args.output_dir, PRED_FILENAME)
    existing_predictions = load_predictions(pred_path) if args.resume else {}

    # 如果从头开始，则覆盖旧结果
    if not args.resume:
        if os.path.isfile(pred_path):
            logger.info("Not resuming: removing existing prediction file: %s", pred_path)
            os.remove(pred_path)
        existing_predictions = {}

    # 构建 DataLoader，跳过已经处理过的样本
    if next_index < total:
        indices = list(range(next_index, total))
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    else:
        dataloader = []  # type: ignore

    # 构建模型 wrapper
    model_kwargs = {}
    if args.model_checkpoint:
        model_kwargs["model_checkpoint"] = args.model_checkpoint
    if args.pretrain_checkpoint:
        model_kwargs["pretrain_checkpoint"] = args.pretrain_checkpoint
    model = build_model_wrapper(args.model_name, **model_kwargs)

    # 主循环：批量生成预测，逐步写出预测和 state，支持中断恢复
    processed = next_index
    failed_indices: List[int] = []
    failed_errors: List[str] = []

    for batch in dataloader:
        try:
            # Subset 返回的 idx 是原始数据集的索引
            idx_list: List[int] = batch["idx"].tolist()
            # 默认使用 eeg 字段，wrapper 可根据需要选择其他字段
            eeg = batch["eeg"]
            mask = batch["mask"]

            # 为了便于 Dummy 模型使用 input_text，这里把 input_text 放进 meta
            # 同时传递整个 batch，让 wrapper 可以选择需要的数据格式
            meta_batch: List[Dict[str, Any]] = []
            for i in range(len(idx_list)):
                meta = dict(batch["meta"][i])
                meta["input_text"] = batch["input_text"][i]
                meta_batch.append(meta)

            # 调用模型生成，传递整个 batch 以便 wrapper 选择数据格式
            preds = model.generate_text(eeg, mask, meta_batch, batch=batch)

            records: List[Dict[str, Any]] = []
            for local_i, idx in enumerate(idx_list):
                ref = batch["reference_text"][local_i]
                pred = preds[local_i]
                records.append(
                    {
                        "idx": int(idx),
                        "reference": ref,
                        "prediction": pred,
                        "meta": meta_batch[local_i],
                    }
                )

            append_predictions(pred_path, records)
            processed = idx_list[-1] + 1

            # 更新 state，确保中断后可以继续
            state["next_index"] = processed
            state["total"] = total
            save_state(state_path, state)
            logger.info("Processed up to index %d / %d", processed, total)
        except Exception as e:
            # 记录详细异常信息，但不中断整个评估
            idx_list_str = batch["idx"].tolist() if "idx" in batch else "unknown"
            logger.exception("Error while processing batch (indices: %s), skipping this batch.", idx_list_str)
            failed_indices.extend(idx_list_str if isinstance(idx_list_str, list) else [])
            failed_errors.append(str(e))

    # 输出失败统计
    if failed_indices:
        logger.warning("=" * 60)
        logger.warning("Evaluation completed with %d failed samples", len(failed_indices))
        logger.warning("Failed indices (first 20): %s", failed_indices[:20])
        # 保存失败记录
        failed_path = os.path.join(args.output_dir, "failed_samples.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump({
                "failed_count": len(failed_indices),
                "failed_indices": failed_indices,
                "errors": failed_errors[:20],  # 只保存前20个错误
            }, f, indent=2, ensure_ascii=False)
        logger.warning("Failed samples record saved to %s", failed_path)
        logger.warning("=" * 60)

    # 重新加载所有预测，计算指标
    all_preds = load_predictions(pred_path)
    if not all_preds:
        logger.warning("No predictions found, skip metric computation.")
        return

    # 按 idx 排序，保证顺序一致
    ordered_indices = sorted(all_preds.keys())
    references: List[str] = []
    predictions: List[str] = []
    for idx in ordered_indices:
        rec = all_preds[idx]
        references.append(rec["reference"])
        predictions.append(rec["prediction"])

    # 计算整体指标
    metrics = compute_corpus_metrics(references, predictions)

    # 计算分组指标
    all_preds_list = list(all_preds.values())
    grouped_metrics = compute_all_grouped_metrics(all_preds_list)

    # 合并所有指标
    full_results = {
        "overall": metrics,
        "grouped": grouped_metrics,
        "failed_count": len(failed_indices),
    }

    metrics_path = os.path.join(args.output_dir, METRIC_FILENAME)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    logger.info("Finished evaluation. Metrics saved to %s", metrics_path)
    logger.info("Overall Metrics: %s", metrics)
    logger.info("Grouped Metrics - By Task: %s", list(grouped_metrics["by_task"].keys()))
    logger.info("Grouped Metrics - By Subject: %s", list(grouped_metrics["by_subject"].keys()))


if __name__ == "__main__":
    main()
