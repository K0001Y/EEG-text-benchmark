"""本地测试脚本 - 用于验证评估流程是否正常。

在正式服务器推理前，使用此脚本在本地测试：
1. 构建小规模测试数据集
2. 数据验证
3. dummy 模型评估流程测试
4. (可选) 小规模真实模型测试

Usage:
    python -m benchmark_eval.scripts.validation.local_test --step all
    python -m benchmark_eval.scripts.validation.local_test --step build_data
    python -m benchmark_eval.scripts.validation.local_test --step validate
    python -m benchmark_eval.scripts.validation.local_test --step eval_dummy
"""

import argparse
import os
import pickle
import sys
import shutil
from typing import List, Dict, Any

import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)


def create_mock_dataset(output_path: str, num_samples: int = 50, logger=None) -> str:
    """创建模拟数据集用于测试流程。
    
    不需要真实的 ZuCo 数据，生成格式正确的模拟数据。
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量
        logger: 日志记录器
    
    Returns:
        输出文件路径
    """
    if logger:
        logger.info("Creating mock dataset with %d samples...", num_samples)
    else:
        print(f"Creating mock dataset with {num_samples} samples...")
    
    # 模拟句子
    mock_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
        "Neural networks can process complex patterns.",
        "Brain signals contain rich information.",
        "EEG data reveals cognitive processes.",
        "Language understanding requires context.",
        "Deep learning models need large datasets.",
        "Attention mechanisms improve performance.",
        "Transformer architecture is widely used.",
        "Natural language processing advances rapidly.",
    ]
    
    samples: List[Dict[str, Any]] = []
    
    # 先按 text_uid 分配 phase，避免数据泄露
    n_texts = len(mock_sentences)
    n_train_texts = max(int(n_texts * 0.6), 1)
    n_val_texts = max(int(n_texts * 0.2), 1)
    
    text_phases = {}
    for idx, sent in enumerate(mock_sentences):
        if idx < n_train_texts:
            text_phases[sent] = "train"
        elif idx < n_train_texts + n_val_texts:
            text_phases[sent] = "val"
        else:
            text_phases[sent] = "test"
    
    # 生成模拟样本
    for i in range(num_samples):
        # 随机选择句子
        sentence = mock_sentences[i % len(mock_sentences)]
        
        # 生成随机 EEG 数据：(L_max=58, C=840)
        eeg = np.random.randn(58, 840).astype(np.float32)
        
        # 随机序列长度
        seq_len = np.random.randint(10, 58)
        mask = [1.0] * seq_len + [0.0] * (58 - seq_len)
        
        # 分配任务和被试
        task_idx = i % 3
        tasks = ["task1-SR", "task2-NR", "task3-TSR"]
        subjects = ["ZAB", "ZDM", "ZDN", "YAC", "YAG"]
        datasets = ["ZuCo1", "ZuCo2"]
        
        # 根据句子分配 phase（基于 text_uid）
        phase = text_phases[sentence]
        
        sample = {
            "eeg": eeg,
            "mask": mask,
            "input_text": sentence,
            "reference_text": sentence,
            "phase": phase,
            "meta": {
                "task": tasks[task_idx],
                "subject": subjects[i % len(subjects)],
                "dataset": datasets[i % len(datasets)],
                "sentence_index": i,
                "source": "mock-data",
                "text_uid": i % len(mock_sentences),
            }
        }
        
        samples.append(sample)
    
    # 保存数据集
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    
    if logger:
        logger.info("Mock dataset saved to %s", output_path)
    else:
        print(f"Mock dataset saved to {output_path}")
    
    # 打印统计
    phases = {}
    for s in samples:
        phase = s["phase"]
        phases[phase] = phases.get(phase, 0) + 1
    
    print(f"  Total samples: {len(samples)}")
    print(f"  By phase: {phases}")
    
    return output_path


def run_validation(data_path: str, output_dir: str):
    """运行数据验证。"""
    print("\n" + "=" * 60)
    print("Step: Data Validation")
    print("=" * 60)
    
    from benchmark_eval.data_processing.validate_dataset import validate_unified_dataset, check_label_coverage
    from benchmark_eval.utils.logging_utils import setup_logging, get_logger
    
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logger = get_logger("local_test")
    
    # 验证数据集
    stats = validate_unified_dataset(data_path, logger)
    
    # 加载样本检查标签
    with open(data_path, "rb") as f:
        samples = pickle.load(f)
    check_label_coverage(samples, logger)
    
    # 检查是否有问题
    if stats["nan_count"] > 0:
        print(f"[WARNING] Found {stats['nan_count']} samples with NaN values!")
    if stats["phase_leakage"]:
        print("[WARNING] Phase leakage detected!")
    else:
        print("[OK] Data validation passed!")
    
    return stats


def run_dummy_evaluation(data_path: str, output_dir: str, batch_size: int = 4):
    """使用 dummy 模型运行评估流程。"""
    print("\n" + "=" * 60)
    print("Step: Dummy Model Evaluation")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    from benchmark_eval.data_processing.dataset import UnifiedDataset, custom_collate_fn
    from benchmark_eval.evaluation.model_wrappers import build_model_wrapper
    from benchmark_eval.evaluation.metrics import compute_corpus_metrics
    from benchmark_eval.utils.logging_utils import setup_logging, get_logger
    
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logger = get_logger("local_test")
    
    # 加载数据集
    print(f"Loading dataset from {data_path}...")
    dataset = UnifiedDataset(data_path, phase="test")
    print(f"  Test samples: {len(dataset)}")
    
    # 创建 DataLoader（使用自定义 collate 函数）
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 构建 dummy 模型
    model = build_model_wrapper("dummy")
    print("  Model: DummyEchoWrapper (echoes input_text)")
    
    # 运行评估
    references = []
    predictions = []
    
    print("\nRunning inference...")
    for batch_idx, batch in enumerate(dataloader):
        eeg = batch["eeg"]
        mask = batch["mask"]
        
        # 构建 meta
        meta_batch = []
        for i in range(len(batch["idx"])):
            meta = dict(batch["meta"][i])
            meta["input_text"] = batch["input_text"][i]
            meta_batch.append(meta)
        
        # 生成预测
        preds = model.generate_text(eeg, mask, meta_batch)
        
        for i in range(len(preds)):
            references.append(batch["reference_text"][i])
            predictions.append(preds[i])
        
        print(f"  Batch {batch_idx + 1}/{len(dataloader)} processed")
    
    # 计算指标
    print("\nComputing metrics...")
    metrics = compute_corpus_metrics(references, predictions)
    
    print("\n" + "-" * 40)
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 40)
    
    # Dummy 模型应该有完美分数（因为预测 == 参考）
    if metrics.get("bleu4", 0) > 0.99:
        print("[OK] Dummy evaluation passed! (Perfect scores expected)")
    else:
        print("[WARNING] Dummy scores are not perfect, might indicate issues.")
    
    return metrics


def check_gpu():
    """检查 GPU 状态。"""
    print("\n" + "=" * 60)
    print("GPU Status Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"  CUDA available: True")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # 显存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        
        print(f"  Total memory: {total_mem:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Available: {total_mem - allocated:.2f} GB")
        
        # RTX 4070 通常有 12GB，足够运行较小的模型
        if total_mem >= 10:
            print("[OK] GPU memory sufficient for most models")
        else:
            print("[WARNING] Limited GPU memory, consider using CPU or smaller batch size")
    else:
        print("  CUDA available: False")
        print("[WARNING] No GPU detected, will use CPU (slower)")
    
    return torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser(description="Local test script for EEG-to-Text benchmark")
    parser.add_argument("--step", type=str, default="all",
                       choices=["all", "build_data", "validate", "eval_dummy", "gpu_check"],
                       help="Which step to run")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of mock samples to generate")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: benchmark_eval/test_outputs)")
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "benchmark_eval", "test_outputs")
    
    os.makedirs(args.output_dir, exist_ok=True)
    mock_data_path = os.path.join(args.output_dir, "mock_dataset.pkl")
    
    print("=" * 60)
    print("EEG-to-Text Benchmark - Local Test")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Step: {args.step}")
    
    # GPU 检查
    if args.step in ["all", "gpu_check"]:
        check_gpu()
    
    # 构建测试数据
    if args.step in ["all", "build_data"]:
        print("\n" + "=" * 60)
        print("Step: Build Mock Dataset")
        print("=" * 60)
        create_mock_dataset(mock_data_path, num_samples=args.num_samples)
    
    # 数据验证
    if args.step in ["all", "validate"]:
        if not os.path.exists(mock_data_path):
            print("[ERROR] Mock dataset not found. Run with --step build_data first.")
            return 1
        run_validation(mock_data_path, os.path.join(args.output_dir, "validation"))
    
    # Dummy 模型评估
    if args.step in ["all", "eval_dummy"]:
        if not os.path.exists(mock_data_path):
            print("[ERROR] Mock dataset not found. Run with --step build_data first.")
            return 1
        run_dummy_evaluation(mock_data_path, os.path.join(args.output_dir, "eval_dummy"), args.batch_size)
    
    print("\n" + "=" * 60)
    print("Local Test Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. If all tests passed, you can run real model evaluation")
    print("2. Use smaller batch_size (2-4) for 4070 to avoid OOM")
    print("3. Consider using --num-workers 0 to avoid multiprocessing issues on Windows")
    
    return 0


if __name__ == "__main__":
    exit(main())
