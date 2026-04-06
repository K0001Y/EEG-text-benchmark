"""数据一致性验证脚本

用于验证统一数据与原始模型数据的一致性。
"""

import argparse
import pickle
import numpy as np
import torch
from typing import Dict, Tuple
import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_unified_data(data_path: str, task: str, subject: str, sent_idx: int):
    """加载统一数据中的指定样本"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    for sample in data:
        meta = sample.get('meta', {})
        if (meta.get('task') == task and 
            meta.get('subject') == subject and 
            meta.get('sentence_index') == sent_idx):
            return sample
    
    return None


def load_cet_mae_original(
    pickle_path: str,
    task: str,
    subject: str,
    sent_idx: int
) -> Dict:
    """加载CET-MAE原始pickle数据"""
    with open(pickle_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    sent_list = dataset_dict.get(subject, [])
    if sent_idx < len(sent_list):
        return sent_list[sent_idx]
    return None


def compare_cet_mae(
    unified_data_path: str,
    cet_mae_pickle_path: str,
    task: str = "task2-NR",
    subject: str = "ZAB",
    sent_idx: int = 0,
    tolerance: float = 1e-5
) -> Tuple[bool, Dict]:
    """比较CET-MAE统一数据和原始数据
    
    Returns:
        (是否一致, 详细信息字典)
    """
    print(f"\n=== CET-MAE数据一致性验证 ===")
    print(f"样本: task={task}, subject={subject}, sent_idx={sent_idx}")
    
    # 加载数据
    unified = load_unified_data(unified_data_path, task, subject, sent_idx)
    original = load_cet_mae_original(cet_mae_pickle_path, task, subject, sent_idx)
    
    if unified is None:
        return False, {"error": "统一数据未找到样本"}
    if original is None:
        return False, {"error": "原始数据未找到样本"}
    
    results = {
        "sample_info": {
            "task": task,
            "subject": subject,
            "sent_idx": sent_idx,
            "text": unified.get('input_text', 'N/A')
        },
        "checks": {}
    }
    
    # 检查1: 文本一致性
    original_text = original.get('content', '')
    unified_text = unified.get('input_text', '')
    text_match = (original_text == unified_text)
    results["checks"]["text"] = {
        "match": text_match,
        "original": original_text[:50] + "..." if len(original_text) > 50 else original_text,
        "unified": unified_text[:50] + "..." if len(unified_text) > 50 else unified_text
    }
    
    # 检查2: eeg_normalized_2d vs 原始eeg
    if 'eeg_normalized_2d' in unified and 'eeg' in original:
        unified_eeg = unified['eeg_normalized_2d']
        original_eeg = original['eeg'].numpy() if torch.is_tensor(original['eeg']) else original['eeg']
        
        diff = np.abs(unified_eeg - original_eeg).max()
        results["checks"]["eeg_normalized_2d"] = {
            "match": diff < tolerance,
            "max_diff": float(diff),
            "tolerance": tolerance,
            "shape_unified": list(unified_eeg.shape),
            "shape_original": list(original_eeg.shape)
        }
    else:
        results["checks"]["eeg_normalized_2d"] = {
            "match": False,
            "error": "缺少eeg_normalized_2d或原始eeg字段"
        }
    
    # 检查3: mask一致性
    if 'mask_with_sent' in unified and 'mask' in original:
        unified_mask = np.array(unified['mask_with_sent'])
        original_mask = original['mask'].numpy() if torch.is_tensor(original['mask']) else np.array(original['mask'])
        
        mask_match = np.allclose(unified_mask, original_mask)
        results["checks"]["mask"] = {
            "match": bool(mask_match),
            "sum_unified": int(unified_mask.sum()),
            "sum_original": int(original_mask.sum())
        }
    
    # 总体结果
    all_match = all(check.get("match", False) for check in results["checks"].values())
    results["overall_match"] = all_match
    
    return all_match, results


def load_eeg2text_original(
    spectro_pickle_path: str,
    subject: str,
    sent_idx: int
) -> Dict:
    """加载EEG2Text原始spectro pickle数据"""
    with open(spectro_pickle_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    sent_list = dataset_dict.get(subject, [])
    if sent_idx < len(sent_list):
        return sent_list[sent_idx]
    return None


def compare_eeg2text(
    unified_data_path: str,
    spectro_pickle_path: str,
    task: str = "task2-NR",
    subject: str = "ZAB",
    sent_idx: int = 0,
    tolerance: float = 1e-3
) -> Tuple[bool, Dict]:
    """比较EEG2Text统一数据和原始spectro数据"""
    print(f"\n=== EEG2Text数据一致性验证 ===")
    print(f"样本: task={task}, subject={subject}, sent_idx={sent_idx}")
    
    # 加载统一数据
    unified = load_unified_data(unified_data_path, task, subject, sent_idx)
    
    # 加载原始spectro数据
    original = load_eeg2text_original(spectro_pickle_path, subject, sent_idx)
    
    if unified is None:
        return False, {"error": "统一数据未找到样本"}
    if original is None:
        return False, {"error": "原始数据未找到样本"}
    
    results = {
        "sample_info": {
            "task": task,
            "subject": subject,
            "sent_idx": sent_idx
        },
        "checks": {}
    }
    
    # 检查: eeg_eeg2text vs rawData
    if 'eeg_eeg2text' in unified and 'sentence_level_EEG' in original:
        unified_eeg = unified['eeg_eeg2text']
        raw_data = original['sentence_level_EEG'].get('rawData')
        
        if raw_data is not None:
            original_eeg = raw_data.numpy() if torch.is_tensor(raw_data) else raw_data
            
            # 转置为 (T, 105) 如果原始 shape 是 (105, T)
            if original_eeg.shape[0] == 105 and len(original_eeg.shape) == 2:
                original_eeg = original_eeg.T
            
            # 截取/填充到相同长度
            min_len = min(unified_eeg.shape[0], original_eeg.shape[0])
            unified_eeg_crop = unified_eeg[:min_len]
            original_eeg_crop = original_eeg[:min_len]
            
            # 计算相关性（因为可能有缩放差异）
            if unified_eeg_crop.size > 0 and original_eeg_crop.size > 0:
                try:
                    corr = np.corrcoef(
                        unified_eeg_crop.flatten(),
                        original_eeg_crop.flatten()
                    )[0, 1]
                except:
                    corr = 0.0
            else:
                corr = 0.0
            
            results["checks"]["eeg_eeg2text"] = {
                "correlation": float(corr),
                "match": corr > 0.99,
                "shape_unified": list(unified_eeg.shape),
                "shape_original": list(original_eeg.shape)
            }
        else:
            results["checks"]["eeg_eeg2text"] = {
                "match": False,
                "error": "原始数据缺少rawData"
            }
    else:
        results["checks"]["eeg_eeg2text"] = {
            "match": False,
            "error": "缺少eeg_eeg2text或原始数据字段"
        }
    
    all_match = all(check.get("match", False) for check in results["checks"].values())
    results["overall_match"] = all_match
    
    return all_match, results


def main():
    parser = argparse.ArgumentParser(description="验证统一数据与原始模型数据的一致性")
    parser.add_argument("--unified-data", required=True, help="统一数据pickle路径")
    parser.add_argument("--cet-mae-pickle", help="CET-MAE原始pickle路径")
    parser.add_argument("--eeg2text-spectro", help="EEG2Text原始spectro pickle路径")
    parser.add_argument("--task", default="task2-NR")
    parser.add_argument("--subject", default="ZAB")
    parser.add_argument("--sent-idx", type=int, default=0)
    parser.add_argument("--output", help="输出结果JSON路径")
    
    args = parser.parse_args()
    
    all_results = {}
    
    # 验证CET-MAE
    if args.cet_mae_pickle:
        match, results = compare_cet_mae(
            args.unified_data,
            args.cet_mae_pickle,
            args.task,
            args.subject,
            args.sent_idx
        )
        all_results["cet_mae"] = results
        
        print(f"\nCET-MAE验证结果: {'通过' if match else '失败'}")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✓" if check_result.get("match") else "✗"
            print(f"  {status} {check_name}: {check_result}")
    
    # 验证EEG2Text
    if args.eeg2text_spectro:
        match, results = compare_eeg2text(
            args.unified_data,
            args.eeg2text_spectro,
            args.task,
            args.subject,
            args.sent_idx
        )
        all_results["eeg2text"] = results
        
        print(f"\nEEG2Text验证结果: {'通过' if match else '失败'}")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✓" if check_result.get("match") else "✗"
            print(f"  {status} {check_name}: {check_result}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n详细结果已保存到: {args.output}")
    
    # 总体结果
    overall = all(r.get("overall_match", False) for r in all_results.values())
    print(f"\n{'='*50}")
    print(f"总体验证结果: {'全部通过' if overall else '存在失败'}")
    print(f"{'='*50}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    exit(main())
