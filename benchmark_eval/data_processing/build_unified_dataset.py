import argparse
import os
import pickle
from typing import Any, Dict, List
import csv
import sys

import numpy as np
import torch
from tqdm import tqdm
import scipy.io as io
import h5py
from glob import glob

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.logging_utils import setup_logging, get_logger


DEFAULT_BANDS = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]


def normalize_1d(input_tensor: torch.Tensor) -> torch.Tensor:
    """对 1D 张量做 z-score 归一化，避免除以 0。"""
    mean = input_tensor.mean()
    std = input_tensor.std()
    if std == 0 or torch.isnan(std):
        return torch.zeros_like(input_tensor)
    return (input_tensor - mean) / std


def normalize_2d(input_matrix: torch.Tensor) -> torch.Tensor:
    """对 2D 张量整体做 z-score 归一化。"""
    flattened = input_matrix.view(-1)
    mean = flattened.mean()
    std = flattened.std()
    if std == 0 or torch.isnan(std):
        return torch.zeros_like(input_matrix)
    return (input_matrix - mean) / std


def build_eeg2text_format(raw_data: np.ndarray, target_len: int = 24000) -> np.ndarray:
    """将原始 EEG 时序数据转换为 EEG2Text 格式。
    
    Args:
        raw_data: 原始 EEG 数据，shape 为 (105, T) 或 (T, 105)
        target_len: 目标时间长度，默认 24000
        
    Returns:
        转换后的 EEG 数据，shape 为 (target_len, 105)
    """
    # 确保数据是 numpy 数组
    if hasattr(raw_data, 'numpy'):
        raw_data = raw_data.numpy()
    
    raw_data = np.asarray(raw_data, dtype=np.float32)
    
    # 判断输入 shape，如果是 (105, T) 则转置为 (T, 105)
    if raw_data.shape[0] == 105 and len(raw_data.shape) == 2:
        raw_data = raw_data.T  # 现在 shape 为 (T, 105)
    
    actual_len = raw_data.shape[0]
    
    # 截断或填充到 target_len
    if actual_len >= target_len:
        result = raw_data[:target_len, :]
    else:
        # 填充零
        pad_len = target_len - actual_len
        pad = np.zeros((pad_len, 105), dtype=np.float32)
        result = np.concatenate([raw_data, pad], axis=0)
    
    return result


def build_eeg2text_mask(actual_len: int, target_len: int = 24000) -> List[float]:
    """构建 EEG2Text 的 mask。
    
    Args:
        actual_len: 实际时间长度
        target_len: 目标时间长度
        
    Returns:
        mask 列表，长度为 target_len，有效位置为 1.0，padding 为 0.0
    """
    effective_len = min(actual_len, target_len)
    mask = [1.0] * effective_len + [0.0] * (target_len - effective_len)
    return mask


def get_word_embedding_eeg_tensor(word_obj: Dict[str, Any], eeg_type: str, bands: List[str], dim: int) -> Dict[str, torch.Tensor] | None:
    """从 EEG-To-Text 的 sent_obj.word 中构造单词级 EEG 向量。

    返回：
    - normalized: 归一化后的 1D 张量
    - raw:        未归一化的 1D 张量
    如果维度或数值异常，返回 None。
    """
    features = []
    for band in bands:
        key = eeg_type + band
        try:
            arr = word_obj["word_level_EEG"][eeg_type][key][0:dim]
        except Exception:
            return None
        features.append(arr)
    vec = np.concatenate(features)
    if len(vec) != dim * len(bands):
        return None
    raw = torch.from_numpy(vec.astype("float32"))
    norm = normalize_1d(raw)
    if torch.isnan(norm).any():
        return None
    return {"normalized": norm, "raw": raw}


def get_sent_eeg(sent_obj: Dict[str, Any], bands: List[str], dim: int) -> Dict[str, torch.Tensor] | None:
    """从 sentence_level_EEG 中构造句级 EEG 向量。"""
    features = []
    try:
        for band in bands:
            key = "mean" + band
            features.append(sent_obj["sentence_level_EEG"][key][0:dim])
    except Exception:
        return None
    vec = np.concatenate(features)
    if len(vec) != dim * len(bands):
        return None
    raw = torch.from_numpy(vec.astype("float32"))
    norm = normalize_1d(raw)
    if torch.isnan(norm).any():
        return None
    return {"normalized": norm, "raw": raw}


def build_samples_for_task(
    dataset: Dict[str, List[Any]],
    task_name: str,
    max_len: int = 56,  # 与 EEG-To-Text 原始保持一致
    dim: int = 105,
    eeg_type: str = "GD",
    bands: List[str] | None = None,
    logger=None,
) -> List[Dict[str, Any]]:
    """从 ZuCo 原始 MAT（经统一解析为 dataset_dict）构造统一格式样本。

    参数中的 dataset 结构应为:
    - key: subject 名称
    - value: 该 subject 的句子列表，每个元素是 sent_obj 或 None
      sent_obj 至少包含:
        - "content": 句子文本
        - "sentence_level_EEG": 带有 mean_t1...mean_g2 字段的字典
        - "word": 词级列表, 每个元素含 "word_level_EEG" 下的 eeg_type/bands 数据
    
    存储多种格式供不同模型使用:
    - eeg_raw: 原始词级 EEG（未归一化），shape (max_len, 840)
    - sent_eeg_raw: 句级 EEG（单独存储），shape (840,)
    - eeg_normalized_1d: 逐词 1D 归一化版本（EEG-To-Text 使用）
    - eeg_normalized_2d: 词级+句级 2D 全局归一化版本（CET-MAE 等使用）
    - mask: 基于词数的 mask，不包含句级 EEG
    - seq_len: 原始词数
    """
    if bands is None:
        bands = list(DEFAULT_BANDS)

    if logger is None:
        logger = get_logger("build_unified_dataset")

    samples_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    feature_dim = dim * len(bands)  # 840

    for subject, sent_list in dataset.items():
        subject_records: List[Dict[str, Any]] = []
        for sent_idx, sent_obj in enumerate(sent_list):
            if sent_obj is None:
                continue
            if "word" not in sent_obj or not sent_obj["word"]:
                continue

            # 收集词级 EEG 原始向量
            word_embeddings_raw: List[torch.Tensor] = []
            for word in sent_obj["word"]:
                t = get_word_embedding_eeg_tensor(word, eeg_type=eeg_type, bands=bands, dim=dim)
                if t is None:
                    word_embeddings_raw = []
                    break
                word_embeddings_raw.append(t["raw"])

            if not word_embeddings_raw:
                continue

            # 获取句级 EEG
            sent_eeg = get_sent_eeg(sent_obj, bands=bands, dim=dim)
            if sent_eeg is None:
                continue

            num_words = len(word_embeddings_raw)
            
            # 截断到 max_len（仅针对词级）
            if num_words > max_len:
                word_embeddings_raw = word_embeddings_raw[:max_len]
                num_words = max_len

            # ===== 1. 构建 eeg_raw：原始词级 EEG，padding 到 max_len =====
            raw_padded = word_embeddings_raw.copy()
            while len(raw_padded) < max_len:
                raw_padded.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_raw = torch.stack(raw_padded).numpy().astype("float32")  # (max_len, 840)

            # ===== 2. 构建 eeg_normalized_1d：逐词 1D 归一化（EEG-To-Text 使用）=====
            norm_1d_list = []
            for i, vec in enumerate(word_embeddings_raw):
                norm_1d_list.append(normalize_1d(vec))
            # padding
            while len(norm_1d_list) < max_len:
                norm_1d_list.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_normalized_1d = torch.stack(norm_1d_list)
            if torch.isnan(eeg_normalized_1d).any():
                continue
            eeg_normalized_1d = eeg_normalized_1d.numpy().astype("float32")  # (max_len, 840)

            # ===== 3. 构建 eeg_normalized_2d：词级+句级 2D 归一化（CET-MAE 等使用）=====
            # 将句级 EEG 作为最后一个 time step 加入
            all_raw_with_sent = word_embeddings_raw + [sent_eeg["raw"]]
            raw_matrix_with_sent = torch.stack(all_raw_with_sent)  # (num_words+1, 840)
            norm_2d_matrix = normalize_2d(raw_matrix_with_sent)
            if torch.isnan(norm_2d_matrix).any():
                continue
            # padding 到 max_len (注意这里包含句级，所以有效长度是 num_words+1)
            seq_len_with_sent = len(all_raw_with_sent)
            norm_2d_list = list(torch.unbind(norm_2d_matrix, dim=0))
            if seq_len_with_sent > max_len:
                norm_2d_list = norm_2d_list[:max_len]
                seq_len_with_sent = max_len
            while len(norm_2d_list) < max_len:
                norm_2d_list.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_normalized_2d = torch.stack(norm_2d_list).numpy().astype("float32")  # (max_len, 840)

            # ===== 4. 句级 EEG 单独存储 =====
            sent_eeg_raw = sent_eeg["raw"].numpy().astype("float32")  # (840,)

            # ===== 5. mask：基于词数，不包含句级 EEG =====
            mask = [1.0] * num_words + [0.0] * (max_len - num_words)
            
            # ===== 6. mask_with_sent：包含句级 EEG 的 mask =====
            mask_with_sent = [1.0] * min(seq_len_with_sent, max_len) + [0.0] * (max_len - min(seq_len_with_sent, max_len))

            record: Dict[str, Any] = {
                # 多格式 EEG 数据
                "eeg_raw": eeg_raw,                      # 原始词级，未归一化
                "eeg_normalized_1d": eeg_normalized_1d,  # 逐词 1D 归一化（EEG-To-Text）
                "eeg_normalized_2d": eeg_normalized_2d,  # 词+句 2D 归一化（CET-MAE）
                "sent_eeg_raw": sent_eeg_raw,            # 句级 EEG，单独存储
                # 兼容旧字段（使用 1D 归一化版本作为默认）
                "eeg": eeg_normalized_1d,
                # mask
                "mask": mask,                            # 词级 mask
                "mask_with_sent": mask_with_sent,        # 包含句级的 mask
                # 文本
                "input_text": sent_obj.get("content", ""),
                "reference_text": sent_obj.get("content", ""),
                "phase": None,  # 稍后填入 train/val/test
                "meta": {
                    "task": task_name,
                    "subject": subject,
                    "sentence_index": int(sent_idx),
                    "source": "ZuCo-MAT",
                    "seq_len": num_words,                # 原始词数
                    "seq_len_with_sent": min(seq_len_with_sent, max_len),  # 包含句级的长度
                },
            }
            
            # ===== 7. EEG2Text 格式：原始时序数据 =====
            # 从 sent_obj 中获取 rawData（如果存在）
            raw_data = None
            if "rawData" in sent_obj:
                raw_data = sent_obj["rawData"]
            elif "sentence_level_EEG" in sent_obj and "rawData" in sent_obj["sentence_level_EEG"]:
                raw_data = sent_obj["sentence_level_EEG"]["rawData"]
            
            if raw_data is not None:
                try:
                    record["eeg_eeg2text"] = build_eeg2text_format(raw_data, target_len=24000)
                    # 获取原始时间长度
                    if hasattr(raw_data, 'shape'):
                        if len(raw_data.shape) == 2:
                            actual_len = raw_data.shape[1] if raw_data.shape[0] == 105 else raw_data.shape[0]
                        else:
                            actual_len = raw_data.shape[0]
                    else:
                        actual_len = 24000
                    record["mask_eeg2text"] = build_eeg2text_mask(actual_len, target_len=24000)
                except Exception as e:
                    if logger:
                        logger.warning("Failed to build EEG2Text format for %s sent %d: %s", subject, sent_idx, e)
            subject_records.append(record)

        samples_by_subject[subject] = subject_records
        logger.info("Task %s, subject %s: %d valid sentences", task_name, subject, len(subject_records))

    # 基于 text_uid 划分 train/val/test，避免数据泄露
    unified: List[Dict[str, Any]] = []
    
    # 收集所有唯一的 text_uid（基于 input_text）
    text_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for subject, records in samples_by_subject.items():
        for rec in records:
            text = rec.get("input_text", "")
            if text not in text_to_records:
                text_to_records[text] = []
            text_to_records[text].append(rec)
    
    # 按 text_uid 划分
    import random
    random.seed(42)  # 固定随机种子保证可复现
    
    # 先排序保证初始顺序确定性，与 LazyZuCo_dataset 的划分完全一致
    unique_texts = sorted(text_to_records.keys())
    
    n_total = len(unique_texts)
    n_train = max(int(n_total * 0.8), 1)
    n_val = max(int(n_total * 0.1), 0)
    if n_train + n_val >= n_total:
        n_train = max(n_total - 2, 1)
        n_val = max(min(n_total - n_train - 1, 1), 0)
    
    train_texts = set(unique_texts[:n_train])
    val_texts = set(unique_texts[n_train:n_train + n_val])
    test_texts = set(unique_texts[n_train + n_val:])
    
    # 为每个样本分配 phase
    train_count = val_count = test_count = 0
    for text, records in text_to_records.items():
        if text in train_texts:
            phase = "train"
            train_count += len(records)
        elif text in val_texts:
            phase = "val"
            val_count += len(records)
        else:
            phase = "test"
            test_count += len(records)
        
        for rec in records:
            rec["phase"] = phase
            unified.append(rec)
    
    logger.info(
        "Task %s: total_texts=%d, train_texts=%d, val_texts=%d, test_texts=%d",
        task_name, n_total, len(train_texts), len(val_texts), len(test_texts)
    )
    logger.info(
        "Task %s: total_samples=%d, train=%d, val=%d, test=%d",
        task_name, len(unified), train_count, val_count, test_count
    )

    logger.info("Task %s: collected %d unified samples", task_name, len(unified))
    return unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified EEG-text dataset directly from ZuCo MAT files")
    parser.add_argument("--zuco-root", type=str, default="models/EEG-To-Text-main/dataset/ZuCo", help="Path to root directory that contains ZuCo task folders with Matlab_files")
    parser.add_argument(
        "--tasks",
        type=str,
        default="task1-SR,task2-NR,task3-TSR,task2-NR-2.0",
        help="Comma-separated ZuCo task names to include",
    )
    parser.add_argument("--output", type=str, required=True, help="Output unified dataset pickle path")
    parser.add_argument("--max-len", type=int, default=56, help="Max time steps L_max for EEG sequence (default 56, same as EEG-To-Text)")
    parser.add_argument("--dim", type=int, default=105, help="Number of EEG channels per band")
    parser.add_argument("--eeg-type", type=str, default="GD", help="EEG type key used in word_level_EEG (e.g. GD/FFD/TRT)")
    return parser.parse_args()


def load_dataset_from_mat_v1(zuco_root: str, task_name: str, logger=None) -> Dict[str, List[Any]]:
    """从 ZuCo v1 .mat 文件构造与 EEG-To-Text 相同结构的 dataset_dict。

    dataset_dict: subject -> List[sent_obj or None]
    sent_obj 至少包括 content / sentence_level_EEG / word / word_level_EEG。
    """
    if logger is None:
        logger = get_logger("load_dataset_from_mat_v1")

    input_mat_files_dir = os.path.join(zuco_root, task_name, "Matlab_files")
    mat_files = sorted(glob(os.path.join(input_mat_files_dir, "*.mat")))
    if not mat_files:
        logger.warning("No mat files found for task %s under %s", task_name, input_mat_files_dir)
        return {}

    dataset_dict: Dict[str, List[Any]] = {}
    logger.info("Start processing ZuCo v1 task %s from %s", task_name, input_mat_files_dir)

    for mat_file in tqdm(mat_files, desc=f"ZuCo-{task_name}"):
        subject_name = os.path.basename(mat_file).split("_")[0].replace("results", "").strip()
        dataset_dict[subject_name] = []

        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)["sentenceData"]

        for sent in matdata:
            word_data = sent.word
            if not isinstance(word_data, float):
                sent_obj: Dict[str, Any] = {"content": sent.content}
                sent_obj["sentence_level_EEG"] = {
                    "mean_t1": sent.mean_t1,
                    "mean_t2": sent.mean_t2,
                    "mean_a1": sent.mean_a1,
                    "mean_a2": sent.mean_a2,
                    "mean_b1": sent.mean_b1,
                    "mean_b2": sent.mean_b2,
                    "mean_g1": sent.mean_g1,
                    "mean_g2": sent.mean_g2,
                }
                # 保留原始时序数据（用于 EEG2Text）
                if hasattr(sent, 'rawData'):
                    sent_obj["rawData"] = sent.rawData

                sent_obj["word"] = []
                word_tokens_has_fixation: List[str] = []
                word_tokens_with_mask: List[str] = []
                word_tokens_all: List[str] = []

                for word in word_data:
                    word_obj: Dict[str, Any] = {"content": word.content}
                    word_tokens_all.append(word.content)
                    word_obj["nFixations"] = word.nFixations
                    if word.nFixations > 0:
                        word_obj["word_level_EEG"] = {
                            "FFD": {
                                "FFD_t1": word.FFD_t1,
                                "FFD_t2": word.FFD_t2,
                                "FFD_a1": word.FFD_a1,
                                "FFD_a2": word.FFD_a2,
                                "FFD_b1": word.FFD_b1,
                                "FFD_b2": word.FFD_b2,
                                "FFD_g1": word.FFD_g1,
                                "FFD_g2": word.FFD_g2,
                            }
                        }
                        word_obj["word_level_EEG"]["TRT"] = {
                            "TRT_t1": word.TRT_t1,
                            "TRT_t2": word.TRT_t2,
                            "TRT_a1": word.TRT_a1,
                            "TRT_a2": word.TRT_a2,
                            "TRT_b1": word.TRT_b1,
                            "TRT_b2": word.TRT_b2,
                            "TRT_g1": word.TRT_g1,
                            "TRT_g2": word.TRT_g2,
                        }
                        word_obj["word_level_EEG"]["GD"] = {
                            "GD_t1": word.GD_t1,
                            "GD_t2": word.GD_t2,
                            "GD_a1": word.GD_a1,
                            "GD_a2": word.GD_a2,
                            "GD_b1": word.GD_b1,
                            "GD_b2": word.GD_b2,
                            "GD_g1": word.GD_g1,
                            "GD_g2": word.GD_g2,
                        }
                        sent_obj["word"].append(word_obj)
                        word_tokens_has_fixation.append(word.content)
                        word_tokens_with_mask.append(word.content)
                    else:
                        word_tokens_with_mask.append("[MASK]")
                        # 当前策略: 无注视词在词级 EEG 中直接跳过
                        continue

                sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
                sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
                sent_obj["word_tokens_all"] = word_tokens_all

                dataset_dict[subject_name].append(sent_obj)
            else:
                logger.warning("Missing sentence: subj=%s content=%s, append None", subject_name, getattr(sent, "content", ""))
                dataset_dict[subject_name].append(None)

    return dataset_dict


def load_dataset_from_mat_v2(zuco_root: str, logger=None) -> Dict[str, List[Any]]:
    """从 ZuCo v2 task2-NR-2.0 HDF5 .mat 文件构造 dataset_dict。"""
    if logger is None:
        logger = get_logger("load_dataset_from_mat_v2")

    task_name = "task2-NR-2.0"
    rootdir = os.path.join(zuco_root, task_name, "Matlab_files")

    if not os.path.isdir(rootdir):
        logger.warning("ZuCo v2 rootdir not found for %s: %s", task_name, rootdir)
        return {}

    dataset_dict: Dict[str, List[Any]] = {}
    logger.info("Start processing ZuCo v2 task2-NR-2.0 from %s", rootdir)

    for file in tqdm(os.listdir(rootdir), desc="ZuCo-task2-NR-2.0"):
        if not file.endswith("NR.mat"):
            continue

        file_name = os.path.join(rootdir, file)
        subject = file_name.split("ts")[1].split("_")[0]
        # 排除 YMH（数据不完整）
        if subject == "YMH":
            continue
        if subject in dataset_dict:
            logger.warning("Duplicate subject %s in v2 files, skipping.", subject)
            continue

        dataset_dict[subject] = []

        f = h5py.File(file_name, "r")
        sentence_data = f["sentenceData"]

        mean_t1_objs = sentence_data["mean_t1"]
        mean_t2_objs = sentence_data["mean_t2"]
        mean_a1_objs = sentence_data["mean_a1"]
        mean_a2_objs = sentence_data["mean_a2"]
        mean_b1_objs = sentence_data["mean_b1"]
        mean_b2_objs = sentence_data["mean_b2"]
        mean_g1_objs = sentence_data["mean_g1"]
        mean_g2_objs = sentence_data["mean_g2"]

        rawData = sentence_data["rawData"]
        contentData = sentence_data["content"]
        wordData = sentence_data["word"]

        # 动态导入 EEG-To-Text 的工具模块
        import importlib.util
        eeg_to_text_util_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "EEG-To-Text-main", "util", "data_loading_helpers_modified.py"
        )
        spec = importlib.util.spec_from_file_location("data_loading_helpers_modified", eeg_to_text_util_path)
        dh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dh)

        for idx in range(len(rawData)):
            obj_reference_content = contentData[idx][0]
            sent_string = dh.load_matlab_string(f[obj_reference_content])

            sent_obj: Dict[str, Any] = {"content": sent_string}
            sent_obj["sentence_level_EEG"] = {
                "mean_t1": np.squeeze(f[mean_t1_objs[idx][0]][()]),
                "mean_t2": np.squeeze(f[mean_t2_objs[idx][0]][()]),
                "mean_a1": np.squeeze(f[mean_a1_objs[idx][0]][()]),
                "mean_a2": np.squeeze(f[mean_a2_objs[idx][0]][()]),
                "mean_b1": np.squeeze(f[mean_b1_objs[idx][0]][()]),
                "mean_b2": np.squeeze(f[mean_b2_objs[idx][0]][()]),
                "mean_g1": np.squeeze(f[mean_g1_objs[idx][0]][()]),
                "mean_g2": np.squeeze(f[mean_g2_objs[idx][0]][()]),
            }

            sent_obj["word"] = []

            word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(
                f, f[wordData[idx][0]]
            )

            if word_data == {} or len(word_tokens_all) == 0:
                logger.warning("Missing or empty word-level features: subj=%s content=%s, append None", subject, sent_string)
                dataset_dict[subject].append(None)
                continue

            for widx in range(len(word_data)):
                data_dict = word_data[widx]
                word_obj: Dict[str, Any] = {"content": data_dict["content"], "nFixations": data_dict["nFix"]}
                if "GD_EEG" in data_dict:
                    gd = data_dict["GD_EEG"]
                    ffd = data_dict["FFD_EEG"]
                    trt = data_dict["TRT_EEG"]
                    assert len(gd) == len(trt) == len(ffd) == 8
                    word_obj["word_level_EEG"] = {
                        "GD": {
                            "GD_t1": gd[0],
                            "GD_t2": gd[1],
                            "GD_a1": gd[2],
                            "GD_a2": gd[3],
                            "GD_b1": gd[4],
                            "GD_b2": gd[5],
                            "GD_g1": gd[6],
                            "GD_g2": gd[7],
                        },
                        "FFD": {
                            "FFD_t1": ffd[0],
                            "FFD_t2": ffd[1],
                            "FFD_a1": ffd[2],
                            "FFD_a2": ffd[3],
                            "FFD_b1": ffd[4],
                            "FFD_b2": ffd[5],
                            "FFD_g1": ffd[6],
                            "FFD_g2": ffd[7],
                        },
                        "TRT": {
                            "TRT_t1": trt[0],
                            "TRT_t2": trt[1],
                            "TRT_a1": trt[2],
                            "TRT_a2": trt[3],
                            "TRT_b1": trt[4],
                            "TRT_b2": trt[5],
                            "TRT_g1": trt[6],
                            "TRT_g2": trt[7],
                        },
                    }
                    sent_obj["word"].append(word_obj)

            sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
            sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
            sent_obj["word_tokens_all"] = word_tokens_all

            dataset_dict[subject].append(sent_obj)

    return dataset_dict


def _load_task1_sentiment_labels(materials_dir: str, logger) -> Dict[str, int]:
    """加载 ZuCo1 task1 的情感标签，返回 sentence -> label 映射。"""
    path = os.path.join(materials_dir, "sentiment_labels_task1.csv")
    mapping: Dict[str, int] = {}
    if not os.path.isfile(path):
        logger.warning("Sentiment label file not found: %s", path)
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sid = (row.get("sentence_id") or "").strip()
            if not sid or sid.startswith("#"):
                continue
            sentence = (row.get("sentence") or "").strip()
            label_str = (row.get("sentiment_label") or "").strip()
            if not sentence or not label_str:
                continue
            try:
                label = int(label_str)
            except ValueError:
                continue
            if sentence in mapping and mapping[sentence] != label:
                logger.warning(
                    "Conflicting sentiment labels for sentence %r: %r vs %r",
                    sentence,
                    mapping[sentence],
                    label,
                )
            mapping[sentence] = label

    logger.info("Loaded %d sentiment labels from %s", len(mapping), path)
    return mapping


def _load_task2_relation_labels(materials_dir: str, logger) -> Dict[str, str]:
    """加载 ZuCo1 task2 的关系标签，返回 sentence -> relation_types 映射。"""
    path = os.path.join(materials_dir, "relations_labels_task2.csv")
    mapping: Dict[str, str] = {}
    if not os.path.isfile(path):
        logger.warning("Relation label file for task2 not found: %s", path)
        return mapping

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            rel = (row.get("relation_types") or "").strip()
            if not sentence or not rel:
                continue
            if sentence in mapping and mapping[sentence] != rel:
                logger.warning(
                    "Conflicting relation labels (task2) for sentence %r: %r vs %r",
                    sentence,
                    mapping[sentence],
                    rel,
                )
            mapping[sentence] = rel

    logger.info("Loaded %d task2 relation labels from %s", len(mapping), path)
    return mapping


def _load_task3_relation_labels(materials_dir: str, logger) -> Dict[str, str]:
    """加载 ZuCo1 task3 的关系标签，返回 sentence -> relation_type 映射。"""
    path = os.path.join(materials_dir, "relations_labels_task3.csv")
    mapping: Dict[str, str] = {}
    if not os.path.isfile(path):
        logger.warning("Relation label file for task3 not found: %s", path)
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            rel = (row.get("relation-type") or "").strip()
            if not sentence or not rel:
                continue
            if sentence in mapping and mapping[sentence] != rel:
                logger.warning(
                    "Conflicting relation labels (task3) for sentence %r: %r vs %r",
                    sentence,
                    mapping[sentence],
                    rel,
                )
            mapping[sentence] = rel

    logger.info("Loaded %d task3 relation labels from %s", len(mapping), path)
    return mapping


def _enrich_samples_with_metadata_and_labels(samples: List[Dict[str, Any]], logger) -> None:
    """补全 meta 中的 dataset/text_uid 以及情感与关系标签。"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root 是 benchmark_eval，需要上一级到 benchmark
    project_root = os.path.dirname(this_dir)  # benchmark_eval
    benchmark_root = os.path.dirname(project_root)  # benchmark
    zuco1_materials = os.path.join(benchmark_root, "data", "ZuCo1", "task_materials")

    sentiment_map = _load_task1_sentiment_labels(zuco1_materials, logger)
    rel_task2_map = _load_task2_relation_labels(zuco1_materials, logger)
    rel_task3_map = _load_task3_relation_labels(zuco1_materials, logger)

    text_uid_map: Dict[tuple[str, str, str], int] = {}
    next_uid = 0

    for rec in samples:
        meta = rec.get("meta") or {}
        task_name = meta.get("task", "")

        # 数据集标记：当前简单按 task_name 区分 ZuCo1 / ZuCo2
        dataset_name = "ZuCo2" if task_name == "task2-NR-2.0" else "ZuCo1"
        if "dataset" not in meta:
            meta["dataset"] = dataset_name

        sentence = rec.get("input_text", "")

        # 为 (dataset, task, sentence) 分配稳定的 text_uid
        uid_key = (dataset_name, task_name, sentence)
        if uid_key not in text_uid_map:
            text_uid_map[uid_key] = next_uid
            next_uid += 1
        if "text_uid" not in meta:
            meta["text_uid"] = text_uid_map[uid_key]

        # 情感标签：仅对 task1-SR 生效
        if task_name == "task1-SR":
            label = sentiment_map.get(sentence)
            if label is not None and "sentiment_label" not in meta:
                meta["sentiment_label"] = int(label)

        # 关系标签：task3-TSR 使用 task3 标签；task2-NR 使用 task2 标签
        if task_name == "task3-TSR":
            rel = rel_task3_map.get(sentence)
            if rel and "relation_label" not in meta:
                meta["relation_label"] = rel
        elif task_name == "task2-NR":
            rel = rel_task2_map.get(sentence)
            if rel and "relation_label" not in meta:
                meta["relation_label"] = rel

        rec["meta"] = meta


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logger = get_logger("build_unified_dataset")
    logger.info("Building unified dataset with args: %s", vars(args))

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_samples: List[Dict[str, Any]] = []

    for task_name in task_names:
        if task_name == "task2-NR-2.0":
            dataset = load_dataset_from_mat_v2(args.zuco_root, logger)
        else:
            dataset = load_dataset_from_mat_v1(args.zuco_root, task_name, logger)

        if not dataset:
            logger.warning("No samples found for task %s, skip.", task_name)
            continue

        task_samples = build_samples_for_task(
            dataset=dataset,
            task_name=task_name,
            max_len=args.max_len,
            dim=args.dim,
            eeg_type=args.eeg_type,
            bands=DEFAULT_BANDS,
            logger=logger,
        )
        all_samples.extend(task_samples)

    logger.info("Total unified samples collected before label enrichment: %d", len(all_samples))

    _enrich_samples_with_metadata_and_labels(all_samples, logger)

    tmp_path = args.output + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(all_samples, f)
    os.replace(tmp_path, args.output)
    logger.info("Unified dataset saved to %s", args.output)


if __name__ == "__main__":
    main()
