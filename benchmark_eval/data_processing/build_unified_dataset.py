import argparse
import os
import pickle
import random
from typing import Any, Dict, List
import csv
import sys

import numpy as np
import torch
from tqdm import tqdm
import scipy.io as io
import scipy.signal
import h5py
from glob import glob

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.logging_utils import setup_logging, get_logger
from constants import (
    MAX_LEN,
    EEG_CHANNELS,
    EEG_BANDS,
    EEG_WORD_DIM,
    RAW_SAMPLING_RATE,
    SPECTRO_NPERSEG,
    SPECTRO_NOVERLAP,
    SPECTRO_STEPS,
    SPECTRO_FREQS,
    DEFAULT_SEED,
)


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


def build_spectrogram(
    raw_data: np.ndarray,
    fs: int = RAW_SAMPLING_RATE,
    nperseg: int = SPECTRO_NPERSEG,
    noverlap: int = SPECTRO_NOVERLAP,
    target_steps: int = SPECTRO_STEPS,
    target_freqs: int = SPECTRO_FREQS,
) -> np.ndarray | None:
    """将原始 EEG 时序数据转换为 spectrogram 格式。

    参数与 EEG2Text 原始 data_spectro.py 完全对齐：
      fs=500, nperseg=128, noverlap=64 → 每通道输出 (freqs, steps) ≈ (65, 374)

    Args:
        raw_data: 原始 EEG 数据，shape 为 (105, T) 或 (T, 105)
        fs: 采样率（Hz）
        nperseg: 窗口大小
        noverlap: 重叠大小
        target_steps: 目标时间步数（不足则 padding，超出则截断）
        target_freqs: 目标频率维度

    Returns:
        spectrogram 数组，shape (target_steps, target_freqs)；失败时返回 None。
    """
    if raw_data is None:
        return None

    raw_data = np.asarray(raw_data, dtype=np.float32)

    # 统一 shape 为 (channels, T)
    if raw_data.ndim == 2 and raw_data.shape[0] != EEG_CHANNELS and raw_data.shape[1] == EEG_CHANNELS:
        raw_data = raw_data.T  # (T, 105) -> (105, T)
    if raw_data.ndim != 2 or raw_data.shape[0] != EEG_CHANNELS:
        return None

    channel_spectros: List[np.ndarray] = []
    for ch in range(raw_data.shape[0]):
        signal_1d = raw_data[ch]
        try:
            _, _, Sxx = scipy.signal.spectrogram(
                signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap
            )
        except Exception:
            return None
        # Sxx shape: (freqs, steps) → 取目标 freqs 维度
        freqs_actual = Sxx.shape[0]
        steps_actual = Sxx.shape[1]

        # 截断或 padding 频率维度
        if freqs_actual >= target_freqs:
            Sxx = Sxx[:target_freqs, :]
        else:
            pad = np.zeros((target_freqs - freqs_actual, steps_actual), dtype=np.float32)
            Sxx = np.concatenate([Sxx, pad], axis=0)

        channel_spectros.append(Sxx)  # (target_freqs, steps_actual)

    # 沿通道平均：(target_freqs, steps_actual)
    avg_spectro = np.mean(np.stack(channel_spectros, axis=0), axis=0)

    # 截断或 padding 时间步维度
    steps_actual = avg_spectro.shape[1]
    if steps_actual >= target_steps:
        avg_spectro = avg_spectro[:, :target_steps]
    else:
        pad = np.zeros((target_freqs, target_steps - steps_actual), dtype=np.float32)
        avg_spectro = np.concatenate([avg_spectro, pad], axis=1)

    # 转置为 (steps, freqs) 格式，与 EEG2Text 期望的输入一致
    result = avg_spectro.T  # (target_steps, target_freqs)

    # 1D z-score 归一化（逐样本）
    mean = result.mean()
    std = result.std()
    if std > 0:
        result = (result - mean) / std

    return result.astype(np.float32)


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
        except (KeyError, IndexError, TypeError):
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
    except (KeyError, TypeError):
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
    max_len: int = MAX_LEN,
    dim: int = EEG_CHANNELS,
    eeg_type: str = "GD",
    bands: List[str] | None = None,
    logger=None,
) -> List[Dict[str, Any]]:
    """从 ZuCo 原始 MAT（经统一解析为 dataset_dict）构造统一格式样本。

    参数中的 dataset 结构应为:
    - key: subject 名称
    - value: 该 subject 的句子列表，每个元素是 sent_obj 或 None

    字段命名规范（v2）：
    - eeg_word_raw:       词级原始 EEG，未归一化，shape (max_len, 840)
    - eeg_word_norm1d:    逐词 1D z-score 归一化，shape (max_len, 840)
    - eeg_word_norm2d:    词+句全局 2D z-score 归一化，shape (max_len, 840)
    - sent_eeg_raw:       句级 EEG，单独存储，shape (840,)
    - eeg_spectro:        EEG2Text spectrogram 格式，shape (SPECTRO_STEPS, SPECTRO_FREQS)
    - mask_word:          词级 mask（1=有效, 0=padding），shape (max_len,)
    - mask_word_with_sent:含句级 token 的 mask，shape (max_len,)
    - mask_spectro:       spectrogram mask，shape (SPECTRO_STEPS,)
    - eeg:                eeg_word_norm1d 的别名（向后兼容）
    - mask:               mask_word 的别名（向后兼容）
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

            # 收集词级 EEG 原始向量 + nFixations（同步收集，保证索引对齐）
            word_embeddings_raw: List[torch.Tensor] = []
            word_nfixations: List[float] = []
            for word in sent_obj["word"]:
                t = get_word_embedding_eeg_tensor(word, eeg_type=eeg_type, bands=bands, dim=dim)
                if t is None:
                    word_embeddings_raw = []
                    word_nfixations = []
                    break
                word_embeddings_raw.append(t["raw"])
                word_nfixations.append(float(word.get("nFixations", 0.0)))

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
                word_nfixations = word_nfixations[:max_len]
                num_words = max_len

            # ===== 1. eeg_word_raw：原始词级 EEG，padding 到 max_len =====
            raw_padded = word_embeddings_raw.copy()
            while len(raw_padded) < max_len:
                raw_padded.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_word_raw = torch.stack(raw_padded).numpy().astype("float32")  # (max_len, 840)

            # ===== 2. eeg_word_norm1d：逐词 1D 归一化（EEG-To-Text 使用）=====
            norm_1d_list = [normalize_1d(vec) for vec in word_embeddings_raw]
            while len(norm_1d_list) < max_len:
                norm_1d_list.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_word_norm1d = torch.stack(norm_1d_list)
            if torch.isnan(eeg_word_norm1d).any():
                continue
            eeg_word_norm1d = eeg_word_norm1d.numpy().astype("float32")  # (max_len, 840)

            # ===== 3. eeg_word_norm2d：词级+句级 2D 归一化（CET-MAE 等使用）=====
            all_raw_with_sent = word_embeddings_raw + [sent_eeg["raw"]]
            raw_matrix_with_sent = torch.stack(all_raw_with_sent)  # (num_words+1, 840)
            norm_2d_matrix = normalize_2d(raw_matrix_with_sent)
            if torch.isnan(norm_2d_matrix).any():
                continue
            seq_len_with_sent = len(all_raw_with_sent)
            norm_2d_list = list(torch.unbind(norm_2d_matrix, dim=0))
            if seq_len_with_sent > max_len:
                norm_2d_list = norm_2d_list[:max_len]
                seq_len_with_sent = max_len
            while len(norm_2d_list) < max_len:
                norm_2d_list.append(torch.zeros(feature_dim, dtype=torch.float32))
            eeg_word_norm2d = torch.stack(norm_2d_list).numpy().astype("float32")  # (max_len, 840)

            # ===== 4. sent_eeg_raw：句级 EEG 单独存储 =====
            sent_eeg_raw_arr = sent_eeg["raw"].numpy().astype("float32")  # (840,)

            # ===== 4.5. nfixations_word：词级注视次数 =====
            nfixations_padded = word_nfixations[:max_len] + [0.0] * (max_len - len(word_nfixations[:max_len]))
            nfixations_word_arr = np.array(nfixations_padded, dtype=np.float32)  # (max_len,)

            # ===== 5. mask_word：基于词数，不包含句级 EEG =====
            mask_word = [1.0] * num_words + [0.0] * (max_len - num_words)

            # ===== 6. mask_word_with_sent：包含句级 EEG 的 mask =====
            eff_len_with_sent = min(seq_len_with_sent, max_len)
            mask_word_with_sent = [1.0] * eff_len_with_sent + [0.0] * (max_len - eff_len_with_sent)

            record: Dict[str, Any] = {
                # v2 字段命名
                "eeg_word_raw": eeg_word_raw,           # 原始词级，未归一化
                "eeg_word_norm1d": eeg_word_norm1d,     # 逐词 1D 归一化（EEG-To-Text）
                "eeg_word_norm2d": eeg_word_norm2d,     # 词+句 2D 归一化（CET-MAE）
                "sent_eeg_raw": sent_eeg_raw_arr,       # 句级 EEG，单独存储
                "nfixations_word": nfixations_word_arr, # 词级注视次数（诊断实验 A1b）
                "mask_word": mask_word,                 # 词级 mask
                "mask_word_with_sent": mask_word_with_sent,  # 含句级的 mask
                # 向后兼容别名
                "eeg": eeg_word_norm1d,                 # 与旧字段 eeg 兼容
                "mask": mask_word,                      # 与旧字段 mask 兼容
                # 文本
                "input_text": sent_obj.get("content", ""),
                "reference_text": sent_obj.get("content", ""),
                "phase": None,  # 稍后填入 train/val/test
                "meta": {
                    "task": task_name,
                    "subject": subject,
                    "sentence_index": int(sent_idx),
                    "source": "ZuCo-MAT",
                    "seq_len": num_words,
                    "seq_len_with_sent": eff_len_with_sent,
                },
            }

            # ===== 7. eeg_spectro：spectrogram 格式（EEG2Text 使用）=====
            raw_data = None
            if "rawData" in sent_obj:
                raw_data = sent_obj["rawData"]
            elif "sentence_level_EEG" in sent_obj and "rawData" in sent_obj.get("sentence_level_EEG", {}):
                raw_data = sent_obj["sentence_level_EEG"]["rawData"]

            if raw_data is not None:
                spectro = build_spectrogram(raw_data)
                if spectro is not None:
                    record["eeg_spectro"] = spectro          # (SPECTRO_STEPS, SPECTRO_FREQS)
                    # mask_spectro：全 1（spectrogram 无 padding 概念，长度固定）
                    record["mask_spectro"] = np.ones(SPECTRO_STEPS, dtype=np.float32)
                else:
                    logger.warning(
                        "build_spectrogram returned None for %s sent %d", subject, sent_idx
                    )

            subject_records.append(record)

        samples_by_subject[subject] = subject_records
        logger.info("Task %s, subject %s: %d valid sentences", task_name, subject, len(subject_records))

    # 基于 text_uid 划分 train/val/test，避免数据泄露
    unified: List[Dict[str, Any]] = []

    text_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for subject, records in samples_by_subject.items():
        for rec in records:
            text = rec.get("input_text", "")
            if text not in text_to_records:
                text_to_records[text] = []
            text_to_records[text].append(rec)

    # 固定种子（random + numpy）保证划分可复现
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    unique_texts = sorted(text_to_records.keys())
    n_total = len(unique_texts)

    # H-6：检查 total 是否足够
    if n_total < 10:
        raise ValueError(
            f"Task {task_name}: too few unique texts ({n_total}) to split into train/val/test. "
            "Need at least 10."
        )

    n_train = max(int(n_total * 0.8), 1)
    n_val = max(int(n_total * 0.1), 0)
    if n_train + n_val >= n_total:
        n_train = max(n_total - 2, 1)
        n_val = max(min(n_total - n_train - 1, 1), 0)

    train_texts = set(unique_texts[:n_train])
    val_texts = set(unique_texts[n_train:n_train + n_val])
    test_texts = set(unique_texts[n_train + n_val:])

    # H-6：验证三个 phase 都非空
    if not train_texts:
        raise ValueError(f"Task {task_name}: train set is empty after split.")
    if not val_texts:
        raise ValueError(f"Task {task_name}: val set is empty after split.")
    if not test_texts:
        raise ValueError(f"Task {task_name}: test set is empty after split.")

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
    parser.add_argument("--zuco-root", type=str, default="models/EEG-To-Text-main/dataset/ZuCo",
                        help="Path to root directory that contains ZuCo task folders with Matlab_files")
    parser.add_argument(
        "--tasks",
        type=str,
        default="task1-SR,task2-NR,task3-TSR,task2-NR-2.0",
        help="Comma-separated ZuCo task names to include",
    )
    parser.add_argument("--output", type=str, required=True, help="Output unified dataset pickle path")
    parser.add_argument("--max-len", type=int, default=MAX_LEN,
                        help=f"Max time steps L_max for EEG sequence (default {MAX_LEN}, same as EEG-To-Text)")
    parser.add_argument("--dim", type=int, default=EEG_CHANNELS, help="Number of EEG channels per band")
    parser.add_argument("--eeg-type", type=str, default="GD", help="EEG type key used in word_level_EEG")
    return parser.parse_args()


def load_dataset_from_mat_v1(zuco_root: str, task_name: str, logger=None) -> Dict[str, List[Any]]:
    """从 ZuCo v1 .mat 文件构造与 EEG-To-Text 相同结构的 dataset_dict。"""
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
                # 保留原始时序数据（用于 spectrogram 计算）
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
                        continue

                sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
                sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
                sent_obj["word_tokens_all"] = word_tokens_all

                dataset_dict[subject_name].append(sent_obj)
            else:
                logger.warning(
                    "Missing sentence: subj=%s content=%s, append None",
                    subject_name, getattr(sent, "content", "")
                )
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
                logger.warning(
                    "Missing or empty word-level features: subj=%s content=%s, append None",
                    subject, sent_string
                )
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
                            "GD_t1": gd[0], "GD_t2": gd[1], "GD_a1": gd[2], "GD_a2": gd[3],
                            "GD_b1": gd[4], "GD_b2": gd[5], "GD_g1": gd[6], "GD_g2": gd[7],
                        },
                        "FFD": {
                            "FFD_t1": ffd[0], "FFD_t2": ffd[1], "FFD_a1": ffd[2], "FFD_a2": ffd[3],
                            "FFD_b1": ffd[4], "FFD_b2": ffd[5], "FFD_g1": ffd[6], "FFD_g2": ffd[7],
                        },
                        "TRT": {
                            "TRT_t1": trt[0], "TRT_t2": trt[1], "TRT_a1": trt[2], "TRT_a2": trt[3],
                            "TRT_b1": trt[4], "TRT_b2": trt[5], "TRT_g1": trt[6], "TRT_g2": trt[7],
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
                    sentence, mapping[sentence], label,
                )
            mapping[sentence] = label

    logger.info("Loaded %d sentiment labels from %s", len(mapping), path)
    return mapping


def _load_task2_relation_labels(materials_dir: str, logger) -> Dict[str, str]:
    """加载 ZuCo1 task2 的关系标签。"""
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
                    sentence, mapping[sentence], rel,
                )
            mapping[sentence] = rel

    logger.info("Loaded %d task2 relation labels from %s", len(mapping), path)
    return mapping


def _load_task3_relation_labels(materials_dir: str, logger) -> Dict[str, str]:
    """加载 ZuCo1 task3 的关系标签。"""
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
                    sentence, mapping[sentence], rel,
                )
            mapping[sentence] = rel

    logger.info("Loaded %d task3 relation labels from %s", len(mapping), path)
    return mapping


def _assign_session(task_name: str, sentence_index: int, total_task1_sentences: int) -> str:
    """根据 ZuCo 原始实验设计标注样本的 session 归属。

    ZuCo（Hollenstein et al., 2018）每位被试在两个实验阶段完成所有任务：
      - Session 1：task2-NR（全部）+ task1-SR 前半部分
      - Session 2：task3-TSR（全部）+ task1-SR 后半部分
      - task2-NR-2.0（ZuCo v2）单独记录，暂作 `session_unknown`
    """
    if task_name == "task2-NR":
        return "session_1"
    if task_name == "task3-TSR":
        return "session_2"
    if task_name == "task1-SR":
        if total_task1_sentences <= 0:
            return "session_unknown"
        mid = total_task1_sentences // 2
        return "session_1" if sentence_index < mid else "session_2"
    return "session_unknown"


def _enrich_samples_with_metadata_and_labels(samples: List[Dict[str, Any]], logger) -> None:
    """补全 meta 中的 dataset/text_uid/session 以及情感与关系标签。"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)          # benchmark_eval
    benchmark_root = os.path.dirname(project_root)    # benchmark
    zuco1_materials = os.path.join(benchmark_root, "data", "ZuCo1", "task_materials")

    sentiment_map = _load_task1_sentiment_labels(zuco1_materials, logger)
    rel_task2_map = _load_task2_relation_labels(zuco1_materials, logger)
    rel_task3_map = _load_task3_relation_labels(zuco1_materials, logger)

    # 统计 task1-SR 的总句数（按原始 MAT 的句序），用于划分前/后半 session
    total_task1_sentences = 0
    for rec in samples:
        meta = rec.get("meta") or {}
        if meta.get("task") == "task1-SR":
            idx = int(meta.get("sentence_index", -1))
            if idx + 1 > total_task1_sentences:
                total_task1_sentences = idx + 1
    if total_task1_sentences > 0:
        logger.info(
            "Session split for task1-SR uses N_task1=%d (first half -> session_1, second half -> session_2)",
            total_task1_sentences,
        )
    else:
        logger.warning("No task1-SR samples found; task1-SR session assignment will be 'session_unknown'.")

    text_uid_map: Dict[tuple, int] = {}
    next_uid = 0

    session_counts: Dict[str, int] = {}

    for rec in samples:
        meta = rec.get("meta") or {}
        task_name = meta.get("task", "")

        dataset_name = "ZuCo2" if task_name == "task2-NR-2.0" else "ZuCo1"
        if "dataset" not in meta:
            meta["dataset"] = dataset_name

        sentence = rec.get("input_text", "")

        uid_key = (dataset_name, task_name, sentence)
        if uid_key not in text_uid_map:
            text_uid_map[uid_key] = next_uid
            next_uid += 1
        if "text_uid" not in meta:
            meta["text_uid"] = text_uid_map[uid_key]

        # Session 标注
        if "session" not in meta:
            sent_idx = int(meta.get("sentence_index", -1))
            meta["session"] = _assign_session(task_name, sent_idx, total_task1_sentences)
        session_counts[meta["session"]] = session_counts.get(meta["session"], 0) + 1

        if task_name == "task1-SR":
            label = sentiment_map.get(sentence)
            if label is not None and "sentiment_label" not in meta:
                meta["sentiment_label"] = int(label)

        if task_name == "task3-TSR":
            rel = rel_task3_map.get(sentence)
            if rel and "relation_label" not in meta:
                meta["relation_label"] = rel
        elif task_name == "task2-NR":
            rel = rel_task2_map.get(sentence)
            if rel and "relation_label" not in meta:
                meta["relation_label"] = rel

        rec["meta"] = meta

    logger.info("Session distribution over all samples: %s", dict(sorted(session_counts.items())))


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logger = get_logger("build_unified_dataset")
    logger.info("Building unified dataset with args: %s", vars(args))
    logger.info(
        "Field naming convention v2: eeg_word_raw/norm1d/norm2d, eeg_spectro, mask_word/mask_spectro"
    )

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
