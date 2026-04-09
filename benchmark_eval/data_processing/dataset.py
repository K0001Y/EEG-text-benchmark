"""统一 EEG-to-Text 数据集加载模块（v2）。

字段命名规范 v2：
  eeg_word_raw         词级原始 EEG，未归一化，(MAX_LEN, 840)
  eeg_word_norm1d      逐词 1D z-score 归一化，(MAX_LEN, 840)
  eeg_word_norm2d      词+句全局 2D z-score 归一化，(MAX_LEN, 840)
  sent_eeg_raw         句级 EEG，单独存储，(840,)
  eeg_spectro          EEG2Text spectrogram，(SPECTRO_STEPS, SPECTRO_FREQS)
  mask_word            词级 mask，(MAX_LEN,)
  mask_word_with_sent  含句级 token 的 mask，(MAX_LEN,)
  mask_spectro         spectrogram mask，(SPECTRO_STEPS,)
  eeg / mask           v1 向后兼容别名

向后兼容：若 PKL 文件仍使用旧字段名（eeg_normalized_1d 等），
UnifiedDataset 会自动回退到旧字段名读取，保证旧 PKL 可正常加载。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pickle
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import SPECTRO_STEPS, SPECTRO_FREQS


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义 collate 函数，正确处理字符串和字典字段。

    Tensor 堆叠，字符串和字典保持为列表。
    """
    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    return collated


def _get_field(item: Dict[str, Any], new_key: str, *fallback_keys: str) -> Any:
    """优先读取新字段名，回退到旧字段名（向后兼容）。"""
    if new_key in item:
        return item[new_key]
    for key in fallback_keys:
        if key in item:
            return item[key]
    return None


@dataclass
class UnifiedSample:
    """单条统一 benchmark 样本（v2 字段命名）。

    Expected fields in the underlying dict:
      eeg_word_raw         raw word-level EEG, (MAX_LEN, 840)
      eeg_word_norm1d      per-word 1D z-score normalized, (MAX_LEN, 840)
      eeg_word_norm2d      global 2D z-score normalized (with sentence EEG), (MAX_LEN, 840)
      sent_eeg_raw         sentence-level EEG, (840,)
      eeg_spectro          spectrogram for EEG2Text, (SPECTRO_STEPS, SPECTRO_FREQS)
      mask_word            word-level mask (1=valid, 0=padding), (MAX_LEN,)
      mask_word_with_sent  mask including sentence token, (MAX_LEN,)
      mask_spectro         spectrogram mask, (SPECTRO_STEPS,)
      eeg / mask           backward-compat aliases
    """
    # 必须字段
    eeg: Any                             # 默认 EEG（= eeg_word_norm1d，向后兼容）
    mask: Any                            # 默认 mask（= mask_word，向后兼容）
    input_text: str
    reference_text: str
    phase: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    # v2 新字段
    eeg_word_raw: Optional[Any] = None
    eeg_word_norm1d: Optional[Any] = None
    eeg_word_norm2d: Optional[Any] = None
    sent_eeg_raw: Optional[Any] = None
    eeg_spectro: Optional[Any] = None
    mask_word: Optional[Any] = None
    mask_word_with_sent: Optional[Any] = None
    mask_spectro: Optional[Any] = None

    # v1 字段（向后兼容，不在 __getitem__ 中输出，由 eeg/mask 别名覆盖）
    eeg_raw: Optional[Any] = None
    eeg_normalized_1d: Optional[Any] = None
    eeg_normalized_2d: Optional[Any] = None
    eeg_eeg2text: Optional[Any] = None
    mask_with_sent: Optional[Any] = None
    mask_eeg2text: Optional[Any] = None


class UnifiedDataset(Dataset):
    """Dataset over a unified EEG-text benchmark pickle file.

    支持新（v2）和旧（v1）字段命名，自动回退。
    支持噪声模式（noise_mode=True）用于对照实验。
    """

    def __init__(
        self,
        data_path: str,
        phase: Optional[str] = None,
        noise_mode: bool = False,
        noise_type: str = "gaussian",
        noise_seed: int = 42,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ):
        """
        Args:
            data_path:   统一数据集 pickle 文件路径
            phase:       按 phase 过滤（"train"/"val"/"test"），None 表示全部
            noise_mode:  True 时返回随机噪声替代真实 EEG
            noise_type:  噪声类型（"gaussian" 或 "uniform"）
            noise_seed:  噪声随机种子（保证跨模型对比公平）
            noise_mean:  高斯噪声均值
            noise_std:   高斯噪声标准差（均匀噪声时为半宽度）
        """
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Unified dataset file not found: {data_path}")

        with open(data_path, "rb") as f:
            raw_samples: List[Dict[str, Any]] = pickle.load(f)

        self.samples: List[UnifiedSample] = []
        for item in raw_samples:
            # 以下 _get_field 调用均优先读新字段名，回退到旧字段名
            eeg_default = _get_field(item, "eeg_word_norm1d", "eeg_normalized_1d", "eeg")
            mask_default = _get_field(item, "mask_word", "mask")
            if eeg_default is None or mask_default is None:
                continue

            sample = UnifiedSample(
                eeg=eeg_default,
                mask=mask_default,
                input_text=item.get("input_text", item.get("text", "")),
                reference_text=item.get("reference_text", item.get("target_text", item.get("text", ""))),
                phase=item.get("phase"),
                meta=item.get("meta", {}),
                # v2 字段
                eeg_word_raw=_get_field(item, "eeg_word_raw", "eeg_raw"),
                eeg_word_norm1d=_get_field(item, "eeg_word_norm1d", "eeg_normalized_1d"),
                eeg_word_norm2d=_get_field(item, "eeg_word_norm2d", "eeg_normalized_2d"),
                sent_eeg_raw=item.get("sent_eeg_raw"),
                eeg_spectro=_get_field(item, "eeg_spectro", "eeg_eeg2text"),
                mask_word=_get_field(item, "mask_word", "mask"),
                mask_word_with_sent=_get_field(item, "mask_word_with_sent", "mask_with_sent"),
                mask_spectro=_get_field(item, "mask_spectro", "mask_eeg2text"),
                # v1 兼容字段（保留供旧代码访问）
                eeg_raw=item.get("eeg_raw"),
                eeg_normalized_1d=item.get("eeg_normalized_1d"),
                eeg_normalized_2d=item.get("eeg_normalized_2d"),
                eeg_eeg2text=item.get("eeg_eeg2text"),
                mask_with_sent=item.get("mask_with_sent"),
                mask_eeg2text=item.get("mask_eeg2text"),
            )
            if phase is None or sample.phase is None or sample.phase == phase:
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples loaded from {data_path} with phase={phase!r}")

        self.noise_mode = noise_mode
        self.noise_type = noise_type
        self.noise_seed = noise_seed
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.data_path = data_path

    def _generate_noise_eeg(self, sample: UnifiedSample, idx: int) -> Dict[str, np.ndarray]:
        """为所有 EEG 字段生成固定 seed 的随机噪声，shape 与真实数据一致。"""
        rng = np.random.default_rng(self.noise_seed + idx)

        def _noise(shape):
            if self.noise_type == "gaussian":
                return rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:  # uniform
                return rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)

        result: Dict[str, np.ndarray] = {}

        # eeg（默认，= eeg_word_norm1d）
        if sample.eeg is not None:
            shape = np.asarray(sample.eeg).shape
            result["eeg"] = _noise(shape)
            result["mask"] = np.ones(shape[0], dtype=np.float32)

        # v2 词级字段
        for key in ("eeg_word_raw", "eeg_word_norm1d", "eeg_word_norm2d"):
            arr = getattr(sample, key, None)
            if arr is not None:
                result[key] = _noise(np.asarray(arr).shape)

        if sample.mask_word is not None:
            shape0 = np.asarray(sample.mask_word).shape[0]
            result["mask_word"] = np.ones(shape0, dtype=np.float32)

        if sample.mask_word_with_sent is not None:
            shape0 = np.asarray(sample.mask_word_with_sent).shape[0]
            result["mask_word_with_sent"] = np.ones(shape0, dtype=np.float32)
            result["eeg_word_norm2d"] = _noise(np.asarray(sample.eeg_word_norm2d or sample.eeg).shape)

        # v2 spectrogram 字段（EEG2Text）
        spectro = sample.eeg_spectro
        if spectro is not None:
            result["eeg_spectro"] = _noise(np.asarray(spectro).shape)
            result["mask_spectro"] = np.ones(SPECTRO_STEPS, dtype=np.float32)

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        if self.noise_mode:
            noise_data = self._generate_noise_eeg(sample, idx)
            result: Dict[str, Any] = {
                "idx": idx,
                "input_text": sample.input_text,
                "reference_text": sample.reference_text,
                "meta": sample.meta or {},
            }
            for key, value in noise_data.items():
                result[key] = torch.as_tensor(value, dtype=torch.float32)
            return result

        # 正常模式
        eeg = torch.as_tensor(np.asarray(sample.eeg), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(sample.mask), dtype=torch.float32)

        result = {
            "idx": idx,
            "eeg": eeg,          # 默认 EEG（= eeg_word_norm1d）
            "mask": mask,        # 默认 mask（= mask_word）
            "input_text": sample.input_text,
            "reference_text": sample.reference_text,
            "meta": sample.meta or {},
        }

        # v2 词级字段
        if sample.eeg_word_raw is not None:
            result["eeg_word_raw"] = torch.as_tensor(np.asarray(sample.eeg_word_raw), dtype=torch.float32)
        if sample.eeg_word_norm1d is not None:
            result["eeg_word_norm1d"] = torch.as_tensor(np.asarray(sample.eeg_word_norm1d), dtype=torch.float32)
        if sample.eeg_word_norm2d is not None:
            result["eeg_word_norm2d"] = torch.as_tensor(np.asarray(sample.eeg_word_norm2d), dtype=torch.float32)
        if sample.sent_eeg_raw is not None:
            result["sent_eeg_raw"] = torch.as_tensor(np.asarray(sample.sent_eeg_raw), dtype=torch.float32)
        if sample.mask_word is not None:
            result["mask_word"] = torch.as_tensor(np.asarray(sample.mask_word), dtype=torch.float32)
        if sample.mask_word_with_sent is not None:
            result["mask_word_with_sent"] = torch.as_tensor(np.asarray(sample.mask_word_with_sent), dtype=torch.float32)

        # v2 spectrogram 字段（EEG2Text）
        if sample.eeg_spectro is not None:
            result["eeg_spectro"] = torch.as_tensor(np.asarray(sample.eeg_spectro), dtype=torch.float32)
        if sample.mask_spectro is not None:
            result["mask_spectro"] = torch.as_tensor(np.asarray(sample.mask_spectro), dtype=torch.float32)

        # v1 字段（向后兼容，同时输出旧名，方便未更新的 wrapper 访问）
        if sample.eeg_normalized_2d is not None:
            result["eeg_normalized_2d"] = torch.as_tensor(np.asarray(sample.eeg_normalized_2d), dtype=torch.float32)
        if sample.mask_with_sent is not None:
            result["mask_with_sent"] = torch.as_tensor(np.asarray(sample.mask_with_sent), dtype=torch.float32)
        if sample.eeg_eeg2text is not None:
            result["eeg_eeg2text"] = torch.as_tensor(np.asarray(sample.eeg_eeg2text), dtype=torch.float32)

        return result
