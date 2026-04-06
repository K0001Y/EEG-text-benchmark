from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pickle
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义 collate 函数，正确处理字符串和字典字段。
    
    默认的 collate_fn 会尝试堆叠所有字段，但对于 meta 字典和字符串字段会出错。
    此函数将 tensor 堆叠，字符串和字典保持为列表。
    """
    # 分类处理不同类型的字段
    collated = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor):
            # Tensor 堆叠
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            # 数值转为 Tensor
            collated[key] = torch.tensor(values)
        else:
            # 字符串、字典等保持为列表
            collated[key] = values
    
    return collated


@dataclass
class UnifiedSample:
    """Single unified benchmark sample.

    Expected fields in the underlying dict:
    - eeg:      default EEG array (1D normalized), shape (L_max, C)
    - eeg_raw:  raw word-level EEG (unnormalized), shape (L_max, C)
    - eeg_normalized_1d: per-word 1D normalized EEG (for EEG-To-Text), shape (L_max, C)
    - eeg_normalized_2d: 2D normalized EEG with sentence-level (for CET-MAE), shape (L_max, C)
    - eeg_eeg2text: raw time-series EEG (for EEG2Text), shape (24000, 105)
    - sent_eeg_raw: sentence-level EEG (separate), shape (C,)
    - mask:     word-level mask, shape (L_max,)
    - mask_with_sent: mask including sentence-level EEG
    - mask_eeg2text: mask for EEG2Text format, shape (24000,)
    - input_text:    original sentence shown to the subject
    - reference_text: target text used for metrics
    - phase:    "train" / "val" / "test" (optional if you only use one split)
    - meta:     arbitrary metadata dict (task, subject, text_uid, seq_len, etc.)
    """

    eeg: Any
    mask: Any
    input_text: str
    reference_text: str
    phase: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    # Additional EEG formats
    eeg_raw: Optional[Any] = None
    eeg_normalized_1d: Optional[Any] = None
    eeg_normalized_2d: Optional[Any] = None
    eeg_eeg2text: Optional[Any] = None
    sent_eeg_raw: Optional[Any] = None
    mask_with_sent: Optional[Any] = None
    mask_eeg2text: Optional[Any] = None


class UnifiedDataset(Dataset):
    """Dataset over a unified EEG-text benchmark pickle file.

    The pickle file is expected to contain a list of dict items. Each dict
    should at least contain the keys required by UnifiedSample.
    
    Supports noise mode for control experiments: when noise_mode=True,
    returns random noise instead of real EEG data.
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
            data_path: Path to the unified dataset pickle file
            phase: Filter by phase ("train"/"val"/"test"), None to include all
            noise_mode: If True, return random noise instead of real EEG
            noise_type: Type of noise ("gaussian" or "uniform")
            noise_seed: Random seed for noise generation (for reproducibility)
            noise_mean: Mean for Gaussian noise
            noise_std: Standard deviation for Gaussian noise (or range for uniform)
        """
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Unified dataset file not found: {data_path}")

        with open(data_path, "rb") as f:
            raw_samples: List[Dict[str, Any]] = pickle.load(f)

        self.samples: List[UnifiedSample] = []
        for item in raw_samples:
            sample = UnifiedSample(
                eeg=item["eeg"],
                mask=item["mask"],
                input_text=item.get("input_text", item.get("text", "")),
                reference_text=item.get("reference_text", item.get("target_text", item.get("text", ""))),
                phase=item.get("phase"),
                meta=item.get("meta", {}),
                # Additional EEG formats
                eeg_raw=item.get("eeg_raw"),
                eeg_normalized_1d=item.get("eeg_normalized_1d"),
                eeg_normalized_2d=item.get("eeg_normalized_2d"),
                eeg_eeg2text=item.get("eeg_eeg2text"),
                sent_eeg_raw=item.get("sent_eeg_raw"),
                mask_with_sent=item.get("mask_with_sent"),
                mask_eeg2text=item.get("mask_eeg2text"),
            )
            if phase is None or sample.phase is None or sample.phase == phase:
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples loaded from {data_path} with phase={phase!r}")
        
        # Noise mode configuration
        self.noise_mode = noise_mode
        self.noise_type = noise_type
        self.noise_seed = noise_seed
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.data_path = data_path
        
    def _generate_noise_eeg(self, sample: UnifiedSample, idx: int) -> Dict[str, torch.Tensor]:
        """Generate random noise EEG data with the same shape as real EEG.
        
        Args:
            sample: The original sample containing real EEG data
            idx: Sample index (used for seeding to ensure reproducibility)
            
        Returns:
            Dictionary containing noise EEG tensors for all formats
        """
        # Create a seeded random generator for this sample
        rng = np.random.default_rng(self.noise_seed + idx)
        
        result = {}
        
        # 1. Default format (eeg_normalized_1d): shape (max_len, 840)
        if sample.eeg is not None:
            shape = sample.eeg.shape
            if self.noise_type == "gaussian":
                result["eeg"] = rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:  # uniform
                result["eeg"] = rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)
            # In noise mode, mask is all ones
            result["mask"] = np.ones(shape[0], dtype=np.float32)
        
        # 2. Raw format: shape (max_len, 840)
        if sample.eeg_raw is not None:
            shape = sample.eeg_raw.shape
            if self.noise_type == "gaussian":
                result["eeg_raw"] = rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:
                result["eeg_raw"] = rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)
        
        # 3. 1D normalized format
        if sample.eeg_normalized_1d is not None:
            shape = sample.eeg_normalized_1d.shape
            if self.noise_type == "gaussian":
                result["eeg_normalized_1d"] = rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:
                result["eeg_normalized_1d"] = rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)
        
        # 4. 2D normalized format (CET-MAE): shape (max_len, 840)
        if sample.eeg_normalized_2d is not None:
            shape = sample.eeg_normalized_2d.shape
            if self.noise_type == "gaussian":
                result["eeg_normalized_2d"] = rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:
                result["eeg_normalized_2d"] = rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)
            result["mask_with_sent"] = np.ones(shape[0], dtype=np.float32)
        
        # 5. EEG2Text format: shape (24000, 105)
        if hasattr(sample, 'eeg_eeg2text') and sample.eeg_eeg2text is not None:
            shape = sample.eeg_eeg2text.shape
            if self.noise_type == "gaussian":
                result["eeg_eeg2text"] = rng.normal(self.noise_mean, self.noise_std, shape).astype(np.float32)
            else:
                result["eeg_eeg2text"] = rng.uniform(-self.noise_std, self.noise_std, shape).astype(np.float32)
            result["mask_eeg2text"] = np.ones(shape[0], dtype=np.float32)
        
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        # Check if noise mode is enabled
        if self.noise_mode:
            # Generate noise data with same shape as real EEG
            noise_data = self._generate_noise_eeg(sample, idx)
            
            result = {
                "idx": idx,
                "input_text": sample.input_text,
                "reference_text": sample.reference_text,
                "meta": sample.meta or {},
            }
            
            # Convert noise data to tensors
            for key, value in noise_data.items():
                result[key] = torch.as_tensor(value, dtype=torch.float32)
            
            return result
        
        # Normal mode: return real EEG data
        eeg = torch.as_tensor(sample.eeg, dtype=torch.float32)
        mask = torch.as_tensor(sample.mask, dtype=torch.float32)
        
        result = {
            "idx": idx,
            "eeg": eeg,
            "mask": mask,
            "input_text": sample.input_text,
            "reference_text": sample.reference_text,
            "meta": sample.meta or {},
        }
        
        # Add optional EEG formats if available
        if sample.eeg_raw is not None:
            result["eeg_raw"] = torch.as_tensor(sample.eeg_raw, dtype=torch.float32)
        if sample.eeg_normalized_1d is not None:
            result["eeg_normalized_1d"] = torch.as_tensor(sample.eeg_normalized_1d, dtype=torch.float32)
        if sample.eeg_normalized_2d is not None:
            result["eeg_normalized_2d"] = torch.as_tensor(sample.eeg_normalized_2d, dtype=torch.float32)
        if sample.eeg_eeg2text is not None:
            result["eeg_eeg2text"] = torch.as_tensor(sample.eeg_eeg2text, dtype=torch.float32)
        if sample.sent_eeg_raw is not None:
            result["sent_eeg_raw"] = torch.as_tensor(sample.sent_eeg_raw, dtype=torch.float32)
        if sample.mask_with_sent is not None:
            result["mask_with_sent"] = torch.as_tensor(sample.mask_with_sent, dtype=torch.float32)
        if sample.mask_eeg2text is not None:
            result["mask_eeg2text"] = torch.as_tensor(sample.mask_eeg2text, dtype=torch.float32)
        
        return result
