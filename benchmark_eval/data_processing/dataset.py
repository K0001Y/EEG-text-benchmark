from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pickle
import os

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
    - eeg:      list or numpy array with shape (L_max, C)
    - mask:     list or numpy/bool array with shape (L_max,)
    - input_text:    original sentence shown to the subject
    - reference_text: target text used for metrics
    - phase:    "train" / "val" / "test" (optional if you only use one split)
    - meta:     arbitrary metadata dict (task, subject, text_uid, etc.)
    """

    eeg: Any
    mask: Any
    input_text: str
    reference_text: str
    phase: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class UnifiedDataset(Dataset):
    """Dataset over a unified EEG-text benchmark pickle file.

    The pickle file is expected to contain a list of dict items. Each dict
    should at least contain the keys required by UnifiedSample.
    """

    def __init__(self, data_path: str, phase: Optional[str] = None):
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
            )
            if phase is None or sample.phase is None or sample.phase == phase:
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples loaded from {data_path} with phase={phase!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        eeg = torch.as_tensor(sample.eeg, dtype=torch.float32)
        mask = torch.as_tensor(sample.mask, dtype=torch.float32)
        # input_text + reference_text 保留为字符串，meta 保留为原始字典
        return {
            "idx": idx,
            "eeg": eeg,
            "mask": mask,
            "input_text": sample.input_text,
            "reference_text": sample.reference_text,
            "meta": sample.meta or {},
        }
