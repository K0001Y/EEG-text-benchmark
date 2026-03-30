from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.logging_utils import get_logger


logger = get_logger("model_wrappers")


class BenchmarkModelWrapper(ABC):
    """抽象基类：所有待评估模型都应实现这个接口。

    评估脚本只依赖两个方法：
    - encode_eeg: 可选，用于将 EEG 序列编码成隐变量（方便复杂模型复用）
    - generate_text: 必须，实现从 EEG 生成文本的自回归解码
    """

    @abstractmethod
    def encode_eeg(self, eeg: torch.Tensor, mask: torch.Tensor, meta: List[Dict[str, Any]] | None = None) -> Any:
        """可选的 EEG 编码接口。

        - eeg:  (B, L_max, C)
        - mask: (B, L_max)
        - meta: 每条样本的额外信息列表（task/subject/text_uid 等）
        """

    @abstractmethod
    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        """核心接口：从 EEG 序列生成文本。

        要求：
        - 内部必须使用自回归生成（如 HuggingFace `generate`），禁止 teacher forcing；
        - 返回长度为 batch_size 的字符串列表，每个元素对应一条样本。

        Args:
            eeg: (B, L_max, C) 默认的 EEG 张量
            mask: (B, L_max) 默认的 mask
            meta: 每条样本的元信息列表
            batch: 完整的 batch 字典，包含多个 EEG 格式：
                   - eeg: 默认 EEG（eeg_normalized_1d）
                   - eeg_raw: 原始词级 EEG（未归一化）
                   - eeg_normalized_1d: 逐词 1D 归一化（EEG-To-Text 使用）
                   - eeg_normalized_2d: 词+句 2D 归一化（CET-MAE 使用）
                   - mask_with_sent: 包含句级的 mask
        """


class DummyEchoWrapper(BenchmarkModelWrapper):
    """占位模型：用于快速测试评估流程是否跑通。

    行为：忽略 EEG，直接返回 input_text 或在前面加上前缀。
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def encode_eeg(self, eeg: torch.Tensor, mask: torch.Tensor, meta: List[Dict[str, Any]] | None = None) -> Any:
        # 这个模型完全不用 EEG，直接返回 None 即可
        logger.debug("DummyEchoWrapper.encode_eeg called, but no-op.")
        return None

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        batch_size = eeg.size(0)
        results: List[str] = []
        meta = meta or [{} for _ in range(batch_size)]
        for m in meta:
            # 尝试从 meta 中拿到 input_text 作为 "预测"，否则返回空串
            text = m.get("input_text", "")
            results.append(self.prefix + text)
        return results


def build_model_wrapper(model_name: str, **kwargs: Any) -> BenchmarkModelWrapper:
    """简单工厂：根据名称构建对应的模型 wrapper。

    - "dummy": 使用 DummyEchoWrapper，用于测试整条评估流程
    - "eeg_to_text": EEG-To-Text 模型 (BrainTranslator/T5Translator)
    - "eeg2text": EEG2Text 模型 (raw/spectro 输入)
    - "cet_mae": CET-MAE 模型 (Cross-modal EEG-Text Masked Autoencoder)
    - "glim": GLIM 模型 (Grounded Language-Interfaced Model)
    """
    name = model_name.lower()
    if name == "dummy":
        return DummyEchoWrapper(prefix=kwargs.get("prefix", ""))

    if name == "eeg_to_text":
        from wrappers.eeg_to_text_wrapper import EEGToTextWrapper
        return EEGToTextWrapper(**kwargs)

    if name == "eeg2text":
        from wrappers.eeg2text_wrapper import EEG2TextWrapper
        return EEG2TextWrapper(**kwargs)

    if name == "cet_mae":
        from wrappers.cet_mae_wrapper import CETMAEWrapper
        return CETMAEWrapper(**kwargs)

    if name == "glim":
        from wrappers.glim_wrapper import GLIMWrapper
        return GLIMWrapper(**kwargs)

    raise ValueError(f"Unknown model_name for BenchmarkModelWrapper: {model_name}")
