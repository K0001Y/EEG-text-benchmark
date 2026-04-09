"""模型 Wrapper 基类与工厂（v2）。

A-1：明确接口契约：
  - `batch` 是所有 wrapper 的主要数据来源
  - `eeg` / `mask` 参数保留供 DummyEchoWrapper 等简单模型使用，以及向后兼容
  - 各 wrapper 应优先从 batch 中取所需格式的 EEG 字段

字段命名（v2）：
  batch["eeg_word_norm1d"]   逐词 1D 归一化，EEG-To-Text 使用
  batch["eeg_word_norm2d"]   全局 2D 归一化，CET-MAE 使用
  batch["eeg_spectro"]       spectrogram 格式，EEG2Text 使用
  batch["eeg_word_raw"]      原始词级特征，GLIM 转换基础
  batch["mask_word"]         词级 mask
  batch["mask_word_with_sent"] 含句级 token 的 mask
  batch["mask_spectro"]      spectrogram mask
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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

    接口契约（A-1）：
    - generate_text 的 batch 参数是主要数据来源，各 wrapper 应从中取所需的 EEG 格式
    - eeg / mask 参数保留作为默认 EEG（= eeg_word_norm1d / mask_word）的引用，
      以及为 DummyEchoWrapper 等简单模型提供向后兼容支持
    - wrapper 内部应使用 batch.get("eeg_word_norm1d", eeg) 模式读取数据
    """

    @abstractmethod
    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """可选的 EEG 编码接口。

        Args:
            eeg:  (B, L_max, C) 默认 EEG（= eeg_word_norm1d）
            mask: (B, L_max) 默认 mask（= mask_word）
            meta: 每条样本的额外信息（task/subject/text_uid 等）
        """

    @abstractmethod
    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """核心接口：从 EEG 序列生成文本（自回归，禁止 teacher forcing）。

        Args:
            eeg:   (B, L_max, C) 默认 EEG（= eeg_word_norm1d），向后兼容用
            mask:  (B, L_max) 默认 mask（= mask_word），向后兼容用
            meta:  每条样本的元信息列表
            batch: 完整 batch 字典（主要数据来源，优先使用）：
                   - "eeg" / "mask"        默认字段（向后兼容）
                   - "eeg_word_norm1d"     逐词 1D 归一化（EEG-To-Text）
                   - "eeg_word_norm2d"     全局 2D 归一化（CET-MAE）
                   - "eeg_spectro"         spectrogram（EEG2Text）
                   - "eeg_word_raw"        原始词级特征（GLIM 转换基础）
                   - "mask_word"           词级 mask
                   - "mask_word_with_sent" 含句级 token 的 mask
                   - "mask_spectro"        spectrogram mask
                   （同时保留 v1 旧字段名供向后兼容）

        Returns:
            长度为 batch_size 的文本字符串列表
        """


class DummyEchoWrapper(BenchmarkModelWrapper):
    """占位模型：用于快速测试评估流程是否跑通。

    行为：忽略 EEG，直接返回 input_text 或加前缀。
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        logger.debug("DummyEchoWrapper.encode_eeg called (no-op).")
        return None

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        batch_size = eeg.size(0)
        meta = meta or [{} for _ in range(batch_size)]
        return [self.prefix + m.get("input_text", "") for m in meta]


def build_model_wrapper(model_name: str, **kwargs: Any) -> BenchmarkModelWrapper:
    """工厂：根据名称构建对应的模型 wrapper。

    可用名称：
    - "dummy"        DummyEchoWrapper，用于测试评估流程
    - "eeg_to_text"  EEG-To-Text（BrainTranslator/T5Translator）
    - "eeg2text"     EEG2Text（spectrogram 输入）
    - "cet_mae"      CET-MAE（Cross-modal EEG-Text Masked Autoencoder）
    - "glim"         GLIM（Grounded Language-Interfaced Model）
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

    raise ValueError(f"Unknown model_name for BenchmarkModelWrapper: {model_name!r}")
