"""
Wrapper for EEG2Text model (BrainTranslator with raw/spectro input) - 简化版。

数据流:
1. 从 batch 获取预处理的 eeg_eeg2text（原始时序数据）
2. 直接输入模型生成文本
3. 不再从 spectro pickle 加载
"""

import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger


logger = get_logger("eeg2text_wrapper")


class EEG2TextWrapper(BenchmarkModelWrapper):
    """EEG2Text 模型的 Wrapper（简化版）。

    直接使用统一数据中的 eeg_eeg2text 字段（原始时序数据），
    不再从 spectro pickle 加载，简化数据流。
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        pretrain_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        **kwargs,
    ):
        """
        Args:
            model_checkpoint: 微调后的模型 checkpoint 路径
            pretrain_checkpoint: 预训练模型 checkpoint 路径（可选）
            device: 设备
            max_new_tokens: 最大生成 token 数
            num_beams: beam search 的 beam 数量（1 表示 greedy）
        """
        super().__init__()
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        logger.info("Loading EEG2Text model from %s", model_checkpoint)
        self._load_model(model_checkpoint, pretrain_checkpoint)
        self.model.eval()
        self.text_decoder.eval()

    def _load_model(self, checkpoint_path: Optional[str], pretrain_checkpoint: Optional[str]):
        """加载 EEG2Text 模型（BrainTranslator + BART decoder）。"""
        # 动态导入 EEG2Text 的模型
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EEG2Text-main")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from model_decoding_pretrain import BrainTranslator

        # 初始化预训练的 encoder
        self.model = BrainTranslator(
            in_feature=40,
            decoder_embedding_size=1024,
            additional_encoder_nhead=5,
            additional_encoder_dim_feedforward=2048
        )

        # 加载预训练权重（如果提供）
        if pretrain_checkpoint and os.path.isfile(pretrain_checkpoint):
            logger.info("Loading pretrained encoder from %s", pretrain_checkpoint)
            pretrain_state = torch.load(pretrain_checkpoint, map_location=self.device)
            self.model.load_state_dict(pretrain_state, strict=False)

        # 初始化 BART decoder
        self.text_decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

        # 加载微调后的模型权重
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            try:
                # 如果 checkpoint 包含两部分（encoder + decoder）
                if "encoder" in state_dict and "decoder" in state_dict:
                    self.model.load_state_dict(state_dict["encoder"], strict=False)
                    self.text_decoder.load_state_dict(state_dict["decoder"], strict=False)
                else:
                    # 否则尝试直接加载
                    self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded model weights successfully")
            except Exception as e:
                logger.warning("Failed to load model weights: %s", e)

        self.model.to(self.device)
        self.text_decoder.to(self.device)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """编码 EEG（简化版，直接使用预处理数据）。

        Args:
            eeg: 不使用，仅作接口兼容
            mask: 不使用，仅作接口兼容
            meta: 元信息列表
            batch: 完整 batch，必须包含 eeg_eeg2text 字段

        Returns:
            encoded_embedding: (B, seq_len, 1024)
        """
        if batch is None or "eeg_eeg2text" not in batch:
            raise ValueError("EEG2TextWrapper requires batch with 'eeg_eeg2text' field")
        
        eeg_raw = batch["eeg_eeg2text"].to(self.device)  # (B, 24000, 105)

        with torch.no_grad():
            encoded_embedding = self.model(eeg_raw)  # (B, seq_len, 1024)

        return encoded_embedding

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        """从 EEG 生成文本（自回归）。
        
        Args:
            eeg: 不使用，仅作接口兼容
            mask: 不使用，仅作接口兼容
            meta: 元信息列表
            batch: 完整 batch，必须包含 eeg_eeg2text 字段
        
        Returns:
            生成的文本列表
        """
        if batch is None or "eeg_eeg2text" not in batch:
            raise ValueError("EEG2TextWrapper requires batch with 'eeg_eeg2text' field")
        
        with torch.no_grad():
            # 编码 EEG（直接使用预处理数据）
            encoded_embedding = self.encode_eeg(eeg, mask, meta, batch)
            
            # 使用 BART decoder 自回归生成
            output_ids = self.text_decoder.generate(
                inputs_embeds=encoded_embedding,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )

            # 解码生成的 token ids
            return self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
