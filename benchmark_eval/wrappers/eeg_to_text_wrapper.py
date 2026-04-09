"""
Wrapper for EEG-To-Text model (BrainTranslator / T5Translator).

v2 更新：
- D-5：优先读取 batch["eeg_word_norm1d"] / batch["mask_word"]（向后兼容旧字段名）
- H-1：生成参数遵循原始论文（beam=5, do_sample=True, repetition_penalty=5.0）
- 使用 MAX_LEN 常量替代硬编码的 56
"""

import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger
from constants import MAX_LEN


logger = get_logger("eeg_to_text_wrapper")


class EEGToTextWrapper(BenchmarkModelWrapper):
    """EEG-To-Text 模型的 Wrapper。

    EEG 输入：eeg_word_norm1d，逐词 1D z-score 归一化，shape (B, MAX_LEN, 840)
    Mask：mask_word，shape (B, MAX_LEN)
    生成策略：beam=5, do_sample=True, repetition_penalty=5.0（与原始论文一致）
    """

    def __init__(
        self,
        model_checkpoint: str,
        model_type: str = "bart",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = MAX_LEN,
        num_beams: int = 5,
        do_sample: bool = True,
        repetition_penalty: float = 5.0,
        no_repeat_ngram_size: int = 2,
        **kwargs,
    ):
        """
        Args:
            model_checkpoint:     模型 checkpoint 路径
            model_type:           "bart" 或 "t5"
            device:               设备
            max_new_tokens:       最大生成 token 数（默认 MAX_LEN=56）
            num_beams:            beam search 数量（原始论文 5）
            do_sample:            是否采样（原始论文 True）
            repetition_penalty:   重复惩罚系数（原始论文 5.0）
            no_repeat_ngram_size: 禁止重复 n-gram 大小（原始论文 2）
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

        logger.info("Loading EEG-To-Text model from %s", model_checkpoint)
        self._load_model(model_checkpoint)
        self.model.eval()

    def _load_model(self, checkpoint_path: str) -> None:
        """加载 EEG-To-Text 模型（BrainTranslator 或 T5Translator）。"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        if self.model_type == "bart":
            pretrained = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large", local_files_only=True
            )
            self.tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large", local_files_only=True
            )
            try:
                models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EEG-To-Text-main")
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                from model_decoding import BrainTranslator
                self.model = BrainTranslator(
                    pretrained_layers=pretrained,
                    in_feature=840,
                    decoder_embedding_size=1024,
                    additional_encoder_nhead=8,
                    additional_encoder_dim_feedforward=2048,
                )
            except ImportError as e:
                logger.error("Failed to import BrainTranslator: %s", e)
                raise

        elif self.model_type == "t5":
            pretrained = T5ForConditionalGeneration.from_pretrained(
                "t5-large", local_files_only=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                "t5-large", local_files_only=True
            )
            try:
                models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EEG-To-Text-main")
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                from model_decoding import T5Translator
                self.model = T5Translator(
                    pretrained_layers=pretrained,
                    in_feature=840,
                    decoder_embedding_size=1024,
                    additional_encoder_nhead=8,
                    additional_encoder_dim_feedforward=2048,
                )
            except ImportError as e:
                logger.error("Failed to import T5Translator: %s", e)
                raise
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")

        # 处理 DataParallel 的 module. 前缀
        if any(k.startswith("module.") for k in state_dict.keys()):
            logger.info("Detected DataParallel checkpoint, stripping 'module.' prefix")
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])
        logger.info(
            "Loaded model weights (missing=%d, unexpected=%d)", len(missing), len(unexpected)
        )
        self.model.to(self.device)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """将词级 EEG 编码为模型内部表示。

        EEG-To-Text 使用 eeg_word_norm1d（逐词 1D z-score 归一化），
        统一数据集默认 eeg 字段即为该格式，无需额外归一化。
        """
        input_embeddings = eeg.to(self.device)          # (B, MAX_LEN, 840)
        input_mask = mask.to(self.device)                # (B, MAX_LEN)
        input_mask_invert = (1 - input_mask).bool()

        return {
            "input_embeddings": input_embeddings,
            "input_mask": input_mask,
            "input_mask_invert": input_mask_invert,
        }

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """从 EEG 生成文本（自回归，遵循原始论文生成参数）。

        Args:
            eeg:   默认 EEG（fallback）
            mask:  默认 mask（fallback）
            meta:  元信息
            batch: 优先从 batch["eeg_word_norm1d"] 和 batch["mask_word"] 读取
        """
        # D-5：优先读取 v2 字段名，向后兼容旧字段名
        if batch is not None:
            eeg_input = batch.get("eeg_word_norm1d", batch.get("eeg_normalized_1d", eeg))
            mask_input = batch.get("mask_word", batch.get("mask", mask))
        else:
            eeg_input = eeg
            mask_input = mask

        with torch.no_grad():
            encoded = self.encode_eeg(eeg_input, mask_input, meta)

            # H-1：使用原始论文生成参数
            output = self.model.generate(
                input_embeddings_batch=encoded["input_embeddings"],
                input_masks_batch=encoded["input_mask"],
                input_masks_invert=encoded["input_mask_invert"],
                target_ids_batch_converted=None,
                max_length=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )

            output_ids = output.sequences if hasattr(output, "sequences") else output

            return self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
