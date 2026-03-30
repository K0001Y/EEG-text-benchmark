"""
Wrapper for EEG-To-Text model (BrainTranslator / T5Translator).
适配 EEG-To-Text 模型到统一 benchmark 接口。
"""

import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, GenerationConfig

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger


logger = get_logger("eeg_to_text_wrapper")


class EEGToTextWrapper(BenchmarkModelWrapper):
    """EEG-To-Text 模型的 Wrapper。
    
    输入：统一的 (B, L_max, C) EEG 序列 + mask。
    输出：自回归生成的文本列表。
    """

    def __init__(
        self,
        model_checkpoint: str,
        model_type: str = "bart",  # "bart" or "t5"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        **kwargs
    ):
        """
        Args:
            model_checkpoint: 模型 checkpoint 路径
            model_type: "bart" 或 "t5"
            device: 设备
            max_new_tokens: 最大生成 token 数
            num_beams: beam search 的 beam 数量（1 表示 greedy）
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        # 加载原始模型结构（需要根据实际路径调整）
        logger.info("Loading EEG-To-Text model from %s", model_checkpoint)
        self._load_model(model_checkpoint)
        self.model.eval()

    def _load_model(self, checkpoint_path: str):
        """加载 EEG-To-Text 模型。"""
        # 从 checkpoint 加载模型参数
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # 根据 model_type 初始化相应的模型（使用本地缓存，离线模式）
        if self.model_type == "bart":
            from transformers import BartForConditionalGeneration
            pretrained = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large",
                local_files_only=True
            )
            self.tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large",
                local_files_only=True
            )
            
            # 动态导入 EEG-To-Text 的 BrainTranslator
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
                    additional_encoder_dim_feedforward=2048
                )
            except ImportError as e:
                logger.error("Failed to import BrainTranslator: %s", e)
                raise

        elif self.model_type == "t5":
            from transformers import T5ForConditionalGeneration
            pretrained = T5ForConditionalGeneration.from_pretrained(
                "t5-large",
                local_files_only=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                "t5-large",
                local_files_only=True
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
                    additional_encoder_dim_feedforward=2048
                )
            except ImportError as e:
                logger.error("Failed to import T5Translator: %s", e)
                raise
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # 去掉 DataParallel 的 module. 前缀
        if any(k.startswith("module.") for k in state_dict.keys()):
            logger.info("Detected DataParallel checkpoint, stripping 'module.' prefix")
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        # 加载训练好的权重
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])
        logger.info("Loaded model weights successfully (missing=%d, unexpected=%d)", len(missing), len(unexpected))

        self.model.to(self.device)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None
    ) -> Any:
        """
        将统一格式的 EEG (B, L_max, C) 编码成模型内部表示。
        
        EEG-To-Text 需要：
        - input_embeddings: (B, seq_len, 840)，词级 EEG 序列
        - input_mask: (B, seq_len)，0/1 mask
        - input_mask_invert: (B, seq_len)，反转的 mask（用于 transformer padding）
        
        注意：统一数据集已经提供了 eeg_normalized_1d（逐词 1D 归一化），
        直接使用即可，无需再次归一化。
        """
        # 统一数据集的 eeg 字段现在默认是 eeg_normalized_1d（逐词 1D 归一化）
        # 直接使用，与 EEG-To-Text 训练时的归一化方式一致
        input_embeddings = eeg.to(self.device)
        
        input_mask = mask.to(self.device)  # (B, L_max)
        input_mask_invert = (1 - input_mask).bool()  # 反转 mask

        return {
            "input_embeddings": input_embeddings,
            "input_mask": input_mask,
            "input_mask_invert": input_mask_invert,
        }

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        """
        从 EEG 生成文本（自回归，禁用 teacher forcing）。
        
        Args:
            eeg: 默认 EEG（已经是 eeg_normalized_1d）
            mask: 词级 mask
            meta: 元信息
            batch: 完整 batch（可包含多种 EEG 格式）
        
        Returns:
            生成的文本列表，长度为 batch_size。
        """
        # EEG-To-Text 使用 eeg_normalized_1d（逐词 1D 归一化）
        # 默认 eeg 字段就是 eeg_normalized_1d，无需额外处理
        with torch.no_grad():
            encoded = self.encode_eeg(eeg, mask, meta)
            
            input_embeddings = encoded["input_embeddings"]
            input_mask = encoded["input_mask"]
            input_mask_invert = encoded["input_mask_invert"]

            # 使用与原始 eval_decoding.py 完全一致的生成参数：
            # max_length=56, num_beams=5, do_sample=True, repetition_penalty=5.0, no_repeat_ngram_size=2
            output = self.model.generate(
                input_embeddings_batch=input_embeddings,
                input_masks_batch=input_mask,
                input_masks_invert=input_mask_invert,
                target_ids_batch_converted=None,
                max_length=56,
                num_beams=self.num_beams if self.num_beams > 1 else 5,
                do_sample=True,
                repetition_penalty=5.0,
                no_repeat_ngram_size=2,
            )

            # 处理输出：可能是 GreedySearchDecoderOnlyOutput 对象或 tensor
            if hasattr(output, 'sequences'):
                output_ids = output.sequences
            else:
                output_ids = output

            # 解码生成的 token ids
            generated_texts = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return generated_texts
