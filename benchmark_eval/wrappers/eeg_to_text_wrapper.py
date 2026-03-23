"""
Wrapper for EEG-To-Text model (BrainTranslator / T5Translator).
适配 EEG-To-Text 模型到统一 benchmark 接口。
"""

import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

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

        # 根据 model_type 初始化相应的模型
        if self.model_type == "bart":
            from transformers import BartForConditionalGeneration
            pretrained = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
            
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
            pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            
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

        # 加载训练好的权重
        try:
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded model weights successfully")
        except Exception as e:
            logger.warning("Failed to load full state dict, trying partial load: %s", e)
            # 部分加载
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)

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
        """
        # 统一输入已经是 (B, L_max, C=840)
        # EEG-To-Text 直接使用这个作为 input_embeddings
        input_embeddings = eeg.to(self.device)  # (B, L_max, 840)
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
        meta: List[Dict[str, Any]] | None = None
    ) -> List[str]:
        """
        从 EEG 生成文本（自回归，禁用 teacher forcing）。
        
        Returns:
            生成的文本列表，长度为 batch_size。
        """
        with torch.no_grad():
            encoded = self.encode_eeg(eeg, mask, meta)
            
            input_embeddings = encoded["input_embeddings"]
            input_mask = encoded["input_mask"]
            input_mask_invert = encoded["input_mask_invert"]

            # 调用模型的 generate 方法（纯自回归）
            output_ids = self.model.generate(
                input_embeddings_batch=input_embeddings,
                input_masks_batch=input_mask,
                input_masks_invert=input_mask_invert,
                target_ids_batch_converted=None,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,  # greedy decoding
            )

            # 解码生成的 token ids
            generated_texts = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return generated_texts
