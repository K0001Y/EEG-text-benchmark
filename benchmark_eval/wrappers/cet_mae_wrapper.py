"""CET-MAE 模型的 Wrapper 实现（简化版）。

数据流:
1. 从 batch 获取预处理的 eeg_normalized_2d
2. 直接输入模型生成文本
3. 不做任何数据转换

CET-MAE 架构：
- EEG encoder: TransformerEncoder (840 -> 840)
- Project: Linear (840 -> 1024)
- Multi-stream encoder: Multi_Stream_TransformerEncoder (1024)
- BART decoder: 使用 BART 进行文本生成
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

import torch

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger

logger = get_logger("cet_mae_wrapper")


class CETMAEWrapper(BenchmarkModelWrapper):
    """CET-MAE 模型的 Wrapper（简化版）。
    
    直接使用预处理的 eeg_normalized_2d 数据，不做任何数据转换。
    """

    def __init__(
        self,
        model_checkpoint: str,
        pretrain_path: str = "./models/huggingface/bart-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        **kwargs
    ):
        """初始化 CET-MAE Wrapper。
        
        Args:
            model_checkpoint: CET-MAE 模型权重路径
            pretrain_path: BART 预训练模型路径
            device: 运行设备
            max_new_tokens: 生成的最大 token 数
            num_beams: beam search 的 beam 大小
        """
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        # 动态导入 CET-MAE 模型代码
        cet_mae_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "CET-MAE"
        )
        if cet_mae_path not in sys.path:
            sys.path.insert(0, cet_mae_path)
        
        from model_mae_bart import CETMAE_project_late_bart
        from transformers import BartTokenizer
        
        logger.info("Loading CET-MAE model from %s", model_checkpoint)
        
        # 初始化模型
        self.model = CETMAE_project_late_bart(
            embed_dim=1024,
            eeg_dim=840,
            multi_heads=8,
            feedforward_dim=2048,
            trans_layers=6,
            decoder_embed_dim=840,
            pretrain_path=pretrain_path,
            device=device
        )
        
        # 加载模型权重
        if os.path.isfile(model_checkpoint):
            checkpoint = torch.load(model_checkpoint, map_location=device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Model weights loaded from %s", model_checkpoint)
        else:
            logger.warning("Checkpoint not found: %s, using random weights", model_checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化 tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(pretrain_path)
        
        logger.info("CET-MAE model loaded successfully")

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> Any:
        """编码 EEG（简化版，直接使用预处理数据）。"""
        # 直接使用预处理的 CET-MAE 格式
        input_eeg = batch["eeg_normalized_2d"].to(self.device)
        input_mask = batch["mask_with_sent"].to(self.device)
        input_mask_invert = (1 - input_mask).bool()
        
        return {
            "eeg": input_eeg,
            "eeg_mask": input_mask,
            "eeg_mask_invert": input_mask_invert,
        }

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
            batch: 完整 batch，必须包含 eeg_normalized_2d 和 mask_with_sent
            
        Returns:
            生成的文本列表
        """
        if batch is None or "eeg_normalized_2d" not in batch:
            raise ValueError("CETMAEWrapper requires batch with 'eeg_normalized_2d' field")
        
        with torch.no_grad():
            encoded = self.encode_eeg(eeg, mask, meta, batch)
            
            input_eeg = encoded["eeg"]
            input_mask_invert = encoded["eeg_mask_invert"]
            
            # 1. 添加位置编码
            eeg_with_pos = input_eeg + self.model.pos_embed_e(input_eeg)
            
            # 2. EEG encoder
            eeg_embeddings = self.model.e_branch(
                eeg_with_pos,
                src_key_padding_mask=input_mask_invert
            )
            
            # 3. 投影到 1024 维
            eeg_embeddings = self.model.act(self.model.fc_eeg(eeg_embeddings))
            
            # 4. Unified branch
            eeg_embeddings = self.model.unify_branch(
                eeg_embeddings,
                src_key_padding_mask=input_mask_invert,
                modality='e'
            )
            
            # 5. BART 生成
            from transformers.modeling_outputs import BaseModelOutput
            
            encoder_outputs = BaseModelOutput(last_hidden_state=eeg_embeddings)
            
            output_ids = self.model.t_branch.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoded["eeg_mask"],
                max_length=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )
            
            return self.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
