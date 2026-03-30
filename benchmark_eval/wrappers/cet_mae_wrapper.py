"""CET-MAE 模型的 Wrapper 实现。

CET-MAE 架构：
- EEG encoder: TransformerEncoder (840 -> 840)
- Project: Linear (840 -> 1024)
- Multi-stream encoder: Multi_Stream_TransformerEncoder (1024)
- BART decoder: 使用 BART 进行文本生成

输入格式：
- 统一格式：(B, L_max, C=840) + mask
输出格式：
- 自回归生成的文本列表
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

import sys
import os
# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger

logger = get_logger("cet_mae_wrapper")


class CETMAEWrapper(BenchmarkModelWrapper):
    """CET-MAE 模型的 Wrapper。
    
    将统一的 (B, L_max, C=840) EEG 序列输入 CET-MAE 模型，
    使用 BART 进行自回归文本生成。
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
        
        try:
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
                # 根据实际 checkpoint 格式调整
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info("Model weights loaded from %s", model_checkpoint)
            else:
                logger.warning("Checkpoint file not found: %s, using randomly initialized weights", 
                             model_checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 初始化 tokenizer（用于解码）
            self.tokenizer = BartTokenizer.from_pretrained(pretrain_path)
            
            logger.info("CET-MAE model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load CET-MAE model: %s", e)
            raise

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> Any:
        """将统一格式的 EEG (B, L_max, C=840) 编码成模型内部表示。
        
        Args:
            eeg: (B, L_max, 840) EEG 序列
            mask: (B, L_max) 1 表示有效，0 表示 padding
            meta: 元信息列表
            batch: 完整 batch，CET-MAE 需要使用 eeg_normalized_2d 和 mask_with_sent
            
        Returns:
            包含编码后 EEG 表示的字典
        """
        # CET-MAE 使用 eeg_normalized_2d（词级+句级 2D 归一化）
        # 如果 batch 中有 eeg_normalized_2d，使用它；否则使用默认 eeg
        if batch is not None and "eeg_normalized_2d" in batch:
            input_eeg = batch["eeg_normalized_2d"].to(self.device)
            # 使用包含句级的 mask
            if "mask_with_sent" in batch:
                input_mask = batch["mask_with_sent"].to(self.device)
            else:
                input_mask = mask.to(self.device)
        else:
            # 回退到默认 eeg
            input_eeg = eeg.to(self.device)
            input_mask = mask.to(self.device)
        
        input_mask_invert = (1 - input_mask).bool()  # 反转 mask
        
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
        """从 EEG 生成文本（自回归，禁用 teacher forcing）。
        
        Args:
            eeg: (B, L_max, 840) EEG 序列
            mask: (B, L_max) 1 表示有效，0 表示 padding
            meta: 元信息列表
            batch: 完整 batch（包含多种 EEG 格式）
            
        Returns:
            生成的文本列表，长度为 batch_size
        """
        with torch.no_grad():
            encoded = self.encode_eeg(eeg, mask, meta, batch)
            
            input_eeg = encoded["eeg"]
            input_mask_invert = encoded["eeg_mask_invert"]
            
            # CET-MAE 的生成流程：
            # 1. 通过 EEG encoder + multi-stream encoder 编码 EEG
            # 2. 使用 BART decoder 生成文本
            
            # Step 1: 编码 EEG
            # 添加位置编码
            eeg_with_pos = input_eeg + self.model.pos_embed_e(input_eeg)
            
            # 通过 EEG branch encoder
            eeg_embeddings = self.model.e_branch(
                eeg_with_pos,
                src_key_padding_mask=input_mask_invert
            )  # (B, L_max, 840)
            
            # 投影到 1024 维
            eeg_embeddings = self.model.act(self.model.fc_eeg(eeg_embeddings))  # (B, L_max, 1024)
            
            # 通过 unified branch（仅使用 EEG 模态）
            eeg_embeddings = self.model.unify_branch(
                eeg_embeddings,
                src_key_padding_mask=input_mask_invert,
                modality='e'
            )  # (B, L_max, 1024)
            
            # Step 2: 使用 BART decoder 生成文本
            # 准备 encoder outputs（BART 期望的格式）
            from transformers.modeling_outputs import BaseModelOutput
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=eeg_embeddings,
            )
            
            # 准备 attention mask（BART 格式：1 表示有效，0 表示 padding）
            bart_attention_mask = encoded["eeg_mask"]
            
            # 使用 BART 的 generate 方法进行自回归生成
            output_ids = self.model.t_branch.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=bart_attention_mask,
                max_length=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,  # greedy decoding
            )
            
            # 解码生成的 token IDs
            generated_texts = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_texts
