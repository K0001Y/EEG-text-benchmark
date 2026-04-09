"""CET-MAE 模型的 Wrapper 实现（带外接 BART 解码器）。

v2 更新：
- D-5：优先读取 batch["eeg_word_norm2d"] / batch["mask_word_with_sent"]，
       向后兼容旧字段名
- M-4：入口处添加输入验证，给出明确错误信息

数据流:
1. 从 batch 获取 eeg_word_norm2d（词+句全局 2D z-score 归一化）
2. CET-MAE encoder 编码 EEG 特征 → 投影 → Multi-stream branch
3. 外接 BARTForConditionalGeneration decoder 生成文本

注意：CET-MAE 原始论文未开源文本解码器，本实现使用预训练 BART decoder
作为代理解码器进行公平对比。该方案未进行 encoder-decoder 联合训练，
建议同时参考检索指标（R@1/R@5/R@10）评估编码器能力。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger
from constants import MAX_LEN

logger = get_logger("cet_mae_wrapper")


class CETMAEWrapper(BenchmarkModelWrapper):
    """CET-MAE 模型的 Wrapper。

    EEG 输入：eeg_word_norm2d，词+句全局 2D z-score 归一化，shape (B, MAX_LEN, 840)
    Mask：mask_word_with_sent，含句级 token 的 mask，shape (B, MAX_LEN)
    """

    def __init__(
        self,
        model_checkpoint: str,
        pretrain_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = MAX_LEN,
        num_beams: int = 1,
        **kwargs,
    ):
        """初始化 CET-MAE Wrapper。

        Args:
            model_checkpoint: CET-MAE 模型权重路径
            pretrain_path:    BART 预训练模型路径（默认自动查找本地缓存）
            device:           运行设备
            max_new_tokens:   生成的最大 token 数（greedy decoding）
            num_beams:        beam search 大小
        """
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        benchmark_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cet_mae_path = os.path.join(benchmark_root, "models", "CET-MAE")
        if cet_mae_path not in sys.path:
            sys.path.insert(0, cet_mae_path)

        from model_mae_bart import CETMAE_project_late_bart
        from transformers import BartTokenizer

        if pretrain_path is None:
            local_path = os.path.join(benchmark_root, "models", "huggingface", "bart-large")
            if os.path.isdir(local_path):
                pretrain_path = local_path
            else:
                pretrain_path = "facebook/bart-large"
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info("Loading CET-MAE model from %s", model_checkpoint)
        logger.info("Using BART pretrain path: %s", pretrain_path)

        self.model = CETMAE_project_late_bart(
            embed_dim=1024,
            eeg_dim=840,
            multi_heads=8,
            feedforward_dim=2048,
            trans_layers=6,
            decoder_embed_dim=840,
            pretrain_path=pretrain_path,
            device=device,
        )

        if os.path.isfile(model_checkpoint):
            checkpoint = torch.load(model_checkpoint, map_location=device)
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Model weights loaded from %s", model_checkpoint)
        else:
            logger.warning("Checkpoint not found: %s, using random weights", model_checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BartTokenizer.from_pretrained(pretrain_path)

        logger.info("Loading external BART decoder for generation...")
        from transformers import BartForConditionalGeneration
        self.text_decoder = BartForConditionalGeneration.from_pretrained(
            pretrain_path, local_files_only=True
        )
        self.text_decoder.to(self.device)
        self.text_decoder.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("CET-MAE with external BART decoder loaded successfully")
        logger.info("Note: Using proxy decoder (not jointly trained with CET-MAE encoder)")

    def _validate_batch(self, batch: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """M-4：验证 batch 包含必需字段，返回目标 EEG 和 mask。"""
        if batch is None:
            raise ValueError(
                "CETMAEWrapper requires a batch dict. "
                "Expected keys: 'eeg_word_norm2d' and 'mask_word_with_sent'."
            )
        # D-5：优先新字段名，向后兼容旧字段名
        eeg_data = batch.get("eeg_word_norm2d", batch.get("eeg_normalized_2d"))
        mask_data = batch.get("mask_word_with_sent", batch.get("mask_with_sent"))

        if eeg_data is None:
            raise ValueError(
                "CETMAEWrapper: batch must contain 'eeg_word_norm2d' "
                "(global 2D z-score normalized EEG, shape B x %d x 840). "
                "Got keys: %s" % (MAX_LEN, list(batch.keys()))
            )
        if mask_data is None:
            raise ValueError(
                "CETMAEWrapper: batch must contain 'mask_word_with_sent' "
                "(mask including sentence token, shape B x %d). "
                "Got keys: %s" % (MAX_LEN, list(batch.keys()))
            )
        return eeg_data, mask_data

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """编码 EEG（直接使用 eeg_word_norm2d）。"""
        input_eeg, input_mask = self._validate_batch(batch)
        input_eeg = input_eeg.to(self.device)
        input_mask = input_mask.to(self.device)
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
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """从 EEG 生成文本（自回归，greedy decoding）。

        Args:
            eeg:   不使用（接口兼容）
            mask:  不使用（接口兼容）
            meta:  元信息
            batch: 必须包含 "eeg_word_norm2d" 和 "mask_word_with_sent"
        """
        with torch.no_grad():
            encoded = self.encode_eeg(eeg, mask, meta, batch)

            input_eeg = encoded["eeg"]
            input_mask_invert = encoded["eeg_mask_invert"]

            # 1. 位置编码
            eeg_with_pos = input_eeg + self.model.pos_embed_e(input_eeg)

            # 2. EEG encoder branch
            eeg_embeddings = self.model.e_branch(
                eeg_with_pos,
                src_key_padding_mask=input_mask_invert,
            )

            # 3. 投影到 1024 维
            eeg_embeddings = self.model.act(self.model.fc_eeg(eeg_embeddings))

            # 4. Unified branch
            eeg_embeddings = self.model.unify_branch(
                eeg_embeddings,
                src_key_padding_mask=input_mask_invert,
                modality="e",
            )

            # 5. 外接 BART decoder 生成
            from transformers.modeling_outputs import BaseModelOutput

            encoder_outputs = BaseModelOutput(last_hidden_state=eeg_embeddings)
            output_ids = self.text_decoder.generate(
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
                clean_up_tokenization_spaces=True,
            )
