"""
Wrapper for EEG2Text model（ShallowNet + TransformerEncoder + BART decoder）。

v2 更新：
- D-1：从 batch["eeg_spectro"] 读取 spectrogram 格式（374, 65），
       不再使用原始时序 (24000, 105)
- D-5：向后兼容旧字段名 "eeg_eeg2text"
- 输入格式：(B, SPECTRO_STEPS, SPECTRO_FREQS) = (B, 374, 65)

注意：EEG2Text 使用完全不同的表征路径（频谱，非词级特征），
与 EEG-To-Text 使用的词级 EEG 属于异构输入对比（H-5 文档化）。
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
from constants import SPECTRO_STEPS, SPECTRO_FREQS, MAX_LEN


logger = get_logger("eeg2text_wrapper")


class EEG2TextWrapper(BenchmarkModelWrapper):
    """EEG2Text 模型的 Wrapper。

    EEG 输入：eeg_spectro，spectrogram 格式，shape (B, SPECTRO_STEPS, SPECTRO_FREQS) = (B, 374, 65)
    该格式由 build_unified_dataset.py 的 build_spectrogram() 预计算，
    与 EEG2Text 原始 data_spectro.py 的参数（fs=500, nperseg=128, noverlap=64）完全对齐。

    异构输入说明（H-5）：
    EEG2Text 使用句级 spectrogram，而 EEG-To-Text 使用词级频域特征，
    两者的 EEG 表征层级不同，对比结果需谨慎解读。
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        pretrain_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = MAX_LEN,
        num_beams: int = 1,
        **kwargs,
    ):
        """
        Args:
            model_checkpoint:   微调后的模型 checkpoint 路径
            pretrain_checkpoint: 预训练 encoder checkpoint 路径（可选）
            device:             设备
            max_new_tokens:     最大生成 token 数（greedy decoding）
            num_beams:          beam search 数量（1 = greedy）
        """
        super().__init__()
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        logger.info("Loading EEG2Text model from %s", model_checkpoint)
        logger.info(
            "Expected EEG input: eeg_spectro (B, %d, %d)", SPECTRO_STEPS, SPECTRO_FREQS
        )
        self._load_model(model_checkpoint, pretrain_checkpoint)
        self.model.eval()
        self.text_decoder.eval()

    def _load_model(
        self,
        checkpoint_path: Optional[str],
        pretrain_checkpoint: Optional[str],
    ) -> None:
        """加载 EEG2Text 模型（BrainTranslator + BART decoder）。"""
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EEG2Text-main")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from model_decoding_pretrain import BrainTranslator

        self.model = BrainTranslator(
            in_feature=40,
            decoder_embedding_size=1024,
            additional_encoder_nhead=5,
            additional_encoder_dim_feedforward=2048,
        )

        if pretrain_checkpoint and os.path.isfile(pretrain_checkpoint):
            logger.info("Loading pretrained encoder from %s", pretrain_checkpoint)
            pretrain_state = torch.load(pretrain_checkpoint, map_location=self.device)
            self.model.load_state_dict(pretrain_state, strict=False)

        self.text_decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

        if checkpoint_path and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            try:
                if "encoder" in state_dict and "decoder" in state_dict:
                    self.model.load_state_dict(state_dict["encoder"], strict=False)
                    self.text_decoder.load_state_dict(state_dict["decoder"], strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded model weights from %s", checkpoint_path)
            except (RuntimeError, KeyError) as e:
                logger.warning("Failed to load model weights: %s", e)

        self.model.to(self.device)
        self.text_decoder.to(self.device)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """编码 EEG spectrogram。

        Args:
            eeg:   不使用（接口兼容）
            mask:  不使用（接口兼容）
            meta:  元信息
            batch: 必须包含 "eeg_spectro"（或旧字段 "eeg_eeg2text"）

        Returns:
            encoded_embedding: (B, seq_len, 1024)
        """
        if batch is None:
            raise ValueError(
                "EEG2TextWrapper requires batch dict. "
                "Expected key: 'eeg_spectro' (shape: B x %d x %d)." % (SPECTRO_STEPS, SPECTRO_FREQS)
            )
        # D-5：优先读取新字段名，向后兼容旧字段名
        eeg_data = batch.get("eeg_spectro", batch.get("eeg_eeg2text"))
        if eeg_data is None:
            raise ValueError(
                "EEG2TextWrapper: batch must contain 'eeg_spectro' key "
                "(spectrogram format, shape B x %d x %d). "
                "Got keys: %s" % (SPECTRO_STEPS, SPECTRO_FREQS, list(batch.keys()))
            )

        eeg_input = eeg_data.to(self.device)  # (B, SPECTRO_STEPS, SPECTRO_FREQS)

        with torch.no_grad():
            encoded_embedding = self.model(eeg_input)  # (B, seq_len, 1024)

        return encoded_embedding

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """从 EEG spectrogram 生成文本（自回归，greedy decoding）。

        Args:
            eeg:   不使用（接口兼容）
            mask:  不使用（接口兼容）
            meta:  元信息
            batch: 必须包含 "eeg_spectro"
        """
        with torch.no_grad():
            encoded_embedding = self.encode_eeg(eeg, mask, meta, batch)

            output_ids = self.text_decoder.generate(
                inputs_embeds=encoded_embedding,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,  # greedy decoding（原始论文设置）
            )

            return self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
