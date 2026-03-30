"""
Wrapper for EEG2Text model (BrainTranslator with raw/spectro input).
适配 EEG2Text 模型（使用 raw EEG 或频谱输入）到统一 benchmark 接口。

EEG 数据来源：直接从 spectro pickle 文件中按 (task, subject, sentence_index) 精确查找原始
raw EEG 时序，不做任何近似转换，保持与训练时完全一致的输入分布。
"""

import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger


logger = get_logger("eeg2text_wrapper")

# 默认 spectro pickle 路径（相对于 benchmark 根目录）
DEFAULT_SPECTRO_PICKLE_PATHS: Dict[str, str] = {
    "task1-SR":    "models/EEG2Text-main/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-spectro.pickle",
    "task2-NR":    "models/EEG2Text-main/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle",
    "task3-TSR":   "models/EEG2Text-main/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-spectro.pickle",
    "task2-NR-2.0":"models/EEG2Text-main/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-spectro.pickle",
}


class EEG2TextWrapper(BenchmarkModelWrapper):
    """EEG2Text 模型的 Wrapper。

    输入：统一 benchmark 格式 (B, L_max, C=840) EEG + meta 信息。
    EEG 获取：通过 meta 中的 (task, subject, sentence_index) 从 spectro pickle 精确查找
              原始 raw EEG 时序 (105, T)，不做任何近似，确保与训练分布一致。
    输出：自回归生成的文本列表。
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        pretrain_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        max_spectro_datapoint: int = 24000,
        spectro_pickle_paths: Optional[Dict[str, str]] = None,
        benchmark_root: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_checkpoint: 微调后的模型 checkpoint 路径
            pretrain_checkpoint: 预训练模型 checkpoint 路径（可选）
            device: 设备
            max_new_tokens: 最大生成 token 数
            num_beams: beam search 的 beam 数量（1 表示 greedy）
            max_spectro_datapoint: raw EEG 的最大时间点数（超出则截断，不足则补零）
            spectro_pickle_paths: task_name -> spectro pickle 绝对路径的映射；
                                  None 时使用 DEFAULT_SPECTRO_PICKLE_PATHS 相对路径
            benchmark_root: benchmark 根目录，用于解析相对路径；
                            None 时自动推断为本文件上两级目录
        """
        super().__init__()
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_spectro_datapoint = max_spectro_datapoint

        # 解析 spectro pickle 路径
        if benchmark_root is None:
            benchmark_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if spectro_pickle_paths is None:
            spectro_pickle_paths = {
                task: os.path.join(benchmark_root, rel_path)
                for task, rel_path in DEFAULT_SPECTRO_PICKLE_PATHS.items()
            }

        # 构建 (task, subject, sentence_index) -> raw_eeg (105, T) 查找表
        logger.info("Loading spectro pickles for EEG2Text wrapper ...")
        self._spectro_lookup: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._load_spectro_data(spectro_pickle_paths)
        logger.info("Spectro lookup table: %d entries", len(self._spectro_lookup))

        logger.info("Loading EEG2Text model from %s", model_checkpoint)
        self._load_model(model_checkpoint, pretrain_checkpoint)
        self.model.eval()
        self.text_decoder.eval()

    def _load_spectro_data(self, spectro_pickle_paths: Dict[str, str]) -> None:
        """加载 spectro pickle，构建 (task, subject, sentence_index) -> raw_eeg 查找表。

        raw_eeg 的 shape 为 (105, T)，存储为 numpy array 节省内存。
        """
        for task_name, pkl_path in spectro_pickle_paths.items():
            if not os.path.isfile(pkl_path):
                logger.warning("Spectro pickle not found, skip: %s", pkl_path)
                continue
            logger.info("Loading %s ...", pkl_path)
            with open(pkl_path, "rb") as f:
                dataset_dict = pickle.load(f)  # {subject: [sent_obj, ...]}
            count = 0
            for subject, sent_list in dataset_dict.items():
                for sent_idx, sent_obj in enumerate(sent_list):
                    if sent_obj is None:
                        continue
                    try:
                        raw = sent_obj["sentence_level_EEG"]["rawData"]  # (105, T)
                        if hasattr(raw, "numpy"):
                            raw = raw.numpy()
                        self._spectro_lookup[(task_name, subject, sent_idx)] = np.asarray(raw, dtype=np.float32)
                        count += 1
                    except (KeyError, TypeError):
                        continue
            logger.info("  task=%s: %d entries loaded", task_name, count)

    def _get_raw_eeg_batch(self, meta: List[Dict[str, Any]]) -> torch.Tensor:
        """根据 meta 列表精确查找 raw EEG，拼成 (B, max_spectro_datapoint, 105) 张量。

        - 超出 max_spectro_datapoint 的部分截断。
        - 不足的部分用 0 补齐。
        - 查找失败时用全零占位并发出警告。
        """
        T = self.max_spectro_datapoint
        batch_tensors: List[torch.Tensor] = []

        for m in meta:
            task = m.get("task", "")
            subject = m.get("subject", "")
            sent_idx = int(m.get("sentence_index", -1))

            raw = self._spectro_lookup.get((task, subject, sent_idx))
            if raw is None:
                logger.warning(
                    "raw EEG not found for (task=%s, subject=%s, idx=%d), using zeros.",
                    task, subject, sent_idx,
                )
                eeg_t = torch.zeros(T, 105, dtype=torch.float32)
            else:
                # raw shape: (105, actual_T)
                actual_T = raw.shape[1]
                if actual_T >= T:
                    arr = raw[:, :T]          # 截断
                else:
                    pad = np.zeros((105, T - actual_T), dtype=np.float32)
                    arr = np.concatenate([raw, pad], axis=1)  # 补零
                eeg_t = torch.from_numpy(arr.T)  # (T, 105)

            batch_tensors.append(eeg_t)

        return torch.stack(batch_tensors, dim=0)  # (B, T, 105)

    def _load_model(self, checkpoint_path: str, pretrain_checkpoint: Optional[str]):
        """加载 EEG2Text 模型（BrainTranslator + BART decoder）。"""
        # 动态导入 EEG2Text 的模型
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EEG2Text-main")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            from model_decoding_pretrain import BrainTranslator
        except ImportError as e:
            logger.error("Failed to import EEG2Text BrainTranslator: %s", e)
            raise

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
            # 尝试加载完整模型权重
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

    def _convert_to_raw_eeg(self, eeg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 已废弃：此方法用近似方式转换 EEG，已由 _get_raw_eeg_batch 替代。
        raise NotImplementedError(
            "_convert_to_raw_eeg is deprecated. Use _get_raw_eeg_batch with meta info instead."
        )

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None
    ) -> torch.Tensor:
        """从 spectro pickle 精确获取 raw EEG 并编码成 EEG2Text 内部表示。

        Args:
            eeg:  统一格式 (B, L_max, 840)，本方法不使用此参数，仅作接口兼容。
            mask: 统一格式 (B, L_max)，同上。
            meta: 必须提供，包含 task / subject / sentence_index 字段。

        Returns:
            encoded_embedding: (B, seq_len, 1024)
        """
        if meta is None:
            raise ValueError("EEG2TextWrapper.encode_eeg requires meta with task/subject/sentence_index.")

        eeg_raw = self._get_raw_eeg_batch(meta).to(self.device)  # (B, 24000, 105)

        with torch.no_grad():
            encoded_embedding = self.model(eeg_raw)  # (B, 957, 1024)

        return encoded_embedding

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
            eeg: 统一格式 (B, L_max, 840)，本方法不使用此参数，仅作接口兼容
            mask: 统一格式 (B, L_max)，同上
            meta: 必须提供，包含 task / subject / sentence_index 字段
            batch: 完整 batch（可选）
        
        Returns:
            生成的文本列表，长度为 batch_size。
        """
        with torch.no_grad():
            # 编码 EEG（从 spectro pickle 精确获取 raw EEG）
            encoded_embedding = self.encode_eeg(eeg, mask, meta)  # (B, seq_len, 1024)
            
            # 使用 BART decoder 自回归生成
            output_ids = self.text_decoder.generate(
                inputs_embeds=encoded_embedding,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )

            # 解码生成的 token ids
            generated_texts = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return generated_texts
