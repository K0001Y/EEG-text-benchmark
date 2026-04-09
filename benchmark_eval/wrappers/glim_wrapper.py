"""GLIM 模型的 Wrapper 实现（v2）。

v2 更新：
- L-1：删除重复的 import sys/os
- M-8：删除废弃的、注释掉的错误转换逻辑
- M-1：input_eeg_len / input_dim 等参数从外部传入，不在 wrapper 内硬编码
- M-4：入口处添加输入验证
- D-5：优先读取 batch["eeg_word_raw"]（向后兼容旧字段名 "eeg_raw"）

GLIM 架构：
  Prompt Embedder → 编码任务/数据集/主题提示
  EEG Encoder    → 编码 EEG 信号（期望 (B, 1280, 128)）
  Aligner        → 对齐 EEG 和文本嵌入
  Text Model     → T5/BART 文本生成

EEG 转换（D-3 待验证）：
  从统一格式 (B, MAX_LEN, 840) → (B, GLIM_EEG_LEN, GLIM_EEG_DIM) 的转换在 wrapper 层进行。
  当前使用分组平均压缩通道维度 + 线性插值扩展时间维度。
  该转换与 GLIM 原始训练数据的分布可能存在差异，评估结果仅供参考。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger
from constants import GLIM_EEG_LEN, GLIM_EEG_DIM, GLIM_HIDDEN_EEG_LEN, MAX_LEN

logger = get_logger("glim_wrapper")


class GLIMWrapper(BenchmarkModelWrapper):
    """GLIM 模型的 Wrapper。

    EEG 输入：eeg_word_raw，原始词级特征，(B, MAX_LEN, 840)
    内部转换为 (B, GLIM_EEG_LEN, GLIM_EEG_DIM) = (B, 1280, 128)
    """

    def __init__(
        self,
        model_checkpoint: str,
        text_model_id: str = "google/flan-t5-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = MAX_LEN,
        num_beams: int = 2,
        # M-1：从外部传入，不在 wrapper 内硬编码
        input_eeg_len: int = GLIM_EEG_LEN,
        hidden_eeg_len: int = GLIM_HIDDEN_EEG_LEN,
        input_dim: int = GLIM_EEG_DIM,
        hidden_dim: int = 256,
        embed_dim: int = 1024,
        **kwargs,
    ):
        """初始化 GLIM Wrapper。

        Args:
            model_checkpoint: GLIM 模型权重路径
            text_model_id:    T5/BART 模型 ID
            device:           运行设备
            max_new_tokens:   生成的最大 token 数
            num_beams:        beam search 大小（原始论文 2）
            input_eeg_len:    输入 EEG 序列长度（从配置读取）
            hidden_eeg_len:   隐层 EEG 长度（从配置读取）
            input_dim:        输入 EEG 特征维度（从配置读取）
            hidden_dim:       隐层维度
            embed_dim:        嵌入维度
        """
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.text_model_id = text_model_id
        self.input_dim = input_dim
        self.input_eeg_len = input_eeg_len

        glim_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "GLIM-main",
        )
        if glim_path not in sys.path:
            sys.path.insert(0, glim_path)

        glim_model_path = os.path.join(glim_path, "model")
        if glim_model_path not in sys.path:
            sys.path.insert(0, glim_model_path)

        try:
            import importlib.util
            import types

            glim_module_path = os.path.join(glim_model_path, "glim.py")
            modules_path = os.path.join(glim_model_path, "modules.py")

            # 创建虚拟 model 包，支持相对导入
            model_package = types.ModuleType("model")
            model_package.__path__ = [glim_model_path]
            model_package.__file__ = os.path.join(glim_model_path, "__init__.py")
            sys.modules["model"] = model_package

            spec_modules = importlib.util.spec_from_file_location("model.modules", modules_path)
            modules = importlib.util.module_from_spec(spec_modules)
            sys.modules["model.modules"] = modules
            spec_modules.loader.exec_module(modules)

            spec = importlib.util.spec_from_file_location("model.glim", glim_module_path)
            glim_module = importlib.util.module_from_spec(spec)
            sys.modules["model.glim"] = glim_module
            spec.loader.exec_module(glim_module)
            GLIM = glim_module.GLIM

            logger.info("Loading GLIM model from %s", model_checkpoint)

            self.model = GLIM(
                input_eeg_len=input_eeg_len,
                hidden_eeg_len=hidden_eeg_len,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                text_model_id=text_model_id,
            )

            self.model.setup(stage="test")

            if os.path.isfile(model_checkpoint):
                checkpoint = torch.load(model_checkpoint, map_location=device)
                if "state_dict" in checkpoint:
                    state_dict = {
                        k: v for k, v in checkpoint["state_dict"].items()
                        if not k.startswith("text_model")
                    }
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                logger.info("Model weights loaded from %s", model_checkpoint)
            else:
                logger.warning(
                    "Checkpoint not found: %s, using randomly initialized weights",
                    model_checkpoint,
                )

            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = self.model.tokenizer
            self.default_prompt = (["<UNK>"], ["<UNK>"], ["<UNK>"])

            logger.info("GLIM model loaded successfully")
            logger.info(
                "EEG conversion: (%d, 840) → (%d, %d)",
                MAX_LEN, input_eeg_len, input_dim,
            )

        except Exception as e:
            logger.error("Failed to load GLIM model: %s", e)
            raise

    def _convert_to_glim_format(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将统一格式 (B, MAX_LEN, 840) 转换为 GLIM 期望的 (B, input_eeg_len, input_dim)。

        转换策略（D-3 待验证，与原始训练分布可能不同）：
        Step 1：通道维度压缩 840 → input_dim（分组平均）
        Step 2：时间维度扩展 MAX_LEN → input_eeg_len（线性插值）

        Args:
            eeg:  (B, MAX_LEN, 840)
            mask: (B, MAX_LEN)

        Returns:
            eeg_final:  (B, input_eeg_len, input_dim)
            mask_final: (B, input_eeg_len)
        """
        B, L_max, C = eeg.shape

        # Step 1：通道维度压缩 840 → input_dim（分组平均）
        group_size = C // self.input_dim
        remainder = C % self.input_dim

        groups = []
        start_idx = 0
        for i in range(self.input_dim):
            end_idx = start_idx + group_size + (1 if i < remainder else 0)
            group = eeg[:, :, start_idx:end_idx].mean(dim=-1, keepdim=True)
            groups.append(group)
            start_idx = end_idx
        eeg_compressed = torch.cat(groups, dim=-1)  # (B, MAX_LEN, input_dim)

        # Step 2：时间维度扩展 MAX_LEN → input_eeg_len（线性插值）
        eeg_t = eeg_compressed.transpose(1, 2)  # (B, input_dim, MAX_LEN)
        eeg_interp = F.interpolate(
            eeg_t, size=self.input_eeg_len, mode="linear", align_corners=False
        )  # (B, input_dim, input_eeg_len)
        eeg_final = eeg_interp.transpose(1, 2)  # (B, input_eeg_len, input_dim)

        # mask 也做相应插值
        mask_t = mask.unsqueeze(1).float()  # (B, 1, MAX_LEN)
        mask_interp = F.interpolate(mask_t, size=self.input_eeg_len, mode="nearest")
        mask_final = mask_interp.squeeze(1)  # (B, input_eeg_len)

        return eeg_final, mask_final

    def _validate_and_get_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        batch: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """M-4：验证输入并返回 EEG 和 mask（D-5：优先新字段名）。"""
        if batch is not None:
            eeg_input = batch.get("eeg_word_raw", batch.get("eeg_raw", batch.get("eeg", eeg)))
            mask_input = batch.get("mask_word", batch.get("mask", mask))
        else:
            eeg_input = eeg
            mask_input = mask

        if eeg_input is None:
            raise ValueError(
                "GLIMWrapper: cannot find EEG data. "
                "Expected batch key: 'eeg_word_raw' (shape: B x %d x 840)." % MAX_LEN
            )
        return eeg_input, mask_input

    def _extract_prompts_from_meta(
        self,
        meta: Optional[List[Dict[str, Any]]],
        batch_size: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        """从 meta 信息中提取 GLIM prompt（task / dataset / subject）。"""
        if meta is None:
            return (
                ["<UNK>"] * batch_size,
                ["<UNK>"] * batch_size,
                ["<UNK>"] * batch_size,
            )

        task_list, dataset_list, subject_list = [], [], []

        for m in meta:
            task = m.get("task", "task1-SR")
            if "NR" in task:
                task_key = "<NR>"
            elif "TSR" in task:
                task_key = "<TSR>"
            else:
                task_key = "<UNK>"
            task_list.append(task_key)

            dataset = m.get("dataset", "ZuCo1")
            dataset_list.append(dataset if dataset in ["ZuCo1", "ZuCo2"] else "<UNK>")

            subject = m.get("subject", "<UNK>")
            if subject in self.model.prompt_keys.get("subject", []):
                subject_list.append(subject)
            else:
                subject_list.append("<UNK>")

        return (task_list, dataset_list, subject_list)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """将统一格式的 EEG (B, MAX_LEN, 840) 编码成 GLIM 内部表示。"""
        glim_eeg, glim_mask = self._convert_to_glim_format(eeg, mask)
        glim_eeg = glim_eeg.to(self.device)
        glim_mask = glim_mask.to(self.device)

        prompts = self._extract_prompts_from_meta(meta, eeg.size(0))
        prompt_ids = self.model.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.model.p_embedder(prompt_ids, self.model.eval_pembed)

        eeg_hiddens, _ = self.model.eeg_encoder(glim_eeg, glim_mask, prompt_embed)
        eeg_embeds, eeg_emb_vector = self.model.aligner.embed_eeg(eeg_hiddens)

        return {
            "eeg_embeds": eeg_embeds,
            "eeg_emb_vector": eeg_emb_vector,
            "glim_mask": glim_mask,
        }

    def generate_text(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """从 EEG 生成文本（自回归，beam=2）。

        Args:
            eeg:   默认 EEG（fallback）
            mask:  默认 mask（fallback）
            meta:  元信息
            batch: 优先从 batch["eeg_word_raw"] 读取
        """
        eeg_input, mask_input = self._validate_and_get_eeg(eeg, mask, batch)

        with torch.no_grad():
            encoded = self.encode_eeg(eeg_input, mask_input, meta)

            eeg_embeds = encoded["eeg_embeds"].to(torch.bfloat16)

            from transformers.modeling_outputs import BaseModelOutput

            gen_ids = self.model.text_model.generate(
                encoder_outputs=BaseModelOutput(eeg_embeds),
                num_beams=self.num_beams,
                min_length=0,
                max_length=self.max_new_tokens,
                early_stopping=True,
                do_sample=False,
            )

            return self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
