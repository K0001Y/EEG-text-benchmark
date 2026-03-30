"""GLIM 模型的 Wrapper 实现。

GLIM 架构：
- Prompt Embedder: 编码任务、数据集、主题提示
- EEG Encoder: 编码 EEG 信号
- Aligner: 对齐 EEG 和文本嵌入
- Text Model: T5/BART 进行文本生成

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
import torch.nn.functional as F

import sys
import os
# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.model_wrappers import BenchmarkModelWrapper
from utils.logging_utils import get_logger

logger = get_logger("glim_wrapper")


class GLIMWrapper(BenchmarkModelWrapper):
    """GLIM 模型的 Wrapper。
    
    将统一的 (B, L_max, C=840) EEG 序列输入 GLIM 模型，
    使用 T5/BART 进行自回归文本生成。
    """

    def __init__(
        self,
        model_checkpoint: str,
        text_model_id: str = "google/flan-t5-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 64,
        num_beams: int = 2,
        input_eeg_len: int = 1280,
        hidden_eeg_len: int = 96,
        input_dim: int = 128,
        hidden_dim: int = 256,
        embed_dim: int = 1024,
        **kwargs
    ):
        """初始化 GLIM Wrapper。
        
        Args:
            model_checkpoint: GLIM 模型权重路径
            text_model_id: T5/BART 模型 ID
            device: 运行设备
            max_new_tokens: 生成的最大 token 数
            num_beams: beam search 的 beam 大小
            input_eeg_len: 输入 EEG 长度
            hidden_eeg_len: 隐藏 EEG 长度
            input_dim: 输入维度
            hidden_dim: 隐藏维度
            embed_dim: 嵌入维度
        """
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.text_model_id = text_model_id
        
        # 动态导入 GLIM 模型代码
        glim_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "GLIM-main"
        )
        if glim_path not in sys.path:
            sys.path.insert(0, glim_path)
        
        # 添加 model 子目录到路径
        glim_model_path = os.path.join(glim_path, "model")
        if glim_model_path not in sys.path:
            sys.path.insert(0, glim_model_path)
        
        try:
            # 使用绝对导入方式加载 GLIM 模块
            # 需要先设置包结构使得相对导入可以工作
            import importlib.util
            import types
            
            glim_module_path = os.path.join(glim_model_path, "glim.py")
            modules_path = os.path.join(glim_model_path, "modules.py")
            
            # 创建一个虚拟的 model 包
            model_package = types.ModuleType('model')
            model_package.__path__ = [glim_model_path]
            model_package.__file__ = os.path.join(glim_model_path, '__init__.py')
            sys.modules['model'] = model_package
            
            # 加载 modules 模块
            spec_modules = importlib.util.spec_from_file_location("model.modules", modules_path)
            modules = importlib.util.module_from_spec(spec_modules)
            sys.modules['model.modules'] = modules
            spec_modules.loader.exec_module(modules)
            
            # 加载 glim 模块
            spec = importlib.util.spec_from_file_location("model.glim", glim_module_path)
            glim_module = importlib.util.module_from_spec(spec)
            sys.modules['model.glim'] = glim_module
            spec.loader.exec_module(glim_module)
            GLIM = glim_module.GLIM
            
            from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
            
            logger.info("Loading GLIM model from %s", model_checkpoint)
            
            # 保存模型参数（因为 GLIM 模型不保存这些属性）
            self.input_dim = input_dim
            self.input_eeg_len = input_eeg_len
            
            # 初始化模型
            self.model = GLIM(
                input_eeg_len=input_eeg_len,
                hidden_eeg_len=hidden_eeg_len,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                text_model_id=text_model_id,
            )
            
            # setup 模型（初始化 tokenizer 和 text_model）
            self.model.setup(stage="test")
            
            # 加载模型权重
            if os.path.isfile(model_checkpoint):
                checkpoint = torch.load(model_checkpoint, map_location=device)
                # 根据实际 checkpoint 格式调整
                if 'state_dict' in checkpoint:
                    # 移除 text_model 相关的权重（因为它们是冻结的）
                    state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                                 if not k.startswith('text_model')}
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                logger.info("Model weights loaded from %s", model_checkpoint)
            else:
                logger.warning("Checkpoint file not found: %s, using randomly initialized weights", 
                             model_checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # tokenizer 已在 setup 中初始化
            self.tokenizer = self.model.tokenizer
            
            # 默认的 prompt（任务、数据集、主题均为 UNK）
            self.default_prompt = (['<UNK>'], ['<UNK>'], ['<UNK>'])
            
            logger.info("GLIM model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load GLIM model: %s", e)
            raise

    def _convert_to_glim_format(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将统一格式 (B, L_max, C=840) 转换为 GLIM 所需的格式。
        
        GLIM 期望的输入格式：
        - eeg: (B, input_eeg_len, input_dim)
        
        转换策略：
        1. 将 C=840 维压缩到 input_dim=128 维（使用线性投影）
        2. 将 L_max interpolate 到 input_eeg_len=1280
        
        Args:
            eeg: (B, L_max, 840) EEG 序列
            mask: (B, L_max) mask
            
        Returns:
            (B, input_eeg_len, input_dim) EEG 序列和对应的 mask
        """
        B, L_max, C = eeg.shape
        
        # Step 1: 压缩通道维度 840 -> 128
        # 简化策略：使用平均池化（每 840/128 ≈ 6.5 个通道取平均）
        # 或使用线性投影（需要额外的参数）
        
        # 这里使用分组平均的方式
        # 840 / 128 ≈ 6.5，我们可以将 840 分成 128 组
        # 方法 1: 简单的 reshape + mean（需要 840 能被某个数整除）
        # 方法 2: 使用 adaptive pooling
        
        # 使用 adaptive average pooling
        eeg_transposed = eeg.transpose(1, 2)  # (B, 840, L_max)
        eeg_pooled = F.adaptive_avg_pool1d(eeg_transposed, self.input_dim)  # (B, 840, 128)
        eeg_compressed = eeg_pooled.transpose(1, 2)  # (B, 128, 840)
        
        # 等等，这样维度不对。让我们重新思考：
        # 我们需要 (B, L_max, 840) -> (B, input_eeg_len, input_dim)
        # input_eeg_len = 1280, input_dim = 128
        
        # 正确的方法：
        # Step 1: 通道维度 840 -> 128 (使用 adaptive pooling)
        eeg_transposed = eeg.transpose(1, 2)  # (B, 840, L_max)
        eeg_channel_pooled = F.adaptive_avg_pool1d(eeg_transposed, L_max)  # 保持长度不变
        
        # 实际上，我们需要在通道维度上做压缩
        # 将 (B, L_max, 840) 重塑为 (B*L_max, 840, 1)，然后使用 adaptive pooling
        eeg_reshaped = eeg.view(B * L_max, C, 1)  # (B*L_max, 840, 1)
        eeg_channel_compressed = F.adaptive_avg_pool1d(eeg_reshaped, self.input_dim)  # (B*L_max, 840, 128)
        
        # 不对，adaptive_avg_pool1d 作用在最后一个维度上
        # 让我们使用更简单的方法：线性插值
        
        # 方法：将 840 维分成 128 组，每组取平均
        group_size = C // self.input_dim  # 840 // 128 = 6
        remainder = C % self.input_dim  # 840 % 128 = 72
        
        # 简化版本：直接使用线性投影（需要一个可学习的投影层）
        # 但这会增加参数，不符合 wrapper 的设计
        
        # 最简单的方法：均匀分组+平均
        # 将 840 分成 128 组，每组大小约为 6-7
        eeg_compressed_list = []
        start_idx = 0
        for i in range(self.input_dim):
            end_idx = start_idx + group_size + (1 if i < remainder else 0)
            group = eeg[:, :, start_idx:end_idx].mean(dim=-1, keepdim=True)  # (B, L_max, 1)
            eeg_compressed_list.append(group)
            start_idx = end_idx
        
        eeg_compressed = torch.cat(eeg_compressed_list, dim=-1)  # (B, L_max, 128)
        
        # Step 2: 时间维度 L_max -> input_eeg_len (1280)
        eeg_transposed = eeg_compressed.transpose(1, 2)  # (B, 128, L_max)
        eeg_interpolated = F.interpolate(
            eeg_transposed,
            size=self.input_eeg_len,
            mode='linear',
            align_corners=False
        )  # (B, 128, 1280)
        eeg_final = eeg_interpolated.transpose(1, 2)  # (B, 1280, 128)
        
        # 同样地，interpolate mask
        mask_unsqueezed = mask.unsqueeze(1).float()  # (B, 1, L_max)
        mask_interpolated = F.interpolate(
            mask_unsqueezed,
            size=self.input_eeg_len,
            mode='nearest'
        )  # (B, 1, 1280)
        mask_final = mask_interpolated.squeeze(1)  # (B, 1280)
        
        return eeg_final, mask_final

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        meta: List[Dict[str, Any]] | None = None
    ) -> Any:
        """将统一格式的 EEG (B, L_max, C=840) 编码成模型内部表示。
        
        Args:
            eeg: (B, L_max, 840) EEG 序列
            mask: (B, L_max) 1 表示有效，0 表示 padding
            meta: 元信息列表（可包含 prompt 信息）
            
        Returns:
            包含编码后 EEG 表示的字典
        """
        # 转换为 GLIM 格式
        glim_eeg, glim_mask = self._convert_to_glim_format(eeg, mask)
        glim_eeg = glim_eeg.to(self.device)
        glim_mask = glim_mask.to(self.device)
        
        # 从 meta 中提取 prompt 信息（如果有的话）
        prompts = self._extract_prompts_from_meta(meta, eeg.size(0))
        
        # 编码 prompt
        prompt_ids = self.model.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.model.p_embedder(prompt_ids, self.model.eval_pembed)
        
        # 通过 EEG encoder 编码 EEG
        eeg_hiddens, _ = self.model.eeg_encoder(glim_eeg, glim_mask, prompt_embed)
        
        # 通过 aligner 获取 EEG 嵌入
        eeg_embeds, eeg_emb_vector = self.model.aligner.embed_eeg(eeg_hiddens)
        
        return {
            "eeg_embeds": eeg_embeds,  # (B, hidden_eeg_len, embed_dim)
            "eeg_emb_vector": eeg_emb_vector,  # (B, embed_dim)
            "glim_mask": glim_mask,  # (B, input_eeg_len)
        }

    def _extract_prompts_from_meta(
        self,
        meta: List[Dict[str, Any]] | None,
        batch_size: int
    ) -> tuple[list, list, list]:
        """从 meta 信息中提取 prompt。
        
        GLIM 的 prompt 格式：
        - (task_list, dataset_list, subject_list)
        - 每个 list 的长度为 batch_size
        
        Args:
            meta: 元信息列表
            batch_size: batch 大小
            
        Returns:
            (task_list, dataset_list, subject_list)
        """
        if meta is None:
            # 使用默认的 UNK prompt
            task_list = ['<UNK>'] * batch_size
            dataset_list = ['<UNK>'] * batch_size
            subject_list = ['<UNK>'] * batch_size
            return (task_list, dataset_list, subject_list)
        
        task_list = []
        dataset_list = []
        subject_list = []
        
        for m in meta:
            # 提取任务信息
            task = m.get('task', 'task1-SR')
            if 'NR' in task or 'task2-NR' in task:
                task_key = '<NR>'
            elif 'TSR' in task or 'task3-TSR' in task:
                task_key = '<TSR>'
            else:
                task_key = '<UNK>'
            task_list.append(task_key)
            
            # 提取数据集信息
            dataset = m.get('dataset', 'ZuCo1')
            if dataset in ['ZuCo1', 'ZuCo2']:
                dataset_list.append(dataset)
            else:
                dataset_list.append('<UNK>')
            
            # 提取主题信息
            subject = m.get('subject', '<UNK>')
            # 检查是否在支持的主题列表中
            if subject in self.model.prompt_keys['subject']:
                subject_list.append(subject)
            else:
                subject_list.append('<UNK>')
        
        return (task_list, dataset_list, subject_list)

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
            encoded = self.encode_eeg(eeg, mask, meta)
            
            eeg_embeds = encoded["eeg_embeds"]  # (B, hidden_eeg_len, embed_dim)
            
            # 转换为 bfloat16 以匹配 T5 模型的精度
            eeg_embeds = eeg_embeds.to(torch.bfloat16)
            
            # 使用 text_model 的 generate 方法进行自回归生成
            from transformers.modeling_outputs import BaseModelOutput
            
            gen_ids = self.model.text_model.generate(
                encoder_outputs=BaseModelOutput(eeg_embeds),
                num_beams=self.num_beams,
                min_length=0,
                max_length=self.max_new_tokens,
                early_stopping=True,
                do_sample=False,  # greedy decoding
            )
            
            # 解码生成的 token IDs
            generated_texts = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_texts
