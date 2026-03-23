"""数据处理模块。

负责从原始 ZuCo MAT 文件构建统一的 EEG-Text 数据集。

主要功能：
- 从 ZuCo MAT 文件读取数据
- 标签加载与补全（情感标签、关系标签）
- 构建统一的 pickle 数据集
- 数据集加载与批处理
"""

from .build_unified_dataset import main as build_dataset
from .dataset import UnifiedDataset

# 为了向后兼容，添加别名
UnifiedEEGDataset = UnifiedDataset

__all__ = [
    'build_dataset',
    'UnifiedDataset',
    'UnifiedEEGDataset',
]
