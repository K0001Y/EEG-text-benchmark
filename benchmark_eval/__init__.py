"""Unified EEG-to-Text benchmark package.

优化后的目录结构：

data_processing/    # 数据处理模块
├── build_unified_dataset.py    # 从 ZuCo MAT 构建统一数据集
└── dataset.py                  # 数据集加载与批处理

evaluation/         # 评估模块
├── model_wrappers.py           # 模型 wrapper 基类
├── eval_runner.py              # 评估流程管理
└── metrics.py                  # 指标计算

wrappers/           # 模型适配器
├── eeg_to_text_wrapper.py      # EEG-To-Text wrapper
├── eeg2text_wrapper.py         # EEG2Text wrapper
├── cet_mae_wrapper.py          # CET-MAE wrapper
└── glim_wrapper.py             # GLIM wrapper

utils/              # 通用工具
└── logging_utils.py            # 日志工具

scripts/            # 脚本文件
└── run_eval_dummy.sh           # 示例评估脚本
"""

# 导出主要接口
from .data_processing import build_dataset, UnifiedEEGDataset
from .evaluation import (
    BenchmarkModelWrapper,
    DummyEchoWrapper,
    build_model_wrapper,
    EvaluationRunner,
    compute_bleu,
    compute_rouge,
    compute_wer,
)
from .utils import setup_logging, get_logger

__all__ = [
    # 数据处理
    'build_dataset',
    'UnifiedEEGDataset',
    # 评估
    'BenchmarkModelWrapper',
    'DummyEchoWrapper',
    'build_model_wrapper',
    'EvaluationRunner',
    'compute_bleu',
    'compute_rouge',
    'compute_wer',
    # 工具
    'setup_logging',
    'get_logger',
]

