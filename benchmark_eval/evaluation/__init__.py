"""评估模块。

负责模型评估、指标计算和结果记录。

主要功能：
- 模型 wrapper 基类定义
- 评估流程管理（加载数据、运行模型、计算指标）
- 指标计算（BLEU、ROUGE、WER 等）
- 结果保存与日志记录
"""

from .model_wrappers import BenchmarkModelWrapper, DummyEchoWrapper, build_model_wrapper
from .metrics import compute_corpus_metrics

# eval_runner 不直接导出类，通过 main 函数调用
# 为了兼容性，添加别名
class EvaluationRunner:
    """评估流程管理器（占位类）。
    
    实际使用时应直接调用 eval_runner.py 中的 main() 函数。
    """
    pass

# 为了向后兼容，添加函数别名
compute_bleu = compute_corpus_metrics
compute_rouge = compute_corpus_metrics
compute_wer = compute_corpus_metrics

__all__ = [
    'BenchmarkModelWrapper',
    'DummyEchoWrapper',
    'build_model_wrapper',
    'EvaluationRunner',
    'compute_corpus_metrics',
    'compute_bleu',
    'compute_rouge',
    'compute_wer',
]
