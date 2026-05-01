"""标准化 CLI 参数解析工具。

从各检索和诊断脚本中抽取共享的 argparse 参数定义，
保证跨脚本参数命名和默认值一致。
"""

import argparse

NOISE_TYPES = ("real", "gaussian", "shuffle", "zero")


def add_common_retrieval_args(parser=None):
    """添加检索评估脚本的共用参数。

    覆盖 CET-MAE / EEG2Text / EEG-To-Text / GLIM 四个检索脚本的
    共同 CLI 参数。各脚本可在调用本函数后追加模型专属参数
    （如 --model-type、--text-model-id 等）。
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True,
                        help="统一数据集路径 (.pkl)")
    parser.add_argument("--model-checkpoint", required=True,
                        help="模型检查点路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--phase", default="test",
                        help="数据集划分 (train/val/test)")
    parser.add_argument("--noise-type", default="real", choices=NOISE_TYPES,
                        help="噪声条件: real(默认)/gaussian/shuffle/zero")
    parser.add_argument("--eeg-batch-size", type=int, default=32,
                        help="EEG 编码批大小")
    parser.add_argument("--text-batch-size", type=int, default=64,
                        help="文本编码批大小")
    return parser


def add_diagnostic_args(parser=None):
    """添加诊断脚本的共用参数。

    覆盖 validate_eeg_signal.py 等诊断脚本的共同 CLI 参数。
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True,
                        help="统一数据集路径 (.pkl)")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（默认由脚本决定）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--skip-a3", action="store_true",
                        help="跳过 A3 去被试化验证")
    parser.add_argument("--skip-tsne", action="store_true",
                        help="跳过 t-SNE 可视化（节省时间）")
    return parser
