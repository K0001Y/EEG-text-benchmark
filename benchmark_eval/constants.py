"""全局命名常量。

统一管理所有关键超参数。
引用方式：from benchmark_eval.constants import MAX_LEN, EEG_WORD_DIM, ...
"""

# ---------- 词级 EEG 常量 ----------
MAX_LEN: int = 56
"""词级 EEG 序列最大长度（与 EEG-To-Text 原始实现一致）。
CET-MAE 的句级 token 追加在 wrapper 层处理，不计入此值。"""

EEG_CHANNELS: int = 105
"""每个频段的 EEG 通道数（ZuCo 标准 105 电极）。"""

EEG_BANDS: int = 8
"""EEG 频段数（_t1, _t2, _a1, _a2, _b1, _b2, _g1, _g2）。"""

EEG_WORD_DIM: int = EEG_CHANNELS * EEG_BANDS  # 840
"""词级 EEG 特征向量维度（105 通道 × 8 频段 = 840）。"""

# ---------- EEG2Text spectrogram 常量 ----------
RAW_SAMPLING_RATE: int = 500
"""ZuCo 原始 EEG 采样率（Hz）。"""

SPECTRO_NPERSEG: int = 128
"""scipy.signal.spectrogram 窗口大小（与 EEG2Text 原始 data_spectro.py 对齐）。"""

SPECTRO_NOVERLAP: int = 64
"""scipy.signal.spectrogram 重叠大小（与 EEG2Text 原始 data_spectro.py 对齐）。"""

SPECTRO_STEPS: int = 374
"""EEG2Text spectrogram 的时间步数（由 scipy.signal.spectrogram 输出，取决于
rawData 时长、nperseg 和 noverlap）。"""

SPECTRO_FREQS: int = 65
"""EEG2Text spectrogram 的频率维度（nperseg // 2 + 1 = 65）。"""

# ---------- GLIM 常量 ----------
GLIM_EEG_LEN: int = 1280
"""GLIM 模型期望的输入 EEG 序列长度。"""

GLIM_EEG_DIM: int = 128
"""GLIM 模型期望的输入 EEG 特征维度。"""

GLIM_HIDDEN_EEG_LEN: int = 96
"""GLIM EEG encoder 隐层输出长度。"""

# ---------- 评估常量 ----------
DEFAULT_SEED: int = 42
"""全局默认随机种子，用于 data split、噪声生成和模型推理的可复现性。"""
