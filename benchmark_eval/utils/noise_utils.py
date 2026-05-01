"""噪声生成工具函数。

提供统一的噪声生成接口，用于检索评估和信号诊断的对照实验。
"""

import numpy as np


def generate_gaussian_noise(shape, seed=42):
    """生成标准高斯噪声。

    使用 np.random.RandomState 保证可复现性，输出 float32。
    行为与 validate_eeg_signal.py 中的 generate_noise_features 一致
    （单种子生成整批噪声）。

    Args:
        shape: 噪声数组形状
        seed: 随机种子

    Returns:
        np.ndarray: 标准正态分布噪声，dtype=np.float32
    """
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def generate_zero_signal(shape):
    """生成全零信号。

    与 dataset.py _generate_noise_eeg 中 zero 类型行为一致。

    Args:
        shape: 输出数组形状

    Returns:
        np.ndarray: 全零数组，dtype=np.float32
    """
    return np.zeros(shape, dtype=np.float32)


def shuffle_signal(data, seed=42, axis=0):
    """沿指定轴随机打乱信号（完全错位排列）。

    生成 derangement（无不动点）并沿指定轴重排数据，
    与 UnifiedDataset 的 shuffle_mode 及 _generate_derangement 语义一致。

    Args:
        data: 原始信号数组
        seed: 随机种子
        axis: 打乱的轴（默认 0，即样本轴）

    Returns:
        np.ndarray: 打乱后的信号数组
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data).copy()
    n = data.shape[axis]
    if n <= 1:
        return data

    # 生成 derangement（完全错位排列）
    perm = np.arange(n)
    for _ in range(1000):
        rng.shuffle(perm)
        if np.all(perm != np.arange(n)):
            break
    else:
        # 兜底修正不动点
        fixed = np.where(perm == np.arange(n))[0]
        for i in fixed:
            j = (i + 1) % n
            while perm[j] == j:
                j = (j + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    # 沿指定轴应用置换
    slices = [slice(None)] * data.ndim
    slices[axis] = perm
    return data[tuple(slices)]


def apply_noise(data, noise_type, seed=42):
    """统一噪声应用接口。

    根据噪声类型对输入数据应用相应的变换：
      - "real":    返回原始数据（无噪声）
      - "gaussian":生成同形状标准高斯噪声替代
      - "zero":    生成同形状全零信号替代
      - "shuffle": 沿样本轴做 derangement 打乱

    Args:
        data: 原始信号数据（numpy 数组或类数组）
        noise_type: "real" | "gaussian" | "zero" | "shuffle"
        seed: 随机种子

    Returns:
        处理后的信号数组

    Raises:
        ValueError: 未知的 noise_type
    """
    if noise_type == "real":
        return data
    elif noise_type == "gaussian":
        return generate_gaussian_noise(np.shape(data), seed)
    elif noise_type == "zero":
        return generate_zero_signal(np.shape(data))
    elif noise_type == "shuffle":
        return shuffle_signal(data, seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
