# Unified EEG-to-Text Benchmark

一个用于 EEG-to-Text 模型公平评估的统一 benchmark 框架，支持 EEG-To-Text、DeWave、EEG2Text、GLIM 等多个模型的标准化对比。

## 项目概述

本项目旨在为多个 EEG-to-Text 模型提供公平、可复现的评估基准。通过统一的数据预处理流程、标准化的模型接口和一致的评估指标，确保不同模型之间的比较具有实际意义。

### 核心特性

- **统一数据格式**：将 ZuCo v1.0/v2.0 的原始 `.mat` 数据转换为标准化的中间格式
- **标准化模型接口**：所有模型通过统一的 wrapper 接口进行适配
- **严格自回归生成**：评估阶段禁用 teacher forcing，真实反映模型能力
- **多维度评估指标**：支持 BLEU、ROUGE、WER 等文本相似度指标
- **鲁棒性测试**：支持噪声 EEG 控制实验（高斯噪声/均匀噪声）
- **编码器检索测试**：支持 EEG-文本检索评估（Top-K 准确率）

## 项目结构

```
benchmark/
├── benchmark_eval/           # 核心评估框架
│   ├── constants.py                  # 全局命名常量（MAX_LEN、EEG_WORD_DIM 等）
│   ├── data_processing/      # 数据处理模块
│   │   ├── build_unified_dataset.py  # 从 ZuCo MAT 构建统一数据集
│   │   └── dataset.py                # 数据集加载与批处理
│   ├── evaluation/           # 评估模块
│   │   ├── model_wrappers.py         # 模型 wrapper 基类
│   │   ├── eval_runner.py            # 评估流程管理（含随机种子、统一输出 schema）
│   │   └── metrics.py                # 指标计算（BLEU、ROUGE、WER、BERTScore）
│   ├── wrappers/             # 模型适配器
│   │   ├── eeg_to_text_wrapper.py    # EEG-To-Text wrapper
│   │   ├── eeg2text_wrapper.py       # EEG2Text wrapper（spectrogram 输入）
│   │   ├── cet_mae_wrapper.py        # CET-MAE wrapper
│   │   └── glim_wrapper.py           # GLIM wrapper
│   ├── config/               # 配置文件
│   │   └── eval_config.yaml          # 评估配置（max_len、生成参数、seed 等）
│   ├── utils/                # 通用工具
│   │   └── logging_utils.py          # 日志工具
│   ├── scripts/              # 脚本文件
│   │   └── run_eval_dummy.sh         # 示例评估脚本
│   └── __init__.py           # 主模块导出
│
├── data/                     # 数据目录
│   ├── ZuCo1/               # ZuCo v1.0 数据
│   │   └── task_materials/  # 任务材料（标签文件）
│   └── ZuCo2/               # ZuCo v2.0 数据
│       └── task_materials/
│
├── models/                   # 模型代码目录
│   ├── CET-MAE/             # CET-MAE 模型
│   ├── DeWave-main/         # DeWave 模型
│   ├── EEG-To-Text-main/    # EEG-To-Text 模型
│   ├── EEG2Text-main/       # EEG2Text 模型
│   └── GLIM-main/           # GLIM 模型
│
└── docs/                     # 文档目录
    ├── EEG_benchmark_eval_scheme.md      # 详细评估方案
    ├── EEG_data_preprocessing_comparison.md  # 数据预处理对比
    └── EEG_models_comparison.md          # 模型对比分析
```

## 快速开始

### 环境准备

1. 克隆仓库并进入项目目录

2. 安装依赖（根据具体模型选择对应的 environment.yml）

```bash
# 示例：使用 EEG-To-Text 的环境
conda env create -f models/EEG-To-Text-main/environment.yml
conda activate eeg-to-text
```

3. 准备数据

下载 ZuCo v1.0 和 v2.0 数据集，放置在 `data/` 目录下：
- `data/ZuCo1/task_materials/` - ZuCo v1.0 任务材料
- `data/ZuCo2/task_materials/` - ZuCo v2.0 任务材料

### 构建统一数据集

```python
from benchmark_eval.data_processing import build_dataset

# 构建统一数据集（max_len=56 为基准值，与 EEG-To-Text 原始训练保持一致）
build_dataset(
    zuco_root="data",
    tasks="task1-SR,task2-NR,task3-TSR",
    output="data/unified_dataset.pkl",
    max_len=56,
    dim=105,
    eeg_type="GD"
)
```

### 运行评估

```python
from benchmark_eval.evaluation import build_model_wrapper
from benchmark_eval.data_processing import UnifiedEEGDataset

# 加载数据集
dataset = UnifiedEEGDataset("data/unified_dataset.pkl", phase="test")

# 加载模型 wrapper
model = build_model_wrapper("eeg_to_text", checkpoint="path/to/checkpoint.pt")

# 运行评估（示例）
for batch in dataset:
    # v2 字段名（旧字段名同样支持，向后兼容）
    eeg = batch["eeg_word_norm1d"]   # (L_max, 840) 词级1D归一化 EEG
    mask = batch["mask_word"]         # (L_max,) 词级有效性 mask
    meta = batch["meta"]
    
    # 生成文本
    generated_text = model.generate_text(eeg, mask, meta, batch=batch)
    print(f"Generated: {generated_text}")
```

## 支持的模型

| 模型 | 论文 | 特点 | Wrapper 路径 | 备注 |
|------|------|------|-------------|------|
| EEG-To-Text | [NeurIPS 2023] | 词级 EEG 序列 + BART-large | `wrappers/eeg_to_text_wrapper.py` | 完整开源；beam=5, do_sample=True |
| EEG2Text | [ACL 2023] | 句级 spectrogram (374, 65) + BART | `wrappers/eeg2text_wrapper.py` | 完整开源；greedy decoding |
| CET-MAE | [NeurIPS 2023] | 跨模态 MAE + Multi-Stream | `wrappers/cet_mae_wrapper.py` | 使用外接 BART 解码器¹；greedy decoding |
| GLIM | [NeurIPS 2024] | Prompt + T5-large | `wrappers/glim_wrapper.py` | 完整开源；beam=2 |

> **注¹**：CET-MAE 原始论文未开源文本解码器组件。本实现使用预训练的 BART decoder 作为代理解码器进行公平对比。该方案未进行 encoder-decoder 联合训练，生成结果仅供参考。建议同时参考 EEG-文本检索指标（R@1/R@5/R@10）评估其编码器能力。

## 统一数据格式

### 样本字段

统一数据集采用 **v2 字段命名规范**，旧字段名（`eeg`、`mask` 等）保留向后兼容：

```python
{
    # 词级特征（EEG-To-Text / CET-MAE / GLIM 共用基础）
    "eeg_word_norm1d": np.ndarray,   # (max_len, 840) 逐词1D z-score 归一化 EEG
    "eeg_word_norm2d": np.ndarray,   # (max_len, 840) 全局2D z-score 归一化（CET-MAE）
    "eeg_word_raw":    np.ndarray,   # (max_len, 840) 原始未归一化词级 EEG
    "mask_word":       np.ndarray,   # (max_len,) int8，1=有效词，0=padding
    "mask_word_with_sent": np.ndarray, # (max_len,) 含句级token的mask（CET-MAE）
    # 频谱特征（EEG2Text 专用）
    "eeg_spectro":     np.ndarray,   # (374, 65) scipy spectrogram，fs=500, nperseg=128
    "mask_spectro":    np.ndarray,   # (374,) int8，频谱有效性mask
    # 句级特征
    "sent_eeg_raw":    np.ndarray,   # (840,) 句级 EEG 特征（CET-MAE 追加到序列末尾）
    # 文本与元信息
    "input_text":  str,              # 输入句子文本
    "phase":       str,              # "train" / "val" / "test"
    "meta": {
        "task":             str,     # 任务类型（如 task1-SR）
        "dataset":          str,     # "ZuCo1" 或 "ZuCo2"
        "subject":          str,     # 被试 ID
        "text_uid":         int,     # 文本唯一 ID
        "sentiment_label":  int,     # 情感标签（task1）
        "relation_label":   str,     # 关系标签（task2/task3）
    },
    # 向后兼容别名（指向 v2 字段）
    "eeg":  np.ndarray,   # → eeg_word_norm1d
    "mask": np.ndarray,   # → mask_word
}
```

### 数据划分

- **训练集 (train)**：80% 样本，用于模型训练
- **验证集 (val)**：10% 样本，用于超参数调优和早停
- **测试集 (test)**：10% 样本，用于最终评估

划分在每个被试（subject）内部独立进行，确保不同被试的数据不会混合。

## 评估指标

### 文本相似度指标

- **BLEU-1/2/3/4**：n-gram 精确度
- **ROUGE-1/2/L**：召回率导向的指标
- **WER**：词错误率

### 语义对齐指标

- **检索准确率**：top-1/top-5/top-10
- **对比学习准确率**：EEG-Text 对齐度

### 鲁棒性指标

- **噪声 EEG 实验**：测试模型对噪声的鲁棒性
  - 高斯噪声：均值为 0，标准差可调
  - 均匀噪声：指定范围内的随机值
  - 对比真实 EEG 与噪声 EEG 的生成质量差异
- **跨被试泛化**：测试模型对新被试的适应能力

### 编码器检索指标

- **EEG-文本检索准确率**：评估 EEG 编码器与文本的语义对齐能力
  - R@1 (Recall@1)：正确文本排在第 1 位的比例
  - R@5 (Recall@5)：正确文本在前 5 名的比例
  - R@10 (Recall@10)：正确文本在前 10 名的比例
- **适用场景**：特别适合评估 CET-MAE 等预训练编码器的表示学习能力

## 开发指南

### 添加新模型

1. 创建新的 wrapper 文件 `benchmark_eval/wrappers/your_model_wrapper.py`

2. 继承 `BenchmarkModelWrapper` 基类：

```python
from benchmark_eval.evaluation import BenchmarkModelWrapper

class YourModelWrapper(BenchmarkModelWrapper):
    def __init__(self, checkpoint_path, **kwargs):
        # 加载模型
        pass
    
    def encode_eeg(self, eeg, mask, meta=None):
        # 将统一格式转换为模型输入格式
        pass
    
    def generate_text(self, eeg, mask, meta=None):
        # 自回归生成文本
        pass
```

3. 在 `benchmark_eval/wrappers/__init__.py` 中导出

4. 更新 `benchmark_eval/evaluation/model_wrappers.py` 中的 `build_model_wrapper` 函数

### 自定义评估流程

```python
from benchmark_eval.evaluation import EvaluationRunner
from benchmark_eval.data_processing import UnifiedEEGDataset

# 创建自定义评估器
class CustomEvaluator(EvaluationRunner):
    def compute_custom_metric(self, references, predictions):
        # 实现自定义指标
        pass

# 运行评估
evaluator = CustomEvaluator()
evaluator.run(dataset, model)
```

## 文档

- [详细评估方案](docs/EEG_benchmark_eval_scheme.md) - 完整的技术方案文档
- [数据预处理对比](docs/EEG_data_preprocessing_comparison.md) - 各模型数据预处理流程对比
- [模型对比分析](docs/EEG_models_comparison.md) - 各模型架构和特点对比

## 引用

如果您使用了本 benchmark，请引用相关论文：

```bibtex
@inproceedings{eeg_to_text_2023,
  title={Decoding Natural Language from EEG during Image-free Thought},
  author={...},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{eeg2text_2023,
  title={Decoding EEG Signals into Natural Language},
  author={...},
  booktitle={ACL},
  year={2023}
}

@inproceedings{cet_mae_2023,
  title={CET-MAE: Cross-modal EEG-Text Masked Autoencoder},
  author={...},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{glim_2024,
  title={GLIM: Grounded Language-Interfaced Model},
  author={...},
  booktitle={NeurIPS},
  year={2024}
}
```

## 许可证

本项目遵循 MIT 许可证。各模型代码遵循其原始仓库的许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 联系方式

如有问题或建议，请通过 GitHub Issue 联系我们。

## 已知限制与注意事项

### 数据重建要求

本次优化（v2 字段命名 + spectrogram 格式变更）只对**重新生成**的 `unified_zuco.pkl` 完全生效：

- **旧 PKL 文件**：`dataset.py` 和所有 wrappers 中已实现向后兼容逻辑，使用旧字段名（`eeg`、`mask` 等）的 PKL 文件仍可正常加载
- **`eeg_spectro` 字段**：只有重新运行 `build_unified_dataset.py` 才能获得正确的 spectrogram 格式 (374, 65)；旧 PKL 文件中该字段存储的是原始时序数据，会被 EEG2Text wrapper 自动 fallback 处理

### 生成参数差异

各模型使用不同的生成策略（在 `eval_config.yaml` 的 `generation.model_overrides` 中配置）：

| 模型 | beam | do_sample | repetition_penalty |
|------|------|-----------|--------------------|
| EEG-To-Text | 5 | True | 5.0 |
| EEG2Text | 1 (greedy) | False | - |
| CET-MAE | 1 (greedy) | False | - |
| GLIM | 2 | False | - |

### GLIM EEG 格式转换

GLIM 期望输入 `(B, 1280, 128)` 格式，而统一数据集存储的是词级 `(max_len, 840)` 格式。当前 wrapper 中通过 `adaptive_avg_pool1d` + `interpolate` 动态转换，该转换与 GLIM 原始训练数据的分布可能存在差异，评估结果仅供参考。

### BERTScore 离线降级

在无法访问 HuggingFace Hub 的环境中，BERTScore 计算会失败并返回 `NaN`（而非错误的 0.0）。其他指标（BLEU、ROUGE、WER）不受影响。

---

**注意**：本项目仍在持续开发中，API 可能会有变动。请关注更新日志以获取最新信息。
