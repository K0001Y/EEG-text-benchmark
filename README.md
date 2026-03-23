# Unified EEG-to-Text Benchmark

一个用于 EEG-to-Text 模型公平评估的统一 benchmark 框架，支持 EEG-To-Text、DeWave、EEG2Text、GLIM 等多个模型的标准化对比。

## 项目概述

本项目旨在为多个 EEG-to-Text 模型提供公平、可复现的评估基准。通过统一的数据预处理流程、标准化的模型接口和一致的评估指标，确保不同模型之间的比较具有实际意义。

### 核心特性

- **统一数据格式**：将 ZuCo v1.0/v2.0 的原始 `.mat` 数据转换为标准化的中间格式
- **标准化模型接口**：所有模型通过统一的 wrapper 接口进行适配
- **严格自回归生成**：评估阶段禁用 teacher forcing，真实反映模型能力
- **多维度评估指标**：支持 BLEU、ROUGE、WER 等文本相似度指标
- **鲁棒性测试**：支持噪声 EEG 控制实验

## 项目结构

```
benchmark/
├── benchmark_eval/           # 核心评估框架
│   ├── data_processing/      # 数据处理模块
│   │   ├── build_unified_dataset.py  # 从 ZuCo MAT 构建统一数据集
│   │   └── dataset.py                # 数据集加载与批处理
│   ├── evaluation/           # 评估模块
│   │   ├── model_wrappers.py         # 模型 wrapper 基类
│   │   ├── eval_runner.py            # 评估流程管理
│   │   └── metrics.py                # 指标计算（BLEU、ROUGE、WER）
│   ├── wrappers/             # 模型适配器
│   │   ├── eeg_to_text_wrapper.py    # EEG-To-Text wrapper
│   │   ├── eeg2text_wrapper.py       # EEG2Text wrapper
│   │   ├── cet_mae_wrapper.py        # CET-MAE wrapper
│   │   └── glim_wrapper.py           # GLIM wrapper
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

# 构建统一数据集
build_dataset(
    zuco_root="data",
    tasks="task1-SR,task2-NR,task3-TSR",
    output="data/unified_dataset.pkl",
    max_len=58,
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
    eeg = batch["eeg"]
    mask = batch["mask"]
    meta = batch["meta"]
    
    # 生成文本
    generated_text = model.generate_text(eeg, mask, meta)
    print(f"Generated: {generated_text}")
```

## 支持的模型

| 模型 | 论文 | 特点 | Wrapper 路径 |
|------|------|------|-------------|
| EEG-To-Text | [NeurIPS 2023] | 词级 EEG 序列 + BART/T5 | `wrappers/eeg_to_text_wrapper.py` |
| EEG2Text | [ACL 2023] | 句级 raw EEG + RoBERTa | `wrappers/eeg2text_wrapper.py` |
| CET-MAE | [NeurIPS 2023] | 跨模态 MAE + Multi-Stream | `wrappers/cet_mae_wrapper.py` |
| GLIM | [NeurIPS 2024] | Prompt + T5/BART | `wrappers/glim_wrapper.py` |

## 统一数据格式

### 样本字段

```python
{
    "eeg": np.ndarray,           # (L_max, C) float32 EEG 序列
    "mask": np.ndarray,          # (L_max,) int8 mask，1 表示有效
    "input_text": str,           # 输入句子文本
    "reference_text": str,       # 参考文本（用于评估）
    "phase": str,                # "train" / "val" / "test"
    "meta": {
        "task": str,             # 任务类型
        "dataset": str,          # "ZuCo1" 或 "ZuCo2"
        "subject": str,          # 被试 ID
        "text_uid": int,         # 文本唯一 ID
        "sentiment_label": int,  # 情感标签（task1）
        "relation_label": str,   # 关系标签（task2/task3）
    }
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
- **跨被试泛化**：测试模型对新被试的适应能力

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

---

**注意**：本项目仍在持续开发中，API 可能会有变动。请关注更新日志以获取最新信息。
