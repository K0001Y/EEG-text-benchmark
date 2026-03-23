# EEG-to-Text 模型 Wrappers

本目录包含了四个 EEG-to-Text 模型的 wrapper 实现，用于统一 benchmark 评估。

## 设计原则

所有 wrapper 都继承自 `BenchmarkModelWrapper` 抽象基类，实现以下接口：

1. **`encode_eeg(eeg, mask, meta)`**: 将统一格式的 EEG 编码成模型内部表示
2. **`generate_text(eeg, mask, meta)`**: 从 EEG 生成文本（纯自回归，禁用 teacher forcing）

### 统一输入格式

- **EEG**: `(B, L_max, C=840)` - 批次大小 × 最大序列长度 × 通道数
- **Mask**: `(B, L_max)` - 1 表示有效，0 表示 padding
- **Meta**: `List[Dict[str, Any]]` - 包含 task、subject、dataset 等元信息

### 统一输出格式

- **Generated Texts**: `List[str]` - 生成的文本列表，长度为 batch_size

## 已实现的模型 Wrappers

### 1. EEGToTextWrapper (`eeg_to_text_wrapper.py`)

**模型架构**: EEG-To-Text (BrainTranslator/T5Translator)

**特点**:
- 支持 BART 和 T5 两种后端模型
- 直接使用统一格式 `(B, L_max, 840)` 作为输入，无需额外转换
- 使用 HuggingFace 的 `generate()` 方法进行自回归生成

**关键参数**:
- `model_checkpoint`: 模型权重路径
- `model_type`: "bart" 或 "t5"
- `max_new_tokens`: 生成的最大 token 数（默认 64）
- `num_beams`: beam search 大小（默认 1，即 greedy decoding）

**使用示例**:
```python
from benchmark_eval.wrappers import EEGToTextWrapper

wrapper = EEGToTextWrapper(
    model_checkpoint="path/to/checkpoint.pt",
    model_type="bart",
    device="cuda",
    max_new_tokens=64,
    num_beams=1
)

texts = wrapper.generate_text(eeg, mask, meta)
```

---

### 2. EEG2TextWrapper (`eeg2text_wrapper.py`)

**模型架构**: EEG2Text

**特点**:
- 需要将统一格式 `(B, L_max, 840)` 转换为 raw EEG 格式 `(B, 24000, 105)`
- 转换策略：
  1. 将 8 个频段平均，得到 `(B, L_max, 105)`
  2. 使用线性插值将时间维度从 `L_max` 扩展到 `24000`
- 使用 RoBERTa 作为 encoder，BART 作为 decoder

**关键参数**:
- `model_checkpoint`: 模型权重路径
- `max_spectro_datapoint`: raw EEG 的时间点数（默认 24000）
- `max_new_tokens`: 生成的最大 token 数（默认 64）
- `num_beams`: beam search 大小（默认 1）

**使用示例**:
```python
from benchmark_eval.wrappers import EEG2TextWrapper

wrapper = EEG2TextWrapper(
    model_checkpoint="path/to/checkpoint.pt",
    device="cuda",
    max_spectro_datapoint=24000,
    max_new_tokens=64
)

texts = wrapper.generate_text(eeg, mask, meta)
```

---

### 3. CETMAEWrapper (`cet_mae_wrapper.py`)

**模型架构**: CET-MAE (Cross-modal EEG-Text Masked Autoencoder)

**特点**:
- 使用 Multi-Stream Transformer Encoder 进行跨模态编码
- EEG 分支: TransformerEncoder (840 -> 840) -> Linear (840 -> 1024)
- 统一分支: Multi_Stream_TransformerEncoder (1024)
- 使用 BART decoder 生成文本

**关键参数**:
- `model_checkpoint`: 模型权重路径
- `pretrain_path`: BART 预训练模型路径（默认 "./models/huggingface/bart-large"）
- `max_new_tokens`: 生成的最大 token 数（默认 64）
- `num_beams`: beam search 大小（默认 1）

**使用示例**:
```python
from benchmark_eval.wrappers import CETMAEWrapper

wrapper = CETMAEWrapper(
    model_checkpoint="path/to/checkpoint.pt",
    pretrain_path="./models/huggingface/bart-large",
    device="cuda",
    max_new_tokens=64
)

texts = wrapper.generate_text(eeg, mask, meta)
```

---

### 4. GLIMWrapper (`glim_wrapper.py`)

**模型架构**: GLIM (Grounded Language-Interfaced Model)

**特点**:
- 需要将统一格式 `(B, L_max, 840)` 转换为 GLIM 格式 `(B, 1280, 128)`
- 转换策略：
  1. 通道维度压缩：840 -> 128（分组平均）
  2. 时间维度插值：L_max -> 1280（线性插值）
- 支持 prompt embedding（task、dataset、subject）
- 使用 T5/BART 作为文本生成器

**关键参数**:
- `model_checkpoint`: 模型权重路径
- `text_model_id`: T5/BART 模型 ID（默认 "google/flan-t5-large"）
- `input_eeg_len`: 输入 EEG 长度（默认 1280）
- `input_dim`: 输入维度（默认 128）
- `max_new_tokens`: 生成的最大 token 数（默认 64）
- `num_beams`: beam search 大小（默认 2）

**使用示例**:
```python
from benchmark_eval.wrappers import GLIMWrapper

wrapper = GLIMWrapper(
    model_checkpoint="path/to/checkpoint.pt",
    text_model_id="google/flan-t5-large",
    device="cuda",
    max_new_tokens=64,
    num_beams=2
)

texts = wrapper.generate_text(eeg, mask, meta)
```

---

## 数据格式转换对比

| 模型 | 统一格式 | 模型内部格式 | 转换方法 |
|------|---------|-------------|---------|
| EEG-To-Text | (B, L_max, 840) | (B, L_max, 840) | 无需转换 |
| EEG2Text | (B, L_max, 840) | (B, 24000, 105) | 频段平均 + 线性插值 |
| CET-MAE | (B, L_max, 840) | (B, L_max, 840) -> (B, L_max, 1024) | Linear 投影 |
| GLIM | (B, L_max, 840) | (B, 1280, 128) | 分组平均 + 线性插值 |

## 生成策略

所有 wrapper 均使用**纯自回归生成**，严格禁用 teacher forcing：

- 使用 HuggingFace 的 `model.generate()` 方法
- 默认使用 greedy decoding（`num_beams=1`, `do_sample=False`）
- 可配置 beam search（`num_beams > 1`）
- 支持早停（`early_stopping=True`）

## Meta 信息使用

Meta 字典可能包含的字段：

- `task`: 任务类型（如 "task1-SR", "task2-NR", "task3-TSR"）
- `dataset`: 数据集名称（"ZuCo1" 或 "ZuCo2"）
- `subject`: 被试 ID（如 "ZAB", "YAC" 等）
- `text_uid`: 文本唯一标识
- `sentiment_label`: 情感标签（仅 task1）
- `relation_label`: 关系标签（仅 task2 和 task3）

不同模型对 meta 信息的利用：

- **EEG-To-Text**: 不使用 meta 信息
- **EEG2Text**: 不使用 meta 信息
- **CET-MAE**: 不使用 meta 信息
- **GLIM**: 使用 task、dataset、subject 作为 prompt embedding

## 注意事项

1. **模型路径**: 所有 wrapper 都使用动态 import 加载原始模型代码，需要确保模型代码在 `models/` 目录下
2. **设备管理**: 默认使用 CUDA，如果不可用则回退到 CPU
3. **预训练模型**: 某些模型（如 CET-MAE、GLIM）需要预训练的 BART/T5 模型，确保路径正确
4. **内存占用**: GLIM 和 CET-MAE 的模型较大，需要足够的 GPU 内存

## 扩展新模型

要添加新的模型 wrapper：

1. 创建新的 wrapper 文件，继承 `BenchmarkModelWrapper`
2. 实现 `encode_eeg()` 和 `generate_text()` 方法
3. 在 `__init__.py` 中导出新的 wrapper
4. 更新本 README 文档

## 测试

建议为每个 wrapper 编写单元测试，验证：

1. 输入输出格式的正确性
2. 自回归生成的正确性（禁用 teacher forcing）
3. 不同 batch size 的兼容性
4. CPU 和 GPU 模式的兼容性
