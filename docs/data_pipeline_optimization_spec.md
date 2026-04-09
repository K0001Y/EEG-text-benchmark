# 数据处理流程优化规格书

> 基于四个 EEG-to-Text 模型的输入需求调研，指导统一数据处理流程的优化改造，并规划随机噪声测试与检索测试方案。

---

## 一、模型输入需求对比

### 1.1 EEG 表征对比总览

| 维度 | EEG-To-Text | EEG2Text | CET-MAE | GLIM |
|------|-------------|----------|---------|------|
| **EEG 表征层级** | 词级频域特征 | 句级频谱（spectrogram） | 词级频域特征 | 词级频域（经维度转换） |
| **原始输入 shape** | (B, 56, 840) | (B, 374, 65) | (B, 58, 840) | (B, 1280, 128) |
| **特征来源** | 8频段×105维 word_level_EEG | rawData → spectrogram(fs=500, nperseg=128, noverlap=64) | 8频段×105维 + 句级EEG追加 | 词级840维 → pool到128维 → 插值到1280步 |
| **归一化方式** | 逐词 1D z-score | 逐句 1D z-score | 词级1D + 句级2D 全局z-score | 无显式归一化（wrapper内转换） |
| **句级 EEG** | 不需要 | 不需要 | **需要**（追加到词序列末尾） | 不需要 |
| **Mask 格式** | (B, 56)，1=有效 | (B, 374)，1=有效 | (B, 58)，含句级token | (B, 1280)，1=有效 |
| **语言模型** | BART-large | BART | BART-large | T5-large (flan-t5-large) |
| **生成参数** | beam=5, sample=True, rep_penalty=5.0 | beam=1 (greedy) | beam=1 (greedy) | beam=2 |
| **max_length** | 56 | 64 | 64 | 64 |

### 1.2 关键发现

**数据层面：**
- EEG-To-Text 和 CET-MAE 共享相同的词级频域特征基础（8频段×105维=840维），区别在于 CET-MAE 额外需要句级EEG 并使用双层归一化
- EEG2Text 使用完全不同的表征路径：从 rawData 计算语音频谱 (374, 65)，不使用词级特征
- GLIM 原始训练使用 (1280, 128) 格式，当前 wrapper 从 (56, 840) 动态转换，这一转换的信息保真度存疑

**生成层面：**
- 三个模型使用 BART 系列，仅 GLIM 使用 T5，这意味着生成参数不可能完全统一
- 各模型原始论文的生成参数差异显著（beam=1~5），强行统一可能导致部分模型性能偏离其最优表现

---

## 二、当前数据处理流程分析

### 2.1 现有 unified_zuco.pkl 存储的字段

| 字段 | Shape | 用途 | 服务模型 |
|------|-------|------|---------|
| `eeg` / `eeg_normalized_1d` | (56, 840) | 词级1D归一化 | EEG-To-Text |
| `eeg_raw` | (56, 840) | 原始词级特征（未归一化） | 备用基础 |
| `eeg_normalized_2d` | (56, 840) | 词+句全局2D归一化 | CET-MAE |
| `eeg_eeg2text` | (24000, 105) | 原始时序信号 | EEG2Text |
| `sent_eeg_raw` | (840,) | 句级EEG特征 | CET-MAE |
| `mask` | (56,) | 词级mask | EEG-To-Text, GLIM |
| `mask_with_sent` | (56,) | 含句级token的mask | CET-MAE |
| `mask_eeg2text` | (24000,) | 时序mask | EEG2Text |

### 2.2 当前存在的问题

**P1（Critical）：EEG2Text 实际需要频谱数据 (374, 65)，而非原始时序 (24000, 105)**
- 当前 `eeg_eeg2text` 字段存储的是 rawData 原始时序信号
- EEG2Text 模型的 ShallowNet 期望输入形状为经过 spectrogram 转换后的数据
- 当前 wrapper 中 model_decoding_pretrain.py 的 ShallowNet 第一层 Conv2d(1, 40, (1, 26)) 暗示输入不是 (24000, 105) 的裸信号
- 需要确认：当前 wrapper 是直接传入 raw 还是做了 spectrogram 转换？如果没有做转换，评估结果无效

**P2（High）：GLIM 维度转换逻辑的正确性未验证**
- 当前 wrapper 将 (B, 56, 840) → (B, 1280, 128)，使用 adaptive_avg_pool1d + interpolate
- 原始 GLIM 训练数据是从独立 pipeline 生成的 (1280, 128)，两者的数据分布可能完全不同
- 理想方案：在 build_unified_dataset.py 中直接生成 GLIM 格式数据，避免 wrapper 层的有损转换

**P3（High）：max_len 不一致**
- build_unified_dataset.py 默认 max_len=56
- eval_config.yaml 定义 max_len=58
- CET-MAE 原始训练使用 max_len=58（含句级token）
- 需明确：统一数据集的 max_len 应为 56（词级基准），CET-MAE 的 +1 句级追加在 wrapper 层处理

**P4（Medium）：`eeg_eeg2text` 字段占用大量存储空间**
- 每个样本 (24000, 105) ≈ 9.6MB float32
- 整个数据集可能因此膨胀到数十GB
- 如果改为存储 spectrogram (374, 65)，每样本仅 ≈ 97KB，节省 99%

---

## 三、数据处理优化方案

### 3.1 设计原则

1. **数据层只存储"源表征"**：存储各模型需要的最接近原始训练格式的数据，避免 wrapper 层的有损转换
2. **归一化分离**：raw 数据和归一化数据分开存储，归一化策略由配置控制
3. **Mask 统一**：所有 mask 使用相同语义（1=有效, 0=padding），特殊 mask（如 CET-MAE 的 mask_with_sent）作为附加字段
4. **可扩展性**：数据结构支持新增模型格式而不影响现有字段

### 3.2 优化后的统一数据集字段设计

#### 基础层（所有模型共享）
| 字段 | Shape | 说明 |
|------|-------|------|
| `input_text` | str | 目标文本 |
| `meta` | dict | 元数据（task, subject, dataset, text_uid, seq_len 等） |
| `phase` | str | train/val/test 划分标签 |

#### 词级特征层（EEG-To-Text, CET-MAE, GLIM 共用基础）
| 字段 | Shape | 说明 |
|------|-------|------|
| `eeg_word_raw` | (max_len, 840) | 词级 EEG 原始特征（8频段×105维），未归一化 |
| `eeg_word_norm1d` | (max_len, 840) | 逐词 1D z-score 归一化 |
| `eeg_word_norm2d` | (max_len, 840) | 全局 2D z-score 归一化（含句级EEG参与归一化） |
| `mask_word` | (max_len,) | 词级有效性mask，1=有效 |
| `sent_eeg_raw` | (840,) | 句级 EEG 特征（CET-MAE 需要追加到词序列末尾） |
| `mask_word_with_sent` | (max_len,) | 含句级token的mask（CET-MAE 用） |

#### 频谱层（EEG2Text 专用）
| 字段 | Shape | 说明 |
|------|-------|------|
| `eeg_spectro` | (374, 65) | spectrogram 格式，由 rawData 经 scipy.signal.spectrogram 转换 |
| `eeg_spectro_norm` | (374, 65) | spectrogram 的 1D z-score 归一化 |
| `mask_spectro` | (374,) | 频谱有效性mask |

#### GLIM 专用层（可选：如果决定预计算）
| 字段 | Shape | 说明 |
|------|-------|------|
| `eeg_glim` | (1280, 128) | GLIM 原始格式，从词级特征转换 |
| `mask_glim` | (1280,) | GLIM mask |

> **设计抉择**：GLIM 格式是在数据层预计算还是在 wrapper 层动态转换？
> - **方案 A（推荐）**：数据层预计算 — 保证转换一致性，可提前验证，便于对比调试
> - **方案 B**：wrapper 层转换 — 节省存储，但每次推理都要转换，且难以保证与原始训练一致

### 3.3 命名规范

统一字段命名模式：`eeg_{表征类型}_{处理方式}`
- 表征类型：`word`（词级）、`spectro`（频谱）、`glim`（GLIM专用）
- 处理方式：`raw`（未归一化）、`norm1d`（逐词/逐句1D归一化）、`norm2d`（全局2D归一化）

### 3.4 build_unified_dataset.py 改造要点

1. **新增 spectrogram 计算函数**
   - 从 rawData 计算 spectrogram：`scipy.signal.spectrogram(signal, fs=500, nperseg=128, noverlap=64)`
   - 参数与 EEG2Text 原始 `data_spectro.py` 完全对齐
   - 替代当前直接存储 rawData 的方案，大幅减小 pkl 体积

2. **新增 GLIM 格式预计算（可选）**
   - 从词级 (max_len, 840) 转换为 (1280, 128)
   - 转换逻辑从 glim_wrapper.py 提取到数据层
   - 确保转换参数可配置且与原始 GLIM 训练一致

3. **统一 max_len 处理**
   - 基准 max_len=56（词级）
   - CET-MAE 的句级追加在构建时处理：eeg_word_norm2d 的归一化包含句级 EEG，但序列长度仍为 56
   - 句级 EEG 作为独立字段 `sent_eeg_raw` 存储，由 wrapper 决定是否追加

4. **字段重命名**
   - `eeg` → `eeg_word_norm1d`（消除歧义）
   - `eeg_raw` → `eeg_word_raw`
   - `eeg_normalized_2d` → `eeg_word_norm2d`
   - `eeg_eeg2text` → `eeg_spectro` 或 `eeg_spectro_norm`（改为频谱格式）

### 3.5 Wrapper 适配改造要点

| Wrapper | 取用字段 | 改造内容 |
|---------|---------|---------|
| EEG-To-Text | `eeg_word_norm1d`, `mask_word` | 仅字段重命名 |
| EEG2Text | `eeg_spectro_norm`, `mask_spectro` | 不再从 raw 时序计算，直接使用预计算频谱 |
| CET-MAE | `eeg_word_norm2d`, `sent_eeg_raw`, `mask_word_with_sent` | 句级EEG追加逻辑保留在wrapper |
| GLIM | `eeg_glim`, `mask_glim` 或 `eeg_word_raw`, `mask_word` | 取决于是否预计算 GLIM 格式 |

---

## 四、随机噪声测试方案

### 4.1 测试目的

验证模型是否真正利用了 EEG 信号中的语义信息，而非仅依赖语言模型先验或统计偏差。如果模型在噪声输入下的性能与真实 EEG 无显著差异，则说明模型未有效利用 EEG 信号。

### 4.2 噪声类型设计

#### 基线噪声（必须实现）

| 噪声类型 | 说明 | 实现方式 |
|---------|------|---------|
| **Gaussian 噪声** | 用标准正态分布随机信号替代 EEG | `torch.randn_like(eeg)` |
| **Uniform 噪声** | 用均匀分布随机信号替代 EEG | `torch.rand_like(eeg) * (max-min) + min` |
| **Shuffle 噪声** | 打乱 batch 内样本的 EEG-文本对应关系 | 随机置换 batch 内的 EEG 索引 |

#### 进阶噪声（推荐实现）

| 噪声类型 | 说明 | 实现方式 |
|---------|------|---------|
| **Zero 输入** | 全零 EEG 输入 | `torch.zeros_like(eeg)` |
| **频段掩码** | 屏蔽特定频段（如仅保留 alpha 或 gamma） | 将目标频段列置零 |
| **时间掩码** | 屏蔽特定时间段 | 将目标时间步置零 |
| **渐进噪声** | 在真实 EEG 上叠加不同强度的高斯噪声 | `eeg + scale * torch.randn_like(eeg)`，scale ∈ {0.1, 0.5, 1.0, 2.0, 5.0} |

### 4.3 实现架构

**数据层**：在 `dataset.py` 的 `UnifiedDataset` 中实现噪声注入
```
noise_config = {
    "mode": "gaussian" | "uniform" | "shuffle" | "zero" | "band_mask" | "progressive",
    "seed": 42,           # 固定随机种子，确保跨模型一致
    "scale": 1.0,         # 渐进噪声强度
    "target_bands": [...] # 频段掩码的目标频段索引
}
```

**关键约束**：
1. **同一噪声用于所有模型**：使用相同的 seed 生成噪声，确保对比公平性
2. **噪声与数据格式匹配**：对每种 EEG 格式（word/spectro/glim）分别生成相应 shape 的噪声
3. **保持 mask 不变**：噪声只替代有效区域的 EEG 信号，padding 区域仍为 0
4. **Shuffle 噪声特殊处理**：在 DataLoader 的 batch 级别操作，不在 sample 级别

### 4.4 评估报告格式

```json
{
    "model": "EEG-To-Text",
    "noise_type": "gaussian",
    "noise_seed": 42,
    "metrics": {
        "BLEU-1": 0.02,
        "ROUGE-L": 0.01,
        "WER": 0.98
    },
    "comparison": {
        "original_BLEU-1": 0.15,
        "delta_BLEU-1": -0.13,
        "relative_drop": "-86.7%"
    }
}
```

---

## 五、检索测试方案

### 5.1 测试目的

评估模型 EEG 编码器学到的表征质量：给定一个 EEG 信号，能否在候选文本集中正确识别出对应的原始文本。这测试的是 EEG 编码与文本语义的对齐程度。

### 5.2 检索任务定义

**任务**：给定一个 EEG 编码向量，从 N 个候选文本编码向量中检索出正确匹配的文本。

**指标**：
- **R@1**：Top-1 检索准确率
- **R@5**：Top-5 检索准确率
- **R@10**：Top-10 检索准确率
- **MRR**：Mean Reciprocal Rank（平均倒数排名）

### 5.3 实现方案

#### 方案概述

```
EEG 样本 → wrapper.encode_eeg() → EEG 向量 (d_embed,)
                                         ↓
                                    余弦相似度排序
                                         ↓
候选文本 → text_encoder.encode() → 文本向量 (d_embed,)
```

#### 接口扩展

在 `model_wrappers.py` 基类中新增方法：

```python
class BenchmarkModelWrapper:
    def encode_eeg_to_embedding(self, eeg, mask, meta=None, batch=None) -> torch.Tensor:
        """
        将 EEG 输入编码为固定维度的向量表征。
        返回: (B, embed_dim) 的嵌入向量
        """
        raise NotImplementedError

    def encode_text_to_embedding(self, text: List[str]) -> torch.Tensor:
        """
        将文本编码为固定维度的向量表征。
        返回: (B, embed_dim) 的嵌入向量
        """
        raise NotImplementedError
```

#### 各模型编码提取点

| 模型 | EEG 编码提取位置 | 输出维度 | 文本编码方式 |
|------|----------------|---------|------------|
| EEG-To-Text | additional_encoder 输出 → mean pooling | (1024,) | BART encoder(text) → mean pooling |
| EEG2Text | ShallowNet + TransformerEncoder → mean pooling | (1024,) | BART encoder(text) → mean pooling |
| CET-MAE | unify_branch 输出 → mean pooling | (1024,) | BART encoder(text) → mean pooling |
| GLIM | EEG encoder + aligner → mean pooling | (1024,) | T5 encoder(text) → mean pooling |

#### 候选集构建策略

| 策略 | N (候选数) | 说明 |
|------|----------|------|
| **In-batch** | batch_size | 同一 batch 内的所有文本作为候选 |
| **Full-test** | test_set_size | 整个测试集的所有文本作为候选（更严格） |
| **Same-task** | task_sample_count | 同一 task 内的所有文本（控制主题偏差） |

推荐使用 **Full-test** 作为主要指标，**In-batch** 作为快速验证。

### 5.4 数据层支持需求

1. **meta 中的 text_uid**：当前已支持，用于确定 ground-truth 匹配
2. **文本去重**：检索候选集中每个唯一文本只出现一次
3. **embedding 缓存**：对整个测试集预计算 EEG/text embedding，避免重复编码

### 5.5 评估报告格式

```json
{
    "model": "EEG-To-Text",
    "retrieval_strategy": "full_test",
    "num_candidates": 465,
    "metrics": {
        "R@1": 0.12,
        "R@5": 0.35,
        "R@10": 0.48,
        "MRR": 0.22
    },
    "by_task": {
        "task1-SR": {"R@1": 0.15, ...},
        "task2-NR": {"R@1": 0.08, ...}
    }
}
```

---

## 六、实施路线图

| 阶段 | 任务 | 依赖 | 状态 |
|------|------|------|------|
| **Phase 1** | 字段重命名 + 新增 spectrogram 计算 + 移除 raw 时序存储 | 无 | ✅ 已完成 |
| **Phase 2** | 各 wrapper 适配新字段名 + 验证数据流 | Phase 1 | ✅ 已完成 |
| **Phase 3** | GLIM 格式预计算（可选） + 验证维度转换正确性 | Phase 1 | ⏭️ 跳过（保留 wrapper 层转换） |
| **Phase 4** | 噪声测试框架实现 + 集成到 eval_runner | Phase 2 | 🔲 待实现 |
| **Phase 5** | 检索测试框架实现 + 各 wrapper 实现 encode_*_to_embedding | Phase 2 | 🔲 待实现 |
| **Phase 6** | 全模型端到端验证 + 结果对比 | Phase 4, 5 | 🔲 待实现 |

---

## 七、实施进度记录

### 7.1 Phase 1 + Phase 2 实施概览

Phase 1（数据层）和 Phase 2（Wrapper 层）已于本次优化中完整实施。下表记录实际实现的字段映射和关键决策。

### 7.2 实际字段映射（v1 → v2）

| 旧字段名（v1） | 新字段名（v2） | Shape | 服务模型 | 变化说明 |
|-------------|-------------|-------|---------|---------|
| `eeg` / `eeg_normalized_1d` | `eeg_word_norm1d` | (max_len, 840) | EEG-To-Text | 仅重命名，保留别名 `eeg` 向后兼容 |
| `eeg_raw` | `eeg_word_raw` | (max_len, 840) | GLIM（via wrapper转换） | 仅重命名 |
| `eeg_normalized_2d` | `eeg_word_norm2d` | (max_len, 840) | CET-MAE | 仅重命名 |
| `eeg_eeg2text` | `eeg_spectro` | **(374, 65)** | EEG2Text | **格式变更**：原始时序 (24000, 105) → scipy spectrogram (374, 65) |
| `mask` | `mask_word` | (max_len,) | EEG-To-Text, GLIM | 仅重命名，保留别名 `mask` 向后兼容 |
| `mask_with_sent` | `mask_word_with_sent` | (max_len,) | CET-MAE | 仅重命名 |
| `mask_eeg2text` | `mask_spectro` | **(374,)** | EEG2Text | 长度从 24000 改为 374 |

### 7.3 新增常量模块（constants.py）

所有魔法数字统一收归 `benchmark_eval/constants.py`：

```python
MAX_LEN = 56          # 词级 EEG 最大长度（基准，统一 eval_config.yaml）
EEG_CHANNELS = 105    # EEG 通道数
EEG_BANDS = 8         # 频段数
EEG_WORD_DIM = 840    # 词级 EEG 特征维度 (105 × 8)
SPECTRO_STEPS = 374   # spectrogram 时间步
SPECTRO_FREQS = 65    # spectrogram 频率维度
RAW_SAMPLING_RATE = 500   # 采样率 (Hz)
SPECTRO_NPERSEG = 128     # spectrogram 窗口大小
SPECTRO_NOVERLAP = 64     # spectrogram 重叠大小
GLIM_EEG_LEN = 1280   # GLIM 输入序列长度
GLIM_EEG_DIM = 128    # GLIM 输入特征维度
DEFAULT_SEED = 42     # 默认全局随机种子
```

### 7.4 关键问题修复状态

| 问题编号 | 描述 | 修复方式 | 状态 |
|---------|------|---------|------|
| P1 | EEG2Text 实际需要频谱 (374, 65)，原存 rawData (24000, 105) | `build_unified_dataset.py` 新增 `build_spectrogram()` 函数，改存 `eeg_spectro` | ✅ 已修复 |
| P2 | GLIM 维度转换逻辑未验证 | 维持 wrapper 层动态转换（Phase 3 跳过）；已移除废弃注释代码，转换逻辑保留 | ⚠️ 部分缓解 |
| P3 | max_len 不一致（build=56, config=58） | `eval_config.yaml` 改为 `max_len: 56`，`constants.py` 定义 `MAX_LEN=56` | ✅ 已修复 |
| P4 | `eeg_eeg2text` 占用存储过大（每样本约 9.6MB） | 改存 spectrogram (374, 65)，每样本约 97KB，节省 **99%** 存储 | ✅ 已修复 |

### 7.5 其他优化

| 优化项 | 实现文件 | 说明 |
|-------|---------|------|
| BERTScore 离线降级 | `metrics.py` | 捕获 `OSError/ValueError`，失败返回 `float('nan')` 而非 0.0 |
| 全局随机种子 | `eval_runner.py` | `set_seed()` 覆盖 random/numpy/torch/cuda/cudnn |
| DataLoader 种子固定 | `eval_runner.py` | `torch.Generator().manual_seed(seed)` 传入 DataLoader |
| 分组指标含样本数 | `metrics.py` | 每组结果附加 `"sample_count"` 字段 |
| 统一输出 schema | `eval_runner.py` | 所有模型输出包含 `"overall"` / `"grouped"` / `"failed_count"` / `"num_samples"` |
| 各模型独立生成参数 | `eval_config.yaml` | `generation.model_overrides` 区段，避免强行统一 beam size |
| 向后兼容 | `dataset.py`, 所有 wrappers | 旧 PKL 文件（v1 字段名）仍可加载，自动 fallback |

### 7.6 重要注意事项

> **数据重建要求**：Phase 1 的字段重命名和 spectrogram 格式变更只对**新生成的** `unified_zuco.pkl` 生效。若使用旧版 PKL 文件，`dataset.py` 中的向后兼容逻辑会自动回退到旧字段名，但 `eeg_spectro` 字段需要重新运行 `build_unified_dataset.py` 才能得到正确格式。

---

*生成时间：2026-04-09*
*更新时间：2026-04-10（Phase 1 + Phase 2 实施完成）*
*基于 models/ 目录四个模型的源码调研 + benchmark_eval/ 现有实现分析*
