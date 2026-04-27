# 统一数据集文件说明

> **代码文件**：
> - `benchmark_eval/data_processing/build_unified_dataset.py` — 构建脚本
> - `benchmark_eval/data_processing/dataset.py` — 加载接口
> - `benchmark_eval/constants.py` — 全局常量

---

## 一、概述

统一数据集文件（`unified_zuco.pkl`）是整个 benchmark 的数据底座。它将 ZuCo 多任务、多被试的原始 `.mat` 文件转换为统一格式的 Python `List[Dict]`，通过 `pickle` 序列化保存。所有模型（EEG-To-Text、EEG2Text、CET-MAE、GLIM）均通过 `UnifiedDataset` 加载同一文件，保证评估公平性。

### 数据来源

| 数据集版本 | 任务 | 被试数 | 格式 |
|-----------|------|--------|------|
| ZuCo 1 | task1-SR（情感阅读） | ~12 人 | v1 `.mat`（`scipy.io.loadmat`）|
| ZuCo 1 | task2-NR（自然阅读） | ~12 人 | v1 `.mat` |
| ZuCo 1 | task3-TSR（任务型阅读）| ~12 人 | v1 `.mat` |
| ZuCo 2 | task2-NR-2.0 | ~18 人 | v2 HDF5 `.mat`（`h5py`）|

### 全局常量（`constants.py`）

| 常量 | 值 | 含义 |
|------|----|------|
| `MAX_LEN` | 56 | 词级 EEG 序列最大长度 |
| `EEG_CHANNELS` | 105 | 每频带通道数（ZuCo 标准 105 电极）|
| `EEG_BANDS` | 8 | 频带数（theta1/2, alpha1/2, beta1/2, gamma1/2）|
| `EEG_WORD_DIM` | 840 | 词级特征维度（105 × 8）|
| `RAW_SAMPLING_RATE` | 500 | 原始 EEG 采样率（Hz）|
| `SPECTRO_NPERSEG` | 128 | spectrogram 窗口大小 |
| `SPECTRO_NOVERLAP` | 64 | spectrogram 重叠大小 |
| `SPECTRO_STEPS` | 374 | spectrogram 时间步数 |
| `SPECTRO_FREQS` | 65 | spectrogram 频率维度（nperseg//2 + 1）|
| `DEFAULT_SEED` | 42 | 全局随机种子 |

---

## 二、构建流程

### 2.1 运行命令

```bash
python benchmark_eval/data_processing/build_unified_dataset.py \
    --zuco-root models/EEG-To-Text-main/dataset/ZuCo \
    --tasks task1-SR,task2-NR,task3-TSR,task2-NR-2.0 \
    --output benchmark_eval/data/unified_zuco.pkl
```

### 2.2 流程总览

```
ZuCo MAT 文件
    │
    ├── v1 tasks (task1-SR / task2-NR / task3-TSR)
    │       └── load_dataset_from_mat_v1()   ← scipy.io.loadmat
    │
    └── v2 task (task2-NR-2.0)
            └── load_dataset_from_mat_v2()   ← h5py
                        │
                        ▼
            build_samples_for_task()     ← 构造每条样本，生成所有 EEG 字段
                        │
                        ▼
            文本级 80/10/10 划分         ← 按唯一句子文本划分 phase
                        │
                        ▼
            _enrich_samples_with_metadata_and_labels()  ← 补充标签与 text_uid
                        │
                        ▼
            pickle.dump(all_samples)     ← 保存为 unified_zuco.pkl
```

### 2.3 MAT 文件解析

#### ZuCo v1（`load_dataset_from_mat_v1`）

- 输入：`{zuco_root}/{task_name}/Matlab_files/*.mat`
- 使用 `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` 加载
- 提取 `sentenceData`，遍历句子与词
- 词级 EEG 类型：`FFD`（首次注视）、`TRT`（总注视时长）、`GD`（注视持续）
- **只保留 `nFixations > 0` 的词**（有眼动注视的词才有 EEG 数据）
- 同时保留 `rawData`（原始时序，用于 spectrogram 计算）

#### ZuCo v2（`load_dataset_from_mat_v2`）

- 输入：`{zuco_root}/task2-NR-2.0/Matlab_files/*NR.mat`
- 使用 `h5py` 解析 HDF5 格式
- 跳过被试 `YMH`（数据异常）
- 通过 `data_loading_helpers_modified.py` 提取词级数据

### 2.4 样本构造（`build_samples_for_task`）

每条有效的（被试, 句子）对生成一条样本 `record`，步骤如下：

**步骤 1：收集词级原始 EEG**

从 `word_level_EEG["GD"]` 读取 8 个频带的特征，拼接为 840 维向量：

```
bands = ["_t1","_t2","_a1","_a2","_b1","_b2","_g1","_g2"]
raw_vec = concat([word["GD_t1"][0:105], word["GD_t2"][0:105], ..., word["GD_g2"][0:105]])
shape: (840,)
```

若任意词的特征缺失或维度异常，**丢弃整句**。

**步骤 2：获取句级 EEG**

从 `sentence_level_EEG` 读取 8 个频带的均值向量 `mean_{band}[0:105]`，拼接为 840 维。
若包含 NaN，**丢弃整句**。

**步骤 3：词序列截断**

若词数超过 `MAX_LEN=56`，截断到前 56 个词。

**步骤 4：生成各 EEG 字段**（详见字段说明）

**步骤 4.5：收集词级注视次数（`nfixations_word`）**

对每个有效词记录 `word.nFixations`，padding 位置填 0，生成 shape `(MAX_LEN,)` 的浮点数组：

```python
nfixations = [float(word.nFixations) for word in valid_words[:max_len]]
nfixations += [0.0] * (max_len - len(nfixations))
record["nfixations_word"] = np.array(nfixations, dtype=np.float32)  # (56,)
```

该字段供 A1b Duration-Weighted Pool 计算注视时长权重：

$$w_{i,t} = \frac{\text{nfixations\_word}_{i,t}}{\sum_{t'=1}^{T_i} \text{nfixations\_word}_{i,t'}}$$

> **注**：ZuCo v1 有效词均已过滤 `nFixations > 0`，因此有效位的值均 ≥ 1；padding 位为 0，计算权重时需用 mask 截取有效长度 $T_i$。

**步骤 5：构造 spectrogram（可选）**

从 `rawData`（句子原始时序，shape `(105, T)`）计算 spectrogram：
- `scipy.signal.spectrogram(signal_1d, fs=500, nperseg=128, noverlap=64)`
- 对 105 通道平均，裁剪/padding 到 `(374, 65)`
- 逐样本 z-score 归一化
- 若 `rawData` 缺失则不生成此字段

### 2.5 训练/验证/测试划分

**策略：文本级 80/10/10 划分**，保证同一句子的所有被试样本都在同一 phase，避免数据泄露。

```python
random.seed(42)
np.random.seed(42)

unique_texts = sorted(text_to_records.keys())  # 按字典序固定顺序
n_total = len(unique_texts)

n_train = max(int(n_total * 0.8), 1)
n_val   = max(int(n_total * 0.1), 0)
# 若 n_train + n_val >= n_total，则自动缩减以保证 test 非空

train_texts = set(unique_texts[:n_train])
val_texts   = set(unique_texts[n_train : n_train + n_val])
test_texts  = set(unique_texts[n_train + n_val:])
```

**划分后的典型数量（各任务独立划分，最终合并）：**

| Phase | 唯一句子数 | 样本数（含多被试） |
|-------|-----------|------------------|
| train | ~880 | ~14903 |
| val   | ~110 | ~1863 |
| test  | ~130 | ~1858 |

> **注意**：train 与 test 的句子集合**完全不重叠**。两者的 `sentence_id` 由 `UnifiedDataset` 各自独立分配，不可跨 phase 比较。

### 2.6 标签与元数据补充（`_enrich_samples_with_metadata_and_labels`）

- **`meta.dataset`**：`"ZuCo1"` 或 `"ZuCo2"`
- **`meta.text_uid`**：全局唯一文本 ID（`(dataset, task, sentence)` 三元组的递增整数映射）
- **`meta.session`**：样本所属的实验 session（供跨 session 诊断实验使用，见下文 Session 标注规则）
- **`meta.sentiment_label`**（仅 task1-SR）：从 `data/ZuCo1/task_materials/sentiment_labels_task1.csv` 加载
- **`meta.relation_label`**（task2-NR / task3-TSR）：从对应的 `relations_labels_*.csv` 加载

#### Session 标注规则

根据 ZuCo 原始实验设计（Hollenstein et al., 2018），每位被试在两个实验阶段完成所有任务：

- **Session 1**：`task2-NR`（全部）+ `task1-SR` 前半部分
- **Session 2**：`task3-TSR`（全部）+ `task1-SR` 后半部分
- **ZuCo v2 (`task2-NR-2.0`)**：单独记录，暂标为 `"session_unknown"`

句子呈现顺序对所有被试完全一致，故可根据 `meta.task` 和 `meta.sentence_index` 在构建阶段统一判定（见 [`_assign_session`](file:///root/autodl-tmp/benchmark/benchmark_eval/data_processing/build_unified_dataset.py)）：

| 任务 | Session 1 | Session 2 | 判定条件 |
|------|-----------|-----------|----------|
| `task1-SR` | 前半部分 | 后半部分 | `sentence_index < N_task1 // 2` → Session 1；否则 Session 2 |
| `task2-NR` | 全部 | — | 均为 Session 1 |
| `task3-TSR` | — | 全部 | 均为 Session 2 |
| `task2-NR-2.0` | — | — | `session_unknown` |

其中 $N_\text{task1}$ 为所有样本中 `task1-SR` 的最大 `sentence_index + 1`（约 400）。构建时会输出日志：

```
Session split for task1-SR uses N_task1=<N> (first half -> session_1, second half -> session_2)
Session distribution over all samples: {'session_1': ..., 'session_2': ..., 'session_unknown': ...}
```

> **Task 与 Session 的混淆**：task2-NR 仅在 Session 1、task3-TSR 仅在 Session 2，因此跨 session 诊断需在 `task1-SR` 内部单独对照，以分离真正的 session 效应。此规则与 [`experiment_A_details.md`](file:///root/autodl-tmp/benchmark/docs/detail/experiment_A_details.md) 步骤 4 完全一致。

---

## 三、数据结构

### 3.1 文件格式

```
unified_zuco.pkl
└── List[Dict[str, Any]]   # 每个元素为一条样本（一个被试阅读一个句子）
    总长度：约 18,624 条
```

### 3.2 单条样本字段详表

| 字段名 | 类型 | Shape | 含义 | 用于哪个模型 |
|--------|------|-------|------|------------|
| `eeg_word_raw` | `np.ndarray[float32]` | `(56, 840)` | 词级原始 EEG，未归一化，padding 为 0 | 原始分析 |
| `eeg_word_norm1d` | `np.ndarray[float32]` | `(56, 840)` | 逐词 1D z-score 归一化 | EEG-To-Text |
| `eeg_word_norm2d` | `np.ndarray[float32]` | `(56, 840)` | 词+句全局 2D z-score 归一化 | CET-MAE |
| `sent_eeg_raw` | `np.ndarray[float32]` | `(840,)` | 句级 EEG（`sentence_level_EEG.mean_*` 拼接），未归一化 | CET-MAE（句级 token）|
| `eeg_spectro` | `np.ndarray[float32]` | `(374, 65)` | Spectrogram（可能缺失） | EEG2Text |
| `mask_word` | `List[float]` | `(56,)` | 词级有效 mask（1=有效，0=padding）| EEG-To-Text、诊断实验 |
| `mask_word_with_sent` | `List[float]` | `(56,)` | 含句级 token 的 mask | CET-MAE |
| `mask_spectro` | `np.ndarray[float32]` | `(374,)` | Spectrogram mask（全 1）| EEG2Text |
| `nfixations_word` | `np.ndarray[float32]` | `(56,)` | 每词注视次数（`nFixations`），padding 位为 0 | 诊断实验 A1b |
| `eeg` | `np.ndarray[float32]` | `(56, 840)` | `eeg_word_norm1d` 的别名（向后兼容）| 旧代码 |
| `mask` | `List[float]` | `(56,)` | `mask_word` 的别名（向后兼容）| 旧代码 |
| `input_text` | `str` | — | 句子文本（模型输入，与 `reference_text` 相同）| 所有模型 |
| `reference_text` | `str` | — | 句子文本（评估参考） | 所有模型 |
| `phase` | `str` | — | `"train"` / `"val"` / `"test"` | 数据加载过滤 |
| `meta` | `Dict` | — | 元数据（见下表）| 诊断分析 |

#### meta 子字段

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `meta.task` | `str` | 任务名，如 `"task1-SR"`、`"task2-NR-2.0"` |
| `meta.subject` | `str` | 被试 ID，如 `"ZAB"`、`"ZPH"` |
| `meta.sentence_index` | `int` | 在原始 MAT 文件中的句子序号 |
| `meta.source` | `str` | `"ZuCo-MAT"` |
| `meta.seq_len` | `int` | 有效词数（不含 padding，不含句级 token）|
| `meta.seq_len_with_sent` | `int` | 含句级 token 的有效长度 |
| `meta.dataset` | `str` | `"ZuCo1"` 或 `"ZuCo2"` |
| `meta.text_uid` | `int` | 全局唯一文本整数 ID |
| `meta.session` | `str` | 样本所属 session：`"session_1"` / `"session_2"` / `"session_unknown"`（ZuCo v2）；用于诊断线 A 的 A1d 与 A3-SessionRetrieval 等跨 session 实验 |
| `meta.sentiment_label` | `int` | 情感标签（仅 task1-SR，可能缺失）|
| `meta.relation_label` | `str` | 关系标签（仅 task2-NR / task3-TSR，可能缺失）|

### 3.3 EEG 字段归一化方式详解

#### `eeg_word_norm1d`（逐词 1D z-score）

每个词的 840 维向量独立做 z-score：

$$\text{norm1d}(\mathbf{x}) = \frac{\mathbf{x} - \text{mean}(\mathbf{x})}{\text{std}(\mathbf{x})}, \quad \text{若 std}=0 \text{ 则返回全零}$$

#### `eeg_word_norm2d`（词+句全局 2D z-score）

将同一句的所有词向量 + 句级向量拼为矩阵 $M \in \mathbb{R}^{(T+1) \times 840}$，对整个矩阵展平后做 z-score：

$$\text{norm2d}(M) = \frac{M - \text{mean}(\text{flatten}(M))}{\text{std}(\text{flatten}(M))}$$

#### `eeg_spectro`（Spectrogram）

对每个通道的原始时序信号做短时傅里叶变换，跨 105 通道取均值，裁剪/padding 到 `(374, 65)`，再整体 z-score：

$$S_{ch} = \text{spectrogram}(\text{raw}_{ch},\ fs=500,\ \text{nperseg}=128,\ \text{noverlap}=64) \in \mathbb{R}^{65 \times T}$$

$$S_{\text{avg}} = \frac{1}{105}\sum_{ch=1}^{105} S_{ch} \in \mathbb{R}^{65 \times 374}$$

$$\text{eeg\_spectro} = \text{z-score}(S_{\text{avg}}^T) \in \mathbb{R}^{374 \times 65}$$

---

## 四、加载接口（`UnifiedDataset`）

### 4.1 基础用法

```python
from benchmark_eval.data_processing.dataset import UnifiedDataset

ds = UnifiedDataset("path/to/unified_zuco.pkl", phase="test")
sample = ds[0]  # Dict[str, Tensor]

# 常用字段
eeg = sample["eeg_word_norm1d"]    # Tensor (56, 840)
mask = sample["mask_word"]         # Tensor (56,)
text = sample["reference_text"]    # str
subject = sample["meta"]["subject"] # str
session = sample["meta"]["session"] # "session_1" / "session_2" / "session_unknown"
```

### 4.2 `__getitem__` 返回字段

每条样本作为 `Dict[str, Any]` 返回，所有 `np.ndarray` 字段自动转为 `torch.float32` Tensor：

| 字段 | 说明 |
|------|------|
| `idx` | 样本下标（int）|
| `eeg` | 默认 EEG（= `eeg_word_norm1d`），Tensor `(56, 840)` |
| `mask` | 默认 mask（= `mask_word`），Tensor `(56,)` |
| `eeg_word_raw` | 原始词级 EEG，Tensor `(56, 840)`（若存在）|
| `eeg_word_norm1d` | 逐词 1D 归一化，Tensor `(56, 840)` |
| `eeg_word_norm2d` | 全局 2D 归一化，Tensor `(56, 840)` |
| `sent_eeg_raw` | 句级 EEG，Tensor `(840,)` |
| `eeg_spectro` | Spectrogram，Tensor `(374, 65)`（若存在）|
| `mask_word` | 词级 mask，Tensor `(56,)` |
| `mask_word_with_sent` | 含句级 mask，Tensor `(56,)` |
| `mask_spectro` | Spectrogram mask，Tensor `(374,)` |
| `input_text` | str |
| `reference_text` | str |
| `meta` | Dict |

### 4.3 噪声模式（`noise_mode=True`）

用于对照实验，将真实 EEG 替换为随机噪声，文本标签不变：

```python
ds_noise = UnifiedDataset(
    data_path, phase="test",
    noise_mode=True,
    noise_type="gaussian",  # "gaussian" | "uniform" | "zero"
    noise_seed=42,          # 固定种子保证跨模型公平
    noise_std=1.0,
)
```

- 每条样本的噪声种子 = `noise_seed + idx`，保证样本级独立可复现
- 噪声 shape 与对应真实字段相同
- mask 在噪声模式下全为 1（无 padding 概念）

### 4.4 Shuffle 模式（`shuffle_mode=True`）

用于打乱 EEG-文本配对关系（derangement，保证无不动点）：

```python
ds_shuffle = UnifiedDataset(
    data_path, phase="test",
    shuffle_mode=True,
    shuffle_seed=42,
)
# ds_shuffle.shuffle_perm  ← 权威 permutation，供 EEG2Text 等外部脚本查询
```

- 文本标签原位不动，仅 EEG 字段（及 mask）按 permutation 重排
- 跨模型使用相同 `shuffle_seed` 保证排列一致

### 4.5 向后兼容

若加载旧版 PKL（v1 字段名），`UnifiedDataset` 自动回退：

| 新字段名（v2） | 旧字段名（v1 fallback）|
|---------------|----------------------|
| `eeg_word_norm1d` | `eeg_normalized_1d` |
| `eeg_word_norm2d` | `eeg_normalized_2d` |
| `eeg_word_raw` | `eeg_raw` |
| `eeg_spectro` | `eeg_eeg2text` |
| `mask_word_with_sent` | `mask_with_sent` |
| `mask_spectro` | `mask_eeg2text` |

---

## 五、各模型字段对应关系

| 模型 | 主要 EEG 字段 | 主要 Mask 字段 |
|------|-------------|--------------|
| EEG-To-Text | `eeg_word_norm1d`（= `eeg`）| `mask_word`（= `mask`）|
| EEG2Text | `eeg_spectro` | `mask_spectro` |
| CET-MAE | `eeg_word_norm2d`、`sent_eeg_raw` | `mask_word_with_sent` |
| GLIM | `eeg_word_norm1d`（经 wrapper 重采样至 1280 步、128 维）| `mask_word` |
| 诊断实验线 A（A1a/A1c）| `eeg_word_norm1d`（= `eeg`）、`sent_eeg_raw` | `mask_word`（= `mask`）|
| 诊断实验线 A（A1b）| `eeg_word_norm1d` + `nfixations_word`（注视时长加权）| `mask_word` |
