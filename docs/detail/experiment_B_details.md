# 实验线 B 细节

> **代码文件**：
> - `benchmark_eval/scripts/run_cet_mae_retrieval.py`
> - `benchmark_eval/scripts/run_eeg_to_text_retrieval.py`
> - `benchmark_eval/scripts/run_eeg2text_retrieval.py`
> - `benchmark_eval/scripts/run_glim_retrieval.py`
>
> **目标**：通过向模型输入替代信号（高斯噪声 / 配对打乱 / 全零），验证四个 EEG-to-Text 模型编码器是否从 EEG 中学到了有意义的表征，量化"模型利用率下界"。
>
> **运行环境**：需要 GPU（PyTorch），依赖各模型对应的 checkpoint 文件。

---

## 公共实验设置

### 评估任务：EEG-文本检索

以每条 EEG 样本作为 query，从测试集所有唯一参考文本组成的**候选池**中，通过余弦相似度检索出对应的参考文本。

| 参数 | 值 |
|------|-----|
| 查询集 | `phase="test"`，约 **1858 条** EEG 样本 |
| 候选池 | 同 phase 所有唯一参考文本，约 **130 条** |
| 相似度函数 | 余弦相似度（L2 归一化后点积）|
| 随机基线 R@1 | $1/130 \approx 0.77\%$ |
| 随机基线 Mean Rank | $\approx 65.5$ |

### 评估指标

$$\text{R@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[r_i \leq K], \quad K \in \{1, 5, 10\}$$

$$\text{MRR} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{r_i}$$

$$\text{Mean Rank} = \frac{1}{N}\sum_{i=1}^{N} r_i, \quad \text{Median Rank} = \text{median}(\{r_i\})$$

其中 $r_i$ 为第 $i$ 条 EEG query 对应正确文本在候选池排序中的位次（从 1 起计）。

此外按分组维度输出：`by_task`、`by_subject`、`by_dataset`。

### 噪声条件矩阵

**4 模型 × 4 条件 = 16 组实验**：

| 条件 | 缩写 | 含义 | 诊断目标 |
|------|------|------|----------|
| 真实 EEG | `real` | 原始数据，不做任何修改 | 基线参照 |
| 高斯噪声 | `gaussian` | 用 $\mathcal{N}(0,1)$ 随机信号完全替代 EEG | 编码器是否学到了任何 EEG 特征 |
| 配对打乱 | `shuffle` | 真实 EEG，但随机打乱 EEG-文本配对关系 | 编码器是否学到了跨模态对应关系 |
| 全零输入 | `zero` | 全零张量替代 EEG | 排除模型 bias/shortcut 依赖 |

---

## 三层噪声架构

为保证跨模型可比性，采用**三层架构**分离逻辑与实现：

| 层级 | 职责 | 说明 |
|------|------|------|
| **逻辑层** | 定义"什么"是噪声/shuffle | 所有模型完全一致的语义定义 |
| **协调层** | 生成全局权威配置 | `UnifiedDataset` 生成 permutation 和种子序列 |
| **实现层** | 决定"如何"应用 | CET-MAE / EEG-To-Text / GLIM 在数据加载时应用；EEG2Text 在编码阶段应用 |

### 协调层：权威配置机制

**Shuffle permutation（derangement）**：

```python
# dataset.py: _generate_derangement(n, seed=42)
# 生成无不动点的完全错位排列，保证 perm[i] != i 对所有 i 成立
perm[i] != i,  ∀ i ∈ {0, ..., N-1}
```

- `UnifiedDataset(shuffle_mode=True)` 自动生成并存储于 `self.shuffle_perm`
- EEG2Text 脚本通过导入 `_generate_derangement(len(ds), seed=42)` 独立生成相同排列
- `shuffle_seed=42` 保证跨模型一致

**噪声种子公式**：

$$\text{seed}_i = 42 + i \quad (i \text{ 为样本下标})$$

- 相同样本索引在不同模型中产生相同的噪声统计特性
- 不同样本独立，保证样本间无相关性

---

## 各条件数据准备

### gaussian / zero（CET-MAE / EEG-To-Text / GLIM）

在 `UnifiedDataset.__getitem__` 中，调用 `_generate_noise_eeg(sample, idx)` 替换 EEG 字段：

$$\mathbf{f}_i^\text{gaussian} \sim \mathcal{N}(0, 1)^{\text{shape}(\mathbf{f}_i^\text{real})}, \quad \text{seed} = 42 + i$$

$$\mathbf{f}_i^\text{zero} = \mathbf{0}^{\text{shape}(\mathbf{f}_i^\text{real})}$$

mask 字段统一设为全 1（`np.ones`），保持有效长度与真实数据相同。

脚本调用方式：
```python
UnifiedDataset(data_path, phase="test", noise_mode=True, noise_type="gaussian")
UnifiedDataset(data_path, phase="test", noise_mode=True, noise_type="zero")
```

### shuffle（CET-MAE / EEG-To-Text / GLIM）

在 `UnifiedDataset.__init__` 中，按 derangement 重排所有样本的 EEG 相关字段，文本标签保持不变：

$$\text{EEG}\_\text{sample}[i] \leftarrow \text{EEG}\_\text{sample}[\text{perm}[i]]$$

涉及字段：`eeg`、`eeg_word_raw`、`eeg_word_norm1d`、`eeg_word_norm2d`、`sent_eeg_raw`、`eeg_spectro` 及对应 mask 字段。

脚本调用方式：
```python
UnifiedDataset(data_path, phase="test", shuffle_mode=True)
```

### gaussian / zero（EEG2Text，实现层差异）

EEG2Text 使用原始时序数据（spectro pickle），不经过 `UnifiedDataset`。噪声在 `encode_eegs()` 函数中于编码阶段生成：

$$\mathbf{r}_i^\text{gaussian} \sim \mathcal{N}(0, 1)^{24000 \times 105}, \quad \text{seed} = 42 + i$$

$$\mathbf{r}_i^\text{zero} = \mathbf{0}^{24000 \times 105}$$

### shuffle（EEG2Text，实现层差异）

EEG2Text 脚本从 `UnifiedDataset` 导入 `_generate_derangement`，生成与协调层一致的 permutation，在收集阶段重排 `raw_list`：

```python
shuffle_perm = _generate_derangement(len(ds), seed=42)
raw_list = [all_raw_data[shuffle_perm[i]] for i in range(len(all_raw_data))]
```

---

## 各模型编码路径

### B1：CET-MAE

**脚本**：`run_cet_mae_retrieval.py`

#### EEG 输入格式

`eeg_word_norm2d`（词+句全局 2D z-score 归一化）：

$$\text{EEG\_input} \in \mathbb{R}^{B \times L \times 840}, \quad \text{mask\_word\_with\_sent} \in \{0,1\}^{B \times L}$$

#### EEG 编码路径

$$\text{EEG\_input} \xrightarrow{+\text{pos\_embed\_e}} x \xrightarrow{\text{e\_branch (TransformerEncoder)}} x \xrightarrow{\text{fc\_eeg + act}} x \xrightarrow{\text{unify\_branch}} \mathbf{h} \in \mathbb{R}^{B \times L \times D}$$

$$\mathbf{v}^\text{EEG} = \text{L2Norm}\!\left(\text{MeanPool}(\mathbf{h},\ \text{mask})\right) \in \mathbb{R}^{B \times 1024}$$

其中 `src_key_padding_mask = (1 - mask).bool()`（True 表示 padding 位置）。

#### 文本编码路径

$$\text{text} \xrightarrow{\text{BartTokenizer}} (\text{input\_ids, attn\_mask}) \xrightarrow{\text{t\_branch\_encoder (BART Encoder)}} \mathbf{h}^\text{text} \in \mathbb{R}^{B \times L \times 1024}$$

$$\mathbf{v}^\text{text} = \text{L2Norm}\!\left(\text{MeanPool}(\mathbf{h}^\text{text},\ \text{attn\_mask})\right) \in \mathbb{R}^{B \times 1024}$$

#### 候选池构建

```python
unique_texts = list(dict.fromkeys(ref_texts))  # 去重，保持顺序
t2i = {t: i for i, t in enumerate(unique_texts)}
gt_idx = [t2i[t] for t in ref_texts]
```

---

### B2：EEG-To-Text

**脚本**：`run_eeg_to_text_retrieval.py`

#### EEG 输入格式

`eeg_word_norm1d`（逐词 1D z-score 归一化）：

$$\text{EEG\_input} \in \mathbb{R}^{B \times L \times 840}, \quad \text{mask\_word} \in \{0,1\}^{B \times L}$$

#### EEG 编码路径（`addin_forward`）

$$\text{EEG\_input} \xrightarrow{\text{additional\_encoder (TransformerEncoder ×6, d=840)}} \mathbf{h} \in \mathbb{R}^{B \times L \times 840} \xrightarrow{\text{fc1 + ReLU}} \mathbf{h}' \in \mathbb{R}^{B \times L \times 1024}$$

$$\mathbf{v}^\text{EEG} = \text{L2Norm}\!\left(\text{MeanPool}(\mathbf{h}',\ \text{mask\_word})\right) \in \mathbb{R}^{B \times 1024}$$

其中 `src_key_padding_mask = (1 - mask).bool()`。

#### 文本编码路径

$$\text{text} \xrightarrow{\text{BartTokenizer}} \xrightarrow{\text{pretrained.model.encoder (BART Encoder)}} \mathbf{h}^\text{text} \xrightarrow{\text{MeanPool}} \mathbf{v}^\text{text} \in \mathbb{R}^{B \times 1024}$$

---

### B3：EEG2Text（BrainTranslator + ShallowNet）

**脚本**：`run_eeg2text_retrieval.py`

#### 数据来源：spectro pickle 查找表

EEG2Text 不使用 `UnifiedDataset` 的频域特征，而是直接从原始 spectro pickle 读取时序数据：

```python
SPECTRO_PICKLE_PATHS = {
    "task1-SR":     ".../task1-SR-dataset-spectro.pickle",
    "task2-NR":     ".../task2-NR-dataset-spectro.pickle",
    "task3-TSR":    ".../task3-TSR-dataset-spectro.pickle",
    "task2-NR-2.0": ".../task2-NR-2.0-dataset-spectro.pickle",
}
lut[(task_name, subject, sentence_index)] = rawData  # rawData shape: (105, time)
```

#### EEG 输入预处理（`raw_to_tensor`）

$$\text{rawData}^\top \in \mathbb{R}^{T \times 105} \xrightarrow{\text{pad/truncate}} \text{EEG} \in \mathbb{R}^{24000 \times 105}$$

截断或零填充至固定长度 $T_\text{max} = 24000$，然后整体 z-score 归一化：

$$\text{EEG\_norm} = \frac{\text{EEG} - \bar{\mu}}{\bar{\sigma}}, \quad \text{若 } \bar{\sigma} > 10^{-8}$$

#### EEG 编码路径（BrainTranslator）

$$\text{EEG\_norm} \in \mathbb{R}^{B \times 24000 \times 105} \xrightarrow{\text{BrainTranslator.forward()}} \mathbf{h} \in \mathbb{R}^{B \times 957 \times 1024}$$

$$\mathbf{v}^\text{EEG} = \text{L2Norm}\!\left(\text{MeanPool}(\mathbf{h})\right) \in \mathbb{R}^{B \times 1024}$$

BrainTranslator 内部含 ShallowNet（时序卷积提取空间特征）→ Transformer。

#### 噪声/零输入（实现层）

噪声直接替代 `raw_to_tensor` 的结果，维度保持 $(24000, 105)$：

| 条件 | 实现 |
|------|------|
| `gaussian` | `np.random.default_rng(42 + i).normal(0, 1, (24000, 105))` |
| `zero` | `torch.zeros(24000, 105)` |
| `shuffle` | `raw_list = [all_raw_data[perm[i]] for i in range(N)]`（收集阶段重排） |

#### 文本编码路径

$$\text{text} \xrightarrow{\text{BartTokenizer}} \xrightarrow{\text{text\_decoder.model.encoder}} \mathbf{h}^\text{text} \xrightarrow{\text{MeanPool}} \mathbf{v}^\text{text} \in \mathbb{R}^{B \times 1024}$$

> **注**：EEG2Text 的文本编码器与 EEG-To-Text 共享相同的 BART Encoder 架构（均来自 `facebook/bart-large`），但 checkpoint 不同。

#### 缺失样本处理

若某条样本在 spectro pickle 查找表中找不到对应 `rawData`（`miss_count`），该样本的 EEG 以全零替代：

```python
if raw is None:
    tensors.append(torch.zeros(24000, 105))
```

---

### B4：GLIM

**脚本**：`run_glim_retrieval.py`

#### EEG 输入格式

`eeg_word_raw`（词级原始 EEG，未归一化）：

$$\text{EEG\_input} \in \mathbb{R}^{B \times L \times 840}, \quad \text{mask\_word} \in \{0,1\}^{B \times L}$$

#### 格式转换（`_convert_to_glim_format`）

$$\text{EEG\_input}\ (B \times L \times 840) \rightarrow \text{GLIM\_EEG}\ (B \times 1280 \times 128)$$

将词级频域特征重塑为 GLIM 期望的 $(1280, 128)$ 格式。

#### EEG 编码路径

$$\text{GLIM\_EEG} \xrightarrow{\text{eeg\_encoder}(eeg, mask, \text{prompt\_embed})} \mathbf{h}^\text{EEG} \in \mathbb{R}^{B \times 96 \times 256}$$

$$\mathbf{v}^\text{EEG} = \text{L2Norm}\!\left(\text{aligner.embed\_eeg}(\mathbf{h}^\text{EEG})\right) \in \mathbb{R}^{B \times 1024}$$

其中 `aligner.embed_eeg` 通过 cross-attention（$\mathbf{q\_x}$ 作为 query）将 EEG 序列压缩为单向量。

**Prompt 嵌入**：GLIM 的 EEG 编码器需要任务 prompt，从 meta 信息中提取 `task` 字段：

$$\text{prompt\_ids} = \text{p\_embedder.encode}(\text{prompts}) \rightarrow \text{prompt\_embed} = \text{p\_embedder}(\text{prompt\_ids},\ \text{eval\_pembed})$$

#### 文本编码路径

$$\text{text} \xrightarrow{\text{T5Tokenizer (max\_length=96)}} \xrightarrow{\text{text\_model.get\_encoder() (Flan-T5-large)}} \mathbf{h}^\text{text} \in \mathbb{R}^{B \times L \times 1024}$$

$$\mathbf{v}^\text{text} = \text{L2Norm}\!\left(\text{aligner.embed\_text}(\mathbf{h}^\text{text},\ \text{attn\_mask})\right) \in \mathbb{R}^{B \times 1024}$$

`aligner.embed_text` 通过 cross-attention（$\mathbf{q\_y}$ 作为 query）将文本序列压缩为单向量，与训练时完全一致。

> **注**：GLIM 是四个模型中唯一明确使用 CLIP 损失训练 EEG-文本对齐的模型，因此该检索评估对 GLIM 最具直接意义。

---

## 模型对比总览

| 维度 | CET-MAE | EEG-To-Text | EEG2Text | GLIM |
|------|---------|-------------|----------|------|
| EEG 输入字段 | `eeg_word_norm2d` | `eeg_word_norm1d` | spectro pickle rawData | `eeg_word_raw` |
| EEG 输入 shape | $(B, L, 840)$ | $(B, L, 840)$ | $(B, 24000, 105)$ | $(B, L, 840)$ |
| EEG 编码器 | e_branch + fc_eeg + unify_branch | additional_encoder + fc1 | BrainTranslator (ShallowNet+Transformer) | eeg_encoder + aligner.embed_eeg |
| 文本编码器 | BART t_branch_encoder | BART pretrained.model.encoder | BART text_decoder.model.encoder | Flan-T5-large + aligner.embed_text |
| 嵌入维度 | 1024 | 1024 | 1024 | 1024 |
| 训练目标 | MAE + 对比学习 | seq2seq | seq2seq | CLIP（EEG-文本对齐）|
| 噪声注入层 | `UnifiedDataset` | `UnifiedDataset` | `encode_eegs()` | `UnifiedDataset` |
| batch size（默认）| 32 | 32 | 16 | 16 |

---

## 检索评估流程（各模型一致）

### 步骤 1：构建候选文本池

```python
unique_texts = list(dict.fromkeys(ref_texts))   # 保持首次出现顺序
t2i = {t: i for i, t in enumerate(unique_texts)}
gt_idx = [t2i[t] for t in ref_texts]            # 每条 EEG 对应的候选文本下标
```

$M = |\text{unique\_texts}| \approx 130$，$N = |\text{ref\_texts}| \approx 1858$

### 步骤 2：编码

1. 编码所有 $M$ 条唯一参考文本 → $\mathbf{V}^\text{text} \in \mathbb{R}^{M \times 1024}$（L2 归一化）
2. 编码所有 $N$ 条 EEG → $\mathbf{V}^\text{EEG} \in \mathbb{R}^{N \times 1024}$（L2 归一化）

### 步骤 3：余弦相似度矩阵

$$\mathbf{S} = \mathbf{V}^\text{EEG} \cdot (\mathbf{V}^\text{text})^\top \in \mathbb{R}^{N \times M}$$

（L2 归一化后点积等价于余弦相似度）

### 步骤 4：排名计算

对每个 query $i$，将 $\mathbf{S}_{i,:}$ 降序排列，正确答案位次：

$$r_i = \text{argsort}(\mathbf{S}_{i,:})[\text{::-1}].\text{index}(\text{gt\_idx}[i]) + 1$$

### 步骤 5：指标计算

$$\text{R@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[r_i \leq K], \quad K \in \{1, 5, 10\}$$

$$\text{MRR} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{r_i}, \quad \text{Mean Rank} = \frac{1}{N}\sum_{i=1}^{N} r_i$$

### 步骤 6：嵌入落盘（供降维可视化）

编码完成后将两类向量并同元数据落盘至输出目录的 `embeddings.npz`：

```python
np.savez(
    output_dir / "embeddings.npz",
    v_eeg=V_eeg.cpu().numpy(),      # (N, 1024)
    v_text=V_text.cpu().numpy(),    # (M, 1024)
    gt_idx=np.array(gt_idx),        # (N,)
    subjects=np.array(subjects),    # (N,)
    tasks=np.array(tasks),          # (N,)
    datasets=np.array(datasets),    # (N,)
    noise_type=noise_type,          # str: real/gaussian/shuffle/zero
)
```

该文件为下文降维可视化分析的唯一输入来源。

---

## 降维可视化

**目的**：将检索指标的数字结果（R@K / MRR）转为可见的空间结构，视觉化编码器学习质量与跨模态对齐情况。

**通用流程**：输入 $\mathbf{V} \in \mathbb{R}^{K \times 1024}$，统一执行 PCA(50) → t-SNE(2)，perplexity=30，`random_state=42`，输出二维坐标 $Z \in \mathbb{R}^{K \times 2}$。以下六种可视化均基于步骤 6 落盘的 `embeddings.npz`，无需重新编码。

### V1：单模型四条件 EEG 嵌入对比图

**数据**：同一模型的 $\mathbf{V}^\text{EEG}_\text{real}, \mathbf{V}^\text{EEG}_\text{gaussian}, \mathbf{V}^\text{EEG}_\text{shuffle}, \mathbf{V}^\text{EEG}_\text{zero}$拼接为 $4N \times 1024$，按条件染色。

**观察目标**（与结果解读表中六种典型诊断对应）：

| 降维图形态 | 对应模式 |
|------------|---------|
| 四色完全重叠为单一簇 | 模式 A：编码器完全无效 |
| real 略偏离 gaussian/shuffle/zero | 模式 B：学到 EEG 统计特性 |
| real 与 shuffle 重叠、与 gaussian/zero 分离 | 模式 C：学到跨模态对应 |
| zero 向量偏离 real 但靠近文本簇 | 特殊异常（b4 GLIM）：解码器偏差 |

输出：`tsne_{model}_4cond_by_noise_p30.png`，每模型一张。

### V2：EEG 与文本的跨模态联合降维图

**数据**：将 $\mathbf{V}^\text{EEG}_\text{real}$（$N \times 1024$）与 $\mathbf{V}^\text{text}$（$M \times 1024$）拼接为 $(N+M) \times 1024$。

**标记**：模态用不同形状（EEG 圆点 / 文本三角），颜色按 $\text{gt\_idx}$（sentence_id）上色。

**观察目标**：

| 模型 | 预期 | 含义 |
|------|------|------|
| **GLIM**（CLIP 训练）| 同句 EEG 和文本点靠近 | 跨模态对齐生效 |
| CET-MAE / EEG-To-Text / EEG2Text | 两模态形成两个分离簇 | 未专门优化对齐 |

**转衍用法**：对同一 sentence_id 的 EEG 点与文本点绘制连接线，线段越短 → 对齐越好。输出：`tsne_{model}_cross_modal_p30.png`。

### V3：跨模型同条件对比图

**数据**：固定 `real` 条件，将四模型的 $\mathbf{V}^\text{EEG}_\text{real}$ 各取一份拼接为 $4N \times 1024$，按**模型**染色。

**观察目标**：判断在相同 EEG 输入下四个编码器产出的分布差异，与 R@1 数据配合评价哪个编码器的嵌入最可分。输出：`tsne_cross_model_real_p30.png`。

### V4：分组维度染色图

**数据**：任一模型的 $\mathbf{V}^\text{EEG}_\text{real}$。

**染色方案**（对应 `by_subject` / `by_task` / `by_dataset` 分组指标）：

| 染色维度 | 验证目标 |
|---------|---------|
| `subject_id` | 编码后是否仍存在被试簇（与 A2-Eta subject_dominant 互证）|
| `task` | task1-SR / task2-NR / task3-TSR 是否形成簇 |
| `dataset` | ZuCo1 / ZuCo2 跨数据集漂移 |
| `sentence_id` | 同句 EEG 嵌入是否聚拢 |

输出：`tsne_{model}_real_by_{subject|task|dataset|sentence}_p30.png`。

### V5：Shuffle 专项诊断图

**数据**：同模型的 $\mathbf{V}^\text{EEG}_\text{real}$ 与 $\mathbf{V}^\text{EEG}_\text{shuffle}$（同一条 EEG 样本在两种条件下的向量）。

**预期**：real 与 shuffle 的同一样本应完全重合（EEG 内容不变）；若按**文本标签**染色，real 下同句聚簇、shuffle 下同句随机分散 → 直接验证跨模态对应是否存在。输出：`tsne_{model}_shuffle_diag_p30.png`。

### V6：Zero 响应图（异常诊断）

**数据**：$\mathbf{V}^\text{EEG}_\text{zero}$ 单独降维。

**观察**：

| 现象 | 含义 |
|------|------|
| 所有点聩缩为单一点 | 全零输入→密集输出，符合预期 |
| 点弥散且靠近某些文本簇 | 存在 BN/Dropout 影响或解码器偏差 |

特别适用于 **GLIM**（zero 指标反而高于 real）的根因定位。输出：`tsne_{model}_zero_response_p30.png`。

### 降维可视化执行脚本

在 `benchmark_eval/scripts/visualize_b_embeddings.py` 中统一实现 V1-V6，输入为 `--model {cet_mae|eeg_to_text|eeg2text|glim}` 与 `--viz {v1|v2|v3|v4|v5|v6|all}`，读取对应 `eval_*_retrieval*/embeddings.npz`，输出至 `benchmark_eval/test_outputs/tsne_b/`。

---

## 结果解读框架

### 单模型诊断决策树

```
            real vs gaussian
           /                \
   real > gaussian        real ≈ gaussian
  （编码器学到了东西）   （编码器未利用 EEG 特征）
       |                        |
 real vs shuffle          zero 是否异常？
  /          \              /          \
real > shuffle  real ≈ shuffle  zero 异常    zero ≈ 随机
（学到跨模态   （仅学到统计   （pipeline     （编码器
  对应关系）    偏置/分布）    存在 bug）     完全无效）
```

### 六种典型诊断结论

| 情况 | real vs gaussian | real vs shuffle | zero 行为 | 归因 |
|------|------------------|-----------------|-----------|------|
| A | real ≈ gaussian ≈ shuffle ≈ zero | — | ≈ 随机 | **编码器完全无效**，未从 EEG 学到任何信息 |
| B | real > gaussian | real ≈ shuffle | — | 学到 EEG **统计特性**（功率谱分布等），未学到跨模态对应 |
| C | real > gaussian | real > shuffle | — | **学到跨模态对应**，但检索天花板受限于信号 SNR |
| D | real ≈ gaussian | — | zero 显著偏离 | **数据管道 bug**：噪声与真实信号经处理后特征坍缩 |
| E | real ≈ gaussian ≈ zero | — | — | 模型存在 **shortcut/bias**，不依赖输入内容 |
| F | real > gaussian, real > shuffle，但绝对值仍低 | — | — | 模型有效但 **EEG SNR 太低**，语义信息本身稀疏 |

### 启发式差异阈值（仅作初筛）

以下阈值用于快速初筛对比，**不作为最终结论依据**，所有诊断必须配套下文《显著性检验》中的 permutation test 与 bootstrap CI 来确认。

| 指标 | 判定为显著差异的阈值 |
|------|---------------------|
| Mean Rank 差值 | $> 3.0$ |
| R@10 差值 | $> 2$ 个百分点 |
| MRR 差值 | $> 0.005$ |

---

## 显著性检验

所有对比结论（模式 A/B/C/D/E/F、zero 异常、跨模型排名、分组异质性）均需配套定量显著性检验，接受阈值 $\alpha = 0.05$（多重比较校正后使用 $\alpha_\text{adj}$），所有结果落盘至 `significance_tests.json`。

### 检验单位：per-query rank

每一条 EEG query 产出一个排名 $r_i$，N=1858 个配对样本为一级检验单位，允许在 real/gaussian/shuffle/zero 四条件间做样本级配对对比（同一 EEG 样本在四条件下各有一个 $r_i$）。

### 检验方法统一约定

| 场景 | 首选方法 | 效应量 | 置信区间 |
|------|---------|--------|---------|
| 条件间 R@K 对比（每模型）| **Permutation test**（1000 次打乱 gt_idx）| 与 null 分布的标准差偏离 | 经验分位数 95% CI |
| 条件间 per-query rank 对比 | **Wilcoxon 符号秩检验**（按样本配对）| Cohen's $d_z$ | bootstrap 95% CI |
| R@K vs 随机基线 $K/M$ | **二项检验**（精确 binomial，$n=N$，$p_0=K/M$）| 查准率提升 | Clopper-Pearson CI |
| Mean Rank / MRR 对比 | **Bootstrap resampling**（按 query 有放回重采样 1000 次）| 均值差 + 效应量 | 95% CI + p 值由 CI 推导 |
| Rank 分布 real vs 随机均匀 | **Kolmogorov-Smirnov 检验** | KS 统计量 | — |
| 多模型同指标对比（4 模型）| **Friedman 检验 + 事后 Nemenyi** | Kendall's $W$ | — |
| 分组指标异质性（subject / task / dataset）| **Kruskal-Wallis** | $\eta^2_H$ | — |

### 多重比较校正

总检验数量 = **4 模型 × 6 条件对（real-vs-gaussian, real-vs-shuffle, real-vs-zero, gaussian-vs-shuffle, gaussian-vs-zero, shuffle-vs-zero）× 3 指标 = 72 组**，**必须校正**：

| 检验类别 | 范围 | 校正方法 |
|---------|------|---------|
| 单模型条件对检验（6 对 × 3 指标）| 18 组 | **Holm-Bonferroni** |
| 跨 4 模型汇总（72 组）| 全局 | **Benjamini-Hochberg FDR** |
| Friedman 事后多重比较（4 模型 × 2 两两对比）| 6 组 | **Nemenyi**（已隐含校正）|
| 分组事后对比（多被试两两）| $\binom{P}{2}$ | **Dunn 检验 + BH-FDR** |

### 核心诊断的检验要求

以下结论必须有显著性依据，否则在报告中标注为“当前样本量下不显著”：

| 结论 | 必需的显著性证据 |
|------|-----------------|
| “模式 A：编码器完全无效” | real-vs-gaussian + real-vs-shuffle + real-vs-zero 的 permutation test 均 $p > \alpha_\text{adj}$ |
| “模式 B：学到统计特性未学到对应” | real-vs-gaussian $p < \alpha_\text{adj}$ 且 real-vs-shuffle $p > \alpha_\text{adj}$ |
| “模式 C：学到跨模态对应” | real-vs-gaussian 与 real-vs-shuffle 均 $p < \alpha_\text{adj}$，且 Cohen's $d_z > 0.2$ |
| “b4 GLIM zero > real 异常” | zero-vs-real 在 R@10 / MRR 上均 $p < \alpha_\text{adj}$ 且效应方向为 zero > real |
| “模型 X 显著优于模型 Y” | Friedman 整体 $p < 0.05$，事后 Nemenyi 下 X vs Y $p < \alpha_\text{adj}$ |
| “被试/任务之间存在异质性” | Kruskal-Wallis $p < 0.05$，且 Dunn 事后证实具体子组对差异 |

### b4 GLIM 异常检验清单

针对“zero R@10 = 13.02% > real R@10 = 8.83%”的异常观察，必须执行：

1. **Permutation test**：按照 real 与 zero 关于同一批 EEG query 的样本配对 rank 差，检验均值差的统计显著性。
2. **Bootstrap 95% CI**：对 MRR、R@10 差值构建置信区间，确认区间不包含 0。
3. **分组 robustness**：在 `by_task` / `by_subject` 分组上重复上述检验，判断异常是否为全局现象。
4. **Kruskal-Wallis**：判断异常是否由某少数被试导致。

### 输出约定

每组实验的输出目录额外生成 `significance_tests.json`：

```json
{
  "model": "cet_mae",
  "alpha": 0.05,
  "alpha_adjusted": 0.0028,
  "correction_method": "holm_bonferroni",
  "pairwise": {
    "real_vs_gaussian": {
      "r@1":  { "delta": ..., "p_perm": ..., "p_bootstrap_ci": [..., ...], "cohens_dz": ..., "significant": true },
      "r@5":  { ... },
      "r@10": { ... },
      "mrr":  { ... },
      "mean_rank": { ... }
    },
    "real_vs_shuffle": { ... },
    "real_vs_zero":    { ... },
    ...
  },
  "vs_random_baseline": {
    "r@1":  { "baseline": 0.0077, "observed": ..., "p_binomial": ..., "ci_95": [..., ...] },
    ...
  },
  "rank_distribution": {
    "ks_vs_uniform": { "statistic": ..., "p": ... }
  },
  "grouped": {
    "by_subject": { "kruskal_wallis": { ... }, "dunn_posthoc": { ... } },
    "by_task":    { ... }
  }
}
```

跨模型汇总专用文件 `benchmark_eval/test_outputs/significance_summary.json`：

```json
{
  "friedman": {
    "r@10_real": { "statistic": ..., "p": ..., "kendalls_w": ... }
  },
  "nemenyi_posthoc": {
    "r@10_real": { "cet_mae_vs_glim": ..., ... }
  }
}
```

---

## 运行命令

### 参数说明

所有脚本均支持 `--noise-type {real,gaussian,shuffle,zero}` 参数，输出目录自动添加对应后缀：

| `--noise-type` | 输出目录后缀 |
|----------------|-------------|
| `real` | 无后缀（默认）|
| `gaussian` | `_gaussian` |
| `shuffle` | `_shuffle` |
| `zero` | `_zero` |

### 示例命令（以 CET-MAE 为例）

```bash
# real（基线）
python benchmark_eval/scripts/run_cet_mae_retrieval.py \
    --data-path benchmark_eval/data/unified_zuco.pkl \
    --model-checkpoint models/CET-MAE/checkpoints/decoding/cet_mae_benchmark_best.pt \
    --output-dir benchmark_eval/test_outputs/eval_cet_mae_retrieval \
    --phase test --noise-type real

# gaussian
python benchmark_eval/scripts/run_cet_mae_retrieval.py \
    ... --noise-type gaussian   # → 输出到 eval_cet_mae_retrieval_gaussian/

# shuffle
python benchmark_eval/scripts/run_cet_mae_retrieval.py \
    ... --noise-type shuffle    # → 输出到 eval_cet_mae_retrieval_shuffle/

# zero
python benchmark_eval/scripts/run_cet_mae_retrieval.py \
    ... --noise-type zero       # → 输出到 eval_cet_mae_retrieval_zero/
```

### 批量运行（全部 16 组）

```bash
for model in cet_mae eeg_to_text eeg2text glim; do
  for noise in real gaussian shuffle zero; do
    python benchmark_eval/scripts/run_${model}_retrieval.py \
      --noise-type $noise \
      --data-path benchmark_eval/data/unified_zuco.pkl \
      --model-checkpoint <对应 checkpoint 路径> \
      --output-dir benchmark_eval/test_outputs/eval_${model}_retrieval \
      --phase test
  done
done
```

---

## 输出文件结构

每组实验输出一个独立目录，内含：

```
benchmark_eval/test_outputs/
├── eval_cet_mae_retrieval/                  # real
│   ├── retrieval_metrics.json
│   └── retrieval_eval.log
├── eval_cet_mae_retrieval_gaussian/
│   ├── retrieval_metrics.json
│   └── retrieval_eval.log
├── eval_cet_mae_retrieval_shuffle/
│   └── ...
├── eval_cet_mae_retrieval_zero/
│   └── ...
├── eval_eeg_to_text_retrieval/              # real
│   └── ...（同上结构）
├── eval_eeg_to_text_retrieval_gaussian/
├── eval_eeg_to_text_retrieval_shuffle/
├── eval_eeg_to_text_retrieval_zero/
├── eval_eeg2text_retrieval/
├── eval_eeg2text_retrieval_gaussian/
├── eval_eeg2text_retrieval_shuffle/
├── eval_eeg2text_retrieval_zero/
├── eval_glim_retrieval/
├── eval_glim_retrieval_gaussian/
├── eval_glim_retrieval_shuffle/
└── eval_glim_retrieval_zero/
```

每个 `eval_*_retrieval*/` 目录均额外包含 `embeddings.npz`（供降维可视化）与 `significance_tests.json`（供显著性检验），另外 `benchmark_eval/test_outputs/significance_summary.json` 统一汇总 4 模型的 Friedman / Nemenyi 结果。

### `retrieval_metrics.json` 字段说明

```json
{
  "overall": {
    "r@1": 0.0108,
    "r@5": 0.0506,
    "r@10": 0.0888,
    "mrr": 0.0484,
    "num_queries": 1858,
    "candidate_pool_size": 130,
    "random_baseline_r@1": 0.007692,
    "mean_rank": 63.5,
    "median_rank": 52.0
  },
  "grouped": {
    "by_task":    { "<task_name>": { "sample_count": ..., "metrics": {...}, "mean_rank": ..., "median_rank": ... } },
    "by_subject": { "<subject_id>": { ... } },
    "by_dataset": { "<dataset_name>": { ... } }
  }
}
```

> EEG2Text 额外输出 `"missing_raw_count"` 字段，记录 spectro pickle 中未找到对应 rawData 的样本数。

### `embeddings.npz` 字段说明

| 字段 | shape | 含义 |
|------|-------|------|
| `v_eeg` | $(N, 1024)$ | L2 归一化后的 EEG 嵌入 |
| `v_text` | $(M, 1024)$ | L2 归一化后的候选文本嵌入 |
| `gt_idx` | $(N,)$ | 每条 EEG 对应的文本池下标 |
| `subjects` | $(N,)$ | 被试 ID |
| `tasks` | $(N,)$ | 任务名 |
| `datasets` | $(N,)$ | 数据集名 |
| `noise_type` | 标量 | `real` / `gaussian` / `shuffle` / `zero` |

该文件供降维可视化章节的 V1–V6 调用，无需重新执行编码。

### 降维可视化输出文件

```
benchmark_eval/test_outputs/tsne_b/
├── tsne_{model}_4cond_by_noise_p30.png          # V1：四条件对比，4 模型各一张
├── tsne_{model}_cross_modal_p30.png             # V2：EEG+文本联合降维，4 模型各一张
├── tsne_cross_model_real_p30.png                # V3：real 条件下四模型同图对比
├── tsne_{model}_real_by_{subject|task|dataset|sentence}_p30.png  # V4：分组染色
├── tsne_{model}_shuffle_diag_p30.png            # V5：shuffle 专项诊断
└── tsne_{model}_zero_response_p30.png           # V6：zero 响应异常诊断
```
