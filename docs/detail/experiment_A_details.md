# 实验线 A 细节

> **代码文件**：`benchmark_eval/scripts/validate_eeg_signal.py`
> **目标**：在当前特征表示下，从数据层面验证 ZuCo EEG 是否包含可检测的句子级语义信息。
> **运行环境**：纯 CPU，依赖 `sklearn / scipy / matplotlib`。

---

## 公共数据准备

### 数据来源与设计原则

- 统一数据集文件：`unified_zuco.pkl`（由 `build_unified_dataset.py` 从 ZuCo MAT 文件构建）
- 数据集划分策略：**文本级 80 / 10 / 10**，按唯一句子文本划分，保证同一句子的所有被试样本都在同一 phase

**实验范围**：全部 A 线实验**只在 `phase="test"` 的小范围测试集内进行**：
- 约 **130 个唯一句子**，约 **1858 条样本**，约 **14 位被试**
- 每位被试均阅读了相同的 130 个句子（跨被试设计）
- 只加载 `test_data`（`collect_samples(data_path, phase="test")`），不依赖 train 集

**为何只用测试集**：
- A 线的目标是"诊断数据集有效性"，而非训练可泛化模型
- 文本级划分导致 train 与 test 句子集合完全不重叠，跨集分类任务在语义上不成立
- 测试集内部 ~1858 条样本（14 被试 × 130 句）已足够做统计诊断

### 实验内部划分策略

A1 Linear Probe 采用 **LOSO 5 折交叉验证**（Leave-Subject-Out，`StratifiedGroupKFold(n_splits=5)`）：

| 折数 | 训练集规模 | 测试集规模 | 特点 |
|------|-----------|-----------|------|
| 5 折 | ~11-12 被试 × 130 句 ≈ 1460 条 | ~2-3 被试 × 130 句 ≈ 390 条 | 无被试泄露，每折测试集统计稳定 |

> 不采用 14 折完整 LOSO 的原因：14 折时每折测试集仅 ~130 条（每类 1 条），指标方差极大，无统计意义。

> 不在受试者内部实验原因：单个受试者单个句子仅一条数据，难以训练

### 步骤 1：样本加载（`collect_samples`）

从 `UnifiedDataset` 加载每条样本：

- EEG 特征字段：优先读取 `eeg_word_norm1d`，回退 `eeg`
  - shape：`(MAX_LEN, 840)`，其中 840 = 8 频带 × 105 通道
  - `eeg_word_norm1d` 为逐词 1D z-score 归一化后的 GD 型眼动 EEG
- Mask 字段：优先读取 `mask_word`，回退 `mask`
  - shape：`(MAX_LEN,)`，1 = 有效词位，0 = padding

局部句子 ID 映射（仅在当前 phase 内）：

$$\text{text\_to\_id}[t] = \text{按首次出现顺序递增分配整数 ID}$$

输出字段：`eeg_list`、`sentence_id_list`、`subject_list`、`task_list`、`n_classes`

### 步骤 2：特征提取（`extract_features`）

输入：`(eeg, mask)` 列表，其中 eeg shape `(MAX_LEN, 840)`

**有效长度截取**：

$$T_i = \sum_{t=1}^{L} \text{mask}_{i,t}, \quad T_i = \max(T_i, 1)$$

$$\text{eeg\_valid}_i = \text{eeg}_i[0 : T_i, :] \in \mathbb{R}^{T_i \times 840}$$

**变体 A：mean\_pool**

$$\mathbf{f}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} \text{eeg\_valid}_{i,t,:} \in \mathbb{R}^{840}$$

**变体 B：band\_separated**

将 `eeg_valid` 重塑为 `(T_i, 8, 105)`，在时间轴取均值后展平：

$$\mathbf{f}_i = \text{flatten}\!\left(\frac{1}{T_i}\sum_{t=1}^{T_i} \text{eeg\_valid}_{i,t,:}.reshape(8,105)\right) \in \mathbb{R}^{840}$$

> **注**：`mean_pool` 与 `band_separated` 在数值上等价（均为沿时间轴取均值），结果完全相同。

### 步骤 3：噪声基线特征生成

为量化"EEG 是否优于随机信号"，对每个真实样本生成一条**等形状高斯噪声特征**，与真实 EEG 特征在完全相同的流程下平行运行。

**噪声类型**：高斯噪声（`gaussian`），参数 $\mu=0,\ \sigma=1$（与 `eeg_word_norm1d` 的逐词 z-score 分布一致）

**噪声生成**（样本级固定种子，保证可复现）：

$$\mathbf{f}_i^\text{noise} \sim \mathcal{N}(\mathbf{0},\ \mathbf{I}_{840}), \quad \text{seed}_i = 42 + i$$

其中 $i$ 为样本在当前 phase 中的下标。不同样本使用不同种子，保证相互独立。

**在 A 线各子实验中的使用方式**：将 $X^\text{noise} \in \mathbb{R}^{N \times 840}$ 替代 $X^\text{EEG}$ 代入相同流程，其余（标签、被试分组、分类器参数）完全不变。若噪声结果贴近理论随机基线，说明实验流程无 shortcut；若 EEG 结果显著超越噪声，说明 EEG 中确实存在有效信号。

### 步骤 4：Session 标注（用于跨 session 分析）

根据 ZuCo 原始实验设计（Hollenstein et al., 2018），每位被试在两个实验阶段完成所有任务：

- **Session 1**：完成 Task 2（NR，300 句）+ Task 1（SR）前半部分
- **Session 2**：完成 Task 3（TSR，407 句）+ Task 1（SR）后半部分

句子呈现顺序对所有被试完全一致，因此可根据任务类型和句序确定 session 归属：

| 任务 | Session 1 | Session 2 | 判定条件 |
|------|-----------|-----------|----------|
| task1-SR | 前半部分 | 后半部分 | `sentence_index < N_task1 // 2` → Session 1；`sentence_index ≥ N_task1 // 2` → Session 2 |
| task2-NR | 全部 | — | 均为 Session 1 |
| task3-TSR | — | 全部 | 均为 Session 2 |
| task2-NR-2.0 | 待确认 | 待确认 | ZuCo v2 需单独核实 session 结构，暂标注为 `"session_unknown"` |

其中 $N_\text{task1}$ 为 task1-SR 的总句数（约 400），`sentence_index` 为样本在 MAT 文件中该任务的原始序号（即 `build_unified_dataset.py` 中的 `sent_idx`）。

**前置要求**：需在 `build_unified_dataset.py` 的 `_enrich_samples_with_metadata_and_labels` 函数中为每条样本的 `meta` 添加 `session` 字段，然后重新构建 `unified_zuco.pkl`；`collect_samples` 同步输出 `session_list`。

```python
def _assign_session(task_name: str, sentence_index: int, total_task1_sentences: int) -> str:
    if task_name == "task2-NR":
        return "session_1"
    elif task_name == "task3-TSR":
        return "session_2"
    elif task_name == "task1-SR":
        mid = total_task1_sentences // 2
        return "session_1" if sentence_index < mid else "session_2"
    else:
        return "session_unknown"
```

> **Task 与 Session 的混淆**：task2-NR 仅在 Session 1，task3-TSR 仅在 Session 2，因此跨 session 分析需在 task1-SR 内部单独做一次对照，以分离真正的 session 效应。

---

## 实验 A1：Linear Probe 分类

**目的**：用线性分类器检验 EEG 表示是否可区分不同句子，评估句子级语义信息的可检测性。

**数据**：测试集（`test_data`），约 1858 条样本，130 类句子，约 14 位被试。

**分类器**：`LogisticRegression`（`solver=lbfgs`，`max_iter=1000`，`random_state=42`，`n_jobs=-1`）

**评估指标**：Top-1 / Top-5 / Top-10 Accuracy，随机基线 $1/130 \approx 0.77\%$

### 三组信号并行

A1 下的 A1a / A1b / A1c 每种池化方法均对以下**三组信号**各跑一遍，结果并列报告：

| 信号组 | 特征来源 | shape | 说明 |
|--------|---------|-------|------|
| **词级 EEG**（主实验）| `eeg_word_norm1d` + mask 截取后 mean-pool / band-sep | `(N, 840)` | 逐词 1D z-score 归一化，只含有眼动注视的词 |
| **句级 EEG**（新增对照）| `sent_eeg_raw` 逐样本 z-score 归一化 | `(N, 840)` | MAT 文件预计算句级频带均值；ZuCo v2 缺失时用零向量填充 |
| **高斯噪声**（基线）| $\mathcal{N}(0,1)$，seed = 42 + i | `(N, 840)` | 验证流程无 shortcut |

> **句级 EEG 特征提取**：`sent_eeg_raw` 已是 840 维向量，不需要时间轴池化，直接做逐样本 z-score：
> $$\mathbf{f}_i^{\text{sent}} = \frac{\mathbf{s}_i - \text{mean}(\mathbf{s}_i)}{\max(\text{std}(\mathbf{s}_i),\ 10^{-8})}$$
> 因此在 A1a/A1b/A1c 三种池化变体下，句级 EEG 的特征值**完全相同**（无池化可做），三种变体结果一致，均标注 `"source": "sent_eeg_raw"`。

---

### A1a：Mean-Pool Linear Probe

#### 步骤 1：特征提取

三组信号的特征矩阵 $X \in \mathbb{R}^{N \times 840}$ 分别构造如下：

**词级 EEG**：
$$X_i^{\text{word}} = \frac{1}{T_i}\sum_{t=1}^{T_i} \text{eeg\_word\_norm1d}_{i,t,:}$$

**句级 EEG**：
$$X_i^{\text{sent}} = \frac{\mathbf{s}_i - \text{mean}(\mathbf{s}_i)}{\max(\text{std}(\mathbf{s}_i),\ 10^{-8})}, \quad \mathbf{s}_i = \text{sent\_eeg\_raw}_i$$

**高斯噪声**：
$$X_i^{\text{noise}} \sim \mathcal{N}(\mathbf{0},\ \mathbf{I}_{840}), \quad \text{seed}_i = 42 + i$$

#### 步骤 2：LOSO 5 折交叉验证

使用 `StratifiedGroupKFold(n_splits=5)`，按被试分组，共 **5 折**。
（采用 5 折而非完整 LOSO-14 折的原因：14 折时每折测试集仅约 130 条，每类只有 1 条，指标方差过大；5 折每折约 390 条测试样本，统计更稳定。）

第 $k$ 折划分（$k = 1, \ldots, 5$）：
$$\mathcal{D}_\text{train}^{(k)} = \{i : \text{subj}[i] \notin \text{test\_subj}_k\}, \quad \mathcal{D}_\text{test}^{(k)} = \{i : \text{subj}[i] \in \text{test\_subj}_k\}$$

每折约 2～3 位被试进入测试集，保证：$\text{subj}[i] \neq \text{subj}[j]$，$\forall i \in \mathcal{D}_\text{train}^{(k)},\ j \in \mathcal{D}_\text{test}^{(k)}$（无被试泄露）

每折执行：
1. `StandardScaler.fit_transform(X_train)` → `StandardScaler.transform(X_test)`
2. `LogisticRegression.fit(X_train_scaled, y_train)` → 预测 `X_test_scaled`
3. 计算 Top-1 / Top-5 / Top-10

最终结果取 5 折均值 ± 标准差：

$$\text{Top-K}_\text{LOSO} = \frac{1}{5}\sum_{k=1}^5 \text{Top-K}^{(k)}, \quad \text{std} = \sqrt{\frac{1}{5}\sum_{k=1}^5 \left(\text{Top-K}^{(k)} - \text{Top-K}_\text{LOSO}\right)^2}$$

#### 步骤 3：指标计算

预测概率矩阵：$\mathbf{P} \in \mathbb{R}^{N_\text{test} \times C}$，$C = 130$

$$\text{Top-1} = \frac{1}{N_\text{test}} \sum_{i=1}^{N_\text{test}} \mathbf{1}\!\left[\arg\max_k P_{i,k} = y_i\right]$$

$$\text{Top-K} = \frac{1}{N_\text{test}} \sum_{i=1}^{N_\text{test}} \mathbf{1}\!\left[y_i \in \text{argtop-K}(\mathbf{P}_{i,:})\right], \quad K \in \{5, 10\}$$

$$\text{Random Baseline} = \frac{1}{130} \approx 0.77\%$$

#### 结果解读框架

三组信号并列报告，比较关系如下：

| 情况 | 解读 |
|------|------|
| 词级 >> 随机 & 词级 >> 噪声 | 词级注视 EEG 携带跨被试可泛化句子语义 |
| 句级 >> 随机 & 句级 >> 噪声 | 句级频带均值同样携带句子信号 |
| 词级 > 句级 | 词级注视过滤（`nFixations > 0`）提升了特征质量 |
| 词级 ≈ 句级 | 两种粒度信息量相近，注视过滤作用有限 |
| 句级 ≈ 随机 ≈ 噪声 | 句级信号几乎不携带可区分的句子语义 |
| std 很大 | 被试效应显著，跨被试泛化困难 |

**三组运行方式**：将相同 LOSO 5 折 CV 流程分别应用于 $X^\text{word}$、$X^\text{sent}$、$X^\text{noise}$，结果以 `"word_eeg"` / `"sent_eeg"` / `"noise"` 三个键并列记录在 JSON 中。

---

### A1b：Duration-Weighted Pool

**状态：未实现（数据缺失），当前 fallback 为 A1a-LOSO 结果**

理论公式（词级 EEG）：

$$\mathbf{f}_i = \sum_{t=1}^{T_i} w_{i,t} \cdot \text{eeg\_valid}_{i,t,:}, \quad w_{i,t} = \frac{\text{dur}_{i,t}}{\sum_{t'} \text{dur}_{i,t'}}$$

其中 $\text{dur}_{i,t}$ 为第 $t$ 个词的注视时长（nFixations）。

**原因**：`UnifiedDataset` 当前未包含 `fixation duration` 字段，需在 `build_unified_dataset.py` 中添加。

当前输出字段包含 `"status": "fallback_to_mean_pool"` 标记。

> **句级 EEG / 噪声**：与 A1a 完全相同（句级向量无时间轴可加权，噪声无注视时长），三组均 fallback 到 A1a 结果。

---

### A1c：Band-Separated Linear Probe

词级 EEG 特征提取使用 `band_separated` 变体，其余步骤（LOSO 5 折 CV）与 A1a 完全相同：

$$\mathbf{f}_i^{\text{word}} = \text{flatten}\!\left(\frac{1}{T_i}\sum_{t} \text{eeg\_valid}_{i,t,:}.reshape(8,105)\right)$$

> **注**：词级 `mean_pool` 与 `band_separated` 数值结果完全相同（公式等价），该变体强调将 840 维视为 8 频带 × 105 通道的独立贡献。

> **句级 EEG**：`sent_eeg_raw` 已是 (840,) 向量，reshape 为 (8, 105) 取均值再 flatten 仍等于原向量，结果与 A1a-sent 完全相同，标注 `"note": "band_sep_equiv_to_a1a_for_sent_eeg"`。

> **噪声**：同上，与 A1a-noise 完全相同。

---

### A1d：跨 Session 可分性 Linear Probe

**目的**：检验线性分类器能否从 EEG 特征区分**同一被试**的不同 session。若可分性显著高于随机基线，则同一被试的两个 session 在 EEG 表示空间存在系统性差异，跨 session 可比性存疑。

**数据**：测试集中同时出现在两个 session 的被试子集（排除 `session_unknown`）。每位被试单独建模：

$$\mathcal{I}_p = \{i : \text{subj}[i] = p,\ \text{session}[i] \in \{\text{session\_1}, \text{session\_2}\}\}$$

**标签**：$y_i = 0$ 对应 Session 1，$y_i = 1$ 对应 Session 2。若某被试仅有单个 session 数据，则跳过。

**特征**：复用 A1a 的 `mean_pool` 词级 EEG、句级 EEG、高斯噪声三组信号，并行报告。

**方法**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，每折 `StandardScaler` + `LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)`，5 折均值 accuracy。

**随机基线**（多数类基线，逐被试计算后取均值）：

$$\text{baseline}_p = \frac{\max(n_{p,1},\ n_{p,2})}{n_{p,1} + n_{p,2}}, \quad \text{baseline} = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \text{baseline}_p$$

**输出**：每位被试的 5 折均值 accuracy 及全体被试均值 ± std。

**task1-SR 内部对照**：仅在 `task1-SR` 任务内部重复上述流程，排除 task2-NR / task3-TSR 的 task 混淆后的 session 可分性。若 task1-SR 内 accuracy 显著低于全局结果，说明全局 session 可分主要由 task 差异驱动；若两者相当，则 session 效应独立于 task。

**解读**：

| 情况 | 含义 |
|------|------|
| accuracy ≈ baseline ≈ 噪声结果 | 同被试两个 session 的 EEG 分布无显著差异，跨 session 可直接合并 |
| accuracy 显著 > baseline | Session 内存在可检测的系统性差异（疑疲劳 / 电极偏移 / 实验归一化差异） |
| accuracy 远超 baseline | Session 效应强烈，需 session-level 归一化后方可跨 session 合并 |

---

## 实验 A2：被试效应 vs 句子效应分析

**数据**：测试集（`test_data`）所有样本，特征 $X \in \mathbb{R}^{N \times 840}$，使用 `mean_pool` 变体（`test_feats_mp`）。

### A2-Cosine：余弦相似度分组对比

#### 步骤 1：计算全体余弦相似度矩阵

$$\text{cos}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \cdot \|\mathbf{x}_j\|_2}$$

矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，$S_{ij} = \text{cos}(\mathbf{x}_i, \mathbf{x}_j)$

#### 步骤 2：样本对分组

遍历所有上三角对 $(i, j)$，$i < j$：

| 组别 | 条件 | 含义 |
|------|------|------|
| 同句异被试 | $\text{sent}[i]=\text{sent}[j]$ 且 $\text{subj}[i]\neq\text{subj}[j]$ | 句子语义效应 |
| 同被试异句 | $\text{subj}[i]=\text{subj}[j]$ 且 $\text{sent}[i]\neq\text{sent}[j]$ | 被试个体特征 |
| 同被试异句同 session | $\text{subj}[i]=\text{subj}[j]$，$\text{sent}[i]\neq\text{sent}[j]$，$\text{session}[i]=\text{session}[j]$ | 同人同阶段的基线相似度 |
| 同被试异句跨 session | $\text{subj}[i]=\text{subj}[j]$，$\text{sent}[i]\neq\text{sent}[j]$，$\text{session}[i]\neq\text{session}[j]$ | **同一被试跨 session 的相似度** |
| 异句异被试 | $\text{sent}[i]\neq\text{sent}[j]$ 且 $\text{subj}[i]\neq\text{subj}[j]$ | 全局基线 |

同句同被试对（即自身重复）不计入任何组。ZuCo 中每被试每句仅记录一次，故不存在"同句同被试跨 session"组。

#### 步骤 3：统计

每组计算：$\text{mean}$、$\text{std}$、$\text{median}$

**判定排序**：按各组 mean 降序排列，理想情况下应为：同句异被试 > 异句异被试（句子信号存在的证据）。

**跨 session 读解**（致力回答"同被试不同 session 的相似度"）：

| 排序关系 | 含义 |
|---------|------|
| 同被试异句跨session ≈ 同被试异句同session | Session 不引入额外差异，同一被试的跨 session EEG 与同 session 内同等相似 |
| 同被试异句跨session 显著 < 同被试异句同session | Session 效应显著，同一被试在不同 session 的 EEG 表示发生系统性漂移 |
| 同被试异句跨session ≈ 异句异被试 | 同一被试的跨 session EEG 与随机两个人相似，被试个体特征在跨 session 时丢失 |

**噪声对照**：对 $X^\text{noise}$ 运行相同分组余弦相似度分析。噪声下各组 mean 应接近相等（因为随机向量彼此正交，相似度接近 0），且各组间无显著差异。若 EEG 的"同句异被试"组显著高于噪声对应组，说明 EEG 中存在可检测的句子一致性；若"同被试异句跨session"组显著低于同 session 组，则证实 session 效应超出随机波动。

---

### A2-Eta：η² 方差分解

#### 步骤 1：对每个特征维度 $d$（$d = 1, \ldots, 840$）

设 $\mathbf{y}_d = X_{:,d} \in \mathbb{R}^N$

**总体均值与总方差**：

$$\bar{y}_d = \frac{1}{N}\sum_{i=1}^N y_{i,d}$$

$$\text{SS\_total}_d = \sum_{i=1}^{N}(y_{i,d} - \bar{y}_d)^2$$

若 $\text{SS\_total}_d < 10^{-12}$，则令 $\eta^2_{\text{sent},d} = \eta^2_{\text{subj},d} = 0$，跳过。

#### 步骤 2：计算各因子组间方差

**句子因子**（$S$ 个唯一句子）：

$$\text{SS\_sent}_d = \sum_{s=1}^{S} n_s \left(\bar{y}_{s,d} - \bar{y}_d\right)^2, \quad \bar{y}_{s,d} = \frac{1}{n_s}\sum_{i:\text{sent}[i]=s} y_{i,d}$$

**被试因子**（$P$ 个唯一被试）：

$$\text{SS\_subj}_d = \sum_{p=1}^{P} n_p \left(\bar{y}_{p,d} - \bar{y}_d\right)^2, \quad \bar{y}_{p,d} = \frac{1}{n_p}\sum_{i:\text{subj}[i]=p} y_{i,d}$$

**Session 因子**（分组数 $\leq 2$，排除 `session_unknown`）：

$$\text{SS\_session}_d = \sum_{s \in \{1,2\}} n_s \left(\bar{y}_{s,d} - \bar{y}_d\right)^2$$

#### 步骤 3：计算 η²

$$\eta^2_{\text{sent},d} = \frac{\text{SS\_sent}_d}{\text{SS\_total}_d}, \quad \eta^2_{\text{subj},d} = \frac{\text{SS\_subj}_d}{\text{SS\_total}_d}, \quad \eta^2_{\text{session},d} = \frac{\text{SS\_session}_d}{\text{SS\_total}_d}$$

#### 步骤 4：汇总与判定

对 840 个维度取 mean/median/std。

**判定准则**：

$$r_\text{subj\_vs\_sent} = \frac{\text{median}(\eta^2_\text{subj})}{\max(\text{median}(\eta^2_\text{sent}),\ 10^{-12})}$$

$$\text{conclusion} = \begin{cases} \text{subject\_dominant} & r_\text{subj\_vs\_sent} > 3 \\ \text{comparable} & 0.5 < r_\text{subj\_vs\_sent} \leq 3 \\ \text{sentence\_dominant} & r_\text{subj\_vs\_sent} \leq 0.5 \end{cases}$$

**Session 效应判定**：额外计算

$$r_\text{session\_vs\_sent} = \frac{\text{median}(\eta^2_\text{session})}{\max(\text{median}(\eta^2_\text{sent}),\ 10^{-12})}$$

| $r_\text{session\_vs\_sent}$ | 结论 |
|------|------|
| $\gg 1$ | Session 效应主导，跨 session 可比性存疑 |
| $\approx 1$ | Session 效应与句子效应相当 |
| $\ll 1$ | Session 效应微弱，跨 session 数据可安全合并 |

**task1-SR 内部对照**：仅在 `task1-SR` 样本上重跑 η² 分析，分离 session 与 task 混淆。若 task1-SR 内 $\eta^2_\text{session}$ 显著低于全局，说明全局 session 效应主要由 task 差异驱动；若两者相当，则 session 效应独立于 task。

**噪声对照**：对 $X^\text{noise}$ 运行相同 η² 分析。噪声下 $\eta^2_\text{sent} \approx \eta^2_\text{subj} \approx \eta^2_\text{session} \approx 0$（随机向量不携带任何结构性方差）。若 EEG 的 η² 值显著高于噪声，说明方差分解捕捉到真实效应而非数值噪声。

---

### A2-band：频带级 η²

将特征矩阵 $X \in \mathbb{R}^{N \times 840}$ 重塑为 $\tilde{X} \in \mathbb{R}^{N \times 8 \times 105}$

对每个频带 $b$（$b = 0, \ldots, 7$）：

$$\text{band\_feat}_b = \tilde{X}_{:,b,:} \in \mathbb{R}^{N \times 105}$$

对该频带的每个通道 $d$（$d = 0, \ldots, 104$），执行与 A2-Eta 相同的 η² 计算流程。

输出：每个频带的 $\text{median}(\eta^2_\text{sent})$ 和 $\text{median}(\eta^2_\text{subj})$（以及 mean）。

频带顺序：theta1、theta2、alpha1、alpha2、beta1、beta2、gamma1、gamma2

**Session 频带级分解**：同时对每个频带计算 $\eta^2_\text{session}$，识别 session 效应在哪些频带更显著（预期低频带 theta/alpha 更易受疲劳等因素影响）。输出字段在 `band_level_eta_squared.json` 中与句子/被试的频带 η² 并列。

---

### A2-tSNE：t-SNE 可视化

#### 步骤 1：PCA 预降维

$$X_\text{pca} = \text{PCA}(X,\ n\_\text{components}=\min(50, 840),\ \text{random\_state}=42) \in \mathbb{R}^{N \times 50}$$

#### 步骤 2：t-SNE 降维

$$X_\text{tsne} = \text{t-SNE}(X_\text{pca},\ n\_\text{components}=2,\ n\_\text{iter}=1000,\ \text{random\_state}=42)$$

对三个 perplexity 值运行：$p \in \{5, 30, 50\}$

#### 步骤 3：可视化输出

| perplexity | 着色方案 |
|-----------|---------|
| 5, 30, 50 | 按被试 ID |
| 30 only | 按句子 ID |
| 30 only | 按 task |
| 30 only | 按 session（新增）|

若"按 session"图中两个 session 呈现明显聚类，说明 session 效应视觉可见；若两 session 样本均匀混合，则 session 效应微弱。同一被试的两 session 样本可进一步用线段连接，观察是否沿固定方向偏移。

输出文件：`tsne_by_{subject|sentence|task|session}_p{5|30|50}.png`

---

## 实验 A3：去被试化信号恢复验证

**触发条件**：A2-Eta 结论为 `subject_dominant`（或无条件执行作为参考，取决于 `--skip-a3`）

**目的**：验证去除被试个体特征后，EEG 是否仍保留句子信号。包含两个子实验。

**入口函数**：`run_desubject_analysis(data_path, output_dir, logger)`

### A3-LP：去被试化 Linear Probe

**数据**：测试集（`test_data`），与 A1 使用相同的数据范围。

#### 步骤 1：LOSO 框架下计算 per-subject 统计量（无数据泄露）

采用与 A1a-LOSO 相同的 `StratifiedGroupKFold` 划分，共 $K$ 折。

在第 $k$ 折的**训练 fold** 内部，对每位被试 $p$ 计算统计量：

$$\mathcal{I}_p^{(k)} = \{i \in \mathcal{D}_\text{train}^{(k)} : \text{subj}[i] = p\}$$

$$\mu_{p,d}^{(k)} = \frac{1}{|\mathcal{I}_p^{(k)}|} \sum_{i \in \mathcal{I}_p^{(k)}} X_{i,d}$$

$$\sigma_{p,d}^{(k)} = \sqrt{\frac{1}{|\mathcal{I}_p^{(k)}|} \sum_{i \in \mathcal{I}_p^{(k)}} (X_{i,d} - \mu_{p,d}^{(k)})^2}$$

> **注**：统计量仅在 train fold 内计算，测试 fold 中被试的统计量不参与 fit，避免泄露。

#### 步骤 2：对 train/test fold 分别应用 per-subject z-score 归一化

$$\tilde{X}_{i,d} = \frac{X_{i,d} - \mu_{p(i),d}^{(k)}}{\max(\sigma_{p(i),d}^{(k)},\ 10^{-8})}$$

- $p(i)$：样本 $i$ 所属被试
- 对于测试 fold 中的被试：若其统计量已在训练 fold 计算（LOSO 保证测试被试不在训练 fold），则使用训练 fold 内其他折的统计量，否则不归一化

#### 步骤 3：在归一化特征上运行 Linear Probe

与 A1a-LOSO 步骤相同（Scaler + LogisticRegression），但输入特征换为归一化后的 $\tilde{X}$。

$K$ 折均值 ± std 与 A1a-LOSO 结果对比，量化去被试化的效果：

$$\Delta\text{Top-1} = \text{Top-1}_\text{A3-LP} - \text{Top-1}_\text{A1a-LOSO}$$

若 $\Delta > 0$，说明去被试化提升了句子信号的可检测性。

---

### A3-Retrieval：被试聚合检索（分组交叉验证）

**目的**：将同一句子跨被试的 EEG 聚合，检验聚合后的句子表示是否具有跨组一致性。

**所用特征**：`test_feats`（`test_data` 上的原始 `mean_pool` 特征，**未**经过 per-subject 去被试化归一化）

#### 步骤 1：被试分组

将测试集中 $P$ 位被试按字典序等分为两组：

$$\text{group\_A} = \text{sorted\_subj}[0 : \lfloor P/2 \rfloor], \quad \text{group\_B} = \text{sorted\_subj}[\lfloor P/2 \rfloor :]$$

$$|\text{group\_A}| = \lfloor P/2 \rfloor, \quad |\text{group\_B}| = P - \lfloor P/2 \rfloor$$

#### 步骤 2：组内按句子聚合（均值）

$$\mathbf{v}^A_s = \frac{1}{|\{i : \text{subj}[i] \in A, \text{sent}[i]=s\}|} \sum_{i : \text{subj}[i] \in A, \text{sent}[i]=s} \mathbf{x}_i$$

$$\mathbf{v}^B_s = \frac{1}{|\{i : \text{subj}[i] \in B, \text{sent}[i]=s\}|} \sum_{i : \text{subj}[i] \in B, \text{sent}[i]=s} \mathbf{x}_i$$

仅保留两组均有数据的 $M$ 个公共句子 `common_sids`。若 $M = 0$，输出 `{"error": "no_common_sentences"}` 并跳过。

#### 步骤 3：L2 归一化（`.clip(min=1e-8)`）

$$\hat{\mathbf{u}}_s = \frac{\mathbf{v}^A_s}{\max(\|\mathbf{v}^A_s\|_2, 10^{-8})}, \quad \hat{\mathbf{v}}_s = \frac{\mathbf{v}^B_s}{\max(\|\mathbf{v}^B_s\|_2, 10^{-8})}$$

#### 步骤 4：余弦相似度矩阵

$$\mathbf{S} \in \mathbb{R}^{M \times M}, \quad S_{ij} = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{v}}_j$$

query = group\_A，candidate = group\_B（避免自检索，$S_{ii} \neq 1$）

#### 步骤 5：检索排名

对每个 query $i$，按 $\mathbf{S}_{i,:}$ 降序排列，正确答案为下标 $i$：

$$r_i = \text{position of } i \text{ in } \text{argsort}(\mathbf{S}_{i,:})[\text{::-1}] + 1$$

#### 步骤 6：检索指标

$$\text{R@K} = \frac{1}{M}\sum_{i=1}^{M} \mathbf{1}[r_i \leq K], \quad K \in \{1, 5, 10\}$$

$$\text{MRR} = \frac{1}{M}\sum_{i=1}^{M} \frac{1}{r_i}$$

$$\text{Mean Rank} = \frac{1}{M}\sum_{i=1}^{M} r_i, \quad \text{Median Rank} = \text{median}(\{r_i\})$$

**噪声对照**：对 $X^\text{noise}$ 运行相同聚合检索流程。噪声向量聚合后仍为随机方向，理论期望为：

$$\mathbb{E}[r_i^\text{noise}] = \frac{M+1}{2}, \quad \text{R@1}^\text{noise} \approx \frac{1}{M} \approx 0.77\%$$

若 EEG 的 R@1 / MRR 显著超越噪声基线，说明跨被试组聚合后 EEG 表示仍保留句子级一致性。

---

### A3-SessionRetrieval：同被试跨 Session 聚合检索

**目的**：直接量化"同一被试在不同 session 的 EEG 相似度"。将检索单位换为 session 内按句子聚合的向量，query 为 Session 1 表示，candidate 为 Session 2 表示，检验同一句子在跨 session 时能否互相匹配。

**数据**：测试集中同时出现在两个 session 的被试子集（排除 `session_unknown`），使用未经 per-subject 归一化的 `mean_pool` 特征。

#### 步骤 1：按 session 内聚合（全体被试合并）

对每个句子 $s$ 和每个 session $k \in \{1, 2\}$：

$$\mathbf{v}^{(k)}_s = \frac{1}{|\mathcal{I}_s^{(k)}|} \sum_{i \in \mathcal{I}_s^{(k)}} \mathbf{x}_i, \quad \mathcal{I}_s^{(k)} = \{i : \text{sent}[i] = s,\ \text{session}[i] = k\}$$

仅保留两个 session 均有样本的 $M$ 个公共句子。由于 task1-SR 仅有约 130/2 = 65 句的重叠可能性，实际测试集内 M 受 task1-SR 前后半句子分布限制，若 $M < 5$ 则输出 `{"error": "too_few_common_sentences"}` 并跳过。

#### 步骤 2：L2 归一化与余弦相似度矩阵

$$\hat{\mathbf{u}}_s = \frac{\mathbf{v}^{(1)}_s}{\max(\|\mathbf{v}^{(1)}_s\|_2, 10^{-8})}, \quad \hat{\mathbf{v}}_s = \frac{\mathbf{v}^{(2)}_s}{\max(\|\mathbf{v}^{(2)}_s\|_2, 10^{-8})}$$

$$\mathbf{S} \in \mathbb{R}^{M \times M}, \quad S_{ij} = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{v}}_j$$

query = Session 1 聚合，candidate = Session 2 聚合，避免自检索。

#### 步骤 3：检索指标

与 A3-Retrieval 相同，输出 R@1 / R@5 / R@10 / MRR / Mean Rank / Median Rank。

#### 步骤 4：被试内版本（可选，更直接地测同一被试）

对每位同时在两个 session 的被试 $p$ 单独执行步骤 1–3（只用该被试的样本构建 $\mathbf{v}^{(k)}_s$），获得每人的 R@1 等指标，再对被试取均值 ± std。与全体聚合版本对比可判断 session 内聚合异被试是否增强信号。

**task1-SR 内部对照**：仅在 `task1-SR` 样本上重跑步骤 1–4，排除 task 混淆后的跨 session 检索能力（因 task2-NR 仅在 Session 1、task3-TSR 仅在 Session 2，这两个 task 的句子无法提供跨 session 聚合向量）。

**噪声对照**：对 $X^\text{noise}$ 运行相同流程，预期 R@1 $\approx 1/M$。

**解读**：

| 结果 | 含义 |
|------|------|
| R@1 显著 > 随机基线 | 同一句子在两个 session 的 EEG 表示存在稳定对应，跨 session 相似度可用 |
| R@1 ≈ 随机基线 | 同一句子在两 session 的聚合表示几乎不相关，session 效应完全破坏跨 session 可比性 |
| R@1 介于两者之间 | 存在部分跨 session 信号，但受 session 差异干扰 |
| 被试内 R@1 均值 $\gg$ 全体聚合 R@1 | Session 内聚合异被试的操作引入了额外噪声，个体内跨 session 一致性更强 |

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `linear_probe_results.json` | A1a/A1b/A1c 三种池化变体各含三组（`word_eeg` / `sent_eeg` / `noise`）LOSO 5 折结果；A1d 跨 session Linear Probe 各被试 accuracy 及汇总（含 task1-SR 对照，`word_eeg` / `sent_eeg` / `noise` 三组并列）；A3-LP 去被试化 Linear Probe；A3-Retrieval 被试聚合检索；A3-SessionRetrieval 同被试跨 session 聚合检索（全体 / 被试内 / task1-SR 对照，EEG + Noise）|
| `subject_effect_analysis.json` | A2-Cosine 五组余弦相似度统计（含"同被试异句同 session"/"同被试异句跨 session"，EEG + Noise）；A2-Eta 三因素 η²（sentence / subject / session），含 `r_subj_vs_sent`、`r_session_vs_sent` 判定与 task1-SR 内部对照 |
| `band_level_eta_squared.json` | A2-band 每个频带的 η² 结果，含 sentence / subject / session 三因素频带级分解（EEG）|
| `tsne_by_*.png` | A2-tSNE 可视化图像，命名为 `tsne_by_{subject\|sentence\|task\|session}_p{5\|30\|50}.png`，其中 `session` 着色固定 perplexity=30 |
| `validate_eeg_signal.log` | 运行日志 |

