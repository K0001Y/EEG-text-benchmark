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
| 异句异被试 | $\text{sent}[i]\neq\text{sent}[j]$ 且 $\text{subj}[i]\neq\text{subj}[j]$ | 基线 |

同句同被试对（即自身重复）不计入任何组。

#### 步骤 3：统计

每组计算：$\text{mean}$、$\text{std}$、$\text{median}$

**判定排序**：按各组 mean 降序排列，理想情况下应为：同句异被试 > 异句异被试（句子信号存在的证据）

**噪声对照**：对 $X^\text{noise}$ 运行相同分组余弦相似度分析。噪声下三组 mean 应接近相等（因为随机向量彼此正交，相似度接近 0），且各组间无显著差异。若 EEG 的"同句异被试"组显著高于噪声对应组，说明 EEG 中存在可检测的句子一致性。

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

#### 步骤 3：计算 η²

$$\eta^2_{\text{sent},d} = \frac{\text{SS\_sent}_d}{\text{SS\_total}_d}, \quad \eta^2_{\text{subj},d} = \frac{\text{SS\_subj}_d}{\text{SS\_total}_d}$$

#### 步骤 4：汇总与判定

对 840 个维度取 mean/median/std。

**判定准则**：

$$r = \frac{\text{median}(\eta^2_\text{subj})}{\max(\text{median}(\eta^2_\text{sent}),\ 10^{-12})}$$

$$\text{conclusion} = \begin{cases} \text{subject\_dominant} & r > 3 \\ \text{comparable} & 0.5 < r \leq 3 \\ \text{sentence\_dominant} & r \leq 0.5 \end{cases}$$

**噪声对照**：对 $X^\text{noise}$ 运行相同 η² 分析。噪声下 $\eta^2_\text{sent} \approx \eta^2_\text{subj} \approx 0$（随机向量不携带任何结构性方差）。若 EEG 的 η² 值显著高于噪声，说明方差分解捕捉到真实效应而非数值噪声。

---

### A2-band：频带级 η²

将特征矩阵 $X \in \mathbb{R}^{N \times 840}$ 重塑为 $\tilde{X} \in \mathbb{R}^{N \times 8 \times 105}$

对每个频带 $b$（$b = 0, \ldots, 7$）：

$$\text{band\_feat}_b = \tilde{X}_{:,b,:} \in \mathbb{R}^{N \times 105}$$

对该频带的每个通道 $d$（$d = 0, \ldots, 104$），执行与 A2-Eta 相同的 η² 计算流程。

输出：每个频带的 $\text{median}(\eta^2_\text{sent})$ 和 $\text{median}(\eta^2_\text{subj})$（以及 mean）。

频带顺序：theta1、theta2、alpha1、alpha2、beta1、beta2、gamma1、gamma2

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

输出文件：`tsne_by_{subject|sentence|task}_p{5|30|50}.png`

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

## 输出文件

| 文件 | 内容 |
|------|------|
| `linear_probe_results.json` | A1a/A1b/A1c 各含三组（`word_eeg` / `sent_eeg` / `noise`）LOSO 5 折结果、A2 余弦相似度（EEG + Noise）、η² 分析（EEG + Noise）、A3 聚合检索（EEG + Noise）|
| `band_level_eta_squared.json` | A2-band 每个频带的 η² 结果（EEG）|
| `subject_effect_analysis.json` | A2 余弦相似度 + η² 的完整字段（EEG）|
| `tsne_by_*.png` | A2-tSNE 可视化图像 |
| `validate_eeg_signal.log` | 运行日志 |

