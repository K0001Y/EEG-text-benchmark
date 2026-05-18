# 实验总结
## 数据处理
1. 数据清洗：丢弃缺失值与异常值
2. 特征拼接：按频带顺序拼接为840维（105*8），单句最多56词，超出截断不足补零
3. 归一化：1D（词向量归一化），2D（词拼接句为矩阵后展平归一），spectrogram归一（整个spectrogram矩阵全局归一）
4. 掩码生成：遮罩padding的0值
5. 数据划分：按句子文本划分80/10/10（不同于模型论文随机划分）
---
## A线实验（仅取130个句子）

### 实验A1
**目的**：用线性分类器检验 EEG 表示是否可区分不同句子，评估句子级语义信息的可检测性。
**线性分类器**： 带softmax的多类逻辑回归，130头（仅在测试集范围内回归）
**十折交叉验证**：每次选取2-3个被试作为测试集

#### A1-a 
词级别池化信号vs原始句子级信号vs噪声

三组信号的特征矩阵 $X \in \mathbb{R}^{N \times 840}$ 分别构造如下：
（显著性）
**词级 EEG**：（平均池化）
$$X_i^{\text{word}} = \frac{1}{T_i}\sum_{t=1}^{T_i} \text{eeg\_word\_norm1d}_{i,t,:}$$

**句级 EEG**：
（降维可视化）图
$$X_i^{\text{sent}} = \frac{\mathbf{s}_i - \text{mean}(\mathbf{s}_i)}{\max(\text{std}(\mathbf{s}_i),\ 10^{-8})}, \quad \mathbf{s}_i = \text{sent\_eeg\_raw}_i$$

**高斯噪声**：
$$X_i^{\text{noise}} \sim \mathcal{N}(\mathbf{0},\ \mathbf{I}_{840}), \quad \text{seed}_i = 42 + i$$

###### 结果：
| 信号组 | Top-1 | Top-5 | Top-10 | 随机基线 |
|--------|-------|-------|--------|----------|
| **词级 EEG** | **1.79%** (±0.84) | **7.67%** | **15.67%** | 0.77% |
| **句级 EEG** | **2.01%** (±0.78) | **9.15%** | **17.14%** | 0.77% |
| **高斯噪声** | 0.96% (±0.58) | 4.62% | 9.95% | 0.77% |

##### 显著性：
**Binomial 二项检验（n=1858, H₀: p=1/130=0.77%）：**
$$p\text{-value} = P(X \ge k) = \sum_{i=k}^{n} \binom{n}{i} p_0^i (1-p_0)^{n-i}$$
| 信号组 | 观测 Top-1 | Δvs 基线 | p 值 | 95%CI 下界 |
|--------|-----------|----------|------|-----------|
| **词级 EEG (A1a)** | 1.78% | +1.01% | **1.5 × 10⁻⁵** ★★★ | 1.30% |
| **句级 EEG (A1a)** | 1.99% | +1.22% | **3.4 × 10⁻⁷** ★★★ | 1.49% |
| 高斯噪声 (A1a) | 0.97% | +0.20% | 0.193 (n.s.) | 0.63% |


**Wilcoxon 折内配对检验（n=10）+ Holm 校正：**

| 对比 | 指标 | Cohen's dz | 原始 p | Holm 校正 p_adj |
|-----|------|------------|--------|-----------------|
| A1a 词级 vs 噪声 | Top-1 | **0.84 (large)** | 0.039 ★ | 0.117 (n.s.) |
| A1a 词级 vs 噪声 | Top-5 | **1.14 (large)** | 0.014 ★ | 0.068 (n.s.) |
| A1a 词级 vs 噪声 | Top-10 | **1.28 (large)** | 0.010 ★★ | 0.069 (n.s.) |
| A1a 句级 vs 噪声 | Top-1 | **1.08 (large)** | 0.020 ★ | — |
| A1a 句级 vs 噪声 | Top-5 | **1.45 (large)** | 0.008 ★★ | — |
| A1a 句级 vs 噪声 | Top-10 | **1.54 (large)** | 0.006 ★★ | — |
| A1a 词级 vs 句级 | Top-1 | −0.27 (small) | 0.438 (n.s.) | — |
| A1a 词级 vs 句级 | Top-5 | −0.74 (medium) | 0.037 ★ | — |
| A1a 词级 vs 句级 | Top-10 | −0.77 (medium) | 0.049 ★ | — |

> 注：Holm 校正仅覆盖 9 个 word_vs_noise 测试（A1a/A1b/A1c × Top1/5/10），最小 p_adj=0.053（A1b Top-10），恰超 α=0.05。句级 vs 噪声、词级 vs 句级未纳入该族。

**解读：10 折后 Wilcoxon 原始 p 已显著，Holm 校正仍不通过**

1. **5→10 折消除 p 下界**：n=5 时 Wilcoxon 最小可达 p = $2/2^5 = 0.0625$，无论效应量多大均无法显著。n=10 后最小可达 p = $2/2^{10} = 0.002$，词级 vs 噪声 Top-10 原始 $p = 0.010$，句级 vs 噪声 Top-10 原始 $p = 0.006$，均在 $\alpha = 0.05$ 下显著。

2. **Holm 校正抵消功效增益**：9 个 word_vs_noise 测试中，最小原始 $p = 0.006$，Holm 修正 $p_{\text{adj}} = 0.006 \times 9 = 0.054$，恰超 $\alpha = 0.05$。折数增加带来的功效提升被多重比较乘数部分抵消。

3. **Binomial vs Wilcoxon 根本差异**：Binomial 用 $n=1858$ 条样本检验 $H_0: p = 1/130$，标准误 $\text{SE} = \sqrt{p_0(1-p_0)/n} \approx 0.203\%$，词级 EEG 偏离基线 $\approx 5\text{ SE}$；Wilcoxon 用 $n=10$ 折层面差异，功效受限于折数与 Holm 乘数之比。

4. **效应量一致性**：Cohen's $d_z$ 范围 0.84–1.54（均为 large），与 Binomial $p < 10^{-5}$ 共同支撑"EEG 显著优于噪声"。Holm 校正后 n.s. 属于多重比较校正偏保守，而非效应不存在。

#### A1-b
使用注视时长对每个词级别信号进行加权
$$\mathbf{f}_i = \sum_{t=1}^{T_i} w_{i,t} \cdot \text{eeg\_valid}_{i,t,:}, \quad w_{i,t} = \frac{\text{dur}_{i,t}}{\sum_{t'} \text{dur}_{i,t'}}$$

**结果**
| 信号组 | Top-1 | Top-5 | Top-10 | 备注 |
|--------|-------|-------|--------|------|
| 词级 EEG | 1.65% (±0.81) | 8.88% | 15.62% | nfixations加权 |
| 句级 EEG | 2.01% (±0.78) | 9.15% | 17.14% | fallback至A1a |
| 高斯噪声 | 0.96% (±0.58) | 4.62% | 9.95% | fallback至A1a |

#### A1-c
频带分解后进行分类（公式上与平均池化等价）
$$\mathbf{f}_i^{\text{word}} = \text{flatten}\!\left(\frac{1}{T_i}\sum_{t} \text{eeg\_valid}_{i,t,:}.reshape(8,105)\right)$$

#### A1-d
跨session实验：取一个被试的所有样本，针对不同session进行分类（二分类），12名有双Session数据的被试各做5折Session分类

**结果**

| 信号组 | Acc (均值±std) | Baseline | Δvs Baseline |
|--------|---------------|----------|-------------|
| **词级 EEG** | **98.85%** (±1.51) | 58.91% | **+39.94%** |
| **句级 EEG** | **97.42%** (±2.46) | 58.91% | **+38.51%** |
| 高斯噪声 | 57.26% (±4.87) | 58.91% | -1.65% |

**Task1-SR 子集**

| 信号组 | Acc (均值±std) | Baseline | Δvs Baseline |
|--------|---------------|----------|-------------|
| **词级 EEG** | **99.31%** (±1.22) | 60.46% | **+38.85%** |
| **句级 EEG** | **96.18%** (±5.24) | 60.46% | **+35.72%** |
| 高斯噪声 | 58.29% (±5.40) | 60.46% | -2.17% |

##### 显著性
**Wilcoxon 配对检验（n=12 被试）—— Overall：**

| 对比 | W 统计量 | p 值 | Cohen's dz | mean_diff | 显著性 |
|------|---------|------|------------|-----------|--------|
| word_eeg acc vs baseline | 78.0 | **2.44 × 10⁻⁴** ★★★ | 15.62 (large) | +0.3994 | ★★★ |
| sent_eeg acc vs baseline | 78.0 | **2.44 × 10⁻⁴** ★★★ | 10.94 (large) | +0.3851 | ★★★ |
| noise acc vs baseline | 27.0 | 0.830 (n.s.) | −0.37 (medium) | −0.0165 | n.s. |
| word_eeg vs noise | 78.0 | **2.44 × 10⁻⁴** ★★★ | 7.02 (large) | +0.4159 | ★★★ |
| sent_eeg vs noise | 78.0 | **2.44 × 10⁻⁴** ★★★ | 7.08 (large) | +0.4016 | ★★★ |
| word_eeg vs sent_eeg | 1.0 | 0.031 ★ | 0.72 (medium) | +0.0143 | ★ |

**Wilcoxon 配对检验（n=12 被试）—— Task1-SR 子集：**

| 对比 | W 统计量 | p 值 | Cohen's dz | mean_diff | 显著性 |
|------|---------|------|------------|-----------|--------|
| word_eeg acc vs baseline | 78.0 | **2.44 × 10⁻⁴** ★★★ | 14.89 (large) | +0.3885 | ★★★ |
| sent_eeg acc vs baseline | 78.0 | **2.44 × 10⁻⁴** ★★★ | 6.31 (large) | +0.3572 | ★★★ |
| noise acc vs baseline | 17.5 | 0.857 (n.s.) | −0.39 (medium) | −0.0217 | n.s. |
| word_eeg vs noise | 78.0 | **2.44 × 10⁻⁴** ★★★ | 7.89 (large) | +0.4101 | ★★★ |
| sent_eeg vs noise | 78.0 | **2.44 × 10⁻⁴** ★★★ | 7.58 (large) | +0.3789 | ★★★ |
| word_eeg vs sent_eeg | 0.0 | 0.031 ★ | 0.61 (medium) | +0.0313 | ★ |

---

### 实验A2

#### A2-a 计算余弦相似度
$$\text{cos}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \cdot \|\mathbf{x}_j\|_2}$$

矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，$S_{ij} = \text{cos}(\mathbf{x}_i, \mathbf{x}_j)$

**结果**
| 比较组 | 样本数 | 均值 | 含义 |
|--------|--------|------|------|
| 同被试异句 | 76,976 | **0.8052** | 被试内部高度相似 |
| 同句异被试 | 13,990 | 0.5798 | 跨被试句子信号微弱 |
| 异句异被试 | 1,634,121 | 0.5722 | 背景噪声水平 |

#### A2-b η² 方差分解
对每个特征为度d

设 $\mathbf{y}_d = X_{:,d} \in \mathbb{R}^N$

**总体均值与总方差**：

$$\bar{y}_d = \frac{1}{N}\sum_{i=1}^N y_{i,d}$$

$$\text{SS\_total}_d = \sum_{i=1}^{N}(y_{i,d} - \bar{y}_d)^2$$

若 $\text{SS\_total}_d < 10^{-12}$，则令 $\eta^2_{\text{sent},d} = \eta^2_{\text{subj},d} = 0$，跳过。

**句子因子**（$S$ 个唯一句子）：

$$\text{SS\_sent}_d = \sum_{s=1}^{S} n_s \left(\bar{y}_{s,d} - \bar{y}_d\right)^2, \quad \bar{y}_{s,d} = \frac{1}{n_s}\sum_{i:\text{sent}[i]=s} y_{i,d}$$

**被试因子**（$P$ 个唯一被试）：

$$\text{SS\_subj}_d = \sum_{p=1}^{P} n_p \left(\bar{y}_{p,d} - \bar{y}_d\right)^2, \quad \bar{y}_{p,d} = \frac{1}{n_p}\sum_{i:\text{subj}[i]=p} y_{i,d}$$

**Session 因子**（分组数 $\leq 2$，排除 `session_unknown`）：

$$\text{SS\_session}_d = \sum_{s \in \{1,2\}} n_s \left(\bar{y}_{s,d} - \bar{y}_d\right)^2$$

**计算 η²**

$$\eta^2_{\text{sent},d} = \frac{\text{SS\_sent}_d}{\text{SS\_total}_d}, \quad \eta^2_{\text{subj},d} = \frac{\text{SS\_subj}_d}{\text{SS\_total}_d}, \quad \eta^2_{\text{session},d} = \frac{\text{SS\_session}_d}{\text{SS\_total}_d}$$

##### 结果

| 因素 | η² 中位数 [EEG] | η² 中位数 [噪声] |
|------|----------------|-----------------|
| **被试效应** | **0.4813** | 0.0152 |
| 句子效应 | 0.0697 | 0.0695 |
| 两者比值 | **6.9×** | 0.22× |

##### 显著性 
**η² 方差分解（n=840 维，Wilcoxon 配对检验 η²_subject vs η²_sentence）：**
- 统计量 W=0.0，**p = 4.1 × 10⁻¹³⁹** ★★★
- Cohen's dz = **3.05 (large)**
- 均值差 = 0.3875，95%CI = [0.379, 0.396]

**5 维 Permutation 检验（n_perm=200）：**

| 特征维 | 句子效应 p | 被试效应 p |
|--------|-----------|-----------|
| 647 | 1.0 (n.s.) | **0.005** ★★ |
| 368 | **0.005** ★★ | **0.005** ★★ |
| 548 | **0.005** ★★ | **0.005** ★★ |
| 74 | 0.97 (n.s.) | **0.005** ★★ |
| 363 | **0.005** ★★ | **0.005** ★★ |

#### A2-c 频带 η² 方差分解

| 频带 | η²_句子 (median) | η²_被试 (median) | 比值 |
|------|-----------------|-----------------|------|
| theta1 | 0.0595 | 0.4575 | 7.7× |
| theta2 | 0.0565 | 0.4511 | 8.0× |
| alpha1 | 0.0618 | 0.4597 | 7.4× |
| alpha2 | 0.0830 | 0.4740 | 5.7× |
| beta1 | 0.0884 | 0.5131 | 5.8× |
| beta2 | 0.0695 | 0.4861 | 7.0× |
| gamma1 | 0.0645 | 0.4417 | 6.8× |
| **gamma2** | **0.2019** | **0.5808** | **2.9×** |

---

### 实验A3 去被试化验证实验
#### A3-a

**步骤 1**：LOSO 框架下计算 per-subject 统计量（无数据泄露）

在第 $k$ 折的**训练 fold** 内部，对每位被试 $p$ 计算统计量：

$$\mathcal{I}_p^{(k)} = \{i \in \mathcal{D}_\text{train}^{(k)} : \text{subj}[i] = p\}$$

$$\mu_{p,d}^{(k)} = \frac{1}{|\mathcal{I}_p^{(k)}|} \sum_{i \in \mathcal{I}_p^{(k)}} X_{i,d}$$

$$\sigma_{p,d}^{(k)} = \sqrt{\frac{1}{|\mathcal{I}_p^{(k)}|} \sum_{i \in \mathcal{I}_p^{(k)}} (X_{i,d} - \mu_{p,d}^{(k)})^2}$$

**步骤 2**：对 train/test fold 分别应用 per-subject z-score 归一化

$$\tilde{X}_{i,d} = \frac{X_{i,d} - \mu_{p(i),d}^{(k)}}{\max(\sigma_{p(i),d}^{(k)},\ 10^{-8})}$$

- $p(i)$：样本 $i$ 所属被试
- 对于测试 fold 中的被试：若其统计量已在训练 fold 计算（LOSO 保证测试被试不在训练 fold），则使用训练 fold 内其他折的统计量，否则不归一化

**步骤 3**：在归一化特征上运行 Linear Probe

##### 结果

| 实验 | Top-1 | Top-5 | Top-10 |
|------|-------|-------|--------|
| A1a 原始词级EEG | **1.79%** (±0.84) | **7.67%** | **15.67%** |
| A3-LP 去被试化后 | 0.99% (±0.51) | 4.36% | 8.74% |
| 变化 | **-0.80%** | **-3.31%** | **-6.93%** |

##### 显著性
**A3-LP vs A1a 词级（Wilcoxon 配对 n=10）：**

| 指标 | mean_diff | Cohen's dz | 原始 p |
|-----|----------|-----------|--------|
| Top-1 | -0.80% | **-0.96 (large)** | 0.023 ★ |
| Top-5 | -3.32% | **-1.64 (large)** | 0.004 ★★ |
| Top-10 | -6.93% | **-1.72 (large)** | 0.002 ★★ |

#### A3-c：被试聚合检索（分组交叉验证）

**目的**：将同一句子跨被试的 EEG 聚合，检验聚合后的句子表示是否具有跨组一致性。

**步骤 1**：被试分组

将测试集中 $P$ 位被试按字典序等分为两组：

$$\text{group\_A} = \text{sorted\_subj}[0 : \lfloor P/2 \rfloor], \quad \text{group\_B} = \text{sorted\_subj}[\lfloor P/2 \rfloor :]$$

$$|\text{group\_A}| = \lfloor P/2 \rfloor, \quad |\text{group\_B}| = P - \lfloor P/2 \rfloor$$

**步骤 2**：组内按句子聚合（均值）

$$\mathbf{v}^A_s = \frac{1}{|\{i : \text{subj}[i] \in A, \text{sent}[i]=s\}|} \sum_{i : \text{subj}[i] \in A, \text{sent}[i]=s} \mathbf{x}_i$$

$$\mathbf{v}^B_s = \frac{1}{|\{i : \text{subj}[i] \in B, \text{sent}[i]=s\}|} \sum_{i : \text{subj}[i] \in B, \text{sent}[i]=s} \mathbf{x}_i$$

**步骤 3***：L2 归一化（`.clip(min=1e-8)`）

$$\hat{\mathbf{u}}_s = \frac{\mathbf{v}^A_s}{\max(\|\mathbf{v}^A_s\|_2, 10^{-8})}, \quad \hat{\mathbf{v}}_s = \frac{\mathbf{v}^B_s}{\max(\|\mathbf{v}^B_s\|_2, 10^{-8})}$$

**步骤 4**：余弦相似度矩阵
（降维K近邻）
$$\mathbf{S} \in \mathbb{R}^{M \times M}, \quad S_{ij} = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{v}}_j$$

**步骤 5**：检索排名

$$r_i = \text{position of } i \text{ in } \text{argsort}(\mathbf{S}_{i,:})[\text{::-1}] + 1$$

**步骤 6**：检索指标

$$\text{R@K} = \frac{1}{M}\sum_{i=1}^{M} \mathbf{1}[r_i \leq K], \quad K \in \{1, 5, 10\}$$

$$\text{MRR} = \frac{1}{M}\sum_{i=1}^{M} \frac{1}{r_i}$$

$$\text{Mean Rank} = \frac{1}{M}\sum_{i=1}^{M} r_i, \quad \text{Median Rank} = \text{median}(\{r_i\})$$

**结果**

| 信号组 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|--------|-----|-----|------|-----|-----------|
| **EEG** | **0.00%** | **11.76%** | **26.47%** | **0.1024** | **18.06** |
| **噪声** | 2.94% | 17.65% | 20.59% | 0.1203 | 19.12 |



---

#### A3-SessionRetrieval：同被试跨 Session 聚合检索

**目的**：直接量化"同一被试在不同 session 的 EEG 相似度"。将检索单位换为 session 内按句子聚合的向量，query 为 Session 1 表示，candidate 为 Session 2 表示，检验同一句子在跨 session 时能否互相匹配。

**数据**：测试集中同时出现在两个 session 的被试子集（排除 `session_unknown`），使用未经 per-subject 归一化的 `mean_pool` 特征。

**步骤 1**：按 session 内聚合（全体被试合并）

对每个句子 $s$ 和每个 session $k \in \{1, 2\}$：

$$\mathbf{v}^{(k)}_s = \frac{1}{|\mathcal{I}_s^{(k)}|} \sum_{i \in \mathcal{I}_s^{(k)}} \mathbf{x}_i, \quad \mathcal{I}_s^{(k)} = \{i : \text{sent}[i] = s,\ \text{session}[i] = k\}$$

仅保留两个 session 均有样本的 $M$ 个公共句子。由于 task1-SR 仅有约 130/2 = 65 句的重叠可能性，实际测试集内 M 受 task1-SR 前后半句子分布限制，若 $M < 5$ 则输出 `{"error": "too_few_common_sentences"}` 并跳过。

**步骤 2**：L2 归一化与余弦相似度矩阵

$$\hat{\mathbf{u}}_s = \frac{\mathbf{v}^{(1)}_s}{\max(\|\mathbf{v}^{(1)}_s\|_2, 10^{-8})}, \quad \hat{\mathbf{v}}_s = \frac{\mathbf{v}^{(2)}_s}{\max(\|\mathbf{v}^{(2)}_s\|_2, 10^{-8})}$$

$$\mathbf{S} \in \mathbb{R}^{M \times M}, \quad S_{ij} = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{v}}_j$$

**结果**

| 信号组 | R@1 | R@5 | R@10 | MRR | Mean Rank | 随机基线 R@1 |
|--------|-----|-----|------|-----|-----------|-------------|
| **EEG** | 0.00% | 100% | 100% | **0.323** | **3.4** | 20% |
| **噪声** | 0.00% | 100% | 100% | 0.237 | 4.4 | 20% |

> 注：仅 5 个公共句子（M=5），R@5/R@10 的 baseline=100%，Binomial 检验已按规范跳过。

##### 显著性
**Binomial 二项检验（n=5, H₀: p₀=20%）：**
- EEG R@1=0% vs 基线：p=1.0 (n.s.)
- 噪声 R@1=0% vs 基线：p=1.0 (n.s.)
- R@5/R@10：k ≥ M，baseline ≥ 1.0，已按规范跳过（先前版本产生的 `p (2.0) must be in range [0,1]` 越界错误已修复）⚠️ 数据修复

## B线实验
使用模型的编码器对EEG信号进行编码，计算编码后的EEG向量与文本向量的相似度。进行检索

### b1 CET-MAE

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | **1.08%** | **5.06%** | **8.88%** | **0.0484** | 63.5 |
| gaussian | 0.75% | 4.04% | 7.91% | 0.0426 | 64.3 |
| shuffle | 0.92% | 4.57% | 8.34% | 0.0448 | 66.0 |
| zero | 0.65% | 4.47% | 8.40% | 0.0425 | 65.2 |

**诊断**：real 与 gaussian 的所有维度在 Wilcoxon + BH-FDR 下均不显著（p_adj ∈ [0.249, 0.569]），数值差异 +0.0058 仅为描述性趋势。但 real 对 shuffle 和 zero 的 Mean-Rank 在 n=1858 下显著更低（p_adj=0.012, dz≈−0.09）。

**结论**：**模式 B（弱化版）** —— 模型对 "句子级分布结构" 有感知（可区分 shuffle 打乱 / zero 全零），但未学到 "真实 EEG vs 同分布噪声" 的判别性表征。⚠️ 数据修复：旧版结论为"模式B—学到EEG统计特性"，v5修正为弱化版

### b2 EEG-To-Text

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | 0.65% | 4.31% | 7.75% | 0.0445 | 63.3 |
| gaussian | 0.65% | 4.31% | 7.80% | 0.0445 | 63.3 |
| shuffle | 0.65% | 4.31% | 7.75% | 0.0444 | 63.3 |
| zero | 0.65% | 4.31% | 7.70% | 0.0444 | 63.4 |

**诊断**：所有条件差异 < 0.001

**结论**：**模式 A** — 编码器完全无效，未从 EEG 学到任何信息。

### b3 EEG2Text

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | **1.18%** | 3.93% | 8.07% | **0.0463** | 65.3 |
| gaussian | 0.97% | 4.74% | 8.02% | 0.0448 | 65.3 |
| shuffle | 0.86% | 4.31% | 7.70% | 0.0434 | 64.7 |
| zero | 0.59% | 3.82% | 7.00% | 0.0402 | 63.9 |

**诊断**：zero R@10=7.00% vs real R@10=8.07%，差异较小

**结论**：**模式 A** — 编码器未有效利用 EEG 信号。

### b4 GLIM（最异常结果）

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | 0.97% | 4.47% | 8.83% | 0.0492 | 60.5 |
| gaussian | 0.59% | 6.19% | **12.27%** | 0.0538 | 58.5 |
| shuffle | **1.13%** | 4.41% | 9.36% | 0.0480 | 63.1 |
| **zero** | 0.97% | **7.80%** | **13.02%** | **0.0596** | **56.8** |

**诊断**：**zero R@10=13.02% > real R@10=8.83%**，**zero MRR=0.0596 > real MRR=0.0492**

**结论**：**模式 B + 异常** — 零输入反而更好，存在严重的文本解码器偏差。

### 跨模型同噪声内 Friedman + Nemenyi CD
（n_blocks=1858, Nemenyi CD₀.₀₅=0.109）⚠️ 数据修复
| 噪声条件 | Friedman χ² | p | Kendall W | avg_rank (cet_mae / eeg_to_text / eeg2text / glim) | 显著对 (Nemenyi, ΔRank > 0.109) |
|---------|------------|---|-----------|----------------------------------------------------|---------------------------------|
| **real** | 15.68 | **1.3 × 10⁻³** ★★ | 0.0028 | 2.433 / 2.517 / 2.588 / 2.462 | cet_mae vs eeg2text (ΔR=0.155)；eeg2text vs glim (ΔR=0.126) |
| **gaussian** | 23.33 | **3.4 × 10⁻⁵** ★★★ | 0.0042 | 2.518 / 2.557 / 2.547 / 2.378 | cet_mae vs glim (ΔR=0.140)；eeg_to_text vs glim (ΔR=0.180)；eeg2text vs glim (ΔR=0.170) |
| shuffle | 4.94 | 0.176 (n.s.) | 0.0009 | 2.456 / 2.490 / 2.548 / 2.506 | 无（所有对 ΔRank < CD） |
| **zero** | 45.92 | **5.9 × 10⁻¹⁰** ★★★ | 0.0082 | 2.523 / 2.541 / 2.604 / 2.333 | cet_mae vs glim (ΔR=0.190)；eeg_to_text vs glim (ΔR=0.208)；eeg2text vs glim (ΔR=0.272) |

**核心观察**：
- **Kendall W 普遍偏低**（≤ 0.008）：即便在 zero 条件下 W 也仅 0.0082，说明模型间的排序一致性极弱。
- **real 条件新发现**：修复 S-1 后，real 下 Friedman p=1.3×10⁻³，Nemenyi 定位出 `cet_mae < eeg2text`、`glim < eeg2text` 两个显著对。
- **shuffle 条件由显著转不显著**（旧版: p=4.2×10⁻⁴ → v5: p=0.176）：S-1 坍缩导致的人为偏差被消除。
- **gaussian / zero 下 GLIM 反常**：GLIM 在这两种条件下均获得最低 avg_rank，说明 GLIM 对退化输入尤其敏感。

### 模型内成对显著性（Holm 校正 α=5/n=0.0017；BH FDR 全局 α=0.05；n=1858）⚠️ 数据修复

| 模型 | BH-FDR 显著的对比 (p_adj<0.05) | 方向 | 解读 |
|------|-------------------------------|------|------|
| **CET-MAE** | real_vs_shuffle/**mean_rank** (p_adj=**0.012**, dz=−0.100)；real_vs_zero/**mean_rank** (p_adj=**0.012**, dz=−0.080) | real < shuffle / zero（**real 更优**） | real 相对 shuffle/zero 的 Mean-Rank 显著更低；但 **real vs gaussian 在所有指标上均不显著** (p_adj 0.249–0.569, |dz|<0.03) —— 模式 B 弱化版 |
| **EEG-To-Text** | **无** | — | real/gaussian/shuffle/zero 两两对比在 BH-FDR 下**全部 n.s.**；模式 A |
| **EEG2Text** | **无** | — | real 对任意噪声均不显著（最接近 real_vs_zero/r@1 p_adj=0.056）；模式 A |
| **GLIM** | real_vs_gaussian/r@10 (0.012)；real_vs_shuffle/mean_rank (0.012)；real_vs_zero/**r@5, r@10** (0.012)、**mrr, mean_rank** (0.045)；gaussian_vs_shuffle/r@10 (0.018)、mean_rank (0.012)；gaussian_vs_zero/r@5 (0.018)、mean_rank (0.012)；shuffle_vs_zero/r@5 (0.012)、r@10 (0.018)、mrr (0.034)、mean_rank (0.012) | **gaussian / zero > real、zero > shuffle** | 12 / 16 个全局显著对比来自 GLIM；`real vs zero` 方向一致为零更优；模式 B + 异常 |

**关键结论**：
1. **CET-MAE 有限真实信号**：是唯一对 `real > shuffle / zero` 的 Mean-Rank 显著的模型（两个 p_adj=0.012），但 `real vs gaussian` 所有指标 p_adj > 0.24。
2. **EEG-To-Text / EEG2Text 的"零显著"铁证**：BH-FDR 下**没有任何 real-vs-noise 显著对比**。
3. **GLIM 异常被大幅加强**：旧版 4 对显著对比扩展到 **12 对**，方向**全部指向退化输入更优**。