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
**五折交叉验证**：每次选取2-3个被试作为测试集

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
| **词级 EEG** | **1.92%** (±0.42) | **8.16%** | **15.71%** | 0.77% |
| **句级 EEG** | **1.87%** (±0.57) | **8.12%** | **15.73%** | 0.77% |
| **高斯噪声** | 1.03% (±0.55) | 4.88% | 9.90% | 0.77% |

##### 显著性：
**Binomial 二项检验（n=1858, H₀: p=1/130=0.77%）：**
$$p\text{-value} = P(X \ge k) = \sum_{i=k}^{n} \binom{n}{i} p_0^i (1-p_0)^{n-i}$$
| 信号组 | 观测 Top-1 | Δvs 基线 | p 值 | 95%CI 下界 |
|--------|-----------|----------|------|-----------|
| **词级 EEG (A1a)** | 1.9376% | +1.1684% | **9.1 × 10⁻⁷** ★★★ | 1.44% |
| **句级 EEG (A1a)** | 1.8837% | +1.1145% | **2.4 × 10⁻⁶** ★★★ | 1.40% |
| 高斯噪声 (A1a) | 1.0226% | +0.2534% | 0.1333 (n.s.) | 0.67% |


**Wilcoxon 折内配对检验（n=5）+ Holm 校正：**

| 对比 | 指标 | Cohen's dz | 原始 p | Holm 校正 p_adj |
|-----|------|------------|--------|-----------------|
| A1a 词级 vs 噪声 | Top-1 | **0.93 (large)** | 0.1875 | 0.5625 (n.s.) |
| A1a 词级 vs 噪声 | Top-5 | **2.77 (large)** | 0.0625 | 0.5625 (n.s.) |
| A1a 词级 vs 噪声 | Top-10 | **2.37 (large)** | 0.0625 | 0.5625 (n.s.) |
| A1a 句级 vs 噪声 | Top-5 | 1.63 (large) | 0.0625 | 0.5625 (n.s.) |
| A1a 词级 vs 句级 | Top-1 | 0.08 (negligible) | 0.875 | 0.5625 (n.s.) |


#### A1-b
使用注视时长对每个词级别信号进行加权
$$\mathbf{f}_i = \sum_{t=1}^{T_i} w_{i,t} \cdot \text{eeg\_valid}_{i,t,:}, \quad w_{i,t} = \frac{\text{dur}_{i,t}}{\sum_{t'} \text{dur}_{i,t'}}$$

**结果**
| 信号组 | Top-1 | Top-5 | Top-10 | 备注 |
|--------|-------|-------|--------|------|
| 词级 EEG | 1.80% (±0.77) | 7.96% | 15.21% | nfixations加权 |
| 句级 EEG | 1.87% (±0.57) | 8.12% | 15.73% | fallback至A1a |
| 高斯噪声 | 1.03% (±0.55) | 4.88% | 9.90% | fallback至A1a |

#### A1-c
频带分解后进行分类（公式上与平均池化等价）
$$\mathbf{f}_i^{\text{word}} = \text{flatten}\!\left(\frac{1}{T_i}\sum_{t} \text{eeg\_valid}_{i,t,:}.reshape(8,105)\right)$$

#### A1-d
跨session实验（尚未实现）
**思路**：取一个被试的所有样本，针对不同session进行分类（二分类）

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
| A1a 原始词级EEG | **1.92%** (±0.42) | **8.16%** | **15.71%** |
| A3-LP 去被试化后 | 1.02% (±0.15) | 4.46% | 9.17% |
| 变化 | **-0.90%** | **-3.70%** | **-6.54%** |

##### 显著性
**A3-LP vs A1a 词级（Wilcoxon 配对 n=5）：**

| 指标 | mean_diff | Cohen's dz | 原始 p |
|-----|----------|-----------|--------|
| Top-1 | -0.90% | **-1.57 (large)** | 0.125 (n.s.) |
| Top-5 | -3.70% | **-2.60 (large)** | 0.0625 (n.s.) |
| Top-10 | -6.54% | **-2.89 (large)** | 0.0625 (n.s.) |

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

---

## B线实验
使用模型的编码器对EEG信号进行编码，计算编码后的EEG向量与文本向量的相似度。进行检索

### b1 CET-MAE

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | **1.08%** | **5.06%** | **8.88%** | **0.0484** | 63.5 |
| gaussian | 0.75% | 4.04% | 7.91% | 0.0426 | 64.3 |
| shuffle | 0.92% | 4.57% | 8.34% | 0.0448 | 66.0 |
| zero | 0.65% | 4.47% | 8.40% | 0.0425 | 65.2 |

**诊断**：real MRR (0.0484) > gaussian MRR (0.0426)，差异 +0.0058

**结论**：**模式 B** — 学到 EEG 统计特性，但未学到跨模态对应关系。

### b2 EEG-To-Text

| 条件 | R@1 | R@5 | R@10 | MRR | Mean Rank |
|------|-----|-----|------|-----|-----------|
| **real** | 0.65% | 4.31% | 7.75% | 0.0445 | 63.3 |
| gaussian | 0.65% | 4.31% | 7.80% | 0.0445 | 63.3 |
| shuffle | 0.65% | 4.31% | 7.75% | 0.0444 | 63.3 |
| zero | 0.65% | 4.31% | 7.70% | 0.0444 | 63.4 |

**诊断**：所有条件差异 < 0.001

**结论**：**模式 A** — 编码器完全无效，未从 EEG 学到任何信息。
（p值显著）
（降维可视化）

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
（n_blocks=54, Nemenyi CD₀.₀₅=0.638）

| 噪声条件 | Friedman χ² | p | Kendall W | 显著对 (Nemenyi) |
|---------|------------|---|-----------|------------------|
| **real** | 8.55 | **0.0359** ★ | 0.053 | 无（所有对 ΔRank < CD） |
| **gaussian** | 45.43 | **7.5 × 10⁻¹⁰** ★★★ | 0.280 | cet_mae vs eeg2text/glim, eeg_to_text vs eeg2text/glim, eeg2text vs glim |
| **shuffle** | 18.10 | **4.2 × 10⁻⁴** ★★★ | 0.112 | cet_mae vs eeg2text, eeg_to_text vs eeg2text |
| **zero** | 55.18 | **6.3 × 10⁻¹²** ★★★ | 0.341 | cet_mae vs eeg2text, eeg_to_text vs eeg2text/glim, eeg2text vs glim |

**核心观察**：
- 在 **real** 条件下，模型间**无显著差异**（所有模型都接近随机），与 §2.1 的"四模型均接近随机基线"一致。
- 在**异常输入** (gaussian/shuffle/zero) 下，Kendall W 显著升高（0.28–0.34），且 GLIM 排名跃升为最优（zero 下 avg_rank=3.31），说明模型对退化输入的响应模式**存在系统性差异**——GLIM 对零输入尤其敏感。

### 模型内成对显著性（Holm 校正 α=5/n=0.0017；BH FDR 全局 α=0.05）

| 模型 | BH-FDR 显著的对比 (p_adj<0.05) | 解读 |
|------|-------------------------------|------|
| **CET-MAE** | real_vs_gaussian MRR (p_adj=0.037), real_vs_zero MRR/MeanRank (p_adj=0.011), shuffle_vs_zero MRR (p_adj=0.020) | **real > gaussian/zero 显著**，支撑"模式 B：学到 EEG 统计特性" |
| **EEG-To-Text** | 仅 shuffle_vs_zero MRR (p_adj=0.011) | real/gaussian/shuffle 之间**全部 n.s.**，定量确认"模式 A：编码器完全无效" |
| **EEG2Text** | gaussian_vs_zero MeanRank (p_adj=0.048), shuffle_vs_zero R@5 (p_adj=0.048) | real 相对于任一噪声**均不显著**，支撑"模式 A：编码器未生效" |
| **GLIM** | real_vs_gaussian MRR/MeanRank, real_vs_zero MRR/MeanRank, gaussian_vs_shuffle MRR/MeanRank, shuffle_vs_zero MRR/MeanRank（全部 p_adj=0.011） | **zero/gaussian 显著优于 real 与 shuffle**，定量确认"模式 B + 异常" |

**关键结论**：
1. **CET-MAE** 是唯一在 BH-FDR 全局校正下 `real > gaussian` 显著的模型（MRR 维度），验证了其微弱但真实的 EEG 信号利用。
2. **EEG-To-Text / EEG2Text** 的 real 与任意噪声差异均未通过 BH-FDR，**统计证据确认编码器失效**。
3. **GLIM** 的 `zero > real` 反常在 MRR 与 MeanRank 上均达到 p_adj=0.011 的显著水平，确认了文本解码器先验主导的架构缺陷。