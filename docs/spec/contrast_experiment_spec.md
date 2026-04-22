# EEG-to-Text 模型性能归因：对比实验规格书（v2 修订版）

> 目标：通过两条独立的诊断线——**噪声对照实验**和**原始数据集有效性验证**——
> 系统性定位四个 EEG-to-Text 模型检索性能接近随机的根本原因。
>
> **创建时间：2026-04**
> **修订时间：2026-04-11**
> **修订说明**：基于 ZuCo 原文数据结构和严谨性分析，修正 v1 版本中的实验设计缺陷

---

## 一、问题背景

### 1.1 当前检索评估结果

四个模型在 test 集（1858 queries, 130 candidates）上的检索性能均接近随机基线：

| 模型 | R@1 | R@5 | R@10 | MRR | Mean Rank | 随机期望 Mean Rank |
|------|-----|-----|------|-----|-----------|--------------------|
| CET-MAE | 1.08% | 5.06% | 8.88% | 0.0484 | 63.5 | 65.5 |
| EEG-To-Text | 0.65% | 4.31% | 7.75% | 0.0445 | 63.3 | 65.5 |
| EEG2Text | 1.18% | 3.93% | 8.07% | 0.0463 | 65.3 | 65.5 |
| GLIM | 0.97% | 4.47% | 8.83% | 0.0492 | 60.5 | 65.5 |

> 随机基线：R@1 ≈ 0.77%（1/130），Mean Rank ≈ 65.5

所有模型 Mean Rank 与随机期望高度接近，R@1 仅略高于或等于随机水平。

### 1.2 待回答的核心问题

```
Q1: 在当前特征提取方式下，ZuCo EEG数据是否包含可检测的句子级语义信息？（数据有效性）
Q2: 模型编码器是否从EEG信号中学到了任何有意义的表征？（模型能力）
Q3: 如果学到了，是跨模态对应关系，还是仅统计偏置？（学习深度）
Q4: 个体差异（被试效应）是否淹没了语义信号？（信号分离）
```

### 1.3 两条诊断线的关系

```
诊断线 A：原始数据集有效性验证          诊断线 B：噪声对照实验
（与模型无关，验证信号本身）           （与数据无关，验证模型能力）
────────────────────────              ──────────────────────
"在当前特征表示下，                    "模型有没有学到EEG信息？"
 EEG数据里有没有可检测的语义信息？"

  Linear Probe / 被试效应分析            Gaussian / Shuffle / Zero 对照
           │                                       │
           ▼                                       ▼
    当前表示下的信息量上界                   模型利用率下界
```

**关键认知**：诊断线 A 的结论是**条件性的**——它回答的是"在当前特征提取方式下"，而非"绝对意义上"。如果 A 线显示信号不可分，可能是：
- (a) 原始 EEG 确实不含句子级语义信息
- (b) 我们的特征提取方式（mean-pooling 等）丢失了信息

**两条线互补**：
- 如果诊断线 A 发现当前表示下信号不可区分 → 需要检验特征提取方式，或尝试增强表示
- 如果诊断线 A 发现当前表示下信号有区分性，但诊断线 B 显示 real ≈ gaussian → 问题在模型
- 两条线结合才能完整归因

---

## 二、ZuCo 数据结构关键事实（基于原文）

### 2.1 被试信息
- **12 名被试**（非 14 名）：ZKW, ZDN, ZPH, ZMG, ZAB, ZJN, ZKH, ZGW, ZJS, ZKB, ZDM, ZJM
- 每个被试阅读 4-6 小时，覆盖 Task 1/2/3

### 2.2 眼动引导的 EEG 分段
- EEG 数据是 **fixation-related**，按眼动注视点分割
- 每个词对应一个 fixation，每个 fixation 对应一段 EEG
- 同步误差 < 2ms（EYEEEG extension）

### 2.3 频带特征（840 维 = 105 通道 × 8 频带）
| 频带 | 频率范围 | 认知功能关联 |
|------|----------|-------------|
| theta1 | 4-6 Hz | 工作记忆、编码 |
| theta2 | 6.5-8 Hz | 工作记忆、编码 |
| alpha1 | 8.5-10 Hz | 抑制、注意 |
| alpha2 | 10.5-13 Hz | 抑制、注意 |
| beta1 | 13.5-18 Hz | 运动准备、语义整合 |
| beta2 | 18.5-30 Hz | 运动准备、语义整合 |
| gamma1 | 30.5-40 Hz | 特征绑定、高级认知 |
| gamma2 | 40-49.5 Hz | 特征绑定、高级认知 |

### 2.4 关键限制：无重复测量
- 每个被试每句话只读**一次**（单次阅读设计）
- 不存在"同句同被试"的重复样本
- 被试间比较是唯一的可行路径

### 2.5 任务分布
| 任务 | 句子数 | 内容特点 |
|------|--------|----------|
| Task 1 (Sentiment) | 400 | 电影评论情感 |
| Task 2 (Normal) | 300 | 维基百科普通阅读 |
| Task 3 (Task-specific) | 407 | 关系提取（有特定指令）|
| 重复句子 | 48 | 跨任务重复 |

### 2.6 模型输入异构性（关键设计约束）

四个模型使用**两种完全不同的 EEG 表示层级**：

| 模型 | 输入来源 | 表示层级 | 数据格式 |
|------|---------|---------|---------|
| **CET-MAE** | `UnifiedDataset` | 词级频域 + 句级频域 | `(max_len, 840)`，句级作为特殊 token 追加 |
| **EEG-To-Text** | `UnifiedDataset` | 词级频域 | `(max_len, 840)` |
| **GLIM** | `UnifiedDataset` | 词级频域 | `(max_len, 840)` |
| **EEG2Text** | **spectro pickle** | **句级时序** | `(24000, 105)`，原始时序信号 |

**关键区别**：
- CET-MAE/EEG-To-Text/GLIM：使用**预计算频域特征**（Hilbert 变换后的 8 频带功率）
- EEG2Text：使用**原始时序信号**，通过 ShallowNet 在运行时提取特征

**对实验设计的影响**：
- EEG2Text 绕过 `UnifiedDataset`，直接读取原始数据文件
- 噪声注入和 shuffle 需要在**不同层级**干预
- 需要**分层统一**策略保证跨模型可比性

---

## 三、诊断线 A：原始数据集有效性验证

> 核心目标：在当前特征表示下，从数据层面验证 ZuCo EEG 是否包含可检测的句子级语义信息。

### 3.1 实验 A1：线性探针上界（Linear Probe）

**原理**：如果当前 EEG 特征空间中句子不可分，任何编码器都不可能做好检索。

**方法变体**（v2 新增多版本对比）：

#### 变体 A1a：词级 Mean-Pool 基线（v1 原始设计）
- 输入：每个样本的词级 EEG 均值池化向量 `(840,)`
- 计算：`eeg_word_norm1d.mean(axis=0)`
- 特点：最简单，但丢失时序信息

#### 变体 A1b：Duration-Weighted Pool（v2 新增）
- 输入：按 fixation duration 加权的词级 EEG
- 计算：`Σ(duration_i × eeg_i) / Σ(duration_i)`
- 原理：注视时间长的词可能携带更多认知信息

#### 变体 A1c：频带分离表示（v2 新增）
- 输入：保持 8 个频带的结构 `(8, 105)` → flatten `(840,)`
- 特点：保留频带间关系，便于后续频带级分析

**共同设置**：
- 任务：130 类分类（130 个 unique 句子）
- 模型：逻辑回归（`sklearn.linear_model.LogisticRegression`），无非线性
- 数据划分：使用已有 train/test split（train 用于拟合，test 用于评估）
- 评估指标：Top-1 / Top-5 / Top-10 accuracy，对比随机基线 0.77%

**结果解读**（v2 修正表述）：

| Linear Probe 结果 | 含义 |
|-------------------|------|
| accuracy ≈ 0.77%（随机） | 在当前表示下，EEG 特征空间中句子不可分。可能原因：(a) 原始信号缺乏信息；(b) 表示方式（如 mean-pooling）丢失了信息。需进一步实验区分。 |
| accuracy 显著 > random（如 > 5%） | 当前表示下信号包含可检测信息，深度模型未有效利用 |
| accuracy 远超 random（如 > 20%） | 当前表示下信息量充足，问题完全在模型端 |
| A1a ≈ random 但 A1b/A1c >> random | mean-pooling 是瓶颈，需要更精细的特征提取 |

### 3.2 实验 A2：被试效应 vs 句子效应分析（v2 重大修正）

**原理**：ZuCo 数据的每个样本同时受**句子语义**和**被试个体差异**两个因素影响。如果被试效应远大于句子效应，个体差异会淹没语义信号。

**v2 关键修正**：由于 ZuCo 是单次阅读设计，**不存在"同句同被试"分组**。修正后的分组策略：

#### 方法 A：余弦相似度分组对比（修正版）

```
对 test 集计算所有样本对的余弦相似度，按关系分三组：

  同句异被试对：cosine(EEG_i, EEG_j) where sentence_i = sentence_j AND subject_i ≠ subject_j
                → 反映句子语义效应（跨被试一致）
  
  同被试异句对：cosine(EEG_i, EEG_j) where subject_i = subject_j AND sentence_i ≠ sentence_j
                → 反映被试个体特征（跨句子一致）
  
  异句异被试对：cosine(EEG_i, EEG_j) where sentence_i ≠ sentence_j AND subject_i ≠ subject_j
                → 基线（无共享因素）

期望排序（如果语义信息存在）：同句异被试 > 同被试异句 > 异句异被试
实际排序（如果被试效应主导）：同被试异句 >> 同句异被试 ≈ 异句异被试
```

**注意事项**：
- "同被试异句对"可能混杂句子属性差异（不同主题、长度、复杂度）
- 建议按 task 分层计算，控制任务效应

#### 方法 B：方差分解（η² 分析）（v2 增强版）

```
对 EEG 特征的每个维度（840维）做 two-way ANOVA：
  Factor A = 句子 ID（130 levels）
  Factor B = 被试 ID（12 levels）

比较：η²(句子) vs η²(被试)

η²(被试) >> η²(句子) → 个体差异主导，语义信号被淹没
η²(句子) ≈ η²(被试) → 两者效应相当
η²(句子) >> η²(被试) → 语义信号强
```

**v2 新增：频带级 η² 分解**

```
对每个频带（8个）分别计算：
  η²_band(句子) vs η²_band(被试)

输出：
  - 每个频带的效应量对比图
  - 识别"语义相关频带"（高 η²(句子)）vs "个体特征频带"（高 η²(被试)）
```

#### 方法 C：降维可视化（v2 增强版）

```
对全体 test 样本的 EEG (mean-pooled, 840-dim) 做 PCA → t-SNE：
  图 A：按被试 ID 着色 → 观察是否呈现被试聚类
  图 B：按句子 ID 着色 → 观察是否呈现句子聚类
  图 C：按 task 着色   → 观察任务效应
  图 D：按频带能量着色 → 观察频带特异性（v2 新增）

参数设置（v2 新增规范）：
  - PCA: n_components=50
  - t-SNE: perplexity=[5, 30, 50], random_seed=42
  - 输出三种 perplexity 的结果，避免参数敏感性
```

### 3.3 实验 A3：去被试化后的信号恢复（v2 增强版）

> 仅在 A2 确认被试效应显著时执行。

**v2 关键修正：数据泄漏防护**

```
被试内 z-score 归一化（修正后流程）：

  步骤 1：在 train 集上，对每个被试 s 计算：
    μ_s = mean(所有属于被试 s 的 train 样本 EEG)
    σ_s = std(所有属于被试 s 的 train 样本 EEG)
  
  步骤 2：对 train 和 test 集应用：(x - μ_s) / σ_s
    ⚠️ 严禁在 test 集上计算 μ_s 或 σ_s
  
  步骤 3：重新执行 A1（Linear Probe）和检索评估
```

**方法 B：被试聚合（同句多被试平均）**

```
对每个句子 t：
  EEG_agg(t) = mean(所有被试看句子 t 的 EEG)
  
  v2 新增：加权平均选项
  EEG_agg_weighted(t) = Σ(duration_i × EEG_i) / Σ(duration_i)

→ 被试级噪声被平均掉，信噪比提升 ~√N（N=被试数，理想情况下 N≤12）
→ 变成 130 query × 130 candidate 的方阵检索
→ 如果此时性能显著提升 → 信号存在但单样本 SNR 太低
```

### 3.4 实验 A 的诊断决策树（v2 修正版）

```
A1: Linear Probe（多版本）
    │
    ├── A1a/b/c 均 ≈ random ──→ A2: 被试效应分析
    │                              │
    │                              ├── η²(被试) >> η²(句子)
    │                              │       → A3: 去被试化
    │                              │              │
    │                              │              ├── 去被试化后 Linear Probe 提升
    │                              │              │   → 结论：在当前表示下，信号存在但被个体差异淹没
    │                              │              │           建议：探索被试归一化策略 / 多被试聚合
    │                              │              │
    │                              │              └── 去被试化后仍 ≈ random
    │                              │                  → 结论：在当前表示下，ZuCo 特征未显示
    │                              │                          可检测的句子级语义信息
    │                              │                          建议：尝试时序模型或更换特征表示
    │                              │
    │                              └── η²(被试) ≈ η²(句子)（都弱）
    │                                     → 结论：当前 EEG 特征维度本身信息稀疏
    │
    ├── A1a ≈ random, 但 A1b/c >> random ──→ 结论：mean-pooling 是瓶颈，需要更精细的特征提取
    │
    └── A1a/b/c 均 >> random ──→ 结论：当前表示下信号有区分性
                                     模型编码器的问题（转诊断线 B）
```

---

## 四、诊断线 B：噪声对照实验

> 核心目标：通过替换/破坏 EEG 输入，验证模型编码器是否从 EEG 中提取了有意义的信息。

### 4.1 三层归因框架

```
Layer 1: 数据管道正确性（Pipeline Bug?）
  │  数据预处理、格式转换、归一化是否引入了信息损失或错误？
  ↓
Layer 2: 模型表征能力（Model Learned Nothing?）
  │  编码器是否从 EEG 中提取到了区分性特征？
  ↓
Layer 3: 信号固有质量（EEG SNR Too Low?）
  │  EEG 信号本身是否包含足够的语义信息供模型学习？
```

**v2 新增：Layer 1 独立验证**
- 可视化检查：随机采样真实 EEG 和 zero 输入的模型内部激活分布
- 梯度检查：确认梯度确实流经 EEG 编码路径

### 4.2 噪声条件设计

| 条件 | 缩写 | 定义 | 诊断目标 |
|------|------|------|----------|
| **真实 EEG** | `real` | 原始数据，不做任何修改 | 基线参照 |
| **高斯噪声** | `gaussian` | 用 N(0,1) 随机信号完全替代 EEG，shape 保持一致 | 编码器是否学到了任何 EEG 特征（vs 纯随机） |
| **配对打乱** | `shuffle` | 使用真实 EEG 数据，但随机打乱 EEG-文本配对关系 | 编码器是否学到了跨模态对应关系（vs 统计偏置） |
| **全零输入** | `zero` | 用全零张量替代 EEG | 排除模型 bias/shortcut 依赖 |

**v2 新增：频谱匹配高斯噪声（可选）**
- 保持 EEG 的功率谱密度 (PSD) 特性，仅打乱相位
- 更好地区分"模型利用频谱特性" vs "模型利用时序结构"

### 4.3 结果解读决策树

```
                    real vs gaussian
                   /                \
           real > gaussian        real ≈ gaussian
          （模型学到了东西）      （模型没学到 EEG 特征）
               |                        |
         real vs shuffle          zero 是否异常？
          /          \              /          \
   real > shuffle  real ≈ shuffle  zero 异常     zero ≈ 随机
   （学到跨模态    （仅学到统计   （pipeline      （编码器
     对应关系）      偏置）       有 bug）        完全无效）
```

**六种典型诊断结论**：

| 情况 | real vs gaussian | real vs shuffle | zero 行为 | 归因 |
|------|------------------|-----------------|-----------|------|
| A | real ≈ gaussian ≈ shuffle ≈ zero | — | ≈ 随机 | **编码器完全无效**，未从 EEG 学到任何信息 |
| B | real > gaussian | real ≈ shuffle | — | 学到 EEG **统计特性**（如功率谱分布），但未学到跨模态对应 |
| C | real > gaussian | real > shuffle | — | **学到了跨模态对应**，但检索天花板受限于信号 SNR |
| D | real ≈ gaussian | — | zero 显著偏离 | **数据管道 bug**：噪声与真实信号经处理后特征坍缩 |
| E | real ≈ gaussian ≈ zero | — | — | 模型存在 **shortcut/bias**，不依赖输入内容 |
| F | real > gaussian, real > shuffle, 但绝对值仍低 | — | — | 模型有效但 **EEG SNR 太低**，语义信息本身稀疏 |

### 4.4 显著性判定（v2 重大增强）

**v1 经验阈值**（保留作为快速判断）：
- Mean Rank 差值 > 3.0
- R@10 差值 > 2 个百分点
- MRR 差值 > 0.005

**v2 新增：统计检验**（推荐作为主要判定依据）

#### 置换检验 (Permutation Test)
```
对 real vs gaussian 的差异，进行 1000 次标签置换：
  1. 合并 real 和 gaussian 的检索结果
  2. 随机重分配标签（real/gaussian）
  3. 计算置换后的指标差异
  4. 重复 1000 次，构建零分布

Empirical p-value = (置换差异 ≥ 观测差异的次数) / 1000

判定：p < 0.05 视为显著差异
```

#### Bootstrap 置信区间
```
对每种条件的指标，进行 1000 次 Bootstrap 重采样：
  1. 有放回地抽取样本（保持样本量不变）
  2. 计算重采样后的指标
  3. 重复 1000 次，得到指标分布
  4. 取 2.5% 和 97.5% 分位数作为 95% CI

判定：如果两个条件的 95% CI 不重叠，视为显著差异
```

#### 效应量 (Cohen's d)
```
对于配对比较（real vs noise），计算 Cohen's d：
  d = (mean_real - mean_noise) / std_pooled

解释：
  |d| < 0.2: 可忽略
  0.2 ≤ |d| < 0.5: 小效应
  0.5 ≤ |d| < 0.8: 中效应
  |d| ≥ 0.8: 大效应
```

---

## 五、实验设计

### 5.1 实验矩阵总览

**诊断线 A（数据有效性）**：3 组实验，无需 GPU

| 编号 | 实验 | 输入 | 工具 | 备注 |
|------|------|------|------|------|
| A1a | Linear Probe (mean-pool) | test 集 EEG mean-pool (840,) | sklearn LogisticRegression | v1 原始设计 |
| A1b | Linear Probe (weighted) | test 集 EEG duration-weighted (840,) | sklearn LogisticRegression | v2 新增 |
| A1c | Linear Probe (band-sep) | test 集 EEG band-separated (840,) | sklearn LogisticRegression | v2 新增 |
| A2 | 被试效应分析 | test 集 EEG mean-pool (840,) | scipy ANOVA / matplotlib | v2 修正分组 |
| A2-band | 频带级 η² 分析 | test 集 EEG per-band (105,) | scipy ANOVA | v2 新增 |
| A3 | 去被试化验证 | per-subject z-score 后的 EEG | sklearn + 检索脚本 | v2 修正数据泄漏 |

**诊断线 B（噪声对照）**：4 模型 × 4 条件 = 16 组，需 GPU

| | real | gaussian | shuffle | zero |
|---|---|---|---|---|
| CET-MAE | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| EEG-To-Text | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| EEG2Text | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| GLIM | ✅ 已有 | 待跑 | 待跑 | 待跑 |

### 5.2 诊断线 A 实施细节

#### A1: Linear Probe（多版本）

```
数据准备：
  从 unified_zuco.pkl 加载 train + test 样本
  
  变体 A1a（mean-pool）：
    eeg_feat = eeg_word_norm1d.mean(axis=0)  → (840,)
  
  变体 A1b（weighted-pool）：
    durations = fixation_durations  # 需要从眼动数据获取
    eeg_feat = Σ(duration_i × eeg_i) / Σ(duration_i)  → (840,)
  
  变体 A1c（band-separated）：
    eeg_reshaped = eeg_word_norm1d.reshape(-1, 8, 105)  # (T, 8, 105)
    eeg_feat = eeg_reshaped.mean(axis=0).flatten()  → (840,)
  
  标签 = 句子文本的唯一 ID（0~129）

模型训练：
  LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
  在 train 集上 fit，在 test 集上 predict

输出指标：
  Top-1 / Top-5 / Top-10 accuracy
  对比随机基线 1/130 = 0.77%
  
v2 新增输出：
  - 三个变体的对比表格
  - 混淆矩阵的条件数
  - per-class accuracy 分布（识别哪些句子不可分）
```

#### A2: 被试效应分析（修正版）

```
余弦相似度分组（修正后）：
  计算 test 集所有样本对 (N×(N-1)/2) 的 cosine similarity
  
  分组统计：
    - 同句异被试：sentence_i = sentence_j, subject_i ≠ subject_j
    - 同被试异句：subject_i = subject_j, sentence_i ≠ sentence_j
    - 异句异被试：sentence_i ≠ sentence_j, subject_i ≠ subject_j
  
  输出：每组的均值、标准差、分布直方图
  
  v2 新增：按 task 分层计算（控制任务效应）

方差分解（含频带级）：
  对 840 维特征逐维做 two-way ANOVA：
    Factor A = 句子 ID（130 levels）
    Factor B = 被试 ID（12 levels）
  
  汇总：median η²(句子) vs median η²(被试)
  
  v2 新增频带级分析：
    对每个频带 b ∈ {theta1, theta2, alpha1, alpha2, beta1, beta2, gamma1, gamma2}：
      提取该频带的 105 维特征
      做 two-way ANOVA
      记录 η²_sentence(b) 和 η²_subject(b)
    输出：8 个频带的效应量对比图

可视化：
  PCA(n_components=50) → t-SNE(n_components=2, perplexity=[5,30,50], random_state=42)
  分别按被试/句子/task 着色
  v2 新增：输出三种 perplexity 的结果
```

#### A3: 去被试化（v2 修正数据泄漏）

```
被试内归一化（修正后流程）：
  步骤 1（仅在 train 集计算统计量）：
    for each subject s in train_set:
      μ_s = mean(all EEG samples from subject s in train_set)
      σ_s = std(all EEG samples from subject s in train_set)
      save μ_s, σ_s
  
  步骤 2（应用到 train 和 test）：
    for each sample x from subject s:
      x_normalized = (x - μ_s) / σ_s
  
  步骤 3：重新执行 A1 和检索评估

被试聚合检索：
  对每个句子 t：
    subjects_t = all subjects who read sentence t
    EEG_agg(t) = mean([EEG_{s,t} for s in subjects_t])
    
    v2 新增加权版本：
    EEG_agg_weighted(t) = Σ(duration_{s,t} × EEG_{s,t}) / Σ(duration_{s,t})
  
  得到 130 个聚合向量 → 130×130 余弦相似度矩阵
  计算 R@1 / R@5 / Mean Rank
```

### 5.3 诊断线 B 噪声注入策略：三层统一架构（v3 核心设计）

**核心原则**：所有模型在**逻辑层**使用相同的噪声/shuffle 定义，但在**实现层**允许分层适配。

#### 三层架构

| 层级 | 职责 | 设计要点 |
|------|------|---------|
| **逻辑层** | 定义"什么"噪声/shuffle | 所有模型完全一致（Gaussian/Shuffle/Zero 的语义定义） |
| **协调层** | 生成全局权威配置 | 在 `UnifiedDataset` 层生成 permutation 和种子序列 |
| **实现层** | 决定"如何"应用 | CET-MAE/EEG-To-Text/GLIM 在数据加载时应用；EEG2Text 在编码阶段应用 |

#### 5.3.1 Shuffle 统一方案

**协调层设计（权威机制）**：
- `UnifiedDataset` 生成全局 derangement（无不动点的 permutation）
- 该 permutation 是**黄金标准**，所有模型必须遵循
- 通过 `shuffle_seed=42` 保证可复现

**实现层适配**：
- **CET-MAE/EEG-To-Text/GLIM**：在 `UnifiedDataset.__init__` 中直接应用 permutation，替换 EEG 字段
- **EEG2Text**：查询 `UnifiedDataset` 的 permutation，在数据加载后对 `raw_list` 进行相同重排

**关键约束**：
- 文本标签不动，仅 EEG 数据发生位移
- 置换必须是完全去相关的（derangement）
- EEG2Text 的 meta 信息和 ref_texts 需要同步重排，保持对齐

#### 5.3.2 Gaussian/Zero 噪声统一方案

**协调层设计（种子权威机制）**：
- 统一使用递进种子公式：`seed = 42 + sample_idx`
- 相同样本索引产生相同的噪声统计特性

**实现层适配**：
- **CET-MAE/EEG-To-Text/GLIM**：在 `UnifiedDataset` 数据加载时生成噪声张量，替换 EEG 字段
- **EEG2Text**：在编码阶段（`encode_eegs`）使用相同种子公式生成噪声，替代 rawData

**一致性保证**：
- 噪声分布一致：均为 N(0,1) 或全零
- 种子逻辑一致：相同样本索引 → 相同噪声（统计意义上）
- 应用时机不同：不影响最终的一致性验证

#### 5.3.3 接口抽象

**目标**：用户（实验脚本）不需要感知 EEG2Text 的特殊性

**统一参数接口**：
- 所有检索脚本使用相同的 `--noise-type {real,gaussian,shuffle,zero}`
- 脚本内部根据模型类型自动选择实现方式
- 输出目录自动添加后缀（`_gaussian` / `_shuffle` / `_zero`）

### 5.4 各模型适配说明（v3 更新）

| 层级 | CET-MAE | EEG-To-Text | EEG2Text | GLIM |
|------|---------|-------------|----------|------|
| **逻辑层** | Gaussian/Shuffle/Zero 定义 | 同上 | 同上 | 同上 |
| **协调层** | 查询 `UnifiedDataset` 配置 | 同上 | **查询** `UnifiedDataset.shuffle_perm` | 同上 |
| **实现层** | 数据加载时应用 | 同上 | **编码阶段**应用 | 同上 |
| **干预点** | `UnifiedDataset.__getitem__` | 同上 | `encode_eegs()` 函数 | 同上 |

**v3 关键改进**：
- 明确三层架构，分离逻辑与实现
- EEG2Text 在逻辑层服从统一设计，在实现层有独立适配
- 通过协调层的权威配置保证跨模型一致性

### 5.5 评估指标

与现有检索评估完全一致：

- **R@K**（K=1, 5, 10）：前 K 命中率
- **MRR**：平均倒数排名
- **Mean Rank / Median Rank**：平均排名 / 中位排名
- **分组维度**：by_task, by_subject, by_dataset

**v2 新增统计指标**：
- 置换检验 p-value
- Bootstrap 95% CI
- Cohen's d 效应量

### 5.6 输出目录结构

```
benchmark_eval/test_outputs/
│
│  # 诊断线 A 输出
├── dataset_validity/
│   ├── linear_probe_results.json           # A1a/b/c 结果
│   ├── linear_probe_comparison.png         # 三版本对比图
│   ├── subject_effect_analysis.json        # A2 结果
│   ├── band_level_eta_squared.json         # A2-band 结果（v2 新增）
│   ├── band_level_eta_squared.png          # 频带效应对比图
│   ├── tsne_by_subject_p5.png              # t-SNE perplexity=5
│   ├── tsne_by_subject_p30.png             # t-SNE perplexity=30
│   ├── tsne_by_subject_p50.png             # t-SNE perplexity=50
│   ├── tsne_by_sentence_p30.png
│   ├── tsne_by_task_p30.png
│   └── desubject_results.json              # A3（条件执行）
│
│  # 诊断线 B 输出（real 已有）
├── eval_cet_mae_retrieval/                 # real
├── eval_cet_mae_retrieval_gaussian/
├── eval_cet_mae_retrieval_shuffle/
├── eval_cet_mae_retrieval_zero/
├── eval_eeg_to_text_retrieval/             # real
├── eval_eeg_to_text_retrieval_gaussian/
├── eval_eeg_to_text_retrieval_shuffle/
├── eval_eeg_to_text_retrieval_zero/
├── eval_eeg2text_retrieval/                # real
├── eval_eeg2text_retrieval_gaussian/
├── eval_eeg2text_retrieval_shuffle/
├── eval_eeg2text_retrieval_zero/
├── eval_glim_retrieval/                    # real
├── eval_glim_retrieval_gaussian/
├── eval_glim_retrieval_shuffle/
├── eval_glim_retrieval_zero/
│
│  # 综合分析报告
└── contrast_summary.json                   # 综合归因结论
```

---

## 六、实施计划

### Task 1：数据集有效性验证脚本（诊断线 A）

创建 `benchmark_eval/scripts/validate_eeg_signal.py`：
- 实现 A1a/b/c（Linear Probe 多版本）
- 实现 A2（被试效应分析，修正分组）
- 实现 A2-band（频带级 η² 分析，v2 新增）
- 实现 A3（去被试化，修正数据泄漏）
- 纯 CPU，依赖 sklearn / scipy / matplotlib
- 输出到 `benchmark_eval/test_outputs/dataset_validity/`

**涉及文件**：新建 `benchmark_eval/scripts/validate_eeg_signal.py`

### Task 2：扩展 `dataset.py` 噪声类型和 Shuffle 支持

- 在 `_generate_noise_eeg` 中新增 `noise_type="zero"` 分支
- **v2 新增**：添加 `shuffle_mode` 支持，实现全局 derangement
- Zero：所有 EEG 字段返回 `np.zeros(shape)`
- Shuffle：全局 permutation，保证跨模型一致

**涉及文件**：`benchmark_eval/data_processing/dataset.py`

### Task 3：为检索脚本添加噪声支持（诊断线 B）

为每个现有检索脚本添加 `--noise-type {real,gaussian,shuffle,zero}` 参数：
- `real`：默认行为，不修改
- `gaussian`：使用 `noise_mode=True` 或脚本内生成噪声
- `shuffle`：使用 `shuffle_mode=True`（v2 统一在 dataset 层）
- `zero`：使用 `noise_type="zero"` 或脚本内全零替代
- 自动设置输出目录后缀：`_gaussian` / `_shuffle` / `_zero`

**v2 更新**：EEG2Text 使用全局 permutation 索引，确保与其他模型一致。

**涉及文件**：
- `benchmark_eval/scripts/run_cet_mae_retrieval.py`
- `benchmark_eval/scripts/run_eeg_to_text_retrieval.py`
- `benchmark_eval/scripts/run_eeg2text_retrieval.py`
- `benchmark_eval/scripts/run_glim_retrieval.py`

### Task 4：运行全部实验

```bash
# 诊断线 A（CPU，~10 分钟，含多版本和频带分析）
python benchmark_eval/scripts/validate_eeg_signal.py \
  --data-path benchmark_eval/data/unified_zuco.pkl

# 诊断线 B（GPU，~30 分钟）
for model in cet_mae eeg_to_text eeg2text glim; do
  for noise in gaussian shuffle zero; do
    python benchmark_eval/scripts/run_${model}_retrieval.py \
      --noise-type $noise ...
  done
done
```

### Task 5：生成综合对比分析报告

创建 `benchmark_eval/scripts/compare_contrast_results.py`：
- 读取诊断线 A 的所有结果（含多版本和频带分析）
- 读取诊断线 B 的所有 16 个 `retrieval_metrics.json`
- 计算统计检验（置换检验、Bootstrap CI、Cohen's d）
- 综合两条线的结论，输出最终归因判定
- 输出到 `benchmark_eval/test_outputs/contrast_summary.json`

### Task 6：更新 `todo.md`

- 更新 NC-1 ~ NC-5 对应状态
- 新增 DV-1a/b/c（数据有效性验证任务，含多版本）
- 新增 DV-2-band（频带级分析）

---

## 七、预期结果与后续行动

### 7.1 综合归因矩阵（v2 修正表述）

| 诊断线 A 结论 | 诊断线 B 结论 | 综合归因 | 后续行动 |
|--------------|--------------|----------|----------|
| A1a/b/c 均 ≈ random，被试效应主导 | real ≈ gaussian | **在当前表示下，被试噪声淹没语义信号** | 探索被试归一化策略 / 多被试聚合 |
| A1a/b/c 均 ≈ random，去被试化后仍 ≈ random | real ≈ gaussian | **在当前表示下，未检测到可检测的句子级语义信息** | 尝试时序模型（LSTM/Transformer）或更换特征表示 |
| A1a ≈ random，但 A1b/c >> random | — | **mean-pooling 是瓶颈** | 采用更精细的特征提取（duration-weighted 或时序模型）|
| A1a/b/c 均 >> random | real ≈ gaussian | **模型编码器完全无效** | 检查 checkpoint、训练收敛性、权重冻结 |
| A1a/b/c 均 >> random | real > gaussian, real ≈ shuffle | **模型仅学到统计偏置** | 考虑 CLIP-style 对比学习 fine-tuning |
| A1a/b/c 均 >> random | real > gaussian, real > shuffle | **模型有效但天花板低** | 属于领域固有挑战，可在论文中正面讨论 |

**v2 关键修正**：避免绝对化的"数据集根本局限"结论，改为条件性的"在当前表示下"。

### 7.2 补充实验（可选）

如果综合结论指向信号质量问题，可追加：

- **渐进噪声实验**：在真实 EEG 上叠加不同强度的高斯噪声（σ = 0.1, 0.5, 1.0, 2.0, 5.0），观察性能衰减曲线
- **频段掩码实验**：将特定频段（如 α、β、γ）置零，定位语义信息所在频段
- **时序模型基线**：使用 1D-CNN 或 LSTM 处理词序列，对比 mean-pooling

---

## 八、关键约束与注意事项

### 8.1 三层架构的执行原则

1. **逻辑层一致性**：所有模型在概念上使用相同的噪声/shuffle 定义（Gaussian/Shuffle/Zero 的语义）
2. **协调层权威性**：`UnifiedDataset` 生成的 permutation 和种子序列是黄金标准，所有模型必须查询并遵循
3. **实现层灵活性**：CET-MAE/EEG-To-Text/GLIM 在数据加载时应用；EEG2Text 在编码阶段应用，但逻辑等价

### 8.2 数据与实验约束

4. **数据泄漏防护**：A3 去被试化必须在 train 集上计算 μ_s 和 σ_s，严禁使用 test 集
5. **文本不变性**：所有噪声实验中，文本编码保持不变（仅 EEG 端变化），文本向量可复用
6. **离线环境**：所有脚本运行时需设置 `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`
7. **诊断线 A 优先**：建议先跑诊断线 A（CPU，10分钟），根据结果决定诊断线 B 的优先级
8. **谨慎结论表述**：避免绝对化的否定性结论，使用"在当前表示下"等条件性表述

### 8.3 EEG2Text 特殊性管理

9. **数据对齐验证**：定期检查 `miss_count`（缺失 rawData 的样本数），异常时告警
10. **一致性交叉验证**：运行 Shuffle 实验时，验证 EEG2Text 与其他模型的样本对应关系一致
11. **异构输入标注**：在对比分析报告中，明确标注 EEG2Text 的输入异构性，谨慎解读跨模型比较

---

## 九、建议执行顺序

| 步骤 | 内容 | 耗时 | 依赖 |
|------|------|------|------|
| 1 | Task 1: 创建 `validate_eeg_signal.py` 并运行 A1a/b/c + A2 + A2-band | ~10 min (CPU) | 无 |
| 2 | 根据 A1/A2 结果决定是否执行 A3 | 判断 | Step 1 |
| 3 | Task 2: 扩展噪声类型 + Shuffle 统一实现 | 开发 | 无 |
| 4 | Task 3: 修改检索脚本 | 开发 | Step 3 |
| 5 | Task 4: 运行 12 组噪声检索实验 | ~30 min (GPU) | Step 4 |
| 6 | Task 5: 综合分析报告（含统计检验） | ~10 min | Step 1 + Step 5 |

> **关键路径**：Step 1（数据有效性）和 Step 3/4（噪声脚本开发）可并行。
> Step 1 的结果可能直接回答核心问题，使 Step 3~5 变为验证性而非探索性。

---

## 十、版本变更历史

### v1 → v2 主要变更

| 方面 | v1 | v2 |
|------|-----|-----|
| A1 Linear Probe | 单一 mean-pool 版本 | 三版本对比（mean / weighted / band-separated） |
| A2 分组 | 含不可行的"同句同被试" | 修正为"同句异被试 / 同被试异句 / 异句异被试" |
| A2 频带分析 | 无 | 新增频带级 η² 分解 |
| A3 数据泄漏 | 未明确防护 | 明确在 train 集计算统计量 |
| Shuffle 实现 | 各脚本独立实现 | 统一在 dataset.py 层，保证跨模型一致 |
| 统计检验 | 经验阈值 | 新增置换检验、Bootstrap CI、Cohen's d |
| 结论表述 | 绝对化（如"数据集根本局限"） | 条件化（"在当前表示下"） |
| t-SNE | 单一 perplexity | 多 perplexity 对比 |

### v2 → v3 主要变更

| 方面 | v2 | v3 |
|------|-----|-----|
| EEG2Text 处理 | 标注为"例外"，特殊处理 | 明确**三层架构**（逻辑/协调/实现），分层统一 |
| 噪声注入 | 描述具体实现 | 抽象为**权威机制**（permutation 权威、种子权威） |
| 架构设计 | 隐含在描述中 | 显式定义三层，分离逻辑与实现 |
| 验证策略 | 较少提及 | 新增**一致性验证**和**兜底策略** |

**v3 核心设计哲学**：
- EEG2Text 的特殊性不是"例外"，而是**实现层的差异**
- 通过协调层的权威配置，保证逻辑层的一致性
- 所有模型服从统一的实验设计，但允许实现路径不同

---

*预计总耗时：开发 ~2h + 运行 ~40min。诊断线 A 的 Linear Probe（10分钟）即可给出第一个关键信号。*
