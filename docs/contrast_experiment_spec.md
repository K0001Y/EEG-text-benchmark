# EEG-to-Text 模型性能归因：对比实验规格书

> 目标：通过两条独立的诊断线——**噪声对照实验**和**原始数据集有效性验证**——
> 系统性定位四个 EEG-to-Text 模型检索性能接近随机的根本原因。
>
> **创建时间：2026-04**

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
Q1: 原始 ZuCo EEG 数据是否包含足够的句子级语义信息？（数据有效性）
Q2: 模型编码器是否从 EEG 信号中学到了任何有意义的表征？（模型能力）
Q3: 如果学到了，是跨模态对应关系，还是仅 EEG 的统计偏置？（学习深度）
```

### 1.3 两条诊断线的关系

```
诊断线 A：原始数据集有效性验证          诊断线 B：噪声对照实验
（与模型无关，验证信号本身）           （与数据无关，验证模型能力）
────────────────────────              ──────────────────────
"EEG 数据里有没有语义信息？"           "模型有没有学到 EEG 信息？"

  Linear Probe / 被试效应分析            Gaussian / Shuffle / Zero 对照
           │                                       │
           ▼                                       ▼
    信号信息量上界                            模型利用率下界
```

**两条线互补**：
- 如果诊断线 A 发现原始信号就不可区分 → 即使模型完美也无法检索，问题在数据
- 如果诊断线 A 发现信号有区分性，但诊断线 B 显示 real ≈ gaussian → 问题在模型
- 两条线结合才能完整归因

---

## 二、诊断线 A：原始数据集有效性验证

> 核心目标：不依赖任何深度模型，直接从数据层面验证 ZuCo EEG 数据是否包含句子级语义信息。

### 2.1 实验 A1：线性探针上界（Linear Probe）

**原理**：如果原始 EEG 特征空间中句子不可分，任何编码器都不可能做好检索。

**方法**：

- 输入：每个样本的词级 EEG 均值池化向量 `(840,)`（`eeg_word_norm1d` 的 `mean(axis=0)`）
- 任务：130 类分类（130 个 unique 句子）
- 模型：逻辑回归（`sklearn.linear_model.LogisticRegression`），无非线性
- 数据划分：使用已有 train/test split（train 用于拟合，test 用于评估）
- 评估指标：
  - Top-1 accuracy（对比随机基线 0.77%）
  - Top-5 / Top-10 accuracy
  - 混淆矩阵的条件数（衡量类间重叠程度）

**结果解读**：

| Linear Probe 结果 | 含义 |
|-------------------|------|
| accuracy ≈ 0.77%（随机） | EEG 特征空间中句子不可分，原始信号缺乏句子级区分信息 |
| accuracy 显著 > random（如 > 5%） | 信号包含信息，深度模型未有效利用 |
| accuracy 远超 random（如 > 20%） | 信号信息量充足，问题完全在模型端 |

### 2.2 实验 A2：被试效应 vs 句子效应分析

**原理**：ZuCo 数据的每个样本同时受**句子语义**和**被试个体差异**两个因素影响。如果被试效应远大于句子效应，个体差异会淹没语义信号。

**数据结构**：

```
130 个句子 × ~14 个被试 ≈ 1858 个样本

每个 EEG 样本 = f(句子语义, 被试个体差异, 随机噪声)
                     ↑ 我们关心的       ↑ 可能淹没信号的
```

**方法 A：余弦相似度分组对比**

```
对 test 集计算所有样本对的余弦相似度，按关系分三组：

  同句同被试对：cosine(EEG_i, EEG_j) where sentence_i = sentence_j AND subject_i = subject_j
  同句异被试对：cosine(EEG_i, EEG_j) where sentence_i = sentence_j AND subject_i ≠ subject_j
  异句异被试对：cosine(EEG_i, EEG_j) where sentence_i ≠ sentence_j

期望排序（如果语义信息存在）：同句同被试 > 同句异被试 > 异句异被试
实际排序（如果被试效应主导）：同句同被试 >> 同句异被试 ≈ 异句异被试
```

**方法 B：方差分解（η² 分析）**

```
对 EEG 特征的每个维度（840维）做 two-way ANOVA：
  Factor A = 句子 ID（130 levels）
  Factor B = 被试 ID（~14 levels）

比较：η²(句子) vs η²(被试)

η²(被试) >> η²(句子) → 个体差异主导，语义信号被淹没
η²(句子) ≈ η²(被试) → 两者效应相当
η²(句子) >> η²(被试) → 语义信号强（不太可能，但理想情况）
```

**方法 C：降维可视化**

```
对全体 test 样本的 EEG (mean-pooled, 840-dim) 做 PCA → t-SNE：
  图 A：按被试 ID 着色 → 观察是否呈现被试聚类
  图 B：按句子 ID 着色 → 观察是否呈现句子聚类
  图 C：按 task 着色   → 观察任务效应

输出：三张可视化图，直观判断聚类主导因素
```

### 2.3 实验 A3：去被试化后的信号恢复

> 仅在 A2 确认被试效应显著时执行。

**方法 A：被试内 z-score 归一化**

```
对每个被试 s：
  μ_s = mean(所有属于被试 s 的 EEG 样本)
  σ_s = std(所有属于被试 s 的 EEG 样本)
  EEG_normalized = (EEG - μ_s) / σ_s

归一化后重新执行 A1（Linear Probe）和检索评估
```

**方法 B：被试聚合（同句多被试平均）**

```
对每个句子 t：
  EEG_agg(t) = mean(所有被试看句子 t 的 EEG)

→ 被试级噪声被平均掉，信噪比提升 ~√N（N=被试数）
→ 变成 130 query × 130 candidate 的方阵检索
→ 如果此时性能显著提升 → 信号存在但单样本 SNR 太低
```

### 2.4 实验 A 的诊断决策树

```
A1: Linear Probe
    │
    ├── accuracy ≈ random ──→ A2: 被试效应分析
    │                              │
    │                              ├── η²(被试) >> η²(句子)
    │                              │       → A3: 去被试化
    │                              │              │
    │                              │              ├── 去被试化后 Linear Probe 提升
    │                              │              │   → 结论：信号存在但被个体差异淹没
    │                              │              │           需要被试归一化策略
    │                              │              │
    │                              │              └── 去被试化后仍 ≈ random
    │                              │                  → 结论：ZuCo 词级频域特征不含
    │                              │                          句子级语义（数据集根本局限）
    │                              │
    │                              └── η²(被试) ≈ η²(句子)（都弱）
    │                                     → 结论：EEG 特征维度本身信息稀疏
    │
    └── accuracy >> random ──→ 结论：信号有区分性
                                     模型编码器的问题（转诊断线 B）
```

---

## 三、诊断线 B：噪声对照实验

> 核心目标：通过替换/破坏 EEG 输入，验证模型编码器是否从 EEG 中提取了有意义的信息。

### 3.1 三层归因框架

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

### 3.2 噪声条件设计

| 条件 | 缩写 | 定义 | 诊断目标 |
|------|------|------|----------|
| **真实 EEG** | `real` | 原始数据，不做任何修改 | 基线参照 |
| **高斯噪声** | `gaussian` | 用 N(0,1) 随机信号完全替代 EEG，shape 保持一致 | 编码器是否学到了任何 EEG 特征（vs 纯随机） |
| **配对打乱** | `shuffle` | 使用真实 EEG 数据，但随机打乱 EEG-文本配对关系 | 编码器是否学到了跨模态对应关系（vs 统计偏置） |
| **全零输入** | `zero` | 用全零张量替代 EEG | 排除模型 bias/shortcut 依赖 |

### 3.3 结果解读决策树

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

**六种典型诊断结论：**

| 情况 | real vs gaussian | real vs shuffle | zero 行为 | 归因 |
|------|------------------|-----------------|-----------|------|
| A | real ≈ gaussian ≈ shuffle ≈ zero | — | ≈ 随机 | **编码器完全无效**，未从 EEG 学到任何信息 |
| B | real > gaussian | real ≈ shuffle | — | 学到 EEG **统计特性**（如功率谱分布），但未学到跨模态对应 |
| C | real > gaussian | real > shuffle | — | **学到了跨模态对应**，但检索天花板受限于信号 SNR |
| D | real ≈ gaussian | — | zero 显著偏离 | **数据管道 bug**：噪声与真实信号经处理后特征坍缩 |
| E | real ≈ gaussian ≈ zero | — | — | 模型存在 **shortcut/bias**，不依赖输入内容 |
| F | real > gaussian, real > shuffle, 但绝对值仍低 | — | — | 模型有效但 **EEG SNR 太低**，语义信息本身稀疏 |

### 3.4 显著性判定标准

"real > gaussian" 的判定标准（需同时满足）：

- Mean Rank 差值 > 3.0（约 2.3% 相对偏移）
- R@10 差值 > 2 个百分点
- MRR 差值 > 0.005

如果差异低于上述阈值，视为 ≈（无显著差异）。

---

## 四、实验设计

### 4.1 实验矩阵总览

**诊断线 A（数据有效性）**：3 组实验，无需 GPU

| 编号 | 实验 | 输入 | 工具 |
|------|------|------|------|
| A1 | Linear Probe | test 集 EEG mean-pool (840,) | sklearn LogisticRegression |
| A2 | 被试效应分析 | test 集 EEG mean-pool (840,) | scipy ANOVA / matplotlib |
| A3 | 去被试化验证 | per-subject z-score 后的 EEG | sklearn + 检索脚本 |

**诊断线 B（噪声对照）**：4 模型 × 4 条件 = 16 组，需 GPU

| | real | gaussian | shuffle | zero |
|---|---|---|---|---|
| CET-MAE | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| EEG-To-Text | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| EEG2Text | ✅ 已有 | 待跑 | 待跑 | 待跑 |
| GLIM | ✅ 已有 | 待跑 | 待跑 | 待跑 |

### 4.2 诊断线 A 实施细节

#### A1: Linear Probe

```
数据准备：
  从 unified_zuco.pkl 加载 train + test 样本
  对每个样本：eeg_feat = eeg_word_norm1d.mean(axis=0)  → (840,)
  标签 = 句子文本的唯一 ID（0~129）

模型训练：
  LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
  在 train 集上 fit，在 test 集上 predict

输出指标：
  Top-1 / Top-5 / Top-10 accuracy
  对比随机基线 1/130 = 0.77%
```

#### A2: 被试效应分析

```
余弦相似度分组：
  计算 test 集所有样本对 (N*(N-1)/2) 的 cosine similarity
  分组统计：同句同被试 / 同句异被试 / 异句 的均值和分布

方差分解：
  对 840 维特征逐维做 two-way ANOVA
  汇总：median η²(句子) vs median η²(被试)

可视化：
  PCA(n_components=50) → t-SNE(n_components=2)
  分别按被试/句子/task 着色
```

#### A3: 去被试化（条件执行）

```
被试内归一化：
  对每个被试计算 μ_s, σ_s（在 train 集上）
  对 train + test 集应用 (x - μ_s) / σ_s
  重新执行 A1

被试聚合检索：
  对每个句子，将所有被试的 EEG 取平均 → 130 个聚合向量
  直接计算 130×130 余弦相似度矩阵
  计算 R@1 / R@5 / Mean Rank
```

### 4.3 诊断线 B 噪声注入策略

#### 4.3.1 Gaussian 噪声

- 分布：`N(μ=0, σ=1)`
- 替换范围：**所有 EEG 字段**（eeg_word_raw, eeg_word_norm1d, eeg_word_norm2d, eeg_spectro, sent_eeg_raw）
- Mask：保持全 1（所有时间步均"有效"）
- 随机种子：`seed = 42 + sample_idx`，保证跨模型一致

> 已有实现：`UnifiedDataset(noise_mode=True, noise_type="gaussian")`
> 覆盖 CET-MAE、EEG-To-Text、GLIM（均从 UnifiedDataset 读取）。
>
> **例外：EEG2Text** 检索脚本直接从 spectro pickle 读 rawData (105, time) → (24000, 105)，
> 需在脚本内部单独生成高斯噪声张量替代 rawData。

#### 4.3.2 Shuffle（配对打乱）

- 方式：对 test 集内所有样本的 EEG 进行**固定随机置换**
- 具体实现：生成一个 `permutation = rng.permutation(N)`，样本 i 使用样本 `permutation[i]` 的 EEG
- 关键约束：
  - 文本标签不动，仅 EEG 数据发生位移
  - 置换必须是**完全去相关**的（permutation[i] ≠ i，即 derangement）
  - 随机种子固定 `seed=42`，保证跨模型一致
- 实现层级：在检索脚本中、编码 EEG 之前执行 shuffle，不修改 `dataset.py`

> **注意**：Shuffle 保留了 EEG 数据的真实统计分布（均值、方差、频谱特性均不变），
> 仅破坏了 EEG-文本的一一对应关系。

#### 4.3.3 Zero（全零输入）

- 所有 EEG 字段填充 `0.0`
- Mask 保持全 1
- 无随机性，结果完全确定

> 新增实现：在 `_generate_noise_eeg` 中添加 `noise_type="zero"` 分支。
> EEG2Text 脚本中同样用 `torch.zeros(24000, 105)` 替代。

### 4.4 各模型适配说明

| 模型 | EEG 数据来源 | Gaussian/Zero 方式 | Shuffle 方式 |
|------|-------------|-------------------|-------------|
| **CET-MAE** | `UnifiedDataset` → `eeg_word_norm2d` + `sent_eeg_raw` | `noise_mode=True` | 检索脚本内 permute `eeg_bufs` |
| **EEG-To-Text** | `UnifiedDataset` → `eeg_word_norm1d` | `noise_mode=True` | 检索脚本内 permute `eeg_bufs` |
| **EEG2Text** | spectro pickle → `rawData` (105, T) | 脚本内生成 `N(0,1)` 或 `zeros` 的 `(24000,105)` 张量 | 脚本内 permute `raw_list` |
| **GLIM** | `UnifiedDataset` → `eeg_word_raw` → `_convert_to_glim_format` | `noise_mode=True` | 检索脚本内 permute `eeg_bufs` |

### 4.5 评估指标

与现有检索评估完全一致：

- **R@K**（K=1, 5, 10）：前 K 命中率
- **MRR**：平均倒数排名
- **Mean Rank / Median Rank**：平均排名 / 中位排名
- **分组维度**：by_task, by_subject, by_dataset

### 4.6 输出目录结构

```
benchmark_eval/test_outputs/
│
│  # 诊断线 A 输出
├── dataset_validity/
│   ├── linear_probe_results.json
│   ├── subject_effect_analysis.json
│   ├── tsne_by_subject.png
│   ├── tsne_by_sentence.png
│   ├── tsne_by_task.png
│   └── desubject_results.json          # A3（条件执行）
│
│  # 诊断线 B 输出（real 已有）
├── eval_cet_mae_retrieval/              # real
├── eval_cet_mae_retrieval_gaussian/
├── eval_cet_mae_retrieval_shuffle/
├── eval_cet_mae_retrieval_zero/
├── eval_eeg_to_text_retrieval/          # real
├── eval_eeg_to_text_retrieval_gaussian/
├── eval_eeg_to_text_retrieval_shuffle/
├── eval_eeg_to_text_retrieval_zero/
├── eval_eeg2text_retrieval/             # real
├── eval_eeg2text_retrieval_gaussian/
├── eval_eeg2text_retrieval_shuffle/
├── eval_eeg2text_retrieval_zero/
├── eval_glim_retrieval/                 # real
├── eval_glim_retrieval_gaussian/
├── eval_glim_retrieval_shuffle/
├── eval_glim_retrieval_zero/
│
│  # 综合分析报告
└── contrast_summary.json
```

---

## 五、实施计划

### Task 1：数据集有效性验证脚本（诊断线 A）

创建 `benchmark_eval/scripts/validate_eeg_signal.py`：
- 实现 A1（Linear Probe）、A2（被试效应分析 + 可视化）、A3（去被试化）
- 纯 CPU，依赖 sklearn / scipy / matplotlib
- 输出到 `benchmark_eval/test_outputs/dataset_validity/`

**涉及文件**：新建 `benchmark_eval/scripts/validate_eeg_signal.py`

### Task 2：扩展 `dataset.py` 噪声类型

- 在 `_generate_noise_eeg` 中新增 `noise_type="zero"` 分支
- Zero：所有 EEG 字段返回 `np.zeros(shape)`
- Shuffle 不在 dataset 层实现（需要全局 permutation），留给检索脚本处理

**涉及文件**：`benchmark_eval/data_processing/dataset.py`

### Task 3：为检索脚本添加噪声支持（诊断线 B）

为每个现有检索脚本添加 `--noise-type {real,gaussian,shuffle,zero}` 参数：
- `real`：默认行为，不修改
- `gaussian`：使用 `noise_mode=True` 或脚本内生成噪声
- `shuffle`：在编码前对 EEG 数据执行固定 permutation
- `zero`：使用 `noise_type="zero"` 或脚本内全零替代
- 自动设置输出目录后缀：`_gaussian` / `_shuffle` / `_zero`

**涉及文件**：
- `benchmark_eval/scripts/run_cet_mae_retrieval.py`
- `benchmark_eval/scripts/run_eeg_to_text_retrieval.py`
- `benchmark_eval/scripts/run_eeg2text_retrieval.py`
- `benchmark_eval/scripts/run_glim_retrieval.py`

### Task 4：运行全部实验

```bash
# 诊断线 A（CPU，~5 分钟）
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
- 读取诊断线 A 的 `linear_probe_results.json` + `subject_effect_analysis.json`
- 读取诊断线 B 的所有 16 个 `retrieval_metrics.json`
- 综合两条线的结论，输出最终归因判定
- 输出到 `benchmark_eval/test_outputs/contrast_summary.json`

### Task 6：更新 `todo.md`

- 更新 NC-1 ~ NC-5 对应状态
- 新增 DV-1 ~ DV-3（数据有效性验证任务）

---

## 六、预期结果与后续行动

### 6.1 综合归因矩阵

| 诊断线 A 结论 | 诊断线 B 结论 | 综合归因 | 后续行动 |
|--------------|--------------|----------|----------|
| Linear Probe ≈ random，被试效应主导 | real ≈ gaussian | **被试噪声淹没语义信号** | 探索被试归一化策略 / 多被试聚合 |
| Linear Probe ≈ random，去被试化后仍 ≈ random | real ≈ gaussian | **ZuCo 词级频域特征不含句子级语义**（数据集根本局限） | 考虑更换特征表示（如时频域）或数据集 |
| Linear Probe >> random | real ≈ gaussian | **模型编码器完全无效** | 检查 checkpoint、训练收敛性、权重冻结 |
| Linear Probe >> random | real > gaussian, real ≈ shuffle | **模型仅学到统计偏置** | 考虑 CLIP-style 对比学习 fine-tuning |
| Linear Probe >> random | real > gaussian, real > shuffle | **模型有效但天花板低** | 属于领域固有挑战，可在论文中正面讨论 |

### 6.2 补充实验（可选）

如果综合结论指向信号质量问题，可追加：

- **渐进噪声实验**：在真实 EEG 上叠加不同强度的高斯噪声（σ = 0.1, 0.5, 1.0, 2.0, 5.0），观察性能衰减曲线
- **频段掩码实验**：将特定频段（如 α、β、γ）置零，定位语义信息所在频段
- **被试聚合检索**：同句多被试 EEG 取平均后检索，量化 SNR 提升的收益

---

## 七、关键约束与注意事项

1. **跨模型一致性**：同一噪声条件下，所有模型使用相同的随机种子（42），保证噪声数据完全一致
2. **EEG2Text 特殊处理**：该模型检索脚本绕过 `UnifiedDataset`，直接从 spectro pickle 读 rawData，噪声需在脚本内部注入
3. **文本不变性**：所有噪声实验中，文本编码保持不变（仅 EEG 端变化），文本向量可复用
4. **Shuffle 的去相关性**：permutation 必须保证 `perm[i] ≠ i`（derangement），避免部分样本"意外正确配对"
5. **离线环境**：所有脚本运行时需设置 `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`
6. **诊断线 A 优先**：建议先跑诊断线 A（CPU，5分钟），根据结果决定诊断线 B 的优先级

---

## 八、建议执行顺序

| 步骤 | 内容 | 耗时 | 依赖 |
|------|------|------|------|
| 1 | Task 1: 创建 `validate_eeg_signal.py` 并运行 A1 + A2 | ~5 min (CPU) | 无 |
| 2 | 根据 A1/A2 结果决定是否执行 A3 | 判断 | Step 1 |
| 3 | Task 2 + Task 3: 扩展噪声类型 + 修改检索脚本 | 开发 | 无 |
| 4 | Task 4: 运行 12 组噪声检索实验 | ~30 min (GPU) | Step 3 |
| 5 | Task 5: 综合分析报告 | ~10 min | Step 1 + Step 4 |

> **关键路径**：Step 1（数据有效性）和 Step 3（噪声脚本开发）可并行。
> Step 1 的结果可能直接回答核心问题，使 Step 3~4 变为验证性而非探索性。

---

*预计总耗时：开发 ~1h + 运行 ~35min。诊断线 A 的 Linear Probe（5分钟）即可给出第一个关键信号。*
