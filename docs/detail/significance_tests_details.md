# 显著性检验细节

> **代码文件**：`benchmark_eval/evaluation/significance.py`（统一检验封装）；`benchmark_eval/scripts/analysis/run_significance_tests.py`（B 线检验主脚本）
> **目标**：为 A/B 两线的所有基线对比、噪声对照、跨模型排名提供统一、可复现、带效应量与置信区间的显著性检验。
> **运行环境**：纯 CPU，依赖 `scipy / numpy`，可选 `scikit-posthocs`（缺失时自动回退到简化 Nemenyi）。

---

## 公共约定

### 显著性阈值与默认参数

| 项 | 值 | 说明 |
|----|----|------|
| 原始阈值 $\alpha$ | $0.05$ | 单个检验的接受阈值 |
| 校正后阈值 $\alpha_\text{adj}$ | 依校正方法给出 | 族内多重比较校正后的实际阈值 |
| `DEFAULT_N_PERM` | $1000$ | permutation test 默认置换次数 |
| `DEFAULT_N_BOOT` | $1000$ | bootstrap 默认重采样次数 |
| `DEFAULT_SEED` | $42$ | 统一随机种子，保证结果可复现 |

### 统一返回结构

所有单项检验函数返回字典：

```json
{
  "test":      "方法名",
  "statistic": "检验统计量",
  "p":         "p 值",
  "n":         "样本量",
  "effect":    {"key": "value"},
  "ci95":      [lo, hi],
  "extra":     {"...": "..."}
}
```

所有 p 值 / 效应量 / 置信区间最终汇总至 A 线 `significance_tests.json` 与 B 线 `line_b/{model}/significance_tests.json` 及 `line_b/significance_summary.json`。

---

## 效应量定义

### 配对 Cohen's $d_z$

对配对样本 $a_i, b_i$（$i=1,\ldots,n$），差值 $\delta_i = a_i - b_i$：

$$d_z = \frac{\bar{\delta}}{s_\delta}, \quad \bar{\delta} = \frac{1}{n}\sum_{i=1}^n \delta_i, \quad s_\delta = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (\delta_i - \bar{\delta})^2}$$

若 $s_\delta < 10^{-12}$ 置 $d_z = 0$。

### 独立样本 Cohen's $d$（pooled）

对独立样本 $a \in \mathbb{R}^{n_a}$、$b \in \mathbb{R}^{n_b}$：

$$s_\text{pooled} = \sqrt{\frac{(n_a-1)s_a^2 + (n_b-1)s_b^2}{n_a + n_b - 2}}, \quad d = \frac{\bar{a} - \bar{b}}{s_\text{pooled}}$$

### 秩移动效应量（Mann-Whitney）

由 U 统计量推导：

$$r = 1 - \frac{2U}{n_a \cdot n_b}$$

### Kruskal-Wallis 效应量 $\eta^2_H$

设 $H$ 为 Kruskal-Wallis 检验统计量，$k$ 组、总样本量 $N$：

$$\eta^2_H = \frac{H - k + 1}{N - k}, \quad N > k$$

### 效应量解读标签

#### Cohen's $d / d_z$（`interpret_effect`）

| $|d|$ 区间 | 标签 |
|-----------|------|
| $< 0.2$ | negligible |
| $[0.2, 0.5)$ | small |
| $[0.5, 0.8)$ | medium |
| $\geq 0.8$ | large |

#### $\eta^2 / \eta^2_H$（`interpret_eta2`，Cohen 1988）

| $\eta^2$ 区间 | 标签 |
|-------------|------|
| $< 0.01$ | negligible |
| $[0.01, 0.06)$ | small |
| $[0.06, 0.14)$ | medium |
| $\geq 0.14$ | large |

> 注意：$\eta^2$ 是方差解释比例，与 Cohen's $d$ 不可混用同一阈值。`kruskal_dunn` 的 `effect.label` 现已切换至 `interpret_eta2`（S-5 修复）。

---

## 检验方法一：Wilcoxon 符号秩（小样本配对）

**函数**：`wilcoxon_paired(a, b, alternative, n_boot, seed)`

**使用场景**：5 折 LOSO Top-K 配对、14 被试 accuracy 配对、per-query rank 配对、维度级 $\eta^2$ 配对。

### 步骤 1：符号秩统计量

对差值 $\delta_i = a_i - b_i$，取非零 $\delta_i$ 绝对值秩 $R_i$：

$$W_+ = \sum_{\delta_i > 0} R_i, \quad W_- = \sum_{\delta_i < 0} R_i, \quad W = \min(W_+, W_-)$$

双尾 p 值由 `scipy.stats.wilcoxon(..., zero_method="wilcox")` 给出（零差值样本剔除）。

### 步骤 2：效应量

配对 Cohen's $d_z$（见上）。

### 步骤 3：配对 bootstrap 95% CI

对差值数组 $\boldsymbol{\delta}$ 有放回重采样 $B=1000$ 次：

$$\bar{\delta}^{(b)} = \frac{1}{n}\sum_{i=1}^n \delta_{\pi^{(b)}(i)}, \quad b = 1,\ldots,B$$

$$[\text{CI}_{2.5}, \text{CI}_{97.5}] = \text{percentile}(\{\bar{\delta}^{(b)}\},\ 2.5,\ 97.5)$$

### 步骤 4：异常处理

若所有 $\delta_i = 0$，返回 `p=1.0, d_z=0.0, ci=[0,0]`；若样本量不足（$n<2$）返回 `error="invalid_input"`。

---

## 检验方法二：Mann-Whitney U（大样本独立）

**函数**：`mannwhitney_u(a, b, alternative, n_boot, seed)`

**使用场景**：A2-Cosine 五组相似度对比（百万量级样本对），EEG 各组 vs 噪声组的独立对比。

### 步骤 1：U 统计量

$$U_a = \sum_{i \in a}\sum_{j \in b} \mathbf{1}[a_i > b_j] + \frac{1}{2}\mathbf{1}[a_i = b_j]$$

代码采用 `scipy.stats.mannwhitneyu` 返回的 $U_a$ 直接进入秩移动效应量计算（不再取 $\min(U_a, n_a n_b - U_a)$），保留了 $r$ 的方向性。

p 值由 `scipy.stats.mannwhitneyu(..., alternative=...)` 给出。

### 步骤 2：双效应量

- 秩移动 $r = 1 - 2U/(n_a n_b)$
- 独立 Cohen's $d$（见上）

### 步骤 3：Bootstrap 均值差 CI

两组独立有放回重采样：

$$\Delta^{(b)} = \bar{a}^{(b)} - \bar{b}^{(b)}, \quad b = 1,\ldots,B$$

$$\text{CI}_{95} = \text{percentile}(\{\Delta^{(b)}\},\ 2.5,\ 97.5)$$

### 判定准则

百万级样本量下 p 值极易显著，**必须**同时报告 $|d|$ 与 $|r|$；$|d| > 0.2$ 视为有实际意义的小效应。

---

## 检验方法三：二项检验 vs 随机基线

**函数**：`binomial_vs_baseline(k, n, p0, alternative="greater")`

**使用场景**：Top-K vs $1/130$；R@K vs $K/M$；A1d 单被试 accuracy vs 多数类基线。

### 步骤 1：精确二项 p 值

观测命中 $k$ 次、样本量 $n$，零假设命中率 $p_0$：

$$P(X \geq k \mid n, p_0) = \sum_{j=k}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j}$$

由 `scipy.stats.binomtest(k, n, p0, alternative)` 精确求得。

### 步骤 2：Clopper-Pearson 95% CI

观测比例 $\hat{p} = k/n$ 的精确 CI：

$$\hat{p}_\text{low} = B^{-1}(\alpha/2;\ k,\ n-k+1), \quad \hat{p}_\text{high} = B^{-1}(1-\alpha/2;\ k+1,\ n-k)$$

### 步骤 3：效应量

$$\Delta_\text{vs\_baseline} = \hat{p} - p_0$$

---

## 检验方法四：Bootstrap 均值差（配对）

**函数**：`bootstrap_mean_diff(a, b, n_boot, seed)`

**使用场景**：B 线 per-query R@K / MRR / mean_rank 差值的 CI；Cohen's $d_z$ 之外需要分布派生 p 值的场景。

### 步骤 1：配对差值重采样

对 $\delta_i = a_i - b_i$ 有放回重采样 $B$ 次，每次计算 $\bar{\delta}^{(b)}$。

### 步骤 2：经验 p 值

根据原始均值方向自适应：

$$p_\text{emp} = \max\!\left(2 \cdot (1 - q),\ \frac{1}{B}\right), \quad q = \begin{cases} \#\{\bar{\delta}^{(b)} > 0\}/B & \bar{\delta} \geq 0 \\ \#\{\bar{\delta}^{(b)} < 0\}/B & \bar{\delta} < 0 \end{cases}$$

下限 $1/B$ 防止出现 $p=0$。

### 步骤 3：CI 与效应量

$$\text{CI}_{95} = \text{percentile}(\{\bar{\delta}^{(b)}\},\ 2.5,\ 97.5)$$

同时返回 $d_z$。CI 跨 0 即视为不显著。

---

## 检验方法五：Permutation Retrieval（固定表示打乱真值）

**函数**：`permutation_retrieval(sim, gt_idx, ks, n_perm, seed)`

**使用场景**：B 线每个噪声条件下检索指标的 null 分布；A3-Retrieval / A3-SessionRetrieval 的显著性。

### 步骤 1：观测指标

给定 $(N_q, N_c)$ 余弦相似度矩阵 $\mathbf{S}$ 与真值下标 $\mathbf{g} \in \{0,\ldots,N_c-1\}^{N_q}$，按行降序排序得排名：

$$r_i = \text{rank of } g_i \text{ in } \text{argsort}(-\mathbf{S}_{i,:}) + 1$$

$$\text{R@K}_\text{obs} = \frac{1}{N_q}\sum_{i=1}^{N_q} \mathbf{1}[r_i \leq K], \quad \text{MRR}_\text{obs} = \frac{1}{N_q}\sum_{i=1}^{N_q} \frac{1}{r_i}$$

### 步骤 2：固定 $\mathbf{S}$，置换真值 $P$ 次

$$\mathbf{g}^{(p)} = \mathbf{g}[\pi^{(p)}], \quad \pi^{(p)} \sim \text{Uniform}(\mathcal{S}_{N_q})$$

> 注：$\pi^{(p)}$ 是 $\{1,\ldots,N_q\}$ 上的随机排列（无放回），等价于在已观测的真值集合内置换标签——保留 $\mathbf{g}$ 的频率分布，不在 $[0, N_c)$ 整个候选池上均匀重采样。`evaluation.significance.permutation_retrieval` 的实现 `gt_idx[rng.permutation(N)]` 即此规约。

重算每次 $\text{R@K}^{(p)}$、$\text{MRR}^{(p)}$、$\text{mean\_rank}^{(p)}$，得 null 分布。

### 步骤 3：经验 p 值（方向自适应）

$$p_\text{R@K} = \max\!\left(\frac{1}{P}\sum_{p=1}^P \mathbf{1}[\text{R@K}^{(p)} \geq \text{R@K}_\text{obs}],\ \frac{1}{P}\right)$$

$$p_\text{mean\_rank} = \max\!\left(\frac{1}{P}\sum_{p=1}^P \mathbf{1}[\text{mean\_rank}^{(p)} \leq \text{mean\_rank}_\text{obs}],\ \frac{1}{P}\right)$$

> mean rank 越小越好，故反向比较。

### 步骤 4：z 偏离与 CI

$$z = \frac{\text{obs} - \bar{\text{null}}}{s_\text{null} + 10^{-12}}, \quad \text{CI}_{95} = \text{percentile}(\text{null},\ 2.5,\ 97.5)$$

### 步骤 5：大样本跳过策略

`run_significance_tests.py` 中 $N_q > 5000$ 时跳过置换（耗时 $O(N_q^2 \log N_q \cdot P)$），输出 `{"skipped": true, "reason": "N>5000"}`。

---

## 检验方法六：Kolmogorov-Smirnov vs 均匀

**函数**：`ks_vs_uniform(ranks, M)`

**使用场景**：检验真实排名分布 $\{r_i\}_{i=1}^N$ 是否显著偏离随机均匀 $\text{Uniform}[1, M]$。

### 统计量

$$D = \sup_r \left| F_N(r) - F_{\text{Uniform}[1,M]}(r) \right|$$

$F_N$ 为经验 CDF。由 `scipy.stats.kstest(ranks, "uniform", args=(1, M - 1))` 给出 p 值。

> scipy 的 `uniform(loc, scale)` 对应 $\text{Uniform}[\text{loc},\ \text{loc}+\text{scale}]$，故 $\text{Uniform}[1, M]$ 需 `args=(1, M-1)`（S-2 修复）。

**解读**：$p < 0.05$ 意味着排名分布显著偏离"完全随机"，是模型学到某种可区分结构的必要条件（非充分）。

---

## 检验方法七：Friedman + 事后 Nemenyi

**函数**：`friedman_nemenyi(matrix, group_names)`

**使用场景**：跨 4 模型 per-query rank 对比；A2-band 8 频带 $\eta^2_\text{sent}$ 对比。

### 步骤 1：Friedman $\chi^2$ 统计量

输入矩阵 $\mathbf{X} \in \mathbb{R}^{n \times k}$（$n$ 个 block × $k$ 个条件）。对每行排名得 $\mathbf{R} \in \mathbb{R}^{n \times k}$，列均秩 $\bar{R}_j = \frac{1}{n}\sum_i R_{ij}$：

$$\chi^2_F = \frac{12n}{k(k+1)}\sum_{j=1}^k \left(\bar{R}_j - \frac{k+1}{2}\right)^2$$

p 值 $\sim \chi^2_{k-1}$ 分布。

### 步骤 2：Kendall's W 一致性

$$W = \frac{\chi^2_F}{n(k-1)} \in [0, 1]$$

$W=0$ 完全随机，$W=1$ 完全一致。

### 步骤 3：Nemenyi 事后两两比较

**首选**：`scikit_posthocs.posthoc_nemenyi_friedman(X)` 直接给出 $\binom{k}{2}$ 校正后 p 值矩阵。

**回退**（无 scikit-posthocs）：临界差 CD 法，

$$\text{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6n}}$$

$q_\alpha$ 查 Nemenyi 表（$k \leq 10$），若 $|\bar{R}_i - \bar{R}_j| > \text{CD}$ 则判定显著。

---

## 检验方法八：Kruskal-Wallis + Dunn

**函数**：`kruskal_dunn(groups, dunn_correction)`

**使用场景**：A1d / A3 等按 subject / task / dataset 的异质性检验；B 线分组诊断。

### 步骤 1：Kruskal-Wallis H 统计量

$k$ 组独立样本合并后整体排名 $R_{ij}$，组均秩 $\bar{R}_{j\cdot}$，总样本量 $N$：

$$H = \frac{12}{N(N+1)}\sum_{j=1}^k n_j \left(\bar{R}_{j\cdot} - \frac{N+1}{2}\right)^2$$

p 值 $\sim \chi^2_{k-1}$。

### 步骤 2：效应量

$$\eta^2_H = \frac{H - k + 1}{N - k}$$

### 步骤 3：Dunn 事后

对每对 $(i, j)$：

$$z_{ij} = \frac{\bar{R}_{i\cdot} - \bar{R}_{j\cdot}}{\sqrt{\frac{N(N+1)}{12}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}}$$

p 值由 $z_{ij}$ 转双尾正态，随后按 `dunn_correction ∈ {bh, holm, bonferroni}` 校正。

---

## 检验方法九：Permutation $\eta^2$（方差分解的置换检验）

**函数**：`permutation_eta(feature_dim, labels, n_perm, seed)`

**使用场景**：A2-Eta 维度级 $\eta^2_\text{subj}$、$\eta^2_\text{sent}$、$\eta^2_\text{session}$ 的显著性；A2-band 频带级 $\eta^2$。

### 步骤 1：观测 $\eta^2$

单维度 $\mathbf{x} \in \mathbb{R}^N$ 与标签 $\mathbf{y}$，类别 $\{c_1,\ldots,c_K\}$：

$$\text{SS}_\text{total} = \sum_{i=1}^N (x_i - \bar{x})^2$$

$$\text{SS}_\text{between} = \sum_{k=1}^K n_k (\bar{x}_k - \bar{x})^2, \quad \bar{x}_k = \frac{1}{n_k}\sum_{i:\ y_i=c_k} x_i$$

$$\eta^2_\text{obs} = \frac{\text{SS}_\text{between}}{\text{SS}_\text{total}}$$

若 $\text{SS}_\text{total} < 10^{-12}$ 置 $\eta^2 = 0$。

### 步骤 2：标签置换 null 分布

固定 $\mathbf{x}$，独立打乱 $\mathbf{y}$ 得 $\mathbf{y}^{(p)} = \mathbf{y}[\pi^{(p)}]$，重算 $\eta^{2\,(p)}$，$p = 1,\ldots,P$。

### 步骤 3：p 值与 CI

$$p = \max\!\left(\frac{1}{P}\sum_{p=1}^P \mathbf{1}[\eta^{2\,(p)} \geq \eta^2_\text{obs}],\ \frac{1}{P}\right)$$

$$\text{CI}_\text{null} = \text{percentile}(\{\eta^{2\,(p)}\},\ 2.5,\ 97.5)$$

$$z = \frac{\eta^2_\text{obs} - \bar{\eta}^2_\text{null}}{s_{\eta^2_\text{null}} + 10^{-12}}$$

### 解读

EEG 真实 $\eta^2_\text{obs}$ 需**显著高于 null 95% 上分位**，方为"方差确实由该因子解释"而非数值噪声。

---

## 多重比较校正

### Holm-Bonferroni（强控 FWER）

**函数**：`holm_bonferroni(pvals, alpha)`

将 $m$ 个 p 值升序排列 $p_{(1)} \leq \ldots \leq p_{(m)}$。校正后 p：

$$\tilde{p}_{(i)} = \max_{j \leq i}\,\min(1,\ (m - j + 1) \cdot p_{(j)})$$

（单调非递减投影）；首个 p 值对应阈值 $\alpha/m$；所有 $\tilde{p}_{(i)} < \alpha$ 为显著。

**适用**：A1 三组信号 × Top-K（9 组）；单模型 36 组条件对（6 pair × 6 p-values）；跨被试两两 $p$ 值。

### Benjamini-Hochberg FDR（控假发现率）

**函数**：`bh_fdr(pvals, alpha)`

升序排列后：

$$\tilde{p}_{(i)} = \min_{j \geq i}\,\min\!\left(1,\ \frac{m}{j} p_{(j)}\right)$$

**适用**：A2-Eta 840 维度级 $\eta^2$；B 线跨 4 模型 120 组全局校正（4 模型 × 6 pair × 5 指标）；A2-band 跨频带。

### 校正策略映射

| 范围 | 校正方法 | 依据 |
|------|---------|------|
| 单族少量对比（$m \leq 10$）| Holm-Bonferroni | 强控，结论保守但可靠 |
| 大规模维度 / 频带（$m \geq 20$）| BH-FDR | 保留检出力 |
| Friedman 事后（$k$ 条件两两）| Nemenyi（已隐含校正）| 标准做法 |
| Kruskal 事后（多组两两）| Dunn + BH-FDR | `scikit-posthocs.posthoc_dunn` 自动化 |

---

## 按实验分层的检验策略

### A 线（`scripts/validate_eeg_signal.py` 相关）

| 子实验 | 比较对 | 方法 | 校正 |
|--------|--------|------|------|
| A1a/A1b/A1c | 词级 EEG Top-K vs 噪声 Top-K（5 折配对） | `wilcoxon_paired` | Holm-Bonferroni（9 组） |
| A1a/A1b/A1c | Top-K vs 随机基线 $1/130$ | `binomial_vs_baseline` | — |
| A1a/A1b/A1c | 词级 vs 句级 EEG（5 折配对） | `wilcoxon_paired` | Holm-Bonferroni |
| A1d | 单被试 accuracy vs 多数类基线 | `binomial_vs_baseline` | Holm-Bonferroni（按被试数） |
| A1d | 全局 vs task1-SR 配对 accuracy | `wilcoxon_paired` | — |
| A2-Cosine | 同句异被试 vs 异句异被试 | `mannwhitney_u` | — |
| A2-Cosine | EEG 各组 vs 噪声对应组 | `mannwhitney_u` | Holm-Bonferroni |
| A2-Eta | 840 维 $\eta^2_\text{subj}$ vs $\eta^2_\text{sent}$ 配对 | `wilcoxon_paired` | BH-FDR |
| A2-Eta | $\eta^2_\text{EEG}$ vs null | `permutation_eta` | BH-FDR |
| A2-band | 8 频带 $\eta^2$ 对比 | `friedman_nemenyi` | Nemenyi |
| A3-LP | $\Delta$Top-K（5 折配对） | `wilcoxon_paired` | Holm-Bonferroni |
| A3-Retrieval / A3-SessionRetrieval | R@K vs 随机基线 $K/M$ | `binomial_vs_baseline` | Holm-Bonferroni |
| A3-Retrieval / A3-SessionRetrieval | EEG vs 噪声 R@K null | `permutation_retrieval` | Holm-Bonferroni |

### B 线（`scripts/analysis/run_significance_tests.py`）

| 块 | 比较对 | 方法 | 校正 |
|----|--------|------|------|
| `pairwise` | 6 条件对 × {R@1, R@5, R@10, MRR, mean_rank} 配对 rank | `compare_pair`（= `wilcoxon_paired` + `bootstrap_mean_diff`）| Holm-Bonferroni 单模型 36 组（6 pair × (1 wilcoxon_rank + 5 delta)）|
| `vs_random_baseline` | R@K vs $K/M$ | `binomial_vs_baseline` | — |
| `rank_distribution` | ranks vs Uniform$[1, M]$ | `ks_vs_uniform` | — |
| `grouped` | 按 subject / task / dataset 的 rank 异质性 | `kruskal_dunn` | Dunn + BH-FDR |
| `permutation_retrieval` | 每 noise 条件固定表示 → null R@K / MRR | `permutation_retrieval`（$N_q > 5000$ 跳过）| — |
| `friedman_by_noise` | 4 模型 per-query rank 横向对比 | `friedman_nemenyi` | Nemenyi |
| `bh_fdr_global` | 跨模型 120 组 $p$ 值 | `bh_fdr` | BH-FDR |

### 模式诊断与显著性门限（B 线）

| 结论 | 充要证据 |
|------|---------|
| 模式 A（编码器完全无效）| real-vs-{gaussian, shuffle, zero} 三对 permutation $p > \alpha_\text{adj}$ |
| 模式 B（学到统计性未学到对应）| real-vs-gaussian $p < \alpha_\text{adj}$，且 real-vs-shuffle $p > \alpha_\text{adj}$ |
| 模式 C（学到跨模态对应）| real-vs-gaussian 与 real-vs-shuffle 均 $p < \alpha_\text{adj}$，且 $|d_z| > 0.2$ |
| b4 GLIM `zero > real` 异常 | R@10 / MRR 的 zero-vs-real 配对 $p < \alpha_\text{adj}$，且方向 zero > real |
| 模型 X 显著优于模型 Y | Friedman 整体 $p < 0.05$ 且 Nemenyi X vs Y $p < \alpha_\text{adj}$ |

---

## `compare_pair`：B 线常用组合 API

**函数**：`compare_pair(name_a, ranks_a, name_b, ranks_b, metric_ks, n_boot, seed)`

将两条件 per-query ranks 一次性产出以下四类对比：

$$\text{result} = \begin{cases}
\text{wilcoxon\_rank}: & \text{wilcoxon\_paired}(r_a, r_b) \\
\text{r@k\_delta}:     & \text{bootstrap\_mean\_diff}\big(\mathbf{1}[r_a \leq k],\ \mathbf{1}[r_b \leq k]\big),\ k \in \{1,5,10\} \\
\text{mrr\_delta}:     & \text{bootstrap\_mean\_diff}(1/r_a,\ 1/r_b) \\
\text{mean\_rank\_delta}: & \text{bootstrap\_mean\_diff}(r_a,\ r_b)
\end{cases}$$

`run_significance_tests.py` 将 `r@1 wilcoxon` + 5 个 delta（共 6 个 p 值）× 6 条件对 = 36 个 p 值送入 Holm-Bonferroni，得到单模型条件对的校正结论。

---

## 数值稳定性与边界处理

| 场景 | 处理 |
|------|------|
| 所有 $\delta_i = 0$ | Wilcoxon 返回 `p=1.0, dz=0.0, ci=[0,0]` |
| $n_a < 2$ 或 $n_b < 2$ | 返回 `error="insufficient_samples"` |
| $\text{SS}_\text{total} < 10^{-12}$ | $\eta^2 = 0$ 避免除零 |
| $p_\text{perm} = 0$ | 下限 $1/P$ 防止 $\log p = -\infty$ |
| $p_\text{bootstrap} > 1$ | 上限 $\min(1, \cdot)$ 防止 $2(1-q) > 1$（S-4 修复） |
| $s_\text{null} < 10^{-12}$ | z-偏离分母加 $10^{-12}$ |
| $N_q > 5000$ | permutation retrieval 跳过，标注 `skipped=true` |
| `scikit-posthocs` 缺失 | Nemenyi 回退至 CD 法；Dunn 标注 `error="scikit-posthocs_not_installed"` |
| 显著性判定 | `p_adj <= alpha`（含等号，与统计学惯例一致；S-9 修复） |
| A3 检索 $k \geq M$ | 跳过二项检验，标注 `skipped=true, reason="k>=M"`（S-8 修复） |

---

## 输出文件结构

### A 线：`test_outputs/line_a/dataset_validity/significance_tests.json`

```json
{
  "a1a": {
    "word_vs_noise":   { "top_1": { "test": "wilcoxon", "p": ..., "effect": {"cohens_dz": ...}, "ci95": [..., ...] }, ... },
    "vs_random":       { "top_1": { "test": "binomial", "p": ..., "ci95": [..., ...] }, ... }
  },
  "a1d":        { ... },
  "a2_cosine":  { "same_sent_cross_subj_vs_diff_sent": { "test": "mannwhitney_u", ... } },
  "a2_eta":     { "subj_vs_sent_wilcoxon": { ... }, "eta_permutation": { ... } },
  "a2_band":    { "friedman": { ... }, "nemenyi": { ... } },
  "a3_lp":      { "delta_top_k": { ... } },
  "a3_retrieval":         { "eeg_vs_noise": { ... }, "vs_random": { ... } },
  "a3_session_retrieval": { ... }
}
```

### B 线：`test_outputs/line_b/{model}/significance_tests.json`

```json
{
  "model": "cet_mae",
  "alpha": 0.05,
  "alpha_adjusted_global": 0.000417,
  "n_noises_loaded": 4,
  "pairwise": {
    "real_vs_gaussian": {
      "wilcoxon_rank":    { "p": ..., "effect": {"cohens_dz": ...}, "ci95": [..., ...], "p_adjusted_holm": ..., "significant_holm": true },
      "r@1_delta":        { "p": ..., "ci95": [..., ...], "p_adjusted_holm": ..., "significant_holm": true },
      "r@5_delta":        { ... },
      "r@10_delta":       { ... },
      "mrr_delta":        { ... },
      "mean_rank_delta":  { ... }
    },
    "real_vs_shuffle": { ... },
    ...,
    "_correction": { "method": "holm_bonferroni", "n_tests": 36, "alpha_first": 0.00139 }
  },
  "vs_random_baseline":   { "real": { "r@1": { ... }, "r@5": { ... }, "r@10": { ... } }, ... },
  "rank_distribution":    { "real": { "test": "ks_uniform", ... }, ... },
  "grouped":              { "real": { "subjects": { "test": "kruskal", ... }, "tasks": { ... } }, ... },
  "permutation_retrieval":{ "real": { "observed": { ... }, "null": { "r@1": { ... } } }, ... }
}
```

### B 线跨模型：`test_outputs/line_b/significance_summary.json`

```json
{
  "models": ["cet_mae", "eeg_to_text", "eeg2text", "glim"],
  "friedman_by_noise": {
    "real":     { "statistic": ..., "p": ..., "kendalls_w": ..., "nemenyi": { "cet_mae_vs_glim": ... } },
    "gaussian": { ... }
  },
  "bh_fdr_global": {
    "method": "bh_fdr", "n_tests": 120,
    "per_test": [ { "key": "cet_mae/real_vs_gaussian/r@1", "p": ..., "p_adj": ..., "significant": true }, ... ]
  }
}
```

---

## 运行命令

### A 线（在 `validate_eeg_signal.py` 内联调用）

A 线显著性检验作为数据有效性诊断的尾部模块，由 `validate_eeg_signal.py` 直接调用 `evaluation/significance.py` 的函数，并写入 `significance_tests.json`。无单独入口脚本。

### B 线（独立主脚本）

```bash
python benchmark_eval/scripts/analysis/run_significance_tests.py \
    --results-dir benchmark_eval/test_outputs \
    --n-perm 1000 \
    --n-boot 1000 \
    --alpha 0.05
```

可选参数：

| 参数 | 作用 |
|------|------|
| `--skip-permutation` | 跳过 `permutation_retrieval` 以节省运行时间 |
| `--skip-grouped` | 跳过 `Kruskal-Wallis + Dunn` 分组检验 |
| `--n-perm` | 置换次数（默认 1000） |
| `--n-boot` | bootstrap 次数（默认 1000） |

### 依赖

必选：`numpy`、`scipy`。
可选：`scikit-posthocs`、`pandas`（若缺失则 Nemenyi 回退 CD 法、Dunn 不可用）。

---

## 输出摘要

| 文件 | 内容 |
|------|------|
| `benchmark_eval/evaluation/significance.py` | 9 类检验 + 2 类多重校正 + 1 组合 API 的统一封装 |
| `benchmark_eval/scripts/analysis/run_significance_tests.py` | B 线显著性检验主脚本 |
| `line_a/dataset_validity/significance_tests.json` | A 线 7 个子实验的检验结果 |
| `line_b/{model}/significance_tests.json` | 每模型 5 块（pairwise / vs_random / KS / grouped / permutation）|
| `line_b/significance_summary.json` | 跨模型 Friedman + Nemenyi + BH-FDR 全局 120 组 |
| `line_b/significance_tests.log` | B 线运行日志 |
