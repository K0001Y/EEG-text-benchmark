# EEG-to-Text Benchmark 待办清单

> 基于全面代码审计 + 显著性检验专项审查生成，按优先级排序。
> **最后更新：2026-05（v5，新增显著性检验 Bug 清单 + A/B 线悖论修复）**

---

## 一、【最高优先级】显著性检验 Bug 修复（新增）

> 依据：2026-05 对 `benchmark_eval/evaluation/significance.py`、`benchmark_eval/scripts/analysis/run_significance_tests.py`、`docs/detail/significance_tests_details.md` 的系统审计。
> 以及 A/B 线结论悖论溯源：A 线显示数据有句子信号，B 线却 real≈gaussian 的直接原因。

### Critical（必须修）

- [x] **S-1: `_align_ranks_by_query` 样本量坍缩（N=1858 → N=54，功效损失 97%）** ✅ 已修复（2026-05-11）
  - 文件：`benchmark_eval/scripts/analysis/run_significance_tests.py` L97-L143
  - 修复方案：按行索引对齐 + 强校验所有 noise 行顺序一致；跨模型 Friedman 同步修复
  - 前置验证：cet_mae / eeg_to_text / eeg2text / glim 四模型在 real/gaussian/shuffle/zero 下 1858 行 `(subject, sentence_id, task)` 序列完全一致
  - 结果：重跑后所有检验 n=1858（原 54），Friedman shape=(1858, 4)
  - **结论订正**：原判断"B 线 real≈gaussian 是功效不足的产物"需修正——在 N=1858 下 real vs gaussian 的 Wilcoxon p 仍不显著（CET-MAE p=0.169, EEG-To-Text p=0.463, EEG2Text p=0.460, GLIM p=0.114），效应量 |d_z| 均 < 0.04（negligible）。说明**预训练模型确实没有区分真实 EEG 和同分布高斯噪声的能力**，A/B 悖论本质是数据含句子信号但这些模型的目标函数没有激励它们学到判别性表征
  - 副产物发现：CET-MAE 对 real vs shuffle (p_adj=0.0015) / real vs zero (p_adj=0.010) 显著；GLIM 对 real vs zero 的 r@5/r@10 显著；说明模型能感知结构破坏（shuffle/zero）但感知不到幅度谱一致的 gaussian 噪声

- [ ] **S-2: `ks_vs_uniform` 的 scipy `args` 传参错误**
  - 文件：`benchmark_eval/evaluation/significance.py` L336-L351
  - 现状：`kstest(ranks, "uniform", args=(1, M))`，scipy 的 uniform 用 `(loc, scale)` 参数化，实际对应 Uniform[1, 1+M]
  - 应为：`args=(1, M - 1)`（表示 Uniform[1, M]，离散 rank 的连续化近似）或 `args=(0.5, M)`（Uniform[0.5, M+0.5]）
  - 影响：所有 KS vs Uniform 的 p 值均偏差（M 越大越显著偏保守）

- [x] **S-3: `N_GLOBAL_TESTS` 常量硬编码错误** ✅ 已修复（2026-05-11）
  - 文件：`benchmark_eval/scripts/analysis/run_significance_tests.py` L59-L62
  - 修复：改为 `N_GLOBAL_TESTS = len(MODELS) * len(PAIR_ORDER) * len(METRICS_FOR_CORRECTION)` = 4 × 6 × 5 = 120
  - 重跑后 `alpha_adjusted_global = 0.05/120 = 4.17e-4`（原 0.05/72 = 6.94e-4）
  - BH-FDR 全局：n_tests=120，16 个显著

### High

- [ ] **S-4: `bootstrap_mean_diff` 的 p 值可能 > 1.0**
  - 文件：`benchmark_eval/evaluation/significance.py` L249-L250
  - 现状：`p_emp = max(2.0 * (1.0 - prop), 1.0 / n_boot)`，当 prop < 0.5 时 `2*(1-prop) > 1`
  - 修复：`p_emp = min(1.0, max(2.0 * (1.0 - prop), 1.0 / n_boot))`

- [ ] **S-5: `kruskal_dunn` 的 η²_H 错用 Cohen's d 阈值判级**
  - 文件：`benchmark_eval/evaluation/significance.py` L458
  - 现状：`"label": interpret_effect(eta2_h)`，复用 Cohen's d 的 0.2/0.5/0.8 阈值
  - 应为：η²_H 的常用阈值（0.01 small / 0.06 medium / 0.14 large），或新增 `interpret_eta2()` 函数

- [ ] **S-6: 文档与代码数值不一致**
  - 文件：`docs/detail/significance_tests_details.md`
  - 问题清单：
    - L406 "Holm-Bonferroni 单模型 18 组" vs L439 "36 个 p 值" vs L496 示例 "n_tests: 36"
    - L369 "72 组全局校正" vs 实际应为 120（与 S-3 同步）
    - L122 Mann-Whitney U 公式用 `min(U_a, n_a·n_b - U_a)`，但代码用 scipy 返回的 U_a
  - 修复：与 S-3 修复同步更新，确认单模型 n_tests = 6 × 5 = 30 还是 6 × 3 = 18 后统一

### Medium

- [ ] **S-7: `permutation_retrieval` 采用 gt_idx 置换而非均匀候选**
  - 检查是否符合 `significance_tests_details.md` 描述的原假设
  - 若不符，修正实现或更新文档

- [ ] **S-8: A 线 `A3_session_retrieval` 的 n 用 M 的不一致**
  - 文件：`benchmark_eval/scripts/diagnostics/validate_eeg_signal.py`
  - 确认 A3 检索的样本量统计口径

- [ ] **S-9: `adj < alpha` vs `adj <= alpha` 约定统一**
  - 文件：`benchmark_eval/evaluation/significance.py`
  - 建议统一用 `<=`（与统计学界惯例一致），并在文档注明

- [ ] **S-10: `significance_tests.json` 重复写入问题**
  - 排查 `run_significance_tests.py` 的写盘路径逻辑

- [ ] **S-11: 嵌入未归一化校验**
  - 检索前确认所有模型的 embeddings 已 L2 归一化（或统一归一化），避免度量口径不一致

### 验证步骤（修复后）

- [x] **S-V: 在 N=1858 下重跑 B 线显著性，验证悖论溯源** ✅ 已完成（2026-05-11）
  - 已重跑 `run_significance_tests.py --skip-permutation`
  - 所有 pair 的 `"n"` 从 54 → 1858，跨模型 Friedman shape=(1858, 4)
  - **与预期不符的关键发现**：real vs gaussian 在 N=1858 下**仍不显著**（CET-MAE p=0.169, EEG-To-Text p=0.463, EEG2Text p=0.460, GLIM p=0.114），|d_z| < 0.04
  - 修正后的 A/B 悖论解释：数据层有句子信号（A 线），但四个预训练模型在其原始训练目标下没有激励去学判别性表征，导致它们对"真实 EEG"与"同分布高斯噪声"几乎无法区分
  - 进一步细分结论：
    - CET-MAE 对 real vs shuffle (p_adj=0.0015, d_z=-0.100) 和 real vs zero (p_adj=0.010, d_z=-0.080) 显著 → 能感知结构破坏但对幅度谱保持的 gaussian 失效
    - GLIM 对 real vs zero 的 r@5/r@10 (p_adj=0.034) 显著
    - EEG-To-Text / EEG2Text 对所有 noise 条件几乎均不显著 → 检索性能几乎独立于输入
  - 待办：将订正结论写入 `docs/EEG_to_Text_Contrast_Experiment_Report.md`

---

## 二、剩余待处理的代码/评估问题

### High

- [ ] **H-5: EEG 输入格式异构性未文档化**（部分处理）
  - `eeg2text_wrapper.py` 注释中已说明异构输入对比注意事项，完整评估报告文档标注待补充

- [ ] **H-7: 分组评估指标缺乏统计学保障**（部分修复）
  - Bootstrap 置信区间未实现（原 Phase 4，现可并入 S-V 扩展）

### Medium

- [ ] **M-3: wrapper 间存在大量重复代码**
  - 提取公共逻辑到基类，工作量较大，列为后续优化

- [ ] **M-7: 日志级别使用不规范**

### Architecture

- [ ] **A-2: 配置管理分散**（部分修复）
  - 生成参数已整合到 `eval_config.yaml`，统一 `Config` 类层级管理待实现

---

## 三、实验设计问题

- [ ] **E-2: 配置中定义的消融实验未实现**
- [ ] **E-3: 缺少结果可视化与对比工具**

---

## 四、已观测到的异常现象

- [ ] **O-1: GLIM 评估性能异常低**（待排查）
- [ ] **O-2: EEG2Text 评估的 WER 值异常**（待排查）

---

## 五、数据处理流程遗留

- [ ] **D-3: 验证并修正 GLIM 维度转换逻辑**（部分处理）
  - 当前转换逻辑已整理（分组平均 + 线性插值）
  - 与 GLIM 原始训练数据分布的对齐验证待完成

---

## 六、噪声测试进阶

- [ ] **N-2: 实现进阶噪声测试（频段掩码 / 时间掩码 / 渐进噪声）**
- [ ] **N-3: 噪声测试结果输出标准化**

---

## 七、诊断线 A 代码实现对齐（v3 规范）

> 依据：`docs/detail/experiment_A_details.md`（三组信号并行 + LOSO 5折CV）
> 和 `docs/detail/unified_dataset.md`（nfixations_word 字段）
> **必须先完成 A-impl-1~3，再运行实验（DV-1a/b/c）**

### 子任务：build_unified_dataset.py

- [ ] **A-impl-1: 新增 nfixations_word 字段**
  - 文件：`benchmark_eval/data_processing/build_unified_dataset.py`
  - 在 `build_samples_for_task` 的 record 构造块中，在步骤4（生成EEG字段）之后新增步骤4.5
  - v1路径：从 `sent_obj["word"]` 中读取已过滤词的 `word_obj["nFixations"]`
  - v2路径：从 `word_obj["nFixations"]`（`data_dict["nFix"]`）读取
  - 代码：`nfixations = [float(w["nFixations"]) for w in valid_words[:max_len]]`，padding 填 0.0，shape `(max_len,)`，`dtype=np.float32`
  - **完成后需重新构建 unified_zuco.pkl**

- [ ] **A-impl-1b: 重新构建 unified_zuco.pkl**
  - 运行 `python benchmark_eval/data_processing/build_unified_dataset.py --zuco-root ... --output benchmark_eval/data/unified_zuco.pkl`
  - 验证新字段：`nfixations_word` shape `(56,)` 且有效词位 ≥ 1.0

### 子任务：validate_eeg_signal.py 逻辑重构

- [ ] **A-impl-2: collect_samples 改为只加载 test 集 + 读取 sent_eeg_raw / nfixations_word**
  - 文件：`benchmark_eval/scripts/diagnostics/validate_eeg_signal.py`
  - 移除 `collect_train_samples` / `train_data` 分路加载（A1 只用 test 集做 LOSO CV）
  - 新增读取 `sent_eeg_raw`（shape (840,)）和 `nfixations_word`

- [ ] **A-impl-3: 实现 LOSO 5折CV 框架**
  - `StratifiedGroupKFold(n_splits=5).split(X, y, groups)`
  - 每折：`StandardScaler` fit on train → transform test，`LogisticRegression` fit/predict
  - 收集 Top-1/5/10，输出均值 ± std

- [ ] **A-impl-4: 实现三组信号特征提取**
  - 词级 EEG：已有 `extract_features`，无需改动
  - 句级 EEG：新函数 `extract_sent_features(sent_eeg_list)` → 逐样本 z-score，返回 `(N, 840)`
  - 高斯噪声：新函数 `generate_noise_features(N, dim=840, base_seed=42)`

- [ ] **A-impl-5: 重写 A1a/A1b/A1c 入口，三组信号各跑 LOSO 5折**
  - A1a：三组特征（词级 mean_pool / 句级 / 噪声）各跑
  - A1b：词级用 nfixations 加权；句级和噪声 fallback 到 A1a
  - A1c：词级用 band_separated；句级和噪声复用 A1a

- [ ] **A-impl-6: 修复 A3-LP 的 per-subject 统计泄漏（改为 LOSO fold 内计算）**
  - 当前：全局 train 集计算 per-subject μ/σ → 有数据泄漏
  - 修复：在 LOSO 每一折 train fold 内部，仅对 train fold 中的被试计算 μ/σ

- [ ] **A-impl-7: 更新 main() 流程和输出格式**

### 诊断线 A 实验运行

- [ ] **DV-1a: Linear Probe — Mean-Pool 基线**（三组信号并行）
- [ ] **DV-1b: Linear Probe — Duration-Weighted Pool**
- [ ] **DV-1c: Linear Probe — Band-Separated**
- [ ] **DV-2: 被试效应 vs 句子效应分析**（余弦相似度 + η² 方差分解 + t-SNE）
- [ ] **DV-2-band: 频带级 η² 分析**
- [ ] **DV-3: 去被试化信号恢复验证**（LOSO 框架下 per-subject z-score）

---

## 八、诊断线 B 剩余任务

- [ ] **NC-3: 运行 4 模型 × 3 噪声条件 = 12 组实验**（已有 16 组结果，确认完整性）
- [ ] **NC-4: 生成综合对比分析报告**（已创建 `compare_contrast_results.py`，需结合 S-V 修复后重跑）
- [ ] **NC-5: 根据双诊断线决策树得出归因结论，更新文档**
  - 关键：结合 S-1 修复后的 N=1858 结果重写 `docs/EEG_to_Text_Contrast_Experiment_Report.md`

---

## 九、建议执行顺序

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase S**（新增，最高优先级） | S-1, S-2, S-3（3 个 Critical Bug） | 待修复 |
| **Phase S+** | S-4~S-11（显著性检验其他问题） | 待修复 |
| **Phase S-V** | 修复后重跑 B 线显著性验证 A/B 悖论 | 待执行 |
| **Phase 9a** | A-impl-1~7（诊断线 A 代码对齐） | 待实现 |
| **Phase 9b** | DV-1a/b/c + DV-2 + DV-2-band（运行 A 线） | 待 Phase 9a 完成 |
| **Phase 9d** | NC-3~NC-5（B 线收尾 + 综合报告） | 待 Phase S-V 完成 |
| **Phase 4** | E-2, E-3（消融 + 可视化） | 待实现 |
| **Phase 5** | O-1, O-2（异常现象定位） | 待排查 |

---

*生成时间：2026-04-09；v2 2026-04；v3 2026-04-11；v4 2026-04-23；*
*v5 2026-05-11：新增第一节显著性检验 Bug 清单（S-1~S-11），溯源 A/B 线结论悖论；删除已完成项（C-1/C-2/H-1/H-2/H-3/H-4/H-6/M-1/M-2/M-4~M-8/L-1/L-2/A-1/A-4/D-1~D-5/N-1/R-1~R-3/NC-1~NC-2）*
*审计范围：benchmark_eval 全部核心模块 + evaluation/significance.py + scripts/analysis/run_significance_tests.py + docs/detail/significance_tests_details.md*
