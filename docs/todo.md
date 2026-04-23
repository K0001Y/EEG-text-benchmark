# EEG-to-Text Benchmark 优化待办清单

> 基于全面代码审计生成，按优先级排序。
> **最后更新：2026-04（v2 优化完成）**

---

## 一、评估严谨性问题

### Critical

- [x] **C-1: max_len 参数全局不一致**（已修复）
  - `eval_config.yaml` 已统一为 `max_len: 56`
  - `build_unified_dataset.py` 默认值已改为 `MAX_LEN = 56`（从 `constants.py` 引入）
  - 修复文件：`benchmark_eval/config/eval_config.yaml`、`benchmark_eval/data_processing/build_unified_dataset.py`、`benchmark_eval/constants.py`

- [x] **C-2: BERTScore 计算在离线环境中失效**（已修复）
  - 添加了 `OSError / ValueError` 捕获，失败时返回 `float('nan')` 而非 0.0
  - 日志会输出离线降级提示
  - 修复文件：`benchmark_eval/evaluation/metrics.py`

### High

- [x] **H-1: 生成参数在不同 wrapper 间严重不一致**（已修复）
  - 在 `eval_config.yaml` 的 `models.{name}.generation` 中定义各模型独立生成参数
  - EEG-To-Text：beam=5, do_sample=True, repetition_penalty=5.0（原始论文）
  - EEG2Text：greedy（beam=1）
  - CET-MAE：greedy（beam=1）
  - GLIM：beam=2

- [x] **H-2: 随机种子设置不完整，结果不可复现**（已修复）
  - 在 `eval_runner.py` 中新增 `set_seed()` 函数，覆盖 random / numpy / torch / torch.cuda / cudnn
  - 在 `main()` 入口处调用 `set_seed(args.seed)`

- [x] **H-3: DataLoader 多进程下结果不确定**（已修复）
  - DataLoader 使用 `generator = torch.Generator().manual_seed(seed)` 固定随机性
  - 默认 `num_workers: 0`

- [x] **H-4: 异常捕获过于宽泛，掩盖真实错误**（已修复）
  - `metrics.py` 中缩小为具体异常类型（`ZeroDivisionError`, `OSError`, `ValueError`）
  - 所有捕获点均添加 `logger.warning()` 记录具体错误信息

- [ ] **H-5: EEG 输入格式异构性未文档化**（部分处理）
  - `eeg2text_wrapper.py` 注释中已说明异构输入对比注意事项
  - 完整的评估报告文档标注待补充

- [x] **H-6: 数据划分边界条件未处理**（已修复）
  - `build_unified_dataset.py` 中当 `n_total < 10` 时抛出 `ValueError`
  - 三个 phase 非空校验已添加

- [x] **H-7: 分组评估指标缺乏统计学保障**（部分修复）
  - 每个分组添加了 `sample_count` 字段
  - Bootstrap 置信区间未实现（待 Phase 4）

---

## 二、代码质量问题

### Medium

- [x] **M-1: 关键参数硬编码，与配置文件重复**（已修复）
  - GLIM wrapper 的 `input_eeg_len`, `input_dim` 等改为从外部传入
  - `eval_config.yaml` 中 `models.glim` 统一定义所有参数

- [x] **M-2: 大量魔法数字未定义为命名常量**（已修复）
  - 新增 `benchmark_eval/constants.py`
  - 定义：`MAX_LEN`, `EEG_CHANNELS`, `EEG_WORD_DIM`, `SPECTRO_STEPS`, `SPECTRO_FREQS`, `GLIM_EEG_LEN`, `GLIM_EEG_DIM` 等

- [ ] **M-3: wrapper 间存在大量重复代码**（未处理）
  - 提取公共逻辑到基类的工作量较大，列为后续优化

- [x] **M-4: 部分 wrapper 缺少输入验证**（已修复）
  - `eeg2text_wrapper.py`、`cet_mae_wrapper.py`、`glim_wrapper.py` 均添加了入口验证

- [x] **M-5: 指标返回值类型不一致**（已修复）
  - 统一约定：计算失败返回 `float('nan')`，成功返回实际数值
  - `_nan_metrics()` 辅助函数统一空数据返回值

- [x] **M-6: 类型标注不完整或有误**（已修复）
  - `metrics.py` 中 `any`（小写）改为 `Any`

- [ ] **M-7: 日志级别使用不规范**（未处理）

- [x] **M-8: 废弃代码未清理**（已修复）
  - `glim_wrapper.py` 中删除了第 208-223 行的废弃转换注释代码

### Low

- [x] **L-1: 存在未使用的导入和重复导入**（已修复）
  - `glim_wrapper.py` 中删除了重复的 `import sys` 和 `import os`

- [ ] **L-2: 文档与代码不同步**（通过本次文档更新修复）

---

## 三、架构设计问题

### Medium

- [x] **A-1: Wrapper 接口设计违反 Liskov 替换原则**（部分修复）
  - `model_wrappers.py` 明确了接口契约：`batch` 是主要数据来源
  - 各 wrapper 使用 `batch.get("eeg_word_norm1d", eeg)` 模式

- [ ] **A-2: 配置管理分散**（部分修复）
  - 生成参数已整合到 `eval_config.yaml`
  - 统一 `Config` 类层级管理待实现

- [ ] **A-3: 噪声控制实验支持不完整**（已实现）
  - `dataset.py` 新增 `noise_type="zero"` 和 `shuffle_mode` 支持
  - 四个检索脚本均已添加 `--noise-type {real,gaussian,shuffle,zero}` 参数

- [x] **A-4: 评估输出 schema 不统一**（已修复）
  - `eval_runner.py` 统一输出 `"overall"` + `"grouped"` + `"failed_count"` + `"num_samples"` schema

---

## 四、实验设计问题

### Medium

- [ ] **E-1: 缺少统计显著性检验**（未处理）
- [ ] **E-2: 配置中定义的消融实验未实现**（未处理）
- [ ] **E-3: 缺少结果可视化与对比工具**（未处理）

---

## 五、已观测到的异常现象

- [ ] **O-1: GLIM 评估性能异常低**（待排查）
- [ ] **O-2: EEG2Text 评估的 WER 值异常**（待排查）

---

## 六、数据处理流程优化

- [x] **D-1: 新增 spectrogram 格式替代 raw 时序存储**（已实现）
  - `build_unified_dataset.py` 新增 `build_spectrogram()` 函数
  - 参数：`fs=500, nperseg=128, noverlap=64`（与 EEG2Text 原始对齐）
  - 输出字段：`eeg_spectro`，shape (374, 65)，替代原 (24000, 105)
  - 存储节省：约 99%（~9.6MB → ~97KB per sample）

- [x] **D-2: 统一字段命名规范**（已实现）
  - `eeg_raw` → `eeg_word_raw`
  - `eeg_normalized_1d` → `eeg_word_norm1d`
  - `eeg_normalized_2d` → `eeg_word_norm2d`
  - `eeg_eeg2text` → `eeg_spectro`（同时更改格式）
  - `mask` → `mask_word`（保留别名）
  - `mask_with_sent` → `mask_word_with_sent`
  - `mask_eeg2text` → `mask_spectro`
  - 旧字段名别名保留，向后兼容旧 PKL 文件

- [ ] **D-3: 验证并修正 GLIM 维度转换逻辑**（部分处理）
  - 当前转换逻辑已整理（分组平均 + 线性插值）
  - 与 GLIM 原始训练数据分布的对齐验证待完成

- [x] **D-4: 统一 max_len 为 56 并修正 CET-MAE 句级追加逻辑**（已修复）
  - `build_unified_dataset.py` 默认 `max_len = MAX_LEN = 56`
  - `eval_config.yaml` `max_len: 56`
  - CET-MAE 句级 EEG 作为独立 `sent_eeg_raw` 字段存储，wrapper 决定是否追加

- [x] **D-5: 各 wrapper 适配新字段名与数据格式**（已实现）
  - 所有 wrapper 已更新为优先读取新字段名，向后兼容旧字段名

## 七、随机噪声测试方案

- [x] **N-1: 实现基线噪声测试（Gaussian / Uniform / Shuffle / Zero）**（已完成）
  - `dataset.py` 支持 gaussian / uniform / zero 噪声模式
  - `dataset.py` 新增 shuffle_mode（全局 derangement）
  - 四个检索脚本统一 `--noise-type` 接口
- [ ] **N-2: 实现进阶噪声测试（频段掩码 / 时间掩码 / 渐进噪声）**
- [ ] **N-3: 噪声测试结果输出标准化**

## 八、检索测试方案

- [x] **R-1: 扩展 wrapper 基类，新增 encode_eeg_to_embedding / encode_text_to_embedding 接口**（已通过独立脚本实现）
- [x] **R-2: 实现检索评估管道**（已完成四个模型的检索评估）
- [x] **R-3: 支持分组检索评估**（已支持 by_task / by_subject / by_dataset 分组）

## 九、性能归因对比实验

> 详见 `docs/detail/experiment_A_details.md`（v3 修订版，含三组信号并行规范）
> 和 `docs/detail/unified_dataset.md`（v3 修订版，含 nfixations_word 字段规范）

### 诊断线 A：原始数据集有效性验证（CPU，无需 GPU）

> ⚠️ **代码与规范存在差距，需要对齐后才能运行**（见第十节）

- [ ] **DV-1a: Linear Probe — Mean-Pool 基线**（三组信号：词级EEG / 句级EEG / 高斯噪声，各跑 LOSO 5折CV）
- [ ] **DV-1b: Linear Probe — Duration-Weighted Pool**（词级 EEG 用 nfixations_word 加权；句级/噪声 fallback 到 DV-1a 结果）
- [ ] **DV-1c: Linear Probe — Band-Separated**（词级 band_sep；句级/噪声与 DV-1a 等价）
- [ ] **DV-2: 被试效应 vs 句子效应分析**（余弦相似度分组对比 + η² 方差分解 + t-SNE 多 perplexity 可视化）
- [ ] **DV-2-band: 频带级 η² 分析**（对 8 个频带分别计算 η²_sentence vs η²_subject）
- [ ] **DV-3: 去被试化信号恢复验证**（LOSO框架下per-subject z-score + 被试聚合检索，严格无数据泄漏）

**实施**：`benchmark_eval/scripts/validate_eeg_signal.py`，输出到 `benchmark_eval/test_outputs/dataset_validity/`
**状态**：脚本已创建但逻辑与 v3 规范不符，需对齐后运行（见第十节实现对齐任务）

### 诊断线 B：噪声对照实验（需 GPU，三层统一架构）

- [x] **NC-1: 扩展 dataset.py 噪声类型和 Shuffle 支持**
  - 新增 `noise_type="zero"` 分支（全零张量替代 EEG）
  - 新增 `shuffle_mode` 支持，实现全局 derangement（无不动点 permutation）
  - `UnifiedDataset` 作为协调层，生成权威 permutation 和种子序列
- [x] **NC-2: 为检索脚本添加统一噪声接口**
  - 所有脚本新增 `--noise-type {real,gaussian,shuffle,zero}` 参数
  - CET-MAE/EEG-To-Text/GLIM：在 `UnifiedDataset` 数据加载时应用噪声
  - EEG2Text：查询 `UnifiedDataset` 的 permutation，在编码阶段应用（三层架构实现层适配）
  - 输出目录自动添加后缀（`_gaussian` / `_shuffle` / `_zero`）
- [ ] **NC-3: 运行 4 模型 × 3 噪声条件 = 12 组实验**
- [ ] **NC-4: 生成综合对比分析报告**
  - 已创建 `benchmark_eval/scripts/compare_contrast_results.py`
  - 读取诊断线 A 全部结果 + 诊断线 B 的 16 个 `retrieval_metrics.json`
  - v2 新增统计检验：置换检验 p-value、Bootstrap 95% CI、Cohen's d 效应量
  - 输出到 `benchmark_eval/test_outputs/contrast_summary.json`
- [ ] **NC-5: 根据双诊断线决策树得出归因结论，更新文档**

---

## 十、诊断线 A 代码实现对齐（v3 规范）

> 依据：`docs/detail/experiment_A_details.md`（三组信号并行 + LOSO 5折CV）
> 和 `docs/detail/unified_dataset.md`（nfixations_word 字段）
> **必须先完成 A-impl-1~3，再运行实验（DV-1a/b/c）**

### 子任务：build_unified_dataset.py

- [ ] **A-impl-1: 新增 nfixations_word 字段**
  - 文件：`benchmark_eval/data_processing/build_unified_dataset.py`
  - 在 `build_samples_for_task` 的 record 构造块中，在步骤4（生成EEG字段）之后新增步骤4.5
  - v1路径：从 `sent_obj["word"]` 中读取已过滤词的 `word_obj["nFixations"]`
  - v2路径：从 `word_obj["nFixations"]`（`data_dict["nFix"]`）读取（注意v2词已按 `"GD_EEG" in data_dict` 过滤）
  - 代码：`nfixations = [float(w["nFixations"]) for w in valid_words[:max_len]]`，padding 填 0.0，shape `(max_len,)`，`dtype=np.float32`
  - 注意：v1 有效词均已过滤 `nFixations > 0`，v2 需确认 `nFix` 字段名
  - **完成后需重新构建 unified_zuco.pkl**

- [ ] **A-impl-1b: 重新构建 unified_zuco.pkl**
  - 运行 `python benchmark_eval/data_processing/build_unified_dataset.py --zuco-root ... --output benchmark_eval/data/unified_zuco.pkl`
  - 验证新字段：从pkl随机采样，确认 `nfixations_word` shape `(56,)` 且有效词位 ≥ 1.0

### 子任务：validate_eeg_signal.py 逻辑重构

- [ ] **A-impl-2: collect_samples 改为只加载 test 集 + 读取 sent_eeg_raw / nfixations_word**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - 移除 `collect_train_samples` / `train_data` 分路加载（A1 只用 test 集做 LOSO CV）
  - 在 `collect_samples` 中额外读取：
    - `sent_eeg_raw`：`sample["sent_eeg_raw"].numpy()`，shape `(840,)`；缺失时用 `np.zeros(840, dtype=np.float32)`
    - `nfixations_word`：`sample.get("nfixations_word")`；缺失时为 None
  - 输出新增字段：`sent_eeg_list`（list of ndarray (840,)）、`nfixations_list`（list of ndarray (56,) or None）

- [ ] **A-impl-3: 实现 LOSO 5折CV 框架**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - 引入 `from sklearn.model_selection import StratifiedGroupKFold`
  - 新函数 `run_loso_linear_probe(X, y, groups, n_classes, variant_name, logger)`：
    - `StratifiedGroupKFold(n_splits=5).split(X, y, groups)`
    - 每折：`StandardScaler` fit on train → transform test，`LogisticRegression` fit/predict
    - 收集 Top-1/5/10，输出均值 ± std
  - 输出格式：`{"mean_top1": ..., "std_top1": ..., "mean_top5": ..., "std_top5": ..., "mean_top10": ..., "std_top10": ..., "folds": [...]}`

- [ ] **A-impl-4: 实现三组信号特征提取**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - **词级 EEG**：已有 `extract_features(eeg_list, "mean_pool"/"band_separated")`，无需改动
  - **句级 EEG**：新函数 `extract_sent_features(sent_eeg_list)` → 逐样本 z-score，返回 `(N, 840)`
    - `f = (s - s.mean()) / max(s.std(), 1e-8)` for each `s` in sent_eeg_list
  - **高斯噪声**：新函数 `generate_noise_features(N, dim=840, base_seed=42)` → `np.random.default_rng(42+i).standard_normal(dim)` for each i

- [ ] **A-impl-5: 重写 A1a/A1b/A1c 入口，三组信号各跑 LOSO 5折**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - A1a：三组特征（词级mean_pool / 句级 / 噪声）各调用 `run_loso_linear_probe`
  - A1b：词级用 nfixations 加权（新函数 `extract_weighted_features(eeg_list, nfix_list)`）；句级和噪声标注 `"status": "fallback_to_a1a"`，直接复用 A1a 结果
  - A1c：词级用 band_separated；句级和噪声标注 `"note": "band_sep_equiv_to_a1a_for_sent_and_noise"`，复用 A1a 结果
  - 输出 JSON key 结构：`{"A1a": {"word_eeg": ..., "sent_eeg": ..., "noise": ...}, "A1b": {...}, "A1c": {...}}`

- [ ] **A-impl-6: 修复 A3-LP 的 per-subject 统计泄漏（改为 LOSO fold 内计算）**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - 当前实现：在完整 train 集计算全局 per-subject μ/σ → 有数据泄漏（test 被试的 train fold 数据混入）
  - 规范要求：在 LOSO 每一折的 train fold 内部，仅对 train fold 中的被试计算 μ/σ；test fold 被试使用各自在 train fold 中的统计量（或不归一化）
  - 修改 `run_desubject_analysis`：移除全局 train 集统计，改为在 LOSO split 内按折计算

- [ ] **A-impl-7: 更新 main() 流程和输出格式**
  - 文件：`benchmark_eval/scripts/validate_eeg_signal.py`
  - main() 中移除对 train_data 的依赖（仅保留 test_data）
  - A1a/A1b/A1c 各产出三组结果，写入 `linear_probe_results.json` 的对应 key
  - A2/A3 流程基本不变，但 A3 改为 LOSO 框架后结果结构更新

---

## 十一、建议执行顺序

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | C-1, C-2, H-2（参数统一 + BERTScore + 随机种子） | 已完成 |
| **Phase 2** | H-1, H-4, H-5（生成参数统一 + 异常处理 + 文档化） | 已完成 |
| **Phase 3** | A-1, A-2, M-1~M-6（接口重构 + 配置统一 + 代码清理） | 已完成（部分） |
| **Phase 4** | E-1, E-2, E-3（统计检验 + 消融实验 + 可视化） | 待实现 |
| **Phase 5** | O-1, O-2（异常现象定位与修复） | 待排查 |
| **Phase 6** | D-1~D-5（数据处理流程优化） | 已完成 |
| **Phase 7** | N-1~N-3（噪声测试实现） | 部分完成 |
| **Phase 8** | R-1~R-3（检索测试实现） | 已完成 |
| **Phase 9a** | A-impl-1~7（诊断线A代码对齐：nfixations字段 + LOSO重构 + 三组信号） | 待实现 |
| **Phase 9b** | DV-1a/b/c + DV-2 + DV-2-band（运行诊断线A，CPU ~10~30min） | 待 Phase 9a 完成后运行 |
| **Phase 9c** | NC-1~NC-2（噪声/Shuffle 支持开发，三层架构） | 已完成 |
| **Phase 9d** | NC-3~NC-5（运行实验 + 综合分析报告） | 脚本已创建，待运行 |

> **注意（数据重建）**：完成 A-impl-1 后，必须重新运行 `build_unified_dataset.py` 重新构建
> `unified_zuco.pkl`，才能使 `nfixations_word` 字段生效（A-impl-1b）。
> 旧 PKL 文件缺少该字段，A1b 会 fallback 到 A1a 结果。

---

*生成时间：2026-04-09，v2 更新：2026-04，v3 更新：2026-04-11（同步 contrast_experiment_spec v2 修订），v4 更新：2026-04-23（同步 experiment_A_details v3 三组信号并行 + LOSO 5折CV规范，新增第十节代码对齐任务）*
*审计范围：benchmark_eval 全部核心模块（18 个源文件 + 3 组评估输出）*
