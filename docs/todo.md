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

> 详见 `docs/contrast_experiment_spec.md`（v2 修订版，2026-04-11）

### 诊断线 A：原始数据集有效性验证（CPU，无需 GPU）

- [ ] **DV-1a: Linear Probe — Mean-Pool 基线**（词级 EEG 均值池化 `(840,)` → sklearn LogisticRegression 130 类分类）
- [ ] **DV-1b: Linear Probe — Duration-Weighted Pool**（v2 新增，按 fixation duration 加权的词级 EEG）
- [ ] **DV-1c: Linear Probe — Band-Separated**（v2 新增，保持 8 频带结构 `(8,105)` → flatten `(840,)`）
- [ ] **DV-2: 被试效应 vs 句子效应分析**（余弦相似度分组对比 + η² 方差分解 + t-SNE 多 perplexity 可视化，v2 修正分组策略）
- [ ] **DV-2-band: 频带级 η² 分析**（v2 新增，对 8 个频带分别计算 η²_sentence vs η²_subject，输出效应量对比图）
- [ ] **DV-3: 去被试化信号恢复验证**（被试内 z-score + 被试聚合检索，v2 修正数据泄漏：仅在 train 集计算 μ/σ）

**实施**：已创建 `benchmark_eval/scripts/validate_eeg_signal.py`，输出到 `benchmark_eval/test_outputs/dataset_validity/`

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

## 九、建议执行顺序

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
| **Phase 9a** | DV-1a/b/c + DV-2 + DV-2-band（数据有效性验证，CPU ~10min） | 脚本已创建 |
| **Phase 9b** | NC-1~NC-2（噪声/Shuffle 支持开发，三层架构） | 已完成 |
| **Phase 9c** | NC-3~NC-5（运行实验 + 综合分析报告） | 脚本已创建，待运行 |

> **注意**：代码修改完成后，需要重新运行 `build_unified_dataset.py` 重新构建 `unified_zuco.pkl`，
> 才能使 D-1（spectrogram 格式）和 D-2（字段重命名）生效。
> 旧 PKL 文件通过向后兼容字段名仍可正常加载，但不包含 `eeg_spectro` 字段（EEG2Text 无法正常评估）。

---

*生成时间：2026-04-09，v2 更新：2026-04，v3 更新：2026-04-11（同步 contrast_experiment_spec v2 修订）*
*审计范围：benchmark_eval 全部核心模块（18 个源文件 + 3 组评估输出）*
