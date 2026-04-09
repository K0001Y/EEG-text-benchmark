## EEG 数据预处理流程对比（EEG-To-Text / DeWave / EEG2Text / GLIM / CET-MAE / Unified Benchmark）

本文件对比了几个代表性 EEG-to-Text 模型在 **数据预处理部分** 的方法，并与当前 Benchmark 中的 **统一数据处理流程** 进行分步骤对照，聚焦 ZuCo 数据从 `.mat` 到模型输入的关键差异。

### 一、总体数据流对比

| **步骤** | **EEG-To-Text** | **DeWave** | **EEG2Text** | **GLIM** | **CET-MAE** | **统一 Benchmark 流程** |
| --- | --- | --- | --- | --- | --- |
| **1. 原始数据起点** | ZuCo v1/v2 原始 `.mat`（`sentenceData`），先用 `construct_dataset_mat_to_pickle_v1/v2` 转成 per-subject `dataset_dict` pickle | 基于 EEG-To-Text 代码，沿用其 ZuCo `.mat` → `dataset_dict` 的预处理；之后接离散编码模块 | ZuCo `.mat`，用 `construct_dataset_mat_to_pickle_v1_spectro` / `_v2_spectro` 直接抽取句级 `rawData`，保存为 spectro pickle | 直接使用预先构建好的 ZuCo 统一 DataFrame（包含 `eeg` / `mask` / `text uid` / `input text` 等列） | ZuCo v1/v2 原始 `.mat`，用 `data2pickle_v2.py` 解析 `sentenceData`，提取词级 + 句级 EEG，保存为 per-sample pickle | 直接从 ZuCo 原始 `.mat`（v1 + v2）读取，`build_unified_dataset_eegtotext.py` 内部解析为 `dataset_dict`，不再依赖各模型自己的 pickle 作为起点 |
| **2. 句/词结构抽取** | `construct_dataset_mat_to_pickle_*` 中，从 `sentenceData` 提取 `content`、`sentence_level_EEG`（8 频段均值）和 `word` 列表 + `word_level_EEG`（GD/FFD/TRT） | 同 EEG-To-Text：沿用 `sentence_level_EEG` + `word_level_EEG` 的结构，作为后续离散编码的输入 | 只保留 `content` 与 `sentence_level_EEG['rawData']`（形状约为 `105 × T`），不再使用词级 EEG；`word` 主要用于过滤坏句子 | 统一 DataFrame 中已经有 `input text`、`task`、`dataset`、`subject`、`text uid` 等文本与元信息；不在 GLIM 内重新解析 `.mat` | 在 `data2pickle_v2.py` 中从 `sentenceData` 提取：`content`、句级 `mean_*`（8 频段）、词级 `word_level_EEG`（GD/FFD/TRT 各 8 频段）；**独特之处**：将句级 EEG 向量拼接到词级序列末尾，形成 "词序列 + 句向量" 结构 | 在 `load_dataset_from_mat_v1/v2` 中统一抽取 `content`、句级 `mean_*`、词级 `word_level_EEG`，得到结构兼容 EEG-To-Text 的 `dataset_dict`，为后续统一编码做准备 |
| **3. EEG 表示形态** | 训练时主要使用 **词级 EEG 序列**：对每个词按 8 频段拼接成 1D 向量（维度 `105×8`），再组成长度 `max_len` 的序列；另有一句级向量作为辅助特征 | 在 EEG-To-Text 的词级表示基础上，加 VQ-VAE 等离散编码，将连续的 EEG 序列映射为离散码序列，再送入文本解码器 | 使用 **句级 raw EEG 序列**：`rawData`（`105 × T`），按时间维 padding 到 24000 点，得到 `(105 × 24000)` 的连续信号（在 `data_raw.py` 中进一步归一化） | 使用 **统一句级 EEG 序列**：`eeg` 字段为 `(L_max, C)`，`mask` 为 `(L_max,)`；在 DataFrame 中已经完成长度对齐与归一化 | 使用 **词级 + 句级拼接序列**：每个词的 EEG 向量为 `105×8=840` 维；将所有词向量拼接后，**追加句级向量作为最后一个 token**，形成 `(n_words+1, 840)` 序列；这是 CET-MAE 的独特设计，用于 MAE 重建与对比学习 | 使用统一的 **句级“伪词序列” EEG**：将词级 + 句级向量堆叠成 `(seq_len, C)`，再 2D z-score，最后时间维截断/补零到固定 `L_max`，得到 `(L_max, C)` + `mask` 序列 |
| **4. 时间长度对齐** | 以 **词数** 为时间维：`get_input_sample` 中若词数 < `max_len` 就补零，> `max_len` 则截断；句级向量本身是固定维度，无时间轴 | 同 EEG-To-Text：离散编码前的 EEG 序列仍以词为时间步，`max_len` 固定，通过 padding/truncation 对齐 | 在构建 spectro pickle 时限制 `rawData.shape[1] ≤ 23631`；`data_raw.py` 中 pad 到 `max_spectro_datapoint = 24000`，并构造长度为 24000 的 `sent_mask` | DataFrame 中的 `eeg` 已经统一到某个 `L_max`（如 24000 或更小），`mask` 标记有效时间步；GLIM 不再自行对齐时间长度 | 以 **词数+1** 为时间维：`max_len = 58`；若 `(n_words+1) < max_len` 则补 840 维零向量；构造 `input_attn_mask`（1 为有效，0 为 padding）；**注意**：句级向量始终在序列末尾，不参与词数计数 | 在 `build_samples_for_task` 中：先按词级 + 句级得到 `seq_len`，然后 **截断到 `max_len`**，再补 0 到 `max_len`，同时构建同长的 `mask`；属于对各模型统一的时间轴标准 |
| **5. 归一化策略** | 每个词向量与句级向量在构建时做 **1D z-score**；词级序列整体不再做额外 2D 归一化，直接送入模型（`data.py` 中） | 在 EEG-To-Text 的 1D 归一化基础上，离散编码模块内部可能再做归一化/量化（依赖 DeWave 具体实现） | `data_raw.py` 中对整条 `(105 × 24000)` 的句级 Tensor 做 **1D z-score**（flatten 后统计均值/方差），归一化后作为 `sent_level_EEG` | GLIM 直接使用 DataFrame 中预处理好的 `eeg`；在 `ZuCoDataset` 中不会再对 EEG 做归一化，只在 `eval_noise_input` 时用高斯噪声替换 | **双重归一化策略**：1) 每个词/句级向量单独做 **1D z-score**（`normalize_1d`）；2) 将词+句向量堆叠后，整体再做一次 **2D z-score**（`normalize_2d`），保存两份副本 `input_embeddings` 和 `normalized_input_embeddings` | 在 `build_unified_dataset_eegtotext.py` 中：先对每个词/句级 1D 归一化，然后把它们堆叠成矩阵，整体再做一次 **2D z-score**，确保不同长度句子在整体尺度上可比 |
| **6. 训练/验证/测试划分** | 在 `ZuCo_dataset` 构造时，对每个 subject 按句子顺序做 80%/10%/10% 划分（train/dev/test），不同任务独立划分 | 通常沿用 EEG-To-Text 的划分策略（80/10/10），以便横向对比；具体实现依赖 DeWave 训练脚本 | 在 `ZuCo_dataset` 中同样以 80%/10%/10%（按句子索引）划分 train/dev/test，按 subject 独立处理 | 在 `GLIMDataModule` 中依据 DataFrame 的 `phase` 字段（train/val/test）来划分；具体 `phase` 字段由上游脚本在构建 DataFrame 时写死 | 在 `data2pickle_v2.py` 中 **按 subject 做 80%/10%/10% 划分**，每个样本独立保存为 `{version}-{task}-{subject}-{idx}.pickle` 到 train/valid/test 目录；训练时用 `ConcatDataset` 合并 train + valid | 在 `build_unified_dataset_eegtotext.py` 中 **统一在每个 subject 内做 8:1:1 划分**，将 `phase` 写进样本字典；所有下游模型共享这一个 split（评估时通过 `UnifiedDataset(phase=...)` 过滤） |
| **7. 中间存储格式** | 每个任务保存一个 per-subject 的 `dataset_dict` pickle：`{subject: [sent_obj or None]}`，结构紧耦合 EEG-To-Text 自己的 `data.py` | 同 EEG-To-Text：复用其 `dataset_dict` pickle 作为上游输入，之后再额外保存离散向量或 codebook | 以 spectro 为主的 per-subject pickle：`{subject: [sent_obj]}`，只有 `content` + `sentence_level_EEG['rawData']` | 统一单个 DataFrame（`.pkl`）：每行一个 EEG-text 样本，字段包括 `eeg`、`mask`、`prompt`、`text uid`、`input text`、`target text`、`sentiment label`、`relation label` 等 | **Per-sample pickle**：每个样本独立保存为一个 pickle 文件，按 train/valid/test 目录组织；文件名格式 `{version}-{task}-{subject}-{idx}.pickle`；字段包括 `input_embeddings`、`normalized_input_embeddings`、`input_attn_mask`、`target_ids`、`target_tokenized`、`selected_words` 等 | 单一 pickle：包含 `List[Dict]`，每个字典字段为 `eeg`、`mask`、`input_text`、`reference_text`、`phase`、`meta`；是 Benchmark 层面的 **唯一官方数据入口** |
| **8. 送入模型前的再处理** | `ZuCo_dataset` 中进一步构造：`input_embeddings`（词级 EEG 序列）、`input_attn_mask`、`target_ids`、`target_mask` 等，形成 EEG-to-Text 解码器输入 | 在 EEG-To-Text 的 `input_embeddings` 基础上插入离散编码模块，将连续 EEG 序列压缩成离散 token，再接 LLM 解码器 | `ZuCo_dataset` 中从 spectro pickle 构造 `sent_level_EEG`（归一化的 105×24000）、`sent_mask`、`target_ids` 等，直接作为解码器输入 | `ZuCoDataset` 中直接从 DataFrame 取出 `eeg`/`mask`/`prompt`/`input text`/`target text` 等，配合自定义 `GLIMSampler` 按 `text uid` 采样 batch；不关心原始 `.mat` 细节 | 在 `EEG_dataset_add_sentence_mae` 中从 pickle 加载样本，返回 `(input_embeddings, normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, target_tokenized, text)`；用于 MAE 预训练 + BART 文本生成 | 统一通过 `benchmark_eval/UnifiedDataset` → `eval_runner` 给各模型：batch 内每条样本传入 `(eeg, mask, meta)`；具体模型的 wrapper 在内部再映射到各自原始输入形式（词级/句级/频谱/DataFrame 等） |

### 二、关键差异小结（面向 Benchmark 的意义）

- **统一起点与统一中间格式**：
  - 既有模型（EEG-To-Text、DeWave、EEG2Text）都从 `.mat` 出发，但各自定义了不同的中间 pickle 结构，导致原始 work 之间难以直接复用数据与划分；
  - **GLIM** 和 **当前 Benchmark 流程** 都引入了 DataFrame/统一样本字典的中间层，将 `eeg`、`mask` 与文本元信息统一封装；
  - **CET-MAE** 采用 per-sample pickle 存储策略，每个样本独立保存，按 train/valid/test 目录组织，粒度更细但数量更多；
  - Benchmark 现在选择：**从 ZuCo `.mat` 直接构建统一的 `(L_max, C)+mask` 表示和统一划分**，避免依赖某一个模型的特定 `*-dataset.pickle`。

- **时间轴与归一化的一致性**：
  - EEG-To-Text / DeWave 以词为时间步，EEG2Text 以 **采样点** 为时间步，GLIM/Benchmark 则以固定长度的 EEG 序列 `(L_max, C)` 为时间步；
  - **CET-MAE** 以 **词数+1** 为时间步（句级向量作为最后一个 token），维度固定为 840，采用独特的 **双重归一化策略**（1D + 2D z-score）；
  - Benchmark 在构建统一数据时对 **词级 + 句级特征** 做了 1D + 2D 归一化，并统一到固定 `L_max`，为不同模型适配提供了一个公共“时间网格”。

- **EEG 表示形态的独特性**：
  - **CET-MAE** 是唯一将句级 EEG 向量显式拼接到词级序列末尾的模型，形成 `(n_words+1, 840)` 结构；这种设计便于 MAE 预训练时对整体序列进行掩码重建，同时保留句子级全局信息用于对比学习。

- **划分策略与可复现性**：
  - EEG-To-Text / EEG2Text 的 80/10/10 划分在各自代码内实现，split 细节（按句子索引、按被试等）存在差别，难以完全对齐；
  - GLIM 通过 DataFrame 的 `phase` 字段固定划分；
  - **CET-MAE** 在预处理脚本 `data2pickle_v2.py` 中完成划分，将样本直接写入对应目录，训练时合并 train+valid；
  - Benchmark 现在把 8:1:1 划分 **写进统一样本** 的 `phase` 字段，并在所有评估脚本和模型 wrapper 中强制使用同一份划分，从而保证各模型在相同样本集上进行对比。
  
  ---
  
  ### 三、统一 Benchmark v2 字段更新（2026-04-10）
  
  本次优化对统一数据集字段命名进行了规范化，并将 EEG2Text 的存储格式从原始时序改为 spectrogram。
  
  #### 3.1 字段命名变更（v1 → v2）
  
  | 旧字段名（v1） | 新字段名（v2） | Shape | 变化说明 |
  |-------------|-------------|-------|---------|
  | `eeg` / `eeg_normalized_1d` | `eeg_word_norm1d` | (max_len, 840) | 仅重命名；保留别名 `eeg` 向后兼容 |
  | `eeg_raw` | `eeg_word_raw` | (max_len, 840) | 仅重命名 |
  | `eeg_normalized_2d` | `eeg_word_norm2d` | (max_len, 840) | 仅重命名 |
  | `eeg_eeg2text` | `eeg_spectro` | **(374, 65)** | **格式变更**：原始时序 (24000, 105) → scipy spectrogram |
  | `mask` | `mask_word` | (max_len,) | 仅重命名；保留别名 `mask` 向后兼容 |
  | `mask_with_sent` | `mask_word_with_sent` | (max_len,) | 仅重命名 |
  | `mask_eeg2text` | `mask_spectro` | **(374,)** | 长度从 24000 改为 374 |
  
  命名规范遵循 `eeg_{表征类型}_{处理方式}` 模式：
  - 表征类型：`word`（词级）、`spectro`（频谱）
  - 处理方式：`raw`（未归一化）、`norm1d`（逐词1D归一化）、`norm2d`（全局2D归一化）
  
  #### 3.2 EEG2Text 数据格式变更说明
  
  **原存储方式（v1）**：直接存储 `rawData` 原始时序 `(24000, 105)`，每样本约 9.6 MB float32。
  
  **新存储方式（v2）**：在 `build_unified_dataset.py` 中预计算 spectrogram：
  - 使用 `scipy.signal.spectrogram(signal, fs=500, nperseg=128, noverlap=64)` 逐通道计算
  - 对所有通道的频谱取均值，得到 `(374, 65)` 格式
  - 每样本约 97 KB，**节省 99% 存储空间**
  - 与 EEG2Text 原始 `data_spectro.py` 的预处理逻辑完全对齐
  
  #### 3.3 向后兼容策略
  
  旧版 PKL 文件（v1 字段名）通过 `dataset.py` 中的 `_get_field()` 辅助函数自动向后兼容：
  
  ```python
  def _get_field(item, new_key, *fallback_keys):
      if new_key in item: return item[new_key]
      for key in fallback_keys:
          if key in item: return item[key]
      return None
  ```
  
  各 wrapper 同样使用 `batch.get(new_key, batch.get(old_key))` 模式，确保新旧 PKL 文件均可使用。
  
  #### 3.4 统一 Benchmark v2 流程总览
  
  | 步骤 | 说明 |
  |------|------|
  | 原始数据起点 | ZuCo v1/v2 `.mat` 文件（不变） |
  | 构建工具 | `build_unified_dataset.py`（新增 `build_spectrogram()` 函数） |
  | 中间存储 | 单一 `unified_zuco.pkl`，包含 v2 字段名的 `List[Dict]` |
  | max_len | **56**（统一基准，与 EEG-To-Text 原始训练一致） |
  | 分割验证 | 三个 phase 均非空，否则抛出明确异常 |
  | 随机种子 | `numpy.seed + random.seed` 双重固定 |
  | 模型访问 | 通过 `UnifiedDataset(phase=...)` 加载，统一使用 v2 字段名 |