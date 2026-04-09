## EEG-to-Text 模型方法横向对比

本文件基于当前工作目录中的四个仓库 README：DeWave、EEG-To-Text（“Are EEG-to-Text Models Working?” 主分支）、EEG2Text、GLIM，整理它们在问题设定、EEG 表示方式、核心建模思路、与语言模型的结合方式以及评估观点等方面的横向对比。内容严格以各仓库 README 所公开的信息为基础，不对未明确给出的技术细节进行额外推断。

### 一、各方法简要概述

DeWave 的核心思想是在 EEG-to-Text 基线框架上，引入类似 VQ-VAE 的离散编码机制，将原本连续的 EEG 波形通过 codebook 量化为离散 token 序列，再作为条件输入驱动文本生成。README 强调这是一个“离散编码 EEG 波形 → 文本翻译”的框架，并展示了在特定 codebook 大小（如 2048）和潜在维度（如 512）下在 ZuCo task 2.0 上取得的 BLEU 与 ROUGE 指标，同时给出了不同被试上的性能雷达图。整体来说，DeWave 通过在 encoder 端增加“离散瓶颈”，期望获得更结构化、更稳定的 EEG 表示，再与下游语言解码器结合完成文本生成。

当前 EEG-To-Text 仓库的主分支并非专注于提出新结构，而是为论文 “Are EEG-to-Text Models Working?” 提供代码，并对早期 AAAI 2022 工作“Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification”的评估进行修正。README 指出，原始代码在生成预测字符串时存在隐式 teacher forcing 问题：直接在整段 logits 上取 argmax，再整体 decode，而非通过自回归的 model.generate。为此，该仓库修改了 model_decoding.py 与 eval_decoding.py，改为使用 model.generate 进行真正的自回归生成，在这种更加真实的评估设置下，模型表现明显下降。该仓库因此更多扮演“经过评估修正的基线实现”的角色，其方法意义在于：用严格的生成过程重新审视 EEG-to-Text 模型是否真的学习到了 EEG 与文本之间的有意义映射。

EEG2Text 仓库对应 IEEE BigData 2024 的工作，其 README 虽然简洁，但清楚表明其目标是在 ZuCo v1.0 与 v2.0 多任务场景下，对不同 EEG 表示形式与掩码预训练策略进行系统比较。其流程首先复用 EEG-To-Text 的数据预处理思路，将原始 ZuCo Matlab 文件转换为统一的 pickle 数据集（命名中常带 spectro 后缀，用以区分与原始 EEG-To-Text 生成的数据版本），然后通过一系列 get_dataset_xxx.py 脚本构建多种变体数据集，例如仅使用句子级数据的 raw 格式、使用谱图形式的 spectro 格式、以及带有 token 级或 span 级遮蔽的 masked_raw 和 span_masked_raw_all 等。随后再通过对应的 pretrain_xxx.sh 脚本进行预训练。结合脚本命名与数据管线可以看出，该工作核心是以类似 RoBERTa 的掩码建模思想，对 EEG 序列进行自监督预训练，并比较不同表示和掩码策略在下游 EEG-to-Text 任务上的效果。

GLIM 仓库对应论文“Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation”，其 README 给出了较完整的动机与方法论。作者从“预训练生成模型是否真正反映大脑语义，还是仅仅在幻觉”这一问题出发，聚焦 EEG-to-Text 任务，从 posterior collapse（后验坍缩）的视角来审视 EEG 与文本信息容量不匹配的问题。GLIM 认为，将任务简单视作对刺激文本的逐字重构容易导致解码器几乎不依赖 EEG 表示，而纯粹依赖语言模型先验。为此，该工作将 EEG-to-Text 重新表述为“语义摘要”任务，即要求生成文本抓住核心语义而不是逐字对齐原文，并提出 Generative Language Inspection Model（GLIM）这一架构，通过强调 EEG latent 表示的信息性与可解释性，在异质且小规模的数据条件下提升语义对齐能力。README 还特别强调模型在生成时不使用 teacher forcing，并引入 EEG-文本检索、零样本语义分类等多种评估指标，以验证生成是否真正受到 EEG 表示约束。

### 二、关键维度的横向对比

下表在若干关键维度上对这四个仓库代表的方法进行对比：

| 维度 | EEG-To-Text（主分支 / “Are EEG-to-Text Models Working?”） | DeWave | EEG2Text | GLIM |
|------|-------------------------------------------------------|--------|---------|------|
| 主要问题设定 | 在严格的无 teacher forcing 自回归生成设置下重新评估 EEG-to-Text 模型，回答“这些模型是否真的在工作？” | 在 EEG-to-Text 基线之上引入离散编码（类似 VQ-VAE）将 EEG 波形量化为 codebook token，再进行文本翻译 | 在 ZuCo 多任务设置下系统比较不同 EEG 表示与掩码预训练策略对 EEG-to-Text 性能的影响 | 将 EEG-to-Text 重新表述为语义摘要任务，通过可解释表示与改进架构提升生成的语义忠实度并缓解幻觉 |
| EEG 表示方式 | 继承原始 EEG-To-Text 的特征设计，重点在评估方式而非新特征（具体特征在论文与代码中） | 使用 VQ-VAE 式离散瓶颈，将连续 EEG 波形编码为离散 codebook 索引序列 | 针对多种数据形式构建数据集变体，如 raw（句子级数据）、spectro（谱图形式）、masked_raw、span_masked_raw_all 等 | 在 ZuCo 数据上构建强调信息性与可解释性的 EEG latent 表示，具体结构在 method 图中给出 |
| 核心建模思路 | 保持原有 EEG-to-Text 架构，关键在于改用 model.generate 进行真正的自回归生成，从而获得更真实的模型性能估计 | 在 encoder 端增设离散化层，以离散 EEG token 作为语言解码器的输入，期望增强表示能力并提升翻译质量 | 使用类似 RoBERTa 的自监督掩码建模对 EEG 序列预训练，通过比较不同数据格式与掩码策略，寻找更适合下游 EEG-to-Text 的表示 | 提出 GLIM 架构，从 posterior collapse 视角设计模型和训练策略，使 EEG latent 表示既可解释又能驱动语义忠实的文本生成 |
| 与语言模型/decoder 的结合 | 采用 seq2seq 语言模型作为解码器，强调在评估中必须使用自回归的生成接口而非 implicit teacher forcing | 沿用 EEG-To-Text 中的文本解码器框架，只是将其条件输入改为离散化后的 EEG token 序列 | 借助掩码预训练得到更好的 EEG encoder，再与文本生成模块结合，分析不同预训练配置对解码器效果的影响 | 使用生成式语言模型作为输出层，配合专门设计的 EEG encoder，使生成文本与 EEG latent 中的语义信息紧密对齐 |
| 任务目标与评估 | 目标是澄清模型真实能力，指出此前由于 teacher forcing 导致的性能虚高，并为后续工作提供更可靠的基线实现 | 目标是通过离散 EEG 表示在 ZuCo task 2.0 上取得更好 BLEU/ROUGE 表现，并探索离散编码对被试间差异的影响 | 目标是在同一任务框架下，通过系统实验验证不同 EEG 表示和自监督预训练策略在 EEG-to-Text 上的优劣 | 目标是获得语义更加忠实、更加 EEG-grounded 的生成，评估不仅包含文本相似度，还包括 EEG-文本检索与零样本语义分类等 |
| 与原始 EEG-To-Text 的关系 | 直接基于原始代码，对评估流程进行修正，并围绕“模型是否工作”这一问题构建新的实验 | 明确声明基于 EEG-To-Text 代码实现，在其基础上加入 VQ-VAE 式离散编码模块 | 前三步数据预处理与 EEG-To-Text 基本一致，然后扩展出多种数据格式与预训练脚本，形成一个系统化方法比较平台 | README 中直接引用“Are EEG-to-text models working?”，并在此基础上提出 GLIM 作为一个更系统、更积极的回答框架 |

从表格可以看到，这四个方法并非简单的互斥竞争关系，而是围绕 EEG-to-Text 问题形成了相对连贯的演进脉络。EEG-To-Text（修正版）首先为社区确立了更加可靠的评估方式，把“模型是否真的在工作”这一关键问题提到台前。DeWave 在此基础上尝试通过离散编码增强 EEG 表示能力，以期在传统 BLEU/ROUGE 指标上取得更强表现。EEG2Text 则从 EEG 表示与自监督预训练的角度出发，系统比较不同数据格式和掩码策略为 EEG-to-Text 带来的影响，为后续模型设计提供经验基础。GLIM 则进一步把问题上升到“语义摘要与可解释表示”的层面，通过重新定义任务目标、引入更全面的语义对齐评估，为 EEG-to-Text 的可靠应用提出一套更完整的方法论框架。

### 三、小结

综合来看，当前这几个仓库代表的工作，从不同侧面回答了“EEG-to-Text 模型是否可靠、如何更好地建模 EEG 表示以及如何保证生成语义忠实”这一系列问题。EEG-To-Text 主分支通过评估修正提醒我们必须谨慎对待 teacher forcing 带来的虚高指标；DeWave 探索了离散 EEG 表示在文本生成中的潜力；EEG2Text 系统性地研究了多种 EEG 表示与掩码预训练策略；GLIM 则从语义忠实与可解释性的角度重新设计任务与模型。对于后续研究而言，可以在更加严格的评估前提下，一边借鉴 EEG2Text 对表示与预训练的经验，一边结合 DeWave 的离散编码思想和 GLIM 的语义摘要框架，共同推进 EEG-to-Text 任务在可靠性和可解释性上的发展。

### 四、数据处理流程对比（EEG-To-Text / EEG2Text / GLIM）

这一节专门从“数据是如何一步步变成模型输入”的角度，对 EEG-To-Text、EEG2Text 和 GLIM 三个仓库的数据处理管线做分步骤说明，便于你后续在自己的实验（例如 pixel/pixl 输入）中对齐或替换数据流。

#### 4.1 EEG-To-Text：从 ZuCo .mat 到按词对齐的 EEG 序列

1. 准备原始 ZuCo 数据
   - 下载 ZuCo v1.0 / v2.0 的 Matlab 文件，按官方目录放到 `./dataset/ZuCo/.../Matlab_files` 中。

2. 使用 `construct_dataset_mat_to_pickle_v1.py` / `construct_dataset_mat_to_pickle_v2.py` 生成 pickle
   - 逐个 `.mat` 文件读取 `sentenceData` 结构；
   - 对每个句子构建一个 `sent_obj`，包含：
     - 文本字段 `content`（完整英文句子）；
     - 句级 EEG 字典 `sentence_level_EEG`，保存 8 个频带统计量（mean_t1/t2, mean_a1/a2, mean_b1/b2, mean_g1/g2）；
     - 词级列表 `word`，每个元素是一个 `word_obj`，包含该词内容、注视次数 `nFixations`，以及 FFD/GD/TRT 三种注视类型在 8 通道上的 EEG；
     - 三套 token 序列：所有词、仅有注视的词、无注视词被 `[MASK]` 替代的序列。
   - 把每个被试的所有 `sent_obj` 收集成 `dataset_dict[subject]`，最终序列化为 `taskX-dataset.pickle`。

3. 在 `data.py` 中用 `ZuCo_dataset` 构造训练/验证/测试集
   - 读取一个或多个 `taskX-dataset.pickle`，按被试划分；
   - 对每个被试内部按句子顺序做 8:1:1 划分为 train / dev / test（不打乱）；
   - 对每条 `sent_obj` 调 `get_input_sample`：
     - 用 tokenizer 编码目标句子，得到 `target_ids` 和 `target_mask`；
     - 从 `sentence_level_EEG` 取出选定频带特征拼接成 1D 向量并标准化，作为句级 EEG；
     - 遍历 `word` 列表，对每个词取对应 EEG 类型（如 GD）和 8 通道特征拼接成 1D 向量，组成“时间步 = 单词”的 EEG 序列；
     - 将该序列补零或截断到固定长度 `max_len`，并构造 `input_attn_mask` / `input_attn_mask_invert` 与 `seq_len`。

4. 训练与评估阶段的数据使用方式
   - 训练脚本（如 `train_decoding.py`）通过 `DataLoader(ZuCo_dataset)` 迭代 `(EEG 序列张量, seq_len, mask, target_ids, target_mask, ...)`；
   - 评估脚本（`eval_decoding.py`）在严格禁用 teacher forcing 的前提下，用 `model.generate` 进行自回归生成，对比预测文本与 `target_ids` 对应的真实句子。

#### 4.2 EEG2Text：统一句级 raw EEG，多视图派生（raw / 频谱 / 掩码）

1. 从 ZuCo .mat 提取句级 raw EEG
   - 使用 `util/construct_dataset_mat_to_pickle_v1_spectro.py` / `construct_dataset_mat_to_pickle_v2_spectro.py`：
     - 逐句读取 `sentenceData.rawData`，得到形状约为 `105 × T` 的时间序列；
     - 过滤掉缺失或极端过长的句子（例如 `T > 23631`）并记为 `None`；
     - 为每个句子构造仅包含 `content` 和 `sentence_level_EEG['rawData']` 的 `sent_obj`；
     - 汇总为 `taskX-dataset-spectro.pickle`，这是 EEG2Text 的统一底层数据表示。

2. 视图一：raw 序列输入（`data_raw.py`）
   - `ZuCo_dataset` 从 spectro pickle 读出 `rawData`：
     - 在时间维上补零到统一长度 24000，得到 `105 × 24000` 的矩阵；
     - 对每个时间点按通道标准化，形成 `sent_level_EEG`；
     - 构造 `sent_mask`（前 T 个为 1，之后补零为 0）与其反掩码；
     - 目标文本仍然用 BART tokenizer 得到 `target_ids`，`seq_len` 为句子单词数 + 1。

3. 视图二：时间–频率谱图输入（`data_spectro.py`）
   - 使用句级 rawData 的**通道平均**作为 1D 信号：
     - 对 `105 × T` 沿通道求平均得到 `v`，补零到统一长度；
     - 使用 `scipy.signal.spectrogram` 计算短时谱，得到 `time × freq` 的谱图（代码中固定检查为 `374 × 65`）；
     - 标准化后作为 `sent_level_EEG`，并根据原始时长推算有效时间步数，构造 `spectro_mask` / `spectro_mask_invert`。

4. 视图三：点级 BERT-MLM 掩码 raw（`data_masked_raw_all.py`）
   - 将 `rawData` 转置为 `T × 105`，把每个时间步视为一个 token：
     - 调用 `bert_mlm_mask`：随机选取约 15% 的时间步索引；
     - 其中 80% 的位置替换为全 0 行，10% 替换为随机噪声行，10% 保留原值；
     - 在时间维上补零到 24000，产生 `masked_EEG`（`24000 × 105`），同时产生长度 24000 的 `mask_indices`（真实 mask 位置，其余为 -1）；
     - 保留归一化后的完整 `sent_level_EEG` 以及 `sent_mask` / `sent_mask_invert`，作为自监督任务的“标签”和有效区域标记。

5. 视图四：span 级 BERT-MLM 掩码 raw（`data_span_masked_raw_all.py`）
   - 同样使用 `T × 105` 的时间步表示，但掩码单位改为连续跨度：
     - 按几何分布采样“单词长度”，再乘以固定系数近似换算成时间步跨度；
     - 在不与已有 span 重叠的条件下，不断采样 span，直到覆盖时间轴约 15%；
     - 对 span 内的时间步应用同样的 80/10/10 掩码策略，生成 `masked_EEG` 与 `mask_indices`；
     - 其余流程（`sent_level_EEG`、`sent_mask` 等）与点级 MLM 版本一致，用于对比不同掩码策略的效果。

6. 视图五：RoBERTa 风格 raw 序列（`data_masked_raw_robert.py`）
   - 对 `rawData` 做统一 padding 和归一化，得到 `24000 × 105` 的 `sent_level_EEG`；
   - 记录真实有效时间步 `sent_eeg_len`，生成 `sent_mask` / `sent_mask_invert`；
   - 不在这里显式生成 `masked_EEG`，而是为 RoBERTa 类自监督预训练提供“干净的连续输入序列”。

7. 通过 `get_dataset_pickle_*.py` 打包 train/dev/test 集
   - raw / spectro / masked / span-masked / robert 等不同视图都用对应的 `get_dataset_pickle_*.py`：
     - 调用 `get_config('train_decoding')` 读入任务组合（task1/task2/task3/taskNRv2）、被试选择、GPU 设备等；
     - 按任务名加载一个或多个 `taskX-dataset-spectro.pickle`；
     - 构造对应视图的 `ZuCo_dataset`，按 8:1:1 划分 train/dev/test；
     - 将 Dataset 本身 pickle 到 `train_set_xxx.pkl` 等文件，供后续训练脚本直接加载使用。

#### 4.3 GLIM：基于 DataFrame 的统一表格输入 + 多文本变体

1. 离线整理 ZuCo + 文本变体为单一 DataFrame
   - 通过 `data/__STEP*.ipynb` 等 notebook 把 ZuCo1/2 的 EEG 与文本信息清洗、对齐：
     - 每行是一条 EEG–文本样本，包括 `eeg`（`L × C` 数组）、`mask`（长度 L 的 0/1）、`input text`、任务 ID、数据集标记、被试 ID、情感/关系标签、唯一 `text uid` 等；
   - 用 `_gen_variants_llm_general.py` / `_gen_variants_llm_regular.py` 调用大语言模型，为每条 `input text` 生成多种“语法简化 / 词汇简化 / 语义澄清 / naive 重写”等目标文本变体，并写入多列（例如 `lexical simplification (v0/v1)` 等）。

2. 在 `ZuCoDataset` 中按 phase 切分并组织目标文本
   - `__init__` 时按 `phase` 过滤 DataFrame：
     - train 阶段：
       - 指定一个 `pt_target_keys` 列表，包含八种不同类型的 target 文本列；
       - 对每个 target 列单独抽取一份数据字典（含 EEG、mask、input text、target text、标签等），然后用 `collate_fn` 拼接，等价于：同一 EEG–原文对可以对应多种目标句作为训练信号；
       - 把所有目标句组合进 `all target texts`（一个 tuple），作为后续评估和多目标解码使用。
     - val/test 阶段：
       - 只从 DataFrame 中取 `input text` 和所有 `pt_target_keys` 对应列，构造 `all target texts`，用于在固定 EEG 输入下对多种目标句进行生成和检索评估。

3. 构造文本输入模板与多维 prompt
   - 在 `__fetch_from_df` 内部：
     - 用固定模板（如 `"To English: <MASK>"`）包裹 `input text`，形成统一的输入格式；
     - 同时为每条样本生成 `prompt`：`(task_prompt, dataset_prompt, subject_prompt)`，例如 `<NR>` 或 `<TSR>` 区分任务，数据集名区分 ZuCo1/2，被试 ID 形成 subject prompt；
     - 保留 `sentiment label` 和 `relation label` 字段，为 GLIM 在评估阶段的情感/关系分类提供标签信号。

4. 支持“噪声 EEG”评估模式
   - 如果 `eval_noise_input=True`：
     - 在构造 Dataset 时，使用与原始 `eeg` 同形状的高斯噪声替换掉所有 EEG；
     - 将 `mask` 改为全 1，表示所有时间步可用；
     - 这样即可用“完全噪声的 EEG”输入测试模型生成表现，从而检验解码器是不是严重依赖语言模型先验而忽视脑信号。

5. 使用 DataModule 与自定义 Sampler 组织批次
   - `GLIMDataModule`：
     - `setup("fit")` 时构造 train_set / val_set；`setup("test")` 时构造 test_set，并记录目标文本数量；
     - `train_dataloader` / `val_dataloader` / `test_dataloader` 分别用 `GLIMSampler` 构造 batch。
   - `GLIMSampler`：
     - 继承 `DistributedSampler`，但在采样时按 `text uid` 保证同一个 batch 内的样本对应不同的文本；
     - 支持多卡训练时自动估算每卡的 batch 数、补齐不足部分，保持各 GPU 上 batch 数一致，方便做 EEG–文本对比学习和检索任务。

---

### 五、各模型在统一 Benchmark 中的 EEG 输入对应关系（v2）

本节说明各模型 wrapper 从统一数据集 v2 字段中读取的具体字段，以及必要的格式适配。

| 模型 | 读取字段（v2） | 实际传入模型的 shape | 适配方式 |
|------|-------------|---------------------|---------|
| **EEG-To-Text** | `eeg_word_norm1d`（fallback: `eeg`）<br>`mask_word`（fallback: `mask`） | (B, 56, 840) | 直接使用；`max_new_tokens=56, num_beams=5, do_sample=True, rep_penalty=5.0` |
| **EEG2Text** | `eeg_spectro`（fallback: `eeg_eeg2text`）<br>`mask_spectro`（fallback: `mask_eeg2text`） | (B, 374, 65) | 直接使用预计算 spectrogram；greedy decoding |
| **CET-MAE** | `eeg_word_norm2d`（fallback: `eeg_normalized_2d`）<br>`mask_word_with_sent`（fallback: `mask_with_sent`） | (B, 56, 840) | wrapper 内追加句级 EEG（`sent_eeg_raw`）到序列末尾；greedy decoding |
| **GLIM** | `eeg_word_raw`（fallback: `eeg_raw` / `eeg`）<br>`mask_word`（fallback: `mask`） | (B, 1280, 128) | wrapper 内动态转换：`adaptive_avg_pool1d` 840→128 + `interpolate` 56→1280；beam=2 |

#### 生成参数配置

各模型在 `eval_config.yaml` 的 `generation.model_overrides` 中独立配置，不强行统一 beam size：

```yaml
generation:
  defaults:
    max_new_tokens: 56
    num_beams: 1
    do_sample: false
  model_overrides:
    eeg_to_text:
      max_new_tokens: 56
      num_beams: 5
      do_sample: true
      repetition_penalty: 5.0
    glim:
      num_beams: 2
```

#### 说明

- **EEG-To-Text** 和 **CET-MAE** 共享相同的词级频域特征基础（`eeg_word_norm1d` / `eeg_word_norm2d`），区别在于 CET-MAE 使用双重归一化和句级 EEG 追加。
- **EEG2Text** 使用完全不同的路径：从 `eeg_spectro`（预计算 spectrogram）直接送入 ShallowNet，不使用词级特征。
- **GLIM** 的维度转换（840→128, 56→1280）在 wrapper 层完成，转换与原始训练数据的分布可能存在差异，评估结果仅供参考。
