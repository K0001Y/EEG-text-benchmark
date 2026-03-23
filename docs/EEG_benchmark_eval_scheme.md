## 统一 EEG-to-Text Benchmark 评估方案

本方案面向 EEG-To-Text、DeWave、EEG2Text、GLIM 以及你后续自定义的 EEG-to-Text 模型，目标是在统一的数据格式、统一的解码设置和统一的指标体系下进行公平可复现的比较，尤其强调在生成阶段关闭隐式教师强制，真实反映模型依赖 EEG 信号的能力。

### 一、设计目标与基本原则

本 benchmark 的设计目标可以分为以下三个方面：

1. 提供标准数据视图：将 ZuCo v1.0/v2.0 的原始 `.mat` 数据统一整理为便于各模型直接使用的中间格式。
2. 统一模型调用与解码协议：强制使用自回归生成，避免 teacher forcing 导致的虚高指标。
3. 构建一致的评估与对照实验：从文本相似度、语义对齐和鲁棒性三个维度，使不同模型的表现可直接比较。

在此基础上，整体方案遵循以下基本原则：

1. 数据统一性：所有模型在评估时必须使用同一份预处理后的数据，不允许各自重新划分或过滤样本。
2. 接口统一性：评估脚本只通过统一的包装接口与模型交互，不直接依赖模型内部实现细节。
3. 随机性可复现：所有随机性（数据划分、掩码、噪声等）都需要显式固定随机种子，并写入配置以保证结果可重复。

### 二、统一的数据格式与划分方案

#### 2.1 数据来源与任务范围

在数据层面可以按如下步骤统一处理和使用 ZuCo 数据：

1. 确定任务范围：选择 ZuCo v1.0 和 v2.0 中的 task1（情感）、task2/NR（自然阅读）、task3/TSR（关系）作为统一评估的任务集合。
2. 收集原始文件：下载各任务对应的 Matlab `.mat` 文件，保证与 EEG-To-Text、EEG2Text、GLIM 使用的数据版本一致。
3. 抽取基础信息：在统一预处理脚本中，从 `.mat` 中解析句级 EEG（例如 `rawData`）和刺激句子文本，但不在此阶段做与具体模型结构绑定的特殊加工。
4. 生成中间表示：将上述 EEG 与文本信息存入统一的中间结果（DataFrame 或 pickle），作为所有模型后续处理的唯一数据源。

#### 2.2 统一样本格式

统一样本格式建议按如下字段与步骤设计：

1. 定义核心字段：
   - `eeg`：`float32` 二维数组，形状为 \(L_\text{max} \times C\)，例如 `L_max=24000`、`C=105`。
   - `mask`：`int8` 一维数组，长度为 `L_max`，有效时间步为 1，padding 为 0。
   - `input_text`：ZuCo 中的刺激句子文本，作为解码目标。
   - `prompt`：例如 `"<NR> <ZuCo1> <ZAB>"`，编码任务、数据集和被试信息。
   - `text_uid`：文本唯一 ID，用于识别相同句子的不同 EEG 片段。
2. 统一时间长度：
   - 对每条 EEG，将原始长度 T 的序列在时间维上补零或截断到统一的 `L_max`；
   - 同时在 `mask` 中将前 T 个位置置为 1，补零部分置为 0。
3. 补充元信息字段（可选但推荐）：
   - `task`（task1/task2/task3）、`dataset`（ZuCo1/ZuCo2）、`subject`（被试 ID）；
   - `sentiment_label`（情感标签，仅对 task1 有意义）、`relation_label`（关系标签，仅对 task3 有意义）。
4. 汇总为统一表格：
   - 使用 pandas DataFrame 将上述字段组织为一张表，或等价的统一 pickle 文件，作为 benchmark 的官方数据入口。

#### 2.3 统一训练/验证/测试划分

统一训练/验证/测试划分可以按以下步骤实施：

1. 在统一 DataFrame 中新增 `phase` 字段，取值为 `train` / `val` / `test`。
2. 选择划分策略：
   - 方案 A：直接沿用某个已有工作（如 GLIM）的划分方案，将其作为 benchmark 官方 split；
   - 方案 B：自定义基于 `text_uid` 的划分，例如对每个任务：
     1）按 `text_uid` 排序；
     2）前 80% 的文本划入 train，并将对应的所有 EEG 样本标记为 train；
     3）剩余 20% 的文本按 1:1 分成 val 与 test。
3. 固化划分规则：
   - 在数据构建脚本中写死划分逻辑；
   - 将划分结果随数据一起保存，保证所有模型共享相同的 train/val/test 划分。
4. 强制遵守 phase 约束：
   - 训练阶段只能使用 phase==train 的样本；
   - 调参与早停只能基于 phase==val；
   - 最终报告的指标只能在 phase==test 上计算，不允许将 test 样本混入训练或验证集。

### 三、统一的模型接口与适配器设计

#### 3.1 通用模型包装接口

为了屏蔽不同仓库实现的差异，可以按如下步骤设计统一包装接口：

1. 定义包装类职责：
   - 构造函数负责加载模型配置或 checkpoint；
   - `encode_eeg` 接收统一 DataFrame 的 `eeg`、`mask`、`prompt` 和元信息，输出模型 encoder 输入张量；
   - `generate_text` 在严格禁用教师强制的前提下，调用模型的自回归解码接口，生成完整预测文本。
2. 约定方法签名示例：
   - `__init__(config_or_checkpoint)`：完成模型加载与设备放置；
   - `encode_eeg(eeg, mask, prompt, metadata) -> EncodedEEG`：返回可直接送入 decoder 的 EEG 表示；
   - `generate_text(encoded_eeg, max_new_tokens, decoding_config) -> str`：返回预测文本字符串。
3. 限定评估脚本与模型交互方式：
   - 评估脚本只调用上述接口，不直接访问模型内部结构；
   - 这样可以确保不同模型在评估时遵守同一协议，便于维护和扩展。

#### 3.2 不同模型的适配策略

针对不同类型模型，可以采用以下适配策略：

1. EEG-To-Text / DeWave（词级 EEG 序列模型）：
   - 在 `encode_eeg` 中，将句级 `(L_max, C)` 表示沿时间维按固定窗口划分，聚合为长度为 `max_len` 的伪词序列；
   - 聚合方式可选平均池化或卷积下采样，将每个窗口映射为固定维度向量；
   - 将得到的“词 × 特征”序列作为原始模型的输入，实现与旧代码兼容。
2. EEG2Text（句级 raw / 频谱模型）：
   - 直接将统一 DataFrame 的 `eeg` 和 `mask` 填入原始代码中的 `sent_level_EEG` 与相应 mask；
   - 若原模型使用频谱，可在数据层统一生成频谱视图，并在包装器中选择对应字段。
3. GLIM（DataFrame 驱动模型）：
   - 统一数据格式本身参考 GLIM 的 DataFrame 设计，包装器只需做字段名映射；
   - 将 `eeg`、`mask`、`prompt`、`input_text`、`text_uid` 等直接传入 GLIM 的数据模块和模型。
4. 自定义新模型：
   - 在模型设计阶段直接以统一 DataFrame 字段为输入，避免事后再做复杂适配；
   - 仅在包装层中实现与 benchmark 接口的轻量封装。

### 四、统一的生成与评估流程

#### 4.1 严格禁止隐式教师强制的生成协议

生成与评估流程需要在协议层面做以下约定：

1. 自回归生成强制要求：
   - 所有模型在评估时必须使用自回归生成接口，如 `model.generate`；
   - 解码过程中每一步只能依赖模型先前生成的 token，不得在任何位置加入 ground truth token。
2. 明确禁止的做法：
   - 一次性将整条目标句子作为 decoder 输入前向传播，再对 logits 全序列取 argmax 解码；
   - 在解码时混入部分真实 token（显式或隐式的教师强制）。
3. 统一解码超参数：
   - 统一设置 `max_new_tokens`，例如略大于数据集中最长参考句子的长度（64 或 128）；
   - 统一使用 greedy 解码，或约定好的 beam search 参数（如 beam_size=4，temperature=0.0，top_p=1.0）；
   - 明确 EOS 与 padding token 的约定，保证各模型在结束条件上行为一致。

#### 4.2 文本层面指标

文本层面的指标可以按以下步骤计算与组织：

1. 对每条样本记录预测文本与参考文本：
   - 在评估循环中，保存 `pred_text` 与 `ref_text`，并记录关联的 task、subject 等元信息；
2. 统一计算基础文本指标：
   - 在整个 test 集上计算 BLEU-1/2/3/4；
   - 计算 ROUGE-L、ROUGE-1、ROUGE-2 等指标；
3. 引入语义相似度指标（推荐）：
   - 使用 BERTScore 或句向量余弦相似度衡量语义接近程度；
4. 分任务与分被试统计：
   - 按 task1/task2/task3 分别统计上述指标；
   - 按 subject 分层统计，观察不同被试上的鲁棒性，并与 DeWave 等工作中的 per-subject 分析对齐。

#### 4.3 下游语义任务评估

下游语义任务评估可以从以下几个方面展开：

1. EEG–文本检索任务：
   - 定义候选文本集合（例如所有 test 集句子）；
   - 将模型生成的文本或其中间语义表示映射到向量空间；
   - 计算与候选文本的相似度，评估 top-k 准确率或检索排名质量。
2. 情感分类任务：
   - 对 task1 的样本，将生成文本输入预训练的情感分类器；
   - 比较预测情感标签与 ZuCo 提供的 sentiment_label 的一致性。
3. 关系分类任务：
   - 对 task3 的样本，将生成文本输入关系分类器；
   - 比较预测关系类型与 relation_label 的一致性。
4. 结果解读：
   - 将这些下游任务指标与文本相似度指标结合，看模型是否在语义空间内也保持较高的一致性。

### 五、控制实验与鲁棒性检验

#### 5.1 噪声 EEG 输入实验

噪声 EEG 控制实验可按如下步骤执行：

1. 构造噪声版 EEG：
   - 在统一 DataFrame 中，为每条样本生成形状与原始 `eeg` 相同的高斯白噪声；
   - 使用固定随机种子，保证不同运行之间噪声可复现。
2. 设置噪声 mask：
   - 对噪声 `eeg`，将 `mask` 直接设为全 1，表示所有时间步都视为有效；
3. 双轨评估：
   - 对每个模型，在同一配置下分别用真实 EEG 和噪声 EEG 进行生成与评估；
   - 记录两种输入条件下的 BLEU/ROUGE/BERTScore 等指标。
4. 分析预期现象：
   - 真实 EEG 条件下，指标应显著优于噪声条件；
   - 若噪声输入表现接近真实 EEG，则说明模型严重依赖语言先验，脑信号贡献有限，需要在报告中明确指出。

#### 5.2 打乱配对与标签的额外对照

在噪声实验之外，可以增加如下对照：

1. EEG–文本配对打乱实验：
   - 随机打乱 DataFrame 中 `eeg` 与 `input_text` 的配对关系，保持各自边缘分布不变；
   - 在打乱后的数据上重复生成与评估流程，观察指标是否接近噪声输入情形。
2. 标签打乱实验（针对下游分类）：
   - 针对情感/关系分类任务，随机重排 sentiment_label 或 relation_label；
   - 重新计算分类准确率，若仍能取得接近真实标签的表现，提示存在评估泄漏或分类器过度依赖语言先验等问题。
3. 综合解读：
   - 将噪声输入、配对打乱和标签打乱结果与正常评估结果对比，有助于判断模型真正使用了多少 EEG 信息。

### 六、实施建议与工程落地

在工程实现上，可以按三层结构逐步落地：

1. 数据层：
   - 从 ZuCo 原始 `.mat` 或各仓库已有 pickle/DataFrame 读取信息；
   - 按第二节方案构建统一的 DataFrame 或统一格式的 pickle，并固化 train/val/test 划分；
   - 将该数据作为所有模型的唯一评估入口。
2. 模型包装层：
   - 为每种模型实现 `BenchmarkModelWrapper` 接口；
   - 在 `encode_eeg` 中完成从统一数据格式到模型内部表示的转换；
   - 在 `generate_text` 中统一通过 `model.generate` 或等价接口进行自回归解码。
3. 评估脚本层：
   - 编写通用评估脚本，循环遍历 test 集，对不同模型重复相同的生成与评估流程；
   - 统一输出 BLEU/ROUGE/BERTScore 等文本指标、下游任务指标，以及噪声输入和打乱对照实验的结果；
   - 将评估配置、随机种子和结果日志一并保存，便于复现和对比。

### 七、当前实现的数据处理流程（ZuCo → 统一数据 → 评估脚本 → 模型）

#### 7.1 从 ZuCo `.mat` 到统一 EEG-Text pickle

1. 入口脚本：使用 `benchmark_eval/build_unified_dataset_eegtotext.py`。
   - 通过命令行参数指定 `--zuco-root`（默认为 `models/EEG-To-Text-main/dataset/ZuCo`）、`--tasks`（如 `task1-SR,task2-NR,task3-TSR,task2-NR-2.0`）、`--output`（统一数据 pickle 路径），以及 `--max-len`、`--dim`、`--eeg-type` 等超参数。
2. 解析 ZuCo v1 `.mat`：
   - 调用 `load_dataset_from_mat_v1(zuco_root, task_name)`，从 `ZuCo/<task>/Matlab_files/*.mat` 读取原始 Matlab 结构。
   - 对于每个被试 subject，构造 `dataset_dict[subject] = List[sent_obj or None]`，其中 `sent_obj` 至少包含：
     - `content`：句子字符串；
     - `sentence_level_EEG`：`mean_t1`/`mean_t2`/`mean_a1`/.../`mean_g2` 这 8 个频段的句级 EEG 向量；
     - `word`：按词的列表，每个元素带有 `word_level_EEG`（包含 FFD/TRT/GD 各频段特征）。
3. 解析 ZuCo v2 `.mat`：
   - 调用 `load_dataset_from_mat_v2(zuco_root)`，从 v2 的 HDF5 `.mat` 中读取 `sentenceData`。
   - 使用 `models.EEG-To-Text-main.util.data_loading_helpers_modified` 中的工具，把 HDF5 引用解码成句子字符串和词级 EEG 特征，同样组织成 `dataset_dict[subject]` 结构。
4. 构建统一样本：
   - 对每个任务/被试，调用 `build_samples_for_task(dataset_dict, task_name, max_len, dim, eeg_type)`。
   - 对每条句子：
     - 从 `word_level_EEG[eeg_type][band]` 中按 `DEFAULT_BANDS = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]` 拼接出单词级向量，并做 1D z-score 归一化；
     - 从 `sentence_level_EEG` 中按相同频段拼接句级向量，并做 1D 归一化；
     - 将所有单词向量 + 句级向量堆叠成二维矩阵，整体做 2D z-score 归一化，再在时间维上截断/补零到固定 `L_max = max_len`；
     - 得到 `eeg`：形状为 `(L_max, C)` 的 `float32` 数组，其中 `C = dim * len(bands)`；
     - 构造 `mask`：前 `seq_len` 位置为 1.0，padding 部分为 0.0；
     - 同时填充：
       - `input_text` / `reference_text`：均为 ZuCo 的原始句子文本；
       - `meta`：包含 `task`、`subject`、`sentence_index`、`source="ZuCo-MAT"` 等信息；
       - `phase` 先置为 `None`，后续统一划分。
5. 统一划分 train/val/test：
   - 在 `build_samples_for_task` 末尾，按被试内样本顺序做 8:1:1 划分：
     - 前 80% 设为 `phase="train"`；
     - 中间 ~10% 设为 `phase="val"`；
     - 剩余样本设为 `phase="test"`；
   - 所有任务和被试的样本合并成一个 `List[Dict]`，最终通过 `pickle.dump` 保存到 `--output` 指定的统一数据文件中。

#### 7.2 从统一 pickle 到评估用 DataLoader

1. 入口类：`benchmark_eval/dataset.py` 中的 `UnifiedDataset`。
   - 初始化时读取统一 pickle：`UnifiedDataset(data_path, phase="test")`；
   - 遍历 pickle 中的每个样本字典，封装为 `UnifiedSample`（包含 `eeg`、`mask`、`input_text`、`reference_text`、`phase`、`meta`）；
   - 若指定了 `phase`，只保留 `sample.phase == phase`（或 `phase` 缺失时全部保留）。
2. 提供给 PyTorch 的样本形式：
   - `__getitem__` 返回一个字典：
     - `idx`：样本在统一数据中的全局索引；
     - `eeg`：转换为 `torch.float32` 的张量，形状 `(L_max, C)`；
     - `mask`：转换为 `torch.float32` 的张量，形状 `(L_max,)`；
     - `input_text`：原始句子文本；
     - `reference_text`：用于计算指标的目标文本；
     - `meta`：包含任务、被试等元信息的字典。

#### 7.3 评估脚本如何把数据喂给模型

1. 入口脚本：`benchmark_eval/eval_runner.py`。
   - 命令行参数：
     - `--data-path`：统一数据 pickle 路径；
     - `--phase`：评估使用的划分（默认为 `test`）；
     - `--output-dir`：保存日志和预测结果的目录；
     - `--model-name`：选择具体的模型封装器（如 `dummy`、后续扩展的 `eeg_to_text`、`cet_mae`、`glim` 等）。
   - 脚本会构造 `UnifiedDataset` 和对应的 `DataLoader`，并支持中断恢复（通过 `state.json` 和增量写入的 `predictions.jsonl`）。
2. 批处理数据并调用模型：
   - 从 `DataLoader` 取到 `batch` 后，得到：
     - `batch["idx"]`、`batch["eeg"]`、`batch["mask"]`、`batch["input_text"]`、`batch["reference_text"]`、`batch["meta"]`；
   - 为了让模型可以访问原始文本，评估脚本会构造 `meta_batch`：
     - 对每条样本，将 `batch["meta"][i]` 拷贝一份，并额外加入 `"input_text"` 字段；
   - 通过工厂函数 `build_model_wrapper(args.model_name)` 构建对应的 `BenchmarkModelWrapper` 实例 `model`；
   - 调用 `model.generate_text(eeg, mask, meta_batch)` 得到长度为 batch_size 的预测文本列表。
3. 保存预测并计算指标：
   - 对每条样本写入一行 JSON 到 `predictions.jsonl`，内容包括：
     - `idx`、`reference`（参考文本）、`prediction`（模型输出）、`meta`（包含任务/被试/原始 input_text 等）；
   - 所有样本处理完后，重新加载预测文件，按 `idx` 排序后调用 `compute_corpus_metrics` 计算 BLEU/ROUGE 等指标，并写入 `metrics.json`。

#### 7.4 不同模型封装层如何使用这些输入

1. 抽象接口：`benchmark_eval/model_wrappers.py` 中定义了 `BenchmarkModelWrapper` 抽象基类：
   - `encode_eeg(eeg, mask, meta)`：可选，用于复杂模型提前把 `(B, L_max, C)` 的 EEG 序列编码成隐变量；
   - `generate_text(eeg, mask, meta)`：必需，实现从 EEG 到文本的**自回归生成**（内部必须使用诸如 HuggingFace `generate` 之类的接口，禁止 teacher forcing）。
2. 当前占位实现：
   - 目前仓库中提供了 `DummyEchoWrapper`，它忽略 `eeg` 和 `mask`，仅从 `meta["input_text"]` 读出原句并加上前缀，主要用于验证评估流程是否跑通。
3. 后续为具体模型（EEG-To-Text、CET-MAE、EEG2Text、GLIM、DeWave 等）写封装时，遵循以下数据流：
   - 封装器从评估脚本拿到统一格式的 `(eeg, mask, meta)`；
   - 在 `encode_eeg` 或 `generate_text` 内部，将 `(B, L_max, C)` 转换为各自仓库原本使用的输入形式（例如词级 EEG 序列、句级 raw EEG、频谱图、或 DataFrame 结构等）；
   - 再调用原模型的 encoder/decoder 或 `generate` 接口完成自回归预测；
   - 这样可以在不改动原模型训练代码的前提下，让所有模型共享同一份 ZuCo→统一数据→评估脚本的数据路径。

通过这一方案，可以在不强行修改各模型内部架构的前提下，把它们拉到同一实验地面上进行公平比较，并为你后续加入新的 EEG 表示（例如 pixel/pixl 编码）提供一套自然的接口和评估环境。