# GLIM：学习可解释表示实现语义忠实的EEG到文本生成

以下是对GLIM（Generative Language Inspection Model，生成语言检验模型）项目的详细解释，涵盖其背景、方法、架构、实验结果、实现方式以及使用指南，基于官方PyTorch实现。

---

## 背景与问题

脑机接口（BCI）技术近年来发展迅速，特别是通过非侵入式脑电图（EEG）信号解码大脑活动，生成文本或图像等输出。然而，现有EEG到文本生成模型存在**幻觉问题**（hallucination），即生成的文本可能并不忠实反映大脑的语义激活，而是由强大的预训练生成模型“凭空想象”出来的。这种不可靠性限制了脑解码技术的实际应用。

GLIM项目通过解决**后验崩塌**（posterior collapse）问题，重新定义了EEG到文本解码任务。传统方法试图逐字重建刺激文本（verbatim reconstruction），但EEG信号的信息容量远低于文本，导致信息不对称。GLIM将任务重构为**语义摘要**（semantic summarization），即从EEG信号中提取核心语义，生成语义上忠实的文本，而非字面复制。

---

## GLIM的核心方法

GLIM的核心创新在于学习**可解释且信息丰富的EEG表示**，以增强生成文本的语义 grounding（语义依据）。其主要特点包括：

1. **可解释的EEG表示**：
   - GLIM通过设计特定架构，提取EEG信号中的语义信息，减少生成过程中的幻觉。
   - 这些表示在异构（heterogeneous）和小规模（small-scale）数据集上仍能保持鲁棒性。

2. **语义忠实生成**：
   - 模型不再追求与原始刺激文本的逐字匹配，而是生成与EEG信号语义一致的流畅句子。
   - 通过避免“教师强制”（teacher forcing），GLIM生成的文本更依赖于EEG信号本身。

3. **多维度评估**：
   - 除了传统的文本相似度评估（如BLEU、ROUGE），GLIM引入了更稳健的评估方法：
     - **EEG-文本检索**：验证生成文本是否能从EEG信号中正确检索。
     - **零样本语义分类**：在情感类别（sentiment）、关系类型（relation types）和语料主题（corpus topics）上进行分类。
   - 这些方法为生成式脑解码提供了可靠且可扩展的基准。

---

## 模型架构

GLIM的架构图如下（参考官方提供的`method.png`）：

- **输入层**：接收原始EEG信号（通常为多通道时间序列数据）。
- **EEG编码器**：将EEG信号映射到可解释的语义表示空间，捕获核心语义信息。
- **生成模块**：基于编码后的表示，生成语义忠实的文本。
- **辅助模块**：包括语义分类和检索任务的损失函数，用于优化表示的语义一致性。

架构设计强调模块化，使用PyTorch Lightning实现，便于扩展和复现。

---

## 实验结果

GLIM在公开的**ZuCo数据集**（包括1.0和2.0版本）上进行了广泛测试，结果表明：

1. **生成质量**：
   - GLIM生成流畅且与EEG信号语义一致的句子，优于传统逐字重建方法。
   - 样本展示（参考`text_samples.png`）表明，生成文本能够捕捉输入EEG的语义核心。

2. **评估指标**：
   - 在EEG-文本检索任务中，GLIM表现出较高的准确性，证明生成文本与EEG信号的高度相关性。
   - 零样本语义分类任务显示，GLIM能够有效区分情感、关系和主题类别。

3. **鲁棒性**：
   - GLIM在噪声输入测试（noise-input test）和无提示测试（prompt-free test）中仍能生成合理的文本，表明其对EEG信号的依赖性较强。

完整生成样本可在[wandb交互式报告](https://wandb.ai/mind-reading/glim-iclr/reports/GLIM-generation-samples--VmlldzoxMjc0Njg1NQ?accessToken=5uqxxv6ug80naqfqlni2xvxa8y8l7u6ouc1cgjt0naxk1g8g0h9lgyf8r0e97xyk)或`results/`目录中查看。

---

## 项目内容

GLIM仓库提供了以下资源：

1. **代码实现**：
   - 核心模型实现：[glim.py](model/glim.py)，基于PyTorch Lightning，模块化设计。
   - 训练脚本：[train.py](train.py)，支持从头训练。
   - 测试脚本：[test.py](test.py)，用于生成文本和计算指标。
   - 语义分类笔记本：[predict_corpus.ipynb](predict_corpus.ipynb)，复现论文中的分类结果。

2. **数据预处理**：
   - 提供完整的预处理笔记本（[__STEP1_text_extract_revise.ipynb](data/__STEP1_text_extract_revise.ipynb) 等），支持从原始ZuCo数据生成训练所需格式。
   - 可跳过部分步骤，直接从[__STEP3_eeg_preproc.ipynb](data/__STEP3_eeg_preproc.ipynb)和[label table](data/tmp/zuco_label_8variants.df)开始。

3. **生成样本**：
   - 包含GLIM生成的所有文本样本（[wandb_export_gen_samples_glim.csv](results/wandb_export_gen_samples_glim.csv)）、噪声输入测试样本和无提示测试样本。

4. **模型检查点**：
   - 可从[figshare](https://doi.org/10.6084/m9.figshare.29115161.v1)下载预训练模型检查点。

---

## 环境配置

### 1. 创建环境
运行以下命令以创建Conda环境：
```bash
conda env create -f environment.yml
```

### 2. 下载ZuCo数据集
ZuCo数据集（1.0和2.0版本）可从以下链接获取：
- [ZuCo 1.0](https://osf.io/q3zws/)
- [ZuCo 2.0](https://osf.io/2urht/)

建议按以下结构组织数据（可选择性下载部分文件）：
```
data/
├── raw_data/
│   ├── ZuCo1/
│   │   ├── task_materials/      # 文本和标签
│   │   ├── task1-SR/
│   │   │   └── Matlab files/    # 句子级EEG片段
│   │   ├── task2-NR/
│   │   │   └── Matlab files/
│   │   └── task3-TSR/
│   │       └── Matlab files/
│   └── ZuCo2/
│       ├── task_materials/
│       ├── task1-NR/
│       │   └── Matlab files/
│       └── task2-TSR/
│           └── Matlab files/
```

---

## 数据预处理

用户可以选择以下两种方式进行数据预处理：

1. **逐个运行笔记本**：
   - 按顺序执行所有四个预处理笔记本（从[__STEP1_text_extract_revise.ipynb](data/__STEP1_text_extract_revise.ipynb)开始）。
   - 适用于需要自定义处理流程的情况。

2. **快速开始**：
   - 直接运行[__STEP3_eeg_preproc.ipynb](data/__STEP3_eeg_preproc.ipynb)，并使用提供的[label table](data/tmp/zuco_label_8variants.df)。
   - 跳过生成文本变体的步骤，适合快速复现。

---

## 复现结果

### 1. 生成文本与计算指标
- 下载[模型检查点](https://doi.org/10.6084/m9.figshare.29115161.v1)并放置在`checkpoints/`目录。
- 运行以下命令（单GPU）：
  ```bash
  python test.py
  ```
- 输出：生成句子及整体指标（如文本相似度、检索准确率等）。

### 2. 复现分类结果
- 运行以下笔记本：
  - `predict_xxx.ipynb`（具体文件名根据任务而定）。
- 支持两种分类方法：
  - **CLIP-like方法**：基于视觉-语言模型的语义嵌入。
  - **LLM辅助方法**：利用大语言模型进行语义分类。

---

## 从头训练

运行以下命令以从头训练GLIM模型：
```bash
python train.py
```
- 默认参数适用于大多数场景，仅需调整设备（GPU）和目录相关参数。
- 训练过程使用ZuCo数据集的预处理结果。

---

## 引用

如果您使用GLIM或参考其代码/结果，请引用以下论文：

```bibtex
@article{liu2025glim,
  title={Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation},
  author={Xiaozhao Liu and Dinggang Shen and Xihui Liu},
  year={2025},
  journal={arXiv preprint arXiv:2505.17099},
}
```

---

## 许可证

GLIM © 2025 由仓库所有者拥有，采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。

---

## 总结

GLIM通过学习可解释的EEG表示，解决了EEG到文本生成中的幻觉问题，重新定义了解码任务为语义摘要。其模块化实现、鲁棒评估方法和公开资源为生成式脑解码研究提供了可靠的基础。用户可通过提供的代码、数据集和检查点快速复现结果，或进一步开发定制模型。