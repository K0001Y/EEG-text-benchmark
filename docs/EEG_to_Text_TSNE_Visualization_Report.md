# EEG-to-Text 对比实验 t-SNE 可视化报告

> 配套于《EEG-to-Text 模型性能归因：双诊断线对比实验报告》v4。本报告汇总**诊断线 A（数据有效性验证）**与**诊断线 B（模型噪声对照）**在 LOSO 5 折与 4 模型 × 4 噪声条件下的全部 t-SNE 降维可视化图像，并附逐图读图要点。

- 线 A 输出目录：`benchmark_eval/test_outputs/line_a/dataset_validity/`
- 线 B 输出目录：`benchmark_eval/test_outputs/tsne_b/`
- 生成脚本：
  - 线 A：`benchmark_eval/scripts/diagnostics/validate_eeg_signal.py`
  - 线 B：`benchmark_eval/scripts/analysis/visualize_b_embeddings.py`
- 默认 perplexity = 30（sklearn TSNE，`max_iter=1000`），被试子图附带 p=5/50 作为稳健性对照。

---

## 1. 诊断线 A：数据层面 t-SNE

对 test 集的 840 维句级 EEG 特征（`sent_eeg_raw` 逐样本 z-score 后 mean-pool）直接做 t-SNE，不经过任何深度模型。共 6 张图。

### 1.1 按句子 ID 着色（130 类）

![t-SNE by sentence (perplexity=30)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_sentence_p30.png)

**读图要点**：
- 若 EEG 句级特征在无监督降维空间中自发聚成 130 个句子簇，则说明信号可直接分离。
- 实际结果：点云弥散，各句子类别高度混叠；这与 A1 Linear Probe Top-1≈1.92% 的弱信号结论一致。

### 1.2 按阅读 session 着色（Session 1 / Session 2）

![t-SNE by session (perplexity=30)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_session_p30.png)

**读图要点**：
- ZuCo 每被试分两个 session 录制，若存在电极漂移则 session 会在 t-SNE 上分离。
- 观察结果：session 之间基本重叠，对应 η²_session 均值 0.006（远低于 η²_subject 0.48），session 效应可忽略。

### 1.3 按被试 ID 着色（30 被试）— 三个 perplexity 稳健性对照

**perplexity = 5**  
![t-SNE by subject (perplexity=5)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_subject_p5.png)

**perplexity = 30（默认）**  
![t-SNE by subject (perplexity=30)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_subject_p30.png)

**perplexity = 50**  
![t-SNE by subject (perplexity=50)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_subject_p50.png)

**读图要点**：
- 在三个 perplexity 下，同一被试的样本均稳定聚成**紧致簇**，30 个被试间界限清晰。
- 这正是 §4.1.2 中 η²_subject=0.48、Wilcoxon dz=3.05 的可视化呈现——**被试身份是 840 维特征最主要的方差来源**，也是 §5.2 所述核心瓶颈的直接证据。
- 从 p=5 到 p=50 结构保持一致，排除了 perplexity 敏感性。

### 1.4 按阅读任务着色（Task1-SR / Task2-NR / Task3-TSR / Task2-NR-2.0）

![t-SNE by task (perplexity=30)](../benchmark_eval/test_outputs/line_a/dataset_validity/tsne_by_task_p30.png)

**读图要点**：
- 同一被试跨任务样本基本聚在一起，任务差异被被试效应淹没。
- 提示若要消除 task 混淆因素，必须先解耦被试身份。

---

## 2. 诊断线 B：模型嵌入 t-SNE

对四个模型（CET-MAE、EEG-To-Text、EEG2Text、GLIM）在 real/gaussian/shuffle/zero 四种输入条件下产生的 EEG 侧嵌入（以及部分视图附带 text 侧嵌入）做 t-SNE，共 33 张图。

每个模型提供 8 类视图：

| 视图 | 作用 |
|------|------|
| `by_dataset` | real 条件下按数据集（ZuCo1 / ZuCo2）着色，诊断跨数据集分布漂移 |
| `by_subject` | real 条件下按被试着色，诊断是否继承了被试偏差 |
| `by_task` | real 条件下按任务着色，诊断任务级聚类 |
| `cross_modal` | real 条件下 EEG 嵌入 vs Text 嵌入联合 t-SNE，诊断跨模态对齐 |
| `four_conditions` | 同模型 real/gaussian/shuffle/zero 四色联合，诊断条件可分性 |
| `shuffle_diag_by_cond` | real vs shuffle 联合图按条件着色 |
| `shuffle_diag_by_gt` | real vs shuffle 联合图按 ground truth 句子 ID 着色 |
| `zero_response` | zero 输入下 EEG 嵌入分布，诊断常量回退模式 |

额外提供 **跨模型 real 条件联合 t-SNE** 1 张。

### 2.1 CET-MAE（模式 B：学到 EEG 统计特性）

**by_dataset**  
![CET-MAE t-SNE by dataset](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_by_dataset_p30.png)

**by_subject**  
![CET-MAE t-SNE by subject](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_by_subject_p30.png)

**by_task**  
![CET-MAE t-SNE by task](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_by_task_p30.png)

**cross_modal（EEG vs Text）**  
![CET-MAE cross-modal](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_cross_modal_p30.png)

**four_conditions**  
![CET-MAE four conditions](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_four_conditions_p30.png)

**shuffle_diag_by_cond**  
![CET-MAE shuffle diag by cond](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_shuffle_diag_by_cond_p30.png)

**shuffle_diag_by_gt**  
![CET-MAE shuffle diag by GT](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_shuffle_diag_by_gt_p30.png)

**zero_response**  
![CET-MAE zero response](../benchmark_eval/test_outputs/tsne_b/tsne_cet_mae_zero_response_p30.png)

**读图要点**：
- `four_conditions` 中 real 与 gaussian/zero 的点云质心存在可见偏移，视觉上呼应 BH-FDR 校正下 `real_vs_gaussian MRR p_adj=0.037★`、`real_vs_zero MRR p_adj=0.011★`。
- `cross_modal` 显示 EEG 与 Text 两朵云仍明显分离，与 MRR≈0.048 的弱对齐一致。
- `by_subject` 上仍能看到明显被试簇，说明 CET-MAE 未能有效消除被试效应。

### 2.2 EEG-To-Text（模式 A：编码器完全无效）

**by_dataset**  
![EEG-To-Text t-SNE by dataset](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_by_dataset_p30.png)

**by_subject**  
![EEG-To-Text t-SNE by subject](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_by_subject_p30.png)

**by_task**  
![EEG-To-Text t-SNE by task](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_by_task_p30.png)

**cross_modal**  
![EEG-To-Text cross-modal](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_cross_modal_p30.png)

**four_conditions**  
![EEG-To-Text four conditions](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_four_conditions_p30.png)

**shuffle_diag_by_cond**  
![EEG-To-Text shuffle diag by cond](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_shuffle_diag_by_cond_p30.png)

**shuffle_diag_by_gt**  
![EEG-To-Text shuffle diag by GT](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_shuffle_diag_by_gt_p30.png)

**zero_response**  
![EEG-To-Text zero response](../benchmark_eval/test_outputs/tsne_b/tsne_eeg_to_text_zero_response_p30.png)

**读图要点**：
- `four_conditions` 中四种条件点云**高度重叠、无可识别差异**，是 §4.4.3 中"real 对任意噪声全部 n.s."的直观图像证据。
- `zero_response` 的点云形态与 real 几乎完全一致，说明编码器对输入不敏感，指向梯度断路或权重退化。

### 2.3 EEG2Text（模式 A：编码器未生效）

**by_dataset**  
![EEG2Text t-SNE by dataset](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_by_dataset_p30.png)

**by_subject**  
![EEG2Text t-SNE by subject](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_by_subject_p30.png)

**by_task**  
![EEG2Text t-SNE by task](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_by_task_p30.png)

**cross_modal**  
![EEG2Text cross-modal](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_cross_modal_p30.png)

**four_conditions**  
![EEG2Text four conditions](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_four_conditions_p30.png)

**shuffle_diag_by_cond**  
![EEG2Text shuffle diag by cond](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_shuffle_diag_by_cond_p30.png)

**shuffle_diag_by_gt**  
![EEG2Text shuffle diag by GT](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_shuffle_diag_by_gt_p30.png)

**zero_response**  
![EEG2Text zero response](../benchmark_eval/test_outputs/tsne_b/tsne_eeg2text_zero_response_p30.png)

**读图要点**：
- `by_subject` 仍显示被试簇结构，说明即便采用原始时序输入 `(24000, 105)`，被试效应仍传递到嵌入空间。
- `four_conditions` 中 zero 云轻微分离而 real/gaussian/shuffle 高度缠绕，对应 BH-FDR 中仅 `gaussian_vs_zero MeanRank` 与 `shuffle_vs_zero R@5` 显著。

### 2.4 GLIM（模式 B + 异常：zero > real）

**by_dataset**  
![GLIM t-SNE by dataset](../benchmark_eval/test_outputs/tsne_b/tsne_glim_by_dataset_p30.png)

**by_subject**  
![GLIM t-SNE by subject](../benchmark_eval/test_outputs/tsne_b/tsne_glim_by_subject_p30.png)

**by_task**  
![GLIM t-SNE by task](../benchmark_eval/test_outputs/tsne_b/tsne_glim_by_task_p30.png)

**cross_modal**  
![GLIM cross-modal](../benchmark_eval/test_outputs/tsne_b/tsne_glim_cross_modal_p30.png)

**four_conditions**  
![GLIM four conditions](../benchmark_eval/test_outputs/tsne_b/tsne_glim_four_conditions_p30.png)

**shuffle_diag_by_cond**  
![GLIM shuffle diag by cond](../benchmark_eval/test_outputs/tsne_b/tsne_glim_shuffle_diag_by_cond_p30.png)

**shuffle_diag_by_gt**  
![GLIM shuffle diag by GT](../benchmark_eval/test_outputs/tsne_b/tsne_glim_shuffle_diag_by_gt_p30.png)

**zero_response**  
![GLIM zero response](../benchmark_eval/test_outputs/tsne_b/tsne_glim_zero_response_p30.png)

**读图要点**：
- `four_conditions` 中 zero 云显著偏离 real/shuffle，且位置更接近 text 嵌入的文本先验中心——这是 §4.2.4 中 `zero R@10=13.02% > real R@10=8.83%` 异常的可视化解释。
- `zero_response` 点云高度紧致（半径小），说明零输入触发了接近常量的回退嵌入，文本解码器在该嵌入附近生成了先验驱动的高质量候选。
- 对应 BH-FDR 中 `real vs gaussian/zero`、`gaussian vs shuffle`、`shuffle vs zero` 四对 MRR/MeanRank 全部 p_adj=0.011★★。

### 2.5 跨模型对比（real 条件）

![Cross-model real](../benchmark_eval/test_outputs/tsne_b/tsne_cross_model_real_p30.png)

**读图要点**：
- 四个模型在 real 条件下的 EEG 嵌入占据 t-SNE 空间的不同区域，形态各异但共同特点是 EEG 云与 Text 云分离——跨模态对齐均未建立。
- 与 Friedman `real p=0.036★` 的弱信号一致：模型间排名差异存在但 Nemenyi CD 下无显著对。

---

## 3. 图像索引

### 线 A（6 张）

| # | 文件 |
|---|------|
| 1 | `line_a/dataset_validity/tsne_by_sentence_p30.png` |
| 2 | `line_a/dataset_validity/tsne_by_session_p30.png` |
| 3 | `line_a/dataset_validity/tsne_by_subject_p5.png` |
| 4 | `line_a/dataset_validity/tsne_by_subject_p30.png` |
| 5 | `line_a/dataset_validity/tsne_by_subject_p50.png` |
| 6 | `line_a/dataset_validity/tsne_by_task_p30.png` |

### 线 B（33 张）

每模型 8 张（`{model}` ∈ {cet_mae, eeg_to_text, eeg2text, glim}）：

```
tsne_b/tsne_{model}_by_dataset_p30.png
tsne_b/tsne_{model}_by_subject_p30.png
tsne_b/tsne_{model}_by_task_p30.png
tsne_b/tsne_{model}_cross_modal_p30.png
tsne_b/tsne_{model}_four_conditions_p30.png
tsne_b/tsne_{model}_shuffle_diag_by_cond_p30.png
tsne_b/tsne_{model}_shuffle_diag_by_gt_p30.png
tsne_b/tsne_{model}_zero_response_p30.png
```

跨模型联合：

```
tsne_b/tsne_cross_model_real_p30.png
```

合计 **6 + 4×8 + 1 = 39 张**。

---

## 4. 结论速览

| 观察 | 可视化证据 | 统计证据 |
|------|-----------|---------|
| 被试效应主导 | 线 A `by_subject` 在 p=5/30/50 下均呈紧致簇；线 B 各模型 `by_subject` 仍继承此结构 | A2 Wilcoxon p=4.1e-139, dz=3.05 |
| 句子信号弱 | 线 A `by_sentence` 点云混叠 | A1 Top-1=1.92% (Binomial p=9.1e-7) |
| Session 可忽略 | 线 A `by_session` 基本重叠 | η²_session 均值 0.006 |
| 跨模态对齐缺失 | 所有 `cross_modal` 两云分离 | 四模型 MRR≈0.045–0.049 |
| EEG-To-Text/EEG2Text 编码器失效 | `four_conditions` 四色重叠 | real 对任意噪声 BH-FDR n.s. |
| GLIM zero > real 异常 | `zero_response` 紧致簇 + `four_conditions` zero 云偏离 | 4 对 MRR/MeanRank p_adj=0.011 |
| CET-MAE 唯一有效利用 EEG | `four_conditions` real 与 gaussian 质心可见偏移 | real_vs_gaussian MRR p_adj=0.037 |

---

*报告生成于 2026-05-01（配合 EEG_to_Text_Contrast_Experiment_Report v4 §4.4）*  
*生成脚本：`benchmark_eval/scripts/diagnostics/validate_eeg_signal.py`, `benchmark_eval/scripts/analysis/visualize_b_embeddings.py`*
