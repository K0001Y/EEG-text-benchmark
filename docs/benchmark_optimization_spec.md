# Unified EEG-to-Text Benchmark 优化方案规格文档

**版本**: v1.0  
**日期**: 2026-04-06  
**作者**: QoderWork  

---

## 1. 文档概述

### 1.1 背景

当前Unified EEG-to-Text Benchmark在适配多个模型时遇到以下核心问题：

1. **CET-MAE模型适配问题**: 数据格式不匹配，缺少评估脚本
2. **EEG2Text模型适配问题**: 完全绕过统一数据，从原始spectro pickle加载，导致数据划分不一致
3. **数据一致性**: 不同模型的预处理流程不统一，难以公平比较

### 1.2 目标

通过优化数据生成流程、简化Wrapper设计、增加验证机制，实现：

- 所有模型使用**同一套统一数据**
- Wrapper只负责**模型调用**，不做**数据转换**
- 支持**快速添加新模型**
- 保证**数据一致性**和**评估公平性**

---

## 2. 当前问题分析

### 2.1 CET-MAE 问题诊断

| 问题 | 影响 | 根本原因 |
|------|------|---------|
| 数据格式不匹配 | wrapper期望`eeg_normalized_2d`，但获取不到或格式不对 | CET-MAE使用特殊的2D归一化（词级+句级拼接后整体归一化） |
| 缺少评估脚本 | 无法独立验证CET-MAE性能 | README提到`eval_decoding_eeg_2_text_cet_mae.py`但文件不存在 |
| 位置编码差异 | 生成结果与原始实现不一致 | CET-MAE使用带频段权重的特殊位置编码 |

### 2.2 EEG2Text 问题诊断

| 问题 | 影响 | 根本原因 |
|------|------|---------|
| 绕过统一数据 | 数据划分不一致，无法公平比较 | EEG2Text需要原始时序数据`(T, 105)`，统一数据只提供词级特征`(max_len, 840)` |
| 查找方式不一致 | 可能用到测试集数据训练 | 通过`(task, subject, sentence_index)`查找，与统一数据的`text_uid`划分冲突 |
| 预处理不一致 | 分布偏移，性能不可比 | spectro pickle的预处理与统一数据的预处理流程不同 |

---

## 3. 优化方案

### 3.1 数据层优化

#### 3.1.1 统一数据格式扩展

在`build_unified_dataset.py`中，为每个模型生成特定的数据格式：

```python
record = {
    # ===== 基础字段（所有模型通用）=====
    "input_text": str,                    # 输入句子文本
    "reference_text": str,                # 参考文本
    "phase": str,                         # "train" / "val" / "test"
    "meta": {
        "task": str,
        "subject": str,
        "sentence_index": int,
        "text_uid": int,                  # 用于划分
        "seq_len": int,                   # 原始词数
    },
    
    # ===== EEG-To-Text 格式 =====
    "eeg": np.ndarray,                    # (max_len, 840) 默认，1D归一化
    "eeg_normalized_1d": np.ndarray,      # (max_len, 840) 逐词1D归一化
    "mask": list,                         # (max_len,) 词级mask
    
    # ===== CET-MAE 格式 =====
    "eeg_cet_mae": np.ndarray,            # (max_len, 840) 词+句2D归一化
    "mask_cet_mae": list,                 # (max_len,) 包含句级的mask
    
    # ===== EEG2Text 格式 =====
    "eeg_eeg2text": np.ndarray,           # (24000, 105) 原始时序数据
    "mask_eeg2text": list,                # (24000,) 时序mask
    
    # ===== 未来扩展格式 =====
    "eeg_dewave": np.ndarray,             # (L,) 离散token序列
    "eeg_spectrogram": np.ndarray,        # (time, freq) 频谱图
}
```

#### 3.1.2 归一化方式标准化

确保每种格式的归一化与原始模型完全一致：

| 格式 | 归一化方式 | 参考实现 |
|------|-----------|---------|
| `eeg_normalized_1d` | 逐词z-score归一化 | EEG-To-Text `normalize_1d` |
| `eeg_cet_mae` | 词+句拼接后整体2D z-score归一化 | CET-MAE `normalize_2d` |
| `eeg_eeg2text` | 通道级标准化（可选） | EEG2Text原始实现 |

#### 3.1.3 数据生成流程修改

```python
def build_samples_for_task(...):
    # ... 现有代码 ...
    
    # 1. 构建基础词级EEG（所有模型需要）
    word_embeddings_raw = [...]  # 从word_level_EEG提取
    sent_eeg = get_sent_eeg(...)  # 句级EEG
    
    # 2. EEG-To-Text格式（默认）
    record["eeg_normalized_1d"] = build_1d_normalized(word_embeddings_raw)
    record["mask"] = build_word_mask(num_words, max_len)
    
    # 3. CET-MAE格式
    record["eeg_cet_mae"] = build_cet_mae_format(
        word_embeddings_raw, sent_eeg, max_len
    )
    record["mask_cet_mae"] = build_cet_mae_mask(num_words, max_len)
    
    # 4. EEG2Text格式（从原始MAT读取rawData）
    if "rawData" in sent_obj:
        record["eeg_eeg2text"] = build_eeg2text_format(
            sent_obj["rawData"], target_len=24000
        )
        record["mask_eeg2text"] = build_eeg2text_mask(
            sent_obj["rawData"].shape[1], target_len=24000
        )
```

### 3.2 Wrapper层优化

#### 3.2.1 简化Wrapper设计原则

**原则1**: Wrapper只负责**模型调用**，不做**数据转换**

**原则2**: 所有数据预处理在`build_unified_dataset.py`中完成

**原则3**: Wrapper通过`batch`字典获取预计算的数据格式

#### 3.2.2 重构后的CETMAEWrapper

```python
class CETMAEWrapper(BenchmarkModelWrapper):
    """CET-MAE模型Wrapper（简化版）
    
    数据流:
    1. 从batch获取预处理的eeg_cet_mae
    2. 直接输入模型生成文本
    3. 不做任何数据转换
    """
    
    def generate_text(
        self,
        eeg: torch.Tensor,                    # 不使用
        mask: torch.Tensor,                   # 不使用
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        # 直接使用预处理的CET-MAE格式
        input_eeg = batch["eeg_cet_mae"].to(self.device)      # (B, L_max, 840)
        input_mask = batch["mask_cet_mae"].to(self.device)    # (B, L_max)
        
        with torch.no_grad():
            # 1. 添加位置编码
            eeg_with_pos = self.model.pos_embed_e(input_eeg)
            
            # 2. EEG encoder
            eeg_embeddings = self.model.e_branch(
                eeg_with_pos,
                src_key_padding_mask=(1 - input_mask).bool()
            )
            
            # 3. 投影到1024维
            eeg_embeddings = self.model.act(
                self.model.fc_eeg(eeg_embeddings)
            )
            
            # 4. Unified branch
            eeg_embeddings = self.model.unify_branch(
                eeg_embeddings,
                src_key_padding_mask=(1 - input_mask).bool(),
                modality='e'
            )
            
            # 5. BART生成
            encoder_outputs = BaseModelOutput(
                last_hidden_state=eeg_embeddings
            )
            output_ids = self.model.t_branch.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=input_mask,
                max_length=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )
            
            return self.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )
```

#### 3.2.3 重构后的EEG2TextWrapper

```python
class EEG2TextWrapper(BenchmarkModelWrapper):
    """EEG2Text模型Wrapper（简化版）
    
    数据流:
    1. 从batch获取预处理的eeg_eeg2text（原始时序）
    2. 直接输入模型生成文本
    3. 不再从spectro pickle加载
    """
    
    def generate_text(
        self,
        eeg: torch.Tensor,                    # 不使用
        mask: torch.Tensor,                   # 不使用
        meta: List[Dict[str, Any]] | None = None,
        batch: Dict[str, Any] | None = None,
    ) -> List[str]:
        # 直接使用预处理的EEG2Text格式
        eeg_timeseries = batch["eeg_eeg2text"].to(self.device)   # (B, 24000, 105)
        # mask可选使用
        
        with torch.no_grad():
            # 1. 编码EEG
            encoded = self.model(eeg_timeseries)   # (B, seq_len, 1024)
            
            # 2. BART decoder生成
            output_ids = self.text_decoder.generate(
                inputs_embeds=encoded,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )
            
            return self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
```

#### 3.2.4 Dataset.__getitem__修改

```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    return {
        "idx": idx,
        
        # 默认格式（EEG-To-Text兼容）
        "eeg": torch.from_numpy(sample["eeg_normalized_1d"]),
        "mask": torch.tensor(sample["mask"]),
        
        # CET-MAE格式
        "eeg_cet_mae": torch.from_numpy(sample["eeg_cet_mae"]),
        "mask_cet_mae": torch.tensor(sample["mask_cet_mae"]),
        
        # EEG2Text格式
        "eeg_eeg2text": torch.from_numpy(sample["eeg_eeg2text"]),
        "mask_eeg2text": torch.tensor(sample["mask_eeg2text"]),
        
        # 文本
        "input_text": sample["input_text"],
        "reference_text": sample["reference_text"],
        "meta": sample["meta"],
    }
```

### 3.3 评估层优化

#### 3.3.1 创建CET-MAE独立评估脚本

```python
# models/CET-MAE/eval_decoding_eeg_2_text_cet_mae.py
"""CET-MAE独立评估脚本

用于:
1. 独立验证CET-MAE模型性能
2. 与benchmark评估结果对比
3. 调试数据一致性问题
"""

import argparse
import yaml
import torch
from model_mae_bart import CETMAE_project_late_bart
from transformers import BartTokenizer
from torch.utils.data import DataLoader

from data_factory.data_loading_helpers_modified import get_dataset
from evaluate_moudle.rouge import compute_rouge
from evaluate_moudle.wer import compute_wer


def load_model(config):
    """加载CET-MAE模型"""
    model = CETMAE_project_late_bart(
        embed_dim=config.get('embed_dim', 1024),
        eeg_dim=config.get('eeg_dim', 840),
        multi_heads=config.get('multi_heads', 8),
        feedforward_dim=config.get('feedforward_dim', 2048),
        trans_layers=config.get('trans_layers', 6),
        decoder_embed_dim=config.get('decoder_embed_dim', 840),
        pretrain_path=config['pretrain_path'],
        device=config.get('device', 0)
    )
    
    checkpoint = torch.load(config['checkpoint'], map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate(config_path):
    """主评估函数"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型和tokenizer
    model = load_model(config)
    tokenizer = BartTokenizer.from_pretrained(config['pretrain_path'])
    device = torch.device(f"cuda:{config.get('device', 0)}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 加载测试数据
    test_dataset = get_dataset(config['test_data'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    
    # 评估
    predictions = []
    references = []
    
    print(f"开始评估，共{len(test_loader)}个batch...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 移动数据到设备
            eeg = batch['eeg'].to(device)                    # (B, L, 840)
            mask = batch['mask'].to(device)                  # (B, L)
            target_texts = batch['target_text']
            
            # 添加位置编码
            eeg_with_pos = model.pos_embed_e(eeg)
            
            # EEG encoder
            mask_invert = (1 - mask).bool()
            eeg_embeddings = model.e_branch(
                eeg_with_pos,
                src_key_padding_mask=mask_invert
            )
            
            # 投影
            eeg_embeddings = model.act(model.fc_eeg(eeg_embeddings))
            
            # Unified branch
            eeg_embeddings = model.unify_branch(
                eeg_embeddings,
                src_key_padding_mask=mask_invert,
                modality='e'
            )
            
            # 生成
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=eeg_embeddings)
            
            output_ids = model.t_branch.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=mask,
                max_length=config.get('max_length', 64),
                num_beams=config.get('num_beams', 1),
                early_stopping=True,
                do_sample=False,
            )
            
            pred_texts = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
            
            predictions.extend(pred_texts)
            references.extend(target_texts)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1}/{len(test_loader)} batches")
    
    # 计算指标
    print("\n计算评估指标...")
    
    rouge_scores = compute_rouge(predictions, references)
    wer_score = compute_wer(predictions, references)
    
    metrics = {
        'rouge': rouge_scores,
        'wer': wer_score,
        'num_samples': len(predictions)
    }
    
    print("\n评估结果:")
    print(f"ROUGE-1: {rouge_scores.get('rouge1', 'N/A')}")
    print(f"ROUGE-2: {rouge_scores.get('rouge2', 'N/A')}")
    print(f"ROUGE-L: {rouge_scores.get('rougeL', 'N/A')}")
    print(f"WER: {wer_score}")
    
    # 保存结果
    if 'output_path' in config:
        import json
        with open(config['output_path'], 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n结果已保存到: {config['output_path']}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CET-MAE EEG-to-Text评估脚本"
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    evaluate(args.config)
```

#### 3.3.2 评估配置文件示例

```yaml
# models/CET-MAE/config/eval_benchmark.yaml

# 模型配置
embed_dim: 1024
eeg_dim: 840
multi_heads: 8
feedforward_dim: 2048
trans_layers: 6
decoder_embed_dim: 840
pretrain_path: "./models/huggingface/bart-large"
device: 0

# 检查点路径
checkpoint: "./checkpoints/cet_mae_best.pt"

# 测试数据
test_data: "./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle"

# 评估配置
batch_size: 32
max_length: 64
num_beams: 1

# 输出
output_path: "./results/cet_mae_eval_results.json"
```

### 3.4 噪声测评实验（新增）

#### 3.4.1 噪声测评概述

噪声测评是验证EEG-to-Text模型是否真正利用EEG信号的关键实验。通过将真实EEG替换为随机噪声，对比模型性能差异：

- **如果模型性能显著下降**：说明模型确实依赖EEG信号
- **如果模型性能无明显变化**：说明模型可能只是记忆训练标签或依赖语言先验

#### 3.4.2 噪声测评实现方案

**参考现有实现**：
- **GLIM**: `models/GLIM-main/data/datamodule.py` (第265-278行)
- **EEG-To-Text**: `models/EEG-To-Text-main/data.py` (第122-129行)

**统一实现设计**：

```python
# benchmark_eval/data_processing/dataset.py

class UnifiedDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        phase: str = "test", 
        noise_mode: bool = False,
        noise_type: str = "gaussian",  # "gaussian" | "uniform"
        noise_seed: int = 42,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ):
        """
        Args:
            noise_mode: 是否启用噪声模式
            noise_type: 噪声类型
            noise_seed: 随机种子（保证可复现）
            noise_mean: 高斯噪声均值
            noise_std: 高斯噪声标准差
        """
        # ... 现有代码 ...
        self.noise_mode = noise_mode
        self.noise_type = noise_type
        self.noise_seed = noise_seed
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        if self.noise_mode:
            # 生成与真实EEG同shape的随机噪声
            eeg_dict = self._generate_noise_eeg(sample, idx)
        else:
            eeg_dict = {
                "eeg": sample["eeg_normalized_1d"],
                "eeg_cet_mae": sample.get("eeg_cet_mae"),
                "eeg_eeg2text": sample.get("eeg_eeg2text"),
                "mask": sample["mask"],
                "mask_cet_mae": sample.get("mask_cet_mae"),
            }
        
        return {
            **eeg_dict,
            "input_text": sample["input_text"],
            "reference_text": sample["reference_text"],
            "meta": sample["meta"],
        }
    
    def _generate_noise_eeg(self, sample: Dict, idx: int) -> Dict[str, np.ndarray]:
        """生成噪声EEG数据"""
        rng = np.random.default_rng(self.noise_seed + idx)
        
        result = {}
        
        # 1. 默认格式 (max_len, 840)
        if "eeg_normalized_1d" in sample:
            shape = sample["eeg_normalized_1d"].shape
            if self.noise_type == "gaussian":
                result["eeg"] = rng.normal(
                    self.noise_mean, self.noise_std, shape
                ).astype(np.float32)
            else:  # uniform
                result["eeg"] = rng.uniform(
                    -self.noise_std, self.noise_std, shape
                ).astype(np.float32)
            # 噪声模式下mask全1
            result["mask"] = np.ones(shape[0], dtype=np.float32)
        
        # 2. CET-MAE格式
        if "eeg_cet_mae" in sample:
            shape = sample["eeg_cet_mae"].shape
            result["eeg_cet_mae"] = rng.normal(
                self.noise_mean, self.noise_std, shape
            ).astype(np.float32)
            result["mask_cet_mae"] = np.ones(shape[0], dtype=np.float32)
        
        # 3. EEG2Text格式 (24000, 105)
        if "eeg_eeg2text" in sample:
            shape = sample["eeg_eeg2text"].shape
            result["eeg_eeg2text"] = rng.normal(
                self.noise_mean, self.noise_std, shape
            ).astype(np.float32)
            result["mask_eeg2text"] = np.ones(shape[0], dtype=np.float32)
        
        return result
```

#### 3.4.3 噪声评估流程

```python
# benchmark_eval/evaluation/eval_runner.py

class EvaluationRunner:
    def run_with_noise_experiment(
        self,
        model: BenchmarkModelWrapper,
        dataset: UnifiedDataset,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行正常评估和噪声对照实验"""
        
        results = {}
        
        # 1. 正常评估
        logger.info("=" * 60)
        logger.info("Running NORMAL evaluation...")
        logger.info("=" * 60)
        normal_results = self.run(model, dataset, config)
        results["normal"] = normal_results
        
        # 2. 噪声评估（如果启用）
        if config.get("control_experiments", {}).get("run_noise_experiment", False):
            logger.info("=" * 60)
            logger.info("Running NOISE evaluation...")
            logger.info("=" * 60)
            
            # 创建噪声数据集
            noise_dataset = UnifiedDataset(
                data_path=dataset.data_path,
                phase=dataset.phase,
                noise_mode=True,
                noise_type=config["control_experiments"].get("noise_type", "gaussian"),
                noise_seed=config["control_experiments"].get("noise_seed", 42),
                noise_mean=config["control_experiments"].get("noise_mean", 0.0),
                noise_std=config["control_experiments"].get("noise_std", 1.0),
            )
            
            noise_results = self.run(model, noise_dataset, config)
            results["noise"] = noise_results
            
            # 3. 对比分析
            comparison = self._compare_normal_vs_noise(normal_results, noise_results)
            results["comparison"] = comparison
            
            # 4. 输出关键结论
            self._print_noise_experiment_summary(comparison)
        
        return results
    
    def _compare_normal_vs_noise(
        self, 
        normal: Dict[str, float], 
        noise: Dict[str, float]
    ) -> Dict[str, Any]:
        """对比正常和噪声实验结果"""
        comparison = {}
        
        metrics_to_compare = ["bleu_1", "bleu_2", "bleu_4", "rouge_l", "wer"]
        
        for metric in metrics_to_compare:
            if metric in normal and metric in noise:
                normal_val = normal[metric]
                noise_val = noise[metric]
                
                # 计算相对下降
                if normal_val != 0:
                    relative_drop = (normal_val - noise_val) / normal_val * 100
                else:
                    relative_drop = 0.0
                
                comparison[metric] = {
                    "normal": normal_val,
                    "noise": noise_val,
                    "absolute_diff": normal_val - noise_val,
                    "relative_drop_%": relative_drop,
                }
        
        # 总体结论
        avg_drop = np.mean([
            v["relative_drop_%"] 
            for v in comparison.values() 
            if "relative_drop_%" in v
        ])
        
        comparison["summary"] = {
            "average_relative_drop_%": avg_drop,
            "is_eeg_dependent": avg_drop > 10.0,  # 下降超过10%认为依赖EEG
            "conclusion": (
                "Model appears to use EEG signal" if avg_drop > 10.0 
                else "WARNING: Model may not be using EEG signal effectively"
            ),
        }
        
        return comparison
    
    def _print_noise_experiment_summary(self, comparison: Dict[str, Any]):
        """打印噪声实验总结"""
        logger.info("\n" + "=" * 60)
        logger.info("NOISE EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        summary = comparison.get("summary", {})
        logger.info(f"Average relative drop: {summary.get('average_relative_drop_%', 0):.2f}%")
        logger.info(f"Conclusion: {summary.get('conclusion', 'N/A')}")
        
        logger.info("\nDetailed metrics:")
        for metric, values in comparison.items():
            if metric != "summary" and isinstance(values, dict):
                logger.info(
                    f"  {metric}: "
                    f"normal={values.get('normal', 0):.4f}, "
                    f"noise={values.get('noise', 0):.4f}, "
                    f"drop={values.get('relative_drop_%', 0):.2f}%"
                )
```

#### 3.4.4 配置文件更新

```yaml
# benchmark_eval/config/eval_config.yaml

# 对照实验配置
control_experiments:
  # 是否运行噪声 EEG 实验
  run_noise_experiment: true
  
  # 噪声类型: "gaussian" | "uniform"
  noise_type: "gaussian"
  
  # 噪声随机种子（保证可复现）
  noise_seed: 42
  
  # 高斯噪声参数
  noise_mean: 0.0
  noise_std: 1.0
  
  # 是否运行 EEG-文本配对打乱实验
  run_shuffle_experiment: false
  shuffle_seed: 42
```

#### 3.4.5 命令行使用方式

```bash
# 1. 运行正常评估
python -m benchmark_eval.evaluation.eval_runner \
    --model eeg_to_text \
    --data data/unified_dataset.pkl \
    --output results/normal_eval.json

# 2. 运行噪声对照实验
python -m benchmark_eval.evaluation.eval_runner \
    --model eeg_to_text \
    --data data/unified_dataset.pkl \
    --noise_experiment \
    --noise_seed 42 \
    --output results/noise_eval.json

# 3. 使用配置文件运行完整实验
python -m benchmark_eval.evaluation.eval_runner \
    --config benchmark_eval/config/eval_config.yaml \
    --model eeg_to_text \
    --output results/full_eval.json
```

#### 3.4.6 结果解读指南

| 相对下降幅度 | 结论 | 建议 |
|-------------|------|------|
| > 30% | 模型强烈依赖EEG信号 | 模型工作正常，EEG编码有效 |
| 10% - 30% | 模型部分依赖EEG信号 | 可能需要改进EEG编码器 |
| < 10% | 模型可能不依赖EEG信号 | **警告**：检查模型是否只是记忆标签 |
| 噪声 > 正常 | 异常情况 | 检查数据或模型实现是否有问题 |

#### 3.4.7 现有模型噪声测评参考

根据文献和现有代码：

| 模型 | 噪声实现位置 | 噪声类型 | 关键发现 |
|------|-------------|---------|---------|
| **GLIM** | `data/datamodule.py:265-278` | 高斯噪声 | README提到噪声测试表现良好，说明EEG依赖性强 |
| **EEG-To-Text** | `data.py:122-129` | 标准正态 | 论文发现噪声和真实EEG表现相似，存在严重问题 |
| **EEG2Text** | 未明确实现 | - | 需要验证 |
| **CET-MAE** | 未明确实现 | - | 需要验证 |

### 3.5 验证层优化

#### 3.5.1 数据一致性验证脚本

```python
# benchmark_eval/scripts/validate_data_consistency.py
"""数据一致性验证脚本

用于验证统一数据与原始模型数据的一致性。
"""

import argparse
import pickle
import numpy as np
import torch
from typing import Dict, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_unified_data(data_path: str, task: str, subject: str, sent_idx: int):
    """加载统一数据中的指定样本"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    for sample in data:
        meta = sample.get('meta', {})
        if (meta.get('task') == task and 
            meta.get('subject') == subject and 
            meta.get('sentence_index') == sent_idx):
            return sample
    
    return None


def load_cet_mae_original(
    pickle_path: str,
    task: str,
    subject: str,
    sent_idx: int
) -> Dict:
    """加载CET-MAE原始pickle数据"""
    with open(pickle_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    sent_list = dataset_dict.get(subject, [])
    if sent_idx < len(sent_list):
        return sent_list[sent_idx]
    return None


def compare_cet_mae(
    unified_data_path: str,
    cet_mae_pickle_path: str,
    task: str = "task2-NR",
    subject: str = "ZAB",
    sent_idx: int = 0,
    tolerance: float = 1e-5
) -> Tuple[bool, Dict]:
    """比较CET-MAE统一数据和原始数据
    
    Returns:
        (是否一致, 详细信息字典)
    """
    print(f"\n=== CET-MAE数据一致性验证 ===")
    print(f"样本: task={task}, subject={subject}, sent_idx={sent_idx}")
    
    # 加载数据
    unified = load_unified_data(unified_data_path, task, subject, sent_idx)
    original = load_cet_mae_original(cet_mae_pickle_path, task, subject, sent_idx)
    
    if unified is None:
        return False, {"error": "统一数据未找到样本"}
    if original is None:
        return False, {"error": "原始数据未找到样本"}
    
    results = {
        "sample_info": {
            "task": task,
            "subject": subject,
            "sent_idx": sent_idx,
            "text": unified.get('input_text', 'N/A')
        },
        "checks": {}
    }
    
    # 检查1: 文本一致性
    original_text = original.get('content', '')
    unified_text = unified.get('input_text', '')
    text_match = (original_text == unified_text)
    results["checks"]["text"] = {
        "match": text_match,
        "original": original_text[:50] + "..." if len(original_text) > 50 else original_text,
        "unified": unified_text[:50] + "..." if len(unified_text) > 50 else unified_text
    }
    
    # 检查2: eeg_cet_mae vs 原始eeg
    if 'eeg_cet_mae' in unified and 'eeg' in original:
        unified_eeg = unified['eeg_cet_mae']
        original_eeg = original['eeg'].numpy() if torch.is_tensor(original['eeg']) else original['eeg']
        
        diff = np.abs(unified_eeg - original_eeg).max()
        results["checks"]["eeg_cet_mae"] = {
            "match": diff < tolerance,
            "max_diff": float(diff),
            "tolerance": tolerance,
            "shape_unified": unified_eeg.shape,
            "shape_original": original_eeg.shape
        }
    else:
        results["checks"]["eeg_cet_mae"] = {
            "match": False,
            "error": "缺少eeg_cet_mae或原始eeg字段"
        }
    
    # 检查3: mask一致性
    if 'mask_cet_mae' in unified and 'mask' in original:
        unified_mask = np.array(unified['mask_cet_mae'])
        original_mask = original['mask'].numpy() if torch.is_tensor(original['mask']) else np.array(original['mask'])
        
        mask_match = np.allclose(unified_mask, original_mask)
        results["checks"]["mask"] = {
            "match": bool(mask_match),
            "sum_unified": int(unified_mask.sum()),
            "sum_original": int(original_mask.sum())
        }
    
    # 总体结果
    all_match = all(check.get("match", False) for check in results["checks"].values())
    results["overall_match"] = all_match
    
    return all_match, results


def compare_eeg2text(
    unified_data_path: str,
    spectro_pickle_path: str,
    task: str = "task2-NR",
    subject: str = "ZAB",
    sent_idx: int = 0,
    tolerance: float = 1e-3
) -> Tuple[bool, Dict]:
    """比较EEG2Text统一数据和原始spectro数据"""
    print(f"\n=== EEG2Text数据一致性验证 ===")
    print(f"样本: task={task}, subject={subject}, sent_idx={sent_idx}")
    
    # 加载统一数据
    unified = load_unified_data(unified_data_path, task, subject, sent_idx)
    
    # 加载原始spectro数据
    with open(spectro_pickle_path, 'rb') as f:
        spectro_dict = pickle.load(f)
    
    sent_list = spectro_dict.get(subject, [])
    original = sent_list[sent_idx] if sent_idx < len(sent_list) else None
    
    if unified is None:
        return False, {"error": "统一数据未找到样本"}
    if original is None:
        return False, {"error": "原始数据未找到样本"}
    
    results = {
        "sample_info": {
            "task": task,
            "subject": subject,
            "sent_idx": sent_idx
        },
        "checks": {}
    }
    
    # 检查: eeg_eeg2text vs rawData
    if 'eeg_eeg2text' in unified and 'sentence_level_EEG' in original:
        unified_eeg = unified['eeg_eeg2text']
        raw_data = original['sentence_level_EEG'].get('rawData')
        
        if raw_data is not None:
            original_eeg = raw_data.numpy() if torch.is_tensor(raw_data) else raw_data
            original_eeg = original_eeg.T  # 转置为 (T, 105)
            
            # 截取/填充到相同长度
            min_len = min(unified_eeg.shape[0], original_eeg.shape[0])
            unified_eeg_crop = unified_eeg[:min_len]
            original_eeg_crop = original_eeg[:min_len]
            
            # 计算相关性（因为可能有缩放差异）
            corr = np.corrcoef(
                unified_eeg_crop.flatten(),
                original_eeg_crop.flatten()
            )[0, 1]
            
            results["checks"]["eeg_eeg2text"] = {
                "correlation": float(corr),
                "match": corr > 0.99,
                "shape_unified": unified_eeg.shape,
                "shape_original": original_eeg.shape
            }
        else:
            results["checks"]["eeg_eeg2text"] = {
                "match": False,
                "error": "原始数据缺少rawData"
            }
    else:
        results["checks"]["eeg_eeg2text"] = {
            "match": False,
            "error": "缺少eeg_eeg2text或原始数据字段"
        }
    
    all_match = all(check.get("match", False) for check in results["checks"].values())
    results["overall_match"] = all_match
    
    return all_match, results


def main():
    parser = argparse.ArgumentParser(description="验证统一数据与原始模型数据的一致性")
    parser.add_argument("--unified-data", required=True, help="统一数据pickle路径")
    parser.add_argument("--cet-mae-pickle", help="CET-MAE原始pickle路径")
    parser.add_argument("--eeg2text-spectro", help="EEG2Text原始spectro pickle路径")
    parser.add_argument("--task", default="task2-NR")
    parser.add_argument("--subject", default="ZAB")
    parser.add_argument("--sent-idx", type=int, default=0)
    parser.add_argument("--output", help="输出结果JSON路径")
    
    args = parser.parse_args()
    
    all_results = {}
    
    # 验证CET-MAE
    if args.cet_mae_pickle:
        match, results = compare_cet_mae(
            args.unified_data,
            args.cet_mae_pickle,
            args.task,
            args.subject,
            args.sent_idx
        )
        all_results["cet_mae"] = results
        
        print(f"\nCET-MAE验证结果: {'通过' if match else '失败'}")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✓" if check_result.get("match") else "✗"
            print(f"  {status} {check_name}: {check_result}")
    
    # 验证EEG2Text
    if args.eeg2text_spectro:
        match, results = compare_eeg2text(
            args.unified_data,
            args.eeg2text_spectro,
            args.task,
            args.subject,
            args.sent_idx
        )
        all_results["eeg2text"] = results
        
        print(f"\nEEG2Text验证结果: {'通过' if match else '失败'}")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✓" if check_result.get("match") else "✗"
            print(f"  {status} {check_name}: {check_result}")
    
    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n详细结果已保存到: {args.output}")
    
    # 总体结果
    overall = all(r.get("overall_match", False) for r in all_results.values())
    print(f"\n{'='*50}")
    print(f"总体验证结果: {'全部通过' if overall else '存在失败'}")
    print(f"{'='*50}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    exit(main())
```

#### 3.4.2 使用示例

```bash
# 验证CET-MAE数据一致性
python benchmark_eval/scripts/validate_data_consistency.py \
    --unified-data data/unified_dataset.pkl \
    --cet-mae-pickle models/CET-MAE/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle \
    --task task2-NR \
    --subject ZAB \
    --sent-idx 0 \
    --output validation_results.json

# 验证EEG2Text数据一致性
python benchmark_eval/scripts/validate_data_consistency.py \
    --unified-data data/unified_dataset.pkl \
    --eeg2text-spectro models/EEG2Text-main/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle \
    --task task2-NR \
    --subject ZAB \
    --sent-idx 0

# 同时验证两者
python benchmark_eval/scripts/validate_data_consistency.py \
    --unified-data data/unified_dataset.pkl \
    --cet-mae-pickle models/CET-MAE/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle \
    --eeg2text-spectro models/EEG2Text-main/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle \
    --output validation_results.json
```

---

## 4. 实施计划

### 4.1 第一阶段：数据层修复（优先级：P0）

**目标**: 修复统一数据生成，支持CET-MAE和EEG2Text格式，增加噪声模式

**任务清单**:
- [ ] 修改`build_unified_dataset.py`，增加`eeg_cet_mae`和`eeg_eeg2text`字段
- [ ] 修改`dataset.py`，增加噪声模式支持（`noise_mode`, `noise_type`, `noise_seed`）
- [ ] 确保归一化方式与原始模型完全一致
- [ ] 重新生成统一数据
- [ ] 运行验证脚本，确保数据一致性

**验收标准**:
- CET-MAE数据max diff < 1e-5
- EEG2Text数据相关系数 > 0.99
- 噪声模式能生成与真实EEG同shape的随机数据

**预计时间**: 2-3天

### 4.2 第二阶段：评估流程优化（优先级：P0）

**目标**: 简化Wrapper，增加噪声测评流程

**任务清单**:
- [ ] 重构`CETMAEWrapper`，删除复杂字段检查，直接使用`eeg_cet_mae`
- [ ] 重构`EEG2TextWrapper`，删除spectro pickle加载，使用`eeg_eeg2text`
- [ ] 更新`Dataset.__getitem__`，返回所有格式
- [ ] 修改`eval_runner.py`，增加`run_with_noise_experiment`方法
- [ ] 实现噪声vs正常结果对比分析
- [ ] 测试wrapper功能和噪声实验流程

**验收标准**:
- Wrapper代码行数减少50%以上
- 生成结果与原始脚本一致（BLEU差异<1%）
- 噪声实验能正常运行并输出对比报告
- 噪声实验结果与论文报道一致

**预计时间**: 2-3天

### 4.3 第三阶段：关键模型噪声测评验证（优先级：P0）

**目标**: 验证主要模型的噪声测评结果，确保与论文报道一致

**任务清单**:
- [ ] 对EEG-To-Text模型运行噪声测评（预期：噪声≈真实，下降<10%）
- [ ] 对GLIM模型运行噪声测评（预期：噪声表现差，下降>30%）
- [ ] 对CET-MAE模型运行噪声测评（验证其EEG依赖性）
- [ ] 对EEG2Text模型运行噪声测评（验证其EEG依赖性）
- [ ] 记录并对比各模型的噪声测评结果

**验收标准**:
- EEG-To-Text: 噪声vs真实差异<10%（验证论文发现）
- GLIM: 噪声性能显著下降>30%（验证其EEG依赖性）
- 生成各模型的噪声测评报告

**预计时间**: 1-2天

### 4.5 第五阶段：CET-MAE评估脚本（优先级：P1）

**目标**: 创建CET-MAE独立评估脚本

**任务清单**:
- [ ] 创建`eval_decoding_eeg_2_text_cet_mae.py`
- [ ] 创建评估配置文件
- [ ] 测试独立评估流程
- [ ] 对比benchmark评估结果

**验收标准**:
- 独立评估脚本可正常运行
- 结果与benchmark评估一致

**预计时间**: 1天

### 4.6 第六阶段：验证与文档（优先级：P1）

**目标**: 完善验证机制和文档

**任务清单**:
- [ ] 完善`validate_data_consistency.py`
- [ ] 编写优化后的使用文档
- [ ] 更新README
- [ ] 创建CI检查（可选）

**验收标准**:
- 验证脚本可检测数据不一致
- 文档清晰完整

**预计时间**: 1-2天

---

## 5. 风险评估与缓解

### 5.1 风险1: 数据重新生成耗时较长

**风险**: 重新生成统一数据可能需要数小时

**缓解**:
- 先在单任务（task2-NR）上测试
- 使用多进程加速数据生成
- 保留旧数据作为备份

### 5.2 风险2: 归一化方式难以完全对齐

**风险**: 不同模型的归一化细节可能有差异

**缓解**:
- 仔细对比原始代码的归一化实现
- 使用验证脚本检测差异
- 必要时在wrapper中做微调

### 5.3 风险3: 模型checkpoint兼容性问题

**风险**: 修改后的wrapper可能无法加载旧checkpoint

**缓解**:
- 保持模型结构不变，只修改数据输入
- 测试checkpoint加载
- 准备重新训练方案

---

## 6. 附录

### 6.1 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `benchmark_eval/data_processing/build_unified_dataset.py` | 修改 | 增加多格式数据生成 |
| `benchmark_eval/data_processing/dataset.py` | 修改 | 增加噪声模式支持 |
| `benchmark_eval/wrappers/cet_mae_wrapper.py` | 重构 | 简化，直接使用eeg_cet_mae |
| `benchmark_eval/wrappers/eeg2text_wrapper.py` | 重构 | 简化，删除spectro加载 |
| `benchmark_eval/evaluation/eval_runner.py` | 修改 | 增加噪声实验流程 |
| `models/CET-MAE/eval_decoding_eeg_2_text_cet_mae.py` | 新增 | 独立评估脚本 |
| `benchmark_eval/scripts/validate_data_consistency.py` | 新增 | 数据一致性验证 |
| `docs/benchmark_optimization_spec.md` | 新增 | 本文档 |

### 6.2 测试检查清单

- [ ] 统一数据生成成功
- [ ] CET-MAE数据一致性验证通过
- [ ] EEG2Text数据一致性验证通过
- [ ] CETMAEWrapper生成测试通过
- [ ] EEG2TextWrapper生成测试通过
- [ ] Benchmark端到端评估通过
- [ ] 指标与原始模型一致

### 6.3 相关资源

- CET-MAE论文: https://aclanthology.org/2024.acl-long.393/
- EEG2Text论文: [IEEE BigData 2024]
- EEG-To-Text论文: [NeurIPS 2023]
- ZuCo数据集: https://osf.io/q3zws/

---

**文档结束**
