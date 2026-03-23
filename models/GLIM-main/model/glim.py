# 导入必要的库
import os  # 操作系统接口
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口
import torch.distributed as dist  # PyTorch分布式训练
import pandas as pd  # 数据处理库
import lightning as L  # PyTorch Lightning框架
from typing import Literal  # 类型提示，用于限制字面量类型
from torch import Tensor  # 张量类型
from copy import deepcopy  # 深拷贝
from collections import defaultdict  # 默认字典
from torch.utils.data import default_collate  # 数据批处理默认整理函数
# 导入torchmetrics中的分类和文本评估指标
from torchmetrics.functional.classification import multiclass_accuracy, binary_accuracy
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
from lightning.pytorch.utilities import rank_zero_only  # Lightning中只在rank 0进程执行的装饰器
# 导入transformers库中的预训练模型和工具
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput  # transformers模型输出基类

# 导入自定义模块
from .modules import PromptEmbedder, EEGEncoder, Aligner


class GLIM(L.LightningModule):  # 继承PyTorch Lightning的LightningModule
    
    # 定义支持的文本模型类型（使用Literal限制选择范围）
    SUPPORTED_TEXT_MODELS = Literal["google/flan-t5-xl", "google/flan-t5-large", 
                                    "facebook/bart-large-cnn", "jbochi/madlad400-3b-mt",]

    def __init__(self, 
                 input_eeg_len = 1280,  # 输入EEG信号长度
                 hidden_eeg_len = 96,   # 隐藏层EEG信号长度
                 input_text_len = 96,   # 输入文本长度
                 tgt_text_len = 64,     # 目标文本长度
                 input_dim = 128,       # 输入维度
                 hidden_dim = 128,      # 隐藏层维度
                 embed_dim = 1024,      # 嵌入维度
                 text_model_id: SUPPORTED_TEXT_MODELS = "google/flan-t5-large",  # 文本模型ID
                 prompt_nums: tuple[int] = (3, 3, 31),  # 提示数量元组（任务、数据集、主题）
                 prompt_dropout_probs: tuple[float] = (0.0, 0.0, 0.0),  # 提示dropout概率
                 evaluate_prompt_embed: Literal['zero', 'sum', 'mean', 'src'] = 'src',  # 评估提示嵌入方式
                 n_in_blocks: int = 6,     # 输入块数量
                 n_out_blocks: int = 6,    # 输出块数量
                 in_temporal_modulate: bool = True,  # 是否使用时间调制
                 out_is_causal: bool = True,         # 输出是否为因果的
                 prompt_tuning_len: bool = 0,        # 提示调优长度
                 num_heads = 8,            # 注意力头数
                 mlp_ratio = 4,            # MLP比率
                 dropout = 0.0,            # Dropout概率
                 clip_loss_weight = 0.5,   # CLIP损失权重
                 commitment_loss_weight = 0.0,  # 承诺损失权重
                 commitment_loss_key: Literal['mse','kl_div']= 'mse',  # 承诺损失类型
                 use_y_mask = False,       # 是否使用y掩码
                 bsz_train = 48,           # 训练批大小
                 bsz_val = 24,             # 验证批大小
                 lr = 1e-5,                # 学习率
                 weight_decay = 0,         # 权重衰减
                 full_val_interval = 10,   # 完整验证间隔
                 bs_retrieval = 24,        # 检索批大小
                ):
        
        super().__init__()  # 调用父类构造函数

        # 保存重要参数为实例变量
        self.input_text_len = input_text_len
        self.tgt_text_len = tgt_text_len
        self.prompt_tuning_len = prompt_tuning_len
        self.eval_pembed = evaluate_prompt_embed

        # 损失函数权重参数
        self.λ = clip_loss_weight
        self.ε = commitment_loss_weight
        
        # 训练相关参数
        self.lr = lr
        self.weight_decay = weight_decay
        self.bsz_train = bsz_train
        self.bsz_val = bsz_val
        self.full_val_interval = full_val_interval
        self.bsz_retrieval = bs_retrieval
        
        # 提示键定义（用于不同任务、数据集、主题的提示）
        self.prompt_keys = {
            # 'task': ['<Normal Reading>'] + ['<Relation Extraction>', '<Sentiment Classification>',],
            'task': ['<UNK>'] + ['<NR>', '<TSR>'],  # 任务类型：未知、正常阅读、时间序列阅读
            # 'task': ['<NR>'] + ['<SC>', '<RE>'],
            'dataset': ['<UNK>'] + ['ZuCo1', 'ZuCo2',],  # 数据集类型
            'subject': ['<UNK>'] + ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN',   # 主题ID列表
                                    'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
                                    'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 
                                    'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 
                                    'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL',],
            }
        
        # 用于日志记录和分类的标签定义
        self.raw_task_keys = ['task1', 'task2', 'task3']  # 原始任务键
        self.sentiment_labels = ['negative', 'neutral', 'positive']  # 情感标签
        # 关系标签
        self.relation_labels = ['awarding', 'education', 'employment',
                                'foundation', 'job title', 'nationality', 
                                'political affiliation','visit', 'marriage']
        
        # 初始化模型组件
        # 提示嵌入器
        self.p_embedder = PromptEmbedder(input_dim, 
                                         prompt_nums, prompt_dropout_probs, self.prompt_keys)
        # self.task_embed_proj = nn.Linear(input_dim, embed_dim * prompt_tuning_len)

        # EEG编码器
        self.eeg_encoder = EEGEncoder(input_eeg_len, hidden_eeg_len, input_dim, hidden_dim, 
                                      prompt_tuning_len, n_in_blocks, n_out_blocks, 
                                      in_temporal_modulate, out_is_causal, 
                                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        
        # 对齐器（用于EEG和文本的对齐）
        self.aligner = Aligner(hidden_dim, embed_dim, num_heads, dropout, commitment_loss_key, use_y_mask) 
        self.use_y_mask = use_y_mask
        self.text_model_id = text_model_id
        self.embed_dim = embed_dim

        # 保存超参数用于日志记录
        self.save_hyperparameters(logger=True)

    def setup(self, stage):  # Lightning生命周期方法，在训练/验证/测试开始前调用
        # 禁用tokenizers并行化以避免警告
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)
        # 初始化文本生成模型，设置为不需要梯度
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            self.text_model_id, device_map = self.device,
            torch_dtype = torch.bfloat16, # FIXME
            ).requires_grad_(False)
        # 确保嵌入维度与模型配置一致
        assert self.embed_dim == self.text_model.config.d_model

    def add_prompt(self, on:Literal['task','dataset','subject'], prompt):  # 添加提示的方法（待实现）
        # TODO: on x -> self.x_prompts += prompt
        # self.p_embedder.
        # maybe self.prompt_nums
        pass

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:  # 保存检查点时的回调
        # 从检查点中移除text_model的参数（因为它们是冻结的）
        for key in deepcopy(list(checkpoint['state_dict'].keys())):
            if 'text_model' in key: 
                checkpoint['state_dict'].pop(key)
        
    def configure_optimizers(self):  # 配置优化器
        # 只优化需要梯度的参数
        params = [p for p in self.parameters() if p.requires_grad == True]
        opt = torch.optim.Adam(params, 
                               lr = self.lr,
                               weight_decay = self.weight_decay)
        # 注释掉的学习率调度器
        # lr_scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, 
        #                                                num_training_steps=self.trainer.max_epochs)
        # return {"optimizer": opt,
        #         "lr_scheduler": lr_scheduler}
        return opt
    
    def tokenize(self, texts: list[str], max_length: int) -> tuple[torch.Tensor]:  # 文本tokenize方法
        # 使用tokenizer处理文本列表
        inputs = self.tokenizer(texts, max_length=max_length, padding='max_length', 
                                truncation=True, return_tensors="pt") 
        # 获取token ids和注意力掩码，并移到设备上
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        return ids, mask
    
    def encode_labels(self, labels:list[str], ignore_idx=-1):  # 编码标签为数字ID
        label_ids = []
        for label in labels:
            if label in self.relation_labels:  # 如果是关系标签
                label_id = self.relation_labels.index(label)
            elif label in self.sentiment_labels:  # 如果是情感标签
                label_id = self.sentiment_labels.index(label)
            else:  # 其他情况（如'nan'）
                assert label == 'nan'
                label_id = ignore_idx
            label_ids.append(label_id)
        # 转换为tensor并移到设备上
        label_ids = torch.tensor(label_ids, dtype=torch.int, device=self.device)
        return label_ids

    def get_inputs(self, batch):  # 从批数据中提取输入
        # 提取EEG相关数据
        eeg = batch['eeg']        # (n, l, c) EEG信号
        eeg_mask = batch['mask']  # (n, l) EEG掩码，1表示未掩码，0表示掩码
        prompts = batch['prompt']  # 提示信息：[tuple('task'), tuple('dataset'), tuple('subject')]
        
        # 提取文本数据
        input_text = batch['input text']        # 输入文本列表
        tgt_text = batch['target text']         # 目标文本列表
        
        # 用于日志记录和计算指标的标签
        sentiment_label = batch['sentiment label']      # 情感标签列表
        relation_label = batch['relation label']      # 关系标签列表
        raw_task_key = batch['raw task key']    # 原始任务键列表
        
        # 将原始任务键转换为数字ID
        raw_task_ids = torch.tensor([self.raw_task_keys.index(key) for key in raw_task_key],
                                    dtype=torch.int,device=self.device)
        
        # 其他文本相关数据
        raw_input_text = batch['raw input text']  # 原始输入文本列表
        all_target_texts = batch['all target texts']    # 所有目标文本：[tuple('v0'), tuple('v1'), ...]
        
        # 编码提示信息
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)  # (n, 3)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)  # (n, c, 3) --> (n, c)
        
        # tokenize文本
        input_ids, input_mask = self.tokenize(input_text, self.input_text_len-self.prompt_tuning_len)
        tgt_ids, _ = self.tokenize(tgt_text, self.tgt_text_len)
        
        # 编码标签
        sentiment_ids = self.encode_labels(sentiment_label)
        relation_ids = self.encode_labels(relation_label)

        # 返回处理后的所有输入
        return (eeg, eeg_mask, prompt_embed, 
                input_ids, input_mask, tgt_ids,
                prompt_ids, raw_task_ids, 
                sentiment_ids, relation_ids,
                tgt_text, raw_input_text, all_target_texts)
    
    def encode_text(self, src_ids, src_mask):  # 编码文本为隐藏状态
        text_encoder = self.text_model.get_encoder()  # 获取文本编码器
        with torch.no_grad():  # 不计算梯度
            outputs = text_encoder(input_ids = src_ids, 
                                    attention_mask = src_mask, 
                                    return_dict = True)
        hidden_mask = src_mask  # 隐藏状态掩码与输入掩码相同

        hidden_states = outputs['last_hidden_state']  # 获取最后一层隐藏状态
        return hidden_states, hidden_mask

    def text_decoder_forward(self, src_embeds, src_mask, tgt_ids):  # 文本解码器前向传播
        # 准备标签，将pad token设为-100（用于损失计算中的忽略）
        labels = tgt_ids.detach().clone()
        labels.masked_fill_(labels == self.text_model.config.pad_token_id, -100)  # 就地操作
        
        # 根据训练状态和配置决定是否使用掩码
        mask = src_mask if (self.use_y_mask and self.training) else None
        
        # 运行文本生成模型
        outputs = self.text_model(encoder_outputs = BaseModelOutput(src_embeds), 
                                  attention_mask = mask,
                                  labels = labels)
        loss = outputs['loss']      # 损失 (1)
        logits = outputs['logits']  # logits (n, l, vocab_sz)
        return loss, logits.detach()
    
    def shared_forward(self, batch):  # 共享的前向传播逻辑
        # 获取输入数据
        (eeg, eeg_mask, prompt_embed, 
         input_text_ids, input_text_mask, target_text_ids, 
         prompt_ids, raw_task_ids, 
         sentiment_ids, relation_ids,
         target_text, raw_input_text, all_target_texts) = self.get_inputs(batch)
     
        # 编码输入文本
        input_text_embeds, hidden_text_mask = self.encode_text(input_text_ids, input_text_mask)
        
        # 编码EEG信号
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)  # TODO: return weights

        # 对齐EEG和文本嵌入
        (loss_clip, logits_clip, loss_commitment, 
         eeg_embeds, eeg_emb, input_text_emb) = self.aligner(eeg_hiddens, input_text_embeds, hidden_text_mask)

        # 文本解码
        loss_lm, logits_lm = self.text_decoder_forward(eeg_embeds, hidden_text_mask, target_text_ids)
        
        # 返回所有计算结果
        return {'loss_commitment': loss_commitment,            # 承诺损失 (1)
                'loss_clip': loss_clip,                        # CLIP损失 (1)
                'loss_lm': loss_lm,                            # 语言模型损失 (1)
                'logits_clip': logits_clip,                    # CLIP logits (n, n)
                'logits_lm': logits_lm,                        # 语言模型logits (n, l, vocab_size)
                'eeg_emb_vector': eeg_emb,                     # EEG嵌入向量 (n, e)
                'text_emb_vector': input_text_emb,             # 文本嵌入向量 (n, e)
                'eeg_embeds': eeg_embeds,                      # EEG嵌入序列 (n, l, e)，用于生成
                # 'input_text_embeds': input_text_embeds,       # 输入文本嵌入 (n, l, e)
                # 'input_text_mask': input_text_mask,           # 输入文本掩码 (n, l)
                
                ### 用于日志记录的数据
                'input_text_ids': input_text_ids,              # 输入文本IDs (n, l)
                'target_text_ids': target_text_ids,            # 目标文本IDs (n, l')
                'prompt_ids': prompt_ids,                      # 提示IDs (n, 3)
                'raw_task_ids': raw_task_ids,                  # 原始任务IDs (n)
                'sentiment_ids': sentiment_ids,    # 情感IDs (n)
                'relation_ids': relation_ids,      # 关系IDs (n)
                
                ### 用于计算指标的数据
                'raw_input_text': raw_input_text,              # 原始输入文本列表
                'all_target_texts': all_target_texts,          # 所有目标文本列表
                }

    def define_metrics(self, metric_keys: list=None) -> None:  # 定义指标的汇总方式
        run = self.logger.experiment
        for key in metric_keys:
            if 'loss' in key:  # 损失指标使用最小值汇总
                run.define_metric(key, summary='min')
            else:  # 其他指标使用最大值汇总
                run.define_metric(key, summary='max')

    def cal_retrieval_metrics(self, logits: torch.Tensor, targets:torch.Tensor=None,
                              strict=False):  # 计算检索指标
        if strict:  # 严格模式：只在指定子集内计算
            assert logits.shape[0] >= self.bsz_retrieval
            logits = logits[:self.bsz_retrieval, :self.bsz_retrieval]  # 截取指定大小
        
        bsz = logits.shape[0]
        # 如果没有提供目标，使用对角线作为目标（即每个样本对应自己）
        targets = torch.arange(bsz, dtype=torch.int, 
                              device=self.device) if targets is None else targets # (n)
        
        probs = torch.softmax(logits, dim=-1)  # 计算概率
        
        # 计算top-k准确率
        acc_top1 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=1)
        acc_top5 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=5)
        acc_top10 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=10)
        
        return {'retrieval_acc_top01': acc_top1,
                'retrieval_acc_top05': acc_top5,
                'retrieval_acc_top10': acc_top10,
                }
    
    def training_step(self, batch, batch_idx):  # 训练步骤
        # 执行共享前向传播
        shared_outputs = self.shared_forward(batch)
        
        # 提取各种损失
        loss_commitment = shared_outputs['loss_commitment']     # 承诺损失 (1)
        loss_clip = shared_outputs['loss_clip']                 # CLIP损失 (1)
        loss_lm = shared_outputs['loss_lm']                     # 语言模型损失 (1)
        
        # 计算总损失（加权组合）
        loss = self.λ * loss_clip + (1-self.λ) * loss_lm + self.ε * loss_commitment
        
        # 组织指标字典
        metrics = {'loss': loss,
                   'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm, 
                #    'learning_rate': self.lr_schedulers().get_last_lr()[0],  # 学习率（已注释）
                    } 

        # 计算检索指标
        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=False)
        metrics.update(retrieval_metrics)

        # 为指标添加'train/'前缀
        metrics = {f'train/{k}': v for k, v in metrics.items()}
        
        # 在第一个epoch的第一个batch定义指标
        if self.current_epoch == 0 and batch_idx == 0:
            self.define_metrics(list(metrics.keys()))
        
        # 记录指标
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_train)
        # `self.log/self.log_dict`会在设备间聚合每个指标
        return loss

    def on_validation_epoch_start(self):  # 验证epoch开始时的回调
        # 只在特定间隔或第一个epoch执行完整验证
        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            # self.val_step_outputs = defaultdict(list)
            self.full_val_step_outputs = []

    def validation_step(self, batch, batch_idx):  # 验证步骤
        # 执行共享前向传播
        shared_outputs = self.shared_forward(batch)
        
        # 提取损失
        loss_commitment = shared_outputs['loss_commitment']     # 承诺损失 (1)
        loss_clip = shared_outputs['loss_clip']                 # CLIP损失 (1)
        loss_lm = shared_outputs['loss_lm']                     # 语言模型损失 (1)
        
        metrics = {'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm,         
                    } 

        # 计算检索指标（允许较小的批大小）
        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=False)  
        metrics.update(retrieval_metrics)
        
        # 添加'val/'前缀
        metrics = ({f'val/{k}':v for k, v in metrics.items()})
        
        # 在第一个epoch的第一个batch定义指标
        if self.current_epoch == 0 and batch_idx == 0:
            self.define_metrics(list(metrics.keys()))
        
        # 记录指标
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_val)

        # 在完整验证间隔执行额外的验证步骤
        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            outputs = self.full_val_step(shared_outputs)
            self.full_val_step_outputs.append(outputs)
            
        
    def full_val_step(self, shared_outputs):  # 完整验证步骤
        bsz = shared_outputs['eeg_embeds'].shape[0]
        
        # 准备需要跨设备聚合的张量
        to_gather_tensors = {'eeg_emb_vector': shared_outputs['eeg_emb_vector'],    # EEG嵌入向量 (n, e)
                             'text_emb_vector': shared_outputs['text_emb_vector'],  # 文本嵌入向量 (n, e)
                             'input_text_ids': shared_outputs['input_text_ids'],    # 输入文本IDs (n, l)
                             'target_text_ids': shared_outputs['target_text_ids'],  # 目标文本IDs (n, l')
                             'prompt_ids': shared_outputs['prompt_ids'],            # 提示IDs (n, 3)
                             'raw_task_ids': shared_outputs['raw_task_ids'],        # 原始任务IDs (n)
                             'sentiment_ids': shared_outputs['sentiment_ids'],  # 情感IDs (n)
                             'relation_ids': shared_outputs['relation_ids'],  # 关系IDs (n)
                             }
        
        # tokenize原始输入文本
        raw_input_ids, _ = self.tokenize(shared_outputs['raw_input_text'], self.input_text_len)   # (n, l)
        
        # 处理所有目标文本
        K = self.trainer.datamodule.n_target_text  # 每个样本的目标文本数量
        all_tgt_text_list = []
        for targets in zip(*shared_outputs['all_target_texts']): # 遍历每个样本
            all_tgt_text_list.extend(list(targets))
        
        # tokenize所有目标文本
        tgt_ids, _ = self.tokenize(all_tgt_text_list, self.tgt_text_len)            # (n*k, l')
        all_tgt_ids = tgt_ids.reshape(bsz, K, -1)                                   # (n, k, l')
        
        # 添加到聚合张量字典
        to_gather_tensors.update({'raw_input_text_ids': raw_input_ids,
                                  'all_target_text_ids': all_tgt_ids,
                                  })

        # 执行生成步骤
        gen_ids_dict = self.generation_step(shared_outputs)
        to_gather_tensors.update(gen_ids_dict)
        return to_gather_tensors

    # @torch.autocast(self.device, dtype=(torch.bfloat16 if self.precision == "bf16-mixed" else torch.half))
    def on_validation_epoch_end(self):  # 验证epoch结束时的回调
        # 只在完整验证间隔处理结果
        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            # 整理验证步骤输出：dict[list[dict[str, tensor]]]
            outputs = default_collate(self.full_val_step_outputs)         # (n_steps, bsz, ...)
            outputs = {k: v.flatten(0,1) for k,v in outputs.items()}      # (n_steps*bsz, ...)
            
            # 如果是分布式训练，聚合所有设备的结果
            if dist.is_initialized():
                outputs = self.all_gather(outputs)                        # (n_devices, n_steps*bsz, ...)
                outputs = {k: v.flatten(0,1) for k,v in outputs.items()}  # (n_devices*n_steps*bsz, ...)
            
            # 只在rank 0进程计算和记录指标
            if self.local_rank == 0:
                with torch.autocast(device_type='cuda', dtype=(torch.bfloat16 if self.trainer.precision == "bf16-mixed" else torch.half)):
                    self.cal_and_log(outputs, prefix='full_val')
            
            # 清空输出列表
            self.full_val_step_outputs.clear()

    def cal_gen_metrics(self, pred_text: list[str], target_texts: list[tuple[str], str], 
                        raw_input_text: list[str], return_more=False) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        """计算生成指标"""
        # 初始化指标列表
        bleu1, bleu2, bleu3, bleu4 = [],[],[],[]
        rouge1_fmeasure, rouge1_precision, rouge1_recall = [],[],[]
        wer = []
        m_dicts = []  # 更详细的指标字典列表
        
        # 遍历每个预测文本、目标文本组和输入文本
        for pred, tgts, input in zip(pred_text, target_texts, raw_input_text):
            # 计算BLEU分数（不同n-gram）
            bleu1.append(bleu_score([pred], [tgts], n_gram=1))
            bleu2.append(bleu_score([pred], [tgts], n_gram=2))
            bleu3.append(bleu_score([pred], [tgts], n_gram=3))
            bleu4.append(bleu_score([pred], [tgts], n_gram=4))
            
            # 计算ROUGE-1分数
            rouge1_dict = rouge_score([pred], [tgts], rouge_keys='rouge1')
            rouge1_fmeasure.append(rouge1_dict['rouge1_fmeasure'])
            rouge1_precision.append(rouge1_dict['rouge1_precision'])
            rouge1_recall.append(rouge1_dict['rouge1_recall'])
            
            # 计算词错误率
            wer.append(word_error_rate([pred], [input]))
            
            # 如果需要更多详细指标
            if return_more:
                # 对每个目标变体计算BLEU1
                bleu1_mtv = {f'BLEU1@MTV{i:02d}': bleu_score([pred], [tgt], n_gram=1)
                            for i, tgt in enumerate(tgts)}
                # 对原始输入计算BLEU1
                bleu1_raw = {'BLEU1@RAW': bleu_score([pred], [input], n_gram=1)}
                
                # 对每个目标变体计算BLEU2
                bleu2_mtv = {f'BLEU2@MTV{i:02d}': bleu_score([pred], [tgt], n_gram=2) 
                            for i, tgt in enumerate(tgts)}
                bleu2_raw = {'BLEU2@RAW': bleu_score([pred], [input], n_gram=2)}
                
                # 对每个目标变体计算ROUGE1
                rouge1_mtv = {f'ROUGE1@MTV{i:02d}': 
                            rouge_score([pred], [tgt], rouge_keys='rouge1')['rouge1_recall']
                            for i, tgt in enumerate(tgts)}
                rouge1_raw = {'ROUGE1@RAW': 
                            rouge_score([pred], [input], rouge_keys='rouge1')['rouge1_recall']}
                
                # 合并所有详细指标
                m_dicts.append({**bleu1_mtv, **bleu1_raw, **bleu2_mtv, **bleu2_raw, **rouge1_mtv, **rouge1_raw})
        
        # 将指标转换为张量（在DDP时需要移到设备上）
        metrics_mean = {'bleu1': torch.stack(bleu1), 
                        'bleu2': torch.stack(bleu2), 
                        'bleu3': torch.stack(bleu3), 
                        'bleu4': torch.stack(bleu4), 
                        'rouge1_fmeasure': torch.stack(rouge1_fmeasure), 
                        'rouge1_precision': torch.stack(rouge1_precision), 
                        'rouge1_recall': torch.stack(rouge1_recall), 
                        'wer': torch.stack(wer), 
                        }
        
        return metrics_mean, m_dicts
    
    def pad_ids(self, ids):  # 填充ID序列到目标长度
        pad_len = self.tgt_text_len - ids.shape[1]  # 计算需要填充的长度
        if pad_len > 0:
            pad_value = self.text_model.config.pad_token_id  # 获取填充token的ID
            ids = F.pad(ids, (0, pad_len), 'constant', pad_value)  # 进行填充
        return ids.int()

    def convert_logits_to_ids(self, logits) -> tuple[list[str], torch.Tensor]:  # 将logits转换为token IDs
        probs = logits.softmax(dim=-1)  # 计算概率 (bs, out_len, vocab_size)
        _, ids = probs.topk(1) # 获取top-1 token IDs (bs, out_len, 1)
        ids = ids.squeeze().int()  # 压缩维度并转换为int
        # text = self.tokenizer.batch_decode(ids, skip_special_tokens=True)  # 可选：解码为文本
        return ids

    def generation_step(self, shared_outputs) -> tuple[dict]:  # 生成步骤
        # 文本生成
        # 注释：在batch=24，4090D*1上：num_beams=2需要3分15秒，4690MB；num_beams=4需要3分30秒，6924MB
        gen_ids = self.text_model.generate(encoder_outputs = BaseModelOutput(shared_outputs['eeg_embeds']), 
                                           num_beams = 2,  # 使用束搜索，束大小为2
                                           min_length = 0, max_length=self.tgt_text_len)
        
        out_ids_dict = {'gen_ids': self.pad_ids(gen_ids)}  # 用于跨设备聚合
        
        # 将teacher forcing的logits转换为token IDs
        tf_ids = self.convert_logits_to_ids(shared_outputs['logits_lm'])
        out_ids_dict.update({'tf_ids': tf_ids})
        return out_ids_dict
    
    def cal_label_embs(self, labels: list[str], template: str=None):  # 计算标签嵌入
        if template:  # 如果提供了模板
            # 将标签插入模板中
            label_sentences = [template.replace("<MASK>", label) for label in labels]
        else:
            label_sentences = labels
        
        # tokenize标签句子
        ids, mask = self.tokenize(label_sentences, 32)
        # 编码为文本嵌入
        embeds, _ = self.encode_text(ids, mask)
        # 通过对齐器获取嵌入向量
        emb_vectors = self.aligner.embed_text(embeds, mask)  # (n, e)
        return emb_vectors
    
    def run_cls(self, eeg_emb_vector, candi_emb_vector):  # 运行分类
        # 归一化EEG嵌入向量
        eeg_norm = eeg_emb_vector / eeg_emb_vector.norm(dim=1, keepdim=True)      # (n, e)
        # 归一化候选嵌入向量
        candi_norm = candi_emb_vector / candi_emb_vector.norm(dim=1, keepdim=True)   # (c, e)
        # 计算相似度并转换为概率
        probs = (eeg_norm @ candi_norm.T).softmax(dim=-1)  # (n, c)
        return probs
    
    def collect_cls_preds(self, probs, target_ids, candidates: list[str], ignore_idx=-1) -> tuple[list, list[dict]]:
        """收集分类预测结果"""
        n = probs.shape[0]  # 样本数
        c = len(candidates)  # 候选标签数
        targets, prob_dicts = [], []
        
        for i in range(n):
            idx = target_ids[i].item()
            # 获取目标标签（如果idx为ignore_idx则为'nan'）
            target = candidates[idx] if idx !=ignore_idx else 'nan'
            targets.append(target)
            
            # 获取top-k概率和对应的候选标签
            topk_probs, indices = probs[i].topk(c)
            prob_dict = {candidates[idx]: prob.item() for prob, idx in zip(topk_probs, indices)}
            prob_dicts.append(prob_dict)
        
        return targets, prob_dicts

    def run_sentiment_cls(self, intermediates: dict, candi_emb_vector: Tensor):  # 运行情感分类
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        c = candi_emb_vector.shape[0]
        assert c == probs.shape[1] 
        
        target_ids = intermediates['sentiment_ids']
        # 计算top-1准确率（忽略index为-1的样本）
        acc_top1 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=1)
        accs = {'sentiment_cls_acc_top01': acc_top1}
        
        # 收集预测标签和概率
        labels, prob_dicts = self.collect_cls_preds(probs, target_ids, self.sentiment_labels)
        return accs, labels, prob_dicts
    
    def run_relation_cls(self, intermediates: dict, candi_emb_vector: Tensor):  # 运行关系分类
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        c = candi_emb_vector.shape[0]
        assert c == probs.shape[1] 
        
        target_ids = intermediates['relation_ids']
        # 计算top-1和top-3准确率
        acc_top1 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=1)
        acc_top3 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=3)
        accs = {'relation_cls_acc_top01': acc_top1,
                'relation_cls_acc_top03': acc_top3}
        
        # 收集预测标签和概率
        labels, prob_dicts = self.collect_cls_preds(probs, target_ids, self.relation_labels)
        return accs, labels, prob_dicts
    
    def run_corpus_cls(self, intermediates: dict, candi_emb_vector: Tensor):  # 运行语料库分类
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        
        # 处理目标ID：将task2(id=2)映射为task1(id=1)，使其成为二分类问题
        target_ids = intermediates['raw_task_ids'].detach().clone()
        target_ids.masked_fill_(target_ids==2, 1)
        
        # 计算二分类准确率
        acc = multiclass_accuracy(probs, target_ids, average='micro', num_classes=2, top_k=1)
        acc_dict = {'corpus_cls_acc': acc}
        return acc_dict

    def cal_and_log(self, outputs, prefix='full_val'):  # 计算并记录指标
        # 预先计算用于分类的标签嵌入
        se_label_embs = self.cal_label_embs(self.sentiment_labels, template="Sentiment classification: It is <MASK>.")
        re_label_embs = self.cal_label_embs(self.relation_labels, template="Relation classification: It is about <MASK>.")
        co_label_embs = self.cal_label_embs(labels=["The topic is about: movie, good or bad", 
                                                    "The topic is about: life experiences, relationship"])
        
        # 计算整体分类指标（微平均）
        se_accs, se_labels, se_prob_dicts = self.run_sentiment_cls(outputs, se_label_embs)
        re_accs, re_labels, re_prob_dicts = self.run_relation_cls(outputs, re_label_embs)
        co_acc = self.run_corpus_cls(outputs, co_label_embs)
        
        # 合并平均指标
        mean_metrics = {**se_accs, **re_accs, **co_acc}
        mean_metrics = {f"{prefix}/mean_{k}":v for k,v in mean_metrics.items()}

        # 计算分组/样本级别指标（生成、分类、检索）
        bsz = outputs['prompt_ids'].shape[0]
        p_keys = self.prompt_keys
        group_dict = defaultdict(list)
        
        # 按任务-数据集-主题-原始任务组织数据
        for i in range(bsz):
            t_id, d_id, s_id = outputs['prompt_ids'][i]
            tds_key = f"{p_keys['task'][t_id]}-{p_keys['dataset'][d_id]}-{p_keys['subject'][s_id]}"
            raw_t_id = outputs['raw_task_ids'][i]
            raw_t_key = f"{self.raw_task_keys[raw_t_id]}"
            group_key = f"{tds_key}-{raw_t_key}"
            group_dict[group_key].append({k: v[i] for k,v in outputs.items()})

        all_rows = []  # 用于创建DataFrame的行列表
        all_group_metrics = {}
        to_mean_metrics = []
        
        # 遍历每个组
        for group_key, intermediates_list_dict in sorted(group_dict.items()):
            t_key, d_key, s_key, raw_t_key = group_key.split('-')
            intermediates = default_collate(intermediates_list_dict)  # 整理组内数据
            n = len(intermediates_list_dict)  # 组内样本数
            
            ### 计算并收集组级别指标
            group_metrics = {}
            if self.current_epoch == 0:  # 第一个epoch记录样本数
                group_metrics.update({'num_samples': n})
            
            # 计算生成指标
            # 解码各种文本
            input_strs = self.tokenizer.batch_decode(intermediates['input_text_ids'], 
                                                     skip_special_tokens=True)
            raw_input_strs = self.tokenizer.batch_decode(intermediates['raw_input_text_ids'], 
                                                     skip_special_tokens=True)
            # 处理所有目标文本
            all_tgt_ids = intermediates['all_target_text_ids'].reshape(-1, self.tgt_text_len)
            all_tgt_strs = self.tokenizer.batch_decode(all_tgt_ids, skip_special_tokens=True)
            k = self.trainer.datamodule.n_target_text
            all_tgt_str_tuples = [tuple(all_tgt_strs[i*k:(i+1)*k]) for i in range(n)]
            
            # 解码生成的文本
            gen_strs = self.tokenizer.batch_decode(intermediates['gen_ids'], skip_special_tokens=True)
            tf_strs = self.tokenizer.batch_decode(intermediates['tf_ids'], skip_special_tokens=True)
            tf_tgt_strs = self.tokenizer.batch_decode(intermediates['target_text_ids'], skip_special_tokens=True)

            # 计算生成指标
            m_gen, mdicts_gen = self.cal_gen_metrics(gen_strs, all_tgt_str_tuples, raw_input_strs, return_more=True)
            m_tf, _ = self.cal_gen_metrics(tf_strs, tf_tgt_strs, raw_input_strs)
            
            # 组织指标名称
            gen_metrics = {f'{k}_gen': v for k,v in m_gen.items()}
            gen_metrics.update({f'{k}_tf': v for k, v in m_tf.items()})
            group_metrics.update({k: v.mean() for k, v in gen_metrics.items()})
            
            # 组内检索准确率（可选）
            # if t_key == '<Normal Reading>': # TODO: 可能随机采样这个子集？
            _, logits = self.aligner.align_emb_vector(intermediates['eeg_emb_vector'],
                                                        intermediates['text_emb_vector'])
            try:
                # 计算检索指标（如果批大小足够）
                retrieval_metrics = self.cal_retrieval_metrics(logits, strict=True)
                group_metrics.update(retrieval_metrics)
            except AssertionError:  # 如果bsz < self.bsz_retrieval则忽略，由`strict`控制
                pass

            # 分类准确率和预测结果
            se_accs, se_labels, se_prob_dicts = self.run_sentiment_cls(intermediates, se_label_embs)
            re_accs, re_labels, re_prob_dicts = self.run_relation_cls(intermediates, re_label_embs)
            co_acc = self.run_corpus_cls(intermediates, co_label_embs)
            group_metrics.update({**se_accs, **re_accs, **co_acc})

            # 添加到所有组指标字典
            all_group_metrics.update({f"{prefix}/{k}-{group_key}":v 
                                      for k,v in group_metrics.items()})
            
            ### 收集样本级别指标用于日志记录
            for i in range(n):
                # 添加到平均指标列表
                to_mean_metrics.append({'BLEU1@MTV': gen_metrics['bleu1_gen'][i], 
                                        'BLEU2@MTV': gen_metrics['bleu2_gen'][i], 
                                        'ROUGE1@MTV': gen_metrics['rouge1_recall_gen'][i],
                                        'ROUGE1@RAW': mdicts_gen[i]['ROUGE1@RAW'],
                                        })
                # 添加到详细结果行
                all_rows.append({
                    'LM': self.text_model_id, 'Task': t_key, 'Dataset': d_key, 'Subject': s_key, 
                    'Raw Task Key': raw_t_key, 
                    'Input Text': raw_input_strs[i], 'Target Texts': all_tgt_str_tuples[i],
                    'Generated Text': gen_strs[i], 
                    'Bleu1': gen_metrics['bleu1_gen'][i], 'Bleu2': gen_metrics['bleu2_gen'][i], 
                    'Rouge1-recall': gen_metrics['rouge1_recall_gen'][i], 
                    'GM': mdicts_gen[i],  # 详细生成指标
                    'Rouge1-precision': gen_metrics['rouge1_precision_gen'][i],  
                    'Rouge1-fmeasure': gen_metrics['rouge1_fmeasure_gen'][i], 'WER': gen_metrics['wer_gen'][i],
                    'Bleu3': gen_metrics['bleu3_gen'][i], 'Bleu4': gen_metrics['bleu4_gen'][i], 
                    
                    'Sentiment label': se_labels[i], 'Sentiment Predictions': se_prob_dicts[i], 
                    'Relation label': re_labels[i], 'Relation Predictions': re_prob_dicts[i], 

                    'Target Text (Current epoch)': tf_tgt_strs[i], 'Generated Text (w/tf)': tf_strs[i], 
                    'Bleu1 (w/tf)': gen_metrics['bleu1_tf'][i], 'Bleu2 (w/tf)': gen_metrics['bleu2_tf'][i], 
                    'Rouge1-precision (w/tf)': gen_metrics['rouge1_precision_tf'][i],   
                    'Rouge1-fmeasure (w/tf)': gen_metrics['rouge1_fmeasure_tf'][i], 'WER (w/tf)': gen_metrics['wer_tf'][i],
                    'Bleu3 (w/tf)': gen_metrics['bleu3_tf'][i], 'Bleu4 (w/tf)': gen_metrics['bleu4_tf'][i], 
                    'Rouge1-recall (w/tf)': gen_metrics['rouge1_recall_tf'][i],       
                })
        
        # 计算样本级指标的平均值
        to_mean_metrics = default_collate(to_mean_metrics)
        mean_metrics.update({f"{prefix}/mean_{k}":v.mean() for k,v in to_mean_metrics.items()})
        
        # 在第一个epoch定义指标
        if self.current_epoch == 0:
            self.define_metrics(list(all_group_metrics.keys())+list(mean_metrics.keys()))
        
        # 记录指标
        self.log_dict(all_group_metrics, rank_zero_only=True)  # 只在rank 0记录组指标
        self.log_dict(mean_metrics, rank_zero_only=True)  # 只在rank 0记录平均指标
        
        # 记录样本详细结果表格
        sample_metrics = pd.DataFrame(all_rows)
        self.logger.log_table(key=f'{prefix}/Samples', dataframe=sample_metrics)

    def on_test_epoch_start(self):  # 测试epoch开始时的回调
        assert not dist.is_initialized()  # 注释：使用单GPU确保可重现性
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):  # 测试步骤
        # 执行共享前向传播
        shared_outputs = self.shared_forward(batch)
        
        # 提取损失
        loss_commitment = shared_outputs['loss_commitment']     # 承诺损失 (1)
        loss_clip = shared_outputs['loss_clip']                 # CLIP损失 (1)
        loss_lm = shared_outputs['loss_lm']                     # 语言模型损失 (1)
        
        metrics = {'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm,         
                    } 
        
        # 计算检索指标
        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=True)  
        metrics.update(retrieval_metrics)
        
        # 为每个batch添加标识符
        metrics = ({f'test/{k}-batch{batch_idx}':v for k, v in metrics.items()})
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_retrieval)
        
        # 执行完整验证步骤并保存输出
        self.full_val_step(shared_outputs)
        outputs = self.full_val_step(shared_outputs)
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):  # 测试epoch结束时的回调
        # 整理测试步骤输出：dict[list[dict[str, tensor]]]
        outputs = default_collate(self.test_step_outputs)         # (n_steps, bsz, ...)
        outputs = {k: v.flatten(0,1) for k,v in outputs.items()}      # (n_steps*bsz, ...)
        
        # 计算并记录测试指标
        with torch.autocast(device_type='cuda', dtype=(torch.bfloat16 if self.trainer.precision == "bf16-mixed" else torch.half)):
            self.cal_and_log(outputs, prefix='test')
        
        # 清空输出列表
        self.test_step_outputs.clear()

    @torch.no_grad()  # 不计算梯度的推理方法
    def predict(self, eeg, eeg_mask, prompts, candidates:list[str]=["It is good.","It is bad."], generate=False):
        """预测方法：输入EEG信号和提示，输出分类概率和生成文本"""
        
        # 编码提示
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)  # (n, 3)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)  # (n, c, 3) --> (n, c)
        
        # 编码EEG信号
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)
        eeg_embs, eeg_emb_vector = self.aligner.embed_eeg(eeg_hiddens)

        # 计算候选标签的嵌入并进行分类
        label_embs = self.cal_label_embs(labels=candidates)
        probs = self.run_cls(eeg_emb_vector, label_embs)

        # 初始化生成结果
        gen_strs = [None]*len(eeg)
        if generate:  # 如果需要生成文本
            gen_ids = self.text_model.generate(encoder_outputs = BaseModelOutput(eeg_embs), 
                                                num_beams = 2, 
                                                min_length=0, max_length=self.tgt_text_len,
                                                )
            gen_strs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        return probs, gen_strs
    
    def predict_text_embedding(self, texts: list[str], input_template: str, candidates:list[str]):
        """使用文本嵌入进行预测（用于对比分析）"""
        if input_template:  # 如果提供了输入模板
            # 将文本插入模板
            texts = [input_template.replace("<MASK>", text) for text in texts]
        
        # tokenize输入文本
        input_ids, input_mask = self.tokenize(texts, self.input_text_len-self.prompt_tuning_len)
        # 编码文本
        input_text_embeds, _ = self.encode_text(input_ids, input_mask)
        # 通过对齐器获取文本嵌入向量
        text_emb = self.aligner.embed_text(input_text_embeds, input_mask)

        # 计算候选标签嵌入并进行分类
        label_embs = self.cal_label_embs(labels=candidates)
        probs = self.run_cls(text_emb, label_embs)
        return probs