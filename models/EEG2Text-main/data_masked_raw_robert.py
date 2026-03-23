import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
from scipy.signal import spectrogram


def normalize_2d(input_tensor):
    mean = torch.mean(input_tensor, dim=1, keepdim=True)
    std = torch.std(input_tensor, dim=1, keepdim=True)

    # Avoid division by zero
    std[std == 0] = 1

    normalized_tensor = (input_tensor - mean) / std
    return normalized_tensor


def bert_mlm_mask(arr, mask_percentage=0.15):
    """
    Mask elements of a 2D float array using BERT's MLM strategy.

    Parameters:
    - arr: 2D numpy array of floats.
    - mask_percentage: Percentage of elements to mask.

    Returns:
    - Masked array.
    - Indices of masked elements.
    """
    # Find total number of elements in the first dimension (K)
    total_elements = arr.shape[0]

    # Calculate number of elements to mask
    num_mask = int(total_elements * mask_percentage)

    # Randomly select elements to mask
    mask_indices = np.random.choice(total_elements, num_mask, replace=False)

    # Make a copy of the array to avoid modifying the original
    masked_arr = arr.copy()

    # Get number of elements for each masking strategy
    num_mask_80 = int(0.8 * num_mask)
    num_mask_10 = int(0.1 * num_mask)
    # Remaining will be for the unchanged strategy

    # 80% replaced by mask_value (using -9999.0 as a [MASK] representation)
    masked_arr[mask_indices[:num_mask_80], :] = 0.0

    # 10% replaced with random float value (assuming float values between 0 and 10 for simplicity)
    for idx in mask_indices[num_mask_80:num_mask_80 + num_mask_10]:
        random_value = np.random.uniform(-10.0, 10.0, arr.shape[1])
        masked_arr[idx, :] = random_value

    # 10% remain unchanged (so we do nothing)
    pad_length = 24000 - masked_arr.shape[0]
    masked_arr = np.pad(masked_arr, ((0, pad_length), (0, 0)), mode='constant', constant_values=0.0)
    #return masked_arr, mask_indices
    assert masked_arr.shape[0] == 24000
    assert masked_arr.shape[1] == 105
    return_tensor = torch.from_numpy(masked_arr)

    pad_length = 24000 - mask_indices.shape[0]
    mask_indices = np.pad(mask_indices, (0, pad_length), mode='constant', constant_values=-1)
    mask_indices = torch.from_numpy(mask_indices)
    return normalize_2d(return_tensor), mask_indices


def get_input_sample(sent_obj, tokenizer, max_len=56, add_CLS_token=False, max_spectro_datapoint=24000):
    def get_sent_eeg(sent_eeg_embedding):
        # np.set_printoptions(threshold=np.inf)
        # max in fact: 23630
        sent_eeg_features = []
        #sent_eeg_embedding = sent_obj['sentence_level_EEG']['rawData']
        # print(sent_eeg_embedding.shape)
        pad_length = max_spectro_datapoint - sent_eeg_embedding.shape[0]
        sent_eeg_embedding = np.pad(sent_eeg_embedding, ((0, pad_length), (0, 0)), mode='constant', constant_values=0.0)
        # print(sent_eeg_embedding.shape)
        assert sent_eeg_embedding.shape[0] == 24000
        assert sent_eeg_embedding.shape[1] == 105

        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_2d(return_tensor)
        # return return_tensor

    if sent_obj is None:
        # print(f'  - skip bad sentence')
        return None
    # print(sent_obj['sentence_level_EEG']['rawData'].shape[1])
    if sent_obj['sentence_level_EEG']['rawData'].shape[1] >= 24000:
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    # print(target_string)
    # print(len(target_string.split(' ')) + 1)
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True,
                                 return_tensors='pt', return_attention_mask=True)

    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    # print(input_sample['target_ids'])
    # get sentence level EEG features
    sent_eeg_embedding = sent_obj['sentence_level_EEG']['rawData'].T
    sent_level_eeg_tensor = get_sent_eeg(sent_eeg_embedding)
    #input_sample['masked_EEG'], input_sample['mask_indices'] = bert_mlm_mask(sent_eeg_embedding, mask_percentage=0.15)
    #print(sent_level_eeg_tensor.size())
    #print(input_sample['masked_EEG'].size())
    #print(input_sample['mask_indices'])
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_eeg_len'] = sent_eeg_embedding.shape[0]
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    print(input_sample['sent_eeg_len'])
    #equal_rows = (input_sample['sent_level_EEG'] == input_sample['masked_EEG']).all(dim=1)
    #for i, are_equal in enumerate(equal_rows):
        #if are_equal.item():
            #print(f"Row {i} is the same in both tensors.")
            #continue
        #else:
            #print(f"Row {i} is different.")
            #print(input_sample['masked_EEG'][i])
    # mask out target padding for computing cross entropy loss
    input_sample['sent_mask'] = torch.zeros(24000)
    # print(len(sent_obj['sentence_level_EEG']['rawData'][0]))
    max_len = len(sent_obj['sentence_level_EEG']['rawData'][0])
    # print(max_timepoint)
    input_sample['sent_mask'][:max_len] = torch.ones(max_len)
    # print(input_sample['spectro_mask'])
    #print(input_sample['sent_mask'].eq(1).sum().item())
    #print(max_len)
    input_sample['sent_mask_invert'] = 1 - input_sample['sent_mask']
    seq_len = len(target_string.split(' ')) + 1

    input_sample['seq_len'] = seq_len

    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', setting='unique_sent',
                 is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]

            total_num_sentence = len(input_dataset_dict[subjects[0]])

            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                                            add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            print('++ adding task to dataset, now we have:', len(self.inputs))

        print('[INFO]input tensor size:', self.inputs[0]['sent_level_EEG'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['seq_len'],
            input_sample['target_ids'],
            input_sample['sent_mask'],
            input_sample['sent_mask_invert'],
            input_sample['sent_eeg_len'],
            input_sample['sent_level_EEG']

        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,


"""for train classifier on stanford sentiment treebank text-sentiment pairs"""


class SST_tenary_dataset(Dataset):
    def __init__(self, ternary_labels_dict, tokenizer, max_len=56, balance_class=True):
        self.inputs = []

        pos_samples = []
        neg_samples = []
        neu_samples = []

        for key, value in ternary_labels_dict.items():
            tokenized_inputs = tokenizer(key, padding='max_length', max_length=max_len, truncation=True,
                                         return_tensors='pt', return_attention_mask=True)
            input_ids = tokenized_inputs['input_ids'][0]
            attn_masks = tokenized_inputs['attention_mask'][0]
            label = torch.tensor(value)
            # count:
            if value == 0:
                neg_samples.append((input_ids, attn_masks, label))
            elif value == 1:
                neu_samples.append((input_ids, attn_masks, label))
            elif value == 2:
                pos_samples.append((input_ids, attn_masks, label))
        print(
            f'Original distribution:\n\tVery positive: {len(pos_samples)}\n\tNeutral: {len(neu_samples)}\n\tVery negative: {len(neg_samples)}')
        if balance_class:
            print(f'balance class to {min([len(pos_samples), len(neg_samples), len(neu_samples)])} each...')
            for i in range(min([len(pos_samples), len(neg_samples), len(neu_samples)])):
                self.inputs.append(pos_samples[i])
                self.inputs.append(neg_samples[i])
                self.inputs.append(neu_samples[i])
        else:
            self.inputs = pos_samples + neg_samples + neu_samples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,


class LazyZuCo_dataset(Dataset):
    """内存高效版本：在 __getitem__ 时才处理EEG数据，而非在 __init__ 时全量预加载。
    spectro pickle 仍整体加载到内存（~5-7GB/task），但 (24000,105) tensor 只在取样时生成。

    划分策略：对所有 task 合并的唯一句子文本用 random.seed(42) 随机打乱后 80/10/10 划分，
    与 benchmark 统一 pickle 的划分逻辑保持一致，避免数据泄露。
    """
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', setting='unique_sent',
                 is_add_CLS_token=False):
        import random as _random

        self.tokenizer = tokenizer
        self.is_add_CLS_token = is_add_CLS_token
        # 存储 (dataset_dict_ref, subject_key, sentence_idx) 的轻量索引
        self.index = []

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[LazyDataset]loading {len(input_dataset_dicts)} task datasets')

        # ── Step 1：收集所有有效候选 (dataset_dict, subject, sent_idx, content) ────────
        all_candidates = []  # list of (dataset_dict, subj, idx, content_text)
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[LazyDataset]using subjects: ', subjects)
            else:
                subjects = [subject]

            for key in subjects:
                for i, sent_obj in enumerate(input_dataset_dict[key]):
                    if sent_obj is None:
                        continue
                    if sent_obj['sentence_level_EEG']['rawData'].shape[1] >= 24000:
                        continue
                    content = sent_obj.get('content', '')
                    all_candidates.append((input_dataset_dict, key, i, content))

        # ── Step 2：按唯一句子文本做 seed=42 随机 80/10/10 划分 ──────────────────────
        # 先排序保证初始顺序确定性，再 shuffle，确保与 build_unified_dataset.py 完全一致
        unique_texts = sorted(set(c[3] for c in all_candidates))
        _random.seed(42)
        _random.shuffle(unique_texts)
        n = len(unique_texts)
        n_train = max(int(n * 0.8), 1)
        n_val   = max(int(n * 0.1), 0)
        if n_train + n_val >= n:
            n_train = max(n - 2, 1)
            n_val   = max(min(n - n_train - 1, 1), 0)
        train_texts = set(unique_texts[:n_train])
        val_texts   = set(unique_texts[n_train:n_train + n_val])
        test_texts  = set(unique_texts[n_train + n_val:])
        print(f'[LazyDataset]split: total_unique={n}, train={len(train_texts)}, '
              f'dev={len(val_texts)}, test={len(test_texts)}')

        # ── Step 3：按 phase 筛选 ─────────────────────────────────────────────────────
        if phase == 'train':
            target_texts = train_texts
        elif phase == 'dev':
            target_texts = val_texts
        else:
            target_texts = test_texts

        for (dd, key, i, content) in all_candidates:
            if content in target_texts:
                self.index.append((dd, key, i))

        print(f'[LazyDataset]++ phase={phase} now have {len(self.index)} refs')
        print(f'[LazyDataset]total refs: {len(self.index)}')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_dict, key, i = self.index[idx]
        input_sample = get_input_sample(dataset_dict[key][i], self.tokenizer,
                                        add_CLS_token=self.is_add_CLS_token)
        if input_sample is None:
            # fallback: try next valid index
            for fallback_idx in range(idx + 1, len(self.index)):
                dataset_dict2, key2, i2 = self.index[fallback_idx]
                sample = get_input_sample(dataset_dict2[key2][i2], self.tokenizer,
                                          add_CLS_token=self.is_add_CLS_token)
                if sample is not None:
                    input_sample = sample
                    break

        return (
            input_sample['seq_len'],
            input_sample['target_ids'],
            input_sample['sent_mask'],
            input_sample['sent_mask_invert'],
            input_sample['sent_eeg_len'],
            input_sample['sent_level_EEG'],
        )


'''sanity test'''
if __name__ == '__main__':

    check_dataset = 'ZuCo'

    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []

        dataset_path_task1 = r'C:\Users\Night\Documents\Code\Python\EEG-To-Text-main\dataset\ZuCo\task1-SR\pickle\task1-SR-dataset-spectro.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        '''dataset_path_task2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-with-tokens_7-10.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        # dataset_path_task3 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-with-tokens_7-10.pickle' 
        # with open(dataset_path_task3, 'rb') as handle:
        #     whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2_v2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle' 
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))'''

        print()
        for key in whole_dataset_dicts[0]:
            print(f'task1, sentence num in {key}:', len(whole_dataset_dicts[0][key]))
        print()

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                                 setting=dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, setting=dataset_setting)
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, setting=dataset_setting)

        print('trainset size:', len(train_set))
        print('devset size:', len(dev_set))
        print('testset size:', len(test_set))

    elif check_dataset == 'stanford_sentiment':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        SST_dataset = SST_tenary_dataset(SST_SENTIMENT_LABELS, tokenizer)
        print('SST dataset size:', len(SST_dataset))
        print(SST_dataset[0])
        print(SST_dataset[1])