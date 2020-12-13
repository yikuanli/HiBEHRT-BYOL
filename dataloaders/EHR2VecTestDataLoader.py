from torch.utils.data import Dataset, DataLoader
import torch
from dataloaders import transform
from torchvision import transforms
import pandas as pd
import math
import random
import numpy as np
from utils.utils import load_obj


def weightedSampling(data, classes, split):
    def make_weights_for_balanced_classes(sampled, nclasses, split):
        count = sampled.label.value_counts().to_list()
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / (float(count[i]))
        weight = [0] * int(N)
        weight_per_class[0] = weight_per_class[0] * split

        for idx, val in enumerate(sampled.label):
            weight[idx] = weight_per_class[int(val)]
        return weight

    w = make_weights_for_balanced_classes(data, classes, split)
    w = torch.DoubleTensor(w)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(w, len(w), replacement=True)
    return sampler


class EHR2VecDset(Dataset):
    def __init__(self, dataset, params):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.data = dataset
        self.params = params
        self._compose = transforms.Compose([
            transform.MordalitySelection(params['mordality'])
            # transform.TruncateSeqence(params['max_seq_length']),
            # transform.CreateSegandPosition(),
            # # transform.RemoveSEP(),
            # transform.TokenAgeSegPosition2idx(params['token_dict_path'], params['age_dict_path']),
            # transform.RetriveSeqLengthAndPadding(params['max_seq_length']),
            # transform.FormatAttentionMask(params['max_seq_length']),
            # transform.FormatHierarchicalStructure(params['segment_length'], params['move_length'],
            #                                       params['max_seq_length'])
        ])

        self.vocab = load_obj(self.params['token_dict_path'])['token2idx']
        self.age2idx = load_obj(self.params['age_dict_path'])['token2idx']

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        sample = {
            'code': self.data.code[index],
            'age': self.data.age[index],
            'label': self.data.label[index]
        }

        sample = self._compose(sample)

        code = sample['code']
        age = sample['age']

        # start to use the original dataloader
        age = age[-self.params['max_seq_length']:]
        code = code[-self.params['max_seq_length']:]

        # separate code and age into segments with max length = seq_len for each segment, move 50 token for each seg
        code_list = []
        age_list = []
        posi_list = []
        seg_list = []
        mask_list = []
        seg_mask_list = []
        for n in range(math.ceil((self.params['max_seq_length'] - self.params['segment_length']) / self.params['move_length'])):
            temp_code = code[n * self.params['move_length']:(self.params['segment_length'] + n * self.params['move_length'])]
            temp_age = age[n * self.params['move_length']:(self.params['segment_length'] + n * self.params['move_length'])]

            if len(temp_code) != 0:
                ori_code, code_idx = code2index(temp_code, self.vocab)
                temp_code = seq_padding(code_idx, self.params['segment_length'], symbol=self.vocab['PAD'])
                temp_age = seq_padding(temp_age, self.params['segment_length'], token2idx=self.age2idx)

                mask = np.ones(self.params['segment_length'])
                mask[len(ori_code):] = 0

                ori_code = seq_padding(ori_code, self.params['segment_length'])
                position = position_idx(ori_code)
                segment = index_seg(ori_code)
                seg_mask_list.append(1)
            else:
                temp_code = np.ones(self.params['segment_length']) * self.vocab.get('PAD')
                temp_age = np.ones(self.params['segment_length']) * self.age2idx.get('PAD')
                mask = np.zeros(self.params['segment_length'])
                position = np.zeros(self.params['segment_length'])
                segment = np.zeros(self.params['segment_length'])
                seg_mask_list.append(0)

            code_list.append(temp_code)
            age_list.append(temp_age)
            mask_list.append(mask)
            posi_list.append(position)
            seg_list.append(segment)

        return {'code': torch.LongTensor(code_list),
                'age': torch.LongTensor(age_list),
                'seg': torch.LongTensor(seg_list),
                'position': torch.LongTensor(posi_list),
                'att_mask': torch.LongTensor(mask_list),
                'h_att_mask': torch.LongTensor(seg_mask_list),
                'label': torch.FloatTensor([sample['label']])}

    def __len__(self):
        return len(self.data)


def EHR2VecTestDataLoader(params):
    if params['data_path'] is not None:
        data = pd.read_parquet(params['data_path'])
        if 'fraction' in params:
            data = data.sample(frac=params['fraction']).reset_index(drop=True)

        if params['selection'] is not None:
            # select patients who have at least one records in the selection list
            for key in params['selection']:
                data[key] = data.code.apply(lambda x: sum([1 for each in x if each[0:3] == key]))
                data = data[data[key] > 1]
            data = data.reset_index(drop=True)

        print('data size:', len(data))

        dset = EHR2VecDset(dataset=data, params=params)

        sampler = None

        if 'ratio' in params:
            if params['ratio'] is not None:
                sampler = weightedSampling(data, 2, params['ratio'])

        dataloader = DataLoader(dataset=dset,
                                batch_size=params['batch_size'],
                                shuffle=params['shuffle'],
                                num_workers=params['num_workers'],
                                sampler=sampler
                                )
        return dataloader
    else:
        return None


def code2index(tokens, token2idx, mask_token=None):
    output_tokens = []
    for i, token in enumerate(tokens):
        if token==mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens


def seq_padding(tokens, max_len, token2idx=None, symbol=None, unkown=True):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                if unkown:
                    seq.append(token2idx.get(tokens[i], token2idx['UNK']))
                else:
                    seq.append(token2idx.get(tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq

def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos