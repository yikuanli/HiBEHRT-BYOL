from torch.utils.data import Dataset, DataLoader
import torch
from dataloaders import transform
from torchvision import transforms
import pandas as pd


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
        self._compose = transforms.Compose([
            transform.TruncateSeqence(params['max_seq_length']),
            transform.CalibratePosition(),
            transform.TokenAgeSegPosition2idx(params['token_dict_path'], params['age_dict_path']),
            transform.RetriveSeqLengthAndPadding(params['max_seq_length']),
            transform.FormatAttentionMask(params['max_seq_length']),
            transform.FormatHierarchicalStructure(params['segment_length'], params['move_length'],
                                                  params['max_seq_length'])
        ])

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        sample = {
            'code': self.data.code[index],
            'age': self.data.age[index],
            'seg': self.data.seg[index],
            'position': self.data.position[index],
            'label': self.data.label[index]
        }

        sample = self._compose(sample)

        return {'code': torch.LongTensor(sample['code']),
                'age': torch.LongTensor(sample['age']),
                'seg': torch.LongTensor(sample['seg']),
                'position': torch.LongTensor(sample['position']),
                'att_mask': torch.LongTensor(sample['att_mask']),
                'h_att_mask': torch.LongTensor(sample['h_att_mask']),
                'label': torch.FloatTensor([sample['label']])}

    def __len__(self):
        return len(self.data)


def EHR2VecDataLoader(params):
    if params['data_path'] is not None:
        data = pd.read_parquet(params['data_path'])
        if 'fraction' in params:
            data = data.sample(frac=params['fraction']).reset_index(drop=True)

        if params['selection'] is not None:
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