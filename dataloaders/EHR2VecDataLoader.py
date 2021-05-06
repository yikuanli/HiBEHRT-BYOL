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
            transform.MordalitySelection(params['mordality']),
            transform.RecordsAugment(aug_prob=params['aug_prob'],
                                     mask_prob=params['mask_prob'],
                                     drop_prob=params['drop_prob'],
                                     is_train=params['is_train']),
            transform.TruncateSeqence(params['max_seq_length']),
            transform.CreateSegandPosition(),
            # transform.RemoveSEP(),
            transform.TokenAgeSegPosition2idx(params['token_dict_path'], params['age_dict_path']),
            transform.RetriveSeqLengthAndPadding(params['max_seq_length']),
            transform.FormatAttentionMask(params['max_seq_length']),
            transform.FormatHierarchicalStructure(params['segment_length'], params['move_length'],
                                                  params['max_seq_length']),
            transform.CalibrateHierarchicalPosition(),
            transform.CalibrateSegmentation()
        ])

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
            # select patients who have at least one records in the selection list
            for key in params['selection']:
                data[key] = data.code.apply(lambda x: sum([1 for each in x if each[0:3] == key]))
                data = data[data[key] > 1]
            data = data.reset_index(drop=True)

        if params['len_range'] is not None:
            data['len'] = data.code.apply(lambda x: len(x))
            data = data[data['len']>params['len_range']['min_len']]
            data = data[data['len']<params['len_range']['max_len']]
            data = data.reset_index(drop=True)

        if params['year_range'] is not None:
            data['duration'] = data.age.apply(lambda x: int(x[-1])-int(x[0]))
            data = data[data['duration'] > params['year_range']['min_year']]
            data = data[data['duration'] < params['year_range']['max_year']]
            data = data.reset_index(drop=True)

        if params['age_range'] is not None:
            data['baseline_age'] = data.age.apply(lambda x: int(x[-1]))
            data = data[data['baseline_age'] > params['age_range']['min_age']]
            data = data[data['baseline_age'] < params['age_range']['max_age']]
            data = data.reset_index(drop=True)

        if params['positive_percent'] is not None:
            pos = data[data['label'] == 1]
            neg = data[data['label'] == 0]

            num_pos = (len(neg) * params['positive_percent'])/(1 - params['positive_percent'])
            pos = pos.sample(n=num_pos)

            data = pd.concat([pos, neg])
            data = data.reset_index(drop=True)

        print('data size:', len(data))
        print('positive sample size', len(data[data['label'] == 1]))
        print('percentage of positive samples:', len(data[data['label'] == 1])/len(data))


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