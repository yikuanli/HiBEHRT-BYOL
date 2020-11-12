from torch.utils.data import Dataset, DataLoader
import torch
from dataloaders import transform
from torchvision import transforms
import pandas as pd


class SSLDset(Dataset):
    def __init__(self, dataset, params):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.data = dataset
        self._compose = transforms.Compose([
            transform.TruncateSeqence(params['max_seq_length']),
            transform.CalibratePosition(),
            transform.EHRAugmentation(),
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
        sample1 = {'code': self.data.code[index],
                   'age': self.data.age[index],
                   'seg': self.data.seg[index],
                   'position': self.data.position[index]}

        sample2 = {'code': self.data.code[index],
                   'age': self.data.age[index],
                   'seg': self.data.seg[index],
                   'position': self.data.position[index]}

        sample1 = self._compose(sample1)
        sample2 = self._compose(sample2)

        return torch.LongTensor(sample1['code']), \
               torch.LongTensor(sample2['code']), \
               torch.LongTensor(sample1['age']), \
               torch.LongTensor(sample1['seg']), \
               torch.LongTensor(sample1['position']), \
               torch.LongTensor(sample1['att_mask']), \
               torch.LongTensor(sample1['h_att_mask'])

    def __len__(self):
        return len(self.data)


def SSLDataLoader(params):
    if params['data_path'] is not None:
        data = pd.read_parquet(params['data_path'])
        dset = SSLDset(dataset=data, params=params)
        dataloader = DataLoader(dataset=dset,
                                batch_size=params['batch_size'],
                                shuffle=params['shuffle'],
                                num_workers=params['num_workers']
                                )
        return dataloader
    else:
        return None