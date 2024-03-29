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
            transform.MordalitySelection(params['mordality']),
            transform.RandomKeepDiagMed(),
            transform.RandomCropSequence(p=params['p'], seq_threshold=params['seq_threshold']),
            transform.TruncateSeqence(params['max_seq_length']),
            transform.EHRAugmentation(),
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
        sample = {'code': self.data.code[index],
                  'age': self.data.age[index]
                  # 'seg': self.data.seg[index],
                  # 'position': self.data.position[index]
                  }

        sample = self._compose(sample)

        return {'code': torch.LongTensor(sample['code']),
                'age': torch.LongTensor(sample['age']),
                'seg': torch.LongTensor(sample['seg']),
                'position': torch.LongTensor(sample['position']),
                'att_mask': torch.LongTensor(sample['att_mask']),
                'h_att_mask': torch.LongTensor(sample['h_att_mask'])}

    def __len__(self):
        return len(self.data)


def MlmByolDataLoader(params):
    if params['data_path'] is not None:
        data = pd.read_parquet(params['data_path'])
        print('number of patients:', len(data))
        dset = SSLDset(dataset=data, params=params)
        dataloader = DataLoader(dataset=dset,
                                batch_size=params['batch_size'],
                                shuffle=params['shuffle'],
                                num_workers=params['num_workers']
                                )
        return dataloader
    else:
        return None