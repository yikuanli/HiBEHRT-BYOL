from utils.utils import load_obj
import numpy as np
import math
import random


class RandomCropSequence(object):
    def __init__(self, p, seq_threshold=50):
        # p probability (0, 1) of doing random crop
        self.p = p
        self.seq_threshold = seq_threshold

    def __call__(self, sample):
        prob = random.random()

        if prob > self.p:
            return sample
        else: # do random crop
            seq_len = len(sample['code'])

            # if seq length <= threshold length , don't do random crop
            if seq_len <= self.seq_threshold:
                return sample
            else:
                # else select start point from 0 to length - threshold
                start = random.randint(0, seq_len-self.seq_threshold)
                len_choise = random.randint(self.seq_threshold, max(self.seq_threshold, seq_len))

                sample.update({
                    'code': sample['code'][start:(start+len_choise)],
                    'age': sample['age'][start:(start+len_choise)],
                    'seg': sample['seg'][start:(start+len_choise)],
                    'position': sample['position'][start:(start+len_choise)]})
                return sample


class TruncateSeqence(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length

    def __call__(self, sample):
        sample.update({
            'code': sample['code'][-self.max_seq_length:],
            'age': sample['age'][-self.max_seq_length:],
            'seg': sample['seg'][-self.max_seq_length:],
            'position': sample['position'][-self.max_seq_length:]})
        return sample


class CalibratePosition(object):
    def __call__(self, sample):
        position = sample['position']
        position_list = []
        for each in position:
            each = int(each) - int(position[0])
            position_list.append(str(each))

        sample.update({'position': position_list})
        return sample


class TokenAgeSegPosition2idx(object):
    def __init__(self, token_dict_path, age_dict_path):
        self.token2idx = load_obj(token_dict_path)['token2idx']
        self.age2idx = load_obj(age_dict_path)['token2idx']

    def __call__(self, sample):
        code, age, seg, position = sample['code'], sample['age'], sample['seg'], sample['position']

        code = [self.token2idx.get(each, 1) for each in code]
        age = [self.age2idx.get(each, 1) for each in age]
        seg = [int(each) for each in seg]
        position = [int(each) for each in position]

        sample.update({'code': code, 'age': age, 'seg': seg, 'position': position})
        return sample


class RetriveSeqLengthAndPadding(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        # padding token need to be 0

    def __call__(self, sample):
        code = sample['code']
        seq_length = len(code)

        def pad(x):
            seq_len = len(x)
            array = np.zeros(self.max_seq_length)
            array[:seq_len] = x
            return array

        sample.update({'code': pad(sample['code']), 'age': pad(sample['age']),
                       'seg': pad(sample['seg']), 'position': pad(sample['position']),
                       'length': seq_length})

        return sample


class FormatAttentionMask(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        # padding token need to be 0

    def __call__(self, sample):
        mask = np.zeros(self.max_seq_length)
        usable = np.ones(sample['length'])
        mask[:len(usable)] = usable

        sample.update({'att_mask': mask})

        return sample


class FormatHierarchicalStructure(object):
    def __init__(self, segment_length, move_length, max_seq_length):
        self.segment_length = segment_length
        self.move_length = move_length
        self.max_seq_length = max_seq_length

    def __call__(self, sample):
        if (self.max_seq_length-self.segment_length) % 50 != 0:
            raise ValueError('Need to set up (max seqence length - segment length) % move length == 0')
        else:
            code = [sample['code'][n*self.move_length:(self.segment_length + n*self.move_length)]
                    for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
            age = [sample['age'][n*self.move_length:(self.segment_length + n*self.move_length)]
                   for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
            seg = [sample['seg'][n*self.move_length:(self.segment_length + n*self.move_length)]
                   for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
            position = [sample['position'][n*self.move_length:(self.segment_length + n*self.move_length)]
                        for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
            att_mask = [sample['att_mask'][n*self.move_length:(self.segment_length + n*self.move_length)]
                        for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]

            mask = np.zeros((self.max_seq_length-self.segment_length)//self.move_length + 1)
            if sample['length'] <= self.segment_length:
                num = 1
            else:
                num = math.ceil((sample['length']-self.segment_length)/self.move_length) + 1
            mask[:num] = np.ones(num)

        sample.update({'code': code, 'age': age, 'seg': seg, 'position': position,
                       'att_mask': att_mask, 'h_att_mask': mask})

        return sample


class EHRAugmentation(object):
    def __call__(self, sample):
        code = sample['code']

        def random_mask(record, category, p=0.25):
            record_list = []
            for token in record:
                if token[0:3] not in category:
                    prob = random.random()
                    if prob < p:
                        record_list.append('MASK')
                else:
                    record_list.append(token)
            return np.array(record_list)

        strategy = ['random_mask', 'mask_diag_med', 'mask_rest']
        p = [0.5, 0.3, 0.2]
        choice = np.random.choice(strategy, p=p)

        if choice == 'random_mask':
            code = random_mask(code, category=[], p=0.15)
        elif choice == 'mask_diag_med':
            code = random_mask(code, category=['TES', 'BMI', 'BPL', 'BPH', 'SMO', 'ALC'], p=0.5)
        else:
            code = random_mask(code, category=['DIA', 'MED'], p=0.5)

        sample.update({'code': code})

        return sample