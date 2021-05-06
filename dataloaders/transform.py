from utils.utils import load_obj
import numpy as np
import math
import random


class CreateSegandPosition(object):
    def index_seg(self, tokens, symbol='SEP'):
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

    def position_idx(self, tokens, symbol='SEP'):
        pos = []
        flag = 0

        for token in tokens:
            if token == symbol:
                pos.append(flag)
                flag += 1
            else:
                pos.append(flag)
        return pos

    def __call__(self, sample):
        code = sample['code']
        position = self.position_idx(code)
        seg = self.index_seg(code)

        sample.update({
            'seg': np.array(seg),
            'position': np.array(position)
        })
        return sample


class RandomKeepDiagMed(object):
    def __init__(self, diag='DIA', med='MED', keep_prob=0.25):
        self.name_list = [diag, med]
        self.keep_prob = keep_prob

    def __call__(self, sample):
        prob = random.random()
        code = sample['code']
        age = sample['age']
        # seg = sample['seg']
        # position = sample['position']

        if prob < self.keep_prob:
            new_code = []
            new_age = []
            # new_seg = []
            # new_position = []
            for i in range(len(code)):
                if code[i][0:3] in self.name_list:
                    new_code.append(code[i])
                    new_age.append(age[i])
                    # new_seg.append(seg[i])
                    # new_position.append(position[i])
            sample.update({
                'code': np.array(new_code),
                'age': np.array(new_age)
                # 'seg': np.array(new_seg),
                # 'position': np.array(new_position)
            })
        return sample


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
                    'age': sample['age'][start:(start+len_choise)]
                    # 'seg': sample['seg'][start:(start+len_choise)],
                    # 'position': sample['position'][start:(start+len_choise)]
                })
                return sample


class TruncateSeqence(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length

    def __call__(self, sample):
        sample.update({
            'code': sample['code'][-self.max_seq_length:],
            'age': sample['age'][-self.max_seq_length:]
            # 'seg': sample['seg'][-self.max_seq_length:],
            # 'position': sample['position'][-self.max_seq_length:]
        })
        return sample


# class MordalitySelection(object):
#     def __init__(self, mordality_list):
#         self.mordality = mordality_list
#         if mordality_list is not None:
#             self.mordality.append('SEP')
#
#     def __call__(self, sample):
#         if self.mordality is not None:
#             code = sample['code']
#             age = sample['age']
#
#             code_list = []
#             age_list = []
#             for i in range(len(code)):
#                 if code[i][0:3] in self.mordality:
#                     code_list.append(code[i])
#                     age_list.append(age[i])
#             sample.update({
#                 'code': np.array(code_list),
#                 'age': np.array(age_list)
#             })
#         return sample

class MordalitySelection(object):
    def __init__(self, mordality_list):
        self.mordality = mordality_list
        if mordality_list is not None:
            self.mordality.append('SEP')

    def __call__(self, sample):
        if self.mordality is not None:
            code = sample['code']
            age = sample['age']

            code_list = []
            age_list = []

            last_code = 0
            for i in range(len(code)):
                if code[i][0:3] in self.mordality:
                    if code[i][0:3] != 'SEP':
                        code_list.append(code[i])
                        age_list.append(age[i])
                        last_code = code[i]
                    else:
                        if last_code != 'SEP':
                            code_list.append(code[i])
                            age_list.append(age[i])
                            last_code = code[i]

            sample.update({
                'code': np.array(code_list),
                'age': np.array(age_list)
            })
        return sample


class CalibrateHierarchicalPosition(object):
    def __call__(self, sample):
        position = sample['position']

        def calibrate(element, value):
            if element != 0:
                return element - value
            else:
                return element

        position_list = []
        for seg in position:
            position_temp = [calibrate(each, seg[0]) for each in seg]
            position_list.append(position_temp)

        sample.update({'position': np.array(position_list)})
        return sample


class CalibrateSegmentation(object):
    def __call__(self, sample):
        segment = sample['seg']

        def reverse(element):
            if element == 0:
                return 1.
            else:
                return 0

        segment_list = []
        for seg in segment:
            if seg[0] == 0:
                segment_list.append(seg)
            else:
                seg_tmp = [reverse(each) for each in seg]
                segment_list.append(seg_tmp)
        sample.update({'seg': np.array(segment_list)})

        return sample


class RecordsAugment(object):
    def __init__(self, aug_prob=0.25, mask_prob=0.05, drop_prob=0.05, is_train=False):
        self.aug_prob = aug_prob
        self.mask_prob = mask_prob
        self.drop_prob = drop_prob
        self.is_train = is_train

    def __call__(self, sample):
        if self.is_train:
            code = sample['code']
            age = sample['age']

            seed = random.random()

            if seed < self.aug_prob:
                # for augmentation 25% mask, 25% replace as UNK, and 50% drop
                code_list = []
                age_list = []
                for i in range(len(code)):
                    if code[i] != 'SEP':
                        seed = random.random()
                        if seed < self.mask_prob:
                            code_list.append('MASK')
                            age_list.append(age[i])
                        elif (seed >= self.mask_prob) and (seed <= self.mask_prob + self.drop_prob):
                            pass
                        else:
                            code_list.append(code[i])
                            age_list.append(age[i])
                    else:
                        code_list.append(code[i])
                        age_list.append(age[i])

                code = code_list
                age = age_list

            sample.update({'coe': np.array(code),
                           'age': np.array(age)})
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


class RemoveSEP(object):
    def __init__(self, symbol='SEP'):
        self.symbol = symbol

    def __call__(self, sample):
        code = sample['code']
        age = sample['age']
        seg = sample['seg']
        position = sample['position']

        code_list = []
        age_list = []
        seg_list = []
        position_list = []
        for i in range(len(code)):
            if code[i] != 'SEP':
                code_list.append(code[i])
                age_list.append(age[i])
                seg_list.append(seg[i])
                position_list.append(position[i])

        sample.update({
            'code': np.array(code_list),
            'age': np.array(age_list),
            'seg': np.array(seg_list),
            'position': np.array(position_list)
        })

        return sample


# class FormatHierarchicalStructure(object):
#     def __init__(self, segment_length, move_length, max_seq_length):
#         self.segment_length = segment_length
#         self.move_length = move_length
#         self.max_seq_length = max_seq_length
#
#     def __call__(self, sample):
#         if (self.max_seq_length-self.segment_length) % 50 != 0:
#             raise ValueError('Need to set up (max seqence length - segment length) % move length == 0')
#         else:
#             code = [sample['code'][n*self.move_length:(self.segment_length + n*self.move_length)]
#                     for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
#             age = [sample['age'][n*self.move_length:(self.segment_length + n*self.move_length)]
#                    for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
#             seg = [sample['seg'][n*self.move_length:(self.segment_length + n*self.move_length)]
#                    for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
#             position = [sample['position'][n*self.move_length:(self.segment_length + n*self.move_length)]
#                         for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
#             att_mask = [sample['att_mask'][n*self.move_length:(self.segment_length + n*self.move_length)]
#                         for n in range((self.max_seq_length-self.segment_length)//self.move_length + 1)]
#
#             mask = np.zeros((self.max_seq_length-self.segment_length)//self.move_length + 1)
#             if sample['length'] <= self.segment_length:
#                 num = 1
#             else:
#                 num = math.ceil((sample['length']-self.segment_length)/self.move_length) + 1
#             mask[:num] = np.ones(num)
#
#         sample.update({'code': code, 'age': age, 'seg': seg, 'position': position,
#                        'att_mask': att_mask, 'h_att_mask': mask})
#
#         return sample

class FormatHierarchicalStructure(object):
    def __init__(self, segment_length, move_length, max_seq_length):
        self.segment_length = segment_length
        self.move_length = move_length
        self.max_seq_length = max_seq_length

    def __call__(self, sample):
        if (self.max_seq_length - self.segment_length) % self.move_length != 0:
            raise ValueError('Need to set up (max seqence length - segment length) % move length == 0')
        else:
            code = sample['code']
            age = sample['age']
            seg = sample['seg']
            position = sample['position']
            att_mask = sample['att_mask']

            code_list = [code[n * self.move_length:(self.segment_length + n * self.move_length)] for n in
                         range(math.ceil((self.max_seq_length - self.segment_length) / self.move_length) + 1)]
            age_list = [age[n * self.move_length:(self.segment_length + n * self.move_length)] for n in
                        range(math.ceil((self.max_seq_length - self.segment_length) / self.move_length) + 1)]
            seg_list = [seg[n * self.move_length:(self.segment_length + n * self.move_length)] for n in
                        range(math.ceil((self.max_seq_length - self.segment_length) / self.move_length) + 1)]
            position_list = [position[n * self.move_length:(self.segment_length + n * self.move_length)] for n in
                             range(math.ceil((self.max_seq_length - self.segment_length) / self.move_length) + 1)]
            att_mask_list = [att_mask[n * self.move_length:(self.segment_length + n * self.move_length)] for n in
                             range(math.ceil((self.max_seq_length - self.segment_length) / self.move_length) + 1)]

            # mask = np.zeros(math.ceil((self.max_seq_length - self.segment_length) / self.move_length))
            # if sample['length'] <= self.segment_length:
            #     num = 1
            # else:
            #     num = math.ceil((sample['length'] - self.segment_length) / self.move_length) + 1
            # mask[:num] = np.ones(num)
            mask = [1. if each[0] != 0 else 0. for each in code_list]

        #             mask = np.zeros(math.ceil(self.max_seq_length-self.segment_length/self.move_length) + 1)
        #             if sample['length'] <= self.segment_length:
        #                 num = 1
        #             else:
        #                 num = math.ceil((sample['length']-self.segment_length)/self.move_length) + 1
        #             mask[:num] = np.ones(num)

        sample.update({'code': code_list, 'age': age_list, 'seg': seg_list,
                       'position': position_list, 'att_mask': att_mask_list, 'h_att_mask': mask})

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
        p = [1, 0, 0]
        choice = np.random.choice(strategy, p=p)

        if choice == 'random_mask':
            code = random_mask(code, category=['SEP'], p=0.2)
        elif choice == 'mask_diag_med':
            code = random_mask(code, category=['LAB', 'BMI', 'BPL', 'BPH', 'SMO', 'ALC'], p=0.5)
        else:
            code = random_mask(code, category=['DIA', 'MED'], p=0.5)

        sample.update({'code': code})

        return sample