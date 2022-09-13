from models.parts.blocks import BertPooler, BertEncoder, BertLayerNorm
from models.parts.embeddings import Embedding
import torch.nn as nn
import pytorch_lightning as pl
import torch
import copy
import torch.nn.functional as F
import pytorch_pretrained_bert as Bert
from torch.distributions.bernoulli import Bernoulli
import copy
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from typing import Any
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pytorch_lightning.metrics.functional.classification import average_precision, auroc
from torchmetrics.functional import average_precision, auroc
from utils.utils import load_obj
from torch.optim import *
from optim.tri_stage_lr_scheduler import TriStageLRScheduler
import math


class MultiEHR2Vec(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())
        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.feature_extractor = HiBEHRT(params)
        self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=self.params['hidden_size'], out_features=1)

        # self.valid_prc = pl.metrics.classification.Precision(num_classes=1)
        # self.valid_recall = pl.metrics.classification.Recall(num_classes=1)
        # self.f1 = pl.metrics.classification.F1(num_classes=1)
        self.nll = torch.nn.BCELoss()

        self.sig = nn.Sigmoid()

        self.manual_valid = self.params['manual_valid']

        if self.manual_valid:
            self.reset_buffer_valid()

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.params['initializer_range'])
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_buffer_valid(self):
        self.pred_list = []
        self.target_list = []

    def forward(self, record, age, seg, position, att_mask, h_att_mask):
        y = self.feature_extractor(record, age, seg, position, att_mask, h_att_mask)
        y = self.pooler(y, encounter=False)
        y = self.classifier(y)
        return y

    def shared_step(self, batch, batch_idx):
        record,  age, seg, position, att_mask, h_att_mask, label = \
            batch['code'], batch['age'], batch['seg'], batch['position'], batch['att_mask'], batch['h_att_mask'], batch['label']

        loss_fct = nn.BCEWithLogitsLoss()

        y = self.forward(record, age, seg, position, att_mask, h_att_mask)

        loss = loss_fct(y.view(-1, 1), label.view(-1, 1))

        return loss, y, label

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        # log results
        self.log("val_loss", loss)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())
        # self.valid_prc(self.sig(pred), label)
        # self.valid_recall(self.sig(pred), label)
        # self.f1(self.sig(pred), label)

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())

        # self.valid_prc(self.sig(pred), label)
        # self.valid_recall(self.sig(pred), label)

    def configure_optimizers(self):
        # optimizer = eval(self.params['optimiser'])

        if self.params['optimiser'] == 'Adam':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(self.named_parameters()) if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.params['optimiser_params']['weight_decay']},
                {'params': [p for n, p in list(self.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0}
            ]

            optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                                   lr=self.params['optimiser_params']['lr'],
                                                   warmup=self.params['optimiser_params']['warmup_proportion'])
        elif self.params['optimiser'] == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.params['optimiser_params']['lr'],
                            momentum=self.params['optimiser_params']['momentum'])
        else:
            raise ValueError('the optimiser is not implimented')

        if self.params['lr_strategy'] == 'fixed':
            return optimizer
        elif self.params['lr_strategy'] == 'warmup_cosine':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                **self.params['scheduler']
            )
            return [optimizer], [scheduler]
        elif self.params['lr_strategy'] == 'stri_stage':
            scheduler = TriStageLRScheduler(
                optimizer,
                **self.params['scheduler']
            )
            return [optimizer], [scheduler]

    def validation_epoch_end(self, outs):
        # log epoch metric
        # self.log('valid_precision', self.valid_prc.compute())
        # self.log('valid_recall', self.valid_recall.compute())
        # self.log('F1_score', self.f1.compute())

        if self.manual_valid:
            label = torch.cat(self.target_list, dim=0).view(-1)
            pred = torch.cat(self.pred_list, dim=0).view(-1)

            auprc_score = average_precision(pred, target=label.type(torch.LongTensor))
            auroc_score = auroc(pred, label.type(torch.LongTensor))
            nll = self.nll(pred, label)

            print('epoch : {} AUROC: {} AUPRC: {} NLL: {}'.format(self.current_epoch, auroc_score, auprc_score, nll))

            self.log('average_precision', auprc_score)
            self.log('nll', nll)
            self.log('auroc', auroc_score)
            self.reset_buffer_valid()

    def test_epoch_end(self, outs):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        PRC = average_precision(pred, target=label.type(torch.LongTensor))
        ROC = auroc(pred, label.type(torch.LongTensor))
        nll = self.nll(pred, label)

        print('average_precision', PRC)
        print('auroc', ROC)
        print('nll', nll)

        return {'auprc': PRC, 'auroc': ROC, 'nll': nll}


class Extractor(nn.Module):
    def __init__(self, params):
        super(Extractor, self).__init__()
        self.encoder = BertEncoder(params, params['extractor_num_layer'])
        self.pooler = BertPooler(params)
        self.params = params

    def reformat_input(self, mask, hidden):
        # mask [batch, segment, segment_length]
        # hidden [batch, segment, hidden_size]
        seg_len = (math.ceil((hidden.size()[1] - self.params['segment_length']) / self.params['move_length']) + 1)
        print('segment length ', seg_len)
        mask = mask[:,:, 0]

        encode_samples = torch.empty((seg_len, hidden.size()[0], self.params['segment_length'], self.params['hidden_size']), device=hidden.device).float()
        mask_samples = torch.empty((seg_len, hidden.size()[0], self.params['segment_length']), device=mask.device).float()

        for i in range(seg_len):
            encode_samples[i] = hidden[:, i * self.params['move_length']:(self.params['segment_length'] + i * self.params['move_length']), :]
            mask_samples[i] = mask[:, i * self.params['move_length']:(self.params['segment_length'] + i * self.params['move_length'])]

        return encode_samples.transpose(0, 1), mask_samples.transpose(0, 1)

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask.to(dtype=next(self.parameters()).dtype)

        attention_mast = mask.unsqueeze(2).unsqueeze(3)

        attention_mast = (1.0 - attention_mast) * -10000.0

        # encode_visit = []
        # for i in range(hidden_state.size(1)):
        #     encoded_layer = self.encoder(hidden_state[:, i, :, :], attention_mast[:, i, :], encounter)
        #     encoded_layer = self.pooler(encoded_layer, encounter)
        #     encode_visit.append(encoded_layer)
        # encode_visit = torch.stack(encode_visit, dim=1)
        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)
        encoded_layer = self.pooler(encoded_layer, encounter)

        pooled_hidden = encoded_layer
        new_hidden, new_mask = self.reformat_input(mask, encoded_layer)

        return pooled_hidden, mask, new_hidden, new_mask  # [batch * seg_len * Dim]

class ExtractorEncoder(nn.Module):
    def __init__(self, params, num_layer):
        super(ExtractorEncoder, self).__init__()
        layer = Extractor(params)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layer)])
    def forward(self, new_hidden, new_mask, encounter=True):
        for layer_module in self.layer:
            pooled_hidden, mask, new_hidden, new_mask = layer_module(new_hidden, new_mask, encounter)
        return pooled_hidden, mask

class Aggregator(nn.Module):
    def __init__(self, params):
        super(Aggregator, self).__init__()
        self.encoder = BertEncoder(params, params['aggregator_num_layer'])

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask[:, :, 0]
        mask = mask.to(dtype=next(self.parameters()).dtype)
        attention_mast = mask.unsqueeze(1).unsqueeze(2)
        attention_mast = (1.0 - attention_mast) * -10000.0
        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)

        return encoded_layer  # batch seq_len dim



class HiBEHRT(nn.Module):
    def __init__(self, params):
        super(HiBEHRT, self).__init__()
        self.embedding = Embedding(params)
        self.extractor = ExtractorEncoder(params, num_layer=params['hi_extractor_layer'])
        self.aggregator = Aggregator(params)

    def forward(self, record, age, seg, position, att_mask, h_att_mask):

        output = self.embedding(record, age, seg, position)
        pooled_output, mask = self.extractor(output, att_mask, encounter=True)
        h = self.aggregator(pooled_output, mask, encounter=False)
        return h