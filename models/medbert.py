from models.parts.blocks import BertPooler, BertEncoder, BertLayerNorm
# from models.parts.embeddings import Embedding
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
from pytorch_lightning.metrics.functional.classification import average_precision, auroc
from utils.utils import load_obj
from torch.optim import *
from optim.tri_stage_lr_scheduler import TriStageLRScheduler

# import torch
# import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
# import copy
# import math
# import sys


class BEHRT2Vec(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())
        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.feature_extractor = BEHRT(params)
        self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=self.params['hidden_size'], out_features=1)

        self.valid_prc = pl.metrics.classification.Precision(num_classes=1)
        self.valid_recall = pl.metrics.classification.Recall(num_classes=1)
        self.f1 = pl.metrics.classification.F1(num_classes=1)
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

    def forward(self, record, age, seg, position, att_mask):
        y = self.feature_extractor(record, age, seg, position, att_mask)
        y = self.pooler(y, encounter=False)
        y = self.classifier(y)
        return y

    def shared_step(self, batch, batch_idx):
        record,  age, seg, position, att_mask, label = \
            batch['code'], batch['age'], batch['seg'], batch['position'], batch['att_mask'], batch['label']

        loss_fct = nn.BCEWithLogitsLoss()

        y = self.forward(record, age, seg, position, att_mask)

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
        self.valid_prc(self.sig(pred), label)
        self.valid_recall(self.sig(pred), label)
        self.f1(self.sig(pred), label)

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())

        self.valid_prc(self.sig(pred), label)
        self.valid_recall(self.sig(pred), label)

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
        self.log('valid_precision', self.valid_prc.compute())
        self.log('valid_recall', self.valid_recall.compute())
        self.log('F1_score', self.f1.compute())

        if self.manual_valid:
            label = torch.cat(self.target_list, dim=0).view(-1)
            pred = torch.cat(self.pred_list, dim=0).view(-1)

            auprc_score = average_precision(pred, target=label)
            auroc_score = auroc(pred, label)
            nll = self.nll(pred, label)

            print('epoch : {} AUROC: {} AUPRC: {} NLL: {}'.format(self.current_epoch, auroc_score, auprc_score, nll))

            self.log('average_precision', auprc_score)
            self.log('nll', nll)
            self.log('auroc', auroc_score)
            self.reset_buffer_valid()

    def test_epoch_end(self, outs):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        PRC = average_precision(pred, target=label)
        ROC = auroc(pred, label)
        nll = self.nll(pred, label)

        print('average_precision', PRC)
        print('auroc', ROC)
        print('nll', nll)

        return {'auprc': PRC, 'auroc': ROC, 'nll': nll}


class Aggregator(nn.Module):
    def __init__(self, params):
        super(Aggregator, self).__init__()
        self.encoder = BertEncoder(params, params['aggregator_num_layer'])

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask.to(dtype=next(self.parameters()).dtype)
        attention_mast = mask.unsqueeze(1).unsqueeze(2)
        attention_mast = (1.0 - attention_mast) * -10000.0
        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)

        return encoded_layer  # batch seq_len dim


class BEHRT(nn.Module):
    def __init__(self, params):
        super(BEHRT, self).__init__()
        self.embedding = Embedding(params)
        self.aggregator = Aggregator(params)

    def forward(self, record, age, seg, position, att_mask):

        output = self.embedding(record, age, seg, position)
        h = self.aggregator(output, att_mask, encounter=False)

        return h


class Embedding(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, params):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(params['vocab_size'], params['hidden_size'])
        # self.segment_embeddings = nn.Embedding(params['seg_vocab_size'], params['hidden_size'])
        # self.age_embeddings = nn.Embedding(params['age_vocab_size'], params['hidden_size'])
        self.posi_embeddings = nn.Embedding(params['max_seq_length'], params['hidden_size']). \
            from_pretrained(embeddings=self._init_posi_embedding(params['max_seq_length'], params['hidden_size']))

        self.LayerNorm = Bert.modeling.BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, word_ids, age_ids, seg_ids, posi_ids):
        word_embed = self.word_embeddings(word_ids)
        # segment_embed = self.segment_embeddings(seg_ids)
        # age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        embeddings = posi_embeddings + word_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table1
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)