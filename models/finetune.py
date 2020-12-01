from models.parts.blocks import BertPooler, BertEncoder
from models.parts.embeddings import Embedding
import torch.nn as nn
import pytorch_lightning as pl
import torch
import copy
import torch.nn.functional as F
from torch.optim import *
import pytorch_pretrained_bert as Bert
from torch.distributions.bernoulli import Bernoulli
import copy
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from typing import Any
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics.functional.classification import average_precision, auroc


class EHR2VecFineTune(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.save_hyperparameters()

        self.online_network = HiBEHRT(params)
        if params['checkpoint_feature'] is not None:
            self.online_network.load_state_dict(torch.load(params['checkpoint_feature'], map_location='cpu')['state_dict'], strict=False)

        self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=self.params['hidden_size'], out_features=1)

        self.valid_prc = pl.metrics.classification.Precision(num_classes=1)
        self.valid_recall = pl.metrics.classification.Recall(num_classes=1)
        self.sig = nn.Sigmoid()

        self.manual_valid = self.params['manual_valid']

        if self.manual_valid:
            self.reset_buffer_valid()

    def reset_buffer_valid(self):
        self.pred_list = []
        self.target_list = []

    def forward(self, record, age, seg, position, att_mask, h_att_mask):
        y = self.online_network(record, age, seg, position, att_mask, h_att_mask)
        y = self.pooler(y, encounter=False)
        y = self.classifier(y)
        return y

    def shared_step(self, batch, batch_idx):
        record,  age, seg, position, att_mask, h_att_mask, label = batch
        loss_fct = nn.BCEWithLogitsLoss()

        y = self.forward(record, age, seg, position, att_mask, h_att_mask)

        loss = loss_fct(y.view(-1, 1), label.view(-1, 1))

        return loss, y, label

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx)

        # log results
        # self.log_dict({'losses': {"train_loss": loss}})
        # self.logger.experiment.add_scalars("losses", {"train_loss": loss}, global_step=self.global_step)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        # log results
        self.log("val_loss", loss)
        # self.logger.experiment.add_scalars("losses", {"val_loss": loss}, global_step=self.global_step)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())

        self.valid_prc(self.sig(pred), label)
        self.valid_recall(self.sig(pred), label)

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())

        self.valid_prc(self.sig(pred), label)
        self.valid_recall(self.sig(pred), label)

    def configure_optimizers(self):
        optimizer = eval(self.params['optimiser'])
        optimizer = optimizer(self.parameters(), **self.params['optimiser_params'])

        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            **self.params['scheduler']
        )
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('valid_precision', self.valid_prc.compute())
        self.log('valid_recall', self.valid_recall.compute())

        if self.manual_valid:
            label = torch.cat(self.target_list, dim=0).view(-1)
            pred = torch.cat(self.pred_list, dim=0).view(-1)

            self.log('average_precision', average_precision(pred, target=label))
            self.log('auroc', auroc(pred, label))
            self.reset_buffer_valid()

    def test_epoch_end(self, outs):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        PRC = average_precision(pred, target=label)
        ROC = auroc(pred, label)

        print('average_precision', PRC)
        print('auroc', ROC)

        return {'auprc': PRC, 'auroc': ROC}

class Extractor(nn.Module):
    def __init__(self, params):
        super(Extractor, self).__init__()
        self.encoder = BertEncoder(params, params['extractor_num_layer'])
        self.pooler = BertPooler(params)

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

        encode_visit = encoded_layer
        return encode_visit  # [batch * seg_len * Dim]


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


class HiBEHRT(nn.Module):
    def __init__(self, params):
        super(HiBEHRT, self).__init__()
        self.embedding = Embedding(params)
        self.extractor = Extractor(params)
        self.aggregator = Aggregator(params)

    def forward(self, record, age, seg, position, att_mask, h_att_mask):

        output = self.embedding(record, age, seg, position)
        output = self.extractor(output, att_mask, encounter=True)

        h = self.aggregator(output, h_att_mask, encounter=False)
        return h
