from pytorch_lightning.metrics.functional.classification import average_precision, auroc
from models.parts.blocks import BertPooler, BertEncoderAdaptor, BertLayerNorm
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
import random
from utils.utils import load_obj


class EHR2VecAdaptorFineTune(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())

        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.online_network = SSLMLMBYOL(params)
        if params['checkpoint_feature'] is not None:
            self.online_network = self.online_network.load_state_dict(
                torch.load(params['checkpoint_feature'], map_location=lambda storage, loc: storage)['state_dict'],
                strict=False
            )

        self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=self.params['hidden_size'], out_features=1)

        self.sig = nn.Sigmoid()
        self.manual_valid = self.params['manual_valid']

        if self.manual_valid:
            self.reset_buffer_valid()

    def reset_buffer_valid(self):
        self.pred_list = []
        self.target_list = []

    def forward(self, record, age, seg, position, att_mask, h_att_mask, if_mask):

        ft = self.current_epoch < self.params['freeze_fine_tune']

        if ft:
            with torch.no_grad():
                y = self.online_network(record, age, seg, position, att_mask, h_att_mask, if_mask=if_mask)
        else:
            y = self.online_network(record, age, seg, position, att_mask, h_att_mask, if_mask=if_mask)

        y = self.pooler(y, encounter=False)
        y = self.classifier(y)
        return y

    def shared_step(self, batch, batch_idx):
        record,  age, seg, position, att_mask, h_att_mask, label = \
            batch['code'], batch['age'], batch['seg'], batch['position'], batch['att_mask'], batch['h_att_mask'], batch['label']

        loss_fct = nn.BCEWithLogitsLoss()

        y = self.forward(record, age, seg, position, att_mask, h_att_mask, if_mask=False)

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

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.shared_step(batch, batch_idx)

        if self.manual_valid:
            self.pred_list.append(self.sig(pred).cpu())
            self.target_list.append(label.cpu())

        # self.valid_prc(self.sig(pred), label)
        # self.valid_recall(self.sig(pred), label)

    def configure_optimizers(self):
        # optimizer = eval(self.params['optimiser'])

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in list(self.named_parameters()) if not any(nd in n for nd in no_decay)],
             'weight_decay': self.params['optimiser_params']['weight_decay']},
            {'params': [p for n, p in list(self.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

        optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters, lr=self.params['optimiser_params']['lr'],
                             warmup=self.params['optimiser_params']['warmup_proportion'])

        # optimizer = optimizer(self.parameters(), **self.params['optimiser_params'])

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            **self.params['scheduler']
        )
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outs):
        # log epoch metric
        # self.log('valid_precision', self.valid_prc.compute())
        # self.log('valid_recall', self.valid_recall.compute())

        if self.manual_valid:
            label = torch.cat(self.target_list, dim=0).view(-1)
            pred = torch.cat(self.pred_list, dim=0).view(-1)

            auprc_score = average_precision(pred, target=label)
            auroc_score = auroc(pred, label)

            print('epoch : {} AUROC: {} AUPRC: {}'.format(self.current_epoch,auroc_score, auprc_score ))

            self.log('average_precision', auprc_score)
            self.log('auroc', auroc_score)
            self.reset_buffer_valid()

    def test_epoch_end(self, outs):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        PRC = average_precision(pred, target=label)
        ROC = auroc(pred, label)

        print('average_precision', PRC)
        print('auroc', ROC)

        return {'auprc': PRC, 'auroc': ROC}


class SSLMLMBYOL(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())

        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.online_network = HiBEHRT(params)
        self.target_network = copy.deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()



    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    # def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    #     self.weight_callback.on_train_batch_end(self.trainer, self, batch, outputs,batch_idx, dataloader_idx)

    def forward(self, record, age, seg, position, att_mask, h_att_mask, bournilli_mask=None, if_mask=False):
        y, _, _ = self.online_network(record, age, seg, position, att_mask, h_att_mask, bournilli_mask,  if_mask)
        return y

    def cosine_similarity(self, a, b, att_mask, bournilli_mask):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1)  # sum dimension level
        loss = 2 - 2 * sim

        # put all paddings into 0 loss
        att_mask = att_mask.to(dtype=sim.dtype)
        loss = loss * att_mask

        # put all un-masked token with 0 loss
        bournilli_mask = (1 - bournilli_mask.to(dtype=sim.dtype))
        loss = loss * bournilli_mask

        # sum loss across sequence
        loss = loss.sum(-1)  # mean over sequence level
        return loss.mean()  # mean over batch level

    def shared_step(self, batch, batch_idx):
        record, age, seg, position, att_mask, h_att_mask = \
            batch['code'], batch['age'], batch['seg'], batch['position'], batch['att_mask'], batch['h_att_mask']

        bournilli_mask = Bernoulli(torch.ones_like(h_att_mask) * self.params['random_mask']).sample()

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(record, age, seg, position, att_mask, h_att_mask, bournilli_mask, if_mask=True)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(record, age, seg, position, att_mask, h_att_mask, bournilli_mask, if_mask=False)
        loss_a = self.cosine_similarity(h1, z2, h_att_mask, bournilli_mask)

        # Image 2 to image 1 loss
        # y1, z1, h1 = self.online_network(record_aug_2, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        # with torch.no_grad():
        #     y2, z2, h2 = self.target_network(record_aug_1, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        # L2 normalize
        # loss_b = self.cosine_similarity(h1, z2, h_att_mask)

        # Final loss
        total_loss = loss_a

        return loss_a, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, 'train_loss': total_loss})
        self.logger.experiment.add_scalar('Loss/Train', total_loss, self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, 'train_loss': total_loss})

        return total_loss

    def configure_optimizers(self):
        if self.params['optimiser'] == 'Adam':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(self.named_parameters()) if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.params['optimiser_params']['weight_decay']},
                {'params': [p for n, p in list(self.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0}
            ]

            optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters, lr=self.params['optimiser_params']['lr'],
                                                   warmup=self.params['optimiser_params']['warmup_proportion'])
        elif self.params['optimiser'] == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.params['optimiser_params']['lr'])
        else:
            raise ValueError('the optimiser is not implimented')
        # optimizer = optimizer(self.parameters(), **self.params['optimiser_params'])

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     **self.params['scheduler']
        # )

        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                **self.params['scheduler']
            )

        return [optimizer], [scheduler]



class HiBEHRT(nn.Module):
    def __init__(self, params):
        super(HiBEHRT, self).__init__()
        self.embedding = Embedding(params)
        self.extractor = Extractor(params)
        self.aggregator = Aggregator(params)

        self.projector = MLP(params)
        self.predictor = MLP(params)

    #     self.apply(self.init_bert_weights)
    #
    # def init_bert_weights(self, module):
    #     """ Initialize the weights.
    #     """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #     elif isinstance(module, BertLayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def forward(self, record, age, seg, position, att_mask, h_att_mask, bournilli_mask=None, if_mask=False):
        # bournilli = Bernoulli(torch.ones_like(h_att_mask) * prob)

        output = self.embedding(record, age, seg, position)
        output = self.extractor(output, att_mask, encounter=True)

        # output = output * bournilli.sample().unsqueeze(-1)
        if if_mask:
            prob = random.random()

            # 85% time mask token to 0, 15% time add random noise to those masked tokens
            if prob <0.85:
                output = output * bournilli_mask.unsqueeze(-1)
            else:
                noise_mask = torch.randn_like(output) * bournilli_mask.unsqueeze(-1)
                output = output + noise_mask

        y = self.aggregator(output, h_att_mask, encounter=False)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class Extractor(nn.Module):
    def __init__(self, params):
        super(Extractor, self).__init__()
        self.encoder = BertEncoderAdaptor(params, params['extractor_num_layer'])
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
        self.encoder = BertEncoderAdaptor(params, params['aggregator_num_layer'])

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask.to(dtype=next(self.parameters()).dtype)
        attention_mast = mask.unsqueeze(1).unsqueeze(2)
        attention_mast = (1.0 - attention_mast) * -10000.0
        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)

        return encoded_layer  # batch seq_len dim


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(params['hidden_size'], params['projector_size']),
            Bert.modeling.BertLayerNorm(params['projector_size'], eps=1e-12),
            nn.ReLU(),
            nn.Linear(params['projector_size'], params['hidden_size'])
        )

    def forward(self, x):
        return self.net(x)