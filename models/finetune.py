import pytorch_lightning as pl
from models.mlmbyol import SSLMLMBYOL
from models.parts.blocks import BertPooler, BertEncoder, BertLayerNorm
import torch.nn as nn
import torch
from pytorch_lightning.metrics.functional.classification import average_precision, auroc
from utils.utils import load_obj
import pytorch_pretrained_bert as Bert


class EHR2VecFineTune(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())

        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.online_network = SSLMLMBYOL(params)
        if params['checkpoint_feature'] is not None:
            self.online_network = self.online_network.load_from_checkpoint(params['checkpoint_feature'])

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

        ft = self.global_step < self.params['freeze_fine_tune']

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

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     **self.params['scheduler']
        # )
        return optimizer

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