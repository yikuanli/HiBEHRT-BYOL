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


class SelfSupervisedLearning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.save_hyperparameters()

        self.online_network = HiBEHRT(params)
        self.target_network = copy.deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.weight_callback.on_train_batch_end(self.trainer, self, batch, batch_idx, dataloader_idx)

    def forward(self, record, age, seg, position, att_mask, h_att_mask, prob):
        y, _, _ = self.online_network(record, age, seg, position, att_mask, h_att_mask, prob)
        return y

    def cosine_similarity(self, a, b, att_mask):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1)  # sum dimension level
        loss = 2 - 2 * sim
        att_mask = att_mask.to(dtype=sim.dtype)
        loss = loss * att_mask
        loss = loss.sum(-1)  # mean over sequence level
        return loss.mean()  # mean over batch level

    def shared_step(self, batch, batch_idx):
        record_aug_1, record_aug_2, age, seg, position, att_mask, h_att_mask = batch

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(record_aug_1, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        with torch.no_grad():
            y2, z2, h2 = self.target_network(record_aug_2, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        loss_a = self.cosine_similarity(h1, z2, h_att_mask)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(record_aug_2, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        with torch.no_grad():
            y2, z2, h2 = self.target_network(record_aug_1, age, seg, position, att_mask, h_att_mask, self.params['random_mask'])
        # L2 normalize
        loss_b = self.cosine_similarity(h1, z2, h_att_mask)

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})
        self.logger.experiment.add_scalar('Loss/Train', total_loss, self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return total_loss

    def configure_optimizers(self):
        optimizer = eval(self.params['optimiser'])
        optimizer = optimizer(self.parameters(), **self.params['optimiser_params'])

        optimizer = LARSWrapper(optimizer)
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

    def forward(self, record, age, seg, position, att_mask, h_att_mask, prob):
        bournilli = Bernoulli(torch.ones_like(h_att_mask) * prob)

        output = self.embedding(record, age, seg, position)
        output = self.extractor(output, att_mask, encounter=True)

        output = output * bournilli.sample().unsqueeze(-1)

        y = self.aggregator(output, h_att_mask, encounter=False)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


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