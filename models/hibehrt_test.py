from models.parts.blocks import BertLayerNorm
# from models.parts.embeddings import Embedding
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
from utils.utils import load_obj
import numpy as np
import sys
import math


class EHR2VecTest(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        self.params.update({'vocab_size': vocab_size})

        self.save_hyperparameters()

        self.feature_extractor = HiBEHRT(params)
        # self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=self.params['hidden_size'], out_features=1)

        self.valid_prc = pl.metrics.classification.Precision(num_classes=1)
        self.valid_recall = pl.metrics.classification.Recall(num_classes=1)
        self.f1 = pl.metrics.classification.F1(num_classes=1)

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
        # y = self.pooler(y, encounter=False)
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
        optimizer = eval(self.params['optimiser'])
        optimizer = optimizer(self.parameters(), **self.params['optimiser_params'])

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     **self.params['scheduler']
        # )
        return optimizer

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

            print('epoch : {} AUROC: {} AUPRC: {}'.format(self.current_epoch,auroc_score, auprc_score ))

            self.log('average_precision', auprc_score)
            self.log('auroc', auroc_score)
            self.reset_buffer_valid()

    def test_epoch_end(self, outs):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        PRC = average_precision(pred, target=label)
        ROC = auroc(pred, label)

        # print('average_precision', PRC)
        # print('auroc', ROC)

        return {'auprc': PRC, 'auroc': ROC}


# class Extractor(nn.Module):
#     def __init__(self, params):
#         super(Extractor, self).__init__()
#         self.encoder = BertEncoder(params, params['extractor_num_layer'])
#         self.pooler = BertPooler(params)
#
#     def forward(self, hidden_state, mask, encounter=True):
#         mask = mask.to(dtype=next(self.parameters()).dtype)
#
#         attention_mast = mask.unsqueeze(2).unsqueeze(3)
#
#         attention_mast = (1.0 - attention_mast) * -10000.0
#
#         encoded_layer = self.encoder(hidden_state, attention_mast, encounter)
#         encoded_layer = self.pooler(encoded_layer, encounter)
#
#         encode_visit = encoded_layer
#         return encode_visit  # [batch * seg_len * Dim]
#
#
# class Aggregator(nn.Module):
#     def __init__(self, params):
#         super(Aggregator, self).__init__()
#         self.encoder = BertEncoder(params, params['aggregator_num_layer'])
#
#     def forward(self, hidden_state, mask, encounter=True):
#         mask = mask.to(dtype=next(self.parameters()).dtype)
#         attention_mast = mask.unsqueeze(1).unsqueeze(2)
#         attention_mast = (1.0 - attention_mast) * -10000.0
#         encoded_layer = self.encoder(hidden_state, attention_mast, encounter)
#
#         return encoded_layer  # batch seq_len dim

class TokenEmbedding(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, params):
        super(TokenEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(params['vocab_size'], params['hidden_size'])
        self.segment_embeddings = nn.Embedding(params['seg_vocab_size'], params['hidden_size'])
        self.age_embeddings = nn.Embedding(params['age_vocab_size'], params['hidden_size'])
        self.posi_embeddings = nn.Embedding(params['max_seq_length'], params['hidden_size']). \
            from_pretrained(embeddings=self._init_posi_embedding(params['max_seq_length'], params['hidden_size']))

        self.LayerNorm = Bert.modeling.BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, word_ids, age_ids, seg_ids, posi_ids):
        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        embeddings = posi_embeddings + age_embed + segment_embed + word_embed

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


class BertSelfAttention(nn.Module):
    def __init__(self, params):
        super(BertSelfAttention, self).__init__()
        if params['hidden_size'] % params['num_attention_heads'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (params['hidden_size'], params['num_attention_heads']))
        self.num_attention_heads = params['num_attention_heads']
        self.attention_head_size = int(params['hidden_size'] / params['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(params['hidden_size'], self.all_head_size)
        self.key = nn.Linear(params['hidden_size'], self.all_head_size)
        self.value = nn.Linear(params['hidden_size'], self.all_head_size)

        self.dropout = nn.Dropout(params['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x, token=True):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        if token is False:
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(0, 1, 3, 2, 4)

    def forward(self, hidden_states, attention_mask, token):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, token)
        key_layer = self.transpose_for_scores(mixed_key_layer, token)
        value_layer = self.transpose_for_scores(mixed_value_layer, token)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        if token is False:
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        else:
            context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, params):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.LayerNorm = Bert.modeling.BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, params):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(params)
        self.output = BertSelfOutput(params)

    def forward(self, input_tensor, attention_mask, token):
        self_output = self.self(input_tensor, attention_mask, token)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, params):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['intermediate_size'])
        if isinstance(params['hidden_act'], str) or (sys.version_info[0] == 2 and isinstance(params['hidden_act'], unicode)):
            self.intermediate_act_fn = Bert.modeling.ACT2FN[params['hidden_act']]
        else:
            self.intermediate_act_fn = params['hidden_act']

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, params):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(params['intermediate_size'], params['hidden_size'])
        self.LayerNorm = Bert.modeling.BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, params):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(params)
        self.intermediate = BertIntermediate(params)
        self.output = BertOutput(params)

    def forward(self, hidden_states, attention_mask, token):
        attention_output = self.attention(hidden_states, attention_mask, token)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoderToken(nn.Module):
    def __init__(self, params):
        super(BertEncoderToken, self).__init__()
        layer = BertLayer(params)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(params['num_hidden_layers_token'])])

    def forward(self, hidden_states, attention_mask, token):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, token)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, params):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states, token):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if token is False:
            first_token_tensor = hidden_states[:, 0]
        else:
            first_token_tensor = hidden_states[:, :, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModelToken(nn.Module):
    def __init__(self, params):
        super(BertModelToken, self).__init__()
        self.visit_embeddings = TokenEmbedding(params)
        self.visit_encoder = BertEncoderToken(params)
        self.pooler = BertPooler(params)
        # self.apply(self.init_bert_weights)

    def forward(self, word, age, seg, posi, mask, token):
        embedding_output_visit = self.visit_embeddings(word, age, seg, posi)
        mask = mask.to(dtype=next(self.parameters()).dtype)

        attention_mast = mask.unsqueeze(2).unsqueeze(3)

        attention_mast = (1.0 - attention_mast) * -10000.0

        encode_visit = []
        for i in range(embedding_output_visit.size(1)):
            encoded_layer = self.visit_encoder(embedding_output_visit[:, i, :, :], attention_mast[:, i, :], token)
            encoded_layer = self.pooler(encoded_layer, token)
            encode_visit.append(encoded_layer)
        encode_visit = torch.stack(encode_visit, dim=1)
        return encode_visit  # [batch * seg_len * Dim]


class BertEncoderVisit(nn.Module):
    def __init__(self, params):
        super(BertEncoderVisit, self).__init__()
        layer = BertLayer(params)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(params['num_hidden_layers_visit'])])

    def forward(self, hidden_states, attention_mask, token):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, token)
        return hidden_states


class BertModelVisit(nn.Module):
    def __init__(self, params):
        super(BertModelVisit, self).__init__()
        self.visit_encoder = BertEncoderVisit(params)
        self.pooler = BertPooler(params)

    def forward(self, token_output, visit_mask, token):
        embedding_output_visit = token_output
        mask = visit_mask.to(dtype=next(self.parameters()).dtype)
        attention_mast = mask.unsqueeze(1).unsqueeze(2)
        attention_mast = (1.0 - attention_mast) * -10000.0
        encoded_layer = self.visit_encoder(embedding_output_visit, attention_mast, token)
        pooled_output = self.pooler(encoded_layer, token)
        return pooled_output


class HiBEHRT(nn.Module):
    def __init__(self, params):
        super(HiBEHRT, self).__init__()
        self.bertToken = BertModelToken(params)
        self.bertVisit = BertModelVisit(params)

    def forward(self, record, age, seg, position, att_mask, h_att_mask):
        token_output = self.bertToken(record,
                                      age,
                                      seg,
                                      position,
                                      att_mask,
                                      token=False)  # [batch, visit, dim]

        visit_output = self.bertVisit(token_output,
                                      h_att_mask,
                                      token=False)

        # output = self.embedding(record, age, seg, position)
        # output = self.extractor(output, att_mask, encounter=True)

        # h = self.aggregator(output, h_att_mask, encounter=False)
        return visit_output