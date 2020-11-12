import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import copy
import math
import sys


class Embedding(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, params):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(params['vocab_size'], params['hidden_size'])
        self.segment_embeddings = nn.Embedding(params['seg_vocab_size'], params['hidden_size'])
        self.age_embeddings = nn.Embedding(params['age_vocab_size'], params['hidden_size'])
        self.posi_embeddings = nn.Embedding(params['max_position_length'], params['hidden_size']). \
            from_pretrained(embeddings=self._init_posi_embedding(params['max_position_length'], params['hidden_size']))

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