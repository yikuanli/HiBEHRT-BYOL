import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import copy
import math
import sys
from models.parts.attention import BertSelfAttention


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEncoder(nn.Module):
    def __init__(self, params, num_layer):
        super(BertEncoder, self).__init__()
        layer = BertLayer(params)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layer)])

    def forward(self, hidden_states, attention_mask, encounter):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, encounter)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, params):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(params)
        self.intermediate = BertIntermediate(params)
        self.output = BertOutput(params)

    def forward(self, hidden_states, attention_mask, encounter):
        attention_output = self.attention(hidden_states, attention_mask, encounter)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertSelfOutput(nn.Module):
    def __init__(self, params):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.LayerNorm = BertLayerNorm(params['hidden_size'], eps=1e-12)
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

    def forward(self, input_tensor, attention_mask, encounter):
        self_output = self.self(input_tensor, attention_mask, encounter)
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
        self.LayerNorm = BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, params):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states, encounter):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if encounter is False:
            first_token_tensor = hidden_states[:, 0]
        else:
            first_token_tensor = hidden_states[:, :, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeedforwardAdaptor(nn.Module):
    def __init__(self, params):
        super(FeedforwardAdaptor, self).__init__()
        self.down_proj = nn.Linear(params['hidden_size'], params['adaptor_size'])
        self.act_fn = Bert.modeling.ACT2FN['gelu']
        self.up_proj = nn.Linear(params['adaptor_size'], params['hidden_size'])

    def forward(self, hidden_state):
        net = self.down_proj(hidden_state)
        net = self.act_fn(net)
        net = self.up_proj(net)

        return hidden_state + net


class BertSelfOutputAdaptor(nn.Module):
    def __init__(self, params):
        super(BertSelfOutputAdaptor, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.LayerNorm = BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.adaptor = FeedforwardAdaptor(params)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adaptor(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionAdaptor(nn.Module):
    def __init__(self, params):
        super(BertAttentionAdaptor, self).__init__()
        self.self = BertSelfAttention(params)
        self.output = BertSelfOutputAdaptor(params)

    def forward(self, input_tensor, attention_mask, encounter):
        self_output = self.self(input_tensor, attention_mask, encounter)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertOutputAdaptor(nn.Module):
    def __init__(self, params):
        super(BertOutputAdaptor, self).__init__()
        self.dense = nn.Linear(params['intermediate_size'], params['hidden_size'])
        self.LayerNorm = BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])
        self.adaptor = FeedforwardAdaptor(params)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adaptor(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayerAdaptor(nn.Module):
    def __init__(self, params):
        super(BertLayerAdaptor, self).__init__()
        self.attention = BertAttentionAdaptor(params)
        self.intermediate = BertIntermediate(params)
        self.output = BertOutputAdaptor(params)

    def forward(self, hidden_states, attention_mask, encounter):
        attention_output = self.attention(hidden_states, attention_mask, encounter)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoderAdaptor(nn.Module):
    def __init__(self, params, num_layer):
        super(BertEncoderAdaptor, self).__init__()
        layer = BertLayerAdaptor(params)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layer)])

    def forward(self, hidden_states, attention_mask, encounter):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, encounter)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, params, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(params, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, params, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(params)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, params):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(params['hidden_size'], params['hidden_size'])

        self.transform_act_fn = Bert.modeling.ACT2FN['relu']

        self.LayerNorm = BertLayerNorm(params['hidden_size'], eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states