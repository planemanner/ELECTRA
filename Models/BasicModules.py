from torch import nn
import torch
from torch.nn import functional as F
import json


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


class MultiHeadAttention(nn.Module):
    def __init__(self, config, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        self.W_Q = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)
        self.W_K = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)
        self.W_V = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)

        self.scale = config.d_head ** (1/4)
        self.mask = mask
        
    def forward(self, inputs, mask=None):
        bs = inputs.size(0)
        # (bs, n_head, n_q_seq, d_head)
        out_dim, in_dim = self.W_Q.weight.shape
        d_head = out_dim // self.n_head

        q_s = self.W_Q(inputs).view(bs, -1, self.n_head, d_head).transpose(1, 2)

        # (bs, n_pos, n_head, d_head) -> (bs, n_head, n_pos, d_head)
        k_s = self.W_K(inputs).view(bs, -1, self.n_head, d_head).transpose(1, 2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(inputs).view(bs, -1, self.n_head, d_head).transpose(1, 2)
        '''This form is more memory-efficient way'''
        q_s = q_s / self.scale
        k_s = k_s / self.scale
        
        # (bs, n_head, n_pos, d_head) * (bs, n_head, d_head, n_pos) -> (bs, n_head, n_pos, n_pos)

        scores = q_s @ k_s.transpose(-2, -1)

        if self.mask:
            # attn_mask : (bs, n_pos, n_pos) -> (bs, 1, n_pos, n_pos) -> (bs, n_head, n_pos, n_pos)
#             attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            mask = mask[:, None, None, :].float()
            scores = scores + mask
            
        # v_s : (bs, n_head, n_pos, d_head)
        scores = self.dropout(F.softmax(scores, dim=-1))
        # attn : (bs, n_head, n_pos, n_pos) -> (bs, n_head, n_pos, n_pos)
        # out : (bs, n_head, n_pos, d_head) -> (bs, n_pos, n_head, d_head)
#         out = torch.matmul(attn, v_s).transpose(1, 2).contiguous().view(bs, -1, out_dim)
        out = (scores @ v_s).permute(0, 2, 1, 3).contiguous().view(bs, -1, out_dim)
        return self.linear(out)


class EncoderLayer(nn.Module):
    def __init__(self, config, mask=True):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config, mask)
        self.dropout = nn.Dropout(config.dropout)
        self.intermediate = nn.Linear(config.d_head * config.n_head, config.d_ff) 
        self.intermediate_act = nn.GELU()
        self.output_dense = nn.Linear(config.d_ff, config.d_head * config.n_head)
        self.attention_layernorm = nn.LayerNorm(config.d_head * config.n_head)
        self.output_layernorm = nn.LayerNorm(config.d_head * config.n_head)

        self.config = config
        
    def forward(self, inputs, mask):
        bs = inputs.shape[0]
        hidden = self.self_attn(inputs, mask)
        hidden = self.intermediate_act(self.intermediate(hidden))
        hidden = self.dropout(self.output_dense(hidden))
        output = self.output_layernorm(inputs + hidden)
        return output


class Encoder(nn.Module):
    def __init__(self, config, device):
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(config.n_enc_vocab, config.d_model)
        self.pos_embedding = nn.Embedding(config.n_enc_seq + 1, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model)
        self.intermediate = nn.Linear(config.d_model, config.d_head * config.n_head)
        self.dropout = nn.Dropout(config.dropout)
        layers = []

        for i in range(config.n_layer):
            layers.append(EncoderLayer(config))

        self.layers = nn.Sequential(*layers)

        self.pad_idx = config.i_pad
        self.device = device

    def forward(self, inputs, mask):
        tokens = self.token_embedding(inputs)
        b, t, e = tokens.size()

#         positions = self.pos_embedding(torch.arange(t, device=self.device))[None, :, :].expand(b, t, e)
        positions = self.pos_embedding(torch.arange(t, device=self.device))
        x = self.embed_norm(tokens + positions)
        x = self.dropout(x)
        x = self.intermediate(x)
        
        for layer in self.layers:
            x = layer(x, mask)

        return x


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)



