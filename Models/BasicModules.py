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
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        self.W_Q = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)
        self.W_K = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)
        self.W_V = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)

        self.linear = nn.Linear(config.n_head * config.d_head, config.n_head * config.d_head)

        self.scale = config.d_head ** (1/4)

    def forward(self, Q, K, V):
        bs = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)

        out_dim, in_dim = self.W_Q.weight.shape
        d_head = out_dim // self.n_head

        q_s = self.W_Q(Q).view(bs, -1, self.n_head, d_head).transpose(1, 2).contiguous().view(bs * self.n_head,
                                                                                              -1, d_head)

        # (bs, n_pos, n_head, d_head) -> (bs, n_head, n_pos, d_head)
        k_s = self.W_K(K).view(bs, -1, self.n_head, d_head).transpose(1, 2).contiguous().view(bs * self.n_head,
                                                                                              -1, d_head)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(bs, -1, self.n_head, d_head).transpose(1, 2).contiguous().view(bs * self.n_head,
                                                                                              -1, d_head)
        '''This form is more memory-efficient way'''
        q_s = q_s / self.scale
        k_s = k_s / self.scale

        dot = torch.bmm(q_s, k_s.transpose(1, 2))
        mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        # (bs, n_head, n_q_seq, n_k_seq)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, v_s).view(bs, self.n_head, -1, d_head)
        out = out.transpose(1, 2).contiguous().view(bs, -1, out_dim)

        return self.linear(out)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_layernorm = nn.LayerNorm(config.d_head * config.n_head)

        self.pos_ffn_layernorm = nn.LayerNorm(config.d_head * config.n_head)
        self.pos_ff = nn.Sequential(
            nn.Linear(config.d_head * config.n_head, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_head * config.n_head)
        )

    def forward(self, inputs):
        attention = self.self_attn(inputs, inputs, inputs)
        output = self.attention_layernorm(inputs + attention)
        output = self.dropout(output)
        ff = self.pos_ff(output)
        output = self.pos_ffn_layernorm(ff+output)
        output = self.dropout(output)
        return output


class Encoder(nn.Module):
    def __init__(self, config, device):
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(config.n_enc_vocab, config.d_model)
        self.pos_embedding = nn.Embedding(config.n_enc_seq + 1, config.d_model)
        self.intermediate = nn.Linear(config.d_model, config.d_head * config.n_head)

        layers = []

        for i in range(config.n_layer):
            layers.append(EncoderLayer(config))

        self.layers = nn.Sequential(*layers)

        self.pad_idx = config.i_pad
        self.device = device

    def forward(self, inputs):
        tokens = self.token_embedding(inputs)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=self.device))[None, :, :].expand(b, t, e)

        x = self.intermediate(tokens + positions)
        x = self.layers(x)

        return x


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


class get_attn_pad_mask(nn.Module):
    def __init__(self):
        super(get_attn_pad_mask, self).__init__()

    def __call__(self, seq_q, seq_k, i_pad):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(i_pad)  # Pad 랑 값이 일치하면 True. seq_k 는 attention mask
        pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
        return pad_attn_mask



