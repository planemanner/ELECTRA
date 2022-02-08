from torch import nn
import torch
from torch.nn import functional as F
import json


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))  # Q 의 n_k_seq 는 K 의 n_k_seq 와 동일.

        # bs, n_head, n_q_seq, n_q_seq
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        self.W_Q = nn.Linear(config.d_model, config.n_head * config.d_head)
        self.W_K = nn.Linear(config.d_model, config.n_head * config.d_head)
        self.W_V = nn.Linear(config.d_model, config.n_head * config.d_head)

        self.scaled_dot_attn = ScaledDotProductAttention(config)
        self.linear = nn.Linear(config.n_head * config.d_head, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # (bs, n_enc_seq, self.config.n_head * self.config.d_head)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        out_dim, in_dim = self.W_Q.weight.shape
        print(Q.shape)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, out_dim // self.n_head).transpose(1, 2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, out_dim // self.n_head).transpose(1, 2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, out_dim // self.n_head).transpose(1, 2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        
        # (bs, n_head, n_q_seq, d_head) -> (bs, n_head, n_q_seq, n_k_seq)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        # BS, -1, n_head * d_head
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, out_dim)

        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)

        output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)

        return output, attn_prob


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=config.d_model,
                               out_channels=config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=config.d_ff,
                               out_channels=config.d_model, kernel_size=1)
        self.active = F.gelu


    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        residual = inputs
        output = self.conv1(inputs.transpose(1, 2))  # input 자체가 (bs, d_head * n_head, n_seq)
        output = self.active(output)
        # (bs, n_seq, n_head * d_head)
        output = self.conv2(output).transpose(1, 2)  # output : (BS, WORD_LENGTH, Rep_DIM) = (BS, WORD_LEN, n_head * d_head)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_layernorm = nn.LayerNorm(config.d_model)
        self.pos_ffn = PoswiseFeedForwardNet(config)
        self.pos_ffn_layernorm = nn.LayerNorm(config.d_model)
    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        attn_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        '''-> bs, n_enc_seq, -1'''
        attn_outputs = self.attention_layernorm(inputs + self.dropout(attn_outputs))
        # (bs, n_enc_seq, d_hidn)
        outputs = self.pos_ffn_layernorm(attn_outputs + self.dropout(self.pos_ffn(attn_outputs)))
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.enc_emb = nn.Embedding(config.n_enc_vocab, config.d_model)
        self.pos_emb = nn.Embedding(config.n_enc_seq + 1, config.d_model)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.pad_idx = config.i_pad
        
    def forward(self, inputs, attn_mask):
        positions = torch.arange(inputs.size(1),
                                 device=inputs.device,
                                 dtype=inputs.dtype).expand(inputs.size(0),
                                                            inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.pad_idx)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)
        # (bs, n_enc_seq, self.config.n_head * self.config.d_head)

        # (bs, n_enc_seq, n_enc_seq)
        # attn_mask = self.get_attention_mask(inputs, inputs, self.pad_idx)

        attn_probs = []
        for idx, layer in enumerate(self.layers):
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    """
    Default configuration for ELECTRA
    config = Config({
    "n_enc_vocab": len(vocab),  # 30522 is the vocab size of ELECTRA
    "n_enc_seq": 128,
    "n_seg_type": 2,
    "n_layer": 6,
    "d_hidn": 128,
    "i_pad": 0,
    "d_ff": 1024,
    "n_head": 12,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12
    })
    """

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



