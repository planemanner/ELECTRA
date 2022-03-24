from torch import nn
import torch
from .BasicModules import Encoder
from torch.nn.functional import log_softmax
import random
from torch.nn.functional import gumbel_softmax



def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.0, orginal_prob=0.15, ignore_index=-100):
    """
    Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence.
    * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
    """

    device = inputs.device
    labels = inputs.clone()

    # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
    for sp_id in special_token_indices:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

    # mask  (mlm_probability * (1-replace_prob-orginal_prob))
    mask_prob = 1 - replace_prob - orginal_prob
    mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    # replace with a random token (mlm_probability * replace_prob)
    if int(replace_prob) != 0:
        rep_prob = replace_prob / (replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(
            torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

    # inputs : mask 에 해당하는 위치의 token 들 random token으로 바꿈 (만약 replace prob가 0이 아니면...)
    # labels : mask 가 없는 곳은 모두 ignore index를 할당하여 이에 대한 loss 는 계산하지 않게끔 함.
    # mlm_mask : 말 그대로 masked language modeling을 위한 mask
    return inputs, labels, mlm_mask


def sampler(Dist, logits, device):
    Gumbel = Dist.sample(logits.shape).to(device)
    return (logits.float() + Gumbel).argmax(dim=-1)


def mask_token_filler(dist, Generator_logits, device, masked_tokens, masking_indices, labels):
    Generated_tokens = masked_tokens.clone()
    sampled_tks = sampler(dist, Generator_logits[masking_indices, :], device)
    #     Generated_tokens[masking_indices] = gumbel_softmax(Generator_logits[masking_indices, :], hard=True).argmax(-1)
    Generated_tokens[masking_indices, :] = sampled_tks
    Disc_labels = (Generated_tokens != labels)
    return Generated_tokens, Disc_labels.float()


def masking_seq(seq, mask_ratio=0.15):
    len_with_pad = len(seq)
    seq_len = len_with_pad - (seq.eq(0).sum() + 2)  # sos, eos is denoted by 2 and pad is the other
    masking_list = []
    mask_size = int(seq_len * mask_ratio)
    masked_tokens = seq.clone()
    for _ in range(mask_size):
        tmp_idx = random.randint(1, (seq_len - 1))
        if tmp_idx not in masking_list:
            masking_list += [tmp_idx]

    masked_tokens[masking_list] = 103
    masked_list = (masked_tokens != seq).tolist()

    return masked_tokens, masked_list


def batch_wise_masking(tokens, mask_ratio=0.163):
    # mask_ratio is empirically determined by examining thousand times to meet 15 % in every iteration
    # tokens shape is : (BS, Num Pos)
    masked_outputs = []
    masked_lists = []
    for tok in tokens:
        masked_tks, masked_list = masking_seq(tok, mask_ratio)  # Tensor format
        masked_outputs += [masked_tks]
        masked_lists += [masked_list]

    return torch.stack(masked_outputs), masked_lists


class BERT(nn.Module):
    def __init__(self, config, device):
        super(BERT, self).__init__()
        self.encoder = Encoder(config, device=device)
        self._init_weights()
        
    def _init_weights(self):
        modules = self.modules()
        for m in modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                
    
    def forward(self, inputs, input_mask):
        outputs = self.encoder(inputs, input_mask)
        return outputs


class ELECTRA_DISCRIMINATOR(nn.Module):
    def __init__(self, config, device):
        super(ELECTRA_DISCRIMINATOR, self).__init__()
        self.bert = BERT(config, device)
        self.projector = nn.Linear(config.d_head * config.n_head
                                   , config.d_model)
        self.activation = nn.GELU()
        self.discriminator = nn.Linear(config.d_model, 1)

    def forward(self, inputs, input_mask):
        bs, num_pos = inputs.shape
        outputs = self.bert(inputs, input_mask)
        outputs = self.projector(outputs)  # b, n_pos, d_head * n_head -> b, n_pos, d_,model
        outputs = self.activation(outputs)
        outputs = self.discriminator(outputs).squeeze(-1)
        return outputs


class ELECTRA_GENERATOR(nn.Module):
    def __init__(self, config, device):
        super(ELECTRA_GENERATOR, self).__init__()
        self.bert = BERT(config, device=device)
        self.activation = torch.nn.GELU()
        self.projection = nn.Linear(config.d_head * config.n_head, config.d_model)
        self.language_model = nn.Linear(config.d_model, config.n_enc_vocab, bias=True)
        self.layernorm = nn.LayerNorm(config.d_model)

                
    def forward(self, inputs, input_mask):
        bs, num_pos = inputs.shape
        outputs = self.bert(inputs, input_mask)
        outputs = self.projection(outputs)
        outputs = self.activation(outputs)
        outputs = self.layernorm(outputs) # bs, n_pos, d_model
#         outputs = self.language_model(outputs.view(bs * num_pos, -1)).view(bs, num_pos, -1)
        outputs = self.language_model(outputs)
        return outputs


class ELECTRA_MODEL(nn.Module):
    def __init__(self, D_config, G_config, device):
        super(ELECTRA_MODEL, self).__init__()
        self.discriminator = ELECTRA_DISCRIMINATOR(D_config, device)
        self.generator = ELECTRA_GENERATOR(G_config, device)
        self.distribution = torch.distributions.gumbel.Gumbel(0, 1)
        self.device = device
        self.cfg = G_config
        
        weight_sync(self.discriminator, self.generator)
        self.generator.language_model.weight = self.generator.bert.encoder.token_embedding.weight
            
    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save
        attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes 
        loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.config.i_pad
        seq_len = input_ids.shape[1]
#         token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
#                                       device=input_ids.device)
        
        return attention_mask
    
    def forward(self, seq_tokens, input_mask):
        """
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        [100, 102, 0, 101, 103]
        """
        with torch.no_grad():
            masked_tokens, generator_labels, replace_mask = mask_tokens(inputs=seq_tokens, mask_token_index=103,
                                                                        vocab_size=self.cfg.n_enc_vocab,
                                                                        special_token_indices=[100, 102, 0, 101, 103])

        g_logits = self.generator(masked_tokens, input_mask)

        m_g_logits = g_logits[replace_mask, :]
        
        with torch.no_grad():
            sampled_tokens = sampler(Dist=self.distribution, logits=m_g_logits, device=self.device)
            generated_tokens = masked_tokens.clone()
            generated_tokens[replace_mask] = sampled_tokens
            disc_labels = replace_mask.clone()
            disc_labels[replace_mask] = (sampled_tokens != generator_labels[replace_mask])  

        disc_logits = self.discriminator(generated_tokens, input_mask)

        return m_g_logits, disc_logits, replace_mask, disc_labels.float(), generator_labels


def weight_sync(src_model, tgt_model):
    tgt_model.bert.encoder.token_embedding = src_model.bert.encoder.token_embedding
    tgt_model.bert.encoder.pos_embedding = src_model.bert.encoder.pos_embedding



