import torch
from torchsummary import summary
from torch import nn
from BERT import BERT
from utils import Config
import sentencepiece as sp
from torch.utils.data import DataLoader
from DataProcess import PretrainDataSet, pretrin_collate_fn
from tqdm import tqdm
import numpy as np


"""
BERT 는 Transformer 의 Encoder 만 사용함.
BERT 사전학습을 위한 기본 Task 는 2가지
MLM (Masked Language Model)
 Masking 이 된 부분의 단어를 예측하는 Task
 전체 단어 중 15 % 를 선택하고, 15 % 의 단어 중 80 %는 Masking 10 % 는 현재 단어 유지 나머지 10 % 는 
 임의의 단어로 대체
NSP (Next Sentence Prediction)
 CLS Token 으로 문장 A와 B의 관계를 예측하는 것
 ex) A 다음 문장이 B가 맞다면 True 틀리면 False  
"""


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BERT(self.config)
        # classfier
        self.projection_cls = nn.Linear(self.config.d_hidn, 2, bias=False)
        # lm
        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_enc_vocab, bias=False)
        self.projection_lm.weight = self.bert.encoder.enc_emb.weight

    def forward(self, inputs, segments):
        # (bs, n_enc_seq, d_hidn), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        # (bs, 2)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_enc_seq, n_enc_vocab)
        logits_lm = self.projection_lm(outputs)
        # (bs, n_enc_vocab), (bs, n_enc_seq, n_enc_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return logits_cls, logits_lm, attn_probs


def train(args):
    # vocab_file = "./kowiki.model"
    # vocab = sp.SentencePieceProcessor()
    # vocab.load(vocab_file)
    vocab = sp.SentencePieceProcessor()
    vocab.load(args.vocab_path)
    dataset = PretrainDataSet(vocab, args.data_path)

    CONFIG = Config({
        "n_enc_vocab": len(vocab),
        "n_enc_seq": args.num_enc_seq,  # 256,
        "n_seg_type": args.seg_type,  # 2,
        "n_layer": args.num_layer,  # 6,
        "d_hidn": args.dim_hidden,  # 256,
        "i_pad": args.pad_idx,  # 0,
        "d_ff": args.ff_dim,  # 1024,
        "n_head": args.num_head,  # 4,
        "d_head": args.d_head,  # 64,
        "dropout": args.dropout,  # 0.1,
        "layer_norm_epsilon": args.layer_norm_eps  # 1e-12
    })

    train_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=pretrin_collate_fn)

    model = BERTPretrain(CONFIG)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()
    losses = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for i, value in enumerate(train_loader):
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(args.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

        """
        ------------------------
        Evaluation Code Line 필요
        ------------------------
        """