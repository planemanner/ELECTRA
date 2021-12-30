import torch
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
from torch.utils.data import DataLoader
from Models.BERT import ELECTRA_GENERATOR, ELECTRA_DISCRIMINATOR, weight_sync
import argparse
from transformers import AutoTokenizer
import random

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

PreSet
VOCAB 만들어야 함
- Pretraining hyperparameters
Architecture type     | Small | Base | Large
--------------------------------------------
Number of layers      | 12    | 12   |  24
Hidden Size           | 256   | 768  |  1024
FFN inner hidden size | 1024  | 3072 |  4096
Attention heads       |  4    |  12  |   16
Attention head size   |  64   |  64  |   64
Embedding Size        |  128  |  768 |  1024
Generator Size        |  1/4  |  1/3 |   1/4
 (multiplier for hidden-size, FFN-size, and num-attention-heads)
Mask percent          |  15   |  15  | 25
Lr decay type         |Linear |Linear|Linear
Warmup steps          | 1e4   | 1e4  | 1e4
Learning Rate         | 5e-4  | 2e-4 | 2e-4
Adam eps              | 1e-6  | 1e-6 | 1e-6
Adam β1               | 0.9   | 0.9  | 0.9
Adam β2               | 0.999 |0.999 |0.999
Attention Dropout     | 0.1   | 0.1  | 0.1
Dropout               | 0.1   | 0.1  | 0.1
Weight Decay          | 0.01  | 0.01 | 0.01
Batch Size            | 128   | 256  | 2048
Train Steps (ELECTRA) | 1M    | 766K | 400K
--------------------------------------------
Hyperparameter GLUE Value
Learning Rate 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
Adam eps 1e-6
Adam β1 0.9
Adam β2 0.999
Layerwise LR decay | 0.8 for Base/Small | 0.9 for Large
Lr decay type | Linear
Warmup fraction | 0.1
Attention Dropout | 0.1
Dropout | 0.1
Weight Decay | 0
Batch Size | 32
Train Epochs | 10 for RTE and STS | 2 for SQuAD | 3 for other tasks
"""

"""
Masked Output Decoding & Creating Discriminator labels

pred_tokens = self.sample(mlm_gen_logits) # ( #mlm_positions, )
# produce inputs for discriminator
generated = masked_inputs.clone() # (B,L) mask 가 포함된 token 을 가져오고
generated[is_mlm_applied] = pred_tokens # (B,L) 그 위치에 복사 붙여넣 
# produce labels for discriminator
is_replaced = is_mlm_applied.clone() # (B,L)
is_replaced[is_mlm_applied] = (pred_tokens != labels[is_mlm_applied]) # (B,L) label 의 값은 0 또는 1
"""


def sampler(Dist, logits, device):
    Gumbel = Dist.sample(logits.shape).to(device)
    return (logits.float() + Gumbel).argmax(dim=-1)


def mask_token_filler(sampling_distribution, Generator_logits,
                      device, masked_tokens, masking_indices, labels):
    """
    :param sampling_distribution: It should be Gumbel Distribution
    :param logits: Generator Language Model Outputs
    (Batch Size, Num Positions, Num Vocab)
    :param device: cpu or gpu
    :param masking_indices: target masking token indices
    (Batch Size, Num Masking Tokens)
    Num Masking Tokens < Num Positions
    Typically, Num Masking Tokens is less than 15 % of Num Positions
    :return:
    """

    masked_logits = Generator_logits[masking_indices, :]
    replaced_tokens = sampler(Dist=sampling_distribution, logits=masked_logits, device=device)
    Generated_tokens = masked_tokens.clone()
    Generated_tokens[masking_indices] = replaced_tokens
    Disc_labels = (labels[masking_indices] != replaced_tokens)  # 실제 잘 바꿨으면 False 를 못바꿨으면 True

    return Generated_tokens, Disc_labels


def masking_seq(seq, mask_ratio=0.15):
    len_with_pad = len(seq)
    seq_len = len_with_pad - (seq.eq(0).sum() + 2)  # sos, eos is denoted by 2 and pad is the other
    masking_list = []
    mask_size = int(seq_len * mask_ratio)
    for _ in range(mask_size):
        masking_list += [random.randint(1, (seq_len-1))]
    seq[masking_list] = 103

    return seq, masking_list


def batch_wise_masking(tokens, mask_ratio=0.163):
    # mask_ratio is empirically determined by examining thousand times to meet 15 % in every iteration
    # tokens shape is : (BS, Num Pos)
    masked_outputs = []
    masked_lists = []
    for tok in tokens:
        masked_tks, masked_list = masking_seq(tok, mask_ratio)  # Tensor format
        masked_outputs += [masked_tks]
        masked_list += [masked_lists]

    return torch.stack(masked_outputs), masked_lists


def pretrain(args):
    cfg = Config({"n_enc_vocab": 30522,  # correct
                  "n_enc_seq": 512,  # correct
                  "n_seg_type": 2,
                  "n_layer": 12,
                  "d_hidn": 128,  # correct
                  "i_pad": 0,
                  "d_ff": 256,
                  "n_head": 4,
                  "d_head": 64,
                  "dropout": 0.1,
                  "layer_norm_epsilon": 1e-12
                  })

    # train_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=pretrin_collate_fn)
    Gumbel_Distribution = torch.distributions.gumbel.Gumbel(0, 1)

    Generator = ELECTRA_GENERATOR(config=cfg)
    Discriminator = ELECTRA_DISCRIMINATOR(config=cfg)

    weight_sync(Generator.bert, Discriminator.bert)

    D_Loss_Weight = args.d_loss_weight
    criterion_D = torch.nn.BCEWithLogitsLoss()
    criterion_G = torch.nn.CrossEntropyLoss()

    losses = {"Generator Loss": 0.0,
              "Discriminator Loss": 0.0,
              "Iteration_cnt": 0}

    optimizer = torch.optim.Adam([{'params': Generator.parameters()},
                                  {'params': Discriminator.parameters()}],
                                 lr=args.lr,
                                 weight_decay=args.wd,
                                 eps=args.Adam_eps)

    train_dataset = LM_dataset(d_path=args.train_data_path)
    val_dataset = LM_dataset(d_path=args.val_data_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    collater = LM_collater(tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collater)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collater)

    """
    Generator and Discriminator are jointly trained
    """
    for epoch in range(args.epochs):
        for i, seq_tokens in enumerate(train_loader):
            """
            Data 에서 return 해줬으면 하는 형태는
            input-token-ids, masked-token-ids, segments ids
            """
            seq_tokens = seq_tokens.to(args.device)

            optimizer.zero_grad()
            masked_tokens, masked_lists = batch_wise_masking(seq_tokens)
            Generated_Logits = Generator(masked_tokens)

            # seq_tokens 도 masked 된 애들만 살려야 함
            G_LOSS = criterion_G(Generated_Logits.view(-1, cfg.n_enc_vocab), seq_tokens.view(-1))
            # 반면에 Discriminator 는 전체를 봄
            with torch.no_grad():
                Generated_tokens, Disc_labels = mask_token_filler(sampling_distribution=Gumbel_Distribution,
                                                                  Generator_logits=Generated_Logits, device=args.device,
                                                                  masked_tokens=masked_tokens,
                                                                  masking_indices=masked_lists, labels=seq_tokens)
            Disc_logits = Discriminator(Generated_tokens)
            D_Loss = criterion_D(Disc_logits, Disc_labels)

            loss = G_LOSS + args.d_loss_weight * D_Loss

            loss.backward()
            optimizer.step()

            """
            -------------------------------------
            |       Logger 및 validation 추가      |
            -------------------------------------
            """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bs", type=int, default=128, help="Batch Size")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--d_loss_weight", type=float, default=50)
    parser.add_argument("--Adam_eps", type=float, default=1e-6)
    parser.add_argument("--warm_up_steps", type=int, default=1e4, help="Based on iteration")
    parser.add_argument("--total_iteration", type=int, default=1e6)
    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    pretrain(args)
