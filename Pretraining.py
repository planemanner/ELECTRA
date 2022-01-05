import torch
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
from torch.utils.data import DataLoader
from Models.BERT import ELECTRA_GENERATOR, ELECTRA_DISCRIMINATOR, weight_sync
import argparse
from transformers import AutoTokenizer
import random
from torch.utils.tensorboard import SummaryWriter
import os
from itertools import chain
import numpy as np


def g_loss(criterion, g_logits, masked_lists, labels):
    """
    :param g_logits: (b, num_pos, num_voca)
    :param masked_lists: (b, dynamic lists)
    :param labels: (b, num_pos)
    :return:
    """
    loss = 0.
    effective_batch_cnt = 0
    for idx, values in enumerate(zip(g_logits, masked_lists)):
        g_logit, mask_list = values  # (num_pos, num_voca) , (locs)
        
        if g_logit[mask_list].shape[0] != 0:
            loss += (criterion(g_logit[mask_list], labels[idx][mask_list]) / len(mask_list))  # -> (locs, num_voca)
            effective_batch_cnt += 1
    return loss / effective_batch_cnt


def lr_warmup(optimizer, tgt_init_lr, cur_iter, max_iter=10000):
    warm_lr = tgt_init_lr / (max_iter - cur_iter + 1)
    optimizer.param_groups[0]['lr'] = warm_lr


def lr_decay(optimizer, init_lr, cur_iter, max_iter):
    fraction = init_lr / max_iter
    decayed_lr = init_lr - fraction * cur_iter
    optimizer.param_groups[0]['lr'] = decayed_lr


def model_save(model, optimizer, root_dir, cur_iter):
    save_path = os.path.join(root_dir, f"ITER_{str(cur_iter).zfill(6)}_LM_MODEL.pth")
    torch.save(
        {'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         },
        save_path
    )
    print(f"\n Trained model is saved at {save_path} \n")


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
    Generated_tokens = masked_tokens.clone()
    Disc_labels = torch.zeros_like(labels).bool()
    for idx, values in enumerate(zip(Generator_logits, masking_indices)):
        g_logit, mask_indices = values # (num_pos, num_voca), (num_mask)
        tgt_logits = g_logit[mask_indices, :]
        replaced_tokens = sampler(Dist=sampling_distribution, logits=tgt_logits, device=device)
        Generated_tokens[idx, mask_indices] = replaced_tokens 
        Disc_labels[idx, mask_indices] = labels[idx, mask_indices] != replaced_tokens  # 실제 잘 바꿨으면 False 를 못바꿨으면 True
        
    return Generated_tokens, Disc_labels.float()


def masking_seq(seq, mask_ratio=0.15):
    len_with_pad = len(seq)
    seq_len = len_with_pad - (seq.eq(0).sum() + 2)  # sos, eos is denoted by 2 and pad is the other
    masking_list = []
    mask_size = int(seq_len * mask_ratio)
    masked_tokens = seq.clone()
    for _ in range(mask_size):
        tmp_idx = random.randint(1, (seq_len-1))
        if tmp_idx not in masking_list:
            masking_list += [tmp_idx]
            
    masked_tokens[masking_list] = 103
    return masked_tokens, masking_list


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


def pretrain(args):
    # d_hidn=d_head*n_head
    cfg = Config({"n_enc_vocab": 30522,  # correct
                  "n_enc_seq": 512,  # correct
                  "n_seg_type": 2,
                  "n_layer": 12,
                  "d_hidn": 128,  # correct
                  "i_pad": 0,
                  "d_ff": 256,
                  "n_head": 4,
                  "d_head": 16,
                  "dropout": 0.1,
                  "layer_norm_epsilon": 1e-12
                  })

    # train_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=pretrin_collate_fn)
    Gumbel_Distribution = torch.distributions.gumbel.Gumbel(0, 1)

    Generator = ELECTRA_GENERATOR(config=cfg).to(args.device)
    print(Generator)
    Discriminator = ELECTRA_DISCRIMINATOR(config=cfg).to(args.device)

    weight_sync(Generator.bert, Discriminator.bert)

    criterion_D = torch.nn.BCEWithLogitsLoss()
    criterion_G = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(set(list(Generator.parameters()) + list(Discriminator.parameters())),
                                 lr=args.lr,
                                 weight_decay=args.wd,
                                 eps=args.Adam_eps)

    train_dataset = LM_dataset(d_path=args.train_data_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    collater = LM_collater(tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collater)

    Logger = SummaryWriter(log_dir=args.log_dir)
    Train_iter_cnt = 0

    print("Learning start !")
    for epoch in range(100):
        for i, seq_tokens in enumerate(train_loader):

            if Train_iter_cnt < 10000:
                lr_warmup(optimizer=optimizer, tgt_init_lr=args.lr, cur_iter=Train_iter_cnt)
            else:
                lr_decay(optimizer=optimizer, init_lr=args.lr, cur_iter=Train_iter_cnt, max_iter=args.total_iteration)
            Generator.train()
            Discriminator.train()

            '''lr modification'''

            seq_tokens = seq_tokens.to(args.device)

            optimizer.zero_grad()
            with torch.no_grad():
                masked_tokens, masked_lists = batch_wise_masking(seq_tokens)
            Generated_Logits = Generator(masked_tokens.to(args.device))
            # seq_tokens 도 masked 된 애들만 살려야 함
            G_LOSS = g_loss(criterion=criterion_G, g_logits=Generated_Logits, masked_lists=masked_lists, labels=seq_tokens)
            # 반면에 Discriminator 는 전체를 봄
            with torch.no_grad():
                Generated_tokens, Disc_labels = mask_token_filler(sampling_distribution=Gumbel_Distribution,
                                                                  Generator_logits=Generated_Logits, device=args.device,
                                                                  masked_tokens=masked_tokens,
                                                                  masking_indices=masked_lists, labels=seq_tokens)
            Disc_logits, _ = Discriminator(Generated_tokens)
            D_Loss = criterion_D(Disc_logits.squeeze(), Disc_labels)

            loss = G_LOSS + args.d_loss_weight * D_Loss
            torch.nn.utils.clip_grad_norm_(set(list(Generator.parameters()) + list(Discriminator.parameters())), 1)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                Logger.add_scalar(tag="G_Loss / Train",
                                  scalar_value=G_LOSS.item(),
                                  global_step=Train_iter_cnt)
                Logger.add_scalar(tag="D_Loss / Train",
                                  scalar_value=D_Loss.item(),
                                  global_step=Train_iter_cnt)


            if ((Train_iter_cnt+1) % args.save_period) == 0:
                print("Start to save a checkpoint....")
                model_save(model=Discriminator, optimizer=optimizer, root_dir=args.model_save, cur_iter=Train_iter_cnt)
                print("Done !!!")
            if ((Train_iter_cnt + 1) % args.verbose_period) == 0:
                with torch.no_grad():
                    print(f"ITER : {str(Train_iter_cnt).zfill(6)}, G_LOSS : {G_LOSS.item()}, D_LOSS : {D_Loss.item()}")
                    
            Train_iter_cnt += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.875e-4)  # for 128 batch, 5e-4
    parser.add_argument("--batch_size", type=int, default=48, help="Batch Size")
    parser.add_argument("--wd", type=float, default=2.5e-2, help="weight decay")  # for 128 batch, 1e-2
    parser.add_argument("--d_loss_weight", type=float, default=50)
    parser.add_argument("--Adam_eps", type=float, default=1e-6)
    parser.add_argument("--warm_up_steps", type=int, default=1e4, help="Based on iteration")
    parser.add_argument("--total_iteration", type=int, default=1000000)
    parser.add_argument("--train_data_path", type=str, default="./merged_lm.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_save", type=str, default="./check_points")
    parser.add_argument("--save_period", type=int, default=10000)
    parser.add_argument("--verbose_period", type=int, default=50)

    args = parser.parse_args()
    pretrain(args)
