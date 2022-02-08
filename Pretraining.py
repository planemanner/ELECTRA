import torch
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
from torch.utils.data import DataLoader
from Models.BERT import ELECTRA_GENERATOR, ELECTRA_DISCRIMINATOR, weight_sync
from Models.BasicModules import get_attn_pad_mask
import argparse
from transformers import AutoTokenizer
import random
from torch.utils.tensorboard import SummaryWriter
import os
import gc
from torch.nn.functional import gumbel_softmax


def GPU_MEMORY_CHECK(status):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Current status : {status}, Allocated memory : {a} / {t} \n Reserved memory : {f} / {t} \n")

    
class EFF_GEN_LOSS(torch.nn.Module):
    def __init__(self):
        super(EFF_GEN_LOSS, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def __call__(self, G_Logits, Masked_list, Labels):
        masked_g_logits = G_Logits[Masked_list, :]
        masked_labels = Labels[Masked_list]
        loss = self.criterion(masked_g_logits, masked_labels)
        return loss
    
    
def lr_warmup(optimizer, tgt_init_lr, cur_iter, warm_iter=10000):
    fraction = (cur_iter+1) / warm_iter
    warm_lr = tgt_init_lr * fraction
    for param in optimizer.param_groups:
        param['lr'] = warm_lr


def lr_decay(optimizer, init_lr, cur_iter, max_iter, warm_iter=10000):
    fraction = (cur_iter - warm_iter + 1) / (max_iter - warm_iter)
    decayed_lr = init_lr - fraction * init_lr
    for param in optimizer.param_groups:
        param['lr'] = decayed_lr


def model_save(model, optimizer, root_dir, cur_iter, model_type):
    save_path = os.path.join(root_dir, f"{model_type}_ITER_{str(cur_iter+1).zfill(6)}_LM_MODEL.pth")
    torch.save(
        {'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         },
        save_path
    )
    print(f"\n Trained model is saved at {save_path} \n")


# def sampler(Dist, logits, device):
#     Gumbel = Dist.sample(logits.shape).to(device)
#     return (logits.float() + Gumbel).argmax(dim=-1)


def mask_token_filler(Generator_logits, device, masked_tokens, masking_indices, labels):

    Generated_tokens = masked_tokens.clone()
    Generated_tokens[masking_indices] = gumbel_softmax(Generator_logits[masking_indices, :], hard=True).argmax(-1)
    Disc_labels = (Generated_tokens != labels)
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


def pretrain(args):
    # d_hidn=d_head*n_head
    G_cfg = Config({"n_enc_vocab": 30522,  # correct
                    "n_enc_seq": 512,  # correct
                    "n_seg_type": 2,  # correct
                    "n_layer": 12,  # correct
                    "d_model": 128,  # correct
                    "i_pad": 0,  # correct
                    "d_ff": 256,  # correct
                    "n_head": 1,  # correct
                    "d_head": 64,  # correct
                    "dropout": 0.1,  # correct
                    "layer_norm_epsilon": 1e-12  # correct
                    })

    D_cfg = Config({"n_enc_vocab": 30522,  # correct
                    "n_enc_seq": 512,  # correct
                    "n_seg_type": 2,  # correct
                    "n_layer": 12,  # correct
                    "d_model": 128,  # correct
                    "i_pad": 0,  # correct
                    "d_ff": 1024,  # correct
                    "n_head": 4,  # correct
                    "d_head": 64,  # correct
                    "dropout": 0.1,  # correct
                    "layer_norm_epsilon": 1e-12  # correct
                    })

    # train_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=pretrin_collate_fn)

    Generator = ELECTRA_GENERATOR(config=G_cfg)

    Discriminator = ELECTRA_DISCRIMINATOR(config=D_cfg)
    
    weight_sync(Generator.bert, Discriminator.bert)

    criterion_D = torch.nn.BCELoss()
    criterion_G = EFF_GEN_LOSS()

    Generator = Generator.to(args.device)
    Discriminator = Discriminator.to(args.device)
    params = list(Generator.parameters())+list(Discriminator.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd, eps=args.Adam_eps)
        
    train_dataset = LM_dataset(d_path=args.train_data_path)
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_path = "/vision/7032593/NLP/ELECTRA/tokenizer_files"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    collator = LM_collater(tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=args.num_workers)

    Logger = SummaryWriter(log_dir=args.log_dir)

    Train_iter_cnt = 0
    get_attn_mask = get_attn_pad_mask()
    print("Learning start !")

    while Train_iter_cnt < args.total_iteration:
        for i, seq_tokens in enumerate(train_loader):
            with torch.no_grad():
                if Train_iter_cnt < args.warm_up_steps:
                    lr_warmup(optimizer=optimizer, tgt_init_lr=args.lr, cur_iter=Train_iter_cnt, warm_iter = args.warm_up_steps)
                else:
                    lr_decay(optimizer=optimizer, init_lr=args.lr, cur_iter=Train_iter_cnt, max_iter=args.total_iteration, warm_iter = args.warm_up_steps)

            optimizer.zero_grad()
            seq_tokens = seq_tokens.to(args.device)
            attn_mask = get_attn_mask(seq_tokens, seq_tokens, 0)
            non_pad = (seq_tokens != 0)
            with torch.no_grad():
                masked_tokens, masked_lists = batch_wise_masking(seq_tokens)

            Generated_Logits = Generator(masked_tokens, attn_mask)
            G_LOSS = criterion_G(G_Logits=Generated_Logits, Masked_list=masked_lists, Labels=seq_tokens)
            # 반면에 Discriminator 는 전체를 봄
            with torch.no_grad():
                Generated_tokens, Disc_labels = mask_token_filler(Generator_logits=Generated_Logits, device=args.device,
                                                                  masked_tokens=masked_tokens, masking_indices=masked_lists, 
                                                                  labels=seq_tokens)
                
            Disc_logits = Discriminator(Generated_tokens, attn_mask).squeeze()
            D_Loss = criterion_D(Disc_logits.masked_select(non_pad), Disc_labels.masked_select(non_pad))
            loss = G_LOSS + args.d_loss_weight * D_Loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1)
            optimizer.step()
#             gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                
                Logger.add_scalar(tag="G_Loss / Train",
                                  scalar_value=G_LOSS.item(),
                                  global_step=Train_iter_cnt)
                Logger.add_scalar(tag="D_Loss / Train",
                                  scalar_value=D_Loss.item(),
                                  global_step=Train_iter_cnt)
                
                if ((Train_iter_cnt+1) % args.save_period) == 0:
                    print("Start to save a checkpoint....")
                    model_save(model=Discriminator, optimizer=optimizer,
                               root_dir=args.model_save, cur_iter=Train_iter_cnt, model_type="DISC")
                    model_save(model=Generator, optimizer=optimizer,
                               root_dir=args.model_save, cur_iter=Train_iter_cnt, model_type="GEN")
                    print("Done !!!")
                    
                if ((Train_iter_cnt + 1) % args.verbose_period) == 0:
                    print(f"ITER : {str(Train_iter_cnt + 1).zfill(6)}, G_LOSS : {G_LOSS.item()}, D_LOSS : {D_Loss.item()}")
                    
                Train_iter_cnt += 1  
    Logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.25e-4)  # for 128 batch, 5e-4
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")  # for 128 batch, 1e-2
    parser.add_argument("--d_loss_weight", type=float, default=50)
    parser.add_argument("--Adam_eps", type=float, default=1e-6)
    parser.add_argument("--warm_up_steps", type=int, default=10000, help="Based on iteration")
    parser.add_argument("--total_iteration", type=int, default=1000000)
    parser.add_argument("--train_data_path", type=str, default="./merged_lm.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_save", type=str, default="./check_points")
    parser.add_argument("--save_period", type=int, default=20000)
    parser.add_argument("--verbose_period", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    pretrain(args)
