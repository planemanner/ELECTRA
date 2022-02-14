import torch
from data_related.utils import Config
from data_related.Custom_dataloader import LM_dataset, LM_collater
from torch.utils.data import DataLoader
from Models.BERT import ELECTRA_MODEL
from Models.BasicModules import get_attn_pad_mask
import argparse
from transformers import AutoTokenizer
import random
from torch.utils.tensorboard import SummaryWriter
import os
import gc


def GPU_MEMORY_CHECK(status):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Current status : {status}, Allocated memory : {a} / {t} \n Reserved memory : {f} / {t} \n")


class lr_scheduler:
    def __init__(self, optimizer, init_lr, warm_iter, max_iter, logger):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warm_iter = warm_iter
        self.max_iter = max_iter
        self.logger = logger
        
    def lr_tune(self, cur_iter):
        if cur_iter < self.warm_iter:
            self.lr_warmup(cur_iter)
        else:
            self.lr_decay(cur_iter)

    def lr_warmup(self, cur_iter):
        fraction = (cur_iter + 1) / self.warm_iter
        warm_lr = self.init_lr * fraction
        for param in self.optimizer.param_groups:
            param['lr'] = warm_lr
        self.logger.add_scalar(tag="Learning Rate", scalar_value=warm_lr, global_step=cur_iter)

    def lr_decay(self, cur_iter):
        fraction = (cur_iter - self.warm_iter + 1) / (self.max_iter - self.warm_iter)
        decayed_lr = self.init_lr - fraction * self.init_lr
        for param in self.optimizer.param_groups:
            param['lr'] = decayed_lr
        self.logger.add_scalar(tag="Learning Rate", scalar_value=decayed_lr, global_step=cur_iter)


def model_save(model, optimizer, root_dir, cur_iter, model_type):
    save_path = os.path.join(root_dir, f"{model_type}_ITER_{str(cur_iter+1).zfill(6)}_LM_MODEL.pth")
    torch.save(
        {'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         },
        save_path
    )
    print(f"\n Trained model is saved at {save_path} \n")

            
def pretrain(args):
    G_cfg = Config({"n_enc_vocab": 30522,  # correct
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

    model = ELECTRA_MODEL(D_cfg, G_cfg, device=args.device).to(args.device)

    criterion_D = torch.nn.BCEWithLogitsLoss()
    criterion_G = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.Adam_eps)
    Logger = SummaryWriter(log_dir=args.log_dir)
    lr_controller = lr_scheduler(optimizer=optimizer, init_lr=args.lr,
                                 warm_iter=args.warm_up_steps, max_iter=args.total_iteration, logger=Logger)

    train_dataset = LM_dataset(d_path=args.train_data_path)
    tokenizer_path = "/vision/7032593/NLP/ELECTRA/tokenizer_files"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    collator = LM_collater(tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=args.num_workers)

    Train_iter_cnt = 0

    print("Learning start !")
    data_iter = iter(train_loader)
    for i in range(args.total_iteration):
        try:
            seq_tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            seq_tokens = next(data_iter)
            
        lr_controller.lr_tune(cur_iter=Train_iter_cnt)    
        optimizer.zero_grad()
        seq_tokens = seq_tokens.to(args.device)
        
        m_g_logits, disc_logits, replace_mask, disc_labels, masked_labels = model(seq_tokens)
        
        non_pad = (~seq_tokens.eq(0)) & (~seq_tokens.eq(101)) & (~seq_tokens.eq(102))
        
        G_LOSS = criterion_G(m_g_logits, masked_labels[replace_mask])

        D_LOSS = criterion_D(disc_logits[non_pad], disc_labels[non_pad])
        
        loss = G_LOSS + args.d_loss_weight * D_LOSS
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
#         gc.collect()
        torch.cuda.empty_cache()
        Train_iter_cnt += 1
        with torch.no_grad():

            Logger.add_scalar(tag="G_Loss / Train",
                              scalar_value=G_LOSS.item(),
                              global_step=Train_iter_cnt)
            Logger.add_scalar(tag="D_Loss / Train",
                              scalar_value=D_LOSS.item(),
                              global_step=Train_iter_cnt)

            if ((Train_iter_cnt+1) % args.save_period) == 0:
                print("Start to save a checkpoint....")
                model_save(model=model.discriminator, optimizer=optimizer,
                           root_dir=args.model_save, cur_iter=Train_iter_cnt, model_type="DISC")
                model_save(model=model.generator, optimizer=optimizer,
                           root_dir=args.model_save, cur_iter=Train_iter_cnt, model_type="GEN")
                print("Done !!!")

            if ((Train_iter_cnt + 1) % args.verbose_period) == 0:
                print(f"ITER : {str(Train_iter_cnt + 1).zfill(6)}, G_LOSS : {G_LOSS.item()}, D_LOSS : {D_LOSS.item()}")
    Logger.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)  # for 128 batch, 5e-4
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")  # for 128 batch, 1e-2
    parser.add_argument("--d_loss_weight", type=float, default=50)
    parser.add_argument("--Adam_eps", type=float, default=1e-6)
    parser.add_argument("--warm_up_steps", type=int, default=10000, help="Based on iteration")
    parser.add_argument("--total_iteration", type=int, default=1000000)
    parser.add_argument("--train_data_path", type=str, default="/vision/7032593/NLP/ELECTRA/merged_lm.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_save", type=str, default="./check_points")
    parser.add_argument("--save_period", type=int, default=50000)
    parser.add_argument("--verbose_period", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    pretrain(args)
