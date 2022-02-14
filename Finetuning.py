import torch
import numpy as np
import argparse
from Models.BERT import ELECTRA_DISCRIMINATOR
from data_related.utils import Config
import os
from torch import nn
from transformers import AutoTokenizer
import random
from data_related.Custom_dataloader import FINE_TUNE_DATASET, FINE_TUNE_COLLATOR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from utils import lr_scheduling, get_layer_decayed_lrs, make_param_groups, get_layer_lrs, linear_warmup_and_then_decay, Evaluation, Downstream_wrapper


def fine_tuner(args):
#     if not os.path.exists(args.task):
#         os.mkdir(args.task)

    '''MANUAL SEED ALLOCATION'''
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''configuration of electra discriminator'''
    cfg = Config({"n_enc_vocab": 30522,  # correct
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

    ED = ELECTRA_DISCRIMINATOR(config=cfg, device=args.device)
#     pretrain_checkpoint = torch.load(args.pretrained_model_weight_path)
#     ED.load_state_dict(pretrain_checkpoint["state_dict"])
    tokenizer_path = "/vision/7032593/NLP/ELECTRA/tokenizer_files"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    downstream_Backbone = ED.bert
    model = Downstream_wrapper(downstream_backbone=downstream_Backbone, task=args.task, config=cfg)
    model = model.to(args.device)

    train_set = FINE_TUNE_DATASET(task=args.task, mode='train', root_dir=args.data_root_dir)
    test_set = FINE_TUNE_DATASET(task=args.task, mode='test', root_dir=args.data_root_dir)

    collator_fn = FINE_TUNE_COLLATOR(tokenizer=tokenizer)

    Train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn, drop_last=True)

    Test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collator_fn, drop_last=True)

    lrs = get_layer_lrs(lr=args.lr, decay_rate=0.8, num_hidden_layers=12)
    param_groups = make_param_groups(model=model, lrs=lrs, global_lr=args.lr)
    optimizer = torch.optim.Adam(param_groups)

    loss_func = nn.MSELoss() if args.task == "STS-B" else nn.CrossEntropyLoss()
    Evaluator = Evaluation(task=args.task, dataloader=Test_loader, logging_dir=args.logging_dir, device=args.device)
    train_iter = 0
    total_iter = Train_loader.__len__() * args.epochs
    total_warm_up = int(args.warmup_fraction * total_iter)
    
    for epoch in range(args.epochs):
        model.train()
        for idx, data in enumerate(Train_loader):
            sentences, labels = data  # two sentences are grafted with [SEP].
            sentences, labels = sentences.to(args.device), labels.to(args.device)
            """
            Learning rate scheduling
            """
            pct = train_iter / total_iter
            global_lr = linear_warmup_and_then_decay(pct=pct, lr_max=args.lr, total_steps=total_iter,
                                                     warmup_steps=total_warm_up, end_lr=0.0, decay_power=1)

            layer_lrs = get_layer_decayed_lrs(lrs=lrs, pct=pct,
                                              warmup_steps=total_warm_up,
                                              total_steps=total_iter)

            lr_scheduling(global_lr=global_lr, layer_lrs=layer_lrs, optimizer=optimizer)
            optimizer.zero_grad()

            outputs = model(sentences)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if train_iter % 50 ==0:
                    print(f"{epoch+1} / {args.epochs}, {idx+1} / {Train_loader.__len__()}, Train Loss : {loss.item()}")
            train_iter += 1

        if (epoch + 1) % args.eval_period == 0:
            Evaluator.task_wise_eval(model=model, cur_epoch=epoch)

    print("End of fine-tuning")
    Evaluator.writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_weight_path", 
                        type=str, 
                        default="/vision2/7032593/ELECTRA/check_points/DISC_ITER_140000_LM_MODEL.pth")
    parser.add_argument("--task", type=str, default="WNLI",
                        choices=["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"])
    parser.add_argument("--data_root_dir", type=str, default="/vision/7032593/NLP/GLUE-baselines/glue_data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root_dir", type=str, default="./save")
    parser.add_argument("--warmup_fraction", type=int, default=0.1)
    parser.add_argument("--logging_dir", type=str, default="./finetune_logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_period", type=int, default=1, help="epoch unit")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    fine_tuner(args)
    """
    Hyperparameter GLUE Value
    Learning Rate 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
    Adam eps 1e-6 v
    Adam β1 0.9 v
    Adam β2 0.999 v
    Layerwise LR decay 0.8 for Base/Small, 0.9 for Large
    Learning rate decay Linear
    Warmup fraction 0.1
    Attention Dropout 0.1
    Dropout 0.1
    Weight Decay 0
    Batch Size 32
    Train Epochs 10 for RTE and STS, 2 for SQuAD, 3 for other tasks
    """