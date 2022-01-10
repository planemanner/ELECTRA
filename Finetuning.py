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


class Evaluation_Metrics:
    def __init__(self, task):
        """
        MatthewCorrCoef : CoLA
        accuracy : SST-2, MNLI, QNLI, RTE, WNLI
        F1 Score : MRPC, QQP
        PearsonCorrCoef, SpearmanCorrCoef : STS-B
        """


class Loss_func:
    def __init__(self, task):
        """
        Cross entropy :  SST-2, MNLI, QNLI, RTE, WNLI, MRPC, QQP, CoLA
        MSE : STS-B
        """


class Downstream_wrapper(nn.Module):
    def __init__(self, downstream_backbone, task):
        super(Downstream_wrapper, self).__init__()
        self.backbone = downstream_backbone
        self.task = task

    def _construct_interpreter(self, task):
        """
        :param task:
        :return:
        """


def fine_tuner(args):
    if not os.path.exists(args.task):
        os.mkdir(args.task)

    '''MANUAL SEED ALLOCATION'''
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''configuration of electra discriminator'''
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

    ED = ELECTRA_DISCRIMINATOR(config=cfg)
    pretrain_checkpoint = torch.load(args.pretrained_model_weight_path)
    ED.load_state_dict(pretrain_checkpoint["state_dict"])
    ED.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    downstream_Backbone = ED.bert
    model = Downstream_wrapper(downstream_backbone=downstream_Backbone, task=args.task)

    train_set = FINE_TUNE_DATASET(task=args.task, mode='train', root_dir=args.data_root_dir)
    test_set = FINE_TUNE_DATASET(task=args.task, mode='test', root_dir=args.data_root_dir)

    collator_fn = FINE_TUNE_COLLATOR(tokenizer=tokenizer, task=args.task)

    Train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn)

    Test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, collate_fn=collator_fn)

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)

    criterion = Loss_func(task=args.task)

    for epoch in range(args.epoch):
        for data in Train_loader:
            if args.task in ["CoLA", "SST-2"]:
                sentences, labels = data
                sentences, labels = sentences.to(args.device), labels.to(args.device)
            else:
                sentences_1, sentences_2, labels = data
                sentences_1, sentences_2, labels = sentences_1.to(args.device), sentences_2.to(args.device), labels.to(args.device)

            optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--task", type=str, default="CoLA", choices=["CoLA", "SST-2", "MRPC", "QQP",
                                                                     "STS-B", "MNLI", "QNLI", "RTE", "WNLI"])
    parser.add_argument("--data_root_dir", type=str, default="")
    parser.add_argument("--pretrained_model_weight_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root_dir", type=str, default="")
    parser.add_argument("--logging_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_period", type=int, default=1, help="epoch unit")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    fine_tuner(args)
